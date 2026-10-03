#!/usr/bin/env python
"""Move the land stations' picks out of obs/ into _archive/obs-land/.

The obs campaign picked 1,993 land stations under reused temporary network
codes (docs/rerun_2026/31_obs_station_selection.md). Their picks are not
wrong, they are in the wrong catalogue, so they are archived, never deleted.

    python scripts/split_obs_land.py classify          # read every object's tid column, checkpointed
    python scripts/split_obs_land.py dry-run           # what would move, rewrite, stay
    python scripts/split_obs_land.py apply --yes       # do it, verifying as it goes
    python scripts/split_obs_land.py verify            # footer totals: obs + archive == before

Objects are partitioned by network, year and month and named after the shard,
so most of the separation is a move. For every object under obs/picks/:

* land-only: copy to _archive/obs-land/picks/<same relative key>, verify by
  size and MD5 of the body (multipart ETags cannot be compared), delete source;
* sea-only: untouched;
* mixed: the original is copied to _archive/obs-land/mixed-originals/<rel>
  first; then the land rows go to the archive key and the sea rows are
  written back to the SAME key with the pick schema, and the footers of both
  are read back so that rows_sea + rows_land == rows_before before anything
  else happens. The original copy is kept until `verify` passes.

Manifests follow the same split: land-only manifests move, mixed manifests
are rewritten with the land records and file rows removed and a `split` field
added, and a land-only manifest is written to the archive so it describes
itself. The station table is backed up and rewritten to the keep rows.
`obs/.dashboard/rowcount.json` is deleted so the dashboard rebuilds.

The land set is `action == return-to-global` in
sb_catalog/configs/networks/offshore_stations.csv; the keep set is
`keep-in-obs`. An object naming a station in neither set stops the run.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUCKET = "quakescope-picks-2026"
REGION = "us-east-2"
CAT = "obs"
ARCHIVE = "_archive/obs-land"
LIST = ROOT / "sb_catalog/configs/networks/offshore_stations.csv"
STATE = ROOT / "docs/rerun_2026/obs_split"          # checkpoints and logs, committed
s3 = boto3.client("s3", config=Config(region_name=REGION, max_pool_connections=96,
                                      retries={"max_attempts": 8, "mode": "adaptive"}))


def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def listing(prefix: str) -> dict[str, tuple[str, int]]:
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = (o["ETag"].strip('"'), o["Size"])
    return out


def get(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def put(key: str, body: bytes, content_type: str) -> None:
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=content_type)


def md5(body: bytes) -> str:
    return hashlib.md5(body).hexdigest()


def sets() -> tuple[set, set]:
    u = pd.read_csv(LIST, low_memory=False)
    land = set(u.loc[u["action"] == "return-to-global", "id"])
    keep = set(u.loc[u["action"] == "keep-in-obs", "id"])
    assert not land & keep
    return land, keep


def rel(key: str) -> str:
    assert key.startswith(f"{CAT}/")
    return key[len(f"{CAT}/"):]


# ------------------------------------------------------------------ classify
def classify_one(key: str, land: set, keep: set) -> dict:
    body = get(key)
    t = pq.read_table(io.BytesIO(body), columns=["tid"])
    tids = set(t.column("tid").to_pylist())
    n_land = len(tids & land)
    n_keep = len(tids & keep)
    unknown = sorted(tids - land - keep)
    kind = ("unknown" if unknown else "land-only" if n_keep == 0 and n_land
            else "sea-only" if n_land == 0 and n_keep else "mixed" if n_land and n_keep
            else "empty")
    return dict(key=key, rows=t.num_rows, bytes=len(body), kind=kind,
                n_land=n_land, n_keep=n_keep, unknown=",".join(unknown[:5]),
                md5=md5(body), classified=utcnow())


def cmd_classify(a) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    land, keep = sets()
    objs = listing(f"{CAT}/picks/")
    ck = STATE / "objects.parquet"
    done = pd.read_parquet(ck) if ck.exists() else pd.DataFrame(columns=["key"])
    todo = [k for k in objs if k not in set(done["key"])]
    print(f"{len(objs):,} objects under {CAT}/picks/, {len(done):,} classified, {len(todo):,} to do", flush=True)
    rows = list(done.to_dict("records"))
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(classify_one, k, land, keep): k for k in todo}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 500 == 0 or i == len(todo):
                pd.DataFrame(rows).to_parquet(ck, index=False)
                print(f"  {i:,}/{len(todo):,}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(ck, index=False)
    report(df)


def report(df: pd.DataFrame) -> None:
    g = df.groupby("kind").agg(objects=("key", "size"), rows=("rows", "sum"),
                               GB=("bytes", lambda s: round(s.sum() / 1e9, 3)))
    print(g.to_string())
    bad = df[df["kind"] == "unknown"]
    if len(bad):
        print(f"\n{len(bad)} objects name stations in neither set, e.g. {bad.iloc[0]['unknown']}; stop.")
        sys.exit(1)


# ------------------------------------------------------------------ manifests
def manifest_plan(land: set, keep: set) -> pd.DataFrame:
    keys = sorted(listing(f"{CAT}/manifests/"))
    def one(k):
        m = json.loads(get(k))
        tids = {r["tid"] for r in m.get("records", [])}
        return dict(key=k, n_land=len(tids & land), n_keep=len(tids & keep),
                    unknown=len(tids - land - keep), records=len(m.get("records", [])))
    with ThreadPoolExecutor(64) as ex:
        rows = list(ex.map(one, keys))
    df = pd.DataFrame(rows)
    df["kind"] = ["unknown" if u else "land-only" if k == 0 and l else "sea-only" if l == 0 and k
                  else "mixed" if l and k else "empty" for l, k, u in zip(df.n_land, df.n_keep, df.unknown)]
    return df


def cmd_dry_run(a) -> None:
    ck = STATE / "objects.parquet"
    if not ck.exists():
        sys.exit("run `classify` first")
    df = pd.read_parquet(ck)
    print("== objects ==")
    report(df)
    land, keep = sets()
    mp = manifest_plan(land, keep)
    mp.to_parquet(STATE / "manifests.parquet", index=False)
    print("\n== manifests ==")
    print(mp.groupby("kind").agg(manifests=("key", "size"), records=("records", "sum")).to_string())
    if (mp["kind"] == "unknown").any():
        sys.exit("manifests name stations in neither set; stop.")
    st = pd.read_parquet(f"s3://{BUCKET}/{CAT}/stations.parquet")
    print(f"\n== station table == {len(st):,} rows; keep {st['id'].isin(keep).sum():,}, "
          f"land {st['id'].isin(land).sum():,}, neither {(~st['id'].isin(keep | land)).sum():,}")


# ---------------------------------------------------------------------- apply
def copy_verified(src: str, dst: str, src_md5: str | None = None) -> str:
    """Server-side copy, then read the copy back and compare MD5 of the body."""
    s3.copy_object(Bucket=BUCKET, Key=dst, CopySource={"Bucket": BUCKET, "Key": src},
                   MetadataDirective="COPY")
    got = md5(get(dst))
    want = src_md5 or md5(get(src))
    if got != want:
        raise RuntimeError(f"copy mismatch {src} -> {dst}: {want} != {got}")
    return got


def move_object(row: dict) -> dict:
    src = row["key"]; dst = f"{ARCHIVE}/{rel(src)}"
    copy_verified(src, dst, row["md5"])
    s3.delete_object(Bucket=BUCKET, Key=src)
    return dict(key=src, archive=dst, action="moved", when=utcnow())


def split_object(row: dict, land: set) -> dict:
    src = row["key"]; dst = f"{ARCHIVE}/{rel(src)}"; orig = f"{ARCHIVE}/mixed-originals/{rel(src)}"
    body = get(src)
    if md5(body) != row["md5"]:
        raise RuntimeError(f"{src} changed since classify; re-run classify")
    copy_verified(src, orig, row["md5"])
    t = pq.read_table(io.BytesIO(body))
    mask = pa.array([x in land for x in t.column("tid").to_pylist()])
    land_t, sea_t = t.filter(mask), t.filter(pa.compute.invert(mask))
    assert land_t.num_rows + sea_t.num_rows == t.num_rows and land_t.num_rows and sea_t.num_rows
    def tobytes(tab):
        buf = io.BytesIO(); pq.write_table(tab, buf, compression="zstd"); return buf.getvalue()
    lb, sb = tobytes(land_t), tobytes(sea_t)
    put(dst, lb, "application/octet-stream")
    put(src, sb, "application/octet-stream")
    n_l = pq.read_metadata(io.BytesIO(get(dst))).num_rows
    n_s = pq.read_metadata(io.BytesIO(get(src))).num_rows
    if n_l != land_t.num_rows or n_s != sea_t.num_rows:
        raise RuntimeError(f"split readback mismatch on {src}: {n_l}+{n_s} != {t.num_rows}")
    return dict(key=src, archive=dst, action="split", rows_before=t.num_rows,
                rows_sea=n_s, rows_land=n_l, original=orig, when=utcnow())


def rewrite_manifest(key: str, land: set, keep: set, object_log: dict) -> dict:
    m = json.loads(get(key))
    recs = m.get("records", [])
    land_recs = [r for r in recs if r["tid"] in land]
    sea_recs = [r for r in recs if r["tid"] in keep]
    assert len(land_recs) + len(sea_recs) == len(recs)
    files = m.get("files", [])
    def key_of(f):
        p = f.get("path", "")
        return p[len(f"s3://{BUCKET}/"):] if p.startswith(f"s3://{BUCKET}/") else p
    sea_files, land_files = [], []
    for f in files:
        k = key_of(f); act = object_log.get(k)
        if act is None:                       # object we did not touch: sea-only or absent
            sea_files.append(f)
        elif act["action"] == "moved":
            land_files.append({**f, "path": f"s3://{BUCKET}/{act['archive']}"})
        else:                                 # split
            sea_files.append({**f, "rows": act["rows_sea"]})
            land_files.append({**f, "path": f"s3://{BUCKET}/{act['archive']}", "rows": act["rows_land"]})
    base = {k: v for k, v in m.items() if k not in ("records", "files", "n_picks", "station_days")}
    split = {"when": utcnow(), "land_records": len(land_recs), "sea_records": len(sea_recs),
             "doc": "docs/rerun_2026/31_obs_station_selection.md"}
    if sea_recs:
        obs_m = {**base, "n_picks": int(sum(r.get("npks", 0) for r in sea_recs)),
                 "station_days": len(sea_recs), "files": sea_files, "records": sea_recs,
                 "split": {**split, "land_manifest": f"s3://{BUCKET}/{ARCHIVE}/{rel(key)}"}}
        put(key, json.dumps(obs_m).encode(), "application/json")
    if land_recs:
        arc_m = {**base, "n_picks": int(sum(r.get("npks", 0) for r in land_recs)),
                 "station_days": len(land_recs), "files": land_files, "records": land_recs,
                 "split": {**split, "archived_from": f"s3://{BUCKET}/{key}"}}
        put(f"{ARCHIVE}/{rel(key)}", json.dumps(arc_m).encode(), "application/json")
    if not sea_recs:
        s3.delete_object(Bucket=BUCKET, Key=key)
    return dict(key=key, land=len(land_recs), sea=len(sea_recs),
                action="moved" if not sea_recs else "rewritten")


def cmd_apply(a) -> None:
    if not a.yes:
        sys.exit("apply needs --yes")
    land, keep = sets()
    df = pd.read_parquet(STATE / "objects.parquet")
    if (df["kind"] == "unknown").any():
        sys.exit("unknown stations in objects; stop")
    log_path = STATE / "apply_objects.jsonl"
    done = {}
    if log_path.exists():
        for line in open(log_path):
            r = json.loads(line); done[r["key"]] = r
    todo = df[(df["kind"].isin(["land-only", "mixed"])) & (~df["key"].isin(done))]
    print(f"objects: {int((df['kind'] == 'land-only').sum()):,} to move, {int((df['kind'] == 'mixed').sum()):,} "
          f"to split, {len(done):,} already done, {len(todo):,} now", flush=True)
    with open(log_path, "a") as log, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(move_object if r["kind"] == "land-only" else split_object, r, *(() if r["kind"] == "land-only" else (land,))): r["key"]
                for r in todo.to_dict("records")}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); done[r["key"]] = r
            log.write(json.dumps(r) + "\n"); log.flush()
            if i % 500 == 0 or i == len(futs):
                print(f"  objects {i:,}/{len(futs):,}", flush=True)
    # manifests
    mp = manifest_plan(land, keep)
    mtodo = mp[mp["kind"].isin(["land-only", "mixed"])]
    mlog = STATE / "apply_manifests.jsonl"
    mdone = set()
    if mlog.exists():
        mdone = {json.loads(l)["key"] for l in open(mlog)}
    mtodo = mtodo[~mtodo["key"].isin(mdone)]
    print(f"manifests: {len(mtodo):,} to rewrite or move ({len(mdone):,} done)", flush=True)
    with open(mlog, "a") as log, ThreadPoolExecutor(32) as ex:
        for i, r in enumerate(ex.map(lambda k: rewrite_manifest(k, land, keep, done), list(mtodo["key"])), 1):
            log.write(json.dumps(r) + "\n"); log.flush()
            if i % 500 == 0 or i == len(mtodo):
                print(f"  manifests {i:,}/{len(mtodo):,}", flush=True)
    # station table: back up, then keep rows only, same schema
    key = f"{CAT}/stations.parquet"
    body = get(key)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    put(f"{CAT}/stations.parquet.bak-{stamp}", body, "application/octet-stream")
    put(f"{ARCHIVE}/stations.parquet.before-split", body, "application/octet-stream")
    t = pq.read_table(io.BytesIO(body))
    ids = t.column("id").to_pylist()
    kept = t.filter(pa.array([x in keep for x in ids]))
    landt = t.filter(pa.array([x in land for x in ids]))
    buf = io.BytesIO(); pq.write_table(kept, buf); put(key, buf.getvalue(), "application/octet-stream")
    buf = io.BytesIO(); pq.write_table(landt, buf); put(f"{ARCHIVE}/stations.parquet", buf.getvalue(), "application/octet-stream")
    print(f"station table: {t.num_rows:,} -> {kept.num_rows:,} kept, {landt.num_rows:,} archived; backup {CAT}/stations.parquet.bak-{stamp}")
    # dashboard cache and README
    for k in (f"{CAT}/.dashboard/rowcount.json",):
        try:
            s3.delete_object(Bucket=BUCKET, Key=k); print(f"deleted {k}")
        except Exception as exc:
            print(f"could not delete {k}: {exc}")
    note = {"archived_from": f"{CAT}/", "when": utcnow(),
            "why": "land stations picked under reused offshore network codes; obs weight picks, "
                   "archived not deleted. docs/rerun_2026/31_obs_station_selection.md",
            "objects_moved": int(sum(1 for r in done.values() if r["action"] == "moved")),
            "objects_split": int(sum(1 for r in done.values() if r["action"] == "split")),
            "stations": len(land)}
    put(f"{ARCHIVE}/README.json", json.dumps(note, indent=1).encode(), "application/json")
    print("done; run `verify`")


def cmd_verify(a) -> None:
    df = pd.read_parquet(STATE / "objects.parquet")
    before = int(df["rows"].sum())
    def total(prefix):
        keys = [k for k in listing(prefix) if k.endswith(".parquet")]
        with ThreadPoolExecutor(64) as ex:
            return sum(ex.map(lambda k: pq.read_metadata(io.BytesIO(get(k))).num_rows, keys)), len(keys)
    obs_n, obs_k = total(f"{CAT}/picks/")
    arc_n, arc_k = total(f"{ARCHIVE}/picks/")
    land_rows = int(df.loc[df["kind"] == "land-only", "rows"].sum())
    sea_rows = int(df.loc[df["kind"] == "sea-only", "rows"].sum())
    ok = obs_n + arc_n == before
    print(f"rows before (classify): {before:,}\n{CAT}/picks/ now: {obs_n:,} in {obs_k:,} objects\n"
          f"{ARCHIVE}/picks/: {arc_n:,} in {arc_k:,} objects\nsum: {obs_n + arc_n:,}  -> {'OK' if ok else 'MISMATCH'}")
    print(f"land-only objects held {land_rows:,} rows, sea-only {sea_rows:,}; mixed contributed the rest")
    land, keep = sets()
    st = pd.read_parquet(f"s3://{BUCKET}/{CAT}/stations.parquet")
    print(f"{CAT}/stations.parquet: {len(st):,} rows, all keep: {st['id'].isin(keep).all()}")
    (STATE / "verify.json").write_text(json.dumps(dict(when=utcnow(), rows_before=before, obs_rows=obs_n,
                                                       archive_rows=arc_n, ok=ok, table_rows=len(st)), indent=1))
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("classify", "dry-run", "apply", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=48)
        if name == "apply":
            p.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    {"classify": cmd_classify, "dry-run": cmd_dry_run, "apply": cmd_apply, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
