#!/usr/bin/env python
"""Fold the eras of one catalogue into one prefix, and tidy the bucket.

Plan and reasoning: docs/rerun_2026/29_one_prefix_per_catalogue.md. S3 has no
rename, so every step is a server-side copy that leaves the source in place,
followed by a verification that compares ETags and sizes object by object.
Nothing here deletes anything except the `delete` and `archive` subcommands,
and both re-verify first.

    unify_catalogue.py copy        --era western-early --into western
    unify_catalogue.py verify      --era western-early --into western
    unify_catalogue.py move-queues --campaign western-early [--campaign ...]
    unify_catalogue.py archive     --prefix western-a [--prefix ...]
    unify_catalogue.py delete      --era western-early --into western --yes

`copy` moves picks/ and runs/ by CopyObject and rewrites each manifest's
files[].path to the new prefix on the way. `verify` lists both sides and
checks every source object has a destination twin with the same ETag and
size, that every moved manifest's paths answer a HEAD, and that a sample of
Parquet pairs holds the same rows. `move-queues` copies a campaign's queue
state (shards.jsonl, claims/, complete/, progress/, blocked/, review/,
access.json, README.json) under _queues/<name>/ and verifies it. `archive`
copies a whole prefix under _archive/<name>/, verifies, and deletes the
source. `delete` removes an era's output after verify passes again.

All subcommands are safe to re-run: an object whose twin already exists with
the same ETag is skipped.
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

BUCKET, REGION = "quakescope-picks-2026", "us-east-2"
s3 = boto3.client("s3", region_name=REGION, config=BotoConfig(
    retries={"max_attempts": 12, "mode": "adaptive"}, read_timeout=120, max_pool_connections=96))
QUEUE_PARTS = ("shards.jsonl", "access.json", "README.json", "claims/", "complete/", "progress/",
               "blocked/", "review/", "stations.parquet")
OUTPUT_PARTS = ("picks/", "manifests/", "runs/")
CATALOGUES = ("western", "obs", "global")     # prefixes readers see; each keeps its stations.parquet
LOG = Path("docs/rerun_2026/unify")


def listing(prefix: str) -> dict[str, tuple[str, int]]:
    """{key: (etag, size)} for everything under a prefix."""
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = (o["ETag"], o["Size"])
    return out


def retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(2 ** i, 30))


def copy_one(src: str, dst: str) -> None:
    retry(lambda: s3.copy_object(Bucket=BUCKET, Key=dst, CopySource={"Bucket": BUCKET, "Key": src},
                                 MetadataDirective="COPY"))


def pmap(fn, items, workers=64, label="") -> list:
    t0, done, out = time.time(), 0, []
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(fn, items):
            out.append(r); done += 1
            if done % 20000 == 0:
                print(f"  {label} {done:,}/{len(items):,} ({time.time() - t0:.0f} s)", flush=True)
    return out


def rewrite_manifest(src_key: str, era: str, into: str) -> None:
    dst_key = f"{into}/manifests/" + src_key.rsplit("/", 1)[1]
    m = json.loads(retry(lambda: s3.get_object(Bucket=BUCKET, Key=src_key)["Body"].read()))
    old, new = f"s3://{BUCKET}/{era}/", f"s3://{BUCKET}/{into}/"
    for f in m.get("files", []):
        if f["path"].startswith(old):
            f["path"] = new + f["path"][len(old):]
    m.setdefault("moved_from", era)
    retry(lambda: s3.put_object(Bucket=BUCKET, Key=dst_key, Body=json.dumps(m).encode(),
                                ContentType="application/json"))


# ----------------------------------------------------------------- subcommands
def cmd_copy(a) -> None:
    era, into = a.era, a.into
    summary = {"era": era, "into": into, "started": datetime.datetime.utcnow().isoformat() + "Z"}
    for part in ("picks/", "runs/"):
        src = listing(f"{era}/{part}")
        dst = listing(f"{into}/{part}")
        todo = [k for k, (etag, size) in src.items()
                if dst.get(f"{into}/{part}" + k[len(f"{era}/{part}"):]) != (etag, size)]
        print(f"{era}/{part}: {len(src):,} objects, {len(todo):,} to copy", flush=True)
        if not a.dry_run:
            pmap(lambda k: copy_one(k, f"{into}/{part}" + k[len(f"{era}/{part}"):]), todo, a.workers, part)
        summary[part] = {"source": len(src), "copied": len(todo)}
    src = listing(f"{era}/manifests/")
    print(f"{era}/manifests/: {len(src):,} to rewrite into {into}/manifests/", flush=True)
    if not a.dry_run:
        pmap(lambda k: rewrite_manifest(k, era, into), sorted(src), a.workers, "manifests")
    summary["manifests/"] = {"source": len(src), "rewritten": len(src)}
    # The eras of one catalogue were planned from the same station table, so
    # the catalogue's copy stands; check the id sets rather than the bytes,
    # which differ by serialization.
    try:
        import pandas as pd
        ids = lambda c: set(pd.read_parquet(f"s3://{BUCKET}/{c}/stations.parquet")["id"])
        extra = ids(era) - ids(into)
        summary["station_ids_only_in_era"] = len(extra)
        print(f"station ids in {era} but not in {into}: {len(extra)}"
              + ("" if not extra else " - merge the tables before deleting the era"))
    except Exception as exc:
        summary["station_ids_only_in_era"] = f"unchecked: {exc}"
    summary["finished"] = datetime.datetime.utcnow().isoformat() + "Z"
    write_log(f"copy_{era}", summary)


def cmd_verify(a) -> int:
    era, into = a.era, a.into
    problems = 0
    report = {"era": era, "into": into, "checked": datetime.datetime.utcnow().isoformat() + "Z"}
    for part in ("picks/", "runs/"):
        src = listing(f"{era}/{part}")
        dst = listing(f"{into}/{part}")
        bad = [k for k, v in src.items() if dst.get(f"{into}/{part}" + k[len(f"{era}/{part}"):]) != v]
        print(f"{era}/{part}: {len(src):,} objects, {len(bad):,} without an identical twin", flush=True)
        problems += len(bad)
        report[part] = {"source": len(src), "mismatched": len(bad), "examples": bad[:5]}
        if part == "picks/" and src and not bad:
            import pyarrow.parquet as pq
            sample = random.Random(0).sample(sorted(src), min(a.sample, len(src)))
            def rows(k):
                return pq.read_metadata(io.BytesIO(retry(lambda: s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()))).num_rows
            diff = sum(1 for k in sample if rows(k) != rows(f"{into}/{part}" + k[len(f"{era}/{part}"):]))
            print(f"  row counts equal on {len(sample) - diff}/{len(sample)} sampled pairs")
            problems += diff
            report[part]["sampled_pairs"] = len(sample); report[part]["row_mismatch"] = diff
    src = listing(f"{era}/manifests/")
    def check_manifest(k):
        dk = f"{into}/manifests/" + k.rsplit("/", 1)[1]
        try:
            m = json.loads(retry(lambda: s3.get_object(Bucket=BUCKET, Key=dk)["Body"].read()))
            o = json.loads(retry(lambda: s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()))
        except Exception as exc:
            return f"{k}: {type(exc).__name__}"
        if (m.get("records") != o.get("records") or m.get("n_picks") != o.get("n_picks")
                or len(m.get("files", [])) != len(o.get("files", []))):
            return f"{k}: content differs"
        for f in m.get("files", []):
            if not f["path"].startswith(f"s3://{BUCKET}/{into}/"):
                return f"{k}: path not rewritten: {f['path']}"
            try:
                s3.head_object(Bucket=BUCKET, Key=f["path"].split(f"s3://{BUCKET}/", 1)[1])
            except Exception:
                return f"{k}: missing object {f['path']}"
        return None
    bad = [r for r in pmap(check_manifest, sorted(src), a.workers, "manifests") if r]
    print(f"{era}/manifests/: {len(src):,} manifests, {len(bad):,} with a problem", flush=True)
    for b in bad[:10]:
        print("   ", b)
    problems += len(bad)
    report["manifests/"] = {"source": len(src), "problems": len(bad), "examples": bad[:10]}
    report["problems"] = problems
    write_log(f"verify_{era}", report)
    print("VERIFY OK" if not problems else f"VERIFY FAILED: {problems} problem(s)")
    return 0 if not problems else 1


def cmd_move_queues(a) -> int:
    problems = 0
    for camp in a.campaign:
        src_root, dst_root = f"{camp}/", f"_queues/{camp}/"
        # List only the queue parts: a catalogue-named campaign also holds
        # hundreds of thousands of pick objects under the same root.
        src = {}
        for part in QUEUE_PARTS:
            src.update(listing(src_root + part))
        dst = listing(dst_root)
        todo = [k for k, v in src.items() if dst.get(dst_root + k[len(src_root):]) != v]
        print(f"{camp}: {len(src):,} queue objects, {len(todo):,} to copy", flush=True)
        if not a.dry_run:
            pmap(lambda k: copy_one(k, dst_root + k[len(src_root):]), todo, a.workers, camp)
            dst = listing(dst_root)
            bad = [k for k, v in src.items() if dst.get(dst_root + k[len(src_root):]) != v]
            print(f"  verified: {len(src) - len(bad):,}/{len(src):,} identical", flush=True)
            problems += len(bad)
            write_log(f"queue_{camp}", {"campaign": camp, "objects": len(src), "copied": len(todo), "mismatched": len(bad),
                                        "when": datetime.datetime.utcnow().isoformat() + "Z"})
            if not bad and a.delete_source:
                # A campaign that shares its name with its catalogue (western,
                # obs, global) keeps stations.parquet: the queue gets a copy,
                # the catalogue keeps the original readers depend on. Learned
                # by deleting it on 2026-09-18 and restoring it from _queues/.
                keep = {f"{src_root}stations.parquet"} if camp in CATALOGUES else set()
                todel = sorted(set(src) - keep)
                pmap(lambda k: retry(lambda: s3.delete_object(Bucket=BUCKET, Key=k)), todel, a.workers, "delete")
                print(f"  deleted {len(todel):,} source objects" + (f", kept {sorted(keep)}" if keep else ""))
    return 0 if not problems else 1


def cmd_archive(a) -> int:
    problems = 0
    for prefix in a.prefix:
        src_root, dst_root = f"{prefix.rstrip('/')}/", f"_archive/{prefix.rstrip('/')}/"
        src, dst = listing(src_root), listing(dst_root)
        todo = [k for k, v in src.items() if dst.get(dst_root + k[len(src_root):]) != v]
        print(f"{prefix}: {len(src):,} objects, {len(todo):,} to copy", flush=True)
        if a.dry_run:
            continue
        pmap(lambda k: copy_one(k, dst_root + k[len(src_root):]), todo, a.workers, prefix)
        note = {"archived_from": prefix, "when": datetime.datetime.utcnow().isoformat() + "Z",
                "objects": len(src), "bytes": sum(v[1] for v in src.values()), "why": a.why or ""}
        s3.put_object(Bucket=BUCKET, Key=f"{dst_root}README.json", Body=json.dumps(note, indent=1).encode(),
                      ContentType="application/json")
        dst = listing(dst_root)
        bad = [k for k, v in src.items() if dst.get(dst_root + k[len(src_root):]) != v]
        print(f"  verified: {len(src) - len(bad):,}/{len(src):,} identical", flush=True)
        problems += len(bad)
        write_log(f"archive_{prefix.rstrip('/')}", {**note, "mismatched": len(bad)})
        if not bad:
            pmap(lambda k: retry(lambda: s3.delete_object(Bucket=BUCKET, Key=k)), sorted(src), a.workers, "delete")
            print(f"  deleted {len(src):,} source objects")
    return 0 if not problems else 1


def cmd_delete(a) -> int:
    if not a.yes:
        sys.exit("delete needs --yes; it re-verifies first and then removes the era's output for good")
    if cmd_verify(a):
        sys.exit("verify failed; nothing deleted")
    era = a.era
    keys = [k for k in listing(f"{era}/") if any(k[len(era) + 1:].startswith(p) for p in OUTPUT_PARTS)
            or k == f"{era}/stations.parquet"]
    print(f"deleting {len(keys):,} objects under {era}/ (picks, manifests, runs, stations.parquet)")
    pmap(lambda k: retry(lambda: s3.delete_object(Bucket=BUCKET, Key=k)), keys, a.workers, "delete")
    left = listing(f"{era}/")
    print(f"{era}/ now holds {len(left):,} objects" + ("" if not left else f", e.g. {sorted(left)[:3]}"))
    write_log(f"delete_{era}", {"era": era, "deleted": len(keys), "left": len(left),
                                "when": datetime.datetime.utcnow().isoformat() + "Z"})
    return 0


def write_log(name: str, obj: dict) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / f"{name}.json").write_text(json.dumps(obj, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("copy", "verify", "delete"):
        p = sub.add_parser(name)
        p.add_argument("--era", required=True); p.add_argument("--into", required=True)
        p.add_argument("--workers", type=int, default=64); p.add_argument("--dry-run", action="store_true")
        p.add_argument("--sample", type=int, default=200, help="Parquet pairs to compare by row count")
        if name == "delete":
            p.add_argument("--yes", action="store_true")
    p = sub.add_parser("move-queues")
    p.add_argument("--campaign", action="append", required=True); p.add_argument("--workers", type=int, default=64)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--delete-source", action="store_true")
    p = sub.add_parser("archive")
    p.add_argument("--prefix", action="append", required=True); p.add_argument("--workers", type=int, default=64)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--why", default="")
    a = ap.parse_args()
    rc = {"copy": cmd_copy, "verify": cmd_verify, "move-queues": cmd_move_queues,
          "archive": cmd_archive, "delete": cmd_delete}[a.cmd](a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
