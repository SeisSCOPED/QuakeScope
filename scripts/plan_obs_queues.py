#!/usr/bin/env python
"""Plan the obs repair queues from the per-station offshore list.

docs/rerun_2026/31_obs_station_selection.md, sections 2 and 4. Four queues,
all writing into the obs catalogue (parquet_uri s3://quakescope-picks-2026/obs)
with the `obs` weight on the job definition that carries the EL change:

    obs-el-test   X9 2012 short-period (EL) stations, 2012.300 to 2012.320:
                  the pre-fill test that the new image picks EL
    obs-el        the kept EL-only stations, 1993.001 to 2026.274
    obs-fill      offshore stations found in the global table plus the
                  pickable ones in no table (FDSN), 1993.001 to 2026.274
    obs-2026      the keep set, 2026.001 to 2026.274; replaces the never-run
                  obs-2026 queue, which was 39 land-only shards of 65

    pixi run python scripts/plan_obs_queues.py --dry-run
    pixi run python scripts/plan_obs_queues.py --write obs-el-test
    pixi run python scripts/plan_obs_queues.py --write obs-2026 --replace

`--dry-run` writes the station tables under sb_catalog/configs/networks/ and
prints the plans; it touches nothing in the bucket. `--write` installs one
queue: its station table, the immutable shards.jsonl and a README.json. For
obs-2026, `--replace` is required and refused if the old queue has claims or
completions; the old plan is copied to _archive/ first.
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BUCKET = "quakescope-picks-2026"
REGION = "us-east-2"
CATALOGUE = f"s3://{BUCKET}/obs"
LIST = ROOT / "sb_catalog/configs/networks/offshore_stations.csv"
ABSENT = ROOT / "sb_catalog/configs/networks/offshore_stations_absent.csv"
CFG = ROOT / "sb_catalog/configs/networks"
SPANS = {
    "obs-el-test": ("2012.300", "2012.320"),
    "obs-el": ("1993.001", "2026.274"),
    "obs-fill": ("1993.001", "2026.274"),
    "obs-2026": ("2026.001", "2026.274"),
}
COLS = ["id", "network_code", "station_code", "location_code", "channels", "latitude",
        "longitude", "elevation", "start_date", "end_date"]
s3 = boto3.client("s3", region_name=REGION)


def read_table(cat: str) -> pd.DataFrame:
    body = s3.get_object(Bucket=BUCKET, Key=f"{cat}/stations.parquet")["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def bands_of(channels: str) -> list[str]:
    return [c.strip() for c in str(channels).split(",") if c.strip()]


def absent_pickable() -> pd.DataFrame:
    """Offshore stations in no table, fetched at channel level, that `obs` can pick."""
    from plan_western_fill import fetch, merge_epochs
    from sb_catalog.src.constants import select_channel
    from sb_catalog.src.s3_state import station_date
    a = pd.read_csv(ABSENT, dtype={"Station": str})
    a = a[a["offshore_class"] == "offshore"]
    pairs = sorted(set(zip(a["Network"], a["Station"])))
    df, missing = fetch(pairs)
    df = merge_epochs(df)
    df = df[[select_channel(bands_of(c), weight="obs") is not None for c in df["channels"]]].copy()
    for c in ("start_date", "end_date"):
        df[c] = [station_date(v) for v in df[c]]
    print(f"absent stations: {len(pairs)} asked, {len(missing)} unknown to FDSN, "
          f"{df['station_code'].nunique()} pickable by obs ({len(df)} station-locations)")
    return df[COLS]


def tables(with_absent: bool = True) -> dict[str, pd.DataFrame]:
    from sb_catalog.src.constants import select_channel
    u = pd.read_csv(LIST, low_memory=False)
    keep = set(u.loc[u["action"] == "keep-in-obs", "id"])
    fill = set(u.loc[u["action"] == "fill-from-global", "id"])
    obs, glob = read_table("obs"), read_table("global")
    t_2026 = obs[obs["id"].isin(keep)].copy()
    el = [select_channel(bands_of(c), weight="obs") == "EL" for c in t_2026["channels"]]
    t_el = t_2026[el].copy()
    t_test = t_el[t_el["network_code"] == "X9"].copy()
    t_fill = glob[glob["id"].isin(fill)].copy()
    if with_absent:
        t_fill = pd.concat([t_fill[COLS], absent_pickable()], ignore_index=True)
    out = {"obs-el-test": t_test, "obs-el": t_el, "obs-fill": t_fill, "obs-2026": t_2026}
    for name, df in out.items():
        assert df["id"].is_unique, f"{name}: repeated id"
    return {k: v[COLS].reset_index(drop=True) for k, v in out.items()}


def dry_run(tabs: dict[str, pd.DataFrame]) -> None:
    from sb_catalog.src.shard_planner import parse_year_day, plan
    for name, df in tabs.items():
        a, b = SPANS[name]
        shards = plan(df, parse_year_day(a), parse_year_day(b))
        sd = sum(s["n_station_days"] for s in shards)
        out = CFG / f"{name.replace('-', '_')}.csv"
        df.to_csv(out, index=False)
        print(f"{name:12s} {len(df):5,} stations  {a}..{b}  {len(shards):6,} shards  {sd:9,} station-days"
              f"  -> {out.relative_to(ROOT)}")


def write(name: str, tabs: dict[str, pd.DataFrame], replace: bool) -> None:
    from sb_catalog.src.s3_state import S3CampaignState
    from sb_catalog.src.shard_planner import parse_year_day, plan
    queue = f"s3://{BUCKET}/_queues/{name}"
    prefix = f"_queues/{name}/"
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=50)
    existing = [o["Key"] for o in r.get("Contents", [])]
    if existing and name != "obs-2026":
        sys.exit(f"{prefix} already holds {len(existing)} objects; refusing")
    if name == "obs-2026" and existing:
        if not replace:
            sys.exit(f"{prefix} exists; pass --replace")
        busy = [k for k in existing if "/claims/" in k or "/complete/" in k or "/progress/" in k]
        if busy:
            sys.exit(f"{prefix} has claims or completions ({busy[:3]}); refusing to replace")
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d")
        for k in existing:
            dst = f"_archive/obs-2026-plan-before-{stamp}/" + k[len(prefix):]
            s3.copy_object(Bucket=BUCKET, Key=dst, CopySource={"Bucket": BUCKET, "Key": k})
        for k in existing:
            s3.delete_object(Bucket=BUCKET, Key=k)
        print(f"moved the old {prefix} plan ({len(existing)} objects) to _archive/obs-2026-plan-before-{stamp}/")
    df = tabs[name]
    a, b = SPANS[name]
    state = S3CampaignState(queue)
    state.write_stations(df)
    shards = plan(state.get_stations(), parse_year_day(a), parse_year_day(b))
    sd = sum(s["n_station_days"] for s in shards)
    print(state.write_shards(shards))
    s3.put_object(Bucket=BUCKET, Key=f"{prefix}README.json", ContentType="application/json",
                  Body=json.dumps(dict(purpose=__doc__.split("\n\n")[0].strip(),
                                       queue=name, parquet_uri=CATALOGUE, weight="obs",
                                       span=[a, b], stations=int(len(df)), shards=len(shards),
                                       station_days=int(sd), list=str(LIST.relative_to(ROOT)),
                                       planned=datetime.datetime.utcnow().isoformat() + "Z"),
                                  indent=1).encode())
    print(f"{name}: {len(df):,} stations, {len(shards):,} shards, {sd:,} station-days -> {queue}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", choices=sorted(SPANS))
    ap.add_argument("--replace", action="store_true", help="obs-2026 only: replace the never-run plan")
    ap.add_argument("--no-absent", action="store_true", help="skip the FDSN fetch of absent stations")
    a = ap.parse_args()
    tabs = tables(with_absent=not a.no_absent)
    if a.dry_run:
        dry_run(tabs)
    else:
        write(a.write, tabs, a.replace)


if __name__ == "__main__":
    main()
