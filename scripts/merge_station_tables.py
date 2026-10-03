#!/usr/bin/env python
"""Make a campaign's station table complete with respect to its picks.

`western/stations.parquet` lists 24,008 station epochs and the catalogue has
picks on 1,383 stations that are not in it. Those arrived with the
`western-fill` and `western-fill2` campaigns, which built their own station
tables under `_queues/` and wrote picks into `western/picks/` without their
station rows ever reaching the published table. `_queues/` is not public-read,
so an outside reader cannot find those stations at all: 178 million picks,
10.3% of the catalogue, unreachable by anyone selecting stations the documented
way.

    python scripts/merge_station_tables.py --campaign western --dry-run
    python scripts/merge_station_tables.py --campaign western --write

The write backs the current table up first, because bucket versioning is off
and an overwrite cannot be undone. It refuses to write a table with a repeated
id, which is the failure that cost 662 shards and $51 in September, and it
refuses to write a table that does not cover every station the manifests say
produced picks.

`--check` alone verifies the invariant for a campaign and writes nothing.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUCKET = "quakescope-picks-2026"
REGION = "us-east-2"          # not us-west-2: a wrong region fails Parquet reads with 301
_s3 = boto3.client("s3", config=Config(region_name=REGION, max_pool_connections=96,
                                       retries={"max_attempts": 6, "mode": "adaptive"}))

# Which queue tables feed which catalogue. A campaign that wrote into another
# campaign's prefix contributes its stations to that prefix's table.
CONTRIBUTORS = {
    "western": ["_queues/western-fill/stations.parquet",
                "_queues/western-fill2/stations.parquet"],
    # obs-fill adds offshore stations found in the global table and in no
    # table at all (docs/rerun_2026/31); obs-el and obs-2026 re-plan stations
    # the obs table already lists.
    "obs": ["_queues/obs-fill/stations.parquet"],
    "global": [],
}


def read_parquet(key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))


def merge_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station id: union the bands, widen the window.

    FDSN returns a station object per epoch, so a reconfigured station comes
    back two or three times. The reader indexes metadata by id and expects one
    row; with several it dies mid-shard on `'Series' object has no attribute
    'split'`.
    """
    if df.empty:
        return df

    def bands(v):
        return ",".join(sorted({b for x in v for b in str(x).split(",") if b and b != "nan"}))

    agg = dict(network_code=("network_code", "first"), station_code=("station_code", "first"),
               location_code=("location_code", "first"), channels=("channels", bands),
               latitude=("latitude", "first"), longitude=("longitude", "first"),
               elevation=("elevation", "first"),
               start_date=("start_date", "min"), end_date=("end_date", "max"))
    for extra in ("state", "start_yearday", "end_yearday"):
        if extra in df.columns:
            agg[extra] = (extra, "first" if extra == "state" else
                          ("min" if extra == "start_yearday" else "max"))
    return df.groupby("id", as_index=False).agg(**agg)


def stations_with_picks(campaign: str) -> tuple[set, collections.Counter]:
    """Every station the manifests say produced a pick, read from the bucket."""
    keys = []
    for page in _s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=f"{campaign}/manifests/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    print(f"  {len(keys):,} manifests")

    def one(k):
        try:
            return json.loads(_s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        except Exception:                                          # noqa: BLE001
            return None

    picks = collections.Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=64) as ex:
        for i, m in enumerate(ex.map(one, keys), 1):
            if m is None:
                continue
            for r in m.get("records", []):
                picks[r.get("tid")] += int(r.get("npks") or 0)
            if i % 30000 == 0:
                print(f"    {i:,}/{len(keys):,}  {time.time() - t0:.0f}s")
    return {k for k, v in picks.items() if v > 0}, picks


def check(campaign: str) -> dict:
    """Is the station table complete with respect to the picks?"""
    inv = read_parquet(f"{campaign}/stations.parquet")
    have = set(inv.id)
    with_picks, counts = stations_with_picks(campaign)
    orphan = with_picks - have
    dup = inv["id"][inv["id"].duplicated()].unique()
    out = dict(campaign=campaign, inventory=len(inv), unique_ids=len(have),
               duplicate_ids=len(dup), with_picks=len(with_picks),
               orphans=len(orphan), orphan_picks=sum(counts[k] for k in orphan),
               total_picks=sum(counts.values()))
    print(f"\n  {campaign}: {len(inv):,} rows, {len(have):,} unique ids, "
          f"{len(dup)} duplicated")
    print(f"  stations with picks: {len(with_picks):,}")
    print(f"  with picks but NOT in the table: {len(orphan):,} "
          f"({out['orphan_picks']:,} picks, "
          f"{100 * out['orphan_picks'] / max(out['total_picks'], 1):.1f}% of the campaign)")
    if orphan:
        nets = collections.Counter(k.split(".")[0] for k in orphan)
        print(f"  by network: {dict(nets.most_common(8))}")
    return out | {"orphan_ids": sorted(orphan)}


def merge(campaign: str, write: bool) -> None:
    inv = read_parquet(f"{campaign}/stations.parquet")
    print(f"  published table: {len(inv):,} rows, {inv.id.nunique():,} unique ids")
    parts = [inv]
    for key in CONTRIBUTORS.get(campaign, []):
        try:
            extra = read_parquet(key)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {key}: {type(exc).__name__}, skipped")
            continue
        new = set(extra.id) - set(inv.id)
        print(f"  {key}: {len(extra):,} rows, {len(new):,} ids not already present")
        parts.append(extra)
    merged = pd.concat(parts, ignore_index=True)
    before = len(merged)
    merged = merge_epochs(merged)
    print(f"  merged: {before:,} rows -> {len(merged):,} unique station ids")

    dup = merged["id"][merged["id"].duplicated()].unique()
    if len(dup):
        raise ValueError(f"{len(dup)} duplicated id(s) survived the merge: {sorted(dup)[:5]}")
    print("  duplicate-id guard: clean")

    with_picks, counts = stations_with_picks(campaign)
    orphan = with_picks - set(merged.id)
    print(f"  stations with picks still missing after the merge: {len(orphan):,}")
    if orphan:
        raise ValueError(f"the merge does not cover {len(orphan)} station(s) that have "
                         f"picks, e.g. {sorted(orphan)[:5]}")

    if not write:
        print("\n  dry run, nothing written. Pass --write to publish.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{campaign}/.backup/stations-{stamp}.parquet"
    body = _s3.get_object(Bucket=BUCKET, Key=f"{campaign}/stations.parquet")["Body"].read()
    _s3.put_object(Bucket=BUCKET, Key=backup, Body=body)
    print(f"\n  backed up the current table to s3://{BUCKET}/{backup} ({len(body) / 1e6:.2f} MB)")

    from sb_catalog.src.s3_state import prepare_station_dates
    merged = prepare_station_dates(merged)
    buf = io.BytesIO()
    merged.to_parquet(buf, index=False)
    _s3.put_object(Bucket=BUCKET, Key=f"{campaign}/stations.parquet", Body=buf.getvalue())
    print(f"  wrote {len(merged):,} stations to s3://{BUCKET}/{campaign}/stations.parquet")

    after = read_parquet(f"{campaign}/stations.parquet")
    missing = with_picks - set(after.id)
    print(f"  verified from the bucket: {len(after):,} rows, "
          f"{len(missing)} station(s) with picks still missing")
    if missing:
        raise SystemExit("the published table is still incomplete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default="western")
    ap.add_argument("--check", action="store_true", help="report the invariant, write nothing")
    ap.add_argument("--dry-run", action="store_true", help="build the merge, write nothing")
    ap.add_argument("--write", action="store_true", help="publish, after backing up")
    a = ap.parse_args()
    if a.check:
        check(a.campaign)
    else:
        merge(a.campaign, write=a.write)
