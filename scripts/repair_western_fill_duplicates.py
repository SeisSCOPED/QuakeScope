#!/usr/bin/env python
"""Recover the western-fill shards that a repeated station id made unrunnable.

`plan_western_fill.py` wrote one row per FDSN *epoch*, so 25 station-locations
appear two or three times in `_queues/western-fill/stations.parquet` (33 extra
rows). `S3DataSource` indexes metadata by id and reads
`meta.loc[station, "channels"]` expecting a string; with a repeated id that is
a Series, and every shard naming one of those stations dies on
`'Series' object has no attribute 'split'`, is released, and is rediscovered by
the next worker. 662 of 28,218 shards span that way from 2026-09-25 to 09-29,
burning 2,409 vCPU-hours (~$51) and completing nothing.

The repeated ids also reached the queue: those 662 shards list the same id two
or three times, so even against a corrected table the deployed worker would
read and pick those station-days twice over. A queue is immutable, so the
shards cannot be edited in place.

This does three things:

1. merges the duplicate epochs in `_queues/western-fill/stations.parquet`
   (union of bands, earliest start, latest end) and writes it back;
2. writes a `blocked/` record for each of the 662 shards, so the queue is
   honest about them and nothing retries them if the target is raised;
3. re-plans exactly those station-days into `_queues/western-fill2/`, with
   de-duplicated station lists, writing into the same `western` catalogue.

    python scripts/repair_western_fill_duplicates.py --dry-run
    python scripts/repair_western_fill_duplicates.py --write
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
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb_catalog.src.shard_planner import shard_id                    # noqa: E402
from sb_catalog.src.utils import station_date                        # noqa: E402

BUCKET = "quakescope-picks-2026"
OLD, NEW, CATALOGUE = "western-fill", "western-fill2", "western"
s3 = boto3.client("s3", region_name="us-east-2")


def listing(prefix: str) -> set[str]:
    out = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            out.add(o["Key"].rsplit("/", 1)[1].removesuffix(".json"))
    return out


def merge_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station-location: union the bands, widen the window."""
    def bands(v):
        return ",".join(sorted({b for x in v for b in str(x).split(",") if b and b != "nan"}))
    agg = dict(network_code=("network_code", "first"), station_code=("station_code", "first"),
               location_code=("location_code", "first"), channels=("channels", bands),
               latitude=("latitude", "first"), longitude=("longitude", "first"),
               elevation=("elevation", "first"),
               start_date=("start_date", "min"), end_date=("end_date", "max"))
    # The float columns must follow their date twins, or the two disagree:
    # start_yearday with the earliest start, end_yearday with the latest end.
    for extra, how in (("state", "first"), ("start_yearday", "min"), ("end_yearday", "max")):
        if extra in df.columns:
            agg[extra] = (extra, how)
    return df.groupby("id", as_index=False).agg(**agg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    st = pd.read_parquet(io.BytesIO(
        s3.get_object(Bucket=BUCKET, Key=f"_queues/{OLD}/stations.parquet")["Body"].read()))
    dups = set(st["id"][st["id"].duplicated()])
    merged = merge_epochs(st)
    print(f"stations: {len(st):,} rows -> {len(merged):,} unique ids "
          f"({len(dups)} ids had more than one epoch)")

    shards = [json.loads(l) for l in s3.get_object(
        Bucket=BUCKET, Key=f"_queues/{OLD}/shards.jsonl")["Body"].read().decode().splitlines() if l.strip()]
    done, blocked, review = listing(f"_queues/{OLD}/complete/"), listing(f"_queues/{OLD}/blocked/"), listing(f"_queues/{OLD}/review/")
    stuck = [s for s in shards
             if s["shard_id"] not in done | blocked | review
             and any(t in dups for t in s["stations"])]
    others = [s for s in shards if s["shard_id"] not in done | blocked | review and s not in stuck]
    print(f"shards: {len(shards):,} | complete {len(done):,} | blocked {len(blocked)} | review {len(review)} "
          f"| stuck on a repeated id {len(stuck)} | other unfinished {len(others)}")

    # Windows from the merged table, so n_station_days is right for the re-plan.
    win = {r.id: (station_date(r.start_date), station_date(r.end_date))
           for r in merged.itertuples()}

    def overlap(tid, d0, d1):
        w = win.get(tid)
        if not w or w[0] is None or w[1] is None:
            return (d1 - d0).days
        lo, hi = max(w[0], d0), min(w[1], d1 - datetime.timedelta(days=1))
        return max((hi - lo).days + 1, 0)

    new_shards = []
    for s in stuck:
        d0 = datetime.datetime.strptime(s["start"], "%Y.%j").date()
        d1 = datetime.datetime.strptime(s["end"], "%Y.%j").date()
        tids = sorted(set(s["stations"]))                     # the whole point
        sd = sum(overlap(t, d0, d1) for t in tids)
        if not sd:
            continue
        new_shards.append(dict(shard_id=shard_id(tids, d0, d1), stations=tids,
                               start=s["start"], end=s["end"], n_station_days=sd))
    print(f"re-plan: {len(new_shards):,} shards, {sum(s['n_station_days'] for s in new_shards):,} station-days")
    if a.dry_run:
        return

    # 1. the corrected table, in place
    buf = io.BytesIO(); pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), buf, compression="snappy")
    s3.put_object(Bucket=BUCKET, Key=f"_queues/{OLD}/stations.parquet", Body=buf.getvalue())
    print(f"wrote the merged table to _queues/{OLD}/stations.parquet")

    # 2. retire the stuck shards
    when = datetime.datetime.utcnow().isoformat() + "Z"
    for s in stuck:
        s3.put_object(Bucket=BUCKET, Key=f"_queues/{OLD}/blocked/{s['shard_id']}.json",
                      ContentType="application/json",
                      Body=json.dumps(dict(shard_id=s["shard_id"], kind="superseded", noted=when,
                                           reason=f"station list repeats an id, which the deployed reader cannot "
                                                  f"handle; re-planned de-duplicated in {NEW}")).encode())
    print(f"blocked {len(stuck):,} shards in {OLD}")

    # 3. the new queue
    key = f"_queues/{NEW}/shards.jsonl"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        sys.exit(f"{key} exists; a queue is immutable. Delete the prefix to re-plan.")
    except s3.exceptions.ClientError:
        pass
    s3.put_object(Bucket=BUCKET, Key=key, ContentType="application/x-ndjson",
                  Body="\n".join(json.dumps(s) for s in new_shards).encode() + b"\n")
    s3.copy_object(Bucket=BUCKET, Key=f"_queues/{NEW}/stations.parquet",
                   CopySource={"Bucket": BUCKET, "Key": f"_queues/{OLD}/stations.parquet"})
    for name in ("access.json",):
        try:
            s3.copy_object(Bucket=BUCKET, Key=f"_queues/{NEW}/{name}",
                           CopySource={"Bucket": BUCKET, "Key": f"_queues/{OLD}/{name}"})
        except s3.exceptions.ClientError:
            pass
    s3.put_object(Bucket=BUCKET, Key=f"_queues/{NEW}/README.json", ContentType="application/json",
                  Body=json.dumps(dict(
                      purpose=f"the {OLD} shards whose station list repeated an id, re-planned de-duplicated",
                      parent=OLD, parquet_uri=f"s3://{BUCKET}/{CATALOGUE}", shards=len(new_shards),
                      station_days=sum(s["n_station_days"] for s in new_shards), planned=when,
                      note="access.json copied from the parent, already surveyed"), indent=1).encode())
    print(f"wrote s3://{BUCKET}/{key}")


if __name__ == "__main__":
    main()
