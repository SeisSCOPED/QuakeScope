#!/usr/bin/env python
"""Build `<catalogue>/availability/`: one row per station-day, what we know about it.

The picks say where arrivals were found; they cannot say why a station-day has
none. The shard manifests can: `records` lists every station-day that reached
the picker with its pick count, and since image d0ccf9b (2026-10-06) `outcomes`
gives every planned station-day a status. This folds both into a table a reader
can query without fetching ~160,000 manifest objects. Asked for by a user
(2026-10-08) who wanted "stations that have records but no picks".

Statuses, most informative first:

    loaded        read and picked; `npks` picks (0 = data, no arrival above 0.2)
    no_data       the archive listing held nothing for the station-day
    not_found     the archive has no such network-year
    no_channel    the station offers no pickable band
    empty_read    an object exists but holds no pickable band
    denied        our account may not read it
    unread        planned and attempted, not read (timeout, throttled, too_big, ...);
                  `detail` says which; a repair queue re-runs these
    unknown       inside the station's operating epochs, no record either way:
                  a shard run before d0ccf9b that found nothing to report

A station-day with several facts keeps the most informative; `loaded` with
picks always wins. Written as Parquet partitioned by network, alongside
`picks/`, never inside it.

    python scripts/build_availability.py --catalogue western --epochs E.parquet --out s3://.../western/availability
    python scripts/build_availability.py --catalogue western --epochs E.parquet --out ./availability --dry-run
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime
import json
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb_catalog.src.shard_planner import _operating_windows  # noqa: E402

BUCKET = "quakescope-picks-2026"
SPAN = {"western": (datetime.date(1986, 1, 1), datetime.date(2026, 9, 8)),
        "obs": (datetime.date(1993, 1, 1), datetime.date(2026, 10, 2))}
UNREAD = {"refused", "throttled", "timeout", "read_error", "too_big"}
# Lower is more informative; a station-day keeps its lowest rank.
RANK = {"loaded": 0, "no_data": 1, "not_found": 2, "no_channel": 3, "empty_read": 4,
        "denied": 5, "unread": 6, "unknown": 7}
SCHEMA = pa.schema([("tid", pa.string()), ("date", pa.date32()), ("status", pa.string()),
                    ("cha", pa.string()), ("npks", pa.int32()), ("detail", pa.string())])


def scan_manifests(cat: str, threads: int = 128) -> list[dict]:
    s3 = boto3.client("s3", region_name="us-east-2",
                      config=Config(max_pool_connections=threads,
                                    retries={"max_attempts": 12, "mode": "adaptive"}))
    keys = [o["Key"] for pg in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=f"{cat}/manifests/") for o in pg.get("Contents", [])
        if o["Key"].endswith(".json")]
    print(f"{cat}: {len(keys):,} manifests", flush=True)

    def get(k):
        for a in range(6):
            try:
                return json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
            except Exception:
                time.sleep(2 ** a)
        return None

    out, failed = [], 0
    with cf.ThreadPoolExecutor(threads) as ex:
        for m in ex.map(get, keys, chunksize=64):
            if m is None:
                failed += 1
                continue
            for r in m.get("records", []):
                out.append((r["tid"], r["yr"], r["doy"], "loaded", r.get("cha"), r.get("npks", 0), None))
            for o in m.get("outcomes", []):
                st = o["status"]
                if st == "done":
                    continue                    # covered by an earlier attempt's records
                if st == "loaded":
                    out.append((o["tid"], o["yr"], o["doy"], "loaded", o.get("cha"), None, None))
                elif st in UNREAD:
                    out.append((o["tid"], o["yr"], o["doy"], "unread", None, None, st))
                else:
                    out.append((o["tid"], o["yr"], o["doy"], st, None, None, o.get("detail")))
    if failed:
        raise SystemExit(f"{failed} manifests could not be read; not writing a partial table")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True, choices=sorted(SPAN))
    ap.add_argument("--epochs", required=True, help="station table with `epochs` (add_station_epochs.py)")
    ap.add_argument("--out", required=True, help="s3://.../<catalogue>/availability or a local dir")
    ap.add_argument("--dry-run", action="store_true", help="summarise, write nothing")
    a = ap.parse_args()
    lo, hi = SPAN[a.catalogue]

    rows = scan_manifests(a.catalogue)
    df = pd.DataFrame(rows, columns=["tid", "yr", "doy", "status", "cha", "npks", "detail"])
    df["date"] = pd.to_datetime(df.yr.astype(str) + df.doy.astype(str).str.zfill(3), format="%Y%j").dt.date
    df["rank"] = df.status.map(RANK)
    # Most informative fact per station-day; among `loaded`, the row that
    # carries a pick count (from `records`) over the outcome that does not.
    df["has_n"] = df.npks.notna()
    df = (df.sort_values(["rank", "has_n"], ascending=[True, False])
            .drop_duplicates(["tid", "date"]).drop(columns=["yr", "doy", "rank", "has_n"]))
    print(f"facts from manifests: {len(df):,} station-days", flush=True)

    # Fill the rest of each station's operating epochs as `unknown`.
    st = pd.read_parquet(a.epochs)
    known = set(zip(df.tid, df.date))
    fill = []
    for tid, windows in _operating_windows(st).items():
        for s, e in windows:
            d, e = max(s, lo), min(e, hi - datetime.timedelta(days=1))
            while d <= e:
                if (tid, d) not in known:
                    fill.append((tid, d))
                d += datetime.timedelta(days=1)
    unk = pd.DataFrame(fill, columns=["tid", "date"]).assign(status="unknown", cha=None, npks=None, detail=None)
    df = pd.concat([df, unk], ignore_index=True)
    df["npks"] = df.npks.astype("Int32")
    df["network"] = df.tid.str.split(".").str[0]
    print(df.status.value_counts().to_string(), flush=True)
    print(f"loaded with zero picks (data, no arrival): {(df.status.eq('loaded') & df.npks.eq(0)).sum():,}")
    if a.dry_run:
        return

    for net, g in df.groupby("network"):
        t = pa.Table.from_pandas(g.sort_values(["tid", "date"])[[f.name for f in SCHEMA]],
                                 schema=SCHEMA, preserve_index=False)
        path = f"{a.out.rstrip('/')}/network={net}/availability.parquet"
        if path.startswith("s3://"):
            import s3fs
            with s3fs.S3FileSystem().open(path, "wb") as fh:
                pq.write_table(t, fh, compression="zstd")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, path, compression="zstd")
    print(f"wrote {df.network.nunique()} network partitions to {a.out}")


if __name__ == "__main__":
    main()
