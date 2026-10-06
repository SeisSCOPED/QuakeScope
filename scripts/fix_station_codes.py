#!/usr/bin/env python
"""Rebuild the code columns of the bucket's station tables from `id` (2026-10-06).

Every `stations.parquet` written before this date carried CSV-parse damage in
`network_code`, `station_code` and `location_code` ("0.0" for "00", "1" for
"001", "" for "NA"); `id` was right throughout, so picking was unaffected.
This rewrites only those three columns, from `id`, through
`utils.normalize_station_codes`, after copying the original to
`_archive/stations-before-codefix-20261006/<key>`. Every other column and the
row order are checked unchanged before the put, and the object is read back
after it. `_archive/` tables are records of past runs and are left as they were.
The bucket is versioned too (noncurrent versions kept 30 days).

    python scripts/fix_station_codes.py --dry-run
    python scripts/fix_station_codes.py --write
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb_catalog.src.utils import STATION_CODE_COLUMNS, normalize_station_codes  # noqa: E402

BUCKET = "quakescope-picks-2026"
BACKUP = "_archive/stations-before-codefix-20261006/"


def table_keys(s3) -> list[str]:
    def sub(p):
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=p, Delimiter="/")
        return [c["Prefix"] for c in r.get("CommonPrefixes", [])]
    keys = []
    for p in ["western/", "obs/", "global/"] + sub("_queues/"):
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=p + "stations.parquet", MaxKeys=1)
        if any(o["Key"] == p + "stations.parquet" for o in r.get("Contents", [])):
            keys.append(p + "stations.parquet")
    return keys


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    s3 = boto3.client("s3", region_name="us-east-2")
    for key in table_keys(s3):
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        old = pd.read_parquet(io.BytesIO(body))
        new = normalize_station_codes(old)
        changed = {c: int((old[c].fillna("").astype(str) != new[c]).sum())
                   for c in STATION_CODE_COLUMNS}
        others = [c for c in old.columns if c not in STATION_CODE_COLUMNS]
        assert list(new.columns) == list(old.columns), key
        pd.testing.assert_frame_equal(old[others], new[others])
        print(f"{key:48s} rows {len(old):6d} changed {changed}", flush=True)
        if not sum(changed.values()) or a.dry_run:
            continue
        s3.copy_object(Bucket=BUCKET, Key=BACKUP + key,
                       CopySource={"Bucket": BUCKET, "Key": key})
        kept = s3.get_object(Bucket=BUCKET, Key=BACKUP + key)["Body"].read()
        assert kept == body, f"backup of {key} does not match the original"
        buf = io.BytesIO()
        new.to_parquet(buf, index=False)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        back = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))
        pd.testing.assert_frame_equal(back, new)
        assert (back["id"] == back["network_code"] + "." + back["station_code"]
                + "." + back["location_code"]).all(), key
        print(f"  written and read back; original at s3://{BUCKET}/{BACKUP}{key}", flush=True)


if __name__ == "__main__":
    main()
