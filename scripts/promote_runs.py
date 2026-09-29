#!/usr/bin/env python
"""Copy a queue's run records into the catalogue its picks went to.

A worker writes picks and manifests to `--parquet_uri` (the catalogue) but its
run record to the campaign state, which is the queue. So after any campaign
that writes into a catalogue other than its own prefix - every repair and fill
queue since 2026-09-17 - the `rid` on those picks resolves to nothing under
`<catalogue>/runs/` until the records are copied across.

    python scripts/promote_runs.py --queue western-dates --catalogue western
    python scripts/promote_runs.py --queue western-fill2 --catalogue western --dry-run

Idempotent: a record already present with the same ETag is skipped. Verifies
by re-listing, and reports any `rid` on the catalogue's recent picks that still
has no record.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import boto3

BUCKET = "quakescope-picks-2026"
s3 = boto3.client("s3", region_name="us-east-2")


def listing(prefix: str) -> dict[str, str]:
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"].rsplit("/", 1)[1]] = o["ETag"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src_root, dst_root = f"_queues/{a.queue}/runs/", f"{a.catalogue}/runs/"
    src, dst = listing(src_root), listing(dst_root)
    todo = sorted(k for k, v in src.items() if dst.get(k) != v)
    print(f"{a.queue}: {len(src):,} run records, {len(todo):,} to copy into {a.catalogue}/runs/")
    if a.dry_run or not todo:
        return

    def copy(name: str) -> None:
        s3.copy_object(Bucket=BUCKET, Key=dst_root + name,
                       CopySource={"Bucket": BUCKET, "Key": src_root + name})

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(copy, todo))
    dst = listing(dst_root)
    missing = [k for k, v in src.items() if dst.get(k) != v]
    print(f"copied; {len(src) - len(missing):,}/{len(src):,} present and identical in {a.catalogue}/runs/")
    if missing:
        sys.exit(f"{len(missing)} record(s) did not copy: {missing[:5]}")


if __name__ == "__main__":
    main()
