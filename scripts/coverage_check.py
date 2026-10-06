#!/usr/bin/env python
"""Is every station-day the station table says existed planned in some queue?

The invariant: union over every queue that writes into a catalogue covers the
table's operating windows inside the catalogue's span. It failed silently
twice in 2026: the date misread (121,692 station-days, docs/rerun_2026/30) and
the fill stations merged into the table after their date repair was planned
(25,919, the 30 addendum). Both were found by hand, weeks late.

Reports never-planned station-days by network and station, and which queues
were planned from an older version of the table (their `plan.json`). Exits 1
when the gap exceeds --max-gap station-days.

    python scripts/coverage_check.py --catalogue western
    python scripts/coverage_check.py --catalogue western --table stations_with_epochs.parquet

Needs credentials: the queues are not public.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import io
import json
import sys
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb_catalog.src.shard_planner import _operating_windows  # noqa: E402

BUCKET = "quakescope-picks-2026"
# The span each catalogue holds, end exclusive like the queues.
SPAN = {"western": (datetime.date(1986, 1, 1), datetime.date(2026, 9, 8)),
        "obs": (datetime.date(1993, 1, 1), datetime.date(2026, 1, 1)),
        "global": (datetime.date(2010, 1, 1), datetime.date(2026, 9, 8))}


def queues_for(s3, cat: str) -> list[str]:
    fleet = json.load(open(Path(__file__).resolve().parents[1] / "fleet.json"))["campaigns"]
    named = {c["queue"].rstrip("/").split("/")[-1] for c in fleet.values()
             if c.get("parquet_uri", "").rstrip("/").endswith(f"/{cat}")}
    listed = {c["Prefix"].split("/")[1] for c in s3.list_objects_v2(
        Bucket=BUCKET, Prefix="_queues/", Delimiter="/")["CommonPrefixes"]
        if c["Prefix"].split("/")[1] == cat or c["Prefix"].split("/")[1].startswith(cat + "-")}
    return sorted(named | listed)


def intervals(windows, lo, hi):
    out = []
    for a, b in windows:
        a, b = max(a, lo), min(b + datetime.timedelta(days=1), hi)
        if b > a:
            out.append((a, b))
    return out


def uncovered(want, have):
    """Days in `want` intervals not in any `have` interval (both [a, b))."""
    have = sorted(have)
    n, runs = 0, []
    for a, b in sorted(want):
        cur = a
        for x, y in have:
            if y <= cur or x >= b:
                continue
            if x > cur:
                runs.append((cur, x)); n += (x - cur).days
            cur = max(cur, y)
            if cur >= b:
                break
        if cur < b:
            runs.append((cur, b)); n += (b - cur).days
    return n, runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True, choices=sorted(SPAN))
    ap.add_argument("--table", help="station table to check against (default: the published one)")
    ap.add_argument("--max-gap", type=int, default=0)
    ap.add_argument("--out", help="CSV of never-planned runs per station")
    a = ap.parse_args()
    s3 = boto3.client("s3", region_name="us-east-2")
    cat = a.catalogue
    lo, hi = SPAN[cat]
    key = f"{cat}/stations.parquet"
    st = pd.read_parquet(a.table) if a.table else pd.read_parquet(
        io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))
    current = s3.head_object(Bucket=BUCKET, Key=key).get("VersionId")

    have = collections.defaultdict(list)
    print(f"{cat}: span {lo} to {hi} (end exclusive), table {len(st):,} rows")
    for q in queues_for(s3, cat):
        try:
            body = s3.get_object(Bucket=BUCKET, Key=f"_queues/{q}/shards.jsonl")["Body"].read()
        except s3.exceptions.NoSuchKey:
            continue
        n = 0
        for line in body.decode().splitlines():
            if not line.strip():
                continue
            d = json.loads(line); n += 1
            s0 = datetime.datetime.strptime(d["start"], "%Y.%j").date()
            s1 = datetime.datetime.strptime(d["end"], "%Y.%j").date()
            for tid in d["stations"]:
                have[tid].append((s0, s1))
        try:
            basis = json.loads(s3.get_object(Bucket=BUCKET, Key=f"_queues/{q}/plan.json")["Body"].read())
            v = (basis.get("stations") or {}).get("version_id")
            note = "current table" if v == current else f"planned from table version {v}"
        except s3.exceptions.NoSuchKey:
            note = "no plan.json (planned before 2026-10-06; basis unknown)"
        print(f"  {q:24s} {n:7,d} shards  {note}")

    win = _operating_windows(st)
    rows, total = [], 0
    for tid in st["id"].astype(str):
        w = win.get(tid) or [(lo, hi - datetime.timedelta(days=1))]
        n, runs = uncovered(intervals(w, lo, hi), have.get(tid, []))
        if n:
            total += n
            rows += [(tid, tid.split(".")[0], r0, r1 - datetime.timedelta(days=1), (r1 - r0).days)
                     for r0, r1 in runs]
    df = pd.DataFrame(rows, columns=["id", "network", "first", "last", "days"])
    print(f"\nnever planned: {total:,} station-days on {df.id.nunique() if len(df) else 0:,} station-locations")
    if len(df):
        print("by network:", df.groupby("network").days.sum().sort_values(ascending=False).head(12).to_dict())
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"runs written to {a.out}")
    sys.exit(1 if total > a.max_gap else 0)


if __name__ == "__main__":
    main()
