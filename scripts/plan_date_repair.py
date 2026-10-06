#!/usr/bin/env python
"""Plan the re-pick of station-days a mis-decoded station date kept out of a queue.

Background: docs/rerun_2026/30_station_dates.md. Until 2026-09-29 the planner
decoded `start_date`/`end_date` with `strptime(str(v), "%Y.%j")`, and `str()`
drops a trailing zero, so a day-of-year divisible by ten read ten times small.
Windows were always clipped at the end, so those days were planned for nothing
and never picked.

This compares, per station, the window the planner *used* against the window
the table *meant*, takes the difference that falls inside the catalogue's span,
drops anything already in a queue, and writes what is left as a queue under
`_queues/<catalogue>-dates/` writing into `<catalogue>/`.

    python scripts/plan_date_repair.py --catalogue western --dry-run
    python scripts/plan_date_repair.py --catalogue western --write

Needs credentials: the queues are not public.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb_catalog.src.utils import station_date            # noqa: E402

BUCKET = "quakescope-picks-2026"
# Catalogue -> (the queues that fed it, the span they cover).
FED_BY = {
    # Every queue that ever wrote into the catalogue, repairs included, so a
    # day already planned anywhere is not planned again. western-fill2 and
    # western-dates were missing until 2026-10-06; the first omission was what
    # hid 54 fill stations the misread had dropped outright.
    "western": (["western-early", "western", "western-2026", "western-fill",
                 "western-fill2", "western-dates", "western-repair",
                 "western-2026-repair"],
                (datetime.date(1986, 1, 1), datetime.date(2026, 9, 8))),
    "obs": (["obs-early", "obs", "obs-2026", "obs-dates", "obs-fill", "obs-el",
             "obs-repair", "obs-early-repair"],
            (datetime.date(1993, 1, 1), datetime.date(2026, 1, 1))),
    "global": (["global", "global-2026"],
               (datetime.date(2010, 1, 1), datetime.date(2026, 9, 8))),
}


def as_read(value) -> datetime.date | None:
    """The window the old planner actually used, reproduced exactly."""
    try:
        return datetime.datetime.strptime(str(value), "%Y.%j").date()
    except (ValueError, TypeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True, choices=sorted(FED_BY))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--write", action="store_true")
    ap.add_argument("--queue", help="queue name under _queues/ (default <catalogue>-dates; "
                                    "a queue is immutable, so a second pass needs a new name)")
    ap.add_argument("--dump", help="also write the planned shards to this local .jsonl")
    a = ap.parse_args()
    cat = a.catalogue
    queues, (c0, c1) = FED_BY[cat]

    import boto3
    s3 = boto3.client("s3", region_name="us-east-2")
    st = pd.read_parquet(f"s3://{BUCKET}/{cat}/stations.parquet")
    # start_yearday/end_yearday are what the planner saw. A table converted
    # before this script ran keeps them; one that never had them is already
    # correct and has nothing to repair.
    if "end_yearday" not in st.columns:
        sys.exit(f"{cat}/stations.parquet has no end_yearday: nothing to compare against")

    # What each queue already covers, so a day is never planned twice.
    planned: dict[str, set] = collections.defaultdict(set)
    for q in queues:
        try:
            body = s3.get_object(Bucket=BUCKET, Key=f"_queues/{q}/shards.jsonl")["Body"].read().decode()
        except s3.exceptions.NoSuchKey:
            continue
        for line in body.splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            s0 = datetime.datetime.strptime(d["start"], "%Y.%j").date()
            s1 = datetime.datetime.strptime(d["end"], "%Y.%j").date()
            for tid in d["stations"]:
                planned[tid].add((s0, s1))
    print(f"{cat}: {sum(len(v) for v in planned.values()):,} station-shard entries across {len(queues)} queues")

    def covered(tid, day):
        return any(s0 <= day < s1 for s0, s1 in planned.get(tid, ()))

    rows, missing = [], collections.Counter()
    for tid, net, sy, ey in zip(st.id, st.network_code, st.start_yearday, st.end_yearday):
        if pd.isna(ey):
            continue
        true_end, read_end = station_date(ey), as_read(ey)
        if read_end is None or read_end >= true_end:
            continue                                   # parsed right, or parsed long
        true_start = station_date(sy) or c0
        lo = max(read_end + datetime.timedelta(days=1), true_start, c0)
        hi = min(true_end, c1)
        day = lo
        while day <= hi:
            if not covered(tid, day):
                rows.append((tid, day)); missing[net] += 1
            day += datetime.timedelta(days=1)
    print(f"{len(rows):,} station-days never planned, on {len({t for t, _ in rows}):,} station-locations")
    print("by network:", dict(missing.most_common(8)))
    if not rows:
        return

    # Runs of consecutive days per station, then shards of identical runs.
    by_station = collections.defaultdict(list)
    for tid, day in rows:
        by_station[tid].append(day)
    runs = collections.defaultdict(list)
    for tid, days in by_station.items():
        days = sorted(set(days)); start = prev = days[0]
        for d in days[1:] + [None]:
            if d is None or (d - prev).days > 1:
                runs[(start, prev + datetime.timedelta(days=1))].append(tid)
                if d is not None:
                    start = d
            prev = d if d is not None else prev
    from sb_catalog.src.shard_planner import shard_id
    shards, MAX_SD = [], 200
    for (s0, s1), tids in sorted(runs.items()):
        ndays = (s1 - s0).days
        per = max(1, MAX_SD // ndays)
        for i in range(0, len(tids), per):
            group = sorted(tids[i:i + per])
            shards.append(dict(shard_id=shard_id(group, s0, s1), stations=group,
                               start=f"{s0:%Y.%j}", end=f"{s1:%Y.%j}",
                               n_station_days=len(group) * ndays))
    total = sum(s["n_station_days"] for s in shards)
    print(f"queue: {len(shards):,} shards, {total:,} station-days, largest {max(s['n_station_days'] for s in shards)}")
    if a.dump:
        Path(a.dump).write_text("\n".join(json.dumps(s) for s in shards) + "\n")
        print(f"plan written to {a.dump}")
    if a.dry_run:
        return

    repair = a.queue or f"{cat}-dates"
    key = f"_queues/{repair}/shards.jsonl"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        sys.exit(f"{key} exists; a queue is immutable. Delete the prefix to re-plan.")
    except s3.exceptions.ClientError:
        pass
    s3.put_object(Bucket=BUCKET, Key=key, ContentType="application/x-ndjson",
                  Body="\n".join(json.dumps(s) for s in shards).encode() + b"\n")
    import io
    from sb_catalog.src.utils import normalize_station_codes
    buf = io.BytesIO()
    normalize_station_codes(st).to_parquet(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=f"_queues/{repair}/stations.parquet", Body=buf.getvalue())
    s3.put_object(Bucket=BUCKET, Key=f"_queues/{repair}/README.json", ContentType="application/json",
                  Body=json.dumps(dict(
                      purpose="station-days a mis-decoded station date kept out of the original queues; "
                              "docs/rerun_2026/30_station_dates.md",
                      catalogue=cat, parquet_uri=f"s3://{BUCKET}/{cat}", span=[str(c0), str(c1)],
                      shards=len(shards), station_days=total,
                      planned=datetime.datetime.utcnow().isoformat() + "Z"), indent=1).encode())
    print(f"wrote s3://{BUCKET}/{key}")


if __name__ == "__main__":
    main()
