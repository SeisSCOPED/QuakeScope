#!/usr/bin/env python
"""Plan the re-read of every station-day a catalogue never recorded.

Background: docs/rerun_2026/32. Until image d0ccf9b the reader chose one band
per station for the whole campaign, and an EarthScope object that lacked it
raised inside libmseed and came back empty; either way the station-day left no
trace. Such a day is inside the station's operating epochs but has no record in
any manifest. This plans exactly those days, on the fixed reader, which records
an outcome for each.

Inputs: an epoch table (scripts/add_station_epochs.py) and the recorded
station-days of the catalogue (a directory of Parquet parts with tid, yr, doy,
from scanning `<catalogue>/manifests/`). Excluded: networks the account cannot
read (`--skip-networks`, default TD,EO,LH: blocked in the queues as "not
readable with this account"), and stations with no pickable band at all.

    python scripts/plan_unread_repair.py --catalogue western --epochs E.parquet \
        --recorded recs/ --queue western-unread --dry-run
"""
from __future__ import annotations

import argparse
import collections
import datetime
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb_catalog.src.constants import channel_priority          # noqa: E402
from sb_catalog.src.shard_planner import _operating_windows, shard_id  # noqa: E402
from sb_catalog.src.utils import normalize_station_codes       # noqa: E402

BUCKET = "quakescope-picks-2026"
SPAN = {"western": (datetime.date(1986, 1, 1), datetime.date(2026, 9, 8))}
MAX_DAYS = 20


def make_shards(runs: dict, lo: datetime.date, max_stations: int = 40,
                max_sd: int = 800, min_sd: int = 20) -> tuple[list, list]:
    """Shards from {(network, first, end): [station ids]} day-offset runs.

    Stations sharing a run go together, capped at `max_stations` and at
    `max_sd` station-days per shard. Station cap as well as station-day cap:
    a shard opens with one FDSN inventory request naming every station, and
    production shards never named more than 40; capping station-days alone
    put up to 800 stations in a one-day shard (706 shards in the first
    western-reread). Runs under `min_sd` station-days come back as holes.
    """
    shards, holes = [], []
    for (net, x, y), tids in sorted(runs.items()):
        d0, d1 = lo + datetime.timedelta(days=int(x)), lo + datetime.timedelta(days=int(y))
        if len(tids) * (y - x) < min_sd:
            holes += [(t, f"{d0:%Y.%j}", f"{d1:%Y.%j}", int(y - x)) for t in tids]
            continue
        per = max(1, min(max_stations, max_sd // (y - x)))
        for j in range(0, len(tids), per):
            g = sorted(tids[j:j + per])
            shards.append(dict(shard_id=shard_id(g, d0, d1), stations=g,
                               start=f"{d0:%Y.%j}", end=f"{d1:%Y.%j}",
                               n_station_days=len(g) * int(y - x)))
    return shards, holes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", default="western", choices=sorted(SPAN))
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--recorded", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--skip-networks", default="TD,EO,LH")
    ap.add_argument("--weight", default="original")
    ap.add_argument("--dump", help="also write the shards to this local .jsonl")
    ap.add_argument("--max-stations", type=int, default=40,
                    help="stations per shard; 40 is the 2025 grouping")
    ap.add_argument("--max-sd", type=int, default=800,
                    help="station-days per shard; 800 is the 2025 grouping, 40 stations x 20 days")
    ap.add_argument("--min-shard-sd", type=int, default=20,
                    help="runs smaller than this are held out of the queue and listed "
                         "(--holes): as shards they cost one FDSN inventory request "
                         "per few station-days")
    ap.add_argument("--holes", help="write the held-out station-day runs to this CSV")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    lo, hi = SPAN[a.catalogue]                       # hi exclusive
    n_days = (hi - lo).days
    st = normalize_station_codes(pd.read_parquet(a.epochs))
    skip = set(a.skip_networks.split(",")) if a.skip_networks else set()
    order = set(channel_priority(a.weight))
    pickable = st["channels"].fillna("").map(
        lambda c: bool({x.strip()[:2] for x in c.split(",")} & order))
    st = st[~st["network_code"].isin(skip) & pickable].reset_index(drop=True)
    idx = {t: i for i, t in enumerate(st["id"])}

    want = np.zeros((len(st), n_days), dtype=bool)
    for tid, windows in _operating_windows(st).items():
        for s, e in windows:
            x, y = max((s - lo).days, 0), min((e - lo).days + 1, n_days)
            if y > x:
                want[idx[tid], x:y] = True
    in_window = int(want.sum())

    recorded = 0
    for batch in ds.dataset(a.recorded, format="parquet").to_batches(columns=["tid", "yr", "doy"]):
        b = batch.to_pandas()
        b = b[b["tid"].isin(idx)]
        if b.empty:
            continue
        off = (pd.to_datetime(b["yr"].astype(str) + b["doy"].astype(str).str.zfill(3),
                              format="%Y%j").dt.date.map(lambda d: (d - lo).days)).to_numpy()
        rows = b["tid"].map(idx).to_numpy()
        ok = (off >= 0) & (off < n_days)
        recorded += int(want[rows[ok], off[ok]].sum())
        want[rows[ok], off[ok]] = False
    unread = int(want.sum())
    print(f"{a.catalogue}: {len(st):,} pickable station-locations (skipping {sorted(skip)}); "
          f"in epochs {in_window:,}; recorded {recorded:,}; never recorded {unread:,}")
    by_net = collections.Counter()
    for i, n in enumerate(want.sum(1)):
        if n:
            by_net[st["network_code"].iat[i]] += int(n)
    print("by network:", dict(by_net.most_common(15)))

    # Runs of consecutive days per station, cut at MAX_DAYS; then shards of
    # stations of one network sharing a run, up to MAX_SD station-days.
    runs = collections.defaultdict(list)
    for i in np.flatnonzero(want.any(1)):
        r = want[i].astype(np.int8)
        d = np.diff(np.r_[0, r, 0])
        for x, y in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            # Cut on a fixed MAX_DAYS grid from the span start, not from each
            # run's own start: stations whose unread runs overlap then share
            # a chunk and a shard. Cutting per run gave 318,145 shards for
            # western (one FDSN inventory request each); the grid gives far
            # fewer for the same station-days.
            k = x
            while k < y:
                edge = min((k // MAX_DAYS + 1) * MAX_DAYS, y)
                runs[(st["network_code"].iat[i], k, edge)].append(st["id"].iat[i])
                k = edge
    shards, holes = make_shards(runs, lo, a.max_stations, a.max_sd, a.min_shard_sd)
    total = sum(s["n_station_days"] for s in shards)
    print(f"queue {a.queue}: {len(shards):,} shards, {total:,} station-days; "
          f"held out {sum(h[3] for h in holes):,} station-days in runs under "
          f"{a.min_shard_sd} station-days per shard")
    if a.holes:
        pd.DataFrame(holes, columns=["id", "start", "end", "days"]).to_csv(a.holes, index=False)
    if a.dump:
        Path(a.dump).write_text("\n".join(json.dumps(s) for s in shards) + "\n")
    if a.dry_run:
        return

    import boto3
    s3 = boto3.client("s3", region_name="us-east-2")
    key = f"_queues/{a.queue}/shards.jsonl"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        sys.exit(f"{key} exists; a queue is immutable.")
    except s3.exceptions.ClientError:
        pass
    s3.put_object(Bucket=BUCKET, Key=key, ContentType="application/x-ndjson",
                  Body="\n".join(json.dumps(s) for s in shards).encode() + b"\n")
    buf = io.BytesIO()
    st.to_parquet(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=f"_queues/{a.queue}/stations.parquet", Body=buf.getvalue())
    s3.put_object(Bucket=BUCKET, Key=f"_queues/{a.queue}/plan.json", ContentType="application/json",
                  Body=json.dumps(dict(
                      planned=datetime.datetime.utcnow().isoformat() + "Z",
                      purpose="station-days inside FDSN epochs with no manifest record; "
                              "docs/rerun_2026/32",
                      catalogue=a.catalogue, parquet_uri=f"s3://{BUCKET}/{a.catalogue}",
                      span=[str(lo), str(hi)], skipped_networks=sorted(skip),
                      stations=dict(source=a.epochs, rows=len(st)),
                      shards=len(shards), station_days=total), indent=1).encode())
    print(f"wrote s3://{BUCKET}/{key}")


if __name__ == "__main__":
    main()
