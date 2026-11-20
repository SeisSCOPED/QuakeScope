#!/usr/bin/env python
"""Plan `pnw-eqt-fill-bands`: re-read the pnw-eqt-fill station-days lost to a stale band list.

The fill (`_queues/pnw-eqt-fill/`, scripts/plan_pnw_fill.py) offers each
station only the bands its station table lists. Stations upgraded after that
table was built (UW.VLM listed EH, recording HH since 2025-10-20; CC.USF,
CC.WIZ, IE.EMI listed BH or EH, recording HH) and stations whose data sit
under another location code than the table's (UW.BURLY: table 00, S3 blank)
came back `empty_read` although the day file holds a pickable band. The
worker records which bands the file held ("object holds HH,EN"), so the lost
days are identifiable from the fill's manifests alone.

A station-day is repaired when, in the fill's manifests:

* some location of the station came back `empty_read` with HH, BH or EH in
  the file, and
* no location of the same station was picked that day (`loaded` or `done`).

The second rule keeps duplicate location codes out: IW.MFID. empty while
IW.MFID.00 was picked is not a loss.

Each repaired station is offered HH, BH and EH at every location code known
for it (fill table, western table) plus the blank code, because the file's
location code cannot be read here without EarthScope read access to every
network. A wrong code costs one empty read. Shards hold one station and one
run of consecutive repaired days (at most --max-days), so the worker reads
nothing but repaired days.

    python scripts/plan_pnw_band_repair.py --dry-run
    python scripts/plan_pnw_band_repair.py --write

Run after the fill drains: the plan is built from complete shards only.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb_catalog.src.shard_planner import shard_id  # noqa: E402
from sb_catalog.src.utils import normalize_station_codes  # noqa: E402

BUCKET = "quakescope-picks-2026"
CATALOGUE = "pnw-eqt-2023"
FILL = "_queues/pnw-eqt-fill"
QUEUE = f"s3://{BUCKET}/_queues/pnw-eqt-fill-bands"
BANDS = ("HH", "BH", "EH")
PICKED = {"loaded", "done"}


def _s3():
    import boto3
    return boto3.client("s3", region_name="us-east-2")


def fill_outcomes() -> list[dict]:
    """Every outcome in the fill's manifests (manifests named by shard id, 2YYYYDDD-...)."""
    s3 = _s3()
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=f"{CATALOGUE}/manifests/2"):
        keys += [o["Key"] for o in page.get("Contents", [])]

    def get(k):
        return json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()).get("outcomes", [])

    with ThreadPoolExecutor(32) as ex:
        return [o for oo in ex.map(get, keys) for o in oo]


def held_bands(detail) -> set[str]:
    """'object holds BH,LH,VH' -> {'BH', 'LH', 'VH'}."""
    text = str(detail or "")
    if not text.startswith("object holds "):
        return set()
    return {b.strip() for b in text[len("object holds "):].split(",") if b.strip()}


def lost_days(outcomes: list[dict]) -> dict[str, set[datetime.date]]:
    """{NET.STA: days} where a pickable band was in the file and nothing was picked."""
    picked, lost = set(), collections.defaultdict(set)
    for o in outcomes:
        ns = ".".join(o["tid"].split(".")[:2])
        day = datetime.date(o["yr"], 1, 1) + datetime.timedelta(days=o["doy"] - 1)
        if o["status"] in PICKED:
            picked.add((ns, day))
        elif o["status"] == "empty_read" and held_bands(o.get("detail")) & set(BANDS):
            lost[ns].add(day)
    out = {}
    for ns, days in lost.items():
        keep = {d for d in days if (ns, d) not in picked}
        if keep:
            out[ns] = keep
    return out


def runs(days: set[datetime.date], max_days: int) -> list[tuple[datetime.date, datetime.date]]:
    """Consecutive runs as [start, end) pairs, each at most max_days long."""
    out, cur = [], []
    for d in sorted(days):
        if cur and ((d - cur[-1]).days != 1 or len(cur) >= max_days):
            out.append((cur[0], cur[-1] + datetime.timedelta(days=1)))
            cur = []
        cur.append(d)
    if cur:
        out.append((cur[0], cur[-1] + datetime.timedelta(days=1)))
    return out


def station_table(stations: list[str], fill: pd.DataFrame, western: pd.DataFrame) -> pd.DataFrame:
    """One row per (station, location) to try, offering HH,BH,EH, coordinates from the fill or western."""
    known = pd.concat([fill, western[fill.columns.intersection(western.columns)]], ignore_index=True)
    known["ns"] = known["network_code"].astype(str) + "." + known["station_code"].astype(str)
    rows = []
    for ns in stations:
        k = known[known["ns"] == ns]
        if k.empty:
            raise SystemExit(f"{ns}: in the fill's manifests but in neither station table")
        base = k.iloc[0].to_dict()
        for loc in sorted(set(k["location_code"].fillna("").astype(str)) | {""}):
            r = dict(base)
            r.update(id=f"{ns}.{loc}", location_code=loc, channels=",".join(BANDS),
                     start_date="2024-01-01", end_date="3000-01-01",
                     start_yearday=2024.001, end_yearday=3000.001)
            r.pop("ns", None)
            rows.append(r)
    return normalize_station_codes(pd.DataFrame(rows)[fill.columns])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-days", type=int, default=20,
                    help="longest run of days in one shard (default 20)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    from sb_catalog.src.s3_state import S3CampaignState
    progress = S3CampaignState(f"s3://{BUCKET}/{FILL}").progress()
    print(f"fill queue: {progress}")
    if a.write and progress["complete"] + progress["blocked"] < progress["total"]:
        raise SystemExit("the fill has not drained; plan from complete shards only")

    outcomes = fill_outcomes()
    print(f"fill outcomes: {collections.Counter(o['status'] for o in outcomes)}")
    lost = lost_days(outcomes)
    fill = pd.read_parquet(f"s3://{BUCKET}/{FILL}/stations.parquet")
    western = pd.read_parquet(f"s3://{BUCKET}/western/stations.parquet")
    table = station_table(sorted(lost), fill, western)
    locs = table.groupby(table["network_code"] + "." + table["station_code"])["id"].apply(list).to_dict()

    shards = []
    for ns in sorted(lost):
        for d0, d1 in runs(lost[ns], a.max_days):
            shards.append({"shard_id": shard_id(locs[ns], d0, d1), "stations": locs[ns],
                           "start": f"{d0:%Y.%j}", "end": f"{d1:%Y.%j}",
                           "n_station_days": (d1 - d0).days * len(locs[ns])})
    assert len({s["shard_id"] for s in shards}) == len(shards), "duplicate shard ids"

    n_days = sum(len(v) for v in lost.values())
    print(f"lost station-days: {n_days:,} over {len(lost)} stations")
    for ns, days in sorted(lost.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {ns:12s} {len(days):5d} days  {min(days):%Y.%j}..{max(days):%Y.%j}  try {locs[ns]}")
    print(f"queue: {len(shards):,} shards, {len(table)} station-locations, "
          f"{sum(s['n_station_days'] for s in shards):,} planned station-location-days")
    if a.dry_run:
        return

    state = S3CampaignState(QUEUE)
    state.write_stations(table)
    print(state.write_shards(shards))
    _s3().put_object(
        Bucket=BUCKET, Key="_queues/pnw-eqt-fill-bands/README.json", ContentType="application/json",
        Body=json.dumps({"purpose": "re-read pnw-eqt-fill station-days lost to a stale band list or "
                                    "location code; scripts/plan_pnw_band_repair.py",
                         "parquet_uri": f"s3://{BUCKET}/{CATALOGUE}", "parent": f"s3://{BUCKET}/{FILL}",
                         "model": "EQTransformer", "weight": "pnw", "threshold": 0.1,
                         "lost_station_days": n_days, "stations": len(lost), "shards": len(shards),
                         "planned": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                        indent=1).encode())


if __name__ == "__main__":
    main()
