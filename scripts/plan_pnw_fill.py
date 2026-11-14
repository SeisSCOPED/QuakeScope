#!/usr/bin/env python
"""Plan `pnw-eqt-fill`: extend the PNW EQTransformer catalogue (pnw-eqt-2023/) past 2023.

The 2002-2023 picks in pnw-eqt-2023/ were made in 2022 by Ni et al. (2023) with
EQTransformer `pnw` weights at threshold 0.1. This queue runs the same model and
threshold through the 2026 worker from 2024-01-01 to --end, writing into the
same prefix, with the 2026 amplitudes (`amp_vel`, Wood-Anderson `amp`) that the
archive lacks.

Station set, both from tables already on S3:

* every station-location with picks in pnw-eqt-2023 in 2023 (its manifests);
* every station-location of western/stations.parquet in Washington, Oregon,
  Idaho or British Columbia.

Only the bands the archive used are kept (HH, BH, EH): accelerometers and other
bands are stripped from `channels`, and a station-location left with none is
dropped. Epochs are western's, so each station is planned only for days it
operated.

    python scripts/plan_pnw_fill.py --dry-run
    python scripts/plan_pnw_fill.py --write

`--write` installs the station table and the queue under _queues/pnw-eqt-fill/.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb_catalog.src.shard_planner import parse_year_day, plan  # noqa: E402
from sb_catalog.src.utils import normalize_station_codes  # noqa: E402

BUCKET = "quakescope-picks-2026"
CATALOGUE = "pnw-eqt-2023"
QUEUE = f"s3://{BUCKET}/_queues/pnw-eqt-fill"
START = "2024.001"
STATES = {"Washington", "Oregon", "Idaho", "British Columbia (Ca)"}
BANDS = ("HH", "BH", "EH")


def keep_bands(channels: str) -> str:
    """'HH,HN,EH' -> 'HH,EH'; order kept, other bands dropped."""
    return ",".join(b for b in str(channels or "").split(",") if b.strip() in BANDS)


def archive_2023_ids() -> set[str]:
    """Station-locations with picks in pnw-eqt-2023 during 2023."""
    import boto3
    s3 = boto3.client("s3", region_name="us-east-2")
    m = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{CATALOGUE}/manifests/pnweqt-2023.json")["Body"].read())
    return {r["tid"] for r in m["records"]}


def select(western: pd.DataFrame, ids_2023: set[str]) -> pd.DataFrame:
    sel = western[western["id"].isin(ids_2023) | western["state"].isin(STATES)].copy()
    sel["channels"] = sel["channels"].map(keep_bands)
    return sel[sel["channels"] != ""].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", default=None,
                    help="last day planned, YYYY.DDD (default: four days ago, for archive latency)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    end = a.end or (datetime.date.today() - datetime.timedelta(days=4)).strftime("%Y.%j")
    western = pd.read_parquet(f"s3://{BUCKET}/western/stations.parquet")
    ids_2023 = archive_2023_ids()
    table = normalize_station_codes(select(western, ids_2023))
    missing = ids_2023 - set(western["id"])
    print(f"2023 archive stations: {len(ids_2023):,}; not in western (not planned): {len(missing)}"
          + (f" e.g. {sorted(missing)[:5]}" if missing else ""))
    print(f"station-locations planned: {len(table):,} "
          f"({table['state'].value_counts().to_dict()})")

    shards = plan(table, parse_year_day(START), parse_year_day(end))
    ids = [s["shard_id"] for s in shards]
    assert len(ids) == len(set(ids)), "duplicate shard ids"
    planned = sum(s["n_station_days"] for s in shards)
    print(f"TOTAL {START}..{end}: {len(shards):,} shards, {planned:,} planned station-location-days")
    if a.dry_run:
        return

    from sb_catalog.src.s3_state import S3CampaignState
    state = S3CampaignState(QUEUE)
    state.write_stations(table)
    print(state.write_shards(shards))
    import boto3
    boto3.client("s3", region_name="us-east-2").put_object(
        Bucket=BUCKET, Key="_queues/pnw-eqt-fill/README.json", ContentType="application/json",
        Body=json.dumps({"purpose": "extend pnw-eqt-2023 past 2023 with EQTransformer pnw at 0.1, "
                                    "with 2026 amplitudes; scripts/plan_pnw_fill.py",
                         "parquet_uri": f"s3://{BUCKET}/{CATALOGUE}", "span": [START, end],
                         "model": "EQTransformer", "weight": "pnw", "threshold": 0.1,
                         "station_locations": len(table), "shards": len(shards),
                         "planned_station_location_days": int(planned),
                         "planned": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                        indent=1).encode())


if __name__ == "__main__":
    main()
