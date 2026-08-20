"""
Plan a campaign into shards and write the work queue to S3.

This is the v3 replacement for `submit_helper.submit_pick_jobs`, minus the
submitting: it does the same station/day grouping the 2025 campaign used - 40
stations x 20 days per unit of work - but instead of calling `batch.submit_job`
once per unit it writes the whole queue to `shards.jsonl` and stops. Workers
then pull from that queue (see `worker.py`).

Splitting planning from execution is what lets a preempted Spot worker be
replaced without re-planning, and what makes a campaign resumable by inspecting
S3 alone. The queue is immutable once written, because completed work is keyed
on shard id.

Usage:
    python -m sb_catalog.src.shard_planner \\
        --campaign s3://quakescope-picks-2026/scedc \\
        --start 2019.001 --end 2019.365 \\
        --network CI --stations s3://.../stations.parquet
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import logging

import pandas as pd

from .s3_state import S3CampaignState

logger = logging.getLogger("shard_planner")

STATION_GROUP_SIZE = 40      # as run in 2025
DAY_GROUP_SIZE = 20


def parse_year_day(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, "%Y.%j").date()


def shard_id(stations: list[str], start: datetime.date, end: datetime.date) -> str:
    """Stable id: same inputs always produce the same shard, so re-planning an
    identical campaign recognises work that is already done."""
    digest = hashlib.sha1(
        ("|".join(sorted(stations)) + f"|{start:%Y.%j}|{end:%Y.%j}").encode()
    ).hexdigest()[:12]
    return f"{start:%Y%j}-{end:%Y%j}-{digest}"


def plan(
    stations: pd.DataFrame,
    start: datetime.date,
    end: datetime.date,
    station_group_size: int = STATION_GROUP_SIZE,
    day_group_size: int = DAY_GROUP_SIZE,
) -> list[dict]:
    """Group stations and days exactly as the 2025 Batch campaign did."""
    ids = sorted(stations["id"].astype(str))
    days = pd.date_range(start, end, freq="D")       # end inclusive
    shards = []
    for i in range(0, len(ids), station_group_size):
        group = ids[i: i + station_group_size]
        for j in range(0, len(days), day_group_size):
            d0 = days[j].date()
            # END IS EXCLUSIVE, matching S3DataSource.load_waveforms, which does
            # np.arange(start, end). An inclusive end here silently drops the
            # last day of every shard - 1 day in 20 at the default grouping.
            last = days[min(j + day_group_size - 1, len(days) - 1)].date()
            d1 = last + datetime.timedelta(days=1)
            shards.append({
                "shard_id": shard_id(group, d0, d1),
                "stations": group,
                "start": f"{d0:%Y.%j}",
                "end": f"{d1:%Y.%j}",
                "n_station_days": len(group) * (d1 - d0).days,
            })
    return shards


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True,
                    help="s3://bucket/campaign - the campaign state prefix")
    ap.add_argument("--start", required=True, type=parse_year_day, help="YYYY.DDD (inclusive)")
    ap.add_argument("--end", required=True, type=parse_year_day, help="YYYY.DDD (inclusive)")
    ap.add_argument("--stations", help="Parquet/CSV of station metadata to install "
                                       "into the campaign before planning")
    ap.add_argument("--network", help="Comma separated network codes")
    ap.add_argument("--extent", help="minlat,maxlat,minlon,maxlon")
    ap.add_argument("--station-group-size", type=int, default=STATION_GROUP_SIZE)
    ap.add_argument("--day-group-size", type=int, default=DAY_GROUP_SIZE)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the plan without writing the queue")
    args = ap.parse_args(argv)

    state = S3CampaignState(args.campaign)

    if args.stations:
        df = (pd.read_parquet(args.stations) if args.stations.endswith(".parquet")
              else pd.read_csv(args.stations))
        state.write_stations(df)

    extent = tuple(float(x) for x in args.extent.split(",")) if args.extent else None
    stations = state.get_stations(extent=extent, network=args.network)
    if stations.empty:
        raise SystemExit("No stations matched - check --network/--extent against stations.parquet")

    shards = plan(stations, args.start, args.end,
                  args.station_group_size, args.day_group_size)
    total_sd = sum(s["n_station_days"] for s in shards)
    logger.info(
        f"{len(stations)} stations x {(args.end - args.start).days + 1} days "
        f"-> {len(shards)} shards, {total_sd:,} station-days"
    )
    if args.dry_run:
        for s in shards[:3]:
            logger.info(f"  e.g. {s['shard_id']}  {s['start']}..{s['end']}  "
                        f"{len(s['stations'])} stations")
        return

    state.write_shards(shards)
    logger.info(f"Queue ready. Launch workers against {args.campaign}")


if __name__ == "__main__":
    main()
