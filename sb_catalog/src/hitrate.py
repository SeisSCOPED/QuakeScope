"""Hit rate by campaign and year, from S3 listings alone.

The cost model divides real work by PLANNED station-days, and the hit rate -
what fraction of them actually hold data - is the largest remaining unknown
(OPTIMISE item 9). It is a property of the QUEUE, not the picker, so answering
it needs no inference, no model and no picking: one LIST per (network, day)
settles every station in that network at once.

**Runs in the container on purpose.** The 69% of planned station-days behind
EarthScope's restricted tier needs the OAuth refresh token, and that token must
not be used from a laptop: `earthscope_sdk`'s refresh grant saves a rotated
token to local SDK state rather than back to Secrets Manager, so if Auth0
rotation is on, one local run invalidates the credential every campaign job
depends on. Inside a task launched from `quakescope_2026_earthscope`, the token
is already injected and rotation is the production path's own business.

Listing goes through `CompositeS3ObjectHelper.get_filesystem`, the same
credentialed filesystem the picker reads with, so the survey cannot drift from
what a worker would actually find - and it reaches both EarthScope tiers.

Calibration, from every shard the 2026-09-01 runs completed:

    SCEDC       5 shards   listed == picked exactly       correction 1.000
    EarthScope  2 shards   listed 82.4%, picked 68.5%     correction 0.831

SCEDC and NCEDC encode the channel in the object name, so a listing answers
exactly what the picker will find. EarthScope stores one object per station-day
covering every channel, so a listing proves the station had data but not that
the object holds the band `select_channel` chose - hence the correction, which
the two shards agreed on to 0.3% (83.0%, 83.3%). It is applied to reported
EarthScope rates and printed alongside the raw number.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import sys

from .constants import NETWORK_MAPPING, select_channel
from .s3_state import S3CampaignState
from .shard_planner import _operating_windows, _overlaps

logger = logging.getLogger("picker")

# Measured 2026-09-01; see the module docstring.
EARTHSCOPE_CORRECTION = 0.831


def sample_days(start_year: int, end_year: int, per_year: int) -> list:
    """Days spread through each year, deterministic so runs are comparable."""
    out = []
    for y in range(start_year, end_year + 1):
        for k in range(per_year):
            doy = int(40 + (300 / max(per_year, 1)) * k + 37 * k) % 360 + 1
            try:
                out.append(datetime.date(y, 1, 1) +
                           datetime.timedelta(days=doy - 1))
            except ValueError:
                pass
    return out


def _present(s3helper, archive: str, net: str, year: str, doy: str):
    """Station keys with data that day, or None if the listing failed."""
    prefix = s3helper.get_prefix(net, year, doy)
    try:
        names = [k.rsplit("/", 1)[-1]
                 for k in s3helper.get_filesystem(net, int(year)).ls(prefix)]
    except FileNotFoundError:
        return set()
    except Exception as exc:
        logger.warning(f"listing {prefix} failed: {type(exc).__name__}: {exc}")
        return None

    found = set()
    if archive == "scedc":
        # NNSSSSSCCLLLYYYYDDD.ms - verified against a real listing,
        # 'AZBZN__BHE___2015100.ms', length 23.
        for n in names:
            if len(n) == 23 and n.endswith(".ms") and n[:2] == net:
                found.add((n[2:7].rstrip("_"), n[10:13].rstrip("_"), n[7:9]))
    elif archive == "ncedc":
        for n in names:
            p = n.split(".")
            if len(p) >= 6 and p[1] == net:
                found.add((p[0], p[3], p[2][:2]))
    else:
        # EarthScope: one object per station-day, optionally "#version".
        for n in names:
            p = n.split(".")
            if len(p) >= 2 and p[1].split("#")[0] == net:
                found.add((p[0],))
    return found


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True,
                    help="s3://bucket/campaign - read for stations.parquet")
    ap.add_argument("--per-year", type=int, default=2)
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--networks", default="",
                    help="comma-separated subset; default is every network "
                         "the campaign's stations belong to")
    ap.add_argument("--out", default="",
                    help="s3:// or local path for the JSON result")
    a = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | hitrate | %(levelname)s | %(message)s",
    )

    from .s3_helper import CompositeS3ObjectHelper

    s3helper = CompositeS3ObjectHelper()
    state = S3CampaignState(a.campaign)
    stations = state.get_stations()

    stations = stations.assign(
        net=[str(i).split(".")[0] for i in stations["id"].astype(str)],
        cha=[select_channel([c for c in str(x).split(",") if c.strip()])
             for x in stations["channels"]],
    )
    stations = stations[stations["cha"].notna()]
    if a.networks:
        keep = set(a.networks.split(","))
        stations = stations[stations["net"].isin(keep)]

    windows = _operating_windows(stations)
    days = sample_days(a.start_year, a.end_year, a.per_year)
    logger.info(f"{len(stations):,} stations, {len(days)} sample days "
                f"{a.start_year}-{a.end_year}")

    planned = hit = 0
    by_year = collections.defaultdict(lambda: [0, 0])
    by_arch = collections.defaultdict(lambda: [0, 0])
    errors = 0

    for d in days:
        y, doy = f"{d:%Y}", f"{d:%j}"
        live = stations[[
            _overlaps(windows.get(str(i)), d, d + datetime.timedelta(days=1))
            for i in stations["id"].astype(str)
        ]]
        for net, group in live.groupby("net"):
            archive = NETWORK_MAPPING.get(net)
            if archive not in ("scedc", "ncedc", "earthscope"):
                continue
            found = _present(s3helper, archive, net, y, doy)
            if found is None:
                errors += 1
                continue
            for _, row in group.iterrows():
                parts = str(row["id"]).split(".")
                sta = parts[1]
                loc = parts[2] if len(parts) > 2 else ""
                key = ((sta,) if archive == "earthscope"
                       else (sta, loc, row["cha"]))
                planned += 1
                by_year[y][0] += 1
                by_arch[archive][0] += 1
                if key in found:
                    hit += 1
                    by_year[y][1] += 1
                    by_arch[archive][1] += 1
        logger.info(f"{y}.{doy}: {hit:,}/{planned:,} so far")

    if not planned:
        logger.error("No station-days sampled - nothing to report")
        sys.exit(1)

    result = {
        "campaign": a.campaign,
        "sampled_station_days": planned,
        "hit": hit,
        "raw_rate": hit / planned,
        "by_year": {k: v for k, v in sorted(by_year.items())},
        "by_archive": {k: v for k, v in sorted(by_arch.items())},
        "earthscope_correction": EARTHSCOPE_CORRECTION,
        "listing_errors": errors,
    }
    # Corrected rate: EarthScope station-days are scaled, the rest are exact.
    es = by_arch.get("earthscope", [0, 0])
    corrected = (hit - es[1] + es[1] * EARTHSCOPE_CORRECTION) / planned
    result["corrected_rate"] = corrected

    logger.info(f"sampled {planned:,} station-days, {hit:,} hit "
                f"= {100*hit/planned:.1f}% raw, "
                f"{100*corrected:.1f}% corrected")
    for arch, (p, h) in sorted(by_arch.items()):
        note = (f"  -> {100*EARTHSCOPE_CORRECTION*h/p:.1f}% corrected"
                if arch == "earthscope" else "")
        logger.info(f"  {arch:11s} {h:,}/{p:,} = {100*h/p:.1f}%{note}")
    if errors:
        logger.warning(f"{errors} listings failed and were skipped")

    blob = json.dumps(result, indent=2)
    if a.out.startswith("s3://"):
        import boto3
        bucket, _, key = a.out[len("s3://"):].partition("/")
        boto3.client("s3").put_object(Bucket=bucket, Key=key,
                                      Body=blob.encode())
        logger.info(f"wrote {a.out}")
    elif a.out:
        with open(a.out, "w") as fh:
            fh.write(blob)
        logger.info(f"wrote {a.out}")
    else:
        print(blob)
