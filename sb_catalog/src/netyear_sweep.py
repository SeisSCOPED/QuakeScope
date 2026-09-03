"""Ask EarthScope which planned network-years actually exist.

Temporary FDSN codes are reused, so our station metadata claims deployments the
archive never held. Three turned up by accident during the 2026-09-02 dry runs:

    ZI 2019   404 network FDSN:ZI year 2019 not found
    XD 2019   404
    5A 2018   404   - cost 16 of west4's 48 shards before the fix

Each one is a shard that either fails or does nothing, and each is planned
station-days that will never be picked. Since the credential exchange answers
404 for a network-year it does not have, the whole plan can be checked against
it directly - one cheap request per (network, year), no data read.

Run it in the deployed image, not on a laptop: it needs the EarthScope refresh
token, and using that token locally risks invalidating the Secrets Manager copy
every campaign job depends on.

    python -m src.picker netyear-sweep --campaign s3://bucket/prefix
    python -m src.picker netyear-sweep --networks XD,ZI,5A --years 2010-2020
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys

logger = logging.getLogger("netyear")


def _pairs_from_campaign(campaign: str) -> list[tuple[str, int]]:
    """Every (network, year) the plan will actually ask for."""
    from .s3_state import S3CampaignState

    shards = S3CampaignState(campaign).read_shards()
    pairs = set()
    for sh in shards:
        # A shard can straddle a year boundary, and the credential is scoped
        # per year, so both years are planned even though the shard is one.
        y0 = int(str(sh["start"]).split(".")[0])
        y1 = int(str(sh["end"]).split(".")[0])
        nets = {str(s).split(".")[0] for s in sh["stations"]}
        for n in nets:
            for y in range(y0, y1 + 1):
                pairs.add((n, y))
    return sorted(pairs)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", help="s3://bucket/prefix - sweep its plan")
    ap.add_argument("--networks", help="comma separated, instead of --campaign")
    ap.add_argument("--years", help="e.g. 2010-2020, with --networks")
    ap.add_argument("--out", help="write the full result as JSON to this S3 URI")
    args = ap.parse_args(argv)

    from .s3_helper import (EARTHSCOPE_OPEN_DATA_NETWORKS, CompositeS3ObjectHelper,
                            EarthScopeNetworkYearNotFound, EarthScopeNotEntitled)

    if args.networks:
        y0, _, y1 = args.years.partition("-")
        years = range(int(y0), int(y1 or y0) + 1)
        pairs = [(n, y) for n in args.networks.split(",") for y in years]
    elif args.campaign:
        pairs = _pairs_from_campaign(args.campaign)
    else:
        ap.error("need --campaign or --networks")

    helper = CompositeS3ObjectHelper()
    # Only the restricted tier has a credential exchange to ask. Open Data is
    # anonymous and SCEDC/NCEDC are not EarthScope at all.
    restricted = [(n, y) for n, y in pairs
                  if helper.get_data_center(n) == "earthscope"
                  and n not in EARTHSCOPE_OPEN_DATA_NETWORKS]
    logger.info(f"{len(pairs)} planned network-years, "
                f"{len(restricted)} on the restricted tier")

    missing, ok, errors, denied = [], [], [], []
    for i, (net, year) in enumerate(restricted, 1):
        try:
            helper.get_es_credential(net, year)
            ok.append((net, year))
        except EarthScopeNetworkYearNotFound:
            missing.append((net, year))
            logger.info(f"  MISSING {net} {year}")
        except EarthScopeNotEntitled:
            # 403 is a different problem from 404 and needs a different
            # response: the data exists, we are not allowed it. Worth a request
            # to EarthScope rather than a correction to the plan.
            denied.append((net, year))
            logger.warning(f"  DENIED  {net} {year} - 403, not entitled")
        except Exception as exc:
            errors.append((net, year, f"{type(exc).__name__}: {exc}"[:120]))
            logger.warning(f"  ERROR   {net} {year}: {type(exc).__name__}")
        if i % 50 == 0:
            logger.info(f"  ... {i}/{len(restricted)}")

    by_net = collections.Counter(n for n, _ in missing)
    print(f"\n=== planned network-years that EarthScope does not have ===")
    print(f"  checked {len(restricted)}  present {len(ok)}  "
          f"MISSING {len(missing)}  DENIED {len(denied)}  errors {len(errors)}")
    if denied:
        dn = collections.Counter(n for n, _ in denied)
        print(f"  403 NOT ENTITLED - these exist but we cannot read them, "
              f"which is a request to EarthScope, not a plan correction:")
        for n, c in dn.most_common():
            print(f"    {n:4s} {c:3d} years  {sorted(y for nn, y in denied if nn == n)}")
    if restricted:
        print(f"  missing fraction: {len(missing)/len(restricted):.1%}")
    for n, c in by_net.most_common(25):
        yrs = sorted(y for nn, y in missing if nn == n)
        print(f"    {n:4s} {c:3d} years  {yrs}")

    if args.out:
        import s3fs
        payload = {"checked": len(restricted), "present": len(ok),
                   "missing": [[n, y] for n, y in missing],
                   "denied": [[n, y] for n, y in denied],
                   "errors": errors}
        with s3fs.S3FileSystem().open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
