#!/usr/bin/env python3
"""
What is running on AWS right now, what it is costing, and how to stop it.

Written because a Spot campaign can quietly cost money in three ways that none
of the usual dashboards make obvious within an hour: an instance that should
have been torn down, a SkyPilot jobs controller that outlives its jobs, and a
managed job silently satisfied on-demand at ~3x the Spot price. Cost Explorer
lags about a day, which is too slow to catch any of them.

Reports, per region:
  * every instance running, its type, lifecycle (spot vs on-demand), and age
  * the hourly burn at the CURRENT spot price for that type and AZ
  * what that projects to per day and per month if nothing changes
  * anything that looks wrong: on-demand where Spot was intended, an idle
    jobs controller, an instance running longer than a campaign shard should

Read-only: it makes no changes to the account. Emergency-stop commands are
printed for a human to run, deliberately not executed.

    python scripts/aws_watch.py                    # human-readable
    python scripts/aws_watch.py --format markdown  # for a GitHub issue
    python scripts/aws_watch.py --quiet            # print nothing when idle
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict

import boto3
from botocore.exceptions import ClientError

DEFAULT_REGIONS = ["us-west-2", "us-east-1", "us-east-2"]

# An instance older than this is worth a second look: campaign shards are sized
# in hours, and a controller left up is the classic way to leak money.
STALE_HOURS = 12.0


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def spot_price(ec2, instance_type: str, az: str) -> float | None:
    """Current spot price for a type in an AZ. None if unavailable."""
    try:
        r = ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            AvailabilityZone=az,
            MaxResults=1,
        )
        hist = r.get("SpotPriceHistory", [])
        return float(hist[0]["SpotPrice"]) if hist else None
    except ClientError:
        return None


def on_demand_price(instance_type: str, region: str) -> float | None:
    """On-demand list price. The pricing API lives only in a few regions."""
    location = {
        "us-west-2": "US West (Oregon)",
        "us-east-1": "US East (N. Virginia)",
        "us-east-2": "US East (Ohio)",
    }.get(region)
    if not location:
        return None
    try:
        pricing = boto3.client("pricing", region_name="us-east-1")
        r = pricing.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "location", "Value": location},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ],
            MaxResults=1,
        )
        if not r["PriceList"]:
            return None
        p = json.loads(r["PriceList"][0])
        term = list(p["terms"]["OnDemand"].values())[0]
        dim = list(term["priceDimensions"].values())[0]
        return float(dim["pricePerUnit"]["USD"])
    except Exception:
        return None


def scan_region(region: str) -> dict:
    ec2 = boto3.client("ec2", region_name=region)
    try:
        pages = ec2.get_paginator("describe_instances").paginate(
            Filters=[{"Name": "instance-state-name",
                      "Values": ["pending", "running", "stopping"]}]
        )
        instances = [i for p in pages for r in p["Reservations"] for i in r["Instances"]]
    except ClientError as exc:
        return {"region": region, "error": str(exc)[:120], "instances": [],
                "hourly": 0.0, "warnings": []}

    rows, warnings, hourly = [], [], 0.0
    for inst in instances:
        tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
        name = tags.get("Name", "")
        cluster = tags.get("skypilot-cluster-name", "")
        itype = inst["InstanceType"]
        az = inst["Placement"]["AvailabilityZone"]
        is_spot = inst.get("InstanceLifecycle") == "spot"
        age_h = (_now() - inst["LaunchTime"]).total_seconds() / 3600

        price = spot_price(ec2, itype, az) if is_spot else on_demand_price(itype, region)
        hourly += price or 0.0

        rows.append({
            "id": inst["InstanceId"], "type": itype, "az": az,
            "lifecycle": "spot" if is_spot else "on-demand",
            "age_h": age_h, "price": price, "name": name,
            "cluster": cluster, "state": inst["State"]["Name"],
        })

        # The three ways a Spot campaign quietly costs more than intended.
        if cluster and not is_spot:
            warnings.append(
                f"{inst['InstanceId']} ({name or itype}) is ON-DEMAND but is a "
                f"SkyPilot instance - roughly 3x the Spot price"
            )
        if "jobs-controller" in name and age_h > STALE_HOURS:
            warnings.append(
                f"{inst['InstanceId']} is a jobs controller up for {age_h:.1f} h - "
                f"controllers outlive their jobs and are easy to forget"
            )
        elif age_h > STALE_HOURS and not name.startswith("sky-jobs-controller"):
            warnings.append(
                f"{inst['InstanceId']} ({name or itype}) has run {age_h:.1f} h"
            )

    return {"region": region, "instances": rows, "hourly": hourly,
            "warnings": warnings, "error": None}


def scan_spot_requests(region: str) -> int:
    """Open/active requests can relaunch after you terminate an instance."""
    try:
        ec2 = boto3.client("ec2", region_name=region)
        r = ec2.describe_spot_instance_requests(
            Filters=[{"Name": "state", "Values": ["open", "active"]}]
        )
        return len(r.get("SpotInstanceRequests", []))
    except ClientError:
        return 0


def render(results: list[dict], spot_open: dict, fmt: str) -> str:
    total_hourly = sum(r["hourly"] for r in results)
    n = sum(len(r["instances"]) for r in results)
    warnings = [(r["region"], w) for r in results for w in r["warnings"]]
    md = fmt == "markdown"
    L: list[str] = []
    h1 = "## " if md else ""

    if n == 0:
        L.append(f"{h1}No instances running")
        L.append("")
        L.append(f"Checked {', '.join(r['region'] for r in results)} at "
                 f"{_now():%Y-%m-%d %H:%M} UTC. Nothing is costing compute.")
        leftover = {k: v for k, v in spot_open.items() if v}
        if leftover:
            L.append("")
            L.append(f"But {sum(leftover.values())} spot request(s) are still open "
                     f"in {', '.join(leftover)} - these can relaunch instances.")
        return "\n".join(L)

    L.append(f"{h1}{n} instance(s) running - ${total_hourly:.4f}/hour")
    L.append("")
    L.append(f"**${total_hourly * 24:.2f}/day, ${total_hourly * 24 * 30:,.0f}/month** "
             f"if nothing changes. Checked {_now():%Y-%m-%d %H:%M} UTC.")
    L.append("")

    if warnings:
        L.append(f"{'### ' if md else ''}Warnings")
        L.append("")
        for region, w in warnings:
            L.append(f"- **{region}**: {w}" if md else f"  ! {region}: {w}")
        L.append("")

    if md:
        L.append("| region | instance | type | lifecycle | age | $/hr | name |")
        L.append("|---|---|---|---|--:|--:|---|")
    for r in results:
        for i in r["instances"]:
            price = f"{i['price']:.4f}" if i["price"] is not None else "?"
            if md:
                L.append(f"| {r['region']} | `{i['id']}` | {i['type']} | "
                         f"{i['lifecycle']} | {i['age_h']:.1f} h | {price} | "
                         f"{i['name'] or '-'} |")
            else:
                L.append(f"  {r['region']:11s} {i['id']:20s} {i['type']:14s} "
                         f"{i['lifecycle']:10s} {i['age_h']:6.1f}h  ${price:>8s}  "
                         f"{i['name']}")
    L.append("")

    open_total = sum(spot_open.values())
    if open_total:
        L.append(f"{open_total} open spot request(s): "
                 + ", ".join(f"{k} ({v})" for k, v in spot_open.items() if v))
        L.append("")

    L.append(f"{'### ' if md else ''}Emergency stop")
    L.append("")
    L.append("Kill the jobs controller **first** - its job is relaunching preempted "
             "workers, so it will resurrect anything you terminate.")
    L.append("")
    L.append("```bash")
    L.append("sky jobs cancel --all && sky down --all      # preferred")
    L.append("")
    for r in results:
        if r["instances"]:
            ids = " ".join(i["id"] for i in r["instances"]
                           if "jobs-controller" in i["name"])
            rest = " ".join(i["id"] for i in r["instances"]
                            if "jobs-controller" not in i["name"])
            if ids:
                L.append(f"aws ec2 terminate-instances --region {r['region']} "
                         f"--instance-ids {ids}   # controller first")
            if rest:
                L.append(f"aws ec2 terminate-instances --region {r['region']} "
                         f"--instance-ids {rest}")
    L.append("```")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--format", choices=["text", "markdown"], default="text")
    ap.add_argument("--quiet", action="store_true",
                    help="Print nothing when nothing is running")
    ap.add_argument("--fail-if-running", action="store_true",
                    help="Exit 1 when anything is running (for CI gating)")
    args = ap.parse_args(argv)

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    results = [scan_region(r) for r in regions]
    spot_open = {r: scan_spot_requests(r) for r in regions}
    n = sum(len(r["instances"]) for r in results)

    if n == 0 and args.quiet:
        return 0
    print(render(results, spot_open, args.format))
    for r in results:
        if r["error"]:
            print(f"\n! {r['region']}: {r['error']}", file=sys.stderr)
    return 1 if (n and args.fail_if_running) else 0


if __name__ == "__main__":
    sys.exit(main())
