#!/usr/bin/env python
"""
Calculate campaign cost from Phase 1 measurements.

Replaces CloudWatch Budget alerts with manual calculation from vCPU-hours.
No AWS permissions required — just the numbers from Phase 1 results.

Usage:
    python scripts/calculate_phase1_cost.py \\
        --phase1-results docs/rerun_2026/phase1_results.md \\
        --output docs/rerun_2026/cost_estimate_calculated.md
"""

import argparse
import json
import sys
from pathlib import Path

# Campaign metadata: (station_days, station_locations, avg_band_days_per_station)
CAMPAIGNS = {
    "scedc": {
        "station_days": 4_106_669,
        "weight": "jma_wc",
        "networks": 1128,
    },
    "ncedc": {
        "station_days": 5_979_675,
        "weight": "jma_wc",
        "networks": 2116,
    },
    "earthscope": {
        "station_days": 67_983_975,
        "weight": "jma_wc",
        "networks": 51846,
    },
    "obs": {
        "station_days": 996_536,
        "weight": "obs",
        "networks": 3389,
    },
    "western": {
        "station_days": 33_799_828,
        "weight": "original",
        "networks": 24113,
    },
}

# Fargate Spot pricing (estimated, varies by region/time)
FARGATE_SPOT_PRICE_PER_VCPU_HR = 0.0148  # dollars

# Channels per station (one per station-location, after channel priority filtering)
CHANNELS_PER_STATION = 1.0


def calculate_cost(
    seconds_per_band_day: float,
    band_days: float,
    procs: int = 1,
    price_per_vcpu_hr: float = FARGATE_SPOT_PRICE_PER_VCPU_HR,
) -> dict:
    """
    Calculate cost for a campaign.

    Args:
        seconds_per_band_day: Wall-clock time per band-day (from profiling)
        band_days: Total band-days (station-days × channels per station)
        procs: Number of processes per vCPU (affects parallelism)
        price_per_vcpu_hr: Spot price (default $0.0148/vCPU-hr)

    Returns:
        dict with vCPU-hours, cost, and breakdown
    """
    # Wall-clock time to vCPU-hours:
    # If one job takes X seconds on 8 vCPU, that's (X / 3600) vCPU-hours
    # But if running --procs N, we can fit more work per vCPU (with diminishing returns)
    vcpu_hours = (band_days * seconds_per_band_day) / 3600 / procs

    cost = vcpu_hours * price_per_vcpu_hr

    return {
        "band_days": int(band_days),
        "seconds_per_band_day": seconds_per_band_day,
        "vcpu_hours": round(vcpu_hours, 0),
        "cost": round(cost, 2),
        "cost_per_band_day": round(cost / band_days, 6),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Phase 1 campaign costs"
    )
    parser.add_argument(
        "--scedc-seconds",
        type=float,
        help="SCEDC seconds per band-day (from Phase 1a profile)",
    )
    parser.add_argument(
        "--earthscope-seconds",
        type=float,
        help="EarthScope seconds per band-day (from Phase 1a profile)",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=1,
        help="Processes per vCPU (from Phase 1b, default 1)",
    )
    parser.add_argument(
        "--price-per-vcpu-hr",
        type=float,
        default=FARGATE_SPOT_PRICE_PER_VCPU_HR,
        help=f"Fargate Spot price (default ${FARGATE_SPOT_PRICE_PER_VCPU_HR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write cost estimate to file (optional)",
    )

    args = parser.parse_args()

    # If no arguments, show usage example
    if not args.scedc_seconds:
        print("Phase 1 Cost Calculator")
        print()
        print("Usage:")
        print("  python scripts/calculate_phase1_cost.py \\")
        print("    --scedc-seconds 34 \\")
        print("    --earthscope-seconds 280 \\")
        print("    --procs 2")
        print()
        print("Example (after Phase 1 results):")
        print("  If Phase 1 shows SCEDC is 34 s/band-day")
        print("  and EarthScope is 8× slower (272 s/band-day)")
        print("  and --procs 2 is optimal:")
        print()
        print("  python scripts/calculate_phase1_cost.py \\")
        print("    --scedc-seconds 34 \\")
        print("    --earthscope-seconds 272 \\")
        print("    --procs 2")
        print()
        return

    # Use EarthScope seconds if provided, else assume same as SCEDC
    es_seconds = args.earthscope_seconds or args.scedc_seconds

    print("\n" + "=" * 70)
    print("Phase 1 Cost Calculation")
    print("=" * 70)
    print(f"\nInputs:")
    print(f"  SCEDC: {args.scedc_seconds} sec/band-day")
    print(f"  EarthScope: {es_seconds} sec/band-day (ratio: {es_seconds/args.scedc_seconds:.1f}×)")
    print(f"  Processes per vCPU: {args.procs}")
    print(f"  Spot price: ${args.price_per_vcpu_hr}/vCPU-hr")

    print(f"\n{'Campaign':<15} {'Station-Days':>15} {'vCPU-Hours':>15} {'Estimated Cost':>15}")
    print("-" * 70)

    total_vcpu_hours = 0
    total_cost = 0
    costs = {}

    for campaign_name, config in CAMPAIGNS.items():
        station_days = config["station_days"]
        band_days = station_days * CHANNELS_PER_STATION

        # Use EarthScope price for EarthScope, SCEDC price for others
        seconds = es_seconds if campaign_name == "earthscope" else args.scedc_seconds

        result = calculate_cost(
            seconds,
            band_days,
            procs=args.procs,
            price_per_vcpu_hr=args.price_per_vcpu_hr,
        )

        costs[campaign_name] = result
        total_vcpu_hours += result["vcpu_hours"]
        total_cost += result["cost"]

        print(
            f"{campaign_name:<15} {station_days:>15,} "
            f"{result['vcpu_hours']:>15,.0f} ${result['cost']:>14,.0f}"
        )

    print("-" * 70)
    print(
        f"{'TOTAL':<15} {sum(c['station_days'] for c in CAMPAIGNS.values()):>15,} "
        f"{total_vcpu_hours:>15,.0f} ${total_cost:>14,.0f}"
    )

    # Calculate daily burn rate (assuming 60-day campaign)
    days_estimate = 60
    daily_burn = total_cost / days_estimate
    print(f"\nDaily burn rate (60-day campaign): ${daily_burn:,.0f}/day")
    print(f"Budget recommendation: ${daily_burn * 1.2:,.0f}/day (with 20% headroom)")

    # Sensitivities
    print(f"\n{'Sensitivities':<15}")
    print(f"  +10% vCPU cost: ${total_cost * 1.1:,.0f}")
    print(f"  -10% vCPU cost: ${total_cost * 0.9:,.0f}")
    print(f"  If procs 1 instead of {args.procs}: ${total_cost * args.procs:,.0f}")

    # Write output
    if args.output:
        output_md = f"""# Cost Estimate (Calculated from Phase 1)

**Generated:** 2026-08-31
**Method:** Manual calculation from Phase 1 profiling results

## Inputs

- SCEDC baseline: {args.scedc_seconds} sec/band-day
- EarthScope: {es_seconds} sec/band-day ({es_seconds/args.scedc_seconds:.1f}× slower)
- Process parallelism: --procs {args.procs}
- Spot price: ${args.price_per_vcpu_hr}/vCPU-hr (Fargate, estimated)

## Costs by Campaign

| Campaign | Station-Days | vCPU-Hours | Cost |
|----------|--------------|-----------|------|
"""
        for name, result in costs.items():
            output_md += f"| {name} | {result['band_days']:,} | {result['vcpu_hours']:,.0f} | ${result['cost']:,.0f} |\n"

        output_md += f"""| **TOTAL** | **112,866,083** | **{total_vcpu_hours:,.0f}** | **${total_cost:,.0f}** |

## Budget Recommendation

- **Estimated total cost:** ${total_cost:,.0f}
- **Daily burn rate:** ${daily_burn:,.0f}/day (60-day campaign)
- **Safety margin:** +20% = ${total_cost * 1.2:,.0f}
- **Proposed budget:** ${max(50000, total_cost * 1.2):,.0f}

## Tracking During Campaign

Since CloudWatch Budgets is restricted on this account, track cost manually:

1. **Weekly:** Check AWS Billing console for actual costs (or use `aws ce` CLI)
2. **Compare to estimate:** vCPU-hours × ${args.price_per_vcpu_hr}/vCPU-hr
3. **Update spreadsheet:** {Path('docs/rerun_2026/weekly_cost_tracking.csv')}
4. **Alert thresholds:** If cost >20% over estimate mid-campaign, investigate

## Go/No-Go Decision

Based on this estimate:
- [ ] ✅ **GO** - Cost is acceptable (<$60k)
- [ ] ⚠️ **GO WITH CAUTION** - Cost is high but manageable ($60–100k)
- [ ] 🛑 **HOLD** - Cost is too high; need to re-plan (>$100k)
"""
        Path(args.output).write_text(output_md)
        print(f"\n✓ Cost estimate written to {args.output}")


if __name__ == "__main__":
    main()
