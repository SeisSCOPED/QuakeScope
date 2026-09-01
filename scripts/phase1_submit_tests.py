#!/usr/bin/env python
"""
Phase 1: Submit measurement test jobs for EarthScope I/O and process parallelism.

This script selects representative shards and submits test jobs to measure:
1. EarthScope I/O latency (--profile, --max-shards 1)
2. Process parallelism efficiency (--procs 1,2,4,8)

Run this ONCE to submit all tests; monitor via dashboard.
Results will land in S3 logs and CloudWatch Logs.

Usage:
    python scripts/phase1_submit_tests.py \\
        --dry-run              # preview jobs, don't submit
        --campaign earthscope  # test only earthscope (default: both)
"""

import argparse
import json
import logging
import os
import sys

import boto3

sys.path.insert(0, os.path.dirname(__file__) + "/..")
from sb_catalog.src.s3_state import S3CampaignState

logger = logging.getLogger("phase1_submit")

# Configuration
CAMPAIGNS = {
    "scedc": {
        "uri": "s3://quakescope-picks-2026/scedc",
        "weight": "jma_wc",
        "kind": "control",  # Baseline
    },
    "earthscope": {
        "uri": "s3://quakescope-picks-2026/earthscope",
        "weight": "jma_wc",
        "kind": "test",  # What we're measuring
    },
}

# Representative shard indices to test
# Pick shards spread across the campaign (early, middle, late, with variety)
SHARD_INDICES = [100, 500, 1000, 3000, 5000]  # 5 shards per campaign

PROCS_VALUES = [1, 2, 4, 8]  # Parallelism levels to test

JOB_QUEUE = "niyiyu_earthscope_missing_station"
JOB_DEFINITION = "quakescope_v3_worker:2"
REGION = "us-east-2"


def select_shards(campaign_uri: str, indices: list[int]) -> list[dict]:
    """Select representative shards by index."""
    state = S3CampaignState(campaign_uri)
    shards = state.read_shards()
    selected = []

    for i in indices:
        if i < len(shards):
            selected.append(shards[i])

    return selected


def submit_job(
    batch_client,
    campaign_name: str,
    campaign_uri: str,
    weight: str,
    max_shards: int = 1,
    profile: bool = True,
    procs: int = 1,
    dry_run: bool = False,
) -> str:
    """Submit a single test job. Returns job ID or empty string if dry-run."""

    job_name = f"phase1-{campaign_name}-procs{procs}-profile{profile}"
    env = [
        {"name": "CAMPAIGN", "value": campaign_uri},
        {"name": "WEIGHT", "value": weight},
        {"name": "MAX_SHARDS", "value": str(max_shards)},
        {"name": "PROCS", "value": str(procs)},
    ]

    if profile:
        env.append({"name": "PROFILE", "value": "1"})

    payload = {
        "jobName": job_name,
        "jobQueue": JOB_QUEUE,
        "jobDefinition": JOB_DEFINITION,
        "containerOverrides": {
            "environment": env,
        },
        "retryStrategy": {
            "attempts": 1,  # Don't retry; we want to see failures
        },
    }

    logger.info(f"Job: {job_name}")
    logger.info(f"  Campaign: {campaign_uri}")
    logger.info(f"  Procs: {procs}")
    logger.info(f"  Profile: {profile}")
    logger.info(f"  Max shards: {max_shards}")

    if dry_run:
        logger.info("  [DRY RUN - not submitted]")
        return ""

    try:
        response = batch_client.submit_job(**payload)
        job_id = response["jobId"]
        logger.info(f"  ✓ Submitted: {job_id}")
        return job_id
    except Exception as exc:
        logger.error(f"  ✗ Failed: {exc}")
        return ""


def main(campaigns: list[str] = None, dry_run: bool = False):
    """Submit all Phase 1 test jobs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if campaigns is None:
        campaigns = list(CAMPAIGNS.keys())

    batch = boto3.client("batch", region_name=REGION)
    submitted_jobs = {}

    logger.info(f"Phase 1 Test Submission")
    logger.info(f"Region: {REGION}")
    logger.info(f"Queue: {JOB_QUEUE}")
    logger.info(f"Job Definition: {JOB_DEFINITION}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info("")

    for campaign_name in campaigns:
        if campaign_name not in CAMPAIGNS:
            logger.warning(f"Unknown campaign: {campaign_name}")
            continue

        config = CAMPAIGNS[campaign_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Campaign: {campaign_name}")
        logger.info(f"{'='*60}")

        # Get representative shards
        logger.info(f"Selecting {len(SHARD_INDICES)} representative shards...")
        try:
            shards = select_shards(config["uri"], SHARD_INDICES)
        except Exception as exc:
            logger.error(f"Failed to load shards: {exc}")
            continue

        logger.info(f"Found {len(shards)} shards (indices: {SHARD_INDICES[:len(shards)]})")

        # Submit one job per procs value
        # (All will read the same shards due to --max-shards 1 per shard)
        logger.info(f"\nSubmitting {len(PROCS_VALUES)} jobs (one per procs value):")
        submitted_jobs[campaign_name] = []

        for procs in PROCS_VALUES:
            job_id = submit_job(
                batch,
                campaign_name=campaign_name,
                campaign_uri=config["uri"],
                weight=config["weight"],
                max_shards=1,  # Profile just one shard at a time
                profile=True,
                procs=procs,
                dry_run=dry_run,
            )
            if job_id:
                submitted_jobs[campaign_name].append(
                    {
                        "job_id": job_id,
                        "procs": procs,
                        "campaign": campaign_name,
                    }
                )

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary")
    logger.info(f"{'='*60}")

    total_submitted = sum(len(jobs) for jobs in submitted_jobs.values())
    logger.info(f"Total jobs submitted: {total_submitted}")

    for campaign_name, jobs in submitted_jobs.items():
        logger.info(f"\n{campaign_name}:")
        for job in jobs:
            logger.info(f"  procs={job['procs']}: {job['job_id']}")

    # Save job IDs for later reference
    if submitted_jobs and not dry_run:
        output_file = "phase1_submitted_jobs.json"
        with open(output_file, "w") as f:
            json.dump(submitted_jobs, f, indent=2)
        logger.info(f"\nJob IDs saved to: {output_file}")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Monitor jobs in AWS Batch console")
    logger.info(f"  2. Check CloudWatch Logs for profiles (search 'amp.wood_anderson', 'model.classify')")
    logger.info(f"  3. Collect results in docs/rerun_2026/profile_*.json")
    logger.info(f"  4. Run Phase 1c: billing alert setup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Submit measurement test jobs"
    )
    parser.add_argument(
        "--campaign",
        choices=list(CAMPAIGNS.keys()) + ["all"],
        default="all",
        help="Which campaign to test (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview jobs without submitting",
    )
    args = parser.parse_args()

    campaigns_to_run = (
        list(CAMPAIGNS.keys()) if args.campaign == "all" else [args.campaign]
    )

    main(campaigns=campaigns_to_run, dry_run=args.dry_run)
