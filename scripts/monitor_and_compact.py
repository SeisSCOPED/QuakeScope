#!/usr/bin/env python
"""
Monitor campaign progress and automatically trigger Parquet compaction.

Runs as a separate Batch job or Lambda, polling campaign state hourly.
When 80% of shards are complete, starts a background compaction job.

Usage:
    python scripts/monitor_and_compact.py \\
        --campaign s3://quakescope-picks-2026/scedc \\
        --poll-interval 3600 \\
        --once                # run once and exit (for testing)
"""

import argparse
import logging
import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sb_catalog.src.s3_state import S3CampaignState
from sb_catalog.src.parquet_compact import _should_compact, COMPLETION_THRESHOLD

logger = logging.getLogger("monitor")


def run_compaction_job(campaign_uri: str, dry_run: bool = False) -> bool:
    """Run compaction as a subprocess. Returns True if successful."""
    cmd = [
        "python",
        "-m",
        "sb_catalog.src.parquet_compact",
        "--campaign",
        campaign_uri,
    ]
    if dry_run:
        cmd.append("--dryrun")

    logger.info(f"Launching compaction: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Compaction succeeded:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"Compaction failed:\n{exc.stderr}")
        return False


def monitor_campaign(
    campaign_uri: str,
    poll_interval: int = 3600,
    once: bool = False,
) -> None:
    """Poll campaign progress and trigger compaction when ready.

    Args:
        campaign_uri: s3://bucket/prefix
        poll_interval: Seconds between polls (default 1 hour)
        once: If True, check once and exit
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info(f"Monitoring campaign {campaign_uri}")
    if once:
        logger.info("Running once (--once mode)")
    else:
        logger.info(f"Polling every {poll_interval} seconds")

    state = S3CampaignState(campaign_uri)
    compaction_started = False

    while True:
        try:
            should_compact, completed, total = _should_compact(state)
            pct = 100 * completed / total if total > 0 else 0

            logger.info(
                f"Status: {completed}/{total} shards ({pct:.1f}%) | "
                f"Compaction threshold: {COMPLETION_THRESHOLD*100:.0f}%"
            )

            # Check if compaction already started.
            if compaction_started:
                logger.info("Compaction already running or completed")
            elif should_compact:
                logger.info("Launching compaction job...")
                if run_compaction_job(campaign_uri):
                    compaction_started = True
                    logger.info(
                        "Compaction launched. Subsequent polls will monitor progress."
                    )
                else:
                    logger.error(
                        "Compaction failed; will retry on next poll"
                    )

        except Exception as exc:
            logger.exception(f"Poll failed: {exc}")

        if once:
            break

        logger.info(f"Next poll in {poll_interval} seconds...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor campaign progress and trigger compaction"
    )
    parser.add_argument("--campaign", required=True, help="Campaign S3 URI")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=3600,
        help="Poll interval in seconds (default 1 hour)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit (for testing)",
    )
    args = parser.parse_args()

    monitor_campaign(args.campaign, poll_interval=args.poll_interval, once=args.once)
