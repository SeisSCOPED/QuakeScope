"""
Automatic Parquet compaction for campaigns.

After 80% of a campaign's shards complete, background compaction consolidates
the thousands of small ~50 KB Parquet files into larger ~1 MB files, making
downstream analysis faster without slowing live picking.

Compaction is safe to run in parallel with picking: picks are immutable once
written, partition keys are unchanged, and re-reading a re-compacted partition
is idempotent.

Usage:
    python -m sb_catalog.src.parquet_compact \\
        --campaign s3://quakescope-picks-2026/scedc \\
        --dryrun              # preview what would compact, then exit
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from collections import defaultdict
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from .parquet_writer import PICK_SCHEMA
from .s3_state import S3CampaignState

logger = logging.getLogger("compact")

TARGET_FILE_SIZE_MB = 1.0  # Consolidate to ~1 MB files
COMPLETION_THRESHOLD = 0.8  # Start compacting when 80% of shards are done


def _should_compact(state: S3CampaignState) -> tuple[bool, int, int]:
    """Check if compaction should run. Returns (should_compact, completed, total)."""
    shards = state.read_shards()
    done = state.completed_ids()
    total = len(shards)
    completed = len(done)
    should_run = completed >= int(total * COMPLETION_THRESHOLD)
    return should_run, completed, total


def _list_partition_objects(
    fs: s3fs.S3FileSystem, bucket: str, prefix: str, partition: str
) -> list[dict]:
    """List all Parquet objects in a partition (network/year/month).

    Returns list of {path, size} dicts.
    """
    part_prefix = f"{prefix}/picks/{partition}/"
    try:
        objects = fs.ls(part_prefix, detail=True)
        return [
            {"path": obj["name"], "size": obj["size"]}
            for obj in objects
            if obj["name"].endswith(".parquet")
        ]
    except FileNotFoundError:
        return []


def _compact_partition(
    fs: s3fs.S3FileSystem,
    bucket: str,
    prefix: str,
    partition: str,
    dryrun: bool = False,
) -> dict:
    """Consolidate small Parquet files in one partition.

    Returns {partition, files_before, files_after, bytes_before, bytes_after, objects_written}.
    """
    objects = _list_partition_objects(fs, bucket, prefix, partition)

    if len(objects) < 2:
        return {
            "partition": partition,
            "files_before": len(objects),
            "files_after": len(objects),
            "status": "skipped (0 or 1 files)",
        }

    total_bytes = sum(o["size"] for o in objects)
    target_files = max(1, int(total_bytes / (TARGET_FILE_SIZE_MB * 1024 * 1024)))

    logger.info(
        f"{partition}: {len(objects)} files, {total_bytes / 1e6:.1f} MB, "
        f"target {target_files} consolidated files"
    )

    if dryrun:
        return {
            "partition": partition,
            "files_before": len(objects),
            "files_after": target_files,
            "bytes_before": total_bytes,
            "status": "dryrun",
            "objects": [o["path"] for o in objects],
        }

    # Read all small files, concatenate, write back as fewer large files.
    tables = []
    for obj in objects:
        try:
            table = pq.read_table(f"s3://{bucket}/{obj['path']}", filesystem=fs)
            tables.append(table)
        except Exception as exc:
            logger.warning(f"Failed to read {obj['path']}: {exc}")
            continue

    if not tables:
        return {
            "partition": partition,
            "status": "failed (no files readable)",
        }

    combined = pa.concat_tables(tables)
    rows_per_file = len(combined) // max(1, target_files)

    # Write consolidated files.
    part_prefix = f"{prefix}/picks/{partition}/"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    written = 0

    for i in range(target_files):
        start = i * rows_per_file
        end = len(combined) if i == target_files - 1 else (i + 1) * rows_per_file
        chunk = combined.slice(start, end - start)

        output_key = f"{part_prefix}compact-{timestamp}-{i:03d}.parquet"
        try:
            pq.write_table(chunk, f"s3://{bucket}/{output_key}", filesystem=fs)
            written += 1
            logger.debug(
                f"  {output_key}: {len(chunk)} rows, "
                f"{chunk.nbytes / 1e6:.1f} MB"
            )
        except Exception as exc:
            logger.error(f"Failed to write {output_key}: {exc}")

    # Delete the old files.
    for obj in objects:
        try:
            fs.rm(f"s3://{bucket}/{obj['path']}")
        except Exception as exc:
            logger.warning(f"Failed to delete {obj['path']}: {exc}")

    return {
        "partition": partition,
        "files_before": len(objects),
        "files_after": written,
        "rows": len(combined),
        "bytes_before": total_bytes,
        "bytes_after": combined.nbytes,
        "status": "completed",
    }


def compact_campaign(
    campaign_uri: str,
    dryrun: bool = False,
    max_partitions: int = 0,
) -> None:
    """Compact a campaign's Parquet output.

    Args:
        campaign_uri: s3://bucket/prefix
        dryrun: If True, preview but don't write
        max_partitions: Limit compaction to N partitions (0 = unlimited)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    state = S3CampaignState(campaign_uri)
    should_run, completed, total = _should_compact(state)

    logger.info(
        f"Campaign {campaign_uri}: {completed}/{total} shards done "
        f"({100*completed/total:.1f}%)"
    )

    if not should_run:
        logger.info(
            f"Waiting for {COMPLETION_THRESHOLD*100:.0f}% completion to start "
            f"compaction ({int(total * COMPLETION_THRESHOLD) - completed} shards to go)"
        )
        return

    if dryrun:
        logger.info("DRYRUN MODE - previewing only, no writes")

    # List all partitions by scanning picks/ prefix.
    fs = s3fs.S3FileSystem()
    bucket, prefix = state.bucket, state.prefix
    picks_prefix = f"{prefix}/picks/"

    partitions = set()
    try:
        for item in fs.ls(picks_prefix, detail=False):
            # item is like s3://bucket/prefix/picks/network=CI/year=2019/month=07
            if "network=" in item and "year=" in item and "month=" in item:
                # Extract network=XX/year=YYYY/month=MM
                parts = item.split(f"{picks_prefix}")[1]
                partitions.add(parts.rstrip("/"))
    except FileNotFoundError:
        logger.warning(f"No picks found at {picks_prefix}")
        return

    logger.info(f"Found {len(partitions)} partitions to compact")

    results = []
    for i, partition in enumerate(sorted(partitions)):
        if max_partitions and i >= max_partitions:
            logger.info(f"Stopping after {max_partitions} partitions (--max-partitions)")
            break

        result = _compact_partition(fs, bucket, prefix, partition, dryrun=dryrun)
        results.append(result)

        if result.get("status") == "completed":
            logger.info(
                f"  {result['partition']}: "
                f"{result['files_before']} → {result['files_after']} files, "
                f"{result['bytes_before']/1e6:.1f} → {result['bytes_after']/1e6:.1f} MB"
            )

    # Summary.
    completed_results = [r for r in results if r.get("status") == "completed"]
    if completed_results:
        total_before = sum(r.get("bytes_before", 0) for r in completed_results)
        total_after = sum(r.get("bytes_after", 0) for r in completed_results)
        total_files_before = sum(r.get("files_before", 0) for r in completed_results)
        total_files_after = sum(r.get("files_after", 0) for r in completed_results)

        logger.info(
            f"\nCompaction summary ({len(completed_results)} partitions):\n"
            f"  Files: {total_files_before} → {total_files_after} "
            f"({100*(1-total_files_after/max(1,total_files_before)):.0f}% reduction)\n"
            f"  Space: {total_before/1e9:.2f} → {total_after/1e9:.2f} GB "
            f"({100*(1-total_after/total_before):.1f}% reduction)\n"
            f"  Mode: {'DRYRUN' if dryrun else 'LIVE'}"
        )

    # Record compaction state in S3.
    if not dryrun:
        state.s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/compaction.jsonl",
            Body="\n".join(json.dumps(r) for r in results).encode(),
        )
        logger.info(f"Wrote compaction log to s3://{bucket}/{prefix}/compaction.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compact Parquet output after campaign reaches 80% completion"
    )
    parser.add_argument("--campaign", required=True, help="Campaign S3 URI")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Preview compaction without writing",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=0,
        help="Limit to N partitions (0 = unlimited)",
    )
    args = parser.parse_args()

    compact_campaign(args.campaign, dryrun=args.dryrun, max_partitions=args.max_partitions)
