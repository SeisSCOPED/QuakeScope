#!/usr/bin/env python
"""
Test FDSN cache under massive parallel reads.

Simulates 1,500 workers reading the same cached file simultaneously
to verify S3 can handle the load without throttling.

Usage:
    # Test with 100 workers (quick)
    python scripts/test_fdsn_cache_parallel.py --workers 100

    # Test with 1,500 workers (production scale)
    python scripts/test_fdsn_cache_parallel.py --workers 1500
"""

import argparse
import concurrent.futures
import logging
import statistics
import time
from datetime import datetime

import boto3

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("test_parallel")


def simulate_worker_read(worker_id: int, bucket: str, key: str) -> dict:
    """Simulate one worker reading the cached FDSN file.

    Returns timing and success info.
    """
    start = time.time()
    s3 = boto3.client("s3")

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        elapsed = time.time() - start

        return {
            "worker_id": worker_id,
            "success": True,
            "bytes": len(data),
            "elapsed_sec": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "worker_id": worker_id,
            "success": False,
            "bytes": 0,
            "elapsed_sec": elapsed,
            "error": str(e),
        }


def test_parallel_reads(
    bucket: str,
    key: str,
    num_workers: int,
    max_concurrent: int = None,
) -> dict:
    """Test parallel reads to the same S3 object.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        num_workers: Number of simulated workers
        max_concurrent: Max concurrent requests (None = unlimited)

    Returns:
        Statistics about the test
    """
    print(f"\n{'='*70}")
    print(f"Testing {num_workers} parallel reads")
    print(f"{'='*70}")
    print(f"Bucket: {bucket}")
    print(f"Key: {key}")
    print(f"Max concurrent: {max_concurrent or 'unlimited'}")
    print()

    # Verify object exists first
    s3 = boto3.client("s3")
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        file_size_mb = head["ContentLength"] / 1024 / 1024
        print(f"✓ Object exists: {file_size_mb:.1f} MB")
    except Exception as e:
        print(f"✗ Object not found: {e}")
        return None

    print(f"\nLaunching {num_workers} workers...\n")

    start_time = time.time()
    results = []

    # Use ThreadPoolExecutor for I/O-bound parallel reads
    max_workers = max_concurrent or num_workers

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulate_worker_read, i, bucket, key)
            for i in range(num_workers)
        ]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Progress indicator
            if completed % max(1, num_workers // 20) == 0:
                pct = 100 * completed / num_workers
                print(f"  {completed}/{num_workers} ({pct:.0f}%)", end="\r")

    total_time = time.time() - start_time
    print(f"\n\nAll {num_workers} workers completed in {total_time:.2f}s\n")

    # Analyze results
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    if successes:
        latencies = [r["elapsed_sec"] for r in successes]
        bytes_read = sum(r["bytes"] for r in successes)

        stats = {
            "total_workers": num_workers,
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": 100 * len(successes) / num_workers,
            "total_time_sec": total_time,
            "throughput_reads_per_sec": num_workers / total_time,
            "latency_min_sec": min(latencies),
            "latency_max_sec": max(latencies),
            "latency_mean_sec": statistics.mean(latencies),
            "latency_median_sec": statistics.median(latencies),
            "latency_p95_sec": sorted(latencies)[int(0.95 * len(latencies))],
            "latency_p99_sec": sorted(latencies)[int(0.99 * len(latencies))],
            "total_bytes": bytes_read,
            "total_mb": bytes_read / 1024 / 1024,
            "throughput_mb_per_sec": (bytes_read / 1024 / 1024) / total_time,
        }

        print(f"{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"\nSuccess rate: {stats['success_rate']:.1f}% ({stats['successes']}/{num_workers})")
        print(f"Total time: {stats['total_time_sec']:.2f}s")
        print(f"Throughput: {stats['throughput_reads_per_sec']:.0f} reads/sec")
        print(f"Data transferred: {stats['total_mb']:.1f} MB")
        print(f"Bandwidth: {stats['throughput_mb_per_sec']:.1f} MB/sec")
        print()
        print("Latency distribution:")
        print(f"  Min:    {stats['latency_min_sec']*1000:.0f} ms")
        print(f"  Median: {stats['latency_median_sec']*1000:.0f} ms")
        print(f"  Mean:   {stats['latency_mean_sec']*1000:.0f} ms")
        print(f"  P95:    {stats['latency_p95_sec']*1000:.0f} ms")
        print(f"  P99:    {stats['latency_p99_sec']*1000:.0f} ms")
        print(f"  Max:    {stats['latency_max_sec']*1000:.0f} ms")

        if failures:
            print(f"\n⚠️  {len(failures)} failures:")
            error_counts = {}
            for f in failures:
                err = f["error"][:100]
                error_counts[err] = error_counts.get(err, 0) + 1
            for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"  {count}x: {err}")

        print(f"\n{'='*70}")
        print("VERDICT")
        print(f"{'='*70}")

        if stats["success_rate"] >= 99.9:
            print("✅ EXCELLENT - S3 handled the load with no issues")
            print(f"   {num_workers} workers can safely read this cache in parallel")
        elif stats["success_rate"] >= 99:
            print("✅ GOOD - Minor failures, likely transient")
            print(f"   Add retry logic; {num_workers} workers should work")
        elif stats["success_rate"] >= 95:
            print("⚠️  ACCEPTABLE - Some throttling detected")
            print(f"   Consider staggering reads or reducing concurrency")
        else:
            print("🚫 POOR - Significant failures")
            print(f"   Need to reduce concurrency or add exponential backoff")

        # Latency assessment
        if stats["latency_p99_sec"] < 1.0:
            print(f"\n✅ Latency is excellent (P99 < 1s)")
        elif stats["latency_p99_sec"] < 5.0:
            print(f"\n✅ Latency is acceptable (P99 < 5s)")
        else:
            print(f"\n⚠️  Latency is high (P99 = {stats['latency_p99_sec']:.1f}s)")
            print("   Consider CloudFront or S3 Transfer Acceleration")

        return stats

    else:
        print("🚫 ALL READS FAILED")
        for f in failures[:5]:
            print(f"  {f['error']}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test FDSN cache under parallel load"
    )
    parser.add_argument(
        "--bucket",
        default="quakescope-picks-2026",
        help="S3 bucket name",
    )
    parser.add_argument(
        "--key",
        default="scedc/cache/fdsn_metadata.xml",
        help="S3 object key",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Number of parallel workers (default 100)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Max concurrent requests (default: unlimited)",
    )
    parser.add_argument(
        "--scale-test",
        action="store_true",
        help="Run scaling test: 10, 50, 100, 500, 1500 workers",
    )

    args = parser.parse_args()

    if args.scale_test:
        print("SCALING TEST")
        print("Testing at increasing concurrency levels\n")

        results = {}
        for n in [10, 50, 100, 500, 1500]:
            print(f"\n\n{'#'*70}")
            print(f"# Testing {n} workers")
            print(f"{'#'*70}")
            stats = test_parallel_reads(
                args.bucket, args.key, n, args.max_concurrent
            )
            if stats:
                results[n] = stats
            time.sleep(2)  # Brief pause between tests

        # Summary table
        print(f"\n\n{'='*70}")
        print("SCALING SUMMARY")
        print(f"{'='*70}\n")
        print(f"{'Workers':<10} {'Success%':<12} {'Throughput':<15} {'P99 Latency':<15}")
        print("-" * 70)
        for n, stats in sorted(results.items()):
            print(
                f"{n:<10} {stats['success_rate']:<12.1f} "
                f"{stats['throughput_reads_per_sec']:<15.0f} "
                f"{stats['latency_p99_sec']*1000:<15.0f}"
            )

    else:
        test_parallel_reads(
            args.bucket, args.key, args.workers, args.max_concurrent
        )


if __name__ == "__main__":
    main()
