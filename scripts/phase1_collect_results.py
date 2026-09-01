#!/usr/bin/env python
"""
Phase 1: Collect and analyze measurement results.

After Phase 1 test jobs complete, this script:
1. Reads CloudWatch Logs for profiling data
2. Extracts per-stage timing (s3.get, model.classify, etc.)
3. Compares EarthScope vs SCEDC latency
4. Compares procs 1,2,4,8 cost efficiency
5. Generates decision report for Phase 2

Usage:
    python scripts/phase1_collect_results.py \\
        --job-file phase1_submitted_jobs.json
"""

import argparse
import json
import logging
import re
from datetime import datetime, timedelta

import boto3

logger = logging.getLogger("phase1_collect")

# CloudWatch Logs group for Batch jobs
LOG_GROUP = "/aws/batch/job"


def parse_profile_line(line: str) -> dict:
    """Parse a profile output line like '| amp.wood_anderson | 25.5 s (47%) |'.

    Returns dict with stage name and seconds.
    """
    # Line format: "| stage_name | 25.5 s (47%) |"
    match = re.search(r"\|\s*(\w+[.\w]*)\s*\|\s*([\d.]+)\s*s", line)
    if match:
        return {"stage": match.group(1), "seconds": float(match.group(2))}
    return None


def fetch_logs(job_id: str, start_time: datetime, end_time: datetime) -> list[str]:
    """Fetch CloudWatch Logs for a Batch job."""
    logs = boto3.client("logs", region_name="us-east-2")
    stream_name = f"{job_id}/default"

    try:
        response = logs.get_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=stream_name,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
        )
        return [event["message"] for event in response.get("events", [])]
    except Exception as exc:
        logger.warning(f"Failed to fetch logs for {job_id}: {exc}")
        return []


def extract_profile_data(logs: list[str]) -> dict:
    """Extract profiling data from job logs."""
    profile_lines = [l for l in logs if "s (" in l and "|" in l]
    stages = {}

    for line in profile_lines:
        parsed = parse_profile_line(line)
        if parsed:
            stages[parsed["stage"]] = parsed["seconds"]

    return {
        "s3_get": stages.get("s3.get", 0),
        "model_classify": stages.get("model.classify", 0),
        "amp_wood_anderson": stages.get("amp.wood_anderson", 0),
        "amp_raw": stages.get("amp.raw", 0),
        "total_stages": sum(stages.values()),
    }


def analyze_results(submitted_jobs: dict) -> dict:
    """Analyze all Phase 1 results."""
    logger.info("Fetching logs and extracting profiles...")

    analysis = {
        "timestamp": datetime.utcnow().isoformat(),
        "campaigns": {},
    }

    for campaign_name, jobs in submitted_jobs.items():
        logger.info(f"\nAnalyzing {campaign_name}...")
        campaign_results = []

        for job_info in jobs:
            job_id = job_info["job_id"]
            procs = job_info["procs"]

            logger.info(f"  Job {job_id} (procs={procs})...")

            # Try to fetch logs (they might not be available immediately)
            # Use a 30-minute window
            now = datetime.utcnow()
            start = now - timedelta(minutes=30)
            logs = fetch_logs(job_id, start, now)

            profile = extract_profile_data(logs)
            profile["job_id"] = job_id
            profile["procs"] = procs
            campaign_results.append(profile)

            logger.info(
                f"    s3.get={profile['s3_get']:.1f}s, "
                f"model.classify={profile['model_classify']:.1f}s"
            )

        analysis["campaigns"][campaign_name] = campaign_results

    return analysis


def generate_report(analysis: dict) -> str:
    """Generate decision report."""
    report = []
    report.append("# Phase 1 Analysis Report\n")
    report.append(f"Timestamp: {analysis['timestamp']}\n")

    # 1a: EarthScope vs SCEDC
    report.append("\n## 1a. EarthScope vs SCEDC I/O Latency\n")
    earthscope_results = analysis["campaigns"].get("earthscope", [])
    scedc_results = analysis["campaigns"].get("scedc", [])

    if earthscope_results and scedc_results:
        es_s3_avg = (
            sum(r["s3_get"] for r in earthscope_results) / len(earthscope_results)
        )
        scedc_s3_avg = sum(r["s3_get"] for r in scedc_results) / len(scedc_results)
        ratio = es_s3_avg / scedc_s3_avg if scedc_s3_avg > 0 else 0

        report.append(f"EarthScope avg s3.get: {es_s3_avg:.1f}s\n")
        report.append(f"SCEDC avg s3.get: {scedc_s3_avg:.1f}s\n")
        report.append(f"Ratio: {ratio:.1f}x\n\n")

        if ratio < 5:
            report.append("**✅ DECISION: EarthScope is acceptable**\n")
            report.append(
                "  → Proceed with EarthScope campaigns as planned\n"
            )
        elif ratio < 15:
            report.append("**⚠️ DECISION: EarthScope is significant but manageable**\n")
            report.append(
                "  → Proceed but monitor cost; may re-plan if procs tests are unfavorable\n"
            )
        else:
            report.append("**🛑 DECISION: EarthScope is TOO SLOW**\n")
            report.append(
                "  → DO NOT launch EarthScope campaigns yet; re-plan required\n"
            )

    # 1b: Process parallelism
    report.append("\n## 1b. Process Parallelism Efficiency\n")

    for campaign_name in ["scedc", "earthscope"]:
        results = analysis["campaigns"].get(campaign_name, [])
        if not results:
            continue

        report.append(f"\n### {campaign_name.upper()}\n\n")
        report.append("| Procs | Classify Time | Cost Model |\n")
        report.append("|-------|---------------|------------|\n")

        baseline = min(r["model_classify"] for r in results)
        for result in sorted(results, key=lambda r: r["procs"]):
            procs = result["procs"]
            classify_time = result["model_classify"]
            # Cost model: wall-clock-time × procs (vCPU-hours proportional)
            cost_factor = (classify_time * procs) / (baseline * 1)
            report.append(f"| {procs} | {classify_time:.1f}s | {cost_factor:.1f}x |\n")

        # Recommendation
        best_cost = min(
            (r["model_classify"] * r["procs"], r["procs"]) for r in results
        )
        report.append(f"\n**Recommendation: Use --procs {best_cost[1]}**\n")

    report.append("\n## Next Steps\n")
    report.append("1. Review this report\n")
    report.append("2. Make go/no-go decision for Phase 2\n")
    report.append("3. If GO: proceed to SCEDC smoke test\n")
    report.append("4. If NO-GO or on-hold: investigate and re-plan\n")

    return "".join(report)


def main(job_file: str = "phase1_submitted_jobs.json"):
    """Collect and analyze Phase 1 results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not job_file:
        logger.error("Must provide --job-file with submitted job IDs")
        return

    logger.info(f"Loading job IDs from {job_file}...")
    try:
        with open(job_file) as f:
            submitted_jobs = json.load(f)
    except FileNotFoundError:
        logger.error(f"Job file not found: {job_file}")
        return

    analysis = analyze_results(submitted_jobs)
    report = generate_report(analysis)

    # Save results
    output_json = "phase1_analysis.json"
    with open(output_json, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved to {output_json}")

    output_md = "docs/rerun_2026/phase1_results.md"
    with open(output_md, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {output_md}")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Collect and analyze results")
    parser.add_argument(
        "--job-file",
        default="phase1_submitted_jobs.json",
        help="JSON file with submitted job IDs",
    )
    args = parser.parse_args()

    main(args.job_file)
