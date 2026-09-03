#!/usr/bin/env python
"""Keep a target number of workers alive on a Spot pool that keeps killing them.

Batch retries a reclaimed attempt, but `attempts` is capped at 10 and cannot be
raised. On a pool reclaiming 37-53% of attempts - measured on
`niyiyu_earthscope_missing_station` over 2026-09-01/02, and *rising* across
those days - a job's expected lifetime is roughly ten reclaims, after which it
fails permanently and is never replaced. The fleet decays to nothing while the
queue still has work.

So something outside Batch has to resubmit. This is that something: it polls
the queue, counts workers that are actually alive, and tops the fleet back up
to `--target`.

Deliberately a poller, not an EventBridge rule plus a Lambda. The reactive
design needs an event bus, a function, a role and a deployment to change any of
it, and it still cannot answer "are enough workers running right now" without
this same query. A loop that can run on a laptop, an EC2 box, or as a Batch job
is easier to reason about, easier to stop, and fails safe: if the governor dies
the fleet stops growing, it does not stampede.

    # watch only - submit nothing
    python scripts/spot_governor.py --campaign s3://.../earthscope \\
        --job-definition quakescope_2026_earthscope:15 --target 20 --dry-run

    # hold 20 workers until the queue drains
    python scripts/spot_governor.py --campaign s3://.../earthscope \\
        --job-definition quakescope_2026_earthscope:15 --target 20

Stops on its own when the campaign is complete, and refuses to exceed
`--max-submissions` so a bug cannot spend the budget.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sb_catalog.src.s3_state import S3CampaignState

logger = logging.getLogger("governor")

SPOT_REASON = "Your Spot Task was interrupted."
# Batch's hard ceiling on retryStrategy.attempts. Not configurable, which is
# the whole reason this script exists.
BATCH_MAX_ATTEMPTS = 10


def alive_count(batch, queue: str, prefix: str) -> dict:
    """Workers THIS campaign has in the pool right now, by status.

    `prefix` is the job-name prefix, and it is not optional. All campaigns share
    one queue, so counting every job in it means each campaign sees the others'
    workers and concludes it is already at target. That is exactly what
    happened on 2026-09-03: obs was set to 59 while western held 101, the
    governor read `alive 101 >= 59`, submitted nothing, and obs never started -
    while reporting deficit 0, so it looked like it had.

    The soak test could not catch it: only one campaign was running.

    RUNNABLE counts: a job waiting on capacity is still a worker we asked for,
    and submitting more because it has not started yet is how you end up with a
    thundering herd the moment capacity returns.
    """
    out = {}
    for status in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"):
        n, tok = 0, None
        while True:
            kw = dict(jobQueue=queue, jobStatus=status, maxResults=100)
            if tok:
                kw["nextToken"] = tok
            r = batch.list_jobs(**kw)
            n += sum(1 for j in r["jobSummaryList"]
                     if j["jobName"].startswith(prefix))
            tok = r.get("nextToken")
            if not tok:
                break
        out[status] = n
    return out


def reclaim_rate(batch, queue: str, hours: float = 6.0) -> tuple[int, int]:
    """(spot-killed attempts, total attempts) in the recent window.

    Reported so the operator can see whether the pool is calm or thrashing
    without leaving the terminal - the number that decides whether this queue
    is the right one to run a campaign on at all.
    """
    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(hours=hours)).timestamp() * 1000
    ids = []
    for status in ("SUCCEEDED", "FAILED", "RUNNING"):
        tok = None
        while True:
            kw = dict(jobQueue=queue, jobStatus=status, maxResults=100)
            if tok:
                kw["nextToken"] = tok
            r = batch.list_jobs(**kw)
            ids += [j["jobId"] for j in r["jobSummaryList"]
                    if j.get("createdAt", 0) >= cutoff]
            tok = r.get("nextToken")
            if not tok:
                break
    spot = total = 0
    for i in range(0, len(ids), 100):
        for j in batch.describe_jobs(jobs=ids[i:i + 100])["jobs"]:
            for a in j.get("attempts", []):
                total += 1
                if SPOT_REASON in (a.get("statusReason") or ""):
                    spot += 1
    return spot, total


def submit(batch, args, n: int) -> list[str]:
    cmd = ["work", "--campaign", args.campaign, "--weight", args.weight,
           "--procs", str(args.procs), "--checkpoint-every", "40",
           "--flush-threshold", "250000",
           "--lease-hours", str(args.lease_hours)]
    env = [{"name": k, "value": str(args.threads)} for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")]
    ids = []
    for _ in range(n):
        r = batch.submit_job(
            jobName=f"{args.name_prefix}-{int(time.time()*1000) % 10**9}",
            jobQueue=args.queue, jobDefinition=args.job_definition,
            containerOverrides={"command": cmd, "environment": env})
        ids.append(r["jobId"])
    return ids


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True, help="s3://bucket/campaign")
    ap.add_argument("--job-definition", required=True)
    ap.add_argument("--queue", default="niyiyu_earthscope_missing_station")
    ap.add_argument("--target", type=int, default=10,
                    help="workers to keep alive")
    ap.add_argument("--weight", default="jma_wc")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--lease-hours", type=float, default=1.0)
    ap.add_argument("--poll-seconds", type=int, default=180)
    ap.add_argument("--max-submissions", type=int, default=500,
                    help="hard cap over this governor's lifetime")
    ap.add_argument("--name-prefix", default="qs-worker")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what it would submit, submit nothing")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)

    batch = boto3.client("batch", region_name="us-east-2")
    state = S3CampaignState(args.campaign)
    submitted = 0

    while True:
        try:
            prog = state.progress()   # {total, complete, in_flight, remaining}
            done, total = prog["complete"], prog["total"]
        except Exception as exc:
            logger.warning(f"Could not read campaign progress: {exc}")
            done, total = 0, 0

        if total and done >= total:
            logger.info(f"Campaign complete ({done}/{total}). Governor exiting; "
                        f"running workers will drain and stop on their own.")
            return 0

        alive = alive_count(batch, args.queue, args.name_prefix)
        n_alive = sum(alive.values())
        deficit = max(0, args.target - n_alive)
        spot, att = reclaim_rate(batch, args.queue)

        logger.info(
            f"{args.name_prefix}: shards {done}/{total} | alive {n_alive} "
            f"(run {alive.get('RUNNING',0)} runnable {alive.get('RUNNABLE',0)}) "
            f"| target {args.target} deficit {deficit} | "
            f"spot reclaims {spot}/{att} attempts in 6h "
            f"({spot/max(att,1):.0%}) | submitted {submitted}")

        if deficit:
            room = args.max_submissions - submitted
            if room <= 0:
                logger.error(
                    f"Hit --max-submissions {args.max_submissions}. Not "
                    f"submitting more. Raise it deliberately if the campaign "
                    f"genuinely needs it.")
                return 1
            n = min(deficit, room)
            if args.dry_run:
                logger.info(f"DRY RUN: would submit {n}")
            else:
                ids = submit(batch, args, n)
                submitted += len(ids)
                logger.info(f"submitted {len(ids)}: {', '.join(i[:8] for i in ids)}")

        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
