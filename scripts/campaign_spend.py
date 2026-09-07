#!/usr/bin/env python
"""What the campaign has cost, split into work that produced picks and work that did not.

A single total hides the thing worth knowing. On 2026-09-05 a 57-worker fleet
spent an hour re-failing shards it could not read; on 2026-09-06 six workers
spent a night on 36 shards and completed none. Both show up in a total as
"campaign spend" and neither produced a pick.

    python scripts/campaign_spend.py
    python scripts/campaign_spend.py --since 2026-09-01 --markdown

CATEGORIES, by job name prefix and outcome:

    productive   a campaign job that completed at least one shard
    spinning     a campaign job that completed NOTHING - the failure modes we
                 spent two days fixing: embargoed shards requeued, our own
                 throttle, a station table that disagreed with the plan, one
                 bad trace killing a shard
    dry run      dryrun* prefixes: deliberate tests, not science
    survey       netyear sweeps, which ask EarthScope what we may read
    other        anything else of ours on the queue

WHAT THIS IS NOT. It is derived from Batch start and stop times times a
published Spot rate, not from a bill: Cost Explorer is blocked on this account
by an organisation policy, so nothing here has been reconciled against what AWS
actually charged. Treat it as the shape of the spend, not the amount.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys

REGION = "us-east-2"
QUEUE = "niyiyu_earthscope_missing_station"
FARGATE_SPOT_RATE = 0.0148          # $/vCPU-hour, us-east-2 published rate
DESCRIBE_CHUNK = 100


def classify(name, completed):
    if name.startswith("survey"):
        return "survey"
    if name.startswith("dryrun"):
        return "dry run"
    if completed:
        return "productive"
    return "spinning"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="YYYY-MM-DD; default all Batch remembers")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", help="write the breakdown as JSON, local path or "
                                  "s3://bucket/key. The dashboard reads this "
                                  "rather than recomputing: this scan opens a "
                                  "log per job and takes the better part of an "
                                  "hour, which is not an hourly budget.")
    a = ap.parse_args(argv)

    import boto3
    b = boto3.client("batch", region_name=REGION)
    logs = boto3.client("logs", region_name=REGION)

    since = 0.0
    if a.since:
        since = datetime.datetime.strptime(
            a.since, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp() * 1000

    try:
        campaigns = tuple(json.load(open("fleet.json"))["campaigns"])
    except Exception:
        campaigns = ("western", "global", "obs")
    ours = campaigns + ("survey", "dryrun")

    ids = []
    for st in ("SUCCEEDED", "FAILED", "RUNNING", "STARTING", "RUNNABLE"):
        tok = None
        while True:
            kw = dict(jobQueue=QUEUE, jobStatus=st, maxResults=100)
            if tok:
                kw["nextToken"] = tok
            r = b.list_jobs(**kw)
            ids += [j["jobId"] for j in r["jobSummaryList"]
                    if j["jobName"].startswith(ours)]
            tok = r.get("nextToken")
            if not tok:
                break

    now = datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000
    spend = collections.Counter()
    hours = collections.Counter()
    jobs = collections.Counter()
    by_campaign = collections.defaultdict(lambda: collections.Counter())

    for i in range(0, len(ids), DESCRIBE_CHUNK):
        for d in b.describe_jobs(jobs=ids[i:i + DESCRIBE_CHUNK])["jobs"]:
            start = d.get("startedAt")
            if not start or start < since:
                continue
            stop = d.get("stoppedAt") or now
            rr = (d.get("container") or {}).get("resourceRequirements") or []
            vcpu = int(([x["value"] for x in rr if x["type"] == "VCPU"] or [8])[0])
            vh = vcpu * (stop - start) / 3600000.0

            # "Completed something" is read from the job's own log, because a
            # SUCCEEDED job that completed no shard is exactly the case worth
            # separating - and Batch cannot tell you that.
            name = d["jobName"]
            completed = False
            stream = ((d.get("container") or {}).get("logStreamName"))
            if stream and not name.startswith(("survey", "dryrun")):
                try:
                    ev = logs.filter_log_events(
                        logGroupName="/aws/batch/job", logStreamNames=[stream],
                        filterPattern='"Completed "', limit=1).get("events", [])
                    completed = bool(ev)
                except Exception:
                    completed = d.get("status") == "SUCCEEDED"

            kind = classify(name, completed)
            spend[kind] += vh * FARGATE_SPOT_RATE
            hours[kind] += vh
            jobs[kind] += 1
            camp = name.rsplit("-", 1)[0]
            by_campaign[camp][kind] += vh * FARGATE_SPOT_RATE

    total = sum(spend.values())
    order = ["productive", "spinning", "dry run", "survey", "other"]
    rows = [(k, jobs[k], hours[k], spend[k]) for k in order if jobs[k]]

    if a.out:
        doc = {
            "generated": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "since": a.since,
            "rate_per_vcpu_hour": FARGATE_SPOT_RATE,
            "reconciled_against_a_bill": False,
            "categories": {k: {"jobs": jobs[k], "vcpu_hours": round(hours[k], 1),
                               "spend": round(spend[k], 2)} for k in order
                           if jobs[k]},
            "total": {"jobs": sum(jobs.values()),
                      "vcpu_hours": round(sum(hours.values()), 1),
                      "spend": round(total, 2)},
            "by_campaign": {c: {k: round(v, 2) for k, v in ct.items()}
                            for c, ct in by_campaign.items()},
        }
        blob = json.dumps(doc, indent=2).encode()
        if a.out.startswith("s3://"):
            bucket, _, obj = a.out[5:].partition("/")
            boto3.client("s3").put_object(Bucket=bucket, Key=obj, Body=blob,
                                          ContentType="application/json")
        else:
            open(a.out, "wb").write(blob)
        print(f"wrote {a.out}", file=sys.stderr)

    if a.markdown:
        print("| category | jobs | vCPU-h | spend | share |")
        print("|---|--:|--:|--:|--:|")
        for k, j, h, s in rows:
            print(f"| {k} | {j:,} | {h:,.0f} | ${s:,.2f} | "
                  f"{100 * s / total if total else 0:.1f}% |")
        print(f"| **total** | **{sum(jobs.values()):,}** | "
              f"**{sum(hours.values()):,.0f}** | **${total:,.2f}** | |")
    else:
        print(f"QuakeScope spend"
              + (f" since {a.since}" if a.since else " (all Batch remembers)"))
        print(f"  {'category':12} {'jobs':>6} {'vCPU-h':>10} {'spend':>11}   share")
        for k, j, h, s in rows:
            print(f"  {k:12} {j:>6,} {h:>10,.0f} {'$' + format(s, ',.2f'):>11}"
                  f"   {100 * s / total if total else 0:5.1f}%")
        print(f"  {'TOTAL':12} {sum(jobs.values()):>6,} "
              f"{sum(hours.values()):>10,.0f} "
              f"{'$' + format(total, ',.2f'):>11}")
        waste = spend["spinning"]
        if waste:
            print(f"\n  ${waste:,.2f} went to jobs that completed no shard "
                  f"({100 * waste / total:.0f}% of the total).")
        print("\n  Derived from Batch start/stop times x "
              f"${FARGATE_SPOT_RATE}/vCPU-h. Cost Explorer is blocked on this "
              "account,\n  so none of this is reconciled against a bill.")
        if by_campaign:
            print("\n  by campaign:")
            for camp, c in sorted(by_campaign.items(),
                                  key=lambda kv: -sum(kv[1].values())):
                tot = sum(c.values())
                if tot < 0.01:
                    continue
                bits = ", ".join(f"{k} ${v:,.2f}" for k, v in c.most_common())
                print(f"    {camp:14} ${tot:>9,.2f}   ({bits})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
