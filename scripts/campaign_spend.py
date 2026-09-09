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
    spot reclaim ran, completed nothing, and Spot took the task back. Not
                 waste: it is the price of the 70% discount, and a reclaimed
                 worker's part-done shard is checkpointed under progress/
    spinning     ran, completed nothing, and was NOT reclaimed - the failure
                 modes we spent two days fixing: embargoed shards requeued,
                 our own throttle, a station table that disagreed with the
                 plan, one bad trace killing a shard
    log expired  the job ran before the log group's retention window, so
                 whether it completed anything is UNKNOWABLE. Reported as its
                 own line rather than guessed into one of the above
    dry run      dryrun* prefixes: deliberate tests, not science
    survey       netyear sweeps, which ask EarthScope what we may read

WHY "log expired" EXISTS. Batch cannot tell you whether a SUCCEEDED job
completed a shard, so this reads the job's log for "Completed ". The log group
keeps 5 days. An earlier version of this script fell back to the Batch status
when the log was gone, which counted every Spot-reclaimed job as spinning and
every old empty job as productive - wrong in both directions, and silently.
A category we cannot determine is named, not imputed.

WHAT THIS IS. vCPU-hours counted from Batch ATTEMPT start and stop times -
every attempt, because every attempt is billed - priced at a rate derived from
a CloudBank bill over days whose usage was measured the same way
(costs_actual.json). Cost Explorer is blocked on this account by an
organisation policy, so that file, edited by a person, is the only place a
real figure can come from; without one the script falls back to a fraction of
list price and says so in the artefact.

ATTEMPTS, NOT JOBS. A job's own startedAt/stoppedAt describe its LAST attempt
only - Batch overwrites them each time a reclaimed task is retried.
global-477782054 ran four attempts for 17.16 h; its job-level span was 1.04 h.
The first version of this script summed job spans, undercounted the three
calibration days by 1.47x (74,030 against 108,664 vCPU-h), and so overstated
the derived rate by the same factor: $0.0313 instead of $0.0213 per vCPU-hour,
published as "calibrated" for two days. See job_usage().
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys

REGION = "us-east-2"
QUEUE = "niyiyu_earthscope_missing_station"
LOG_GROUP = "/aws/batch/job"
# us-east-2 Fargate ON-DEMAND list, from the AWS Pricing API on 2026-09-07.
# Our tasks are 8 vCPU / 16 GB, so memory is 2.06 GB per vCPU and adds 23% on
# top of the vCPU line. An earlier version of this script priced vCPU alone at
# a guessed $0.0148 and left memory out entirely.
FARGATE_ONDEMAND_VCPU_H = 0.04048
FARGATE_ONDEMAND_GB_H = 0.004445
# Fallback only, used when costs_actual.json carries no validated figure to
# calibrate against. Nominal Fargate Spot is about 70% off list.
FALLBACK_SPOT_FRACTION = 0.30
DESCRIBE_CHUNK = 100
# CloudWatch deletes on its own schedule, not on the stroke of the retention
# hour, so the last few hours of the window are unreliable. Treat a job as
# knowable only if it ran wholly inside the window less this margin.
RETENTION_MARGIN_H = 6

ORDER = ["productive", "spot reclaim", "spinning", "log expired",
         "dry run", "survey"]


def completed_streams(logs, start_ms, end_ms):
    """Log streams that printed "Completed " - ONE paged query, not one per job.

    Asking per job is 3,572 filter_log_events calls and takes the better part
    of an hour. Asking the group once and keeping the distinct stream names is
    72 pages and 158 seconds, and answers the same question.
    """
    seen, tok = set(), None
    while True:
        kw = dict(logGroupName=LOG_GROUP, startTime=start_ms, endTime=end_ms,
                  filterPattern='"Completed "', limit=10000)
        if tok:
            kw["nextToken"] = tok
        r = logs.filter_log_events(**kw)
        for e in r.get("events", []):
            seen.add(e["logStreamName"])
        tok = r.get("nextToken")
        if not tok:
            return seen


def is_reclaim(job):
    """Did Spot take this task back, as opposed to it failing on its own?"""
    text = " ".join(str(x) for x in (
        job.get("statusReason"),
        (job.get("container") or {}).get("reason"))).lower()
    return ("spot" in text or "reclaim" in text
            or "host ec2" in text or "terminated" in text)


def job_usage(d, now):
    """What one job consumed, over EVERY attempt it ran.

    Returns (vcpu_hours, gb_hours, {day: vcpu_hours}, log_streams, first_start)
    where first_start is the earliest attempt's start in ms, or None if nothing
    ever ran.

    Batch's job-level startedAt/stoppedAt are overwritten on each retry, so a
    job reclaimed by Spot three times reports the span of its fourth attempt
    and nothing else. Summing those undercounted the calibration days by 1.47x
    and inflated the derived rate by the same factor. Every attempt is a task
    that ran and was billed, so every attempt is summed here; the day split
    follows attempts too, so a job retried across midnight lands on the days
    it actually ran.

    log_streams is the union over attempts, not the last one: a job that
    completed shards in its first attempt and was then reclaimed should count
    as productive, and only the last attempt's stream is on the job record.
    """
    rr = (d.get("container") or {}).get("resourceRequirements") or []
    vcpu = int(([x["value"] for x in rr if x["type"] == "VCPU"] or [8])[0])
    mem = int(([x["value"] for x in rr if x["type"] == "MEMORY"]
               or [16384])[0]) / 1024.0
    spans = [(a["startedAt"], a.get("stoppedAt") or now)
             for a in d.get("attempts") or [] if a.get("startedAt") is not None]
    if not spans and d.get("startedAt") is not None:
        # Running, first attempt not yet on the record.
        spans = [(d["startedAt"], d.get("stoppedAt") or now)]
    streams = {(a.get("container") or {}).get("logStreamName")
               for a in d.get("attempts") or []}
    streams.add((d.get("container") or {}).get("logStreamName"))
    streams.discard(None)
    vh = gh = 0.0
    per_day = collections.Counter()
    for start, stop in spans:
        vh += vcpu * (stop - start) / 3600000.0
        gh += mem * (stop - start) / 3600000.0
        # Spread across the calendar days the attempt actually spanned, so a
        # job can be compared against a billed DAY. Charging a 20-hour attempt
        # wholly to its start day misaligns it from the invoice.
        t = start
        while t < stop:
            day = datetime.datetime.fromtimestamp(
                t / 1000, datetime.timezone.utc).date()
            nxt = min(stop, datetime.datetime.combine(
                day + datetime.timedelta(days=1), datetime.time.min,
                datetime.timezone.utc).timestamp() * 1000)
            per_day[str(day)] += vcpu * (nxt - t) / 3600000.0
            t = nxt
    first = min(s for s, _ in spans) if spans else None
    return vh, gh, per_day, streams, first


def classify(name, completed, knowable, job):
    if name.startswith("survey"):
        return "survey"
    if name.startswith("dryrun"):
        return "dry run"
    if completed:
        return "productive"          # a completion is proof regardless of age
    if not knowable:
        return "log expired"
    return "spot reclaim" if is_reclaim(job) else "spinning"


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

    # list_jobs(jobQueue=...) returns array PARENTS and standalone jobs - never
    # the children of an array. A parent consumes nothing; its 1,500 children
    # consume everything. western_a_full2 ran 1,467 tasks for 2,973 vCPU-hours
    # on 2026-08-31 and this script counted none of it, because the parent was
    # all it could see. Children have to be listed per array id.
    ids, arrays = [], []
    for st in ("SUCCEEDED", "FAILED", "RUNNING", "STARTING", "RUNNABLE"):
        tok = None
        while True:
            kw = dict(jobQueue=QUEUE, jobStatus=st, maxResults=100)
            if tok:
                kw["nextToken"] = tok
            r = b.list_jobs(**kw)
            for j in r["jobSummaryList"]:
                if not j["jobName"].startswith(ours):
                    continue
                if (j.get("arrayProperties") or {}).get("size"):
                    arrays.append(j["jobId"])   # holds no resources itself
                else:
                    ids.append(j["jobId"])
            tok = r.get("nextToken")
            if not tok:
                break
    for aid in set(arrays):
        for st in ("SUCCEEDED", "FAILED", "RUNNING", "STARTING", "RUNNABLE",
                   "PENDING", "SUBMITTED"):
            tok = None
            while True:
                kw = {"arrayJobId": aid, "jobStatus": st}
                if tok:
                    kw["nextToken"] = tok
                r = b.list_jobs(**kw)
                ids += [j["jobId"] for j in r["jobSummaryList"]]
                tok = r.get("nextToken")
                if not tok:
                    break
    if arrays:
        print(f"{len(arrays)} array parent(s) expanded to {len(ids):,} jobs",
              file=sys.stderr)

    now = datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000

    # How far back the log group can still answer for.
    try:
        grp = logs.describe_log_groups(
            logGroupNamePrefix=LOG_GROUP)["logGroups"][0]
        retain_days = grp.get("retentionInDays")
    except Exception:
        retain_days = None
    horizon = (now - (retain_days * 86400 - RETENTION_MARGIN_H * 3600) * 1000
               if retain_days else 0.0)
    print(f"reading {LOG_GROUP} (retention "
          f"{retain_days or 'unlimited'} days)...", file=sys.stderr)
    done_streams = completed_streams(logs, int(max(horizon, since)), int(now))
    print(f"{len(done_streams):,} log streams completed at least one shard",
          file=sys.stderr)
    hours = collections.Counter()        # vCPU-hours by category
    gbh = collections.Counter()          # GB-hours by category
    jobs = collections.Counter()
    kind_of = {}                         # job index -> category
    per_day_vh = collections.Counter()   # calendar day -> vCPU-hours, for
    rows = []                            # calibration against billed days

    for i in range(0, len(ids), DESCRIBE_CHUNK):
        for d in b.describe_jobs(jobs=ids[i:i + DESCRIBE_CHUNK])["jobs"]:
            vh, gh, days, streams, start = job_usage(d, now)
            if start is None or start < since:
                continue
            for day, v in days.items():
                per_day_vh[day] += v

            # "Completed something" comes from the job's own logs - every
            # attempt's, since a job that finished shards and was then
            # reclaimed keeps only its last stream on the record - because a
            # SUCCEEDED job that completed no shard is exactly the case worth
            # separating, and Batch cannot tell you that.
            name = d["jobName"]
            completed = bool(streams & done_streams)
            # Knowable only if the WHOLE run sits inside the retention window.
            # A job that straddles the edge has some of its output deleted, so
            # the absence of a "Completed " line proves nothing about it.
            knowable = start >= horizon

            kind = classify(name, completed, knowable, d)
            hours[kind] += vh
            gbh[kind] += gh
            jobs[kind] += 1
            rows.append((name.rsplit("-", 1)[0], kind, vh, gh))

    # ---- CALIBRATION ------------------------------------------------------
    # Pricing vCPU-hours at a list rate produced $1,111 against a validated
    # $2,976. The gap was three things: array children never counted, memory
    # never priced, and a guessed rate. Rather than guess again, derive ONE
    # all-in rate from a billed figure over a window whose usage we measured:
    #
    #   $/vCPU-h = (billed - standing baseline - things that are not ours)
    #              / (our vCPU-hours over the same days)
    #
    # It bundles memory, logs, requests and IPv4 into the vCPU-hour because our
    # tasks are a fixed 8 vCPU / 16 GB shape, so those scale with vCPU-hours
    # too. That holds only while the shape holds; change the task size and this
    # must be recalibrated.
    rate, basis = None, None
    try:
        act = json.load(open("costs_actual.json"))
        w = act["campaign_window"]
        # Calibrate on days that can actually price compute: inside the
        # window, carrying real load, and not on the unattributed list. A day
        # billing 432% of on-demand LIST against our measured hours is not
        # telling us about Spot - it is telling us something else was billed
        # that day - and averaging it in raised the rate 21%.
        floor = act.get("calibration_min_vcpu_hours_per_day", 0)
        skip = {u["day"] for u in act.get("unattributed", [])}
        per_day_excl = collections.Counter()
        for e in act.get("exclusions", []):
            for dd in e.get("days", []):
                per_day_excl[dd] += e["amount"] / max(len(e.get("days", [])), 1)
        days = [d for d in sorted(act["daily"])
                if w["start"] <= d <= w["end"] and d not in skip
                and per_day_vh.get(d, 0.0) >= floor]
        net = sum(act["daily"][d] - act["baseline_per_day"] - per_day_excl[d]
                  for d in days)
        used = sum(per_day_vh.get(d, 0.0) for d in days)
        if used > 0 and net > 0:
            rate = net / used
            basis = (f"calibrated: ${net:,.2f} billed over {len(days)} day(s) "
                     f"carrying real load ({', '.join(days)}; baseline and "
                     f"non-campaign items removed) / {used:,.0f} vCPU-h "
                     f"measured on those days")
    except Exception as exc:
        print(f"no calibration from costs_actual.json ({exc})", file=sys.stderr)
    if rate is None:
        rate = (FARGATE_ONDEMAND_VCPU_H + 2.0 * FARGATE_ONDEMAND_GB_H) \
            * FALLBACK_SPOT_FRACTION
        basis = (f"UNCALIBRATED fallback: {FALLBACK_SPOT_FRACTION:.0%} of "
                 f"Fargate on-demand list for an 8 vCPU / 16 GB task")

    spend = collections.Counter()
    by_campaign = collections.defaultdict(lambda: collections.Counter())
    for camp, kind, vh, gh in rows:
        spend[kind] += vh * rate
        by_campaign[camp][kind] += vh * rate

    total = sum(spend.values())
    order = ORDER
    table = [(k, jobs[k], hours[k], spend[k]) for k in order if jobs[k]]

    if a.out:
        doc = {
            "generated": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "since": a.since,
            "rate_per_vcpu_hour": round(rate, 6),
            "rate_basis": basis,
            "rate_is_calibrated": "calibrated" in basis,
            "reconciled_against_a_bill": "calibrated" in basis,
            "vcpu_hours_by_day": {d: round(v, 1)
                                  for d, v in sorted(per_day_vh.items())},
            "categories": {k: {"jobs": jobs[k], "vcpu_hours": round(hours[k], 1),
                               "gb_hours": round(gbh[k], 1),
                               "spend": round(spend[k], 2)} for k in order
                           if jobs[k]},
            "total": {"jobs": sum(jobs.values()),
                      "vcpu_hours": round(sum(hours.values()), 1),
                      "gb_hours": round(sum(gbh.values()), 1),
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
        print("| category | jobs | vCPU-h | GB-h | spend | share |")
        print("|---|--:|--:|--:|--:|--:|")
        for k, j, h, s in table:
            print(f"| {k} | {j:,} | {h:,.0f} | {gbh[k]:,.0f} | ${s:,.2f} | "
                  f"{100 * s / total if total else 0:.1f}% |")
        print(f"| **total** | **{sum(jobs.values()):,}** | "
              f"**{sum(hours.values()):,.0f}** | **{sum(gbh.values()):,.0f}** | "
              f"**${total:,.2f}** | |")
        print(f"\n_${rate:.5f}/vCPU-h — {basis}._")
    else:
        print(f"QuakeScope spend"
              + (f" since {a.since}" if a.since else " (all Batch remembers)"))
        print(f"  {'category':12} {'jobs':>6} {'vCPU-h':>10} {'GB-h':>11} {'spend':>11}   share")
        for k, j, h, s in table:
            print(f"  {k:12} {j:>6,} {h:>10,.0f} {gbh[k]:>11,.0f} "
                  f"{'$' + format(s, ',.2f'):>11}"
                  f"   {100 * s / total if total else 0:5.1f}%")
        print(f"  {'TOTAL':12} {sum(jobs.values()):>6,} "
              f"{sum(hours.values()):>10,.0f} {sum(gbh.values()):>11,.0f} "
              f"{'$' + format(total, ',.2f'):>11}")
        waste = spend["spinning"]
        if waste:
            print(f"\n  ${waste:,.2f} went to workers that ran, completed no "
                  f"shard, and were not\n  reclaimed by Spot "
                  f"({100 * waste / total:.0f}% of the total). That is the "
                  f"avoidable part.")
        if spend["log expired"]:
            print(f"\n  ${spend['log expired']:,.2f} "
                  f"({100 * spend['log expired'] / total:.0f}%) is from jobs "
                  f"older than the log group's\n  retention window. Whether "
                  f"they completed anything cannot be recovered;\n  it is "
                  f"reported rather than guessed. Raise retention to shrink it.")
        print(f"\n  ${rate:.5f}/vCPU-h all-in.\n  {basis}.")
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
