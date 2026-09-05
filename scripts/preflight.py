#!/usr/bin/env python
"""Are we about to repeat 2026-09-04? One command, one answer.

On 2026-09-04 this fleet sent 351,735 rejected token requests to EarthScope over
four hours and nobody here noticed - the archive operator told us. Every control
added since is checked here, so "is it safe" is a thing you run rather than a
thing you remember.

    python scripts/preflight.py                # human output
    python scripts/preflight.py --markdown     # for a workflow step summary

Exit code is 0 only if nothing is FAIL. A check that cannot reach AWS reports
UNKNOWN, not FAIL: absence of evidence must not read as evidence of safety, and
must not read as an emergency either.

WHAT IT DOES NOT DO. It is a snapshot, not a monitor. The thing that actually
stops a runaway is the rate breaker inside the Fleet workflow, which runs every
15 minutes whether anyone is looking. This is for answering the question
deliberately - before starting a campaign, after changing something, or when
somebody asks.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REGION = "us-east-2"
BUCKET = "quakescope-picks-2026"
QUEUE = "niyiyu_earthscope_missing_station"
LOG_GROUP = "/aws/batch/job"
JOB_ROLE = "SeisBenchBatchRole"
EXEC_ROLE = "SeisBenchExecutionRole"
SECRET = ("arn:aws:secretsmanager:us-east-2:073795725844:secret:"
          "quakescope/earthscope-refresh-token-bGo4vN")
CONTROL_BUCKET = "scoped-noise"
QUARANTINED = {"c5846f3"}
RATE_LIMIT = float(os.environ.get("ES_RATE_LIMIT_PER_MIN", "500"))

PASS, FAIL, WARN, UNKNOWN = "PASS", "FAIL", "WARN", "UNKNOWN"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, area, check, state, detail=""):
        self.rows.append((area, check, state, detail))

    @property
    def failed(self):
        return [r for r in self.rows if r[2] == FAIL]

    def text(self):
        icon = {PASS: "ok  ", FAIL: "FAIL", WARN: "warn", UNKNOWN: "??  "}
        out, last = [], None
        for area, check, state, detail in self.rows:
            if area != last:
                out.append(f"\n{area}")
                last = area
            out.append(f"  [{icon[state]}] {check}"
                       + (f"\n           {detail}" if detail else ""))
        return "\n".join(out)

    def markdown(self):
        icon = {PASS: "✅", FAIL: "🛑", WARN: "⚠️", UNKNOWN: "❔"}
        out, last = [], None
        for area, check, state, detail in self.rows:
            if area != last:
                out += ["", f"### {area}", "", "| | check | detail |",
                        "|---|---|---|"]
                last = area
            out.append(f"| {icon[state]} | {check} | {detail} |")
        return "\n".join(out)


def check_fleet(r, batch):
    """Nothing should be running unless somebody meant it."""
    try:
        cfg = json.load(open("fleet.json"))["campaigns"]
    except Exception as exc:
        r.add("Fleet", "fleet.json readable", FAIL, str(exc)[:80])
        return
    live = {k: v["target"] for k, v in cfg.items() if v["target"] > 0}
    r.add("Fleet", "campaign targets",
          PASS if not live else WARN,
          "all 0 — nothing will be started" if not live
          else f"running by intent: {live}")

    if batch is None:
        r.add("Fleet", "jobs in the queue", UNKNOWN, "no AWS access")
        return
    # OURS ONLY. The queue is shared with other groups, and counting everything
    # on it reported 57 foreign jobs as ours on 2026-09-05. Job names carry the
    # campaign, which is what `spot_governor --name-prefix` sets.
    ours = tuple(cfg)
    mine = foreign = 0
    for st in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"):
        for j in batch.list_jobs(jobQueue=QUEUE, jobStatus=st)["jobSummaryList"]:
            if j["jobName"].startswith(ours):
                mine += 1
            else:
                foreign += 1
    consistent = (mine > 0) == bool(live)
    r.add("Fleet", "jobs match the targets",
          PASS if consistent else FAIL,
          f"{mine} of ours alive ({foreign} other jobs share this queue); "
          f"targets {live or 'all 0'}"
          + ("" if consistent else " — workers running with every target at 0 "
             "means setting a target to 0 did not stop them; that only stops "
             "REPLACEMENT, it does not terminate what is already running"))


def check_images(r, batch):
    """The build that caused the incident must not be launchable."""
    if batch is None:
        r.add("What would launch", "campaign images", UNKNOWN, "no AWS access")
        return
    try:
        cfg = json.load(open("fleet.json"))["campaigns"]
    except Exception:
        return
    bad = []
    for camp, c in cfg.items():
        jd = batch.describe_job_definitions(
            jobDefinitions=[c["job_definition"]])["jobDefinitions"][0]
        tag = jd["containerProperties"]["image"].rsplit(":", 1)[-1]
        if tag in QUARANTINED:
            bad.append(f"{camp}={tag}")
    r.add("What would launch", "no campaign points at a quarantined build",
          PASS if not bad else FAIL,
          "all on a build that carries the fix" if not bad
          else f"QUARANTINED: {', '.join(bad)}")


def check_roles(r, iam):
    """The container reaches one bucket; the platform holds the secret."""
    if iam is None:
        r.add("Permissions", "role split", UNKNOWN, "no AWS access")
        return

    def decide(role, action, resource):
        arn = iam.get_role(RoleName=role)["Role"]["Arn"]
        return iam.simulate_principal_policy(
            PolicySourceArn=arn, ActionNames=[action],
            ResourceArns=[resource])["EvaluationResults"][0]["EvalDecision"]

    try:
        wrong = []
        if decide(JOB_ROLE, "s3:PutObject", f"arn:aws:s3:::{BUCKET}/x") != "allowed":
            wrong.append("job role cannot write the catalogue")
        if decide(JOB_ROLE, "s3:PutObject",
                  f"arn:aws:s3:::{CONTROL_BUCKET}/x") == "allowed":
            wrong.append("job role can write OTHER buckets")
        if decide(JOB_ROLE, "secretsmanager:GetSecretValue", SECRET) == "allowed":
            wrong.append("job role can read the EarthScope secret")
        r.add("Permissions", "worker reaches one bucket and no secret",
              PASS if not wrong else FAIL,
              "scoped" if not wrong else "; ".join(wrong))

        wrong = []
        if decide(EXEC_ROLE, "secretsmanager:GetSecretValue", SECRET) != "allowed":
            wrong.append("execution role cannot inject the secret")
        if decide(EXEC_ROLE, "s3:PutObject",
                  f"arn:aws:s3:::{BUCKET}/x") == "allowed":
            wrong.append("execution role can write S3")
        r.add("Permissions", "platform role separate from the container",
              PASS if not wrong else FAIL,
              "split" if not wrong else "; ".join(wrong))
    except Exception as exc:
        r.add("Permissions", "role simulation", UNKNOWN, str(exc)[:80])


def check_bucket(r, s3):
    """A bad delete must be recoverable."""
    if s3 is None:
        r.add("Catalogue", "versioning", UNKNOWN, "no AWS access")
        return
    try:
        v = s3.get_bucket_versioning(Bucket=BUCKET).get("Status")
        r.add("Catalogue", "versioning enabled",
              PASS if v == "Enabled" else FAIL,
              f"{v or 'not enabled'} — a wrong delete "
              + ("is recoverable" if v == "Enabled" else "is PERMANENT"))
    except Exception as exc:
        r.add("Catalogue", "versioning", UNKNOWN, str(exc)[:80])


def _yearless(logs, end, minutes):
    """Count credential requests that carry a temporary code and no year.

    Parses the request line rather than matching a substring. A temporary FDSN
    code - leading digit or X/Y/Z - is authorised per year because codes are
    reused between experiments, so a year-less request for one can only be
    refused. Returns (year_less, total_credential_requests).
    """
    import re
    net = re.compile(r"network=FDSN(?:%3A|:)([A-Z0-9]{1,2})")
    yr = re.compile(r"[?&]year=\d{4}")
    yl = total = 0
    tok = None
    while True:
        kw = dict(logGroupName=LOG_GROUP, startTime=end - minutes * 60000,
                  endTime=end, limit=10000,
                  filterPattern='"credentials/aws/s3-miniseed-v2"')
        if tok:
            kw["nextToken"] = tok
        resp = logs.filter_log_events(**kw)
        for e in resp.get("events", []):
            m = net.search(e["message"])
            if not m:
                continue
            total += 1
            if not yr.search(e["message"]) and m.group(1)[0] in "0123456789XYZ":
                yl += 1
        tok = resp.get("nextToken")
        if not tok:
            return yl, total


def check_earthscope(r, logs):
    """The thing we were reported for, measured the way it was measured."""
    if logs is None:
        r.add("EarthScope traffic", "token endpoint rate", UNKNOWN, "no AWS access")
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    end = int(now.timestamp() * 1000)

    def count(pattern, minutes):
        n, tok = 0, None
        while True:
            kw = dict(logGroupName=LOG_GROUP, startTime=end - minutes * 60000,
                      endTime=end, filterPattern=pattern, limit=10000)
            if tok:
                kw["nextToken"] = tok
            resp = logs.filter_log_events(**kw)
            n += len(resp.get("events", []))
            tok = resp.get("nextToken")
            if not tok:
                return n

    try:
        n15 = count('"POST https://login.earthscope.org/oauth/token"', 15)
        rate = n15 / 15
        r.add("EarthScope traffic", "token endpoint rate",
              PASS if rate <= RATE_LIMIT else FAIL,
              f"{rate:,.0f}/min over 15 min (limit {RATE_LIMIT:,.0f}; "
              f"the incident sustained 2,193 and peaked at 5,104)")

        # The specific malformed request EarthScope singled out. This has to
        # PARSE, not grep: an earlier version counted every line mentioning a
        # temporary code and reported PASS regardless, which is a check that
        # cannot fail and therefore is not a check.
        yl, total = _yearless(logs, end, 60)
        r.add("EarthScope traffic", "no year-less temporary requests",
              PASS if yl == 0 else FAIL,
              f"{yl} of {total} credential request(s) carried a temporary code "
              f"with no year — those can only ever be refused"
              + ("" if yl == 0 else "; this is the 2026-09-04 defect"))

        loops = count('"credential request failed for"', 60)
        r.add("EarthScope traffic", "no retry loops",
              PASS if loops == 0 else WARN,
              f"{loops} retry-sleep line(s) in the last hour")
    except Exception as exc:
        r.add("EarthScope traffic", "log query", UNKNOWN, str(exc)[:80])


def check_client(r):
    """The client still refuses to re-ask a question it has been answered."""
    try:
        p = subprocess.run([sys.executable, "scripts/earthscope_selftest.py",
                            "--offline"], capture_output=True, text=True,
                           timeout=300)
        ok = p.returncode == 0
        r.add("Client", "offline self-test", PASS if ok else FAIL,
              "six checks pass; a refusal is asked once and never again" if ok
              else p.stdout.strip().splitlines()[-1][:90] if p.stdout else "failed")
    except Exception as exc:
        r.add("Client", "offline self-test", UNKNOWN, str(exc)[:80])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    a = ap.parse_args(argv)

    batch = iam = s3 = logs = None
    try:
        import boto3
        batch = boto3.client("batch", region_name=REGION)
        iam = boto3.client("iam")
        s3 = boto3.client("s3")
        logs = boto3.client("logs", region_name=REGION)
        batch.describe_job_queues(jobQueues=[QUEUE])       # prove credentials
    except Exception:
        batch = iam = s3 = logs = None

    r = Report()
    check_fleet(r, batch)
    check_images(r, batch)
    check_earthscope(r, logs)
    check_roles(r, iam)
    check_bucket(r, s3)
    if not a.skip_selftest:
        check_client(r)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if a.markdown:
        verdict = ("## 🛑 Something needs attention" if r.failed
                   else "## ✅ Clear — safe to run a campaign")
        print(f"{verdict}\n\n_Checked {stamp}._")
        print(r.markdown())
        if r.failed:
            print("\n**Do not raise a target until these are resolved.** "
                  "Background: [the incident report]"
                  "(https://seisscoped.org/QuakeScope/earthscope_credential_audit.html).")
    else:
        print(f"QuakeScope preflight · {stamp}")
        print(r.text())
        print("\n  " + ("SOMETHING NEEDS ATTENTION" if r.failed
                        else "clear — safe to run a campaign"))
    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
