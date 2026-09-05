#!/usr/bin/env python
"""Export the 2026-09-04 EarthScope incident logs before retention eats them.

CloudWatch keeps `/aws/batch/job` for FIVE DAYS. The incident ran 2026-09-04
15:00-16:40 UTC, so the raw evidence for it disappears on 2026-09-09 and cannot
be recovered from anywhere. Everything downstream - the audit report, the
lessons, anything EarthScope or a reviewer wants to check - rests on log lines
that are about to stop existing. This copies them somewhere durable and writes
down what they mean.

WHAT IT PRODUCES

    raw/<window>.jsonl.gz     every event in the window, one JSON object per
                              line, REDACTED (see below), in timestamp order
    summary/<window>.json     counts that the report quotes, recomputed here
                              from the same events, so a reader can check the
                              numbers rather than trust them
    manifest.json             provenance: log group, region, retention, the
                              query windows, event counts, and a sha256 for
                              every file written

REDACTION IS NOT OPTIONAL. earthscope_sdk logs the refresh token and the access
token at DEBUG:

    logger.debug(f"Refreshed tokens: {self._tokens}")
    logger.debug(f"Refresh token revoked: {refresh_token}")

`s3_helper` pins that logger to INFO precisely so those lines cannot be emitted,
but this export is the wrong place to rely on that having always been true -
these logs span builds. Every event is passed through `redact()` before it is
written, and `--verify-redaction` re-scans what was written and fails if
anything still looks like a credential.

USAGE

    python scripts/export_incident_logs.py --out ./incident_export
    python scripts/export_incident_logs.py --out ./incident_export \\
        --upload s3://quakescope-picks-2026/incident/2026-09-04-earthscope

The upload prefix is deliberately NOT one of the publicly readable ones
(`*/picks/*`, `*/manifests/*`, `*/runs/*`), so the raw export stays private.
Publish the derived documents, not this.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import re
import sys
import time

REGION = "us-east-2"
LOG_GROUP = "/aws/batch/job"

# The windows worth keeping, and why each one exists.
WINDOWS = {
    "incident": (
        "2026-09-04T14:50", "2026-09-04T16:45",
        "The event EarthScope reported. Token-endpoint storm, year-less "
        "requests for temporary networks, and the fleet being stopped.",
    ),
    "dryrun3-broken": (
        "2026-09-05T01:15", "2026-09-05T01:30",
        "First dry run on the merged fix. Credential path correct; three "
        "shards lost to the cached SimpleSyncRunner on their second year.",
    ),
    "dryrun3-fixed": (
        "2026-09-05T01:55", "2026-09-05T02:05",
        "Same eight shards after pinning the SDK sync runner. 8/8 complete, "
        "second-year credentials issued for the first time.",
    ),
    "dryrun4": (
        "2026-09-05T02:25", "2026-09-05T04:15",
        "Read-path validation on 4F. 1.24M picks, and four mid-read credential "
        "renewals - the case that would have failed before the runner fix.",
    ),
}

# Anything that could be a credential. Deliberately broad: a false positive
# costs a redacted line, a false negative publishes a token.
_SECRET_PATTERNS = [
    (re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]*"),
     "<JWT-REDACTED>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<AWS-ACCESS-KEY-REDACTED>"),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), "<AWS-STS-KEY-REDACTED>"),
    (re.compile(r"(?i)(refresh[_\- ]?token[\"'\s:=]+)([A-Za-z0-9_\-\.]{12,})"),
     r"\1<REFRESH-TOKEN-REDACTED>"),
    (re.compile(r"(?i)(access[_\- ]?token[\"'\s:=]+)([A-Za-z0-9_\-\.]{12,})"),
     r"\1<ACCESS-TOKEN-REDACTED>"),
    (re.compile(r"(?i)(session[_\- ]?token[\"'\s:=]+)([A-Za-z0-9/+=]{20,})"),
     r"\1<SESSION-TOKEN-REDACTED>"),
    (re.compile(r"(?i)(authorization:\s*bearer\s+)(\S+)"),
     r"\1<BEARER-REDACTED>"),
    (re.compile(r"(?i)(secret[_\- ]?access[_\- ]?key[\"'\s:=]+)(\S{20,})"),
     r"\1<SECRET-KEY-REDACTED>"),
    (re.compile(r"(?i)(x-amz-security-token[:=]\s*)(\S{20,})"),
     r"\1<SECURITY-TOKEN-REDACTED>"),
    # Presigned URL signature material.
    (re.compile(r"(X-Amz-Signature=)([0-9a-f]{16,})"), r"\1<SIG-REDACTED>"),
    (re.compile(r"(X-Amz-Credential=)([^&\s]+)"), r"\1<CRED-REDACTED>"),
]


def redact(text: str) -> tuple[str, int]:
    """Return the text with anything credential-shaped removed, and a count."""
    n = 0
    for pat, repl in _SECRET_PATTERNS:
        text, k = pat.subn(repl, text)
        n += k
    return text, n


def _ms(stamp: str) -> int:
    dt = datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M").replace(
        tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


# What the report claims, recomputed from the events themselves. Each key is a
# number a reader can check against reports/earthscope_credential_audit.html.
COUNTERS = {
    "token_endpoint_posts": lambda m: "POST https://login.earthscope.org/oauth/token" in m,
    "credential_requests": lambda m: "credentials/aws/s3-miniseed-v2" in m,
    "yearless_temporary_requests": lambda m: bool(
        re.search(r"network=FDSN%3A[0-9XYZ]", m) and "year=" not in m),
    "verdicts_cached": lambda m: "Not retrying" in m,
    "retry_sleeps": lambda m: "credential request failed for" in m,
    "auth_failed": lambda m: "user is blocked" in m or "InvalidRefreshToken" in m,
    "access_denied_branch": lambda m: "Credential refreshed after access denied" in m,
    "station_day_timeouts": lambda m: "Timeout after" in m,
    "shard_failures": lambda m: " failed: " in m,
    "credential_renewed": lambda m: "credential renewed" in m.lower(),
}


def export_window(logs, name, start, end, out_dir, budget_s):
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{name}.jsonl.gz")

    counts = {k: 0 for k in COUNTERS}
    per_minute = {}
    status_codes = {}
    n = redacted = 0
    tok = None
    t0 = time.time()
    truncated = False

    with gzip.open(path, "wt", encoding="utf-8") as fh:
        while True:
            kw = dict(logGroupName=LOG_GROUP, startTime=start, endTime=end,
                      limit=10000)
            if tok:
                kw["nextToken"] = tok
            r = logs.filter_log_events(**kw)
            for e in r.get("events", []):
                msg, k = redact(e.get("message", ""))
                redacted += k
                n += 1
                fh.write(json.dumps({
                    "ts": e["timestamp"],
                    "utc": datetime.datetime.utcfromtimestamp(
                        e["timestamp"] / 1000).isoformat(timespec="milliseconds") + "Z",
                    "stream": e.get("logStreamName"),
                    "message": msg,
                }, separators=(",", ":")) + "\n")
                for key, pred in COUNTERS.items():
                    if pred(msg):
                        counts[key] += 1
                if "oauth/token" in msg:
                    minute = datetime.datetime.utcfromtimestamp(
                        e["timestamp"] / 1000).strftime("%H:%M")
                    per_minute[minute] = per_minute.get(minute, 0) + 1
                m = re.search(r'HTTP/1\.1 (\d{3})', msg)
                if m:
                    status_codes[m.group(1)] = status_codes.get(m.group(1), 0) + 1
            tok = r.get("nextToken")
            if not tok:
                break
            if time.time() - t0 > budget_s:
                truncated = True
                break

    peak = max(per_minute.items(), key=lambda kv: kv[1]) if per_minute else None
    summary = {
        "window": name,
        "why": WINDOWS[name][2],
        "start_utc": datetime.datetime.utcfromtimestamp(start / 1000).isoformat() + "Z",
        "end_utc": datetime.datetime.utcfromtimestamp(end / 1000).isoformat() + "Z",
        "events": n,
        "truncated": truncated,
        "redactions": redacted,
        "counts": counts,
        "http_status_codes": dict(sorted(status_codes.items())),
        "token_posts_per_minute": dict(sorted(per_minute.items())),
        "token_posts_peak_minute": {"minute_utc": peak[0], "posts": peak[1]} if peak else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    sdir = os.path.join(out_dir, "summary")
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, f"{name}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return path, summary


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_redaction(out_dir):
    """Re-read what was written and fail if anything still looks like a secret."""
    bad = []
    raw = os.path.join(out_dir, "raw")
    for fn in sorted(os.listdir(raw)):
        with gzip.open(os.path.join(raw, fn), "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                msg = json.loads(line)["message"]
                for pat, _ in _SECRET_PATTERNS:
                    hit = pat.search(msg)
                    # A replacement marker is the expected outcome, not a leak.
                    if hit and "REDACTED" not in hit.group(0):
                        bad.append((fn, i, hit.group(0)[:60]))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./incident_export")
    ap.add_argument("--upload", help="s3://bucket/prefix (keep it non-public)")
    ap.add_argument("--windows", help="comma separated subset of "
                    + ",".join(WINDOWS))
    ap.add_argument("--budget", type=float, default=900,
                    help="seconds per window before truncating (default 900)")
    a = ap.parse_args(argv)

    import boto3
    logs = boto3.client("logs", region_name=REGION)
    os.makedirs(a.out, exist_ok=True)

    names = [w for w in (a.windows.split(",") if a.windows else WINDOWS)
             if w in WINDOWS]
    files, summaries = [], {}
    for name in names:
        s, e, _ = WINDOWS[name]
        print(f"  {name}: {s} -> {e} ...", flush=True)
        path, summary = export_window(logs, name, _ms(s), _ms(e), a.out, a.budget)
        files.append(path)
        summaries[name] = summary
        print(f"     {summary['events']:,} events, "
              f"{summary['redactions']} redactions, "
              f"{os.path.getsize(path)/1e6:.1f} MB"
              + ("  [TRUNCATED]" if summary["truncated"] else ""))

    bad = verify_redaction(a.out)
    if bad:
        print("\n  REDACTION FAILED - not uploading. Offending lines:")
        for f, i, t in bad[:10]:
            print(f"    {f}:{i}  {t}")
        return 1
    print("  redaction verified: nothing credential-shaped survives")

    try:
        grp = logs.describe_log_groups(
            logGroupNamePrefix=LOG_GROUP)["logGroups"][0]
        retention = grp.get("retentionInDays")
    except Exception:
        retention = None

    # The manifest describes the EXPORT DIRECTORY, not just this run. A run
    # with --windows incident used to overwrite it with that one window and
    # silently drop the other three, so the index disagreed with the files
    # sitting next to it. Pick up every summary and every raw file present.
    sdir = os.path.join(a.out, "summary")
    for fn in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
        if fn.endswith(".json"):
            w = fn[:-5]
            if w not in summaries:
                summaries[w] = json.load(open(os.path.join(sdir, fn)))
    rdir = os.path.join(a.out, "raw")
    for fn in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        fp = os.path.join(rdir, fn)
        if fp not in files:
            files.append(fp)

    manifest = {
        "what": "Raw CloudWatch evidence for the 2026-09-04 EarthScope "
                "credential incident and the dry runs that validated the fix.",
        "log_group": LOG_GROUP,
        "region": REGION,
        "retention_days": retention,
        "evidence_expires_utc": "2026-09-09",
        "exported_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "redaction": "every event passed through export_incident_logs.redact(); "
                     "verified by re-scanning the written files",
        "windows": {k: {"why": WINDOWS.get(k, ("", "", ""))[2], **summaries[k]}
                    for k in sorted(summaries)},
        "files": {os.path.relpath(f, a.out): {
            "bytes": os.path.getsize(f), "sha256": sha256(f)} for f in files},
    }
    mpath = os.path.join(a.out, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  wrote {mpath}")

    if a.upload:
        assert a.upload.startswith("s3://")
        bucket, _, prefix = a.upload[5:].partition("/")
        s3 = boto3.client("s3")
        for root, _, fns in os.walk(a.out):
            for fn in fns:
                p = os.path.join(root, fn)
                key = f"{prefix.rstrip('/')}/{os.path.relpath(p, a.out)}"
                s3.upload_file(p, bucket, key)
                print(f"  uploaded s3://{bucket}/{key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
