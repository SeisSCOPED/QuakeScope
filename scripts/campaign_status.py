"""Hourly campaign status, read from S3 and Batch.

    python scripts/campaign_status.py [--markdown]

Every number here is counted from an API response. Nothing is estimated,
extrapolated or remembered between runs - if a figure cannot be read it is
reported as unknown rather than filled in, because a status page that quietly
invents a number is worse than one that admits a gap.

Read per campaign prefix:
  shards.jsonl      the planned work: shard count, station-days
  complete/         one object per finished shard
  claims/           one per shard a worker holds; minus complete = in flight
  manifests/        what each finished shard wrote: picks, station-days, files
  picks/            the catalogue itself: object count and bytes

Read from Batch: job counts by status, and the reason on anything that failed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys

import boto3

REGION = "us-east-2"
BUCKET = "quakescope-picks-2026"
QUEUES = ["niyiyu_earthscope_missing_station", "pickblue_jobqueue"]
# Restructured 2026-09-03: scedc+ncedc+earthscope -> `global`.
CAMPAIGNS = ["global", "obs", "western"]

# A claim older than this with no completion is not progress, it is a stall:
# the shard timeout is 24 h and a worker that dies without releasing leaves the
# claim behind.
STALE_CLAIM_HOURS = 26


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def count_prefix(s3, prefix: str) -> tuple[int, int, datetime.datetime | None]:
    """(objects, bytes, oldest LastModified) under a prefix."""
    n = size = 0
    oldest = None
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        for o in page.get("Contents", []):
            n += 1
            size += o["Size"]
            if oldest is None or o["LastModified"] < oldest:
                oldest = o["LastModified"]
    return n, size, oldest


def read_campaign(s3, name: str) -> dict:
    d: dict = {"campaign": name}
    try:
        body = s3.get_object(Bucket=BUCKET, Key=f"{name}/shards.jsonl")["Body"].read()
        rows = [json.loads(x) for x in body.decode().splitlines() if x.strip()]
        d["shards"] = len(rows)
        d["planned_station_days"] = sum(r.get("n_station_days", 0) for r in rows)
    except Exception:
        d["shards"] = None            # queue not written
        d["planned_station_days"] = None

    d["complete"], _, _ = count_prefix(s3, f"{name}/complete/")
    n_claims, _, oldest_claim = count_prefix(s3, f"{name}/claims/")
    d["in_flight"] = max(n_claims - d["complete"], 0)
    d["oldest_claim_h"] = (
        (_now() - oldest_claim).total_seconds() / 3600 if oldest_claim else None
    )
    d["picks_objects"], d["picks_bytes"], _ = count_prefix(s3, f"{name}/picks/")

    # Manifests carry the authoritative pick count: the writer records what it
    # actually wrote, so this does not require opening any Parquet.
    picks = sdays = files = 0
    n_man = 0
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=f"{name}/manifests/"
    ):
        for o in page.get("Contents", []):
            n_man += 1
            m = json.loads(s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read())
            picks += m.get("n_picks", 0)
            sdays += m.get("station_days", 0)
            files += len(m.get("files", []))
    d["manifests"] = n_man
    d["picks"] = picks
    d["done_station_days"] = sdays
    d["parquet_files"] = files
    return d


def read_batch(b) -> dict:
    out: dict = {"by_status": {}, "failed": []}
    for q in QUEUES:
        try:
            for st in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING",
                       "RUNNING", "FAILED"):
                jobs = b.list_jobs(jobQueue=q, jobStatus=st)["jobSummaryList"]
                if not jobs:
                    continue
                out["by_status"][st] = out["by_status"].get(st, 0) + len(jobs)
                if st == "FAILED":
                    for j in jobs[:5]:
                        out["failed"].append(
                            {"name": j.get("jobName", "?"),
                             "reason": j.get("statusReason", "")[:110]}
                        )
        except Exception as e:
            out.setdefault("errors", []).append(f"{q}: {type(e).__name__}")
    return out


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def render(rows: list[dict], batch: dict, md: bool) -> tuple[str, str]:
    L = []
    h = "### " if md else ""
    L.append(f"{h}Campaign status — {_now():%Y-%m-%d %H:%M} UTC")
    L.append("")
    if md:
        L.append("| campaign | shards | done | in flight | % | picks | parquet | size |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        if r["shards"] is None and r["complete"] == 0 and r["picks_objects"] == 0:
            continue                      # queue not written and nothing there
        tot = r["shards"]
        pct = f"{100*r['complete']/tot:.1f}%" if tot else "-"
        if md:
            L.append(
                f"| `{r['campaign']}` | {tot or '-':,} | {r['complete']:,} | "
                f"{r['in_flight']:,} | {pct} | {r['picks']:,} | "
                f"{r['parquet_files']:,} | {human(r['picks_bytes'])} |"
                if isinstance(tot, int) else
                f"| `{r['campaign']}` | - | {r['complete']:,} | {r['in_flight']:,} "
                f"| {pct} | {r['picks']:,} | {r['parquet_files']:,} | "
                f"{human(r['picks_bytes'])} |"
            )
        else:
            L.append(f"  {r['campaign']:<11} {r['complete']:>7,}/{tot or 0:<8,} "
                     f"({pct:>6})  {r['picks']:>12,} picks  "
                     f"{human(r['picks_bytes']):>9}")
    L.append("")

    warn = []
    for r in rows:
        if r["oldest_claim_h"] and r["oldest_claim_h"] > STALE_CLAIM_HOURS \
                and r["in_flight"]:
            warn.append(
                f"`{r['campaign']}`: a claim has been held "
                f"{r['oldest_claim_h']:.0f} h with the shard unfinished — the "
                f"shard timeout is 24 h, so a worker likely died without "
                f"releasing it"
            )
        if r["complete"] and not r["picks"]:
            warn.append(
                f"`{r['campaign']}`: {r['complete']:,} shards complete but zero "
                f"picks recorded — shards are finishing without writing"
            )
    for f in batch.get("failed", []):
        warn.append(f"failed job `{f['name']}`: {f['reason'] or 'no reason given'}")
    for e in batch.get("errors", []):
        warn.append(f"could not read queue {e}")

    st = batch["by_status"]
    L.append(f"{'**' if md else ''}Batch{'**' if md else ''}: "
             + (", ".join(f"{k} {v}" for k, v in sorted(st.items())) or "nothing active"))
    L.append("")
    if warn:
        L.append(f"{'#### ' if md else ''}Needs attention")
        for w in warn:
            L.append(f"- {w}" if md else f"  ! {w}")
    else:
        L.append("No stalled claims, no failed jobs, no shard completing without picks.")

    # One line that changes only when something meaningful changes - used to
    # decide whether to notify, so an idle hour stays silent.
    sig = "|".join(
        f"{r['campaign']}:{r['complete']}:{r['picks']}" for r in rows
    ) + "|" + ",".join(f"{k}{v}" for k, v in sorted(st.items())) + \
        "|" + str(len(warn))
    return "\n".join(L), sig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--campaigns", default=",".join(CAMPAIGNS))
    a = ap.parse_args()
    s3 = boto3.client("s3", region_name=REGION)
    b = boto3.client("batch", region_name=REGION)
    rows = [read_campaign(s3, c) for c in a.campaigns.split(",") if c]
    body, sig = render(rows, read_batch(b), a.markdown)
    print(body)
    print(f"\n<!-- sig:{sig} -->" if a.markdown else f"\nsig:{sig}")


if __name__ == "__main__":
    sys.exit(main())
