#!/usr/bin/env python
"""Plan (and optionally launch) the re-pick of station-days a resumed shard lost.

Background: docs/rerun_2026/28_resumed_shard_overwrite.md. Until 2026-09-16 a
worker resuming a checkpointed shard reused the first attempt's Parquet keys,
so the first attempt's earliest checkpoint files were replaced and its
station-days vanished from the manifest. The fix is in parquet_writer.py; this
script repairs the campaigns that ran before it.

What counts as lost, per resumed shard
--------------------------------------
A shard is *resumed* if the bucket holds pick files for it that its manifest
does not list (the first attempt's survivors), or if its progress object was
written by a different worker than the one that completed it. For each:

* P  = station-days with at least one pick in ANY surviving file of the shard
* R  = station-days the manifest records (the last attempt processed them)
* D  = station-days in the progress object when it still belongs to the first
       attempt (progress.worker != complete.worker). When a second attempt
       also checkpointed it overwrote that list, so D is instead the shard's
       plan (stations x days) cut at the first day the manifest records: the
       loader is day-major, a resumed attempt starts where the first stopped,
       so whatever the first attempt lost lies on or before that day.

lost = D - P - R. A station-day in D - P that genuinely had no data or no
arrival is re-picked and yields nothing again, which is harmless; a
station-day with picks anywhere is never re-picked, so nothing is duplicated.

The repair queue
----------------
Lost station-days are grouped by station into runs of consecutive days, runs
with identical dates are grouped into shards of at most --max-sd station-days,
and the queue is written to ``<campaign>-repair/shards.jsonl`` with the parent
campaign's ``stations.parquet`` copied beside it. Workers are then launched on
the parent's job definition with ``--parquet_uri`` pointing at the PARENT
prefix, so the repaired picks and their manifests land where readers already
look, and with ``--checkpoint-every 0`` so a repair shard can never itself be
resumed under an image that predates the fix.

    python scripts/repair_resumed_shards.py western            # plan, write queue
    python scripts/repair_resumed_shards.py western --launch 20   # ...and 20 workers

Needs credentials: progress/, complete/ and shards.jsonl are not public.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb_catalog.src.shard_planner import shard_id as make_shard_id  # noqa: E402

BUCKET, REGION = "quakescope-picks-2026", "us-east-2"


def _fleet():
    """(job definition, weight, queue key, output key) per campaign, from fleet.json.

    Since 2026-09-18 a campaign's queue lives under _queues/<name>/ and its
    picks under the catalogue prefix (docs/rerun_2026/29); both are recorded in
    fleet.json and default to the campaign name at the bucket root.
    """
    cfg = json.loads(Path(__file__).resolve().parents[1].joinpath("fleet.json").read_text())["campaigns"]
    out = {}
    for name, c in cfg.items():
        key = lambda u: u.split(f"s3://{BUCKET}/", 1)[1].rstrip("/") if u else name
        out[name] = (c["job_definition"], c["weight"], key(c.get("queue")), key(c.get("parquet_uri")))
    return out


JOBDEF = _fleet()
QUEUE = "niyiyu_earthscope_missing_station"
s3 = boto3.client("s3", region_name=REGION, config=BotoConfig(
    retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=120, max_pool_connections=64))
FILE_RE = re.compile(r"^(?P<sid>\d{7}-\d{7}-[0-9a-f]{12})(?:-\d{3})?\.parquet$")


def yd(s: str) -> datetime.date:
    y, d = s.split(".")
    return datetime.date(int(y), 1, 1) + datetime.timedelta(days=int(d) - 1)


def get_bytes(key: str, tries: int = 5) -> bytes | None:
    """GET with retries on the read itself: botocore's retry mode covers the
    request, not a body read that times out half way."""
    for i in range(tries):
        try:
            return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        except s3.exceptions.NoSuchKey:
            return None
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def get_json(key: str):
    body = get_bytes(key)
    return None if body is None else json.loads(body)


def list_keys(prefix: str) -> list[str]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        out += [o["Key"] for o in page.get("Contents", [])]
    return out


def pick_files_by_shard(camp: str) -> dict[str, list[str]]:
    files = collections.defaultdict(list)
    for key in list_keys(f"{camp}/picks/"):
        m = FILE_RE.match(key.rsplit("/", 1)[1])
        if m:
            files[m["sid"]].append(key)
    return files


def station_days_in(keys: list[str]) -> set[tuple]:
    """(tid, yr, doy) with at least one pick across these objects."""
    have = set()
    for key in keys:
        body = get_bytes(key)
        if body is None:                                  # gone between LIST and GET
            continue
        t = pq.read_table(io.BytesIO(body), columns=["tid", "peak"]).to_pandas()
        if len(t):
            t["yr"], t["doy"] = t.peak.dt.year, t.peak.dt.dayofyear
            have |= set(map(tuple, t[["tid", "yr", "doy"]].drop_duplicates().values.tolist()))
    return have


def analyse_shard(camp: str, shard: dict, keys: list[str]) -> dict | None:
    sid = shard["shard_id"]
    man = get_json(f"{camp}/manifests/{sid}.json")
    comp = get_json(f"{camp}/complete/{sid}.json")
    if man is None or comp is None:
        return None                                   # not finished: not ours to repair
    listed = {f["path"].split(f"s3://{BUCKET}/", 1)[1] for f in man["files"] if f["kind"] == "picks"}
    survivors = [k for k in keys if k not in listed]
    prog = get_json(f"{camp}/progress/{sid}.json")
    by_other = bool(prog) and prog.get("worker") != comp.get("worker")
    if not survivors and not by_other:
        return None                                   # never resumed
    P = station_days_in(keys)
    R = {(r["tid"], int(r["yr"]), int(r["doy"])) for r in man["records"]}
    if by_other:
        D = {(e[0], int(e[1]), int(e[2])) for e in prog["done"]}
        basis = "progress (first attempt)"
    else:
        # The second attempt checkpointed, so progress no longer describes the
        # first. The loader is day-major and a resumed attempt starts on the day
        # the first stopped, so the first attempt's lost files hold days at or
        # before the earliest day the manifest records.
        d0, d1 = yd(shard["start"]), yd(shard["end"])
        if R:
            cut = min(datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1) for _t, yr, doy in R)
            d1 = min(d1, cut + datetime.timedelta(days=1))
        days = [d0 + datetime.timedelta(days=k) for k in range((d1 - d0).days)]
        D = {(tid, d.year, d.timetuple().tm_yday) for tid in shard["stations"] for d in days}
        basis = "plan up to the resumed attempt's first day (progress overwritten)"
    lost = sorted(D - P - R)
    return dict(shard_id=sid, basis=basis, survivors=len(survivors), files=len(keys),
                planned=len(D), with_picks=len(P), recorded=len(R), lost=lost)


def plan_repair(lost: list[tuple], max_sd: int) -> list[dict]:
    """Runs of consecutive lost days per station -> shards of identical runs."""
    by_station = collections.defaultdict(list)
    for tid, yr, doy in lost:
        by_station[tid].append(datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1))
    runs = collections.defaultdict(list)                # (start, end_exclusive) -> [tid]
    for tid, days in by_station.items():
        days = sorted(set(days))
        start = prev = days[0]
        for d in days[1:] + [None]:
            if d is None or (d - prev).days > 1:
                runs[(start, prev + datetime.timedelta(days=1))].append(tid)
                if d is not None:
                    start = d
            prev = d if d is not None else prev
    shards = []
    for (start, end), tids in sorted(runs.items()):
        ndays = (end - start).days
        per = max(1, max_sd // ndays)
        for i in range(0, len(tids), per):
            group = sorted(tids[i:i + per])
            shards.append(dict(shard_id=make_shard_id(group, start, end), stations=group,
                               start=f"{start:%Y.%j}", end=f"{end:%Y.%j}",
                               n_station_days=len(group) * ndays))
    return shards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", choices=sorted(JOBDEF))
    ap.add_argument("--max-sd", type=int, default=200, help="station-days per repair shard")
    ap.add_argument("--max-shards", type=int, default=0, help="each launched worker stops after N shards (0 = drain)")
    ap.add_argument("--out", default="docs/rerun_2026/resumed_shards")
    ap.add_argument("--launch", type=int, default=0, help="submit this many workers after writing the queue")
    ap.add_argument("--plan-only", action="store_true", help="analyse and write the CSVs; touch nothing on S3")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--launch-only", action="store_true", help="skip the analysis; submit workers on an existing queue")
    a = ap.parse_args()
    camp, repair = a.campaign, f"{a.campaign}-repair"
    jobdef, weight, qkey, okey = JOBDEF[camp]
    if (qkey, okey) != (camp, camp):
        sys.exit(f"{camp}: queue {qkey}, output {okey}. This script predates the prefix reorganisation "
                 f"(docs/rerun_2026/29) and reads state from <campaign>/; adapt it before running.")

    if a.launch_only:
        if not a.launch:
            sys.exit("--launch-only needs --launch N")
        s3.head_object(Bucket=BUCKET, Key=f"{repair}/shards.jsonl")     # raises if there is no queue
        launch(camp, repair, jobdef, weight, a.launch, a.max_shards)
        return

    t0 = time.time()
    files = pick_files_by_shard(camp)
    shards = {s["shard_id"]: s for s in (json.loads(l) for l in
              s3.get_object(Bucket=BUCKET, Key=f"{camp}/shards.jsonl")["Body"].read().decode().splitlines() if l.strip())}
    prog_ids = {k.rsplit("/", 1)[1][:-5] for k in list_keys(f"{camp}/progress/")}
    # A resume needs a checkpoint, so only shards with a progress object can be resumed.
    todo = [sid for sid in prog_ids if sid in shards]
    print(f"{camp}: {len(shards):,} shards, {sum(map(len, files.values())):,} pick objects, "
          f"{len(todo):,} checkpointed shards to inspect ({time.time() - t0:.0f} s)", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        results = [r for r in ex.map(lambda sid: analyse_shard(camp, shards[sid], files.get(sid, [])), todo) if r]
    lost = sorted({e for r in results for e in r["lost"]})
    n_basis = collections.Counter(r["basis"] for r in results)
    print(f"resumed shards: {len(results)} ({dict(n_basis)}); lost station-days: {len(lost):,} ({time.time() - t0:.0f} s)")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cols = ["shard_id", "basis", "files", "survivors", "planned", "with_picks", "recorded", "lost"]
    rows = pd.DataFrame([dict(shard_id=r["shard_id"], basis=r["basis"], files=r["files"], survivors=r["survivors"],
                              planned=r["planned"], with_picks=r["with_picks"], recorded=r["recorded"], lost=len(r["lost"]))
                         for r in results], columns=cols)
    rows.sort_values("lost", ascending=False).to_csv(out / f"repair_shards_{camp}.csv", index=False)
    pd.DataFrame(lost, columns=["tid", "yr", "doy"]).to_csv(out / f"repair_station_days_{camp}.csv", index=False)

    queue = plan_repair(lost, a.max_sd)
    print(f"repair queue: {len(queue)} shards, {sum(s['n_station_days'] for s in queue):,} station-days, "
          f"largest {max((s['n_station_days'] for s in queue), default=0)}")
    if not queue or a.plan_only:
        return

    key = f"{repair}/shards.jsonl"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        sys.exit(f"{key} already exists; the queue is immutable. Delete the prefix to re-plan.")
    except s3.exceptions.ClientError:
        pass
    s3.put_object(Bucket=BUCKET, Key=key, Body="\n".join(json.dumps(s) for s in queue).encode() + b"\n",
                  ContentType="application/x-ndjson")
    s3.copy_object(Bucket=BUCKET, Key=f"{repair}/stations.parquet",
                   CopySource={"Bucket": BUCKET, "Key": f"{camp}/stations.parquet"})
    s3.put_object(Bucket=BUCKET, Key=f"{repair}/README.json", Body=json.dumps(dict(
        purpose="re-pick of station-days lost to the resumed-shard overwrite (docs/rerun_2026/28_resumed_shard_overwrite.md)",
        parent=camp, parquet_uri=f"s3://{BUCKET}/{camp}", planned=datetime.datetime.utcnow().isoformat() + "Z",
        resumed_shards=len(results), lost_station_days=len(lost), repair_shards=len(queue), job_definition=jobdef,
    ), indent=1).encode(), ContentType="application/json")
    print(f"wrote s3://{BUCKET}/{key} and stations.parquet")

    if a.launch:
        launch(camp, repair, jobdef, weight, min(a.launch, len(queue)), a.max_shards)


def launch(camp: str, repair: str, jobdef: str, weight: str, n: int, max_shards: int = 0) -> None:
    """Submit n workers that drain the repair queue and write into the parent."""
    batch = boto3.client("batch", region_name=REGION)
    command = ["work", "--campaign", f"s3://{BUCKET}/{repair}", "--parquet_uri", f"s3://{BUCKET}/{camp}",
               "--weight", weight, "--procs", "4", "--checkpoint-every", "0", "--flush-threshold", "250000"]
    if max_shards:
        command += ["--max-shards", str(max_shards)]
    kw = dict(jobName=f"{repair}-{datetime.datetime.utcnow():%Y%m%d%H%M}", jobQueue=QUEUE, jobDefinition=jobdef,
              parameters={"campaign": f"s3://{BUCKET}/{repair}", "weight": weight, "procs": "4",
                          "checkpoint": "0", "flush": "250000"},
              containerOverrides={"command": command})
    if n > 1:
        kw["arrayProperties"] = {"size": n}
    job = batch.submit_job(**kw)
    print(f"submitted {job['jobName']} ({job['jobId']}): {n} worker(s) on {jobdef}, picks -> s3://{BUCKET}/{camp}")


if __name__ == "__main__":
    main()
