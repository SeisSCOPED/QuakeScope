#!/usr/bin/env python
"""Run a picking campaign as a sequence of gates. docs/rerun_2026/34_campaign_playbook.md.

The 2026 western and obs catalogues needed twelve corrective queues, and every
expensive one repaired a skip that had been recorded as success. This script is
the procedure that would have caught them before a fleet ran: each stage checks
one thing, writes what it saw to `_queues/<campaign>/workflow.json`, and stops
the run when the check fails. Sampling uses fixed seeds, so the same inputs give
the same queue, sample and verdict.

    # 1. station table: FDSN epochs, codes as text, dates as dates
    campaign_workflow.py table   --source s3://.../western/stations.parquet --out table.parquet
    # 2. plan the queue from it (40 stations / 800 station-days per shard, per epoch)
    campaign_workflow.py plan    --campaign western-2027 --catalogue western --table table.parquet \\
                                 --start 1986-01-01 --end 2026-12-31
    # 3. gates before any worker: codes, coverage, image, fleet entry
    campaign_workflow.py check   --campaign western-2027 --catalogue western --table table.parquet
    # 4. yield sample in Batch, then read it
    campaign_workflow.py yield   --campaign western-2027 --catalogue western --size 3000
    campaign_workflow.py yield-report --campaign western-2027
    #    ... launch through the Fleet workflow (access survey first) ...
    # 5. close-out: availability table + FDSN residual gate, then repair queues
    campaign_workflow.py close   --campaign western-2027 --catalogue western --table table.parquet
    campaign_workflow.py repairs --campaign western-2027 --catalogue western --table table.parquet

Writes to S3 only under `_queues/` and `_archive/` and the catalogue's
`availability/`; never to `picks/`. Launching workers stays with the Fleet
workflow and its operator.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import io
import json
import random
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BUCKET = "quakescope-picks-2026"
REGION = "us-east-2"
SEED = 2026
# Every image must descend from this commit: per-day band choice, the libmseed
# selector fix, one outcome per planned station-day, and the completion rule.
REQUIRED_ANCESTOR = "d0ccf9b"
RESIDUAL_GATE = 0.02            # close when < 2% of unloaded days hold FDSN data
FDSN = {"scedc": "https://service.scedc.caltech.edu/fdsnws/dataselect/1/query",
        "ncedc": "https://service.ncedc.org/fdsnws/dataselect/1/query",
        "earthscope": "https://service.earthscope.org/fdsnws/dataselect/1/query"}


# ------------------------------------------------------------------ gates
# Pure functions: what each stage decides, testable without AWS.

def table_problems(st: pd.DataFrame) -> list[str]:
    """Why a station table is not fit to plan from; empty when it is."""
    out = []
    joined = st["network_code"].astype(str) + "." + st["station_code"].astype(str) \
        + "." + st["location_code"].astype(str)
    bad = int((st["id"] != joined).sum())
    if bad:
        out.append(f"{bad} rows whose code columns disagree with id")
    dup = int(st["id"].duplicated().sum())
    if dup:
        out.append(f"{dup} repeated ids (merge epochs into the `epochs` column)")
    for c in ("start_date", "end_date"):
        if c in st and not st[c].map(lambda v: isinstance(v, datetime.date) or pd.isna(v)).all():
            out.append(f"{c} is not a date column")
    if "epochs" not in st:
        out.append("no `epochs` column: the planner would plan the hull of the epochs")
    return out


def shard_problems(shards: list[dict], max_stations: int = 40, max_sd: int = 800) -> list[str]:
    out = []
    big = [s["shard_id"] for s in shards if len(s["stations"]) > max_stations]
    if big:
        out.append(f"{len(big)} shards over {max_stations} stations, e.g. {big[:3]}")
    heavy = [s["shard_id"] for s in shards if s.get("n_station_days", 0) > max_sd]
    if heavy:
        out.append(f"{len(heavy)} shards over {max_sd} station-days")
    mixed = [s["shard_id"] for s in shards if len({t.split('.')[0] for t in s["stations"]}) > 1]
    if mixed:
        out.append(f"{len(mixed)} shards mix networks")
    return out


def yield_verdict(outcomes: list[dict], min_samples: int = 30) -> dict:
    """Loaded fraction per network, and the networks to leave out: zero loaded
    over at least `min_samples` sampled station-days (NP 2026: 0 of 1,408)."""
    by = collections.defaultdict(collections.Counter)
    for o in outcomes:
        by[o["tid"].split(".")[0]][o["status"]] += 1
    nets = {n: dict(c) for n, c in by.items()}
    drop = sorted(n for n, c in by.items() if c["loaded"] == 0 and sum(c.values()) >= min_samples)
    total = collections.Counter()
    for c in by.values():
        total.update(c)
    n = sum(total.values())
    return {"sampled": n, "loaded_fraction": total["loaded"] / n if n else 0.0,
            "by_network": nets, "exclude": drop}


def residual_verdict(answers: list[bool], gate: float = RESIDUAL_GATE) -> dict:
    """The close-out gate: of sampled station-days we did not load, how many
    does FDSN serve? Above the gate, the catalogue is not done."""
    n = len(answers)
    hit = sum(answers)
    frac = hit / n if n else 0.0
    return {"sampled": n, "fdsn_has_data": hit, "fraction": frac, "pass": n > 0 and frac < gate}


def image_is_current(tag: str, ancestor: str = REQUIRED_ANCESTOR) -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", ancestor, tag],
                       capture_output=True)
    return r.returncode == 0


# ------------------------------------------------------------------ state

def _s3():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", region_name=REGION,
                        config=Config(retries={"max_attempts": 12, "mode": "adaptive"}))


def record(campaign: str, stage: str, result: dict) -> None:
    """Append a stage's inputs and verdict to the campaign's workflow.json."""
    s3, key = _s3(), f"_queues/{campaign}/workflow.json"
    try:
        doc = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:
        doc = {"campaign": campaign, "stages": []}
    doc["stages"].append({"stage": stage, "at": datetime.datetime.utcnow().isoformat() + "Z", **result})
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(doc, indent=1, default=str).encode())


def stop(campaign: str, stage: str, problems: list[str], **extra) -> None:
    record(campaign, stage, {"pass": False, "problems": problems, **extra})
    print(f"STOP at {stage}:")
    for p in problems:
        print(f"  - {p}")
    raise SystemExit(1)


def read_shards(campaign: str) -> list[dict]:
    body = _s3().get_object(Bucket=BUCKET, Key=f"_queues/{campaign}/shards.jsonl")["Body"].read()
    return [json.loads(x) for x in body.decode().splitlines() if x.strip()]


# ------------------------------------------------------------------ stages

def stage_table(a) -> None:
    """Epochs from FDSN, codes from id, dates as dates; then validate."""
    from sb_catalog.src.s3_state import prepare_station_dates
    from sb_catalog.src.utils import normalize_station_codes
    tmp = a.out + ".epochs.parquet"
    subprocess.run([sys.executable, str(ROOT / "scripts/add_station_epochs.py"), "--table", a.source,
                    "--out", tmp, "--weight", a.weight], check=True)
    st = prepare_station_dates(normalize_station_codes(pd.read_parquet(tmp)))
    st.to_parquet(a.out, index=False)
    problems = table_problems(st)
    print(f"{len(st):,} station-locations, {st['epochs'].notna().sum():,} with FDSN epochs")
    if problems:
        print("table problems:", *problems, sep="\n  - ")
        raise SystemExit(1)
    print(f"wrote {a.out}")


def stage_plan(a) -> None:
    from sb_catalog.src.s3_state import S3CampaignState
    from sb_catalog.src.shard_planner import plan
    st = pd.read_parquet(a.table)
    problems = table_problems(st)
    if problems:
        raise SystemExit("table not fit to plan from: " + "; ".join(problems))
    start, end = datetime.date.fromisoformat(a.start), datetime.date.fromisoformat(a.end)
    shards = plan(st, start, end)
    sp = shard_problems(shards)
    if sp:
        raise SystemExit("planner produced: " + "; ".join(sp))
    state = S3CampaignState(f"s3://{BUCKET}/_queues/{a.campaign}")
    state.write_stations(st)
    state.write_shards(shards)          # refuses an existing queue; writes plan.json
    total = sum(s["n_station_days"] for s in shards)
    record(a.campaign, "plan", {"pass": True, "catalogue": a.catalogue, "span": [a.start, a.end],
                                "shards": len(shards), "station_days": total})
    print(f"{a.campaign}: {len(shards):,} shards, {total:,} station-days")


def stage_check(a) -> None:
    from coverage_check import intervals, queues_for, uncovered
    from sb_catalog.src.shard_planner import _operating_windows
    s3, problems = _s3(), []
    st = pd.read_parquet(io.BytesIO(s3.get_object(
        Bucket=BUCKET, Key=f"_queues/{a.campaign}/stations.parquet")["Body"].read()))
    problems += table_problems(st)
    shards = read_shards(a.campaign)
    problems += shard_problems(shards)

    # Coverage: every station-day inside the table's epochs is in some queue
    # writing into this catalogue (the check that would have caught the
    # fill stations missing from the date repair).
    plan = json.loads(s3.get_object(Bucket=BUCKET, Key=f"_queues/{a.campaign}/workflow.json")["Body"].read())
    span = next(s for s in reversed(plan["stages"]) if s["stage"] == "plan")["span"]
    lo, hi = (datetime.date.fromisoformat(x) for x in span)
    hi = hi + datetime.timedelta(days=1)
    have = collections.defaultdict(list)
    for q in set(queues_for(s3, a.catalogue)) | {a.campaign}:
        try:
            for s in read_shards(q):
                s0 = datetime.datetime.strptime(s["start"], "%Y.%j").date()
                s1 = datetime.datetime.strptime(s["end"], "%Y.%j").date()
                for t in s["stations"]:
                    have[t].append((s0, s1))
        except Exception:
            continue
    gap = sum(uncovered(intervals(w, lo, hi), have.get(t, []))[0]
              for t, w in _operating_windows(st).items())
    if gap:
        problems.append(f"{gap:,} in-epoch station-days in no queue")

    # Image and fleet entry.
    fleet = json.load(open(ROOT / "fleet.json"))["campaigns"].get(a.campaign)
    if not fleet:
        problems.append("no fleet.json entry")
    else:
        import boto3
        jd = boto3.client("batch", region_name=REGION).describe_job_definitions(
            jobDefinitions=[fleet["job_definition"]])["jobDefinitions"][0]
        tag = jd["containerProperties"]["image"].rsplit(":", 1)[-1]
        if not image_is_current(tag):
            problems.append(f"image {tag} does not descend from {REQUIRED_ANCESTOR} "
                            f"(no per-station-day outcomes or band fallback)")
        if fleet.get("target"):
            problems.append("fleet target is not 0; launch only after the yield sample")
    if problems:
        stop(a.campaign, "check", problems)
    record(a.campaign, "check", {"pass": True, "coverage_gap": 0})
    print("check passed")


def stage_yield(a) -> None:
    """A stratified sample of the queue, run in Batch on the campaign's job
    definition, writing to _archive/ (never the catalogue)."""
    import boto3
    from sb_catalog.src.s3_state import S3CampaignState
    from sb_catalog.src.shard_planner import shard_id
    rng = random.Random(SEED)
    shards = read_shards(a.campaign)
    by_net = collections.defaultdict(list)
    for s in shards:
        d0 = datetime.datetime.strptime(s["start"], "%Y.%j").date()
        d1 = datetime.datetime.strptime(s["end"], "%Y.%j").date()
        by_net[s["stations"][0].split(".")[0]].append((s, d0, d1))
    sd = {n: sum(s["n_station_days"] for s, _, _ in v) for n, v in by_net.items()}
    total = sum(sd.values())
    sample = []
    for net in sorted(by_net):
        k = max(1, round(a.size / 8 * sd[net] / total))       # ~8 stations per network-day
        for s, d0, d1 in rng.sample(by_net[net], min(k, len(by_net[net]))):
            day = d0 + datetime.timedelta(days=rng.randrange((d1 - d0).days))
            tids = sorted(rng.sample(s["stations"], min(8, len(s["stations"]))))
            sample.append(dict(shard_id=shard_id(tids, day, day + datetime.timedelta(days=1)),
                               stations=tids, start=f"{day:%Y.%j}",
                               end=f"{day + datetime.timedelta(days=1):%Y.%j}",
                               n_station_days=len(tids)))
    prefix = f"_archive/yield-{a.campaign}"
    s3 = _s3()
    st = pd.read_parquet(io.BytesIO(s3.get_object(
        Bucket=BUCKET, Key=f"_queues/{a.campaign}/stations.parquet")["Body"].read()))
    state = S3CampaignState(f"s3://{BUCKET}/{prefix}")
    state.write_stations(st[st["id"].isin({t for s in sample for t in s["stations"]})])
    state.write_shards(sample)
    fleet = json.load(open(ROOT / "fleet.json"))["campaigns"][a.campaign]
    cmd = ["work", "--campaign", f"s3://{BUCKET}/{prefix}", "--parquet_uri", f"s3://{BUCKET}/{prefix}",
           "--weight", fleet["weight"], "--procs", str(fleet["procs"]), "--checkpoint-every", "0"]
    job = boto3.client("batch", region_name=REGION).submit_job(
        jobName=f"yield-{a.campaign}", jobQueue="niyiyu_earthscope_missing_station",
        jobDefinition=fleet["job_definition"], arrayProperties={"size": a.workers},
        retryStrategy={"attempts": 3}, containerOverrides={"command": cmd})
    record(a.campaign, "yield-submitted", {"pass": True, "prefix": prefix, "shards": len(sample),
                                            "station_days": sum(s["n_station_days"] for s in sample),
                                            "job": job["jobId"]})
    print(f"yield sample: {len(sample)} shards; Batch job {job['jobId']}")


def stage_yield_report(a) -> None:
    s3 = _s3()
    prefix = f"_archive/yield-{a.campaign}"
    keys = [o["Key"] for pg in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=f"{prefix}/manifests/") for o in pg.get("Contents", [])]
    outs = [o for k in keys for o in json.loads(
        s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()).get("outcomes", [])]
    v = yield_verdict(outs)
    record(a.campaign, "yield", {"pass": True, **v})
    print(f"sampled {v['sampled']:,} station-days, loaded {v['loaded_fraction']:.1%}; "
          f"leave out: {v['exclude'] or 'none'}")


def fdsn_has_data(tid: str, day: datetime.date) -> bool:
    from sb_catalog.src.constants import NETWORK_MAPPING
    net, sta, loc = tid.split(".")
    url = FDSN.get(NETWORK_MAPPING.get(net, "earthscope"), FDSN["earthscope"])
    t0 = datetime.datetime.combine(day, datetime.time(12))
    q = (f"{url}?net={net}&sta={sta}&loc={loc or '--'}&cha=??Z&starttime={t0:%Y-%m-%dT%H:%M:%S}"
         f"&endtime={t0 + datetime.timedelta(minutes=10):%Y-%m-%dT%H:%M:%S}&nodata=404")
    try:
        return urllib.request.urlopen(q, timeout=120).status == 200
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def stage_close(a) -> None:
    """Availability table, then the FDSN residual gate on days we did not load."""
    import pyarrow.dataset as ds
    out = f"s3://{BUCKET}/{a.catalogue}/availability"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_availability.py"), "--catalogue",
                    a.catalogue, "--epochs", a.table, "--out", out], check=True)
    rows = ds.dataset(out, format="parquet", partitioning="hive").to_table(
        columns=["tid", "date", "status"]).to_pandas()
    pool = rows[rows.status.isin(["unknown", "no_data", "unread"])]
    if a.exclude:
        pool = pool[~pool.tid.str.split(".").str[0].isin(a.exclude.split(","))]
    pick = pool.sample(min(a.sample, len(pool)), random_state=SEED)
    answers = [fdsn_has_data(t, d) for t, d in zip(pick.tid, pick.date)]
    v = residual_verdict(answers)
    v["by_status"] = rows.status.value_counts().to_dict()
    if not v["pass"]:
        stop(a.campaign, "close", [f"{v['fraction']:.1%} of {v['sampled']} unloaded station-days "
                                   f"have FDSN data (gate {RESIDUAL_GATE:.0%})"], **v)
    record(a.campaign, "close", v)
    print(f"close-out passed: {v['fraction']:.1%} of {v['sampled']} sampled unloaded days hold FDSN data")


def stage_repairs(a) -> None:
    """Repair queues from outcomes only: too_big at a high limit, the other
    unread reasons on the same settings. Never re-planned from the table."""
    import pyarrow.dataset as ds
    from plan_unread_repair import make_shards
    from sb_catalog.src.s3_state import S3CampaignState
    rows = ds.dataset(f"s3://{BUCKET}/{a.catalogue}/availability", format="parquet",
                      partitioning="hive").to_table(columns=["tid", "date", "status", "detail"]).to_pandas()
    rows = rows[rows.status == "unread"]
    st = pd.read_parquet(a.table)
    lo = datetime.date(1970, 1, 1)
    for name, sel in (("toobig", rows.detail == "too_big"), ("unread", rows.detail != "too_big")):
        part = rows[sel]
        if part.empty:
            continue
        runs = collections.defaultdict(list)
        for tid, g in part.groupby("tid"):
            offs = sorted((pd.Timestamp(d).date() - lo).days for d in g.date)
            start = prev = offs[0]
            for o in offs[1:] + [None]:
                if o is None or o != prev + 1:
                    k = start
                    while k <= prev:
                        edge = min((k // 20 + 1) * 20, prev + 1)
                        runs[(tid.split(".")[0], k, edge)].append(tid)
                        k = edge
                    start = o
                if o is not None:
                    prev = o
        shards, _ = make_shards(runs, lo, max_stations=4 if name == "toobig" else 40,
                                max_sd=100 if name == "toobig" else 800, min_sd=1)
        q = f"{a.campaign}-{name}"
        state = S3CampaignState(f"s3://{BUCKET}/_queues/{q}")
        state.write_stations(st[st["id"].isin(set(part.tid))])
        state.write_shards(shards)
        record(a.campaign, f"repairs-{name}", {"pass": True, "queue": q, "shards": len(shards),
                                               "station_days": len(part),
                                               "fleet_settings": {"limit_mb": 4096, "procs": 2}
                                               if name == "toobig" else {}})
        print(f"{q}: {len(shards):,} shards, {len(part):,} station-days")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("table"); t.add_argument("--source", required=True); t.add_argument("--out", required=True)
    t.add_argument("--weight", default="original")
    p = sub.add_parser("plan")
    for x in ("campaign", "catalogue", "table", "start", "end"):
        p.add_argument(f"--{x}", required=True)
    c = sub.add_parser("check")
    for x in ("campaign", "catalogue", "table"):
        c.add_argument(f"--{x}", required=True)
    y = sub.add_parser("yield"); y.add_argument("--campaign", required=True)
    y.add_argument("--catalogue", required=True); y.add_argument("--size", type=int, default=3000)
    y.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("yield-report"); r.add_argument("--campaign", required=True)
    cl = sub.add_parser("close")
    for x in ("campaign", "catalogue", "table"):
        cl.add_argument(f"--{x}", required=True)
    cl.add_argument("--sample", type=int, default=300)
    cl.add_argument("--exclude", default="", help="networks left out by decision (e.g. NP,TD,EO,LH)")
    rp = sub.add_parser("repairs")
    for x in ("campaign", "catalogue", "table"):
        rp.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    {"table": stage_table, "plan": stage_plan, "check": stage_check, "yield": stage_yield,
     "yield-report": stage_yield_report, "close": stage_close, "repairs": stage_repairs}[a.stage](a)


if __name__ == "__main__":
    main()
