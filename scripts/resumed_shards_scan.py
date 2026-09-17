"""For every checkpointed shard of a campaign: was it finished by a different
worker than the one that wrote the last checkpoint (resumed after a preemption
or a kill), and how many station-day-channels does the manifest carry against
the checkpoint? Needs read access to <campaign>/progress/ and complete/, which
are not public. See docs/rerun_2026/28_resumed_shard_overwrite.md.

    python scripts/resumed_shards_scan.py western --out docs/rerun_2026/resumed_shards
"""
import argparse, boto3, json, sys, time, csv
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("campaign"); ap.add_argument("--out", default="docs/rerun_2026/resumed_shards")
ap.add_argument("--workers", type=int, default=48)
a = ap.parse_args(); camp = a.campaign
B = "quakescope-picks-2026"
s3 = boto3.client("s3", region_name="us-east-2")
keys = []
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=f"{camp}/progress/"):
    keys += [o["Key"].rsplit("/", 1)[1][:-5] for o in page.get("Contents", [])]
print(camp, len(keys), "checkpointed shards", flush=True)

def get(key):
    try:
        return json.loads(s3.get_object(Bucket=B, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None

def one(sid):
    prog = get(f"{camp}/progress/{sid}.json")
    comp = get(f"{camp}/complete/{sid}.json")
    man = get(f"{camp}/manifests/{sid}.json")
    prog_done = set(tuple(d) for d in prog["done"]) if prog else set()
    man_recs = set((r["tid"], r["yr"], r["doy"], r["cha"]) for r in man["records"]) if man else set()
    return dict(shard_id=sid, prog_worker=prog["worker"] if prog else None, prog_n=len(prog_done),
                comp_worker=comp["worker"] if comp else None, complete=bool(comp),
                man_n=len(man_recs), man_files=len(man["files"]) if man else None,
                in_prog_not_man=len(prog_done - man_recs), in_man_not_prog=len(man_recs - prog_done),
                station_days=comp["station_days"] if comp else None, picks_record=comp["picks_record"] if comp else None)

t = time.time()
with ThreadPoolExecutor(a.workers) as ex:
    rows = list(ex.map(one, keys))
print(f"fetched in {time.time()-t:.0f}s", flush=True)
import os; os.makedirs(a.out, exist_ok=True); out = f"{a.out}/scan_{camp}.csv"
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
resumed = [r for r in rows if r["complete"] and r["prog_worker"] != r["comp_worker"]]
under = [r for r in rows if r["complete"] and r["in_prog_not_man"] > 0]
print(f"{camp}: complete {sum(r['complete'] for r in rows)}, finished by a different worker than checkpointed: {len(resumed)}, "
      f"manifest missing records the checkpoint had: {len(under)} shards, {sum(r['in_prog_not_man'] for r in under)} station-day-channels")
