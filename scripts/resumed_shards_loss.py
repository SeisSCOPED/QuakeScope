"""For each resumed shard found by resumed_shards_scan.py: which station-day-channels
checkpointed by the first attempt have no picks in ANY surviving Parquet object of
the shard? Those were overwritten (upper bound: a station-day the first attempt read
and found nothing on looks the same). See docs/rerun_2026/28_resumed_shard_overwrite.md.

    python scripts/resumed_shards_loss.py western --out docs/rerun_2026/resumed_shards
"""
import argparse, boto3, json, sys, io, datetime, csv, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, pyarrow.parquet as pq
ap = argparse.ArgumentParser()
ap.add_argument("campaign"); ap.add_argument("--out", default="docs/rerun_2026/resumed_shards")
a = ap.parse_args(); camp = a.campaign
B = "quakescope-picks-2026"
s3 = boto3.client("s3", region_name="us-east-2")
scan = pd.read_csv(f"{a.out}/scan_{camp}.csv")
resumed = scan[scan.complete & (scan.prog_worker != scan.comp_worker)]
def get(key):
    return json.loads(s3.get_object(Bucket=B, Key=key)["Body"].read())
def yd(s):
    y, d = s.split("."); return datetime.date(int(y), 1, 1) + datetime.timedelta(days=int(d) - 1)

def one(sid):
    prog, man = get(f"{camp}/progress/{sid}.json"), get(f"{camp}/manifests/{sid}.json")
    man_recs = set((r["tid"], r["yr"], r["doy"], r["cha"]) for r in man["records"])
    missing = [tuple(d) for d in prog["done"] if tuple(d) not in man_recs]
    if not missing:
        return dict(shard_id=sid, missing=0, lost=0, survivors=0, manifest_files=len(man["files"]))
    nets = sorted(set(m[0].split(".")[0] for m in missing))
    y0, y1 = sid.split("-")[0], sid.split("-")[1]
    d0, d1 = yd(f"{y0[:4]}.{y0[4:]}"), yd(f"{y1[:4]}.{y1[4:]}")
    months = set()
    d = d0
    while d < d1:
        months.add((d.year, d.month)); d += datetime.timedelta(days=1)
    man_paths = set(f["path"] for f in man["files"])
    have = set()
    n_surv = 0
    for net in nets:
        for (y, m) in months:
            pfx = f"{camp}/picks/network={net}/year={y}/month={m:02d}/{sid}"
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=pfx):
                for o in page.get("Contents", []):
                    if f"s3://{B}/{o['Key']}" in man_paths:
                        continue
                    n_surv += 1
                    body = s3.get_object(Bucket=B, Key=o["Key"])["Body"].read()
                    t = pq.read_table(io.BytesIO(body), columns=["tid", "cha", "peak"]).to_pandas()
                    t["yr"] = t.peak.dt.year; t["doy"] = t.peak.dt.dayofyear
                    have |= set(map(tuple, t[["tid", "yr", "doy", "cha"]].drop_duplicates().values.tolist()))
    lost = [m for m in missing if m not in have]
    return dict(shard_id=sid, missing=len(missing), lost=len(lost), survivors=n_surv, manifest_files=len(man["files"]),
                lost_days=sorted(set((m[1], m[2]) for m in lost))[:3])

t = time.time()
with ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(one, resumed.shard_id.tolist()))
df = pd.DataFrame(rows)
df.to_csv(f"{a.out}/loss_{camp}.csv", index=False)
print(f"{camp}: {len(df)} resumed shards in {time.time()-t:.0f}s; checkpointed-but-not-in-manifest {df.missing.sum():,}; "
      f"of which with no picks in any surviving object (lost, upper bound) {df.lost.sum():,}; "
      f"shards with any loss {(df.lost>0).sum()}; max per shard {df.lost.max()}")
print(df.sort_values("lost", ascending=False).head(8).to_string())
