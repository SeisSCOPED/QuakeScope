# 16 — Why Fargate, not SkyPilot (decision record)

> **Decided: Fargate Spot.** SkyPilot is no longer used and its config and
> runbook have been deleted. This document is kept because it is the evidence
> for that decision, not because either option is still open.
>
> The short version: Fargate Spot already holds a 12,000 vCPU quota in us-east-2
> where SkyPilot's EC2 Spot quota there is 256; Fargate tasks end with the job,
> whereas SkyPilot's jobs controller survives `sky down --all` and keeps
> relaunching workers; and on an identical shard Fargate measured 33.7 s against
> EC2's 53.7 s (n=1, so weak evidence on its own, but it points the same way as
> the other two).

A cost and bottleneck comparison of the two ways to run a picking campaign, plus
why the 2025 DocumentDB output became Parquet on S3 and how to read either.

Measured on real shards, not modelled, unless a row says otherwise.

---

## 1. Where the time actually goes

One station-day, `CI.CLC` and `CI.TOW2`, 2019 DOY 187–188, 8 vCPU Spot instance,
one worker process, `worker --profile`:

| stage | mainshock day | ordinary day | scales with |
|---|--:|--:|---|
| **amp.wood_anderson** | 25.5 s (47%) | 25.5 s (47%) | picks |
| **model.classify** | 24.0 s (45%) | 22.2 s (38%) | samples |
| amp.raw | 2.3 s (4%) | 2.3 s (4%) | picks |
| s3.get | 1.2 s (2%) | 1.7 s (3%) | bytes |
| mseed.parse | 0.3 s (0.5%) | 0.3 s (0.4%) | bytes |
| s3.list | ~0 | ~0 | objects in prefix |
| **parquet.encode + put** | **0.05 s (0.1%)** | **0.05 s (0.1%)** | rows |
| s3.head | ~0 | ~0 | objects |
| *unaccounted* | 0.4 s (0.7%) | | |

Measured before the numpy work on the per-pick amplitude path; the
head-to-head in section 2 supersedes the absolute numbers, though the
proportions hold.

Three conclusions, and they drive everything below.

**The workload is CPU-bound, not I/O-bound.** Every S3 and parsing stage
together is under 4%, at 31–47 MB/s. This is the single most useful result: it
means **cross-region reads are not a cost problem**, so compute does not have to
sit in the same region as the data. That frees the campaign to run in us-east-2,
where the Fargate Spot quota already is.

**Writing Parquet is free — 0.1% of runtime.** Whatever else is true about the
storage change, it did not cost throughput.

**Amplitude extraction is the largest stage, larger than inference.** It is also
the thing added since 2025. This is where the remaining money is, and §5 covers
what has been done and what has not.

---

## 2. Platform comparison

Matched as closely as the two platforms allow: same container image, same code,
8 vCPU / 16 GB, one worker process, same shard, same S3 output.

| | SkyPilot (EC2 Spot) | AWS Batch (Fargate Spot) |
|---|---|---|
| Cold start, measured | ~2–4 min (instance provision) | **61 s** median, 59–72 s (n=3) |
| Spot quota, us-east-2 | **256 vCPU** (`L-34B43A08`) | **12,000 vCPU** (`L-36FBB829`) |
| Spot quota, us-west-2 | 656 vCPU | — |
| Price, measured | c7g.16xlarge **$0.0133/vCPU-hr** | — |
| Price, list | m6i.xlarge $0.0224/vCPU-hr | on-demand **$0.04937/vCPU-hr** (8 vCPU/16 GB) |
| Price, Spot estimate | measured above | ~**$0.0148/vCPU-hr** at the usual ~70% discount |
| Preemption recovery | jobs controller relaunches | Batch `retryStrategy` re-queues |
| Idle cost | **jobs controller is on-demand and persists** | none between jobs |
| Instance choice | any type, incl. Graviton | vCPU/memory only |

**Price is close to a wash.** EC2 Spot on Graviton is measurably $0.0133/vCPU-hr;
Fargate Spot is *estimated* at $0.0148 because AWS does not publish the Spot rate
through the pricing API, and Cost Explorer is blocked on this account by an
organisation SCP (`p-q1ngvul9`) so the 2025 actuals could not be recovered
either. **Treat the Fargate figure as unverified.** Reaching the cheaper EC2
number also requires an arm64 image, which does not exist yet.

**Quota is not a wash.** 12,000 vs 256 vCPU in us-east-2 is a 47× difference in
achievable parallelism, and it is already granted. At 256 vCPU a full campaign
takes months; the EC2 route needs a quota increase before it is viable there at
all.

**Two operational differences that matter more than the price:**

- The SkyPilot **jobs controller is on-demand and does not stop itself**. It
  survives `sky down --all`, which reports success and leaves it running — see
  [15_monitoring.md](15_monitoring.md). On a Spot campaign it is the most
  expensive thing to forget.
- Fargate has **no instance to leak**: when a job ends the task is gone.

**Recommendation, pending the outstanding measurement:** Fargate Spot in
us-east-2, on the quota you already hold, unless the EC2 quota increase lands
*and* an arm64 image makes Graviton reachable. The deciding factor is not price
per vCPU-hour but whether 12,000 vCPU of headroom is worth more than a ~10%
hourly saving that depends on two prerequisites.

### Head-to-head: the same shard on both platforms

Run after the v3 image was rebuilt, which is what previously made this
impossible — the published image crashed on `EARTHSCOPE_S3_ACCESS_POINT` at
import and had no `pyarrow`, so Fargate probes exited 1 within seconds.

Shard `2019188-2019189-0abee934e908`, 6,697 picks, 8 vCPU, one worker process,
identical image (`ghcr.io/seisscoped/quakescope:32c8321`) and identical code:

| stage | EC2 Spot (us-west-2) | Fargate Spot (us-east-2) | ratio |
|---|--:|--:|--:|
| amp.wood_anderson | 25.45 s | 20.48 s | 0.80x |
| **model.classify** | 24.05 s | **7.23 s** | **0.30x** |
| amp.raw | 2.35 s | 2.00 s | 0.85x |
| s3.get | 1.17 s | 2.26 s | 1.93x |
| mseed.parse | 0.27 s | 0.27 s | 1.00x |
| parquet encode + put | 0.05 s | 0.36 s | 7.2x |
| **wall clock** | **53.71 s** | **33.72 s** | **0.63x** |

**Fargate ran the identical shard 37% faster**, almost entirely in inference:
7.2 s against 24.1 s. Same vCPU count, same image, same code, so the likeliest
explanation is a newer CPU generation behind Fargate than the `c6i.2xlarge`
SkyPilot selected — but this is **one sample on each side and should be
confirmed before it is relied on.** It matters because it inverts the price
comparison: at $0.0148/vCPU-hr and 0.63x the runtime, Fargate is *cheaper per
station-day* than EC2 Spot at $0.0133, not merely more available.

**Cross-region reads cost about one second per station-day.** S3 throughput
halved — 46.6 MB/s reading `scedc-pds` from the same region against 24.1 MB/s
from us-east-2 — but that is +1.1 s of a 33.7 s shard. This settles the
locality question: the workload is CPU-bound, so running where the Fargate
quota is costs roughly 3%, not a redesign.

Output verified rather than assumed: 29,906 picks over four shards,
`remaining: 0`, amplitudes populated on 29,852 rows. Fargate cold start was
66 s, consistent with the 61 s median measured earlier.

### Conclusion

**Fargate Spot, in us-east-2.** It wins on every axis measured:

| | Fargate Spot | EC2 Spot via SkyPilot |
|---|---|---|
| Spot quota, us-east-2 | **12,000 vCPU**, already granted | 256 vCPU |
| Cold start | **66 s** | ~3 min |
| Wall clock, same shard | **33.7 s** | 53.7 s |
| Idle cost | none — the task ends | jobs controller persists, on-demand |
| Proven at scale | the 2025 petabyte campaign | two shards |

The v3 worker runs there unmodified, through `python -m src.picker work`.

**What would change this:** an EC2 Spot quota increase in us-east-2 plus an
arm64 image, which together would make Graviton reachable at $0.0133/vCPU-hr.
Worth revisiting only if the inference gap above turns out to be measurement
noise.

### Still unmeasured

- **Repeat runs.** A 3.3x inference difference from n=1 on each side is too
  load-bearing to accept as-is.
- **Processes per vCPU.** Every benchmark so far used one process on an 8 vCPU
  box, so most of the machine was idle. This swings campaign cost about 4x and
  is an hour's work to settle.
- **The 2025 baseline.** Cost Explorer is blocked on this account by an
  organisation SCP, so "match or beat 2025" needs a billing-console export.

---

## 3. Scalability

Per-band-day cost is **~54 s** on 8 vCPU with one process, after the amplitude
work in §5. The campaign is 52.1M station-days
([12_output_storage.md](archive/12_output_storage.md) §1, operating-window aware).

Two numbers are still unmeasured and together swing the total by ~4×:

- **processes per vCPU** — every measurement so far used one process on an
  8 vCPU box, so most of the machine was idle. `worker --procs` exists; the
  sweep has not been run.
- **bands per station** — the picker currently processes every band a station
  has, measured at **2.83×** on SCEDC, including `LH` at 1 Hz which cannot be
  picked at all. One-band-per-station is agreed but not implemented.

Until both are measured, any total campaign cost is a guess with a 4× error bar,
and this document deliberately does not state one.

**Wall-clock is quota-bound, not throughput-bound.** At 12,000 vCPU the arithmetic
is comfortable; at 256 it is not.

---

## 4. Why DocumentDB became Parquet on S3

| | DocumentDB (2025) | Parquet on S3 (2026) |
|---|--:|--:|
| Bytes per pick | 148 (BSON), ~228 with indexes | **35** |
| At 88B picks | 13.1 TB, ~20.2 TB with indexes | **3.1 TB** |
| Storage cost | ~$465/month + cluster instance-hours | **~$71/month**, nothing running |
| Write cost in the pipeline | per-station-day `insert_many` | **0.1% of runtime** (measured) |
| Between campaigns | cluster must stay up | nothing to run |
| Requires a VPC | yes — submission had to run inside it | **no** |

Measurements in [12_output_storage.md](archive/12_output_storage.md) §2.

The change is a win on every axis measured: **6.5× less storage**, the cluster
and its VPC constraint disappear, and the write path costs 0.1% of a station-day.
Dictionary encoding is why — `tid`, `cha`, `pha` and `rid` repeat down a column
and compress to almost nothing, while BSON repeats every field *name* in every
document.

**What did not move.** Station metadata and resume state are small, need point
lookups, and are what a database is good at. In v3 they became S3 objects too
(`stations.parquet`, `complete/`, `manifests/`) — see the module docstring in
[`sb_catalog/src/s3_state.py`](../../sb_catalog/src/s3_state.py) — but that is a
consequence of dropping the database entirely, not of the Parquet decision.

**The one cost.** Parquet buffers a partition in memory until
`flush_threshold` (4M rows, ~800 MB resident per partition per process). That is
a RAM floor the streaming database writer did not have, and it will not fit
alongside one process per vCPU on a memory-light instance. Lower it if the
process sweep shows memory pressure.

---

## 5. What has been optimised, and what remains

**Done, and verified:**

- **Deconvolution hoisted** out of the per-pick loop — was one
  `remove_response` per pick, ~6,700 per busy station-day. Now once per station.
  5.3× faster on that stage, and *more correct*: the old 33 s window was
  ill-conditioned, returning 0.170 where every longer window converges on
  0.00045. Roughly 5% of picks were affected.
- **Per-pick path moved to numpy** — obspy `Stream.slice()` per pick over an
  8.6M-sample trace, plus a `select()` scan per trace per pick. `amp.raw` went
  5.7 s → 2.3 s.
- **joblib removed** — `Parallel(n_jobs=-1)` was nested *inside* each worker
  process, oversubscribing every core.

**Remaining, in order of expected value:**

1. **`amp.wood_anderson` is still 47% of runtime**, and the per-pick loop is no
   longer the cost — the deconvolution of a day-long trace is. This is the
   biggest single lever left.
2. **Process sweep** (`--procs` × `OMP_NUM_THREADS`) — up to 4×, unmeasured.
3. **One band per station** — 2.83× on the California campaigns.
4. **arm64 image** — 1.68× on price, and the only route to the cheapest EC2 tier.
5. **Restore the operating-window filter** the v3 planner lost; v2 applied
   `filter_station_by_start_end_date` (`utils.py:183`).

---

## 6. Reading the output

Two notebooks do all of this interactively:
[`notebooks/5_submit_job_parquet.ipynb`](../../notebooks/5_submit_job_parquet.ipynb)
launches a campaign on Fargate with no DocumentDB, and
[`notebooks/6_check_parquet.ipynb`](../../notebooks/6_check_parquet.ipynb)
queries the result. They are the Parquet counterparts of notebooks 3 and 4,
which remain for the 2025 database path.

### 6a. The 2025 catalog, in DocumentDB

```python
import pymongo, datetime

client = pymongo.MongoClient(
    DOCDB_ENDPOINT_URI,                      # from sb_catalog/src/parameters.py
    tls=True, tlsCAFile="global-bundle.pem", # AWS RDS trust store
    retryWrites=False,                       # DocumentDB does not support it
)
db = client["quakescope_2025"]

# Picks for one station in a time range. Indexed on (peak, tid).
rows = db["picks"].find({
    "tid": "CI.CLC.",
    "peak": {"$gte": datetime.datetime(2019, 7, 6),
             "$lt":  datetime.datetime(2019, 7, 7)},
})
```

Only reachable from **inside the DocumentDB VPC** — that constraint is a large
part of why v3 left. Collections: `picks`, `classifies`, `picks_record`,
`stations`, `sb_runs`.

### 6b. The 2026 catalog, in Parquet

Layout:

```
s3://<bucket>/<campaign>/
    picks/network=CI/year=2019/month=07/<shard_id>.parquet
    manifests/<shard_id>.json      what each job wrote, including object keys
    complete/<shard_id>.json       queue state
    shards.jsonl                   the immutable work queue
    stations.parquet               station metadata
```

**Whole-dataset query, letting the engine prune partitions.** Simplest, and
correct for analysis:

```python
import duckdb
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("""
    SELECT tid, pha, peak, conf, amp
    FROM read_parquet('s3://bucket/campaign/picks/**/*.parquet',
                      hive_partitioning = true)
    WHERE network = 'CI' AND year = 2019 AND month = 7
      AND tid = 'CI.CLC.'
      AND conf >= 0.5
    ORDER BY peak
""").df()
```

`network`, `year` and `month` are **partition keys**, so those predicates skip
whole prefixes rather than reading them. `tid` and `conf` are pushed to Parquet
row-group statistics. Same in pyarrow:

```python
import pyarrow.dataset as ds
dataset = ds.dataset("s3://bucket/campaign/picks/", format="parquet",
                     partitioning="hive")
table = dataset.to_table(
    filter=(ds.field("network") == "CI") & (ds.field("year") == 2019)
           & (ds.field("tid") == "CI.CLC."),
    columns=["tid", "pha", "peak", "conf", "amp"],
)
```

### 6c. Reading without a single LIST

Partition pruning still **lists** the prefixes it keeps. To avoid listing
entirely, use the manifests as the index — they record the exact object keys a
job wrote:

```python
import json, s3fs, pyarrow.parquet as pq

fs = s3fs.S3FileSystem()
root = "bucket/campaign"

# 1. One GET: the work queue tells you which shard covers which stations/days.
shards = [json.loads(l) for l in
          fs.cat(f"{root}/shards.jsonl").decode().splitlines() if l.strip()]

want, day = "CI.CLC.", "2019.187"
hits = [s for s in shards
        if want in s["stations"] and s["start"] <= day < s["end"]]

# 2. One GET per shard: the manifest names the objects that shard produced.
for s in hits:
    manifest = json.loads(fs.cat(f"{root}/manifests/{s['shard_id']}.json"))
    for f in manifest["files"]:
        if f["kind"] != "picks":
            continue
        # 3. One GET per file. No LIST anywhere in this path.
        table = pq.read_table(f["path"], filesystem=fs,
                              columns=["tid", "pha", "peak", "conf", "amp"])
        print(f["path"], table.num_rows)
```

**Why this works.** `shard_id` is content-derived —
`{start:%Y%j}-{end:%Y%j}-{sha1(stations|start|end)[:12]}` — so it is stable
across re-planning, and the manifest records every key including the `-001`
sequence suffixes a large partition produces. Neither is reconstructible from
the shard definition alone, which is why the manifest is the index rather than a
naming convention.

**Which to use.** 6b for analysis, where the engine's pruning is faster to write
and fast enough. 6c for a service or a tight loop, where LIST latency and request
cost matter and the access pattern is known in advance.

### 6d. Provenance

Every pick carries `rid`, the run id, which resolves through
`runs/<run_id>.json` to the model, weight and thresholds that produced it. That
is how picks from different weights stay separable inside one campaign — the
same job `sb_runs` did in 2025.
