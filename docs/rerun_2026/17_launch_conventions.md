# 17 — Launch conventions: buckets, names, roles, job definitions

What is provisioned, what everything is called, and why. Fixed before the first
campaign so that five campaigns produce one coherent catalogue rather than five
that need reconciling afterwards.

## Bucket

**`s3://quakescope-picks-2026`**, region **us-east-2**.

Same region as the Fargate Spot quota, so writes cost nothing. Reads are
cross-region for `scedc-pds` (us-west-2) and `ncedc-pds` (us-east-1), which the
head-to-head measured at about **one second per station-day** — the workload is
CPU-bound, so locality is not worth splitting the output over.

- **Public access:** blocked, all four settings.
- **Versioning: off, deliberately.** Objects are immutable and named after the
  shard that wrote them, so a retried shard overwrites itself byte for byte.
  Versioning would keep every superseded retry and silently double a 3.1 TB
  catalogue.

## Campaign prefixes

One prefix per campaign. Campaigns never share a prefix, because the queue is
keyed on shard id and two campaigns would collide.

```
s3://quakescope-picks-2026/
    scedc/          campaign 1
    ncedc/          campaign 2
    earthscope/     campaign 3
    obs/            campaign 4
    western/        campaign 5  (the stakeholder deliverable)
```

## Layout inside a campaign

```
<campaign>/
    stations.parquet              station metadata
    shards.jsonl                  the work queue, immutable once written
    claims/<shard_id>.json        who holds what
    progress/<shard_id>.json      mid-shard checkpoints
    complete/<shard_id>.json      finished shards
    manifests/<shard_id>.json     what each job wrote, including object keys
    runs/<run_id>.json            model, weight, thresholds
    picks/network=<NET>/year=<YYYY>/month=<MM>/<shard_id>.parquet
```

### Shard id, and why files are named after it

```
<start:%Y%j>-<end:%Y%j>-<sha1(sorted stations | start | end)[:12]>
e.g.  2019187-2019207-8be9ab508637
```

Content-derived, so re-planning an identical campaign reproduces the same ids
and recognises completed work. Naming the Parquet object after it makes a retry
**idempotent**: the same shard always writes the same key.

This is not cosmetic. Before it, the writer fell back to `HOSTNAME`, so every
shard a node ran wrote the *same* key inside a `(network, year, month)`
partition and silently overwrote the last. Under Batch that was safe only by
accident, because the Batch job id happened to be unique.

A partition that exceeds `flush_threshold` emits `-001`, `-002` suffixes. A
reader cannot reconstruct those, which is why **the manifest records the keys**
and is the index for LIST-free reads.

### Pick schema

| column | type | meaning |
|---|---|---|
| `tid` | string | `NET.STA.LOC` |
| `cha` | string | band code, e.g. `HH` |
| `pha` | string | `P` or `S` |
| `start`, `peak`, `end` | timestamp(ms) | pick window |
| `conf` | float32 | model confidence |
| `amp` | float32 | Wood-Anderson displacement, metres. NaN where the window fell inside a taper |
| `amp_raw` | float32 | high-passed counts |
| `rid` | string | run id → `runs/<rid>.json` |

`network`, `year`, `month` are **partition keys**, not columns in the file.

## IAM

**`SeisBenchBatchRole`** serves as both the Batch job role and the Fargate
execution role. It trusts `ecs-tasks.amazonaws.com` and carries
`AmazonECSTaskExecutionRolePolicy` (pull the image, write logs) and
`AmazonS3FullAccess`. Verified by policy simulation against the new bucket.

> `AmazonS3FullAccess` is wider than a campaign needs. A scoped policy limited
> to `quakescope-picks-2026/*` plus read on the public archives would be
> better; it is not a launch blocker but is worth tightening.

## Compute

| | |
|---|---|
| Region | us-east-2 |
| Compute environment | `niyiyu_earthscope` (FARGATE_SPOT, maxvCpus 4000) |
| Job queue | `niyiyu_earthscope_missing_station` |
| Fargate Spot quota | 12,000 vCPU (`L-36FBB829`) |
| Job definition | `quakescope_v3_worker` |
| Image | pin the **short-SHA tag**, never `:latest` |

**Pin the tag.** `:latest` moves when someone pushes to `main`, and a task that
starts tomorrow would run different code from one that started today.

A job is a **worker, not a shard**: it claims from the queue until the queue is
empty, so you choose how many workers to run rather than submitting one job per
unit of work. `maxvCpus` on the compute environment (4000) binds before the
account quota (12,000), and is the number to raise first.

## Weight per campaign

| campaign | weight | why |
|---|---|---|
| scedc, ncedc, earthscope | `jma_wc` | best recall/MCC of the fine-tunes |
| obs | `obs` | `PickBlue(base="phasenet")` |
| **western** | **`original`** | see below |

**Western states uses `original`, superseding
[09_western_states_run.md](09_western_states_run.md), which specified
`instance`.** `instance` has a genuine ceiling on dense near-field aftershock
sequences: at Ridgecrest it emits 246 S picks with its threshold on the floor
where the others reach 684 and 832, and no threshold recovers it. Since this is
a stakeholder deliverable, **confirm the substitution with the stakeholder
before launching** — it changes what they receive.

## Sizing

Measured from station operating windows, not stations × span:

| campaign | station-days (2010–2026) |
|---|--:|
| scedc | 2,467,740 |
| ncedc | 4,551,557 |
| earthscope onshore | 44,127,796 |
| earthscope offshore | 950,793 |
| **western** | **33,776,383** |

The western set is **24,111 stations across 122 networks**, selected by true
state polygons rather than a bounding box — a single box over the six states
sweeps in about 3,000 stations from AZ, UT, MT and CO, which
[09](09_western_states_run.md) explicitly warns against. Nevada is the tell: a
naive per-state rectangle scheme assigns it 3 stations, because California's
rectangle covers it.

At the measured ~34 s per band-day on Fargate, western alone is roughly
**319,000 vCPU-hours ≈ $4,700** at the estimated Spot rate — with the caveats in
[16_skypilot_vs_fargate.md](16_skypilot_vs_fargate.md): processes per vCPU is
unmeasured and swings this about 4×.

**Western is the largest single campaign**, 65% of the launch total, and it
overlaps campaigns 1–3 geographically — it re-picks those stations with
different weights, so it is additional work rather than a subset.
