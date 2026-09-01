# What is left to optimise

Open work on the 2026 picking workflow, ordered by expected value. Current
verified state is in [README.md](README.md); this file is deliberately only the
things that are **not** settled.

Each item says what is measured and what is assumed. The distinction matters —
the last planning round produced a confident cost model on top of a number
nobody had measured, and all of it had to be retracted
([`archive/README.md`](archive/README.md)).

---

## 1. `--procs` scaling — the largest unknown

**Status: being measured now.** Jobs `proctest-p{1,2,4,8}`, job definition
`quakescope_v3_worker:4`, image `dd4fcbc`, submitted 2026-09-01 05:20 UTC. Ids
in [`proctest.json`](../../proctest.json); they run on AWS and need no laptop
attached.

### The first attempt was not a valid experiment

Recorded because the flaw is easy to repeat. The first sweep gave every job
`--max-shards 1`, so a job with `--procs P` claimed P shards from the live queue
and its wall clock was **the maximum of P random draws** — while `--procs 1`
drew once. Even under perfect scaling `procs=8` would finish later, purely
because the max of eight draws exceeds one draw. "Flat wall clock means perfect
scaling" was wrong.

The draws are wildly unequal, which is what made this fatal rather than
cosmetic. Two shards observed in that run:

| shard | wall | station-day-channels |
|---|--:|--:|
| `2018219-2018239-9c91d08e08b6` | **23 s** | **0** — no data at all |
| `2012031-2012051-52c578f41b6a` | 1145 s | 60 |

A 50x spread, and some shards are entirely empty. Any design that hands
different shards to different arms measures shard luck, not parallelism.

### The design that replaces it

Four campaign prefixes under `s3://quakescope-picks-2026/_proctest/p{1,2,4,8}`,
each holding an **identical** `shards.jsonl` of the same 8 shards, with
`--max-shards` set so `procs x max_shards = 8`:

| arm | `--procs` | `--max-shards` | shards |
|---|--:|--:|--:|
| p1 | 1 | 8 | 8, sequential |
| p2 | 2 | 4 | 8 |
| p4 | 4 | 2 | 8 |
| p8 | 8 | 1 | 8, all parallel |

Every arm completes the same work, so wall clock is directly comparable and the
speedup is `p1_wall / pN_wall`.

The 8 shards are deliberately uniform — all 620 planned station-days, the same
31 stations, consecutive 20-day windows through 2015 — so no single straggler
sets `p8`'s wall clock while `p1` pays the sum. That skew, not parallelism, is
what the first attempt actually measured.

Picks land in the test prefixes, not in `scedc`, so the real catalogue stays
clean. The work is duplicated four times; that is the price of a controlled
experiment and comes to well under a dollar.

### Reading it

```bash
python3 -c "
import boto3, json
b = boto3.client('batch', region_name='us-east-2')
for p, jid in sorted(json.load(open('proctest.json'))['jobs'].items(),
                     key=lambda x: int(x[0])):
    d = b.describe_jobs(jobs=[jid])['jobs'][0]
    el = (d['stoppedAt'] - d['startedAt']) / 1000 if d.get('stoppedAt') else None
    print(f\"procs={p}: {d['status']:10} {el and f'{el:.0f}s' or '-'}\")"
```

**You do not have to wait for `p1`.** It runs 8 shards sequentially and will take
hours; `p8` runs them in parallel and should finish in roughly one shard's time.
Comparing completed-shards-per-elapsed-minute answers the question long before
`p1` ends, and `p1` can be killed once the rate is clear.

Also check `p8` for an OOM or memory-pressure stall — 16 GB across 8 processes
is 2 GB each, and the Parquet writer buffers a partition per process.

Record the outcome here and update the cost figure in [README.md](README.md).

Every measurement to date used `--procs 1` on an 8 vCPU task. If the other seven
cores are idle, the campaign costs up to 8× what it needs to; if torch is
already using them, there is nothing to win. **Nobody has looked.** This single
number moves the ~$16,400 estimate more than everything else on this list
combined.

Method: `--procs P --max-shards 1` runs P worker loops, each taking one shard, so
the job completes P shards on the same 8 vCPU. Perfect scaling shows as flat wall
clock across P; the useful metric is station-day-channels per vCPU-hour.

Watch for memory. The Parquet writer buffers a partition until `flush_threshold`
and 16 GB across 8 processes is 2 GB each, so `--procs 8` is where an OOM would
appear if there is one.

## 2. EarthScope I/O — 60% of the campaign, unprofiled

EarthScope stores **one multi-channel object per station-day**, downloaded and
parsed whole; a UW sample held 214 traces across 38 channel codes of which the
picker uses three. SCEDC and NCEDC store one object per channel, so a station-day
fetches only the band it needs.

Every timing in [README.md](README.md) is SCEDC. Campaigns 3–5 are **91% of the
station-days and ~$14,800 of the ~$16,400 estimate**, extrapolated on the
assumption that EarthScope reads at SCEDC speed. That assumption has never been
tested and [19_earthscope_access.md](19_earthscope_access.md) records the
standing suspicion that it is badly wrong — an earlier test sat on
`Load ZI.CAMP.10` for 25 minutes.

That 25-minute stall is now known to have been the stale image, not EarthScope.
It is no longer evidence of anything, which means the question is **fully open
again**, not resolved.

**Do this before launching campaign 3.** Run `--profile` on five EarthScope
shards on `quakescope_v3_worker:4` and compare `s3.get` seconds and MB against
the SCEDC baseline. Two hours; it either confirms the estimate or changes the
campaign plan.

## 3. `amp.wood_anderson` is 58% of runtime

1,179 s of a 2,038 s shard, at 10.262 ms/pick over 114,939 picks. The largest
single stage, larger than inference, and the biggest lever left in the picker
itself.

Already done, and not to be redone: the deconvolution was hoisted out of the
per-pick loop (5.3× on that stage, and *more correct* — the old 33 s window was
ill-conditioned), the per-pick path moved to numpy, and a nested
`joblib.Parallel` that oversubscribed every core was removed. What remains is the
deconvolution of a day-long trace, not the per-pick work.

A 2× here is ~$4,000 of the campaign.

## 4. `s3.get` ran at 8.8 MB/s

Against 46.6 MB/s same-region and 24.1 MB/s cross-region measured in
[16_skypilot_vs_fargate.md](16_skypilot_vs_fargate.md) §2. That is 3–5× slower
than expected and costs 502 s of a 2,038 s shard.

Unexplained. Candidates: `--procs 1` leaving no concurrency to hide latency,
2010 objects being small and numerous, or per-request overhead dominating at
14.7 MB per call. Item 1 may resolve it for free — check this again from the
sweep's profiles before investigating separately.

## 5. Parquet fragmentation — and the compaction code is unverified

A 1,500-worker run produced **29,862 objects at a 49 KB median** for 2.8 GB,
because `flush_threshold` spread across 1,456 partitions and 1,500 writers means
no partition ever accumulates. Writes are unaffected; reads are not — counting
rows opens ~30k footers, which is what stalled the dashboard.

`sb_catalog/src/parquet_compact.py` and `scripts/monitor_and_compact.py` were
written for this and **have never been run against a real campaign prefix**.
They are a sketch, not a tool. Treat the numbers in
[22_parquet_compaction.md](22_parquet_compaction.md) as intentions.

Before relying on either: run against one partition with `--dryrun`, then for
real, and verify pick counts before and after. Compaction deletes its inputs and
the bucket has versioning **off**, so a bug there loses picks. That is the reason
this is not already done.

## 6. Two correctness questions worth closing before their campaigns

**`obs` declares `component_order: Z12H`** — Z, two horizontals, and a
hydrophone — against a `--components` default of `ZNE12`, which has no `H`. A
`component_order` mismatch does not fail loudly; the model loads and picks on
mis-ordered traces. Check before campaign 4 runs.

**`pip install seisbench` is unpinned** in the Dockerfile, while `jma_wc`
declares `seisbench_requirement: 0.9.0` and `original` 0.3.2. Which weight
*version* SeisBench resolves also depends on it — 0.12.3 takes `original` at
`.v2` and `jma_wc` at `.v1`. An unpinned install means two builds of the same
commit can differ. Pin it.

**`UL` has no metadata.** Listed in `ncedc.txt` but `networks/UL.zip` does not
exist, so it is absent from the queue. Decide whether it belongs before calling
NCEDC complete.

## 7. Cheaper compute, if the effort is warranted

**One band per station** is already implemented
([17_launch_conventions.md](17_launch_conventions.md)) — this is done, listed
only because older documents still describe it as pending.

**arm64 image** — 1.68× on price and the only route to the cheapest EC2 tier
($0.0133/vCPU-hr measured on Graviton against ~$0.0148 estimated for Fargate
Spot). Does not exist yet. Only worth it if item 1 shows the campaign is
compute-bound and expensive.

**Cross-region** is settled and needs no work: the workload is CPU-bound, so
reading SCEDC from us-east-2 costs about 1.1 s of a 33.7 s shard. Run where the
Fargate quota is.

## 8. No billing baseline

Cost Explorer is blocked on this account by an organisation SCP (`p-q1ngvul9`),
and CloudWatch Budgets is unavailable, so "match or beat 2025" has nothing to
compare against and a running campaign has no authoritative spend figure.

Two ways out, neither done: request an SCP exemption for the campaign window, or
export from the billing console. Until then the only cost signal is vCPU-hours
from the dashboard × $0.0148, which is an *estimate* — the Fargate Spot rate is
not published through the pricing API.

Weekly, during a campaign: record vCPU-hours and shards complete, compare against
~4.43 s per planned station-day, and investigate anything over 20%.

## 9. Is the planner's station-day count the right cost basis?

The measured shard was **sized at 460 station-days and processed 100** — the rest
had no data and were skipped. Skipping is cheap (`s3.list` 13 s for 20 calls,
`s3.head` 0.28 s for 300), so this is not waste; but it means the campaign's
112.9M planned station-days are not 112.9M units of work.

The ~$16,400 estimate already uses seconds per *planned* station-day, so it is
self-consistent. The risk is that the 22% hit rate is specific to CI in 2010 —
station density and data availability both rise sharply over the 2010–2026 span.
If later years hit 60%, the cost is nearly 3× the estimate.

Cheapest way to close it: record processed-vs-planned from the `--procs` sweep
shards, which span whatever the queue hands them, and compare.

---

## Priority

| # | item | blocks | effort |
|---|---|---|---|
| 1 | `--procs` scaling | the whole cost model | in flight |
| 2 | EarthScope I/O profile | campaigns 3–5, ~$14,800 of the estimate | 2 h |
| 9 | processed-vs-planned ratio | the cost basis | free, from item 1 |
| 6 | `obs` components, seisbench pin | correctness of campaign 4 | 1 h |
| 5 | verify compaction before trusting it | analysis after the campaign | 2 h |
| 3 | `amp.wood_anderson` | ~$4,000 | days |
| 4 | `s3.get` throughput | ~$3,000 | may fall out of item 1 |
| 8 | billing baseline | knowing what it actually cost | 1 h + waiting |
| 7 | arm64 | ~1.68× on price | days |
