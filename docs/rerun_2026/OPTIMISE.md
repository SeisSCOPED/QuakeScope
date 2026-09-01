# What is left to optimise

Open work on the 2026 picking workflow, ordered by expected value. Current
verified state is in [README.md](README.md); this file is deliberately only the
things that are **not** settled.

Each item says what is measured and what is assumed. The distinction matters —
the last planning round produced a confident cost model on top of a number
nobody had measured, and all of it had to be retracted
([`archive/README.md`](archive/README.md)).

---

## 0. RESOLVED — `--procs 4` with `OMP_NUM_THREADS=2`, worth 1.50x

**Measured 2026-09-01** on `quakescope_v3_worker:4` (image `dd4fcbc`), eight
pinned identical shards per arm, throughput counted from `Put` lines over ~26
minutes. Set it and move on.

| `--procs` | threads | station-day-channels/min | speedup | worker spread |
|--:|--:|--:|--:|---|
| 1 | 8 | 4.64 | 1.00x | `120` |
| 2 | 4 | 6.54 | 1.41x | `88,84` |
| **4** | **2** | **6.97** | **1.50x** | `25,53,53,53` |
| 8 | 1 | 7.03 | 1.52x | `44,41,7,1,43,43,7` |

**Take `4 x 2`, not `8 x 1`.** They are within 0.3% of each other, and four
processes hold half the Parquet write buffers of eight — the memory ceiling
flagged in [16_skypilot_vs_fargate.md](16_skypilot_vs_fargate.md) §4. Same
throughput, less risk.

Campaign estimate moves **~$16,400 to ~$11,000**.

Safe to use as of 2026-09-01: the preemption bug that made `--procs > 1`
dangerous is fixed and verified — item 0c.

### The thread count is the whole story

Sweeping `--procs` alone found the *opposite* answer — `procs=8` came out at
1.06x, worse than `procs=2` — because `OMP_NUM_THREADS` was unset and torch
defaults to the core count. Eight processes then meant **64 threads on 8 vCPU**,
and the per-worker spread showed the thrashing plainly: `35,1,205,1,36,1`, most
workers starved.

Pinning `procs x threads = 8` turned that same `procs=8` arm into `7,7,7,1,7,7,7`
— evenly loaded — and moved it from 1.06x to 1.52x. **Never raise `--procs`
without lowering the thread count to match.**

The Batch job definition sets no thread environment at all, so this must be
passed per submission (or added to the job definition):

```
OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
NUMEXPR_NUM_THREADS, VECLIB_MAXIMUM_THREADS
```

All five, not just `OMP_NUM_THREADS`: `amp.wood_anderson` is 58% of runtime and
runs through numpy/BLAS, which threads independently of torch.

### Why the ceiling is 1.5x and not 4x

Adding processes only helps the part of the pipeline a single process leaves
idle. One process at 8 threads already saturates the box, so the gain comes from
filling stalls — I/O waits, single-threaded stretches — not from unused cores.
1.5x is the size of that gap. Do not expect more from this axis; the remaining
multiples are in items 2 and 3.

## 0a. Preempted workers are not replaced

Found while reading the sweep. On SIGTERM the worker releases its claim and
calls `sys.exit(0)` — deliberate and correct for the *shard*, which returns to
the queue and resumes from its checkpoint, costing at most 40 station-day-
channels. Observed working:

```
Checkpointed 120 station-day-channels; a preemption now costs at most 40
Preempted while holding 2015135-2015155-23a5e8f47dbb - releasing it back
Exiting on signal after 0 shards
```

But exit 0 means Batch records the attempt as **SUCCEEDED**, so the
`evaluateOnExit` rule that retries `"Your Spot Task was interrupted."` never
fires. The shard survives; the **worker does not**. Over a multi-day campaign at
1,500 Spot workers the fleet decays instead of self-healing, and the dashboard
shows it as a falling vCPU count with no failures anywhere.

Not yet decided. Exiting non-zero would make Batch retry — the hard-killed
attempts of 2026-09-01 exited 137 and *were* retried, so the `evaluateOnExit`
rule does work when the exit code is non-zero. The cost is that every ordinary
preemption then looks like a failure in the console.

**More pressing since 0c was fixed.** Preemption is now graceful in every
configuration, so every preemption takes the exit-0 path and is never retried.
What was an occasional leak is now the normal path. Whichever way it is
decided, the campaign needs *something* that notices the fleet shrinking —
the dashboard shows it only as a falling vCPU count with no failures anywhere.

## 0b. BLOCKER — no job definition can read restricted EarthScope

**81% of the EarthScope queue cannot currently run.** Found 2026-09-01 while
setting up the I/O profile.

[19_earthscope_access.md](19_earthscope_access.md) states that
`quakescope_2026_earthscope:2` and `quakescope_2026_western:2` carry
`ES_OAUTH2__REFRESH_TOKEN` through `containerProperties.secrets`. They do not:

```
$ aws batch describe-job-definitions --status ACTIVE \
    | jq '.jobDefinitions[] | select(.jobDefinitionName|test("quakescope"))
          | "\(.jobDefinitionName):\(.revision) secrets=\(.containerProperties.secrets//[]|length)"'
... every one of the 30 active definitions reports secrets=0
```

`quakescope_2026_earthscope:2` specifically returns `"secrets": null` and
`"environment": []`.

**The secret itself is fine** — `quakescope/earthscope-refresh-token` exists in
us-east-2, last changed 2026-08-30, last accessed 2026-08-31. It is the wiring
that is missing, so this is a job-definition fix, not a credentials problem.

**And wiring it needs an IAM change.** Fargate injects
`containerProperties.secrets` using the **execution** role, and none of these
definitions set `executionRoleArn` at all — only `jobRoleArn`
(`SeisBenchBatchRole`). The policy doc 19 describes,
`QuakeScopeEarthScopeSecretRead`, is attached to the job role, which is not the
role that performs the injection. So adding `secrets:` alone would fail at task
startup.

Scale of what is blocked, from the written queue:

| | shards | share |
|---|--:|--:|
| fully restricted | 123,771 | 81% |
| mixed | 3,496 | 2% |
| all Open Data (anonymous) | 25,941 | 17% |

Campaigns 3 and 5 both route to EarthScope, so this gates ~$9,000 of the
estimate. Do not schedule either until a job definition demonstrably reads a
restricted network — the evidence doc 19 cites is a log line from a container
that no current definition reproduces.

The I/O profile in item 2 deliberately uses Open Data networks so it is not
blocked on this; per doc 19 both tiers use identical object layout, so the
I/O question is answerable anonymously.

## 0c. FIXED — `--procs > 1` broke graceful preemption

**Fixed and verified 2026-09-01.** Kept in full because the failure was silent
in every direction — no error, no failed job, no alarm — and the shape of it is
worth recognising again.

`worker.py` `main()`:

```python
if args.procs <= 1:
    loop(args, 0)              # loop() installs the SIGTERM handler
    return
procs = [mp.Process(target=loop, args=(args, i)) for i in range(args.procs)]
for p in procs: p.start()
try:
    for p in procs: p.join()
except KeyboardInterrupt:      # SIGINT only - never SIGTERM
    for p in procs: p.terminate()
```

The parent installs **no SIGTERM handler**. Docker delivers SIGTERM to PID 1
only. So with `--procs > 1`:

1. the parent takes the default action and dies immediately;
2. the children are orphaned and never see the signal;
3. ~120 s later the task is SIGKILLed, so **claims are never released**;
4. Batch retries; the retry finds every shard still claimed and logs
   `Queue drained for this worker: 0 shards done, 0 failed`;
5. `n_done == 0 and n_failed == 0` does not trip the `sys.exit(1)` guard, so it
   **exits 0 and the job reports SUCCEEDED having done nothing**.

Observed 2026-09-01 on both I/O-profile arms, which is what destroyed that
measurement:

| arm | attempt 1 | attempt 2 | attempt 3 |
|---|---|---|---|
| es | 2.9 m, **exit 137** | 7.1 m, **exit 137** | 0.5 m, exit 0, 0 shards |
| sc | 9.2 m, **exit 137** | 0.0 m, **exit 137** | 0.5 m, exit 0, 0 shards |

Exit 137 is SIGKILL. The logs of attempts 1 and 2 stop mid-`Load` with no
`Preempted while holding`, no `Exiting on signal`, no checkpoint line — the
handler never ran. Afterwards both prefixes held **8 claims each, 0 shards
complete**, locked for the 6 h lease.

Contrast `--procs 1`, where `loop()` runs in the main process and the handler
does fire:

```
Checkpointed 120 station-day-channels; a preemption now costs at most 40
Preempted while holding 2015135-2015155-23a5e8f47dbb - releasing it back
```

**Campaign impact.** At 1,500 Spot workers on `--procs 4`, every preemption
strands 4 shards for 6 hours and its retries "succeed" instantly doing nothing.
Throughput collapses while the dashboard shows SUCCEEDED jobs and **zero
failures anywhere** — the same class of invisible failure as the stale image pin.

**The fix is small.** Trap SIGTERM in the parent, forward it to the children,
join with a timeout, then escalate. Roughly:

```python
def _forward(signum, frame):
    for p in procs:
        if p.is_alive():
            os.kill(p.pid, signal.SIGTERM)
signal.signal(signal.SIGTERM, _forward)
signal.signal(signal.SIGINT, _forward)
```

Decide item 0a at the same time — it is the same signal path, and the two
interact: 0a is about the graceful exit not being retried, 0c is about the
graceful exit never happening.

### The fix, and how it was verified

`main()` now traps SIGTERM/SIGINT in the parent, forwards to each live child,
and waits out a grace period (`SHUTDOWN_GRACE_SECONDS`, default 90 — inside
Spot's ~120 s window) before escalating to `kill()`. It polls `is_alive()`
rather than calling `join()`, because a plain `join()` can swallow the signal
until a child happens to exit.

Verified against the real worker, not a mock: two loops on `--procs 2`, both
holding claims, SIGTERM sent to the **parent only** exactly as Docker delivers
it:

```
Signal 15 - forwarding to 2 worker loops so they release their claims
worker1 | Preempted while holding 2016030-2016050-2be66209dd1e - releasing it back
worker0 | Preempted while holding 2016010-2016030-e5c832e16dd5 - releasing it back
```

Both claims gone from S3 afterwards; before the fix the same signal stranded
both for the full 6 h lease.

**This raises the priority of 0a.** Every preemption is now graceful, so every
preemption now exits 0 — and therefore is never retried. The fleet-decay
question is no longer occasional, it is the normal path.

## 1. How that was measured, and the two designs that were wrong

Kept because both mistakes are easy to repeat, not because the question is open.

**Attempt 1 measured shard luck.** Every arm got `--max-shards 1`, so `--procs P`
claimed P shards from the live queue and its wall clock was the *maximum of P
random draws* while `--procs 1` drew once. Under perfect scaling `procs=8` would
still finish later. The draws are nowhere near equal — two shards from that run:

| shard | wall | station-day-channels |
|---|--:|--:|
| `2018219-2018239-9c91d08e08b6` | **23 s** | **0** — no data at all |
| `2012031-2012051-52c578f41b6a` | 1145 s | 60 |

A 50x spread, some shards entirely empty. Any design handing different shards to
different arms measures the queue, not the change.

**Attempt 2 swept one axis of a two-axis problem** — `--procs` without
`OMP_NUM_THREADS` — and produced an answer that was not merely imprecise but
*inverted*: `procs=8` looked worse than `procs=2`. See item 0.

**The design that worked.** Four campaign prefixes under
`s3://quakescope-picks-2026/_omptest/{a,b,c,d}`, each holding an **identical**
`shards.jsonl` of the same 8 shards — all 620 planned station-days, the same 31
stations, consecutive 20-day windows through 2015, so no straggler dominates.
Picks land in the test prefixes, keeping the real catalogue clean.

**Do not wait for completion.** Throughput is counted from `Put` lines while the
arms run, so the answer arrives in ~20 minutes instead of the ~7 hours the
`procs=1` arm needs to grind 8 shards sequentially. The ranking was already
correct at 6 minutes and unchanged at 26. Total cost of the sweep: about $0.10.

The `procs=1` arm reproduced the previous run's baseline to within 2% (4.64 vs
4.85 station-day-channels/min), which is the check that the harness measures
anything at all.

```bash
# rate per arm, while running
python3 -c "
import boto3, json, collections, re, time
b  = boto3.client('batch', region_name='us-east-2')
lg = boto3.client('logs',  region_name='us-east-2')
for n, c in json.load(open('omptest.json'))['arms'].items():
    d = b.describe_jobs(jobs=[c['job_id']])['jobs'][0]
    el = (time.time()*1000 - d['startedAt'])/60000
    ev = lg.get_log_events(logGroupName='/aws/batch/job',
             logStreamName=d['container']['logStreamName'],
             startFromHead=True)['events']
    n_put = sum(1 for e in ev if '| Put ' in e['message'])
    print(f\"{n}: procs={c['procs']}x{c['threads']}  {n_put/el:.2f}/min\")"
```

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

Unexplained, but the candidate list narrowed on 2026-09-01. The measurement was
taken on SCEDC, which is the **only cross-region archive** (`scedc-pds` is
us-west-2, compute is us-east-2) **and** has no S3 gateway endpoint — see item
4a. Doc 16's 24.1 MB/s cross-region figure was also SCEDC-from-us-east-2, so
8.8 MB/s is low even against the cross-region baseline, but the comparison to
46.6 MB/s same-region was never like-for-like.

Remaining candidates: no gateway endpoint, `--procs 1` leaving no concurrency to
hide latency, 2010 objects being small and numerous, or per-request overhead at
14.7 MB per call.

**Re-measure on a same-region archive before spending effort here** — NCEDC or
EarthScope, both us-east-2. The number may simply be a SCEDC artefact.

## 4a. No S3 gateway endpoint — free, and covers nearly all reads

Checked 2026-09-01 on compute environment `niyiyu_earthscope` (`vpc-0543376e`,
subnets `subnet-f85fc393`, `subnet-4f37fd32`, `subnet-2b635e67`):

```
endpoints : NONE
NAT gws   : none
default route -> igw-b6f0e6de   (all three subnets)
```

**The good news first.** Tasks run in public subnets behind an Internet Gateway,
not private subnets behind NAT. So the failure mode where NAT data-processing at
$0.045/GB quietly exceeds the compute bill **does not apply here**. Across ~1.1 PB
of campaign reads that would have been a five-figure surprise. Confirm this again
if the VPC is ever rebuilt, because it is invisible until the bill arrives.

**What is missing is a Gateway endpoint for S3.** It has no hourly and no per-GB
charge, so there is no cost argument against one. Verified bucket regions:

| bucket | region | |
|---|---|---|
| `ncedc-pds` | us-east-2 | same region as compute |
| `earthscope-geophysical-data` | us-east-2 | same region |
| `earthscope-mseed-v2-…-s3alias` | us-east-2 | same region |
| `scedc-pds` | **us-west-2** | the only cross-region archive |

A gateway endpoint covers same-region S3 only, but that is nearly all of the
campaign: EarthScope alone is 68.0M of 112.9M station-days, NCEDC another 6.0M,
and western is predominantly EarthScope-routed. Only the SCEDC campaign
(4.1M station-days) plus western's ~3,211 SCEDC/NCEDC-routed stations read
cross-region.

Also a posture note the cost lens does not capture: public subnets mean tasks
carry public IPs. Not a charge, but not least-privilege networking either.

**This reframes item 4.** The 8.8 MB/s `s3.get` was measured on the SCEDC smoke
test — the one archive that is cross-region *and* has no endpoint. EarthScope may
well read faster than that baseline implies, which cuts the opposite way from the
standing worry in item 2. Do not resolve item 4 without re-measuring on a
same-region archive.

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
| ~~0~~ | ~~`--procs` x threads~~ | **done — 1.50x, ~$16,400 to ~$11,000** | — |
| ~~0c~~ | ~~`--procs > 1` breaks graceful preemption~~ | **fixed and verified 2026-09-01** | — |
| **0a** | **preempted workers are not replaced** | **now the normal path — every preemption exits 0** | 1 h |
| 0b | EarthScope credentials unwired | campaigns 3 and 5, ~$9,000 | IAM change |
| 2 | EarthScope I/O profile | campaigns 3–5, ~$14,800 of the estimate | 2 h |
| 9 | processed-vs-planned ratio | the cost basis | free, from item 1 |
| 6 | `obs` components, seisbench pin | correctness of campaign 4 | 1 h |
| 5 | verify compaction before trusting it | analysis after the campaign | 2 h |
| 3 | `amp.wood_anderson` | ~$4,000 | days |
| 4a | add S3 gateway endpoint | free; may explain item 4 | 15 min |
| 4 | `s3.get` throughput | ~$3,000 | re-measure same-region first |
| 8 | billing baseline | knowing what it actually cost | 1 h + waiting |
| 7 | arm64 | ~1.68× on price | days |
