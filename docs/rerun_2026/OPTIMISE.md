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

Revisions `:1`–`:4` set no thread environment at all. **`quakescope_v3_worker:6`
pins all five to 2**, matching its default `procs` of 4, so a submission that
overrides nothing now gets the measured optimum instead of the worst case. Any
submission that changes `--procs` still has to override them together:

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

### Observed in production, 2026-09-01 21:47:50 UTC

A real Spot reclaim took both SCEDC arms of the I/O profile in the same second,
on `quakescope_v3_worker:6`. It settles the question, because it produced the
one combination that cannot be argued with:

| | sc1 | sc4 |
|---|---|---|
| `--procs` | 1 | 4 |
| claims released | 1 of 1 | **4 of 4** |
| exit code | 0 | 0 |
| Batch status | **SUCCEEDED** | **SUCCEEDED** |
| `statusReason` | `Your Spot Task was interrupted.` | `Your Spot Task was interrupted.` |
| retried | **no** | **no** |
| shards completed | 0 | 0 |

The job definition's rule is
`{onStatusReason: "Your Spot Task was interrupted.", action: retry}`, and Batch
set `statusReason` to **exactly that string** — and still did not retry.
**`evaluateOnExit` is only consulted for a failed attempt.** An attempt that
exits 0 is a success, so no rule is evaluated, no matter what the status reason
says. Nothing about the rule can be tuned to fix this; the exit code is the only
lever.

### Worse than "always leaks": at `--procs > 1` it is a coin flip

A fourth preemption the same afternoon, on the `es4` arm, exited **1** and *was*
retried — the opposite of the two above at the same `--procs 4`. The mechanism
is [worker.py:437](../../sb_catalog/src/worker.py#L437):

```python
if any(p.exitcode not in (0, None) for p in procs):
    sys.exit(1)
```

A child that releases its claim within `SHUTDOWN_GRACE_SECONDS` exits 0; a child
still running when the grace expires is `p.kill()`ed and exits −9. So the
parent's exit code — and therefore whether Batch retries the worker at all —
depends on whether every loop happened to finish in time:

| event | loops that released | parent exit | retried |
|---|---|--:|---|
| sc4, 21:47:50 | 4 of 4 | 0 | no |
| sc4, 21:57:48 | 4 of 4 | 0 | no |
| sc4, 22:25:30 | 4 of 4 | 0 | no |
| **es4, 21:50:44** | **2 of 4** | **1** | **yes** |

In the es4 case workers 1 and 2 never logged `Preempted while holding`. The
likely reason is that a Python signal handler only runs between bytecodes, and
these loops were inside a `model.classify` call that averages **20.8 s** at
`--procs 4`; under memory pressure (see item 0d) a child can miss the 90 s
window entirely.

So at `--procs 1` the fleet always decays, and at `--procs > 1` it decays by an
amount nobody can predict. Both are bad, and they are bad in a way that looks
identical on the dashboard.

**The decision is forced: the graceful preemption path has to exit non-zero,
deliberately, rather than as a side effect of which child won a race.** The cost
is cosmetic — ordinary preemptions appear as failed attempts in the console —
and the `evaluateOnExit` catch-all `{onReason: "*", action: exit}` already stops
a genuinely broken job from consuming all 10 attempts.

Note what this cost here: two of four measurement arms died 9 minutes in having
completed no shard, and had to be resubmitted by hand. At 1,500 workers over
days, nobody is resubmitting by hand.

**The same event verified the 0c fix in production.** All four of sc4's loops
released their claims from a real SIGTERM delivered to PID 1 by the platform,
not a hand-sent signal. Before the fix these four shards would have been
stranded for the full 6 h lease.

**Still needed either way:** something that notices the fleet shrinking. The
dashboard shows it only as a falling vCPU count with no failures anywhere.

## 0b. WITHDRAWN — the EarthScope credentials were wired all along

**The blocker recorded here on 2026-09-01 did not exist.** It claimed that no
job definition carried `ES_OAUTH2__REFRESH_TOKEN` and that none set an
`executionRoleArn`. Both were artefacts of the tool used to look, and the entry
is withdrawn rather than deleted because the way it went wrong will recur.

**The local `aws` CLI is `aws-cli/2.0.34`, built mid-2020.** Its service model
predates several Batch fields, and it drops every one of them from
`describe-job-definitions` output silently — no warning, no error, just absent
keys that read exactly like `null`:

```
secrets   executionRoleArn   platformCapabilities
networkConfiguration   fargatePlatformConfiguration   evaluateOnExit
```

The same definition, read two ways on the same day:

| | via `aws-cli/2.0.34` | via boto3 1.40.61 |
|---|---|---|
| `secrets` | absent → read as 0 | `ES_OAUTH2__REFRESH_TOKEN` → the secret ARN |
| `executionRoleArn` | absent | `arn:...:role/SeisBenchBatchRole` |
| `platformCapabilities` | absent | `["FARGATE"]` |
| `evaluateOnExit` | absent | Spot-interruption retry + `*` exit |

**Audit job definitions with boto3, not the local CLI.** `describe-job-definitions`
through that CLI cannot be used as evidence of absence for anything added to the
Batch API after 2020.

What is actually deployed, confirmed through boto3 and by policy simulation:
`quakescope_2026_earthscope:2/:4` and `quakescope_2026_western:2` carry the
secret; the execution role is `SeisBenchBatchRole`, the same role the job runs
as, so the `QuakeScopeEarthScopeSecretRead` inline policy is on the role that
performs the injection; and simulating `secretsmanager:GetSecretValue` against
the token ARN for that role returns **allowed**. Detail and the read-back are in
[19_earthscope_access.md](19_earthscope_access.md).

**Still worth one live check before campaigns 3 and 5.** Configuration being
correct is not the same as a container having read a restricted network on this
image. Doc 19's evidence for that is a log line from an earlier build. One shard
on a restricted network under `quakescope_2026_earthscope:4` settles it; it is
minutes, not an IAM project.

For the record, the shard split this was thought to block:

| | shards | share |
|---|--:|--:|
| fully restricted | 123,771 | 81% |
| mixed | 3,496 | 2% |
| all Open Data (anonymous) | 25,941 | 17% |

The I/O profile in item 2 uses Open Data networks regardless, so it never
depended on this either way.

**This is the second time a document has asserted deployed state that nobody
read back from the account, and the first correction was wrong in the same
direction as the thing it corrected.** Both the original claim and its
retraction were written from a description rather than a live read. The habit
that catches it is not scepticism, it is `boto3.describe_*` and
`simulate_principal_policy`.

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

**Confirmed again by a real reclaim on 2026-09-01 21:47:50 UTC**, this time with
the signal delivered by the platform to PID 1 rather than by hand: all four
loops of a `--procs 4` worker released their claims. See item 0a, which the same
event settles.

**This raises the priority of 0a.** Every preemption is now graceful, so every
preemption now exits 0 — and therefore is never retried. The fleet-decay
question is no longer occasional, it is the normal path.

## 0d. BLOCKER — `--procs 4` runs out of memory on EarthScope data

**Found 2026-09-01 by the I/O profile, on the settled configuration.** The `es4`
arm — `quakescope_v3_worker:6`, `--procs 4`, `OMP_NUM_THREADS=2`, 8 vCPU /
**16 GB** — was killed by Batch after 31 minutes:

```
attempt 1: exit=1  reason=OutOfMemoryError: container killed due to memory usage
```

It had completed exactly one shard of four. `es1` — the same image, same data,
same 16 GB, `--procs 1` — completed its shard with no trouble. The variable is
the process count.

**Why EarthScope and not SCEDC.** EarthScope stores one multi-channel object per
station-day and it is downloaded and parsed *whole*: a UW sample held 214 traces
across 38 channel codes, of which the picker uses three. SCEDC and NCEDC store
one object per channel, so a station-day parses only the band it needs. Four
concurrent whole-station-day parses do not fit in 16 GB; four concurrent
single-band parses do. This is the same asymmetry item 2 predicted would cost
*bandwidth* — it does not (item 2), it costs *memory* instead.

**This is the risk that was accepted when `4 x 2` was chosen over `8 x 1`.**
Item 0 took `4 x 2` explicitly because "four processes hold half the Parquet
write buffers of eight — the memory ceiling flagged in
[16_skypilot_vs_fargate.md](16_skypilot_vs_fargate.md) §4". That reasoning was
right about the direction and wrong about the margin: four is still too many
when the archive is EarthScope. §4 also names the lever —
`flush_threshold` buffers a partition at ~800 MB resident **per partition per
process**, and says to lower it if the process sweep shows memory pressure. It
now has.

**Nothing here invalidates item 0's 1.50x.** That sweep ran on SCEDC, where the
memory shape is different, and its ranking stands for SCEDC and NCEDC.

### Where the memory actually goes, measured

On a real Open Data object, `PS09.AK.2020.309` — 140.4 MB, 18 traces,
64.9M samples — parsed three ways with `tracemalloc`:

| | peak | traces kept |
|---|--:|--:|
| `obspy.read(buff)` | 542.5 MB | 18 |
| `read()` then `.select()` — **what the worker does** | 540.4 MB | 3 |
| `read(sourcename="AK.PS09..HN?")` | **347.7 MB** | 3 (identical) |

**Two things follow, and the second matters more.**

**1. Filtering at read time is worth 1.6x, not the 70x the trace count
suggests.** [`_read_waveform_from_s3`](../../sb_catalog/src/s3_helper.py) calls
`obspy.read(buff)` with no filter, decodes all 18 traces, and
[the caller](../../sb_catalog/src/s3_helper.py) then `.select()`s three — so the
full stream and the copy are both resident. ObsPy passes `sourcename` to
libmseed as a record-level selection, so non-matching records are never decoded.
Output is byte-identical and parse is slightly faster (0.58 → 0.49 s).

But **"214 traces of which we use three" overstates the waste.** It is true by
count and false by volume: on PS09 the 15 discarded traces are almost all 1 Hz
state-of-health channels (`VCO`, `VEA`, `VM0`, `VKI`), and the three kept are
51.8M of the 64.9M samples. The gain is station-dependent — larger on a station
carrying `HH`+`BH`+`HN`, where the discards are whole broadband sets.

**2. The dominant term is the in-flight queue, not the parse.**
[picker.py:383](../../sb_catalog/src/picker.py#L383) is
`asyncio.Queue(data_queue_size)` holding **decoded** Streams, default **5**. Per
process that is ~5 queued plus ~2 in flight at ~0.4–0.5 GB each ≈ 3 GB; at
`--procs 4` that is 10–15 GB before torch and the Parquet buffers. That is the
16 GB, and it is why `es1` at `--procs 1` was untroubled on the same data.

### Options, cheapest first

1. **`--data_queue_size 1` on the EarthScope campaigns.** No code change — it is
   already a CLI flag — and it cuts the dominant term ~3x. Try this first.
2. **Pass `sourcename` at read.** Small, contained change; 1.6x here and more on
   multi-band stations; also removes the redundant `.select()` copy.
3. **Raise memory to 32 GB** for the EarthScope and western job definitions.
   Fargate allows 8 vCPU with up to 60 GB. Works, costs memory-GB-hours, and
   hides the cause rather than fixing it.
4. **Lower `--flush-threshold`** from 4M rows. Free, but trades against the
   Parquet fragmentation of item 5, which is already bad.
5. **Run EarthScope at `--procs 2`.** Gives up part of the 1.50x on the majority
   of the campaign. Last resort.

### Checked and cleared: the band mix is not a hidden multiplier

Worth recording because the station-count view of it is alarming and wrong.

Selected band across the EarthScope campaign, weighted by operating windows
clipped with the planner's own `_operating_windows` / `_overlap_days` — the
total reconciles exactly with the 67,983,975 station-days in
[21_queues_written.md](21_queues_written.md):

| band | nominal Hz | stations | station-days | % sd | % samples |
|---|--:|--:|--:|--:|--:|
| HN | 100 | 6,883 | 27,360,362 | 40.2% | 42.0% |
| HH | 100 | 12,682 | 20,294,599 | 29.9% | 31.1% |
| BH | 40 | 9,347 | 8,197,403 | 12.1% | 5.0% |
| EH | 100 | 5,542 | 7,862,261 | 11.6% | 12.1% |
| SH | 50 | 1,094 | 1,946,817 | 2.9% | 1.5% |
| DP | 250 | 11,967 | 1,835,562 | 2.7% | 7.0% |
| others | | 4,331 | 486,971 | 0.7% | 1.2% |

**Station-day-weighted mean: 95.9 Hz — 0.96x a uniform 100 Hz assumption.** No
correction needed.

Two things that look like problems and are not:

- **`DP` is 23.1% of stations but 2.7% of station-days.** Nodal deployments are
  numerous and short. Counting stations overstates it 9x.
- **`HN` chosen over an available `BH`: 119 stations, 0.2%.** The
  `CHANNEL_PRIORITY` comment sizes this as "10 of 24,111" for western states;
  EarthScope is 5x that rate and still negligible.

**The caveat that remains.** Those percentages use the *nominal* rates in the
`CHANNEL_PRIORITY` comment, and spot checks show the comment is not accurate for
EarthScope: `PS09.AK` runs `HN` at **200 Hz** (not 100), and twelve sampled AK
`BH` channels run at **50 Hz** (not 40). Selection is unaffected — only the
priority *order* matters there — but any memory or cost model keyed to those
annotations is off by whatever the real rates are. `HN` is 40.2% of the campaign,
so if 200 Hz is common rather than a PS09 quirk the weighted mean moves
materially. Sampling record headers with 8 KB range reads answers it for a few
cents; it does not need a run.

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

## 2. RESOLVED — EarthScope I/O is not the problem. It reads 10x *faster*

**Measured 2026-09-01** on `quakescope_v3_worker:6` (image `fe61788`), campaign
prefix `_iotest2/es1`, `--procs 1` on 8 vCPU so it is like-for-like with the
`--procs 1` SCEDC baseline in [README.md](README.md). Shard
`2020309-2020312-acb849e20ec5`: 36 AK stations (Open Data, us-east-2), 3 days,
108 planned station-days, **73 processed, 782.3 s, 66,992 picks**.

A **same-day SCEDC control** ran on the same image at the same `--procs 1`
(`_iotest2/sc1`, shard `2015135-2015138-878a471d41d6`, 31 CI stations, 36 of 93
planned station-days, 719.7 s), so the comparison does not rest on the older
baseline:

| | EarthScope (AK, us-east-2) | SCEDC control, same day | SCEDC baseline (2010) |
|---|--:|--:|--:|
| `s3.get` throughput | **90.3 MB/s** | 11.6 MB/s | 8.8 MB/s |
| `s3.get` seconds | 34.71 | 121.53 | 502.18 |
| `s3.get` share of wall | **4.4%** | 16.9% | 24.6% |
| MB per station-day-channel | 43.0 | 39.3 | 44.2 |
| seconds per processed station-day-channel | **10.7** | 20.0 | 20.4 |

The control reproduces the old baseline's throughput to within 30% and its
seconds-per-processed-unit to within 2%, which is the check that the harness
measures anything at all. **EarthScope reads 7.8x faster than SCEDC measured the
same afternoon**, and moves 9% more bytes per station-day-channel — not the
multiple the object layout suggested.

**The standing suspicion was wrong in both of its parts.**

1. *"EarthScope transfers far more bytes, because it stores one multi-channel
   object per station-day and the picker uses three of ~214 traces."* It does
   not: 43.0 MB per station-day-channel against SCEDC's 44.3. Whatever the
   object layout costs, it does not show up as bytes on the wire.
2. *"EarthScope reads slowly — an earlier test sat on `Load ZI.CAMP.10` for 25
   minutes."* Already known to have been the stale image. At 90.3 MB/s,
   EarthScope is the **fastest** archive measured, and `s3.get` falls from a
   quarter of the shard to 4%.

**This also resolves item 4, and confirms item 4a's reframing.** The unexplained
8.8 MB/s was a property of *SCEDC*, not of the pipeline: it is the one archive
that is cross-region **and** has no gateway endpoint. Read same-region, the same
code moves data at 90.3 MB/s. There is no I/O bug to find.

### What it moved instead: the cost basis, and not favourably

The stage mix on EarthScope-like data is a different shape:

| stage | seconds | %wall | per unit |
|---|--:|--:|---|
| `model.classify` | 543.87 | **69.4** | 7.450 s/call |
| `amp.wood_anderson` | 157.14 | 20.1 | 2.346 ms/pick |
| `amp.velocity` | 80.12 | 10.2 | 1.196 ms/pick |
| `s3.get` | 34.71 | 4.4 | 90.3 MB/s |
| `mseed.parse` | 16.87 | 2.2 | 185.8 MB/s |
| `s3.list` / `s3.head` / parquet | 0.44 | 0.0 | |

**Inference is now the dominant stage at 69.4%, not `amp.wood_anderson`.** That
is the reverse of the SCEDC baseline, where `wood_anderson` was 58% and
`classify` 32%. Item 3 is still worth doing, but on EarthScope — 60% of the
campaign — it is worth far less than the SCEDC shard implied.

**And both of today's shards cost ~1.7x the estimate's cost basis** — see item 9,
which this promotes from a hypothetical to the largest open number in the
campaign. It is *not* an EarthScope effect: the SCEDC control is the worse of
the two.

## 2a. Superseded — the original statement of the question

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

## 3. `amp.wood_anderson` is 20–58% of runtime, and inference is the bigger lever

**Revised 2026-09-01.** The "58%" was one shard. Across the three shards now
measured the ranking *flips*, and `model.classify` is the larger stage in both of
the newer ones:

| shard | `amp.wood_anderson` | `model.classify` | ms/pick |
|---|--:|--:|--:|
| README baseline, CI 2010 | **57.9%** | 32.5% | 10.262 |
| `sc1`, CI 2015 | 32.3% | **54.0%** | 5.979 |
| `es1`, AK 2020 | 20.1% | **69.5%** | 2.346 |

No code changed between them that touches amplitude — the difference is the
data. So neither number is "the" share; the stage mix depends on the shard, and
a lever sized from one shard is sized wrong.

**Weighted by where the campaign actually is** — EarthScope is 60% of planned
station-days — inference dominates and `amp.wood_anderson` is the *smaller*
target. Prefer work on `model.classify` (batching, or the arm64/quantisation
route in item 7) over further amplitude work.

Already done on amplitude, and not to be redone: the deconvolution was hoisted
out of the per-pick loop (5.3× on that stage, and *more correct* — the old 33 s
window was ill-conditioned), the per-pick path moved to numpy, and a nested
`joblib.Parallel` that oversubscribed every core was removed. What remains is the
deconvolution of a day-long trace, not the per-pick work.

## 4. RESOLVED — the 8.8 MB/s was SCEDC, not the pipeline

It was a SCEDC artefact, exactly as item 4a predicted. Measured same-region on
the same image and the same `--procs 1`:

| archive | region | gateway endpoint | `s3.get` |
|---|---|---|--:|
| EarthScope (`es1`) | us-east-2, same as compute | none | **90.3 MB/s** |
| SCEDC (`sc1`) | **us-west-2, cross-region** | none | 11.6 MB/s |
| SCEDC (2010 baseline) | us-west-2, cross-region | none | 8.8 MB/s |

Same code, same day, 7.8x apart. There is no pipeline I/O bug and nothing here
to optimise. The number was low because `scedc-pds` is the one archive that is
cross-region, and doc 16's 46.6 MB/s same-region figure was never a like-for-like
comparison against it.

Adding the gateway endpoint (item 4a) is still free and still worth doing, but it
covers same-region traffic only — which is now known to be the fast path
already. It will not help SCEDC.

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

## 9. CONFIRMED — the cost basis is low by ~1.7x. Now the largest open number

The estimate's basis is **4.43 s per planned station-day**, taken from a single
2010 CI shard that was **78% empty**. Two shards measured on 2026-09-01, on
different archives, both come in at about 1.7x that — and they agree with each
other while disagreeing with the shard the estimate rests on:

| run | archive, year | planned | processed | hit rate | **s / planned sd** | vs basis |
|---|---|--:|--:|--:|--:|--:|
| README baseline | CI 2010 | 460 | 100 | 21.7% | 4.43 | 1.00x |
| `sc1` | CI **2015** | 93 | 36 | 38.7% | **7.74** | **1.75x** |
| `es1` | AK **2020** | 108 | 73 | 67.6% | **7.24** | **1.64x** |

**The mechanism is the hit rate, and only the hit rate.** Seconds per
*processed* station-day-channel are essentially unchanged between the 2010
baseline and the 2015 control — **20.38 vs 19.99 s, 2% apart**. Nothing got
slower. The 2010 shard simply spent most of its planned station-days discovering
there was no data, at `s3.list`/`s3.head` prices, and the estimate then divided
real work by a denominator inflated with cheap misses.

Data availability rises sharply across 2010–2026, so the one year that was
measured is the cheapest year in the campaign.

**What it does to the number.** On its own this correction roughly *doubles* the
campaign — but it is not on its own. The published figure also applied SCEDC and
`jma_wc` economics to campaigns that use neither, and correcting that pulls the
other way by almost the same factor:

| | |
|---|--:|
| published | $10,963 |
| + hit rate 21.7% → 40% | $16,487 |
| + per-campaign archive and weight | **$10,465** |

Full rebuild, per campaign, with the sensitivity table:
[24_cost_model.md](24_cost_model.md). **The near-agreement with the old figure
is two errors cancelling, not confirmation.**

**How to close it properly, and it is cheap.** The hit rate is a property of the
*queue*, not of the picker: it is `objects that exist` ÷ `station-days planned`.
That is answerable with `s3.list` alone — no inference, no GPU, no picking — by
sampling shards stratified across years and campaigns and counting listings.
An hour of listing against 112.9M planned station-days replaces the single
number the whole cost model divides by. **Do this before committing to a spend,
not after.**

Note also that the ~$11,000 figure applies the 1.50x `--procs 4` speedup, which
item 0d shows does not currently hold on EarthScope at 16 GB. The two corrections
push in the same direction.

---

## Priority

| # | item | blocks | effort |
|---|---|---|---|
| ~~0~~ | ~~`--procs` x threads~~ | **done — 1.50x, ~$16,400 to ~$11,000** | — |
| ~~0c~~ | ~~`--procs > 1` breaks graceful preemption~~ | **fixed and verified 2026-09-01** | — |
| **0d** | **`--procs 4` OOMs on EarthScope at 16 GB** | **campaigns 3 and 5 — 60% of the campaign** | 2 h |
| **0a** | **preempted workers are not replaced** | **decided by a live reclaim: must exit non-zero. Not yet implemented** | 1 h |
| ~~0b~~ | ~~EarthScope credentials unwired~~ | **withdrawn — a stale `aws` CLI, not a blocker** | — |
| 0b′ | one live restricted-network read on the current image | campaigns 3 and 5 | 15 min |
| ~~2~~ | ~~EarthScope I/O profile~~ | **done — 90.3 MB/s, 7.8x the same-day SCEDC control; not a risk** | — |
| **9** | **hit rate by campaign and era** | **the dominant cost term — swings the campaign $5.1k–$24.6k** | 1 h of `s3.list` |
| 6a | `wa_min_conf` 0.5 → 0.3 if more amplitudes are wanted | 3x WA coverage for ~+40% cost — [23](23_amplitude_review.md) | decision |
| 6b | one pick in 41 disagrees 48% between short-window and whole-day WA | 0.17 ML, cause unknown — [23](23_amplitude_review.md) §3 | 2 h |
| 6 | `obs` components, seisbench pin | correctness of campaign 4 | 1 h |
| 5 | verify compaction before trusting it | analysis after the campaign | 2 h |
| 3 | `amp.wood_anderson` | ~$4,000 | days |
| 4a | add S3 gateway endpoint | free; may explain item 4 | 15 min |
| 4 | `s3.get` throughput | ~$3,000 | re-measure same-region first |
| 8 | billing baseline | knowing what it actually cost | 1 h + waiting |
| 7 | arm64 | ~1.68× on price | days |
