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

## 0a. FIXED — preempted workers now exit non-zero, so Batch retries them

**Fixed 2026-09-02.** `worker.py` exits **75** (`PREEMPTED_EXIT_CODE`) on the
graceful preemption path instead of 0, in both process modes:

- `loop()`'s `except Preempted:` handler, which covers `--procs 1`;
- `main()`, which now reports preemption **before** inspecting child exit codes,
  so the node's fate no longer depends on whether every child happened to
  release inside the grace period.

**No job-definition change is needed, and that is the point.** The existing
rules were already correct; the exit code was the only missing piece:

```
1. {onStatusReason: "Your Spot Task was interrupted.", action: retry}
2. {onReason: "*", action: exit}
```

Rule 1 fires only for a genuine Spot reclaim, so an operator `TerminateJob`
carries a different `statusReason`, falls to rule 2, and is **not** retried —
the emergency stop in [15_monitoring.md](15_monitoring.md) still works. Verified
against the `es4` arm, which exited 1 with `statusReason` "Your Spot Task was
interrupted." and *was* retried, while its OOM attempt reported "Essential
container in task exited" and was not.

75 rather than 1 so that "preempted" stays separable from "this job is broken"
when reading attempt histories.

The analysis that led here is kept below, because the mechanism is easy to
forget and the failure was invisible in every direction.

## 0a-history. Preempted workers were not replaced

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

## 0h. BLOCKER — high-rate bands still OOM at `--procs 4`

**Found 2026-09-02, on `c4bcd21` with the item 0d fix in place.** An NCEDC arm
completed one shard and was then killed:

```
attempts: [(1, 'OutOfMemoryError: container killed due to memory usage')]
Completed 2018141-2018142-202ee85e425d in 538.1s (28 station-day-channels)
```

**The queue budget was not the whole problem.** Item 0d bounded what waits in
`data_queue`, and that part works. But the peak is upstream of the queue, in the
read path itself:

```
raw = fs.read_bytes(uri)      137 MB for one station-day-channel
obspy.read(buff)              decoded at the FULL native rate
downsample_to_target(st)      only now reduced to 100 Hz
```

These stations are `DP`/`CN` at 250–500 Hz. A 500 Hz day is 43.2M samples per
trace; three components decode to roughly a gigabyte before anything is
downsampled, and obspy's decimate allocates a filtered copy on top. Four loops
doing that at once exceeds 16 GB regardless of how small the queue is.

So there are two distinct memory ceilings, and 0d only moved the second:

| | where | fixed? |
|---|---|---|
| decoded streams waiting in `data_queue` | after the read | yes — node budget |
| **full-rate decode + resample working copy** | **inside the read** | **no** |

**Options, and this needs a decision rather than a default:**

1. **Drop `DP` and `CN` from `CHANNEL_PRIORITY`.** Together they are 4.1M
   station-days, **3.6% of the campaign**, and `DP` is largely unreadable anyway
   — only the 2014 nodal deployment is in miniSEED, the rest is still PH5. This
   removes the high-rate case entirely rather than engineering around it.
2. **Raise memory to 32 GB** for campaigns carrying high-rate bands. Costs
   memory-GB-hours, not vCPU-hours.
3. **`--procs 2` on those campaigns.** Gives up part of the parallel speed-up.

Measure peak RSS on one high-rate shard before choosing — that is what 0d should
have done, and its absence is why this ceiling was missed.

## 0g. FIXED — restricted EarthScope credentials were never scoped

**Root cause, 2026-09-02.** The denial below is real and reproduces exactly as
described. It is **not** an entitlement gap. An unscoped credential for
`s3-miniseed-v2` carries `s3:ListBucket` and not `s3:GetObject`; the request
has to name the network:

```python
client.user.get_aws_credentials(role="s3-miniseed-v2", network="FDSN:AV")
```

Verified interactively from EC2 in us-east-2 — the same object that returns
`AccessDenied` unscoped returns bytes when scoped. Temporary networks (FDSN
codes starting with a digit or X/Y/Z) additionally need `year=`, which is most
of this campaign: `XD`, `ZI`, `ZG`, `1D`, `1B`.

**Fixed** in `s3_helper.py`: `es_scope(net, year)` builds the parameters,
`get_es_filesystem` caches one credential *per scope* — never per object, since
a shard is thousands of GETs behind one token exchange — and `shard_planner`
now groups stations per network so a shard needs one scope rather than several.
`tests/test_credential_scope.py` and `tests/test_shard_networks.py` pin both.

Requires **`earthscope-sdk>=1.8.0`**, the first release that forwards query
parameters. The image's `boto3==1.35.81` pin had to go with it — `aioboto3`
needs `botocore>=1.36.0`. `seisbench==0.12.5` is unmoved, so the picks are
unaffected. See [19](19_earthscope_access.md) for the resolution detail.

**What this unblocks:** the 19,510 restricted stations, ~87% of the EarthScope
campaign. No request to EarthScope is needed, and none should be sent.

### Verified in-container on Fargate, us-east-2 (2026-09-02)

`diag-earthscope` on `5c36d97`, the same code path the workers use:

| network | kind | result |
|---|---|--:|
| `AV` 2019.187 | permanent | 10.4 MB @ **97.2 MB/s** |
| `CC` 2019.187 | permanent | 18.0 MB @ **97.6 MB/s** |
| `ZI` 2011.200 | temporary | 29.0 MB @ **96.4 MB/s** |
| `XD` 2008.200 | temporary | 17.9 MB @ **97.0 MB/s** |
| `1D` 2016.200 | temporary | 18.0 MB @ **97.5 MB/s** |
| `UW` 2019.187 | open data | 27.9 MB @ 94.1 MB/s |

Every one of these returned `AccessDenied` before the fix. Restricted
EarthScope now reads at the same rate as Open Data, which also settles item 2:
there was never an EarthScope throughput problem.

### The two refusals mean opposite things

EarthScope answers a bad credential request in two ways, and they need
**opposite** responses:

| status | body | meaning | response |
|---|---|---|---|
| **400** | `Temporary networks require a year` | scope shape is wrong | escalate: flip the scoping |
| **404** | `network FDSN:ZI year 2019 not found` | scope is **right**, archive has no such network-year | treat as no data; change nothing |

Escalating on the 404 was actively harmful, and briefly shipped: it flipped
`ZI` to network-only, drew a 400 from the next request, and marked both
scopings spent — so **every other year of that network would fail for the rest
of the worker's life**. `ZI` 2019 does not exist; `ZI` 2011 reads at 96 MB/s
and would have been lost with it. The diagnostic only missed it because 2011
happened to be probed first. Now pinned by a regression test and re-verified in
the deliberately hostile order (2019 then 2011).

**404 is a property of the plan, not an error.** Temporary codes are reused,
and our station metadata claims network-years the archive never held — the
metadata lists `ZI` stations starting in 2019, and EarthScope has no `ZI` 2019
at all. That belongs with the metadata-vs-archive gap in
[25](25_metadata_vs_archive.md), and it means the planned station-day count is
an overestimate by however many such network-years exist.

**The lesson.** Listing succeeded at every stage, and that is what made this
take two weeks: a successful LIST reads as proof the role is valid, so every
hypothesis pointed at entitlement rather than at the shape of our own request.
**LIST and GET are separate grants.** The evidence below is preserved because
the diagnosis was sound and only the conclusion was wrong.

---

**Diagnosed 2026-09-02** with `python -m src.picker diag-earthscope`, running in
the container on Fargate in us-east-2 with the secret injected. It walks the
request apart, and the answer is unambiguous:

```
--- AV: RESTRICTED (role)
DNS   2 ms      TCP 443  1 ms  ok
LIST      1.4 s   163 objects
HEAD      0.0 s   403 Forbidden
GET 1 KB  0.0 s   AccessDenied: arn:aws:sts::457219964709:assumed-role/earthscope-...
GET full  0.0 s   AccessDenied
--- UW: OPEN DATA (anonymous)
GET full  0.3 s   27.9 MB at 97.0 MB/s
```

Same for `CC`. **Nothing hangs** — the denial is instant. DNS and TCP are fine.
The role *is* assumed (the ARN is in EarthScope's account, 457219964709). That
role can **list** the access point and is **not permitted to read it**.

> **An earlier framing of this item said "reads hang". That was wrong** — it came
> from reading a worker log filtered for the wrong keywords, where four minutes
> of silence looked like a stall. Corrected here.

### This is item 0b, and withdrawing it was too broad

Item 0b established that the *plumbing* is correct: the secret is wired, the
execution role is right, and `simulate_principal_policy` allows reading it. All
of that is true, and none of it tests whether the assumed role can read S3.
**Listing succeeded, which looked like proof and was not.**

[19_earthscope_access.md](19_earthscope_access.md) makes the same mistake in the
other direction: it cites a container logging `Load ZI.CAMP.10 @ earthscope` as
verification. That line is printed **before** the read. It proves the code
reached the read, not that bytes came back.

### The code did not know — it asserted

`ES_DENIED_ATTEMPTS = 2` exists with a handler whose message was confident and
wrong:

> *refreshing the credential did not help, so this is an entitlement gap, not an
> expiry. The role `s3-miniseed-v2` can list this access point but is not
> permitted to read it. Ask EarthScope to grant `s3:GetObject` for network …*

Read as corroboration — someone had clearly hit this before and written it up —
it was really the same wrong inference, committed earlier and then quoted back
as evidence. A confident error message in the codebase is not a second opinion.

The handler now names the credential scope it actually used, and says to check
that scope *before* suspecting entitlement.

### Scale

Three of the four dry-run arms failed on this one cause: `esr1` and `esr4` on
`AV`, and **`west4` on `2F`** — western is ~70% EarthScope-routed, so this gates
campaign 5 as well as campaign 3.

| | planned station-days | share |
|---|--:|--:|
| EarthScope restricted | 78,041,471 | 69.1% |
| plus western's EarthScope-routed share | ~20,301,703 | ~18% |

**It was a code fix after all** — one query parameter, per the resolution at the
top of this item. The 78.0M restricted station-days and western's ~20.3M
EarthScope-routed share are both unblocked, so the campaign is back to its full
**112.9M planned station-days** rather than the 34.8M reachable without it.

[26_reproduce_earthscope_denial.md](26_reproduce_earthscope_denial.md) is kept
for its method, not its conclusion: §4a — `aws s3api` silently truncating keys
at `#` — is a trap worth keeping written down.

## 0g-history. The symptom before it was diagnosed

**Found 2026-09-02 by the second dry run, on `c4bcd21` — every fix included.**
Both restricted-EarthScope arms died at exit 137 having picked nothing. The log
shows the credential exchange working perfectly and the *first read* hanging:

```
16:30:38 worker1 | POST https://login.earthscope.org/oauth/token          200 OK
16:30:38 worker1 | GET  .../beta/user/credentials/aws/s3-miniseed-v2      200 OK
16:30:39 worker1 | Done preparing inventory for the assigned stations
16:30:39 worker1 | Load AV.ACH.  2018.142 @ earthscope
         (silence - four minutes, all four loops)
         Signal 15 - forwarding to 4 worker loops so they release their claims
         (SIGKILL)
```

**This is doc 19's symptom exactly** — "sat on `Load ZI.CAMP.10` for 25 minutes
without a further log line". Item 2 dismissed that as an artefact of the stale
image. It is not: it reproduces on an image containing every fix made since.
That dismissal is withdrawn.

What is established:

- **Credentials are fine.** Both HTTP calls return 200, the s3-miniseed-v2 role
  is assumed, and `Done preparing inventory` follows. This is not item 0b.
- **Listing is fine.** The hit-rate survey listed 310,347 restricted
  station-days with **zero errors** and took three minutes. `ls` works.
- **It is the GET that hangs.** The first `read_bytes` on the restricted access
  point produces no further log line.
- **Open Data is unaffected** — AK reads at 90.3 MB/s through the same code.

`STATION_DAY_TIMEOUT` is 900 s, so a hung read holds the loop for **15 minutes**
before the timeout fires — longer than the mean time to interruption in this
pool, so in practice Spot kills the worker first and the claim strands.

**Not yet diagnosed.** Candidates, cheapest first: the access-point alias needs
a different addressing style or region pin for GET than for LIST; the temporary
credentials carry list-but-not-get; a request is being retried silently inside
`s3fs`/botocore. `scripts/check_earthscope_getobject.sh` exists from an earlier
round and is the place to start.

**Nothing about campaigns 3 or 5 can be costed or scheduled until this is
closed.** It is 69% of planned station-days and ~66% of the estimate.

## 0e. FIXED — a retry loop was swallowing the preemption

**Found 2026-09-02 by the dry run, not by reading.** An arm exited 137 with four
claims stranded, and the log says why:

```
Signal 15 - forwarding to 4 worker loops so they release their claims
worker0 | FDSN request failed (1/8): Preempted. Sleeping 5 s.
worker1 | FDSN request failed (1/8): Preempted. Sleeping 5 s.
worker2 | FDSN request failed (1/8): Preempted. Sleeping 5 s.
worker3 | FDSN request failed (1/8): Preempted. Sleeping 5 s.
...
worker1 | Load BG.AL4.  2018.156 @ ncedc
```

The SIGTERM landed inside an FDSN metadata request. That retry loop is a broad
`except Exception ... sleep(5)`, so it caught the `Preempted` the handler raised,
logged it as a failed request, and **retried**. All four loops carried on working
after being told to stop, and Docker SIGKILLed the container ~120 s later.

**This is item 0c's failure, reintroduced by a handler three modules away.** And
it is not one handler: there are **19 broad `except Exception` clauses** in the
package plus one bare `except:`. Which one absorbs a preemption depends only on
where the signal happens to land — the EarthScope credential loop has the same
shape, and the FDSN fetch happens at the *start* of every shard, which is exactly
when a frequently-preempted worker is most likely to be interrupted.

**Fix: `Preempted` now inherits from `BaseException`**, for the same reason
`KeyboardInterrupt` and `SystemExit` do — it is control flow, not a failure. No
`except Exception` can catch it, including the ones inside obspy, boto3 and
seisbench. Auditing 19 handlers and hoping nobody adds a twentieth is a losing
game. `tests/test_preemption_not_swallowed.py` pins it.

**It also explains earlier evidence.** The first preemption test had 2 of 4 loops
release and 2 not; that was attributed to a long `model.classify` overrunning the
grace window. A swallowed signal is the better explanation, and it fits the
exit-137 attempts in the original `es4` arm too.

## 0f-plan. What to do about the reclaim rate

Reviewed 2026-09-02 by the `aws-cloud-architect` agent against the live account.
Its findings, with one correction below.

**The pool is the shape, not the AZ.** Of 18 recently stopped tasks, **12 (67%)
had `stopCode: SpotInterruption`**, split 5/3/4 across us-east-2a/b/c. That rules
out a single-AZ pool or a CE misconfiguration and points at scarcity for the
8 vCPU / 16 GB x86 shape itself. AWS publishes **no interruption-rate advisor for
Fargate Spot** — the EC2 Spot Instance Advisor and `get-spot-placement-score` are
EC2-only — so this has to be measured, not looked up.

**Watch ECS, not Batch.** There are currently **zero EventBridge rules** in this
account and region. The rule to add is on
`source: aws.ecs, detail-type: "ECS Task State Change"`, not Batch's Job State
Change: the ECS event fires within seconds of a reclaim, whereas Batch's `FAILED`
only fires after all ten attempts are spent — about 100 minutes late at the
measured rate. Feed it to a small Lambda that tops the fleet back up to target
with fresh `submit-job` calls, each getting a fresh attempt budget, plus a
5-minute scheduled sweep as a backstop. Effectively $0 at this event volume.

**Rejected, with reasons:** array jobs (each child hits the same 10-attempt cap,
and the S3 claim protocol already provides the coordination); Step Functions
(per-transition billing for multi-day churn at this population); SQS (duplicates
a queue-depth signal S3 already exposes); over-submitting (gives the sawtooth
being seen now, not steady state).

**On-demand fallback needs a quota increase first.** Fargate Spot and on-demand
cannot be blended in one Fargate CE — that is an EC2-CE concept — so it would be
two CEs on one queue with `order`. But this account's **Fargate on-demand quota
is 140 vCPU** (`L-3032A538`): 17 concurrent 8-vCPU workers, **1.1% of a
1,500-worker target**. Building the fallback before raising it would be
decoration. And cap it deliberately when built: a sustained Spot crunch could
otherwise reprice the fleet at roughly 3× with no alarm.

**No early warning exists.** EC2-style Capacity Rebalance has no Fargate Spot
equivalent. The ~2 minute `SIGTERM` at actual reclamation is the only signal, and
the worker already uses it. Do not design around a notice that does not exist.

**The public subnets are right.** Placement already spreads across all three AZs.
Going private would need a NAT gateway — interface endpoints cannot carry the
external EarthScope FDSN traffic — and NAT data processing at $0.045/GB across
~1.1 PB would dwarf what public IPs cost.

### Two corrections to the review

- **`evaluateOnExit` is present**, contrary to the finding. The agent read the
  job definition through the local `aws` CLI v2.0.34, which strips it — the trap
  in [README.md](README.md). boto3 shows
  `{attempts: 10, evaluateOnExit: [retry on Spot interruption, exit otherwise]}`.
  A good demonstration that the guardrail applies to reviewers too.
- **`maxvCpus` is 12,000, not 4,000** as recorded elsewhere in these docs. The
  agent is right and the older note was stale.

### A cost line nobody had counted

Since February 2024 AWS charges **$0.005/hour per public IPv4 address**,
including Fargate task ENIs with `assignPublicIp: ENABLED` — which these are.

| | |
|---|--:|
| campaign worker-hours (657,892 vCPU-hr ÷ 8) | 82,236 |
| public IPv4 at $0.005/hr | **$411** |
| against ~$9,940 of compute | **+4.1%** |

Not worth re-architecting away, since NAT would cost more, but it belongs in the
estimate. Unverified against a bill — Cost Explorer is blocked by the org SCP.

## 0i. MEASURED — 4.7% of the plan cannot be read, and 1.5% is an entitlement gap

`netyear-sweep` asks EarthScope's credential exchange whether each planned
`(network, year)` exists. It answers 404 for one it does not hold and 403 for
one we are not entitled to, so the whole plan can be checked without reading a
byte. Run 2026-09-03 against the written queues:

| campaign | restricted network-years | present | 404 not in archive | 403 not entitled |
|---|--:|--:|--:|--:|
| earthscope | 4,235 | 81.6% | 438 (10.3%) | **341 (8.1%)** |
| western | 628 | 87.4% | 75 (11.9%) | 4 (0.6%) |

In station-days, counting a shard dead only when *every* network-year in it is
unreadable:

| campaign | planned | unreadable | 404 | 403 |
|---|--:|--:|--:|--:|
| earthscope | 67,983,975 | 4,394,559 (6.5%) | 2,742,500 | 1,636,988 |
| western | 33,799,828 | 931,101 (2.8%) | 931,101 | 0 |
| **total** | **112,866,683** | **5,325,660 (4.7%)** | 3,673,601 | 1,636,988 |

**The two halves need opposite responses, which is why the sweep separates
them.**

**404 — 3.67M station-days, a correction to us.** Temporary FDSN codes are
reused, and our station metadata claims deployments the archive never held. Not
an error and nothing to ask for; the plan is simply 3.3% smaller than it says.
Handled gracefully in the reader since `d33bf79`, so these shards complete empty
rather than failing.

**403 — 1.64M station-days across 49 networks, a request to EarthScope.** The
data exists and this account may not read it: `AF`, `DR`, `KS`, `PI`, `TR`, `VE`
at 17 years each, then `EC`, `GI`, `TD`, `YF`, `I0`, `YE`, `ZC`, `MP`, `RI`,
`OC`, `DE`, `EO` and 31 more. Either ask for access or drop them from the
plan — but they will fail every time until one or the other happens.

### The campaign is global on purpose

Checking these, 40.9% of the earthscope plan's stations fall outside western
North America — Italy, Spain, Chile, Antarctica, New Zealand. That is
**intended, not a planning bug**: `configs/networks/earthscope_onshore.txt` is
"everything in NETWORK_MAPPING routed to EarthScope, minus the offshore list",
420 networks, with no geographic filter. Only campaign 5 (`western`) is selected
by bounding box. Worth stating because the denied networks look foreign at a
glance and invite the wrong conclusion.

### A bug this found

`LH` returned 403 and the worker retried it five times with five-second sleeps —
25 s per network-year — because `earthscope_sdk` raises `UnauthorizedError`
(403) and `UnauthenticatedError` (401) *instead of* an `HTTPStatusError`, so
neither carries `.response` and the 4xx fast-fail never saw them. Fixed in
`530fc5b`; the re-run logged **zero** retry sleeps and finished western in
1.2 min against ~3 min before. At campaign scale, 49 denied networks retrying
five times each would have been a large amount of worker time spent re-asking an
answered question.

## 0f. OPEN — the Spot pool is volatile enough to exhaust the retry cap

**Measured 2026-09-02.** A validation job was reclaimed **ten times in a row**,
every one handled correctly by the 0a fix — and then hit Batch's retry ceiling
and failed permanently:

```
mean attempt 10.2 min | median 7.1 | max 26.4
cumulative 102.4 min | 2 shards completed | no OOM
```

`attempts: 10` is **Batch's maximum**; it cannot be raised. At a ~10 minute mean
time to interruption, a worker's expected lifetime is about 100 minutes of
runtime however the retries are configured.

Two consequences for launch:

1. **The campaign needs something that resubmits workers**, not just Batch
   retries. The fleet now decays *visibly* (FAILED rather than silently
   SUCCEEDED, which is the 0a fix working) but it still decays. This is the
   concrete version of item 0a's closing note.
2. **`lease_hours: 6` is mismatched to this pool.** A stranded claim removes a
   shard from circulation for six hours. Combined with 0e, that deadlocked a
   dry-run arm completely: all 8 shards claimed by dead attempts, retries
   finding nothing to take. One hour is a better fit, and it is a per-submission
   flag rather than a code change.

Whether this reclaim rate is a transient capacity crunch in
`niyiyu_earthscope` or normal for it is **not established** — one afternoon
cannot tell. Worth sampling across a few days before sizing the fleet.

## 0d. FIXED — `--procs 4` ran out of memory on EarthScope data

**Fixed 2026-09-02**, both halves of the product:

**1. The queue is now sized from a node budget, not per loop.**
`--data_queue_size` defaulted to a flat 5 *per loop*, so the node's exposure was
`5 x procs` and nothing counted it. `worker._resolve_queue_size` now divides a
`NODE_STREAM_BUDGET` (default 8) by `--procs`, capped at the old 5:

| `--procs` | per loop | node total |
|--:|--:|--:|
| 1 | 5 | 5 (unchanged) |
| 2 | 4 | 8 |
| **4** | **2** | **8** |
| 8 | 1 | 8 |

An explicit `--data_queue_size` still wins. The worker logs what it chose.

**2. EarthScope reads now filter at the record level.**
`_read_waveform_from_s3` takes a `sourcename`, and the EarthScope branch passes
`NET.STA.LOC.CHA?`, so libmseed skips non-matching records **before** decoding
them rather than decoding all 18-214 traces and calling `.select()` afterwards.

Verified through the real method on `AK.PS09.2020.309`, comparing against what
the old path kept after its `.select()`:

```
no sourcename (old behaviour)      traces  18  peak   696.0 MB
sourcename=AK.PS09..HN?            traces   3  peak   489.2 MB
identical (id, samples, checksum)  : True
peak reduction                     : 1.42x
```

obspy takes **one** pattern — an `"a|b"` form raises rather than matching both —
so the filter applies in the single-band case, which is what `CHANNEL_PRIORITY`
always yields; more than one band falls back to a full read.

Together these cut the dominant term ~2.5x at `--procs 4` (queue 5→2) and the
per-stream size ~1.4x. **Not yet re-run on Batch** — the arm that OOMed should be
repeated on the new image before campaign 3 is scheduled.

The analysis that led here is kept below.

## 0d-history. Why it ran out of memory

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

> **That dismissal was wrong, and is withdrawn (2026-09-02).** It said the
> 25-minute stall "is now known to have been the stale image, not EarthScope",
> reasoning from the unbounded retry loops that image contained. The symptom
> reproduces on `c4bcd21`, which carries every fix made since. See item **0g**:
> restricted EarthScope reads hang, and it is the top launch blocker.

**Do this before launching campaign 3.** Run `--profile` on five EarthScope
shards on `quakescope_v3_worker:4` and compare `s3.get` seconds and MB against
the SCEDC baseline. Two hours; it either confirms the estimate or changes the
campaign plan.

## 3a. MEASURED cleanly — amplitude is the bigger cost, at least on `original`

**The only profile worth quoting is `--procs 1`.** The profiler says so itself:

> `--profile with --procs 4: worker loops contend for the same cores, so
> per-stage attribution will be distorted. Use --procs 1.`

Every dry-run arm except two ran `--procs 4`, so their stage percentages are
not usable. `west1b` (AZ 2018, `original`, procs 1, OMP 8, 16 station-days,
152 s wall, 1,128 MB read) is the clean one:

| stage | clean (procs 1) | distorted (procs 4) |
|---|--:|--:|
| `model.classify` | 32.5% | 58.1% |
| **`amp.wood_anderson`** | **31.5%** | 24.6% |
| `amp.velocity` | 17.6% | 14.5% |
| `resample` | **7.4%** | 15.8% |
| `s3.get` | 7.1% | 4.1% |

Two things this settles.

**Resample is not a problem.** It was reported here at 29.4% from a single
`--procs 4` shard and treated as a cost-model error against the 12.7% in the
model. Measured cleanly it is **7.4%** — the model if anything over-charges it.
Fragmentation is not the cause either: `esr1` (AV) had the `fragments shorter`
warning on **65%** of station-days and resampled *nothing*, because AV is EH/BH
at ≤100 Hz; `west4` (AZ) had it on 4% and did all the resampling. The cost
tracks **sample rate, not fragment count**, so merging traces before resampling
would buy nothing. A 44 MB 200 Hz SCEDC object read back as **1 trace, 0 gaps**,
and merging it gained 1.02x.

**Amplitude extraction is 49.1% of wall, against inference's 32.5%** — which
inverts the framing in item 3 below.

⚠️ **Do not generalise this to the whole campaign.** Western runs `original`,
which is ~0.35x the inference cost of `jma_wc`. Amplitude looks large here
partly because inference is cheap. The three `jma_wc` campaigns are 78M of the
113M station-days, and inference should dominate there again. This is 16
station-days of one weight on one network; the equivalent `--procs 1` profile on
a `jma_wc` campaign has not been taken.

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

## 6. CLOSED — both correctness questions, 2026-09-02

**`obs` and its missing hydrophone: not a problem.** `obs` declares
`component_order: "Z12H"` and `in_channels: 4` against a `--components` default
of `ZNE12`. Two facts settle it:

- **There is no hydrophone to fetch.** Across the 3,389 obs-campaign stations
  the band+instrument codes present are BH, CN, EH, EL, EP, HH, HN, SH, SL —
  not one pressure channel (`?D`). And the URI builder appends the component
  letter to the band+instrument code, so asking for `H` on a `BH` station
  requests `BHH`, not `BDH`; adding `H` to `--components` could not reach a
  hydrophone even where one existed.
- **SeisBench handles both gaps.** A missing component is zero-filled —
  byte-identical to supplying an explicit all-zero `H` — and `N`/`E` map onto
  the `1`/`2` slots, so the two naming conventions in the campaign (820
  `HHN`/`HHE` against 331 `HH1`/`HH2`) annotate identically. Pinned by
  `tests/test_obs_components.py`.

**`seisbench` is pinned to 0.12.5.** Unpinned, two builds of the same commit
could differ — and not only in the library. SeisBench resolves which *weight
version* to load from its own version, and the repo carries `original` at both
`.v1` and `.v2`. Those share **byte-identical `.pt` tensors** but differ in
`model_args`: `.v2` adds `norm: "std"`. So the resolved version is a property of
the picks, not just of the install. 0.12.5 resolves `original` → `.v2`,
`jma_wc` → `.v1`, `obs` → `.v1`, and is the version every 2026-09 measurement
used. On a synthetic test the two `original` variants gave identical picks, so
the exposure was latent rather than active.

**`UL` has no metadata** — listed in `ncedc.txt` but `networks/UL.zip` does not
exist, so it is absent from the queue. Still open; decide whether it belongs
before calling NCEDC complete.

## 6a. Superseded — the original statement of these questions

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

## 9. MEASURED for 31% of the campaign — and the premise was half wrong

**Surveyed 2026-09-01** with [`scripts/hitrate_survey.py`](../../scripts/hitrate_survey.py),
32 sample days 2010–2025, S3 listings only, calibrated against every completed
shard (SCEDC exact on 5/5; EarthScope ×0.831, the two shards agreeing to 0.3%).
Full results and the cost consequences: [24_cost_model.md](24_cost_model.md).

| archive | measured hit rate | share of campaign |
|---|--:|--:|
| SCEDC | 36.2% | 7.3% |
| NCEDC | 45.3% | 10.6% |
| EarthScope Open Data | 68.1% | 13.0% |
| **EarthScope restricted** | **unmeasured** | **69.1%** |

**The "rises sharply across the span" premise is wrong.** It rises on SCEDC only
(30.3% → 47.5%, 1.57×); NCEDC and EarthScope Open Data are flat across sixteen
years. What was right is that 21.7% was unrepresentative — even SCEDC in 2010
surveys at 30.3%.

**The remaining 69% needs the EarthScope refresh token**, and using it locally
risks invalidating the Secrets Manager copy: the SDK's refresh grant saves a
rotated token to local state, not back to the secret. Run the survey in-container
instead, where the token already is. The Open Data rate is not a substitute —
those eight networks are the permanent ones, which is why they sit at ~82%.

Campaign total: **~$10,800 at a 35% restricted rate, ~$16,700 if restricted
behaves like Open Data, ~$19,700 at 85%.**

The original statement of the item, kept because the reasoning below is what the
survey was built to test:

## 9a. Superseded — the cost basis is low by ~1.7x

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
| ~~0h~~ | ~~high-rate bands OOM at `--procs 4`~~ | **`quakescope_2026_highrate:1` registered at 32 GB** | — |
| **0g** | **restricted EarthScope `GetObject` is denied** | **~87% of planned station-days once western is counted. Needs an entitlement from EarthScope, not a code change** | external |
| **0f** | **Spot reclaims exhaust the 10-attempt cap** | **the fleet still decays; needs worker resubmission** | 1 day |
| ~~0e~~ | ~~a retry loop swallowed the preemption~~ | **fixed — `Preempted` is now a `BaseException`** | — |
| ~~0d~~ | ~~`--procs 4` OOMs on EarthScope at 16 GB~~ | **fixed; 102 min at `--procs 4`, no OOM. Not a clean A/B** | — |
| **0a** | **preempted workers are not replaced** | **decided by a live reclaim: must exit non-zero. Not yet implemented** | 1 h |
| ~~0b~~ | ~~EarthScope credentials unwired~~ | **withdrawn — a stale `aws` CLI, not a blocker** | — |
| 0b′ | one live restricted-network read on the current image | campaigns 3 and 5 | 15 min |
| ~~2~~ | ~~EarthScope I/O profile~~ | **done — 90.3 MB/s, 7.8x the same-day SCEDC control; not a risk** | — |
| **9** | **EarthScope *restricted* hit rate — 69% of the campaign** | **the one number left: $10.8k vs $19.7k** | subcommand + 1 job |
| ~~9~~ | ~~hit rate, SCEDC/NCEDC/EarthScope Open Data~~ | **measured — 36.2% / 45.3% / 68.1%** | — |
| 6a | `wa_min_conf` 0.5 → 0.3 if more amplitudes are wanted | 3x WA coverage for ~+40% cost — [23](23_amplitude_review.md) | decision |
| 6b | one pick in 41 disagrees 48% between short-window and whole-day WA | 0.17 ML, cause unknown — [23](23_amplitude_review.md) §3 | 2 h |
| 6 | `obs` components, seisbench pin | correctness of campaign 4 | 1 h |
| 5 | verify compaction before trusting it | analysis after the campaign | 2 h |
| 3 | `amp.wood_anderson` | ~$4,000 | days |
| 4a | add S3 gateway endpoint | free; may explain item 4 | 15 min |
| 4 | `s3.get` throughput | ~$3,000 | re-measure same-region first |
| 8 | billing baseline | knowing what it actually cost | 1 h + waiting |
| 7 | arm64 | ~1.68× on price | days |
