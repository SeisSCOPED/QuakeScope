# 18 — Launch readiness: western states

State of the campaign 5 launch. What is provisioned, what it should cost, what
is still unknown, and the two decisions that are not ours to make.

Briefing deck for the non-operational audience:
[seisscoped.org/QuakeScope/launch_2026.html](https://seisscoped.org/QuakeScope/launch_2026.html).

---

## Provisioned and verified

| | | verified by |
|---|---|---|
| Bucket | `s3://quakescope-picks-2026`, us-east-2 | created; public access blocked; versioning off |
| Campaign prefix | `s3://quakescope-picks-2026/western` | — |
| Job definition | `quakescope_v3_worker:2` | image pinned to `9abd01c`; no `--classifier` |
| Compute env | `niyiyu_earthscope`, FARGATE_SPOT, maxvCpus 4000 | ENABLED |
| Job queue | `niyiyu_earthscope_missing_station` | ENABLED / VALID |
| Task role | `SeisBenchBatchRole` | S3 write to the new bucket confirmed by policy simulation |
| Station list | `sb_catalog/configs/networks/western_states.csv` | 24,111 stations, 122 networks, true state polygons |
| Cost alerts | `QuakeScopeAWSWatch` via GitHub OIDC | workflow run succeeded; dashboard issue #30 open |
| Notebook | `notebooks/7_launch_western_states.ipynb` | cells parse; preflight checks image pin and classifier flag |

The image was confirmed able to run the worker on Fargate — a probe exited
cleanly after the v3 rebuild, where every earlier attempt exited 1 on the
`EARTHSCOPE_S3_ACCESS_POINT` import.

**The queue has deliberately not been written.** `shards.jsonl` is immutable
once created, so writing it is the point of no return, and it depends on both
decisions below.

## Sizing

| | |
|---|--:|
| Stations | 24,111 |
| Networks | 122 |
| Date range | 2010.001 – 2026.001 |
| Shards | 72,441 |
| Station-days | **33,796,194** |
| Estimated compute | ~319,000 vCPU-hours |
| Estimated cost | **~$4,700** at the estimated Fargate Spot rate |

Cross-checked: the planner's 33,796,194 against an independent sum of clipped
operating windows at 33,776,383 — 0.06% apart.

These are station-days, and with one channel per station they are also band-days, which is what the cost scales with. Before the channel policy the same campaign was 42.7M band-days.

**This estimate was 4.2× higher three hours ago.** The planner built the full
station × day cartesian product, ignoring operating windows: 140.9M station-days
and 176,679 shards, about $19,700 of planned compute. The excess was not free —
every phantom station-day still costs an S3 listing before the picker discovers
there is nothing there. Fixed, and the fix is what the numbers above reflect.

## Blocker: EarthScope access is not working

**86.7% of the western-states stations are served by EarthScope** — 20,900 of
24,111, across the temporary deployments (`XD`, `ZI`, `ZG`, `1D`, …) that
dominate the set. Only 2,095 route to NCEDC and 1,116 to SCEDC.

Two separate things are missing, and both are outside this repository:

**1. The account cannot assume the S3 role.** A live credential exchange from
this machine returns:

```
UnauthorizedError: {"detail":"You are not allowed to assume role 's3-miniseed'"}
```

The EarthScope login itself is fine — the SDK returns the profile for
`mdenolle@uw.edu` — so this is an entitlement on EarthScope's side, not an
expired token. **EarthScope data services have to grant the `s3-miniseed`
role.**

**2. `EARTHSCOPE_S3_ACCESS_POINT` has never been set.** It is `""` in every
commit in the repository's history, which is correct — it is account-specific
and belongs in the controller's `parameters.py`, not in git. Per
[05_submitting_jobs.md](05_submitting_jobs.md) it looks like
`es-miniseed-<...>-s3alias`, and the value should exist in the 2025 campaign's
notes.

**Partial relief:** EarthScope's sponsored open-data bucket
`s3://earthscope-geophysical-data` (us-east-2) is anonymous and needs neither
the role nor the access point — but it holds **8 networks**, covering 1,392 of
the 20,902 EarthScope-routed western stations (6.7%). See
[19_earthscope_access.md](19_earthscope_access.md).

Until both are resolved:

- **Campaign 5 (western states)** can only cover the 3,211 SCEDC/NCEDC
  stations — 13% of the intended deliverable.
- **Campaigns 3 and 4** (EarthScope onshore and offshore, 45.1M station-days,
  52% of the launch) cannot run at all.
- **Campaigns 1 and 2** (SCEDC, NCEDC — 7.0M station-days) are unaffected and
  could run tomorrow.

A campaign that needs EarthScope now **fails at shard startup** with the station
count and the networks involved, rather than raising `KeyError('earthscope')`
once per station. That was the previous behaviour, and it read like a defect in
the reader rather than a misconfigured campaign.

## Two decisions that are not ours

**1. The weight substitution is stakeholder-facing.**
[09](09_western_states_run.md) specifies `instance`;
[11](11_launch_plan.md) supersedes it with `original`. The evidence for the
change is strong — on three minutes of Ridgecrest aftershocks at a shared
threshold, `instance` returns 4 picks where `original` returns 26, a ceiling no
threshold recovers. But this is a deliverable to an external group, and the
substitution changes what they receive. **Confirm before the queue is written.**

**2. Western is the largest campaign, not the smallest.** 33.8M station-days is
65% of the launch total, and it is *additional* work rather than a subset — it
re-picks CA/NCEDC/SCEDC stations with different weights. v3's largest run to
date is four shards. SCEDC, at 2.47M station-days, would prove the same
machinery for about 7% of the exposure.

## Run order

1. **Preflight** (notebook §1) — image pinned, `--classifier` absent, queue and
   compute environment enabled.
2. **Stations** (§2) — writes `stations.parquet`.
3. **Plan** (§3) — prints shard and station-day counts. Read them. The write
   cell is separate and is the irreversible step.
4. **Smoke test** (§4) — `--max-shards 2 --profile`, leaving the rest of the
   queue untouched. Look at the picks, not just the exit code.
5. **Launch** (§5) — 25 workers × 8 vCPU = 200 vCPU against a 4000 cap. Scale
   up once throughput is known; workers are stateless and adding more needs no
   cleanup.
6. **Watch** (§6) and the hourly alerts.

## Stopping

Cancelling loses at most 40 station-day-channels per worker: each shard returns
to the queue and resumes from its last checkpoint. Fargate tasks end with the
job, so there is no instance to terminate — unlike the SkyPilot path, where the
jobs controller survives `sky down --all`.

```python
for jid in json.load(open("western_workers.json")):
    batch.cancel_job(jobId=jid, reason="operator stop")
```

## Known unknowns

Listed because they are what would move the numbers above, not as caveats.

- **Processes per vCPU.** Every benchmark used one process on an eight-vCPU
  box. This swings the cost estimate about 4× in either direction and is an
  hour of work to settle. It is the single highest-value measurement left.
- ~~Bands per station.~~ **Resolved.** One channel code per station-location,
  by the hard-coded order in `constants.CHANNEL_PRIORITY` — see
  [17](17_launch_conventions.md). Western goes from 30,453 band-days to 24,111
  (1.26×); the saving is larger on the permanent-network campaigns.
- **The 2025 baseline.** Cost Explorer is blocked on this account by an
  organisation SCP, so "match or beat 2025" has no measured baseline. A
  billing-console export would settle it.
- **Scale.** v3 has completed four shards end to end. Everything about its
  behaviour at 72,441 is inference from those four plus the claim-protocol
  tests.
- **The Fargate throughput advantage is n=1.** 33.7 s against 53.7 s on
  identical code, almost entirely in inference. Worth repeating before it is
  used to justify anything.

## What was found by running rather than reviewing

Recorded because it is the reason for the smoke-test discipline above. Every
one of these was invisible to inspection and surfaced only on real
infrastructure:

- an inclusive end against `np.arange` silently dropping the last day of every
  shard — 5% of a campaign;
- the Parquet key defaulting to `HOSTNAME`, so every shard on a node overwrote
  the last;
- an unbounded credential retry that hung forever while holding a claim;
- a job reporting SUCCEEDED after failing every shard;
- a stale-claim takeover that was not atomic, so two workers could run the same
  shard;
- the planner's 4.2× overstatement above.
