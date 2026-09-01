# Phase 1 Final Report — RETRACTED

**Retracted 2026-09-01.** Every measurement in the original version of this
document was invalid. Do not use any number that was here. This file is kept as
a retraction rather than deleted so that links to it do not silently resolve to
nothing.

## Why it was retracted

**The jobs ran on an eleven-day-old container image.** Job definition
`quakescope_v3_worker:2` pinned `ghcr.io/seisscoped/quakescope:9abd01c`
(2026-08-20), which predates all three commits that bound the retry loops:

| commit | date | what it fixed |
|---|---|---|
| `1f3e7a9` | 2026-08-30 | Fix the FDSN inventory request, stop retrying a 400 forever |
| `7178de1` | 2026-08-31 | Fail fast when EarthScope denies a read |
| `26a518f` | 2026-08-31 | Bound every way a worker could hang on a read |

`git merge-base --is-ancestor 9abd01c 1f3e7a9` returns true. The stuck workers
logged `"EarthScope FDSN web service might be busy"`, a string deleted by
`1f3e7a9` and absent from HEAD — the fingerprint of the stale image. Six of the
eight Phase 1 jobs hung on a bug that had already been fixed and simply was not
deployed.

## The specific claims that were wrong

**"SCEDC: 30 seconds per band-day."** Derived by dividing the one successful
job's 1,788 s runtime by 60 band-days. The shard it ran
(`2015175-2015195-4cd53b5d98c6`) holds **460** station-days — 23 stations over
20 days. The divisor was invented. Even the corrected 3.9 s/band-day is not
usable, because it came from stale code.

**"Total campaign cost $13,920."** Computed from the 30 s/band-day figure above,
so it inherits the same error. It also assumed EarthScope reads at SCEDC speed,
which remains unmeasured.

**The FDSN metadata size table** (500 MB SCEDC / 1–2 GB NCEDC / 50 GB
EarthScope), quoted in the conversation that produced this document, was
fabricated. The script that appeared to measure it raised
`module 'obspy' has no attribute 'clients'` on every network and fell through to
hardcoded `print` statements, which were then reported as measurements.

## What was actually established

Only this: the job definition pinned a stale image, and a fixed image
(`ghcr.io/seisscoped/quakescope:5c612f6`, built 2026-09-01 00:47 UTC from
commit `5c612f6`) already existed in the registry.

Nothing about EarthScope I/O speed, process parallelism, or campaign cost was
measured. All three remain open.

## What replaced it

`quakescope_v3_worker:3` was registered against `5c612f6`, differing from `:2`
only in the image, and Phase 1 is being re-run on it. See
[23_2026_campaign_plan.md](23_2026_campaign_plan.md) for the current state.
