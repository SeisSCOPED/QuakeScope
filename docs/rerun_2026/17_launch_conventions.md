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
| Compute environment | `niyiyu_earthscope` (FARGATE_SPOT, maxvCpus 12000) |
| Job queue | `niyiyu_earthscope_missing_station` |
| Fargate Spot quota | 12,000 vCPU (`L-36FBB829`) |
| Job definition | `quakescope_v3_worker` |
| Image | pin the **short-SHA tag**, never `:latest` |

**Pin the tag.** `:latest` moves when someone pushes to `main`, and a task that
starts tomorrow would run different code from one that started today.

A job is a **worker, not a shard**: it claims from the queue until the queue is
empty, so you choose how many workers to run rather than submitting one job per
unit of work. `maxvCpus` and the account quota are **both 12,000**, so neither
binds first — at 8 vCPU per worker that is 1,500 concurrent workers, which is
exactly the campaign target with no headroom above it.

Note the **on-demand** quota is a different and much smaller number:
`L-3032A538` is **140 vCPU**, i.e. 17 workers. That matters only if an
on-demand fallback compute environment is ever added
([OPTIMISE.md](OPTIMISE.md) item 0f-plan).

## Channels: one code per station-location

The 2025 study processed `EH? HH? BH? HN? EP? DP? EL? SL? SH? CN?`, recording
each location code and channel type separately. **2026 keeps the location-code
separation but picks one channel code per station-location**, always with all
available components (`Z N E 1 2`).

Picking every band duplicates the same ground motion at different sampling
rates — 2.83× on SCEDC's permanent stations, 1.26× across the western set — and
sweeps in bands like `LH` at 1 Hz that cannot produce a usable arrival at all.

The order is **hard-coded** in `constants.CHANNEL_PRIORITY`, not derived at read
time. SEED band codes carry standard sampling-rate ranges, so the ranking is a
property of the code; rediscovering it per station-day would cost a metadata
lookup to re-derive a constant.

Ordered by |standard rate − 100 Hz|, since PhaseNet resamples to 100 Hz —
**except accelerometers, which sort last whatever their rate**:

| | code | instrument | nominal | observed |
|--:|---|---|--:|--:|
| 1 | `HH` | high-gain seismometer | 100 | 106 |
| 2 | `EH` | high-gain short-period | 100 | 122 |
| 3 | `SH` | high-gain short-period | 50 | 50 |
| 4 | `BH` | high-gain broadband | 40 | 40 |
| 5 | `DP` | geophone (nodal) | 250 | **425** |
| 6 | **`HN`** | **accelerometer** | 100 | **128** |
| 7 | **`CN`** | **accelerometer** | 250 | **500** |

`EP`, `EL` and `SL` were removed from the list entirely — see below. The
observed column is measured from MiniSEED record headers; the nominal figures
the ordering was built on understate three of the seven.

**Corrected 2026-09-02 by M. Denolle.** `HN` previously sat 5th, above `SH` and
`BH`, because it is a 100 Hz code and the ordering was on rate proximity alone.
That is wrong: **an accelerometer should be picked only when the station offers
no seismometer or geophone at all.** Accelerometers do not clip on large events,
but their SNR on the small ones that dominate a catalogue is poor — so picking
one where a broadband exists loses exactly the events the catalogue is built to
find. Rate proximity is a resampling convenience; instrument class is a property
of the data.

Scale of the change, measured across all five campaigns: **139
station-locations, 528,758 station-days (0.47%)** — 460,279 that were taking
`HN` over an available `BH`, and 68,479 over an `SH`. Nothing else moves.

The earlier note dismissed this as "10 of 24,111 western-states stations",
which counted one campaign and only the `HN`+`BH` case.

**No queue rewrite is needed.** The band is chosen at read time in
`s3_helper.load_waveforms`, not stored in `shards.jsonl` — the shard schema is
`shard_id, stations, start, end, n_station_days` — so the correction takes
effect on the next worker run and the immutable queues stand.

### `EP`, `EL` and `SL` are not pickable at all (2026-09-02)

Deep-learning phase pickers are trained on `BH`, `HH` and `EH`. Picking on a
band no training set contains is out of distribution rather than merely
second-best, so a station offering nothing else is **skipped**, not picked
badly. Cost: 3,631 stations, 344,778 station-days, **0.31%** of the campaign.
Another 379,633 station-days move onto a band that is in the training set.

`SH` stays — the same instrument as `EH` at a lower rate, and BK uses it.

`DP` stays, with a caveat rather than a change: only the **2014** nodal
deployment is in miniSEED, the rest of EarthScope's nodal archive is still PH5.
Most `DP` station-days should therefore *miss at read time* rather than pick,
and the plan is consistent with that — 2,092,180 of 3,425,226 planned `DP`
station-days (61%) belong to deployments starting in 2014.

`CN` is SEED band `C` (250–1000 Hz) with instrument code `N`: a high-rate
accelerometer. It sits last, so it is picked only where there is no `HN` either.

### Everything above 100 Hz is downsampled at read time (2026-09-02)

All three 2026 weights declare `sampling_rate = 100`. Traces recorded faster are
brought down to 100 Hz **in `_read_waveform_from_s3`**, before the stream
reaches the queue:

| band | nominal | station-days | |
|---|--:|--:|---|
| `DP` | 250 Hz | 3,636,370 | 2.5× fewer samples |
| `CN` | 250 Hz | 485,692 | 2.5× fewer samples |

**3.7% of the campaign for certain**, plus any `HN` running at 200 Hz — `AK.PS09`
measures 200, not the 100 the band table assumes, so the real figure is higher.

Three reasons to do it at read time rather than leave it to `annotate`:

- **Memory** — the decoded stream waits in `data_queue`, and that queue is what
  put `--procs 4` over 16 GB ([OPTIMISE.md](OPTIMISE.md) item 0d).
- **Amplitude cost** — `annotate` resamples its own copy, but
  `amplitude_extractor` runs on the stream as read, so without this the
  Wood-Anderson and velocity stages process 2.5× more samples than the picks
  were made on.
- **Comparability** — amplitudes are then measured on uniformly 100 Hz data.

**The picks do not change.** `downsample_to_target` calls SeisBench's own
resampler rather than reimplementing it, including its `zerophase=True` default
— the `zerophase_resample` docstring warns that a different filter in
application than in training causes out-of-distribution issues. Verified on
synthetic 200, 250 and 500 Hz data through the real model: pick phases, times
and probabilities identical, both branches of the resampler (integer ratio →
lowpass + decimate, non-integer → FFT). `tests/test_downsampling.py` pins it.

It only ever downsamples. A 40 Hz `BH` trace is left alone: upsampling at read
time would inflate the very queue this is meant to relieve, and `annotate` does
it anyway on a copy that is discarded straight after.

Ties inside the 100 Hz group break on instrument code, by signal quality for
small events: high-gain seismometer (`H`) > geophone (`P`) > low-gain (`L`) >
accelerometer (`N`). Accelerometers do not clip on large events but have poor
SNR on the small ones that dominate a catalogue.

**Two orderings are debatable and were left as the rule states:**

- `HN` (100 Hz accelerometer) outranks `BH` (40 Hz broadband). Better for large
  events, worse for small. It affects **10 of 24,111** western-states stations —
  those carrying both and no 100 Hz high-gain channel.
- `BH` (40 Hz) outranks `DP` (250 Hz). Downsampling 250→100 is lossless across
  the picking band while upsampling 40→100 is not, so `DP` is arguably the
  better choice; |Δ100| says otherwise. Stations carrying both are rare.

Chosen channels across the western set: `DP` 8,548, `HN` 5,128, `EH` 4,354,
`HH` 2,396, `BH` 1,430, `EP` 1,355, `EL` 530, `SH` 294, `CN` 75, `SL` 1. No
station was left without a pickable channel.

Unchanged from 2025: waveforms that are empty, embargoed, or carry more than
50 gaps per component are skipped (`picker.py`, `len(stream_c) > 150`).

## Pick threshold

**`p_threshold = s_threshold = 0.2`** for every 2026 campaign — the default in
`worker.py` and `picker.py`, so a launch that passes no threshold argument gets
it. The launch notebook passes none.

Not to be confused with the **0.3 used across the benchmark notebooks**
(`tutorials/`, `reports/`). That is a *shared comparison* threshold, chosen so
weight sets are ranked at one operating point; the published figures and their
narrative depend on it. Changing the production threshold does not change it,
and it should not be edited to match.

## Weight per campaign

| campaign | weight | why |
|---|---|---|
| scedc, ncedc, earthscope | `jma_wc` | best recall/MCC of the fine-tunes |
| obs | `obs` | `PickBlue(base="phasenet")` |
| **western** | **`original`** | see below |

**Western states uses `original`, superseding
[09_western_states_run.md](archive/09_western_states_run.md), which specified
`instance`.** `instance` has a genuine ceiling on dense near-field aftershock
sequences: at Ridgecrest it emits 246 S picks with its threshold on the floor
where the others reach 684 and 832, and no threshold recovers it. This is a
stakeholder deliverable, so the substitution needed their agreement:
**confirmed 2026-08-29 by M. Denolle.** The `quakescope_2026_western` job
definition carries `original` as its default weight.

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
[09](archive/09_western_states_run.md) explicitly warns against. Nevada is the tell: a
naive per-state rectangle scheme assigns it 3 stations, because California's
rectangle covers it.

At the measured ~34 s per band-day on Fargate, western alone is roughly
**319,000 vCPU-hours ≈ $4,700** at the estimated Spot rate — with the caveats in
[16_skypilot_vs_fargate.md](16_skypilot_vs_fargate.md): processes per vCPU is
unmeasured and swings this about 4×.

**Western is the largest single campaign**, 65% of the launch total, and it
overlaps campaigns 1–3 geographically — it re-picks those stations with
different weights, so it is additional work rather than a subset.
