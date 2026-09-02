# 24 — Campaign cost, rebuilt from measurement

**Supersedes every earlier cost figure**, including the ~$16,400 and ~$11,000 in
[README.md](README.md). Those were revised repeatedly on 2026-09-01 and the
chain of multipliers behind them is no longer traceable. This is rebuilt from
raw Batch attempt durations and nothing else.

---

## The structure, and why the old one misled

The old basis was **seconds per *planned* station-day**, taken from one 2010
SCEDC shard. That conflates two independent things and hides both:

```
cost = planned_sd  x  hit_rate  x  seconds_per_PROCESSED_sdc  x  8 vCPU  x  $/vCPU-hr
                      ^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^
                      a property   a property of the CODE and
                      of the QUEUE the archive - stable, measured
```

**Seconds per processed station-day-channel is stable.** Two shards, five years
and one archive apart, agree to 2.6%:

| | processed sdc | node wall | s per processed sdc |
|---|--:|--:|--:|
| README baseline, CI 2010, procs 1 | 100 | 2,037.7 | 20.38 |
| `sc1`, CI 2015, procs 1 | 36 | 753.1 | **20.92** |

**Hit rate is not.** It is 21.7% (CI 2010), 38.7% (CI 2015), 67.6% (AK 2020) —
a 3× range, and it rises across the campaign span because station density and
data availability rise. Dividing real work by a denominator padded with cheap
misses is what made 2010 look cheap.

## Measured inputs

All on image `fe61788`, `quakescope_v3_worker:6`, 8 vCPU Fargate Spot,
2026-09-01. Node wall clock, so container start and model load are included.

| quantity | value | how |
|---|--:|---|
| SCEDC, `jma_wc`, `--procs 4` | **11.10 s** / processed sdc | `sc4`, 4 shards on one node |
| SCEDC, `jma_wc`, `--procs 1` | **20.92 s** | `sc1` |
| → `--procs 4` speedup | **1.88×** | the two above |
| EarthScope, `jma_wc`, `--procs 1` | **11.14 s** | `es1` |
| → EarthScope vs SCEDC, same procs | **1.88× cheaper** | the two above |
| inference share of wall | **74%** | `sc4` stage profile |
| `jma_wc` inference | 1.00× (1,070,899 params) | local, 2 threads |
| `obs` inference | **0.39×** (268,499 params) | local, 2 threads |
| `original` inference | **0.35×** (268,443 params) | local, 2 threads |

`jma_wc` carries `filter_factor: 2` — double the filters in every convolutional
layer, so 4× the parameters and ~2.85× the wall clock. **Campaigns 1–3 use it;
campaigns 4 and 5 do not**, and they are 31% of the planned station-days.

## Hit rate — measured, for 31% of the campaign

Surveyed 2026-09-01 with [`scripts/hitrate_survey.py`](../../scripts/hitrate_survey.py):
32 sample days across 2010–2025, S3 listings only. **Calibrated against every
shard the runs completed** — the check that it measures what the picker sees:

| archive | shards | listed / planned | picked / planned | correction |
|---|--:|--:|--:|--:|
| SCEDC | 5 | 38.7% | 38.7% | **1.000** |
| EarthScope | 2 | 82.4% | 68.5% | **0.831** |

SCEDC and NCEDC encode the channel in the object name, so a listing answers
exactly what the picker will find. EarthScope stores one object per station-day
covering all channels, so a listing proves the *station* had data but not that
the object holds the band `select_channel` chose — hence the 0.831, which the
two shards agree on to 0.3% (83.0%, 83.3%).

**Measured rates** (EarthScope already corrected):

| archive | hit rate | cross-check |
|---|--:|---|
| SCEDC | **36.2%** | 36.2% in the `western` station set independently |
| NCEDC | **45.3%** | 45.2% independently |
| EarthScope Open Data | **68.1%** | 61.2% in the `western` set |

### The year trend — the item 9 premise was wrong

| year | scedc | ncedc | EarthScope OD |
|--:|--:|--:|--:|
| 2010 | 30.3% | 41.4% | 82.9% |
| 2015 | 31.7% | 47.6% | 79.3% |
| 2020 | 38.9% | 45.5% | 83.2% |
| 2025 | 47.5% | 43.8% | 83.0% |

Item 9 assumed the hit rate "rises sharply over the 2010–2026 span". **It rises
only on SCEDC, and only 1.57×**; NCEDC and EarthScope Open Data are flat across
sixteen years. The campaign is far less year-sensitive than feared. What was
right is that the 21.7% basis was unrepresentative — even SCEDC in 2010 surveys
at 30.3%.

### What the survey could not reach

| tier | planned sd | share | hit rate |
|---|--:|--:|---|
| SCEDC | 8,194,864 | 7.3% | measured 36.2% |
| NCEDC | 11,913,192 | 10.6% | measured 45.3% |
| EarthScope Open Data | 14,717,156 | 13.0% | measured 68.1% |
| **EarthScope restricted** | **78,041,471** | **69.1%** | **unmeasured** |

The restricted tier needs the OAuth refresh token. Using it locally risks
invalidating the copy in Secrets Manager — the SDK's refresh grant saves a
rotated token to *local* state, not back to the secret — so it was not done.
**Its rate cannot be extrapolated from Open Data**: those eight networks are the
permanent ones (AK, TA, IU, II, N4, UU, UW, PB), which is exactly why they sit
at ~82%. The restricted majority are temporary deployments.

## Read-time downsampling, and what it is worth

Everything above 100 Hz is now downsampled when it is read
([17_launch_conventions.md](17_launch_conventions.md)). Only one stage changes:
`model.classify` always ran at 100 Hz because SeisBench resampled its own copy,
and `s3.get`/`mseed.parse` move the same bytes either way — but
`amplitude_extractor` ran on the stream **as read**, so a 425 Hz trace cost
4.25× what it needed to in Wood-Anderson and velocity.

**Nominal SEED rates badly understate this.** Read out of MiniSEED record
headers on 2018.041:

| band | observed mix | mean rate | wall before/after |
|---|---|--:|--:|
| `DP` | 250 Hz 30%, **500 Hz 70%** | 425 | **1.80×** |
| `CN` | **500 Hz 100%** | 500 | **1.99×** |
| `HN` | 100 Hz 72%, **200 Hz 28%** | 128 | 1.07× |
| `EH` | 100 Hz 78%, 200 Hz 22% | 122 | 1.05× |
| `HH` | 100 Hz 94%, 200 Hz 6% | 106 | 1.01× |
| `SH`, `BH` | 50 / 40 Hz | — | 1.00× (never resampled) |

The band table assumed `DP` = 250 and `CN` = 250; both are mostly 500. And `HN`
is not the flat 100 Hz it is listed as.

**Campaign effect: ~5.1%, $10,262 → $9,737.**

| campaign | before | after | saved |
|---|--:|--:|--:|
| scedc | $543 | $521 | 4.0% |
| ncedc | $955 | $883 | 7.5% |
| earthscope | $6,745 | $6,443 | 4.5% |
| obs | $43 | $42 | 2.7% |
| western | $1,976 | $1,847 | 6.5% |
| **total** | **$10,262** | **$9,737** | **5.1%** |

693,404 → 657,892 vCPU-hours.

Two caveats. The rate mix comes from small samples — 18 to 20 objects per band
on one day — so the `HN` 28%-at-200 figure in particular has wide error bars,
and `HN` is 46% of the campaign, so it is where the estimate is most sensitive.
The resample stage's own cost is **no longer assumed**: measured at **12.7% of
wall** on an all-`DP`/`CN` NCEDC shard, against the 2% allowed for here — six
times higher. That cuts the downsampling saving from 5.1% to **3.1%** and moves
the campaign to **~$9,940**. The tables above still show the 2% figures; treat
$9,940 as the current number.

Memory is the other benefit and is not in this table: the decoded stream waits
in `data_queue`, so a 500 Hz `DP` trace was occupying 5× what the model would
ever use — in the queue that put `--procs 4` over 16 GB.

## Per campaign

`--procs 4` throughout, EarthScope assumed to gain **1.4×** from it (less than
SCEDC's measured 1.88×, because EarthScope spends only 4.4% of wall in `s3.get`
against SCEDC's 16.9% and so has less stall to fill).

| campaign | weight | vCPU-hr | $ | % of cost |
|---|---|--:|--:|--:|
| scedc | `jma_wc` | 36,681 | 543 | 5.0% |
| ncedc | `jma_wc` | 65,501 | 969 | 9.0% |
| **earthscope** | `jma_wc` | 486,465 | **7,200** | **66.5%** |
| obs | `obs` | 3,383 | 50 | 0.5% |
| western | `original` | 139,608 | 2,066 | 19.1% |
| **total** | | **731,638** | **~$10,800** | |

That table takes the restricted tier at **35%**. It is the one number left, and
it moves the total more than everything else combined:

| EarthScope restricted hit rate | total |
|--:|--:|
| 35% | **$10,828** |
| 50% | $13,490 |
| 68% (= the Open Data rate) | $16,685 |
| 85% | $19,702 |

**So: ~$11,000–$20,000, and which end depends on one unmeasured number.** If the
restricted temporary deployments behave like the permanent networks, it is
~$16,700. Add ~$3,600 if EarthScope has to run at `--procs 1` because item 0d is
unfixed.

## Why this lands near the old ~$11,000 anyway

Two large corrections in opposite directions very nearly cancel, which is why
the published figure was closer to right than its derivation was:

| | |
|---|--:|
| published (4.43 s/planned-sd ÷ 1.50, SCEDC/`jma_wc` applied to everything) | $10,963 |
| **+** correct the hit rate 21.7% → 40% | $16,487 |
| **+** correct per-campaign archive and weight | **$10,465** |

The old number applied SCEDC + `jma_wc` economics to all 112.9M planned
station-days. But 60% of the campaign reads an archive that is 1.88× cheaper per
unit of work, and 31% uses a model that costs 0.35× the inference. **Do not read
the agreement as confirmation** — it is two errors cancelling, and they will not
cancel if either input moves.

## What would tighten this

1. **The EarthScope restricted hit rate.** 69% of planned station-days, and on
   its own it spans $10.8k–$19.7k. Everything else is now measured. It is the
   same `s3.list` survey, run where the refresh token already is — inside the
   container, under `quakescope_2026_earthscope:4`, which carries the secret.
   That needs a subcommand on `src.picker` (the image's ENTRYPOINT is fixed), so
   it is a small code change plus one Batch job, and it doubles as the live
   restricted-read check that [OPTIMISE.md](OPTIMISE.md) item 0b′ still wants.
   **Do not run it locally from the secret** — see the note above on rotation.
2. **EarthScope at `--procs 4`**, once item 0d is fixed. Worth ~$3,600 and
   currently a guess.
3. **A `western`/`original` shard end to end.** The 0.35× weight factor is
   measured, but on synthetic data locally, not in a campaign.

## Assumptions that are not measured

- Fargate Spot at **$0.0148/vCPU-hr**. Not published through the pricing API and
  Cost Explorer is blocked by an SCP ([OPTIMISE.md](OPTIMISE.md) item 8), so the
  whole table scales linearly with a rate nobody has confirmed.
- Amplitude settings unchanged. Lowering `wa_min_conf` to 0.3 multiplies
  everything here by ~1.40, to 0.2 by ~1.96 —
  [23_amplitude_review.md](23_amplitude_review.md).
- The classifier stays out.
