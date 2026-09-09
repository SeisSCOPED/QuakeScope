# 24 — Campaign cost, rebuilt from measurement

**Supersedes every earlier cost figure**, including the ~$16,400 and ~$11,000 in
[README.md](README.md). Those were revised repeatedly on 2026-09-01 and the
chain of multipliers behind them is no longer traceable. This is rebuilt from
raw Batch attempt durations and nothing else.

---

## 2026-09-09: re-priced at the billed rate, with three campaigns measured outright

Everything below this section prices vCPU-hours at **$0.0148**, a Spot list
rate nobody had checked. Two things have happened since. A CloudBank bill
arrived and gave a real rate, and the western, obs and obs-early campaigns ran
to completion, so their cost is now a measurement and not a model. Multiply any
dollar figure further down by **1.44** to bring it to the billed rate; the
vCPU-hour figures stand or fall on their own, and for western they fell.

### The rate is $0.0213 per vCPU-hour, not $0.0148 and not $0.0313

`costs_actual.json` holds the CloudBank daily breakdown for the campaign
window. Net of the standing account baseline, a GPU instance that is not ours
and one unattributable day, the three days that carried real load bill
**$2,316.89** against **108,658 vCPU-hours** measured over the same days:

| | |
|---|--:|
| billed, 09-03 + 09-04 + 09-06, net of baseline and exclusions | $2,316.89 |
| vCPU-hours on those days, every Batch attempt summed | 108,658 |
| **all-in rate** | **$0.02132 / vCPU-h** |
| Fargate on-demand list for the 8 vCPU / 16 GB shape (vCPU + memory) | $0.04937 |
| discount against list | 57% |

All-in means memory, log ingestion, requests and public IPv4 are inside it,
because the task shape is fixed and they scale with vCPU-hours. The $411 IPv4
line estimated further down is therefore already counted. Change the task
shape and the rate has to be re-derived.

**$0.0313 was published as "calibrated" from 2026-09-07 to 09-09 and was 1.47x
too high.** `campaign_spend.py` summed each job's own `startedAt`/`stoppedAt`,
and Batch overwrites those on every retry: a job Spot reclaimed three times
reports the span of its fourth attempt and nothing else. `global-477782054`
ran four attempts for 17.16 hours; its job-level span was 1.04. On the
calibration days that undercounted usage by 1.47x (74,030 against 108,658
vCPU-hours), and dividing a correct bill by too few hours inflated the rate by
exactly that factor. Fixed the same day; `tests/test_spend_classification.py`
pins attempt-level accounting. This is the third time an estimate on this
account was wrong by a bookkeeping miss rather than a price (array children,
memory, now attempts), and each was invisible from the console.

### What has actually been spent, by campaign

Attempt-level vCPU-hours from Batch, priced at $0.02132. Shard counts and
station-days from the campaign's own S3 state on 2026-09-09.

| campaign | shards complete | planned sd in those shards | processed sd | hit | vCPU-h | $ |
|---|--:|--:|--:|--:|--:|--:|
| western | 72,448 / 72,505 | 33,780,833 | 7,730,960 | 22.9% | 86,879 | **1,852** |
| western-2026 | 3,219 / 3,237 | 1,555,620 | 411,472 | 26.5% | 3,444 | 73 |
| obs | 6,231 / 6,566 | 976,317 | 520,114 | 53.3% | 8,563 | 183 |
| obs-early | 3,631 / 3,631 | 407,368 | 155,101 | 38.1% | 1,178 | 25 |
| global | 7,344 / 202,468 | 2,639,157 | 629,967 | 41.9% readable | 14,387 | 307 |
| dry runs, surveys, tests | | | | | 96 | 2 |
| **total** | | | | | **114,547** | **2,442** |

The rest of western and western-2026 is 54 embargoed LH shards (blocked, cost
nothing) and 21 shards stuck on a guard bug fixed in PR #33; obs's remaining
335 are blocked the same way. Global's 41.9% is on network-years EarthScope
serves; counting the 1.14M planned station-days in network-years it does not
hold, the raw figure is 23.9%, and those cost nothing to discover.

**Western is the check on the model.** The table further down predicted
western at 139,608 vCPU-hours and $2,066. It took 86,879 vCPU-hours and cost
$1,852. The hours were overpredicted 1.61x and the rate underpredicted 1.44x,
so the dollars agree to 10% for the second time by cancellation, and for the
second time that agreement should not be read as the model being right.

### Unit costs, measured

Each completion record carries the shard's wall time. Four shards run at once
on an 8 vCPU task, so a perfectly packed task spends `seconds / 4 x 8` vCPU-seconds
per shard; the ratio of what Batch actually billed to that ideal is the
overhead of the campaign as run: workers that started and found nothing,
attempts killed mid-shard, the 09-05 embargo hour, OOM kills.

| campaign | weight | shard-seconds per processed sd | packed vCPU-h per processed sd | overhead as run | all-in vCPU-h per processed sd | $ per processed sd |
|---|---|--:|--:|--:|--:|--:|
| western | `original` | 11.5 | 0.0064 | 1.77x | 0.0112 | $0.00024 |
| obs | `obs` | 13.5 | 0.0075 | 2.20x | 0.0165 | $0.00035 |
| global | `jma_wc` | 23.6 | 0.0131 | 1.74x | 0.0228 | **$0.00049** |
| western-2026 | `original` | 12.7 | | 1.19x | | |
| obs-early | `obs` | 11.0 | | 1.25x | | |

Two things are worth taking from it. `jma_wc` costs 2.06x `original` per
processed station-day, close to the 2.3x the parameter count predicted. And
the two campaigns that ran after the September fixes, with no incident, carried
about 1.2x overhead; the three that ran through the incidents carried 1.7 to
2.2x. Global's 0.0228 agrees with the dashboard's independently measured
0.0241 to 5%.

### Global, projected from measured hit rates

Global and western share their station-days for CI, NC, UW, UU, BK, NN, PB, TA,
US, GS and IM: western is a geographic re-pick of the same data with different
weights. So western's per-network hit rates are a measurement of 31% of
global's planned station-days, and the 7,344 completed global shards measure
another 5%. Where neither applies, the tier survey rate stands (SCEDC 36%,
NCEDC 45%, Open Data 68%), the global sample's 41% is used for restricted
network-years, and GeoNet is taken at 80%. NP, the strong-motion network, is
**20.4M planned station-days, a quarter of global, and unmeasured**; it is
triggered data and assumed 5% here.

| tier | planned sd | in network-years EarthScope lacks | projected processed | effective hit | share measured |
|---|--:|--:|--:|--:|--:|
| EarthScope restricted | 55,958,302 | 2,512,399 | 14,196,300 | 26.6% | 57% |
| EarthScope Open Data | 11,750,743 | 0 | 7,074,323 | 60.2% | 65% |
| NCEDC | 6,297,425 | 0 | 2,665,715 | 42.3% | 100% |
| GeoNet | 4,526,684 | 0 | 3,621,347 | 80.0% | 0% |
| SCEDC | 4,300,919 | 0 | 1,491,367 | 34.7% | 100% |
| **global + global-2026** | **82,834,073** | **2,512,399** | **29,049,052** | **36.2%** | |

At $0.00049 per processed station-day as global has run so far, and $0.00034
if it runs as cleanly as obs-early and western-2026 did:

| NP hit rate | processed sd | clean (1.2x) | as run (1.74x) |
|--:|--:|--:|--:|
| 0% | 28.0M | $9,400 | $13,600 |
| **5%** | **29.0M** | **$9,800** | **$14,100** |
| 40% | 36.2M | $12,200 | $17,600 |

Of that, $307 is spent. The old table's `scedc + ncedc + earthscope` at the
billed rate would be $12,550 at a 35% restricted hit rate and $24,000 at 68%;
the measured hit rates on the biggest networks run below the survey (UW 45%
and TA 44% against Open Data's 68%; US 28%; GS 17%), which is why the
projection lands lower. The one number still worth measuring is NP, and after
it the 23M restricted station-days that nothing has sampled: moving their 41%
to 25% or 60% moves the total by about $1,500 either way.

### The whole campaign

| | spent | to go | basis |
|---|--:|--:|---|
| western + western-2026 | $1,925 | ~$10 | measured; 21 shards left |
| western-early (28.3M sd, pre-2010) | 0 | $1,000 to $1,600 | western's cost per planned sd; hit rate before 2010 unmeasured |
| obs + obs-early + obs-2026 | $208 | ~$1 | measured; obs-2026 is 15,500 sd |
| global + global-2026 | $307 | $9,500 to $13,800 | above, NP at 5% |
| dry runs, surveys, tests | $2 | | |
| **total** | **$2,442** | **$10,500 to $15,400** | **$13,000 to $18,000 all-in** |

The central figure this document used to carry, $10,500 with a $5,100 to
$24,600 range, becomes **about $15,500, in $13,000 to $18,000**. It moved up
because the billed rate is 1.44x the assumed one and western's actual cost was
within 10% of its estimate; it did not move up further because global's
measured hit rates are lower than the survey's and NP is assumed nearly empty.
The dashboard's plan panel prices at this rate from the same change.

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

> **Priced at $0.0148. Multiply by 1.44 for the billed rate (see the 2026-09-09
> section above), and note that western's measured 86,879 vCPU-hours are 1.61x
> below the figure in this table.**

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

## A cost line that is not compute

Since February 2024 AWS charges **$0.005/hour per public IPv4 address**, and
Fargate task ENIs with `assignPublicIp: ENABLED` — which these are — each hold
one. The compute environment has no NAT and no VPC endpoints, so public IPs are
how tasks reach ECR, S3 and EarthScope's FDSN service at all.

| | |
|---|--:|
| worker-hours (657,892 vCPU-hr ÷ 8 vCPU) | 82,236 |
| public IPv4 at $0.005/hr | **$411** |
| against ~$9,940 of compute | **+4.1%** |

Not worth re-architecting away — going private needs a NAT gateway, and NAT data
processing at $0.045/GB across ~1.1 PB of reads would cost far more — but it
belongs in the total. **2026-09-09: it is inside the billed all-in rate above,
so do not add it again.** Cost Explorer is blocked on this account by an
organisation SCP; the figure came from the CloudBank breakdown instead.

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

- ~~Fargate Spot at **$0.0148/vCPU-hr**.~~ **Measured 2026-09-09 at $0.02132
  all-in** from the CloudBank breakdown; see the section at the top. The tables
  below still carry $0.0148 and scale by 1.44.
- Amplitude settings unchanged. Lowering `wa_min_conf` to 0.3 multiplies
  everything here by ~1.40, to 0.2 by ~1.96 —
  [23_amplitude_review.md](23_amplitude_review.md).
- The classifier stays out.
