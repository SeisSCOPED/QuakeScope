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

## Per campaign

Central case: **hit rate 40%**, EarthScope gaining **1.4×** from `--procs 4`
(less than SCEDC's 1.88×, because EarthScope spends only 4.4% of wall in
`s3.get` against SCEDC's 16.9% and so has less stall to fill).

| campaign | weight | planned sd | s/processed | vCPU-hr | $ | % of cost |
|---|---|--:|--:|--:|--:|--:|
| scedc | `jma_wc` | 4,106,669 | 11.10 | 40,532 | 600 | 5.7% |
| ncedc | `jma_wc` | 5,979,675 | 10.88 | 57,838 | 856 | 8.2% |
| **earthscope** | `jma_wc` | 67,983,975 | 7.96 | 480,781 | **7,116** | **68.0%** |
| obs | `obs` | 996,536 | 4.36 | 3,866 | 57 | 0.5% |
| western | `original` | 33,799,828 | 4.13 | 124,057 | 1,836 | 17.5% |
| **total** | | **112,866,683** | | **707,074** | **~$10,500** | |

Only the `scedc` row is fully measured. `ncedc` assumes SCEDC economics less the
cross-region penalty; `earthscope` and `western` assume the 1.4× and, for
`western`, that its stations are predominantly EarthScope-routed (they are —
~21k of 24.1k).

## Sensitivity — the two things that actually move it

|  | EarthScope at `--procs 1` | 1.4× | 1.88× |
|---|--:|--:|--:|
| hit rate 25% | $8,793 | $6,540 | $5,103 |
| 30% | $10,551 | $7,849 | $6,123 |
| **40%** | $14,068 | **$10,465** | $8,165 |
| 50% | $17,585 | $13,081 | $10,206 |
| 60% | $21,102 | $15,697 | $12,247 |
| 70% | $24,619 | $18,313 | $14,288 |

**Range $5,100–$24,600.** The left column is today's reality for campaigns 3
and 5, because EarthScope OOMs at `--procs 4` ([OPTIMISE.md](OPTIMISE.md)
item 0d) — fixing that is worth roughly $3,600 at a 40% hit rate.

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

1. **The hit rate, by campaign and era.** It is the dominant term and it is
   `objects that exist ÷ station-days planned` — answerable with `s3.list`
   alone, no picking, in about an hour. This is the single highest-value
   measurement left.
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
