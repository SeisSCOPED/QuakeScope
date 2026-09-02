# 23 — Amplitude review: taper, threshold, whole-day deconvolution

Three questions asked on 2026-09-01, answered by measurement on one real
station-day with a real response: **AK.A19K 2020.309**, BH horizontals at 50 Hz,
41 picks from an STA/LTA trigger, response from EarthScope FDSN.

Reference throughout is the **whole-day deconvolution** — one detrend, a 60 s
taper, one `remove_response` over 86,400 s, then Wood-Anderson simulation. That
is the method the per-pick short window replaced.

Conventions (WA damping, component rule, `pre_filt`) are in
[`../amplitude_conventions.md`](../amplitude_conventions.md) and are not
revisited here.

---

## 1. Tapering — not a problem, and the fix would be *more* taper

`extract_amplitudes` deconvolves a 70 s window per pick with a hardcoded 15%
taper, so 10.5 s a side, leaving a 49 s core for a 13 s measurement window.

**Window length, against the whole-day reference:**

| window | taper each side | median | p10 | p90 |
|--:|--:|--:|--:|--:|
| **70 s** | 10.5 s | **1.0010** | 0.9953 | 1.0072 |
| 140 s | 21 s | 1.0001 | 0.9994 | 1.0008 |
| 300 s | 45 s | 1.0000 | 0.9999 | 1.0001 |
| 600 s | 90 s | 1.0000 | 1.0000 | 1.0000 |

**Taper fraction, holding the window at 70 s:**

| fraction | taper each side | core | median | p10 | p90 |
|--:|--:|--:|--:|--:|--:|
| 5% | 3.5 s | 63.0 s | 1.0023 | 0.9881 | 1.0167 |
| 10% | 7.0 s | 56.0 s | 1.0011 | 0.9932 | 1.0100 |
| **15%** | 10.5 s | 49.0 s | **1.0010** | 0.9953 | 1.0072 |
| 25% | 17.5 s | 35.0 s | 1.0006 | 0.9970 | 1.0045 |
| 35% | 24.5 s | 21.0 s | 1.0004 | 0.9978 | 1.0033 |

**At the current setting the spread is ±0.5%, which is 0.002 ML.** Nothing in
the magnitude scale is limited by this.

Two things worth noting anyway:

- **More taper is monotonically better, not worse.** The intuition that a taper
  "eats" the signal is wrong here, because the guard in `_peak_from_arrays`
  already rejects any pick whose measurement window reaches into the taper. What
  the taper actually does is suppress the edge discontinuity that makes the
  deconvolution ring, so a longer one is cleaner. Raising the fraction from 15%
  to 25% halves the spread and costs nothing — the core is still 35 s against a
  13 s measurement window. **A cheap, safe change if anyone wants it.**
- Do not go below 10%. At 5% the spread triples.

## 2. Confidence threshold — 0.3 is the good deal, 0.2 is not

Measured on the picks these runs actually wrote (`_iotest2`, image `fe61788`):

| threshold | AK 2020 picks kept | CI 2015 picks kept |
|--:|--:|--:|
| 0.2 (= all) | 100.0% | 100.0% |
| 0.3 | 51.8% | 51.9% |
| 0.4 | 29.0% | 27.3% |
| **0.5 (current)** | **17.2%** | **15.4%** |
| 0.7 | 6.0% | 4.4% |

Confidence percentiles are near-identical on both archives — median 0.306, p90
0.58–0.61 — so this shape is a property of the model, not of a network.

**The gate is the only thing limiting WA coverage.** Of 6,626 picks at
conf ≥ 0.5, **6,625 got an amplitude**. Data quality rejects essentially
nothing; `wa_min_conf` rejects 83%.

`amp.wood_anderson` is 18–20% of shard wall clock, and its cost is linear in
qualifying picks:

| threshold | picks vs now | amp stage | **total shard cost** |
|--:|--:|--:|--:|
| 0.5 (current) | 1.0× | 1.0× | 1.00× |
| 0.4 | 1.7× | 1.7× | **~1.14×** |
| **0.3** | **3.0×** | **3.0×** | **~1.40×** |
| 0.2 | 5.8× | 5.8× | ~1.96× |

**Recommendation: 0.3 if more amplitudes are wanted.** It triples WA coverage
for about +40% on the campaign. Dropping to 0.2 roughly doubles the campaign and
should not be done with the short-window method — see below.

## 3. Whole-day deconvolution — agrees, but is only cheaper above ~800 picks

**Accuracy: the two methods agree.** Whole-day ÷ short-70 s across all picks:
median **0.9990**, p10 0.9929, p90 1.0047. The short window is not paying for
its speed with accuracy.

**Cost, measured on this station-day:**

| method | cost |
|---|--:|
| whole-day deconvolution | **7.01 s per station-day** (flat) |
| short 70 s window | **8.68 ms per qualifying pick** |
| 140 s window | 22.5 ms per pick |
| 300 s window | 29.0 ms per pick |

**Break-even is 808 picks per station-day-channel.** The campaign sees
~900–1,100 picks per station-day-channel at the 0.2 picking threshold, of which
~160 pass the 0.5 gate. So:

- **at `wa_min_conf` 0.5 (~160 picks): the short window is ~5× cheaper.** Keep it.
- at 0.3 (~500 picks): short window still cheaper, but only ~1.6×.
- **at 0.2 (~1,000 picks): whole-day is cheaper.**

So "lower the threshold to 0.2" and "go back to whole-day deconvolution" are the
same decision, not two. Whole-day at full threshold costs ~1.82× the campaign
against ~1.96× for short windows — and it removes the taper question entirely,
and is the only route to a longer-period measure later (displacement, Mw), which
the short window cannot support at all.

### The one disagreement, unexplained

**1 pick of 41 differs by 48%** (short-window high, whole-day low; 0.17 ML).
Ruled out by measurement:

- **not a data gap** — the day has one 19 s gap at 01:36:31, 22 hours earlier;
  re-running with the day trimmed to after the gap leaves the outlier unchanged;
- **not a large event** — 2,021 peak counts against 2,646 for a median pick;
- **not the per-window detrend** — `linear`, `demean` and none agree to 5e-6;
- **not a rejected component** — both horizontals are used by both methods and
  both show the same ratio (1.489 E, 1.467 N);
- **not the window length** — 70, 140, 300, 600 and 1200 s all agree to 0.35%.
  It is the *whole-day* value that is the outlier, not the short one.

Coherent across components, so it is a property of the deconvolution rather than
the measurement. **Worth resolving before either method is trusted at the
0.1 ML level**, and it argues mildly *against* treating whole-day as the gold
standard. It is 1 pick on 1 station-day; the first step is to see how often it
happens, which is a re-run of this script over a few hundred station-days.

## What this does not cover

- One station, one day, one network, 41 picks. The ratios are tight enough that
  the taper and threshold conclusions are unlikely to move, but the outlier rate
  is a single observation.
- 50 Hz BH data. Cost per pick scales with sample rate, so the 8.68 ms and the
  808-pick break-even shift on 100 Hz data — the *ratio* is what transfers.
- Run with `seisbench 0.10.2` locally; the container's `pip install seisbench`
  is unpinned ([OPTIMISE.md](OPTIMISE.md) item 6).
