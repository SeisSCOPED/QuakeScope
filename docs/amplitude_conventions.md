# Amplitude conventions

What [`sb_catalog/src/amplitude_extractor.py`](../sb_catalog/src/amplitude_extractor.py)
measures, and why the numbers are what they are. These choices define the
magnitudes of the resulting catalog, so they belong in the provenance record.

## Two amplitudes per pick

| Field | Components | Rule | Response removed |
|---|---|---|---|
| `amplitude` | horizontals (`NE12`) | **mean** of the per-component peaks | yes → Wood-Anderson |
| `raw_amplitude` | all | **max** over components | no — counts, high-passed at 1 Hz |

Both are peak values in a window `time_before=3 s` before to `time_after=10 s`
after the pick, sliced out of a window padded by `slack=10 s` on each side.

`amplitude` feeds ML. `raw_amplitude` is a detection-strength proxy that
survives missing or broken response metadata; it is not a physical unit and is
not comparable across instruments.

## Wood-Anderson constants

The IASPEI standard: `T0 = 0.8 s`, damping `h = 0.7`, static gain `2080`,
applied to ground velocity (one zero at the origin).

The 2025 catalog used `h = 0.8` — Richter's original damping — paired with
IASPEI's 2080 gain rather than Richter's 2800, i.e. a mix of two conventions.
Corrected in the 2026 code. Measured on five CI stations, the change raises
amplitudes by a near-uniform **+0.033 ML** (station-to-station spread 0.010),
so 2025 ML values are directly comparable after adding that constant. The point
of the change is comparability with other ML catalogs, not accuracy.

## The deconvolution runs once per station, on the whole trace

The response is removed **once per station**, over whatever span the stream
covers — typically a full day — and each pick's measurement window is then
sliced out of the deconvolved trace.

It previously ran once per *pick*, on a 33 s window: roughly 6,700
deconvolutions for a single busy station-day, each recomputing the same transfer
function. Hoisting it is 5.3x faster, but the reason to do it is correctness.

**A 33 s window was not a reliable measurement.** For the pick at
2019-07-06T03:16:17 on `CI.CLC`, the old window returned 0.170 while every other
window — symmetric 33 s, the same shape doubled, 300 s, 3600 s — converges on
0.00045. The old asymmetric `-13/+20 s` window was ill-conditioned there, and
shifting it by a fraction of a second moved the answer by 380x. About 5% of
picks were affected. This is the same failure as
[Denolle-Lab/cascadia_obs_ensemble#19](https://github.com/Denolle-Lab/cascadia_obs_ensemble/issues/19):
a short deconvolution window is not a measurement.

Across 508 real picks the median ratio to the old implementation is 1.0001
(+0.0001 ML), with 94% within 5%; the disagreements are the ill-conditioned
cases, where the new value is the correct one.

### Two consequences worth knowing

**The taper is specified in seconds, not as a fraction.** obspy tapers 5% by
default, which on a day-long trace is 72 minutes at each end and would silently
null the amplitude of every pick near a day boundary. `taper_seconds` (default
60) keeps it fixed regardless of trace length.

**Picks inside a taper return NaN.** The taper drives the signal smoothly to
zero, so a measurement taken there is wrong rather than imprecise — a pick at a
trace edge measured 200x low before this. Missing is honest; suppressed is not.
It costs about 0.1% of picks on a day-long trace, at the day boundaries and
beside gaps.

The `pre_filt` low corner no longer has to fit inside a short window. What is
still checked is that the *measurement* window can hold the periods being
measured: a 13 s window cannot represent a 50 s period however well the trace
was deconvolved, and the constructor warns when it cannot.

## Known gaps

- **No displacement/Mw amplitude is measured.** Adding one means a second,
  much longer deconvolution window, not a second slice of the existing one.
- **ML is horizontals-only**, which is standard, but leaves OBS stations with
  unknown horizontal orientation and high horizontal noise poorly served. The
  Cascadia code takes the max over all components including Z for that reason;
  the difference is +0.057 ML median but station-dependent (0.020–0.118), so it
  is absorbable by per-station terms and not by a global constant.
- **No distance correction or station terms** are applied here — this stage
  emits amplitudes, not magnitudes.
