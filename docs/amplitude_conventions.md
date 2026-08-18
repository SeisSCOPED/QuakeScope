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

## Why the deconvolution window is short — and when it stops being safe

The response is removed on a 33 s window, while the default `pre_filt` low
corner asks for a 5–10 s period. That is comfortable for Wood-Anderson and
would be badly wrong for a displacement amplitude.

The reason is that the Wood-Anderson simulation is itself a sharp bandpass near
1 Hz, so it discards the long-period deconvolution noise a short window
contaminates. Measured: varying the padding 30× (10 s → 300 s) moves the
Wood-Anderson amplitude by **0.077 ML total**. The same test on a 0.05–2 Hz
displacement amplitude in the equivalent Cascadia code moved it by **up to 12×,
≈1.8 Mw units**, worst at the lowest-amplitude stations — a magnitude-dependent
distortion no linear calibration removes
(Denolle-Lab/cascadia_obs_ensemble#19).

So: **do not reuse `AmplitudeExtractor` for a displacement or Mw amplitude
without raising `slack`.** The rule is that the deconvolved window must span at
least ~3 cycles of the longest period in the passband; 0.05 Hz needs ≥ 60–100 s,
not 33 s. The constructor emits a `UserWarning` when a supplied `pre_filt`
violates this, which is why the default `pre_filt` is `[0.1, 0.2, 40, 45]`
rather than the `[0.02, 0.05, 40, 45]` used in 2025 — the old value asked for a
50 s period from 33 s of data. Switching it changes amplitudes by
**+0.0002 ML**, i.e. nothing; it removes an inconsistency rather than a bias.

`slack` stays at 10 s deliberately. Raising it to 30 s removes about 0.03 ML of
residual window scatter at roughly 2.2× the FFT cost per pick — not a trade
worth making on a per-pick step at order 10^10 picks.

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
