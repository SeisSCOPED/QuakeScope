# Paper outline: what a deployed picker benchmark has to measure

Working draft, 2026-09-29. Not a manuscript yet: a statement of what we can
claim, what the evidence is, and what is still missing. Numbers in this file
are placeholders marked `[from <file>]` and are filled from
`docs/benchmark/results/` by `scripts/build_benchmark_readme.py`, never typed.

## The gap this paper fills

Picker benchmarks score models on **labelled datasets**: every arrival in the
scored window is known, so precision, F1 and MCC are identifiable, and a model
is ranked by numbers computed on data whose labels came from the same
distribution as its training set. Münchmeyer et al. (2022,
doi:10.1029/2021JB023499) is the reference work and made the cross-domain point
that a picker transfers between regions with mild degradation but not from
regional to teleseismic.

What a person deploying a picker actually faces is different in three ways,
and each changes the answer:

1. **The reference is an operator bulletin, not a labelled set.** An analyst
   picked what a location needed. Unmatched model picks are a mixture of false
   positives and real arrivals nobody marked, so precision and F1 are not
   identifiable and any number reported for them is a lower bound. Nearly every
   applied comparison in the literature reports F1 against such a reference
   without saying so.
2. **A threshold is not an operating point.** Weight sets put their
   probabilities on different scales: at a shared 0.3 one emits twice the picks
   of another and collects both more recall and more extra detections. A
   benchmark that fixes the threshold measures liberality as much as skill.
3. **The deployment is the experiment.** A campaign that picks 114 million
   station-days exposes failure modes a windowed benchmark cannot: silent
   skips, metadata that truncates station epochs, archive copies that differ
   from the web service. These change catalogue completeness by more than the
   difference between the candidate models.

## Claims, and the evidence for each

**C1. At matched pick budgets the three land weight sets are within a few
points, and no ordering survives across sequences.**
Evidence: seven sequences on two continents, 2,300 analyst arrivals, scored at
equal emitted picks. `[from summary_matched_budget.csv]`

**C2. A fine-tune selected on a benchmark's timing metric did not transfer.**
`quakescope2026` (v7, fine-tuned from `jma_wc` on 527k windows) was chosen by
reading a benchmark across 19 versions with no held-out split. Against three
operators' bulletins abroad it loses to its own parent on every sequence and
phase, at the shared threshold and at matched budgets, and its timing edge is
0 to 17 ms. Contamination favours v7 (its corpus includes INSTANCE, and Norcia
is in-domain), and it still loses. `[from global_sequences/*]`

**C3. Where a weight cannot reach the others' pick counts, that is a ceiling,
not a calibration offset, and no threshold recovers it.**
`instance` at Ridgecrest emits 246 S picks with its threshold on the floor
where the others reach 684 and 832. At regional distances abroad the same
weight is the most efficient of the four at any common budget. The two facts
together are a deployment rule, not a ranking. `[from us_sequences/saturation.csv]`

**C4. Confidence is not a probability, and the miscalibration differs between
weight sets.** `[from calibration_*.csv]` This is the mechanism behind C1 and
the reason a shared threshold misleads.

**C5. Training on ocean-bottom data, not the hydrophone, is what buys offshore
performance.** Withholding the pressure channel from a four-component model
moved mean P confidence by +0.0002 across 94 windows;
`obstransformer`, which has no hydrophone, competes with the four-component
models. `[from obs_offshore/*]`

**C6. The campaign's own picks reproduce exactly.** 59,298 of 59,315 picks
recovered to the millisecond from a different data path on a different CPU
architecture; the 17 that did not are the same arrivals one to three samples
apart. `[from western_reproduction/*]`

**C7. Deployment failure modes dominate model choice for catalogue
completeness.** Measured on this campaign: resumed shards overwriting their own
output (1.8 % of western's processed station-days), a station-date encoding
that truncated 1,168 station epochs (121,692 station-days), a gap rule that
dropped fragmented days, and archive listings that failed silently. Each is
larger than the recall difference between the candidate weights.

## Methods to write out

- Reference construction: curated reviewed-event lists; arrivals harvested from
  every archive that located the event; manual picks only; the trap that using
  only the authoritative solution gives 2 S picks where all archives give 16.
- Matching: greedy nearest, one-to-one, tolerance stated; detection scored at
  0.5 s and residuals at 2 s, because matching at the detection tolerance
  truncates the residual distribution and makes an outlier rate a statement
  about the tolerance.
- Metric definitions, and which are identifiable against a bulletin:
  recall (yes), precision/F1 (lower bounds), phase swaps (yes), residual
  statistics (yes), calibration (lower bound), multiplicity (yes).
- Threshold handling: shared threshold, matched pick budget, and per-model best
  threshold, reported together because they can disagree.
- The cost axis: vCPU-hours per processed station-day per weight, measured.

## Figures

1. Recall at a shared threshold vs at a matched budget, seven sequences: the
   ordering changes.
2. Recall against picks emitted, one panel per sequence and phase, with each
   model's shared-threshold point marked.
3. Residual distributions, with MAE, RMSE, MedianAE and the gross-error rate.
4. Reliability curves per weight set: confidence against observed agreement.
5. Offshore: five models on the same windows, and the hydrophone ablation.
6. Deployment losses as a share of station-days, against the model differences
   on the same axis.

## Target and honesty constraints

Seismica or SRL; the audience is people choosing a picker for a deployment.
Every number traceable to an executed notebook. The limitations section states:
sample sizes differ by an order of magnitude between sequences; leakage
assessment per weight; Thessaly S is capped near 0.5 for every weight for a
reason we have not identified; no association step, so extra detections remain
unclassified.

## What is missing before submission

1. Association on one sequence, to turn extra detections into a precision
   estimate rather than a bound.
2. A held-out sequence never looked at during any of this work.
3. The global campaign's own stored picks scored on the three foreign
   sequences (section 8 of that notebook does it once the picks exist).
4. Blanco analyst picks; a second AACSE season.
