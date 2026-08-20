# Rendered reports

Executed HTML renders of the notebooks in [`../tutorials/`](../tutorials/), with
all figures and tables in place, for sharing with people who should not have to
run anything.

Regenerate any of them with:

```bash
pixi run -e tutorials jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=quakescope-cpu \
    --ExecutePreprocessor.timeout=7200 \
    --output /tmp/out.ipynb tutorials/<notebook>.ipynb

pixi run -e tutorials jupyter nbconvert --to html --template lab \
    --output-dir reports --output <name> /tmp/out.ipynb
```

| Report | What it answers | Runtime |
|---|---|---|
| [phasenet_sequence_comparison.html](phasenet_sequence_comparison.html) | Do the three weight sets hold up across five sequences, four regions, and two catalogs? | ~25 min |
| [phasenet_aftershock_benchmark.html](phasenet_aftershock_benchmark.html) | Detailed S-recall benchmark on one dense Ridgecrest window, against 533 analyst S picks | ~5 min |
| [phasenet_smoke_test_ridgecrest.html](phasenet_smoke_test_ridgecrest.html) | Does one weight set produce physically sensible picks at all? The first thing to run | ~2 min |
| [phasenet_obs_offshore_benchmark.html](phasenet_obs_offshore_benchmark.html) | What do the ocean-bottom pickers buy over land models on OBS data, and does the hydrophone matter? | ~50 min |
| [quakexnet_alaska_test.html](quakexnet_alaska_test.html) | Does the PNW-trained classifier transfer to Alaska, and how much does window placement matter? | ~30 min |

## What these currently show

**The picker — and a correction.** At the shared 0.3 threshold everyone uses,
`original` appears to recover far more S arrivals than the others: Ridgecrest
0.76 against 0.56, Mendocino 0.73 against 0.61. That reading is wrong, and the
sequence report now shows why.

Holding the threshold fixed across weight sets does not hold the *operating
point* fixed. `original` emits close to twice as many S picks at 0.3 as the
other two, so it sits further along the recall curve and collects both more
recall and more extra detections. **At matched pick budgets the three are
within a few points of each other** and no ordering survives across sequences:

| Ridgecrest, S picks emitted | quakescope2026 | jma_wc | original |
|---|--:|--:|--:|
| 287 | 0.548 | 0.536 | 0.540 |
| 370 | 0.663 | 0.639 | 0.628 |
| 535 | 0.775 | 0.777 | 0.768 |

The operational consequence is that **thresholds belong to the weight set, not
to the pipeline**. Carrying 0.3 across a change of weights silently moves the
operating point and changes catalog completeness with it.

**`instance` — the weight the 2025 campaign ran — is a separate case**, and the
distinction matters. On Mendocino it matches the others on budget and is
competitive, in places best. On Ridgecrest it never reaches their budgets at
all: with its threshold on the floor it emits 246 S picks where the others
reach 684 and 832. No threshold recovers that, so it is a ceiling rather than a
calibration offset. Ridgecrest is the densest and closest-in sequence here,
which reads as the incumbent struggling specifically with heavily overlapping
near-field aftershocks while remaining fine at regional distance.

The three are also not one lineage. `original` is Zhu et al., trained on
Northern California. `jma_wc` is a different architecture — PhaseNetWC, double
the filters per layer — trained on Japanese JMA data, and `quakescope2026` is
fine-tuned from it. Nothing here descends from `original`.

**Offshore.** SeisBench ships three ocean-bottom pickers. `PickBlue` is a
constructor returning the `obs` weights on either a PhaseNet or an
EQTransformer backbone, both taking four components including a hydrophone;
`OBSTransformer` is OBS-trained but takes only three. Across 138 windows on
three deployments, the OBS models lead the land models by roughly 5–15 points
of detection, and **no single one wins everywhere** — `pickblue_eqt` is best at
Cascadia, `obstransformer` at AACSE, and at Blanco all five are within a few
points of each other.

The hydrophone turns out not to be the reason. Re-running the four-component
model on 94 detected windows with the channel withheld moved mean P confidence
by **+0.0002**, helping 32 windows and hurting 35. Its sampling rate spans
100 Hz to 10 Hz across these deployments and the effect is indistinguishable
from noise at any of them. That `obstransformer` competes without a hydrophone
at all points the same way: the gain is from training on ocean-bottom data, not
from the fourth channel.

**The classifier.** QuakeXNet agrees with the Alaska catalog 78% of the time
when the analysis window matches the training convention, and 16% when it does
not — the same events and the same waveforms, cut differently. It is deferred
for the 2026 campaign on that basis; see
[`../docs/quakexnet_generalization_plan.md`](../docs/quakexnet_generalization_plan.md).

## Reading the numbers honestly

Recall is measured against analyst picks, which are authoritative for what they
contain and **not exhaustive**. Unmatched model picks are reported as *extra
detections*, never as false positives, because in a dense sequence most of them
are real earthquakes nobody had time to work through. Deciding which is which
needs association across stations.

The reference is not a pick archive — no such service exists. It is the set of
arrivals attached to published origins, so a pick counts only because an analyst
made it *and* a location used it. Which earthquakes count is fixed by the team's
curated reviewed-event lists in `docs/rerun_2026/comcat_reviewed_*.csv`, and
arrivals for those events are harvested from every archive that located them —
SCEDC and NCEDC solve the same California earthquake with different station
sets, and ComCat carries only the authoritative solution. Only manually reviewed
picks are kept. The provenance section of that report covers what the catalogs
record, which is the operating agency rather than any individual analyst.

Sample sizes still differ by more than an order of magnitude between sequences —
San Simeon rests on 16 analyst S picks where Ridgecrest has 298. The `analyst`
column in every table is the sample size and should be read alongside the
recall.
