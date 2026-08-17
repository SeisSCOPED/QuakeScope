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
| [quakexnet_alaska_test.html](quakexnet_alaska_test.html) | Does the PNW-trained classifier transfer to Alaska, and how much does window placement matter? | ~30 min |

## What these currently show

**The picker.** `original` recovers the most S arrivals on every sequence where
the comparison is well powered — Ridgecrest 0.76, Mendocino 0.73, Monte Cristo
0.67 — against 0.56, 0.61 and 0.42 for `quakescope2026`. The fine-tune leads on
P at Mendocino and on timing in several places, which is the trade its own
training benchmark describes: v7 buys P-MAE and pays in recall. That the pattern
repeats across four regions and two independent analyst catalogs makes it a
property of the weights rather than of one window.

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

Sample sizes differ by more than an order of magnitude between sequences, and
San Simeon has almost no analyst S at all — two picks, a 2003 cataloging
practice rather than anything about the data. The `analyst` column in every
table is the sample size and should be read alongside the recall.
