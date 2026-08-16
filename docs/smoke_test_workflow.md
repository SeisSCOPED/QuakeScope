# Phase-picker smoke test

A two-minute check that a PhaseNet weight set produces sane picks before it is
trusted for a production run. It is not a benchmark — for that, see
[phasenet_v7_model_description.md](phasenet_v7_model_description.md).

## Run it

```bash
pixi install --environment tutorials
pixi run -e tutorials install-kernel
pixi run -e tutorials smoke-test
```

Select the **QuakeScope CPU (tutorials)** kernel. The notebook downloads roughly
300 MB per event from the SCEDC public bucket and runs on CPU in about 90
seconds.

## What it does

1. Fetches five CI stations, 5–48 km from the 2019 Ridgecrest events, from
   `scedc-pds` over anonymous S3.
2. Loads `quakescope2026` if it has been converted locally, otherwise the
   published `original` weights — and says which it used.
3. Picks the M7.1 mainshock, plots per-station waveforms and a record section.
4. Picks a M4.6 aftershock and compares observed S−P against the interval
   implied by hypocentral distance.

Station and event details are in
[ridgecrest_2019_test_stations.md](ridgecrest_2019_test_stations.md).

## Reading the result

**Healthy:** picks sit on visible onsets; P precedes S at every station;
observed S−P grows with distance and lands within a second or two of prediction;
picks in the record section follow the moveout curves.

**Investigate:** S before P; S−P that does not scale with distance (usually a
component-mapping or resampling problem); dense picks in pre-event noise; a
station producing nothing while neighbours at similar distance work.

**Expected, not a failure:** no S picks for the mainshock at close range — a
magnitude 7 ruptures for tens of seconds and buries its own S arrival. Also
expect many extra picks: this window sits inside one of the most active
aftershock sequences on record, so much of that extra energy is real
earthquakes.

## Comparing weight sets

```bash
pixi run -e tutorials compare-models
```

Runs every available weight set over identical waveforms. Pick counts alone do
not rank models — a model emitting more picks may be recovering real
aftershocks or firing on noise, and only the waveform panels separate those
cases. What carries information is whether the weight sets agree on a shared
arrival to within a few tenths of a second, and whether any misses a station its
peers handle.

Note that `jma_wc` fails to load in some SeisBench releases
(`InvalidVersion: '1.partial'`); the notebook skips unloadable weights with a
message rather than aborting.

## Before a production run

If the smoke test passes, convert and install the production weights:

```bash
cd sb_catalog/models/phasenet
python convert_checkpoint.py --checkpoint /path/to/best.pt \
    --name quakescope2026 --verify
```

Then re-run the smoke test — it picks up `quakescope2026` automatically — and
proceed to [rerun_2026/README.md](rerun_2026/README.md).

## References

- Zhu & Beroza (2019), PhaseNet: <https://doi.org/10.1093/gji/ggy423>
- Woollam et al. (2022), SeisBench: <https://doi.org/10.1785/0220210324>
