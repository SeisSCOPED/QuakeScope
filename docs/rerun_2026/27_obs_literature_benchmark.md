# Benchmarking the OBS campaign against published picks

Scoping for the second 2026 benchmark: compare the `obs` campaign's picks with
catalogues other groups have published from the same instruments. Three
deployments were proposed - Axial Seamount, the Blanco transform, and the
Alaska AACSE array. All three are in the campaign's station table. Only one of
them is ready to run today, and the reason is a campaign coverage problem
rather than a literature problem.

Everything below was read from the campaign's own S3 state, not from a previous
document. First written 2026-09-06; the status section is 2026-09-07.

## Status, 2026-09-07, evening: two of the three arms are built

The AACSE and Axial arms now exist, in
[`tutorials/phasenet_obs_offshore_benchmark.ipynb`](../../tutorials/phasenet_obs_offshore_benchmark.ipynb)
sections 8–11, rendered at
[`reports/phasenet_obs_offshore_benchmark.html`](../../reports/phasenet_obs_offshore_benchmark.html).
Three things in the morning's scoping turned out to be wrong or out of date, and
all three made the job easier:

* **Barcheck's dataset is a download, not a request.** The eCommons repository
  has a public API and the pick tables are two tarballs of 0.7 and 1.1 MB
  (`archive_metadata2018.tar.gz`, `archive_metadata2019.tar.gz`), one row per
  (event, OBS station) with the P and S arrival as a sample index and a
  `manual`/`automatic` status. The 2 GB waveform archives are not needed. A
  companion **land** dataset (doi:10.7298/q2fq-9688, 2026) gives the same for
  the 30 land stations of the same network, which makes a free control.
* **Both Axial portals answer now.** The LDEO ML-DD catalogue
  (`Axial.MLDD.v202112.2`, 144,329 events 2014–2021, 15 MB) is events only.
  The UW portal at `axial.ocean.washington.edu` serves `ph2dtInputCatalog.dat`
  (100 MB, regenerated hourly): HypoDD phase format with **per-station P and S
  picks** and weights for ~320,000 events since January 2015. That is the
  pick-to-pick reference the morning note said did not exist.
* **`netyear-sweep` was not needed for the 2018 arm** and has still not been
  run. `XO` 2019 and `X9` 2013 remain empty; the AACSE arm is 2018 only.

What the arms found, in short (the report has the tables):

| | P | S | tolerance | reference |
|---|--:|--:|---|---|
| AACSE 2018, 65 OBS stations, covered station-days | **0.86** | **0.86** | 1 s | analyst manual picks, 9,150 P / 10,572 S |
| AACSE 2018, 30 land stations, same `obs` weight | 0.91 | 0.85 | 1 s | analyst manual picks |
| Axial 2015–2025, EH stations | 0.21 | 0.04 | 0.5 s | UW automatic picks |
| Axial 2015–2025, HH stations | 0.30 | 0.01 | 0.5 s | UW automatic picks |

Re-scored on 358 of the same AACSE windows against analyst picks at 1 s (section
10), the five models keep the ordering of the iasp91 table with wider gaps:
`obstransformer` 0.81 P / 0.87 S, the PickBlue pair 0.74–0.77 P, `quakescope2026`
0.74 P but 0.46 S, `original` 0.49 P; the stored campaign pick, with a full day of
context, scores 0.83 P / 0.81 S on the same windows.

Residuals on AACSE are centred: median +0.02 s (P) and +0.04 s (S), quartiles
inside ±0.1 s for P. Two stations, `WD46` and `WD47`, score zero: they wrote
about 15 picks a day where neighbours wrote hundreds and the nearest campaign
pick sits hours from the analyst's, moving month to month, so the archive's
timing for them is not what the analysts worked with; `WS72` is fine until
August and then drifts. **1,436 analyst arrivals (5%) fall on 350 station-days
the campaign holds no picks for**, in runs of consecutive days inside shards that
reported complete with `picks_record` below `station_days` (the worst wrote 31%) — the
silent-skip pattern from the western campaign, now measured here.

Axial is the weak arm and the reason is the seismicity, not the archive: the
catalogue is mostly M < 0.5 events under the caldera with S–P of 0.5–0.7 s,
the `obs` weights were trained on regional OBS records, and the campaign
downsampled 200 Hz to 100 Hz. Recall of UW P picks climbs to ~0.5 for
M 0.5–1.5 and collapses below M 0; only half of the campaign's own picks above
0.7 confidence have a UW counterpart. At the event level the campaign sees 18%
of the ML-DD events on three or more stations (50% of those M 0.5–1, chance
0.1%). The campaign catalogue at Axial should not be read below M 0.

Remaining: the Blanco request email (step 4 below), and `netyear-sweep` if the
2019 half of AACSE is wanted. The section below is the morning's status and is
kept as written.

---

## Status, 2026-09-07, morning

Nothing has moved on the `obs` campaign since this was written: 6,231 shards
complete, 335 blocked, newest pick object 2026-09-06 09:22. `XO` 2019 and `X9`
2013 still hold no picks, so both the AACSE and the Blanco arms are still
scoped to a single season.

**There is now a pre-2010 plan, and it has not run.** `obs-early` covers
1993-2009: 3,631 shards, 407,368 station-days, 17 networks with stations in
range, written 2026-09-06 14:25. The prefix holds three objects - `access.json`,
`shards.jsonl`, `stations.parquet` - and nothing else. No `complete/`, no
`claims/`, no `manifests/`, no `picks/`. `fleet.json` carries it at **target 0**,
as it does `obs-2026`, `western-2026` and `global-2026`; only `western` has a
non-zero target. The queue is written and idle, so there are no pre-2010 picks
to benchmark against yet - starting it is a fleet target change, not a
replanning job.

Two things worth knowing before spending on it.

**It does not help the three arms below.** Axial starts 2014, Blanco ran
2012-13, AACSE 2018-19. `obs-early` opens *different* deployments - XJ from
1995, XZ from 1993, YO, YR, ZU, YS - and because temporary network codes are
reassigned, the `XO` and `X9` stations in that window are other experiments
entirely, not early AACSE or early Blanco. Whether any of them has a published
ML-picker catalogue is an open question and the first thing to answer if the
pre-2010 data is wanted for benchmarking rather than for coverage.

**The plan is sound, and most of it is pickable.** 4.2% of its station-slots
name a station outside its deployment window, against 3.2% in the `obs` plan
that already ran, so it is not systematically mis-planned. Of the 1,503 stations
active before 2010, 1,388 (92%) have a band `select_channel` will pick - 595 HH,
494 BH, 248 EH, 41 HN, 10 SH. The 115 it skips are almost all two channel sets:

| stations | channels | note |
|--:|---|---|
| 76 | `EL2,EL1,ELZ` | short-period, outside `CHANNEL_PRIORITY` by design |
| 39 | `EPZ,EPE,EPN` | **short-period at 100 Hz - worth reviewing** |

`EP` is the one to look at. It is a short-period code in the same rate class as
`EH`, which `CHANNEL_PRIORITY` already ranks second, and excluding it drops 39
stations silently. That is a small change with a real coverage consequence, and
it should be decided before the campaign runs rather than after.

---

## What the campaign holds

| Deployment | Network | Stations in table | Pickable | Years with picks |
|---|---|--:|--:|---|
| Axial Seamount (OOI cabled) | `OO` | 8 (`AX*`) | 8 — 3 HH, 5 EH | 2014–2026, continuous |
| Blanco transform | `X9` 2012–2013 | 54 | **30** | 2012 only |
| AACSE Alaska | `XO` 2018–2019 | 97 | 97 (HH) | 2018 only |

Two things in that table decide the shape of the exercise.

**The Blanco short-period instruments are not picked at all.** Twenty-four of
the 54 stations advertise `EL1,EL2,ELZ` or `SL1,SL2,SLZ`, and
`constants.select_channel` returns `None` for both, so the reader skips the
station rather than picking it. That is the documented behaviour - codes
outside `CHANNEL_PRIORITY` are ignored deliberately - but it means the campaign
covers the 30 broadband Güralp CMG3T stations and none of the 25 short-period
Mark Products L-28LB ones. Any comparison against a published Blanco catalogue
is a comparison over the broadband subset, and saying so is part of the result.

**The second year of both temporary deployments is missing.** Not blocked, not
failed - recorded as complete, having read nothing:

| | shards | station-days planned | median runtime | `picks_record` | manifests |
|---|--:|--:|--:|--:|--:|
| `XO` 2018 | 120 | 23,285 | 1,786 s | 17,025 | 88 of 120 |
| **`XO` 2019** | 114 | 23,780 | **1.2 s** | **0** | **0 of 114** |
| `X9` 2012 | 42 | 6,336 | 0.6 s | 2,127 | 12 of 42 |
| **`X9` 2013** | 42 | 13,536 | **1.1 s** | **0** | **0 of 42** |

The station metadata says the instruments were in the water: the AACSE
stations carry end dates between 2019.223 and 2019.365, and the Blanco
stations between 2013.266 and 2013.279. The preceding year of the same
deployment produced picks. A shard that finishes in about a second has not
read anything, and `obs/blocked/` contains only `7D` and `2F` — so whatever
happened to these was not recorded as an access block.

This is not confined to the two deployments of interest. Across the whole
`obs` campaign, **2,293 of 6,231 completed shards carry `picks_record: 0`,
covering 211,583 station-days — 21.7% of the completed total.** Some fraction
of that is legitimate: a network-year genuinely absent from the archive, or an
instrument that was not recording. The fraction that is not is invisible from
`progress()`, which counts these as done.

`XO` and `X9` are year-scoped temporary codes behind EarthScope's restricted
access point, and `s3_helper` treats a 404 for a network-year as "a
network-year with no data is a day with no data" — quiet, debug-level, and
indistinguishable in the campaign state from an empty archive. That is the
first hypothesis to test, and the tool for it already exists:

    python -m src.picker netyear-sweep

It asks EarthScope's credential exchange whether each planned `(network, year)`
is there. It has to run from a Batch task on the campaign's job definition,
never from a laptop — the refresh token must not be used here (see the
`picker.main` docstring and `19_earthscope_access.md`).

**Do that before collecting any catalogues.** If `XO` 2019 and `X9` 2013 are
recoverable, the AACSE arm doubles and Blanco becomes viable; if they are not,
both arms are scoped to a single season and the write-up has to say so.

## Reference catalogues

Ranked by whether the picks can actually be obtained, which is the binding
constraint rather than the science.

### AACSE — the strongest arm

Barcheck's *Ocean-bottom P and S arrival waveform dataset from the Alaska
Amphibious Community Seismic Experiment, 2018–19* (Cornell eCommons) is
analyst-checked P and S arrivals on OBS and hydrophone channels, cut around
events within 350 km, built for ML training. It is a pick list rather than an
event catalogue, which is exactly the comparison this campaign needs — our
output is picks, and scoring picks against an event catalogue requires an
association step that introduces its own errors.

The regional context is Ruppert, Barcheck & Abers (2023), *SRL* 94(1) 522–530,
doi:10.1785/0220220226, which added about 30% more events and 60% more phase
picks using AACSE data. The Alaska Earthquake Center published the resulting
catalogues in two halves at ScholarWorks: May–December 2018 and
January–August 2019. The 2018 half aligns with the only year the campaign has.

### Axial Seamount — the arm that can run continuously

The campaign has `OO` picks from 2014 through 2026 with no gaps, which is the
only one of the three where a multi-year comparison is possible, and it spans
the 2015 eruption. Two published catalogues:

* the Wilcock/Waldhauser real-time high-precision catalogue, at
  `axialdd.ldeo.columbia.edu` and `axial.ocean.washington.edu` — built with an
  ML picker plus cross-correlation and double-difference relocation;
* Wilcock, Waldhauser & Tolstoy (2016) for January–November 2015.

Neither portal was reachable from this network — one presents an invalid
certificate, the other refuses the connection — so retrieval is an open step,
not a solved one. Worth noting that the LDEO catalogue is itself ML-picked,
which makes it a picker-to-picker comparison rather than a comparison against
analyst truth. That is still informative, but it is a different claim.

Five of the eight Axial stations are picked on `EH`, not `HH`. Short-period
instruments at 100 Hz are within PhaseNet's range but are not what it was
trained on, and the comparison should report `EH` and `HH` separately rather
than pooling them.

### Blanco — the weakest arm, and the one to defer

Evaluating ML pickers on the Blanco transform is already published: *Evaluating
the performance of machine-learning-based phase pickers when applied to ocean
bottom seismic data: Blanco oceanic transform fault as a case study*, GJI
242(3), ggaf256, which compares EQTransformer, PickBlue and OBSTransformer over
the September 2012 – October 2013 deployment against the Kuna (2020) catalogue
of roughly 8,000 events. Adding PhaseNet to that comparison is a clean
contribution.

Two problems. The paper's data availability statement puts the derived
catalogues "available from the corresponding author upon request" — so the
picks require an email and a wait, not a download. And the campaign holds only
2012, over the broadband subset. Defer this arm until `netyear-sweep` has said
whether 2013 is recoverable, and send the request in the meantime so the reply
arrives when the data does.

## Suggested order

1. Run `netyear-sweep` for `XO` 2019 and `X9` 2013 from a Batch task. This
   decides the scope of two of the three arms and costs one job. *Still open.*
2. ~~Build the AACSE arm against Barcheck's arrival dataset over 2018.~~ Done
   2026-09-07, notebook section 9 and 10.
3. ~~Build the Axial arm once a catalogue is actually in hand, reporting `EH` and
   `HH` separately.~~ Done 2026-09-07, notebook section 11, against the UW
   phase file rather than the LDEO events, which carry no picks.
4. Email the Blanco authors; revisit after step 1. *Still open.*

The method is the one validated in
[`tutorials/western_pick_validation.ipynb`](../../tutorials/western_pick_validation.ipynb):
match on station, phase and time, report what each side has that the other does
not, and never convert a reference catalogue's absence into a false positive —
analyst catalogues are not exhaustive, and neither is an OBS deployment's
coverage.
