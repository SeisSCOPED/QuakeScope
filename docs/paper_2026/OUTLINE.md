# Data report outline: the 2026 QuakeScope catalogues

Working draft, 2026-10-19. Target: *Seismica*, Report (Data Report), as an
update of Ni et al. (2025, doi:10.26443/seismica.v4i2.1738). Not a manuscript
yet: what the paper presents, the evidence for each claim, the figures, and
the decisions still open.

Numbers come from `figures/numbers.json`, written by `make_figures.py` from
public objects in `s3://quakescope-picks-2026` (station tables, availability
tables, 2025 manifests, three network-months of picks), or from the named
source. Re-run before quoting; do not type numbers into the text by hand.

```bash
python docs/paper_2026/make_figures.py all --cache <dir>   # ~3.5 GB pulled anonymously, ~15 min
```

## Working title

*QuakeScope 2026: phase picks for the western United States since 1986, for
ocean-bottom seismometers, and the 2025 global catalogue in an open format*

## What the paper presents

Three catalogues in one bucket, one Parquet layout and one schema, read
anonymously:

| prefix | picker and weights | years | station-locations (with a loaded day) | station-days read | picks (rows) | stored | compute |
|---|---|---|--:|--:|--:|--:|--:|
| `western/` | PhaseNet `original`, P/S 0.2 | 1986-01-01 to 2026-09-08 | 26,377 (13,123) | 16,965,900 | 2,130,446,013 (~2.07 B unique) | 55.3 GB | ~$4,200 (est. $3,700 to $4,800) |
| `obs/` | PickBlue (`obs`), P/S 0.2, adds `EL` | 1997-03-16 to 2026-10-01 | 3,036 (2,018) | 449,541 | 104,420,230 (~103.6 M unique) | 2.9 GB | ~$120 (est. $90 to $170) |
| `quakescope-2025/` | PhaseNet `instance`, P/S 0.2 | 2002 to 2025-03 | 58,403 in the table; 47,354 with data (Ni et al., 2025) | 38,774,015 station-band-days | 4,376,745,115 | 88.3 GB | ~$15,000 (Ni et al., 2025) |

Sources: station-days and date range from the availability tables
(`numbers.json`, `western`/`obs`); rows from Parquet footers on 2026-10-13
(`_archive/inventory-20261013/`); "unique" from the duplicate sample in the
same inventory; stored sizes from the S3 listing of 2026-10-19; 2026 compute
from `reports/western_2026.html` and `obs_2026.html` (vCPU-hours priced at
$0.0213, calibrated on CloudBank billing, `docs/rerun_2026/24_cost_model.md`);
2025 station-band-days from the exporter's per-network `picks_record` counts
(`quakescope-2025/_export/`). Unit cost: $0.00024 per processed western
station-day and $0.00035 per OBS station-day (doc 24), against about $0.00039
per 2025 station-band-day ($15,000 over 38.8 M).

Why an update rather than a new run of the same thing:

1. **A weight choice for dense near-field sequences.** `instance` has a
   ceiling on S picks in dense aftershock sequences that no threshold
   recovers (benchmark claim C3, `docs/benchmark/PAPER_OUTLINE.md`). The West
   Coast Catalog group traced low association recall at Ridgecrest in the
   2025 picks to missing picks. `western` uses `original`.
2. **Ocean-bottom data with an ocean-bottom picker.** Ni et al. (2025) named
   PickBlue for OBS data as future work. `obs` does it, with stations chosen
   per station by a land mask and water depth, not by network code.
3. **Back to 1986.** The 2025 run started in 2002.
4. **A format that needs no server.** Parquet on S3 with manifests, run
   records and a per-station-day availability table, replacing DocumentDB,
   which was retired on 2026-10-12. The 2025 picks were converted, count for
   count, so both generations read with the same code.
5. **An accounting of every station-day.** What was read, what had no data,
   and what was never read, published per station-day.

## Claims and evidence

**D1. The 2026 western catalogue reads 16.97 M station-days with one
configuration, and says what happened to the other 37.4 M inside the
stations' FDSN epochs.** Availability table: loaded 16,965,900 (1,368,764 of
them with data and no pick), no_data 11,627,811, not_found 931,317,
empty_read 207,332, unread 1,030, unknown 24,621,886 (mostly NP, about 23 M,
and the networks our account may not read). Figure 2.

**D2. On the same station-days and band, the two generations differ by band
more than by year.** 2002 to March 2025, 11.7 M station-days picked by both:
2026/2025 median picks per station-day of 2.0 (HH, 3.2 M station-days), 3.0
(BH), 2.7 (HN), 2.0 (DP), and 0.06 (EH, 5.0 M) and 0.48 (SH). Figure 3.

**D3. The EH deficit is 2025 over-picking single-component data, not 2026
missing arrivals.** The EH stations of NC, UW, CI, MB and NN are mostly
vertical-only; PB borehole EH (three-component) shows 6.2 (2015 onward,
232,692 station-days). Two NC EHZ
station-days re-picked locally from NCEDC FDSN reproduce both catalogues
(`original` 23 vs 22 stored, `instance` 224 vs 221; 18 vs 15 and 3,745 vs
3,756) and both weight sets recall the NCEDC analyst picks at those stations
(3 of 3 and 2 of 3). `instance` emits 3,745 picks on a day with two NCEDC
events within 1 degree. **Evidence is thin (two station-days); see "Missing
before submission".** The seasonal pick rate at UW.BVW (EH) shown in Ni et al.
(2025, Fig. 4) as noise-driven seasonality is absent in 2026 (Figure 6).

**D4. The weight sets place arrivals differently, by an amount that matters
for location.** CI, July 2019: matched `original` picks are later than
`instance` by a median 48 ms (P, 849,099 pairs) and 98 ms (S, 662,721) at
`conf` >= 0.5. Against analyst picks, median residuals are `original` +28 ms P
and +28 ms S at Ridgecrest, +20 / +50 ms at Norcia; `instance` −12 / −37 ms
and −10 / −40 ms (`docs/benchmark/results/timing_full.csv`). Figure 4c.

**D5. At Ridgecrest, 2026 holds three times the confident S picks of 2025 on
the same station-days.** CI, July 2019, `conf` >= 0.5: S 1,333,048 (2026) vs
444,816 (2025); 81.5 % of 2025 S picks found in 2026, 49.7 % of 2026 S picks
found in 2025. P: 1,336,978 vs 1,082,753, 73.9 % and 63.5 %. Figure 4a, b.

**D6. On ocean-bottom stations PickBlue gives about four times the picks of
`instance`.** 234,376 OBS station-days picked by both on the same band:
median ratio 4.1. Of the 429,532 OBS station-days with 2026 picks, 166,497
have no 2025 record at all and 194,414 have no 2025 picks. AACSE: 86 % of analyst P and
S arrivals have a PickBlue pick within 1 s (land control 91 / 85 %), from
`reports/phasenet_obs_offshore_benchmark.html`. Figure 5.

**D7. 2026 is not a superset of 2025.** 2002 to March 2025 on western
stations: 12.78 M station-days in both, 0.83 M in 2026 only, 0.56 M in 2025
only. 540,868 of the 2025-only station-days carry 97.3 M 2025 picks. What
2026 says about them: not_found 256,241 (temporary, mostly nodal networks ZG,
ZH, XQ, XD, ZI, XT, XE, 1D: the reader finds no network-year at the
EarthScope access point, and FDSN availability returned 404 for three
samples), unknown 171,803 (TD 63,780, account refused; BK 39,609; NC; CI),
outside FDSN epochs 101,354 (UU, MB, WY, US: 2025 read days that the FDSN
epochs used for 2026 planning exclude), no_data 8,505, empty_read 2,965.
Figure 3c. **Cause of the not_found block not established.**

**D8. Running a catalogue at this scale fails silently unless every planned
unit records an outcome.** Mechanisms found and fixed during the 2026 run,
each larger than the recall difference between candidate weights: the
one-band-per-station reader (15.6 M western station-days re-read,
`docs/rerun_2026/32_outcomes_and_band_fallback.md`); the >150-trace gap rule;
silent day-listing failures; resumed Spot shards overwriting their first
checkpoint files (`28_resumed_shard_overwrite.md`); a station-date encoding
that truncated 1,168 epochs (`30_station_dates.md`); OBS stations selected by
reused temporary network codes, 1,996 of 3,389 on land
(`31_obs_station_selection.md`); duplicate rows from interrupted attempts
(~64 M in western, inventory 2026-10-13). Table 4.

## Figures

All drafts are in `figures/`, built by `make_figures.py`.

| # | file | shows | status |
|---|---|---|---|
| 1 | `fig1_station_years.png` | station-years read and picked: western (1986-2026) and OBS (1993-2026); cf. Ni et al. (2025) Fig. 1 | draft; add a 2025 panel for the same region |
| 2 | `fig2_station_day_accounting.png` | (a) every western station-day inside FDSN epochs by outcome, per year, with 2025's station-days on top; (b) picks per year, 2026 vs 2025 (every band, and one band per day by the 2026 priority) | draft |
| 3 | `fig3_2025_vs_2026_station_days.png` | (a) picks per station-day, 2025 vs 2026, same band; (b) median ratio by band; (c) station-days in both, 2026 only, 2025 only | draft; split (c) by D7's causes |
| 4 | `fig4_ridgecrest_pick_agreement.png` | CI July 2019: fraction matched within 0.5 s by `conf`, confidence distributions, timing of matched picks | draft; add the analyst-pick residuals of both weights as panel (d) |
| 5 | `fig5_obs.png` | (a) PickBlue vs `instance` per OBS station-day; (b) picks per station-day by depth and band; (c) station-days per year | draft; add the AACSE recall panel from the OBS benchmark |
| 6 | `fig6_example_stations.png` | daily picks (28-day mean) at five stations, both generations; cf. Ni et al. (2025) Fig. 4 | draft; choose stations deliberately (one EH vertical-only, one HH, one borehole, one OBS) |
| 7 | not drafted | downstream: events associated from 2025 vs 2026 picks, same associator and period | needs the West Coast Catalog group's results and their agreement (see below) |

Candidate seventh-figure material, already reported to the group and not ours
to publish without them: VORA on the full western picks, 2010-01-01 to
2025-01-01, 3,556,961 events from the 2025 picks and 5,903,385 from the
September 2026 picks (Junhao Song, email of 2026-09-24); PyOcto on five
sequences, about twice the events (Jannes Münchmeyer, 2026-10-01); ComCat
recall at Ridgecrest after association (Ian McBrearty, 2026-09-21). All three
predate the October re-read and fill.

## Tables

1. The three catalogues (above), with the run configuration of each.
2. What changed between the 2025 and 2026 runs: weights, bands per
   station-day, gap handling, amplitude conventions (Wood-Anderson h = 0.8 in
   2025, 0.7 in 2026; `amp_vel` new), station selection, output store,
   orchestration.
3. Cost and storage: compute per station-day, bytes per pick (western 26 B,
   2025 20 B in Parquet: stored size over rows), and the retired database.
4. Failure modes found during the 2026 run, their size and the guard each now
   has (D8; `docs/rerun_2026/34` campaign playbook).

## Sections

1. **Introduction.** What the 2025 database delivered and what its users found;
   the five reasons above.
2. **Data and methods.** Archives (SCEDC, NCEDC, EarthScope S3); station
   selection (western state polygons plus the stakeholder fill; OBS by land
   mask and depth); epochs from FDSN as dates; picker configuration and the
   per-day band priority; amplitudes; the S3 work queue, Fargate Spot fleet
   and per-station-day outcomes; conversion of the 2025 database (exporter,
   count equality per network, what differs).
3. **Results.** D1 to D7, Figures 1 to 6.
4. **Discussion.** D8; what an availability table makes possible for users
   (data without picks is now distinguishable from no data); limits; mixing
   generations (D4).
5. **Data and code availability.** Bucket, prefixes, read rules, the two
   tutorials, the DOI.

## Decisions for Marine

1. **Benchmark paper separate or folded in.** Recommend separate:
   `docs/benchmark/PAPER_OUTLINE.md` is a methods paper about scoring pickers
   against bulletins; this report cites its summary table and timing results.
2. **The 2025-only station-days (D7).** Fill them before submission or report
   them as a limit. The not_found block (256 k station-days on temporary
   networks) needs a cause first; the epoch block (101 k) needs planning from
   the 2025 station windows as well as FDSN epochs.
3. **De-duplicate the bucket and mint a DOI** before submission, and make the
   second copy outside the account first (only one copy exists).
4. **Rewrite the station-code columns** (`scripts/fix_station_codes.py
   --write`), still damaged on 2026-10-19 (2,065 western rows).
5. **`global` stays out** (paused at a few percent, picker to change).
6. **Authorship and the downstream figure:** invite the West Coast Catalog
   associators as co-authors for Figure 7, or leave association out.

## Missing before submission

1. D3 on a real sample: 50 to 100 random EH vertical-only station-days scored
   against ANSS arrivals for both weights, not two.
2. D7's not_found cause: one ZG 2014 station-day traced through the 2025 and
   2026 readers.
3. Figure 7 or its removal.
4. Fresh counts after de-duplication.
