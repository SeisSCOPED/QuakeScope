# 09 — Dedicated run: western states with the original PhaseNet

A separate campaign for a stakeholder group covering the western US: original
PhaseNet weights, a defined station set, and results fully isolated from the
main (global, new-weights) science run.

## Scope

- **Stations**: all NCEDC + SCEDC stations, plus a curated list of stations
  in WA, OR, NV, NM, UT, and ID (provided as a station list — a bounding box
  would drag in AZ/CO/MT/WY stations that are not part of the deliverable).
- **Weights**: the original PhaseNet `instance` weights — already in the
  container image, no container work needed for this run.
- **Isolation**: its own database inside the shared DocumentDB cluster
  (suggested name: `western2026`). Same cluster, same Batch queue, same
  image as the main run; only the database name and weight differ. `sb_runs`
  in each database records which weights produced which picks, so the two
  catalogs can never be confused.

Channel handling needs no special care here: NCEDC and SCEDC store one file
per channel (the reader enumerates them explicitly), and EarthScope stores
per-station files from which the listed channels are selected — the archive
layouts already route the right channels to the picker.

## Cost containment

This run is cheap by construction: one workflow, a bounded station set,
Fargate Spot, and the shared `picks_record` resume logic (re-submissions only
process what's missing). See "Capping spend" in
[06_monitoring.md](06_monitoring.md) for the mechanisms that bound the bill.

## Steps

1. **Prepare the database** (once): on the EC2 controller, run notebook 2
   against the new name:

   ```python
   db = SeisBenchDatabase(DOCDB_ENDPOINT_URI, "western2026")
   ```

2. **Prepare the station list file.** Any of these formats work with the
   submitter's `--station_file` option:

   ```
   # western_states.txt — one NET.STA.LOC per line, # comments allowed
   UW.RATT.
   UO.PINE.
   NN.WAK.
   ```

   or a CSV with an `id` column (extra columns ignored). Ids must match the
   `id` field in the `stations` collection (`NET.STA.LOC`, location possibly
   empty: `UW.RATT.`).

3. **Submit in two slices** (both write to the same database; the NCEDC/SCEDC
   slice is selected by network, the six-state slice by the list):

   ```bash
   # slice 1: all NCEDC + SCEDC networks
   PYTHONPATH=../sb_catalog ~/miniconda/bin/python -m src.submit_helper \
       pick 2023.001 2024.001 \
       --network BG,BK,BP,NC,PG,UL,WR,CI \
       --database western2026 --weight instance
   ```

   ```bash
   # slice 2: the six-state station list (EarthScope-hosted networks)
   PYTHONPATH=../sb_catalog ~/miniconda/bin/python -m src.submit_helper \
       pick 2023.001 2024.001 \
       --station_file western_states.txt \
       --database western2026 --weight instance
   ```

   If a station appears in both slices it is picked only once — the resume
   logic deduplicates at the station-day level within the database.

   `--station_file` combines (AND) with `--network` and `--extent` if you
   ever need to slice the list further. Stations in the file that are not in
   the database are reported in a warning at submission time — check that
   warning against the group's expectations before scaling up.

4. **Smoke test first** (guide 05 §3) with a two-day window; confirm
   `sb_runs` in `western2026` shows `weight: instance`, then submit year
   blocks.

## Relationship to the main run

| | Western-states run | Main science run |
|---|---|---|
| Database | `western2026` | `quakescope2026` |
| Picker weights | `instance` (original PhaseNet) | new weights (guide 08) |
| Stations | NCEDC + SCEDC + six-state list | global |
| Infrastructure | shared (cluster, queue, image) | shared |

The two runs can be in the Batch queue at the same time; they compete only
for the `maxvCpus` pool.
