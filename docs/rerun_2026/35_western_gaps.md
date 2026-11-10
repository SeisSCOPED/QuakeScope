# 35. wc2025 station-days missing from western: causes and the `western-gaps` queue

2026-11-10. Junhao Song compared the 2025 West Coast picks (wc2025) with
`western/` + `obs/` (wc2026), over -134 to -108 E, 22 to 53 N, 2002-01-01 to
2025-03-31. He sent `stations_compared.csv` (16,953 stations) and
`missing_station_days.parquet`: 553,672 station-days with picks in wc2025 and
none in wc2026, each with a reason. This document traces each reason to its
cause and records what was relaunched. Numbers come from his two files, the
three `stations.parquet` tables and `western/availability/` as of 2026-11-10.

## Causes

| his reason | station-days | stations | cause | action |
|---|--:|--:|---|---|
| `not_found` | 231,913 | 5,206 | the EarthScope S3 access point has no such network-year (404 at the credential exchange); temporary and nodal networks (ZG 2014, 1D, 9K, 1B, XQ, XD, XT, XE...) | needs an FDSN read path, below |
| station not in wc2026, US shoreline and islands | 79,055 | 38 | selection bug, below | **relaunched** (32 stations) |
| station not in wc2026, British Columbia | 80,645 | 46 | same bug; BC is in scope since the 2026-09-21 decision to pick the whole stakeholder list | **relaunched** |
| station not in wc2026, Alberta (2K) | 6,895 | 72 | 2K has no metadata at EarthScope FDSN, so `plan_western_fill.py` dropped it | open |
| never processed, TD | 63,780 | 22 | restricted network, refused to this account | open (access) |
| never processed, BK EP-only and other EP/EL/SL-only | ~39,000 | ~700 | the 2026-09-02 band decision dropped EP, EL and SL from land weights | decision |
| never processed, NP | 7,423 | 26 | NP was left out of every western queue (archive at NCEDC) | open |
| never processed, other | 23,800 | 2,096 | scattered days with no outcome (median 4 per station), the held-out holes of the October re-read | **relaunched** |
| `no_data`, `empty_read`, no row | 16,268 | ~300 | an outcome is recorded, or the day is outside the station's epochs | none |

### The selection bug

Both station selections asked which admin-1 polygon (Natural Earth 10 m)
contains a station, and treated "none" as offshore, to be picked by the obs
campaign: the six-state selection of commit c1d7228 and `where()` in
`scripts/plan_western_fill.py`. The obs selection (doc 31) then kept only
stations outside the land mask **and** at or below -10 m, and returned the rest
to `global`, which has not run. A station on an island or the shoreline is
outside the generalized coastline and above sea level, so it was in neither
campaign: NC.CYB (Yerba Buena Island), CI.SBI (Santa Barbara Island), UW.ALKI,
BK.MCCM (22 m from the polygon), the UW Puget Sound and Oregon coast stations,
and the Vancouver Island and BC coast stations (PB.B0xx, CN, C8). All 156 are
on the stakeholder list, `WestCoast_stations.txt`.

Not yet fixed in the planners: a point outside every polygon should go to the
nearest polygon within a few km, and to obs only when it is below -10 m.

### `not_found`

The reader asks EarthScope for credentials scoped to the network and, for a
temporary code, the year (`s3_helper.es_scope`), and records `not_found` when
the exchange answers 404. Sampled on 2026-11-10, one station-day per
network-year, 80 network-years: the public FDSN dataselect service at
service.iris.edu returned data for 70 of them, 99.3% of the `not_found`
station-days by weight; the PH5 service returned none. The data are at
EarthScope, outside the S3 bucket. At 32 to 42 MB per station-day (1-hour
requests scaled to a day), the set is about 7 to 10 TB, too large to pull from
the web service without asking EarthScope first.

### Not missing data

Junhao's slide 3 shows wc2026 with 0.24 times the EH picks per record of
wc2025 and 2.7 to 3.5 times the HH, BH and HN picks. The EH ratio is the 2025
single-component EH over-picking, not a loss. The 2014 spike in wc2025 is ZG
2014, which is in the `not_found` set.

## The `western-gaps` queue

`scripts/plan_western_gaps.py`, written 2026-11-10 to `_queues/western-gaps/`,
picks into `western/` with `quakescope_2026_western:30` (image 6e99850, which
descends from the d0ccf9b reader fix).

| part | station-locations | shards | planned station-location-days |
|---|--:|--:|--:|
| absent stations, per epoch of their pickable vertical channels, 1986.001 to 2026.251 | 84 (78 stations) | 4,678 | 301,209 |
| never-processed days, as 20-day windows around exactly those days | 2,236 (2,096 stations) | 3,871 | 24,496 |
| total | 2,320 | 8,549 | 325,705 |

Epochs come from each network's data center (`add_station_epochs.py`): SCEDC
for CI, NCEDC for NC and BK, EarthScope for the rest. Six stations of the
shoreline set are left out because their data center lists no vertical channel
in a picked band: BK.W02B and BK.BBEB (borehole DP, 4,095 station-days in
wc2025) and PG.OBS1 to OBS4 (418).

Checks before the launch: the hole windows were checked against the
availability table for BK, CI, NC and UW (9,387 location-days, all `unknown`,
so no day already read is re-read); the doc 34 table and shard gates
(`campaign_workflow.py` on PR #66) report no problems.

Cost, estimated: $18 at the $0.0000556 per planned station-day of the
western-fill estimate, and at most $78 if every planned day were processed at
the western benchmark of $0.000239 per processed station-day.

After the run: the absent stations will have picks under `western/` that
`western/stations.parquet` does not list. Merge the queue table into it with
`scripts/merge_station_tables.py --campaign western` (issue #46), then rebuild
`western/availability/`.
