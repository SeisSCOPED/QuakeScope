# 31 — Offshore stations: select per station, then repair the obs catalogue

**Status: plan, written 2026-10-03. Nothing below has been executed.** Every
number was measured from `s3://quakescope-picks-2026` on 2026-10-03 by the
commands in the appendix, not read from the dashboard or an earlier document.

Marine noticed from the dashboard station maps that the obs campaign holds
stations that are not on the sea floor. It does: the campaign was planned from
26 **network codes** ([11_launch_plan.md](11_launch_plan.md), "How onshore and
offshore were split"), and FDSN temporary codes are year-scoped and reused. The
code `XO` is AACSE only in 2018 and 2019; in 2011 to 2014 it was OIINK in
Illinois and Kentucky. `XJ` is Tenerife in 2007, `2F` is Taiwan in 2009, `YN` is
Nevada and Anza, `7D` is Cascadia Initiative only in 2011 to 2014. The survey
behind the list checked each code where it is OBS; the campaign then fetched
every station ever registered under the code, 1993 to 2026, worldwide.

| | land | sea | land share |
|---|--:|--:|--:|
| stations in `obs/stations.parquet` (3,389) | 1,993 | 1,396 | 59% |
| planned station-days, obs + obs-early + obs-2026 + obs-dates | 993,442 | 490,107 | 67% |
| completed station-days (5,984 `obs/manifests/*.json`) | 484,908 | 214,583 | 69% |
| picks written (manifest `npks`) | 98.9 M | 30.6 M | 76% |
| compute, `complete/` seconds at 8 vCPU and $0.0213/vCPU-h, last attempts only | ~$293 | ~$132 | 69% |

The station row uses the rule of section 1; the other rows were measured
first, with land meaning elevation above 0 m. The two agree to within 35
stations, all at or just below sea level on land (Imperial Valley, Ohio), and
none of those 35 wrote picks that change a row.

`obs-early` (1993 to 2009) is 89% land. The land stations are **absent from
`global`**, because `earthscope_onshore.txt` is "every mapped network minus the
offshore list", so about 1 M land station-days have no `jma_wc` picks. 240 of
them sit inside the western polygons and were picked a second time with
`original`. The picks themselves are not wrong: the `obs` weight scored 0.91 P /
0.85 S on the AACSE land control ([27](27_obs_literature_benchmark.md)). They
are in the wrong catalogue, picked with a different weight from their
neighbours.

---

## 1. The selection rule

A station is **offshore** when both hold:

1. it lies **outside the Natural Earth 10 m land polygons**, and
2. its **elevation is at or below -10 m**.

Neither half works alone. Elevation alone calls the Salton Trough (CI, NP, SB,
EN at -60 m), the Dead Sea nodes (`1C`, -214 m) and boreholes (IU, ZL) offshore.
The mask alone misses small islands (Johnston, Midway, Wake, Kwajalein) and
treats ice shelves as sea: the `X9` 2004 to 2006 Amery Ice Shelf stations sit at
+60 m, 100 km from the mask's coast. A -10 m floor sends gauges at exactly 0 m,
atoll stations and ice stations to a review list instead of into the campaign.
Stations whose channel list has no vertical component (tide gauges, hydrophone
strings) are listed but never queued. **Cabled observatories stay in the OBS
track whatever their depth** (M. Denolle, 2026-10-03): any `NV` (Ocean
Networks Canada) or `OO` (OOI) station off the land mask is offshore, which
keeps the Saanich Inlet node at -8 m.

Implemented in [`scripts/select_offshore_stations.py`](../../scripts/select_offshore_stations.py)
(pixi environment, cartopy + shapely). It reads the three catalogue tables
from the bucket and writes
[`sb_catalog/configs/networks/offshore_stations.csv`](../../sb_catalog/configs/networks/offshore_stations.csv),
one row per station-location in any table with `on_land`, `coast_km`,
`offshore_class`, `has_vertical`, `days_1993_2026`, `source_tables` and an
`action`. With `--fdsn` it also sweeps the IRIS, NCEDC and SCEDC station
services and writes `offshore_stations_absent.csv`: offshore stations in
`NETWORK_MAPPING` networks that no table lists. Human decisions go into the
script's `REVIEWED` dict so the list stays reproducible.

```bash
pixi run python scripts/select_offshore_stations.py --write --fdsn
```

## 2. The list (run of 2026-10-03)

| action | station-locations | operating days 1993 to 2026 |
|---|--:|--:|
| keep-in-obs | 1,396 | 439,354 |
| return-to-global (land; obs picks archived, station goes to the next global onshore plan) | 1,993 | 976,774 |
| fill-from-global (offshore, has a vertical channel, not in obs) | 1,398 | 338,235 |
| none (land in global or western) | 52,131 | |

The return-to-global rows are also written in the catalogue table schema to
[`global_onshore_from_obs.csv`](../../sb_catalog/configs/networks/global_onshore_from_obs.csv).
They are **not** re-picked with `jma_wc` now: global will be relaunched with
a new picker, and that plan must include them with the other land stations.

Per code, what the 26-network list actually bought (keep = offshore by the
rule, return = everything else under that code in `obs/stations.parquet`):

| code | keep | return | keep days | return days |
|---|--:|--:|--:|--:|
| XJ | 15 | 437 | 85 | 92,649 |
| XO | 68 | 268 | 32,198 | 124,672 |
| 2F | 29 | 226 | 10,958 | 44,916 |
| YO | 125 | 185 | 11,694 | 91,184 |
| XZ | 46 | 131 | 274 | 78,524 |
| YR | 120 | 108 | 2,436 | 33,110 |
| YN | 64 | 94 | 1,369 | 216,206 |
| X9 | 54 | 89 | 20,115 | 26,220 |
| YS | 62 | 85 | 23,957 | 53,094 |
| X6 | 83 | 68 | 1,372 | 48,433 |
| ZF | 42 | 48 | 178 | 25,639 |
| 1V | 68 | 45 | 630 | 10,679 |
| 7D | 254 | 43 | 77,917 | 21,417 |
| ZU | 124 | 43 | 12,592 | 11,636 |
| 7A | 10 | 29 | 3,479 | 23,929 |
| Z5 | 64 | 28 | 21,455 | 20,364 |
| 3A | 3 | 25 | 203 | 19,691 |
| 7S | 0 | 19 | 0 | 14,554 |
| Z6 | 21 | 11 | 4,783 | 17,161 |
| 9A | 12 | 6 | 3,808 | 2,575 |
| ZS | 16 | 5 | 3,200 | 121 |
| NV | 27 | 0 | 127,637 | 0 |
| OO | 13 | 0 | 57,706 | 0 |
| 2D | 34 | 0 | 12,422 | 0 |
| 9R | 36 | 0 | 7,224 | 0 |
| 7K | 6 | 0 | 1,662 | 0 |

Only five codes were entirely offshore, two of them the cabled observatories.
`7S` never was.

**Fill from the global table.** 1,406 stations in `global/stations.parquet` are
offshore by the rule; 1,398 have a vertical channel. They are in 44 networks
and were picked with `jma_wc`, never with `obs`. The largest:

| code | stations | days 1993 to 2026 | where | depth m | years |
|---|--:|--:|---|--:|---|
| IM | 13 | 43,810 | 34.2, 34.3 | -2034 | 2017 to open |
| 8A | 94 | 37,791 | -4.6, -105.9 | -3272 | 2019 to 2022 |
| BK | 4 | 32,046 | 37.2, -122.6 | -528 | 1997 to open |
| XF | 64 | 30,151 | 17.5, 147.5 | -4504 | 2000 to 2013 |
| 3H | 49 | 22,104 | 0.4, -92.4 | -2945 | 2023 to 2024 |
| YL | 89 | 19,636 | -20.6, -176.2 | -2325 | 2009 to 2010 |
| ZD | 52 | 13,634 | -4.1, -104.5 | -3256 | 2007 to 2009 |
| XS | 36 | 13,120 | -0.5, -13.3 | -4037 | 2016 to 2017 |
| YI | 153 | 11,201 | 54.7, -134.3 | -914 | 2007 to 2022 |
| 7B | 30 | 10,817 | -34.5, -155.0 | -5301 | 2019 to 2020 |
| 8Q | 24 | 10,538 | 28.2, -141.5 | -4906 | 2021 to 2023 |

Of the 338,235 days, 84,856 fall in 1993 to 2009, which `global` never covered
at all, 248,721 in 2010 to 2025 and 4,658 in 2026.

**Absent from every table.** The FDSN sweep finds 507 offshore or shallow
stations in mapped networks that no catalogue table holds (58,864 operating
days). Channel-level metadata (fetched 2026-10-03 for all 507) says 203 of them
have a vertical channel in a `CHANNEL_PRIORITY` band, 45,346 station-days: `3J`
(Samoa 2023 to 2025, 27 stations), `YM` (Aleutians 2007 to 2020, 68), `OO` (5
more cabled nodes), `XB`, `2P`, `XW`, `YF`, one `NV`. The rest are
hydroacoustic (`IM`, `AU`), pressure-only (`7S` 2024 Mendocino, `8F`, `XF`
2012, `Z4`, `7D` 2011 extras) or beach nodes (`YB`, `ZI`). How the original
tables were built is not recorded in the repository, so why these are missing is
unknown; the fill table should be rebuilt from FDSN channel metadata for these
203, the way `plan_western_fill.fetch()` does it.

**Western is clean.** No western station is offshore by the rule. The 83
western stations below -50 m are Imperial Valley land.

### What the manifests hold against the list (checked 2026-10-03)

The 5,984 obs manifests name 1,769 stations with picks. 1,157 of them are
return-to-global land stations (100.1 M picks); 612 are offshore (29.4 M
picks). **784 of the 1,396 offshore stations never wrote a pick.** Their
planned station-days, by cause (a station can appear under more than one):

| cause | station-days | stations |
|---|--:|--:|
| `EL` channels only, skipped by `select_channel`, network-year also unreadable at the S3 access point | 27,100 | 493 |
| `EL` channels only, skipped by `select_channel`, access present | 22,299 | 366 |
| planned on metadata that is wrong: `NV.KEMF.B1` has HN channels for 96 days in 2020, the table gives it 2010 to open | 5,853 | 1 |
| data absent at the DMC too (`7D.M04A`, `7D.M05A` 2011 to 2012, `Z6.03` 2010; one hour requested, 404) | 658 | 3 |
| blocked on EarthScope 403 (2F Axial 2022 to 2023) | 2,482 | 9 |
| network-year unreadable, pickable band (XO 2019 class) | 1,692 | 6 |
| not run (2026) or completed empty on a readable network-year | 405 | 3 |

**The `EL` band is the finding.** 767 of the 1,396 offshore stations carry
only `EL1, EL2, ELZ` (plus an `EDH` hydrophone): the SIO and OBSIP
short-period fleet, Geospace GS-11D geophones at 200 Hz (YN 2009, X6 2012, YO
2014, ZU 2018, YR 2021, 1V 2023) and L28LB at 100 Hz (X9, Z5, 9R). `EL` was
dropped from `CHANNEL_PRIORITY` on 2026-09-02 because land pickers are not
trained on it (`constants.py`, "Low-gain and short-period-geophone bands were
dropped"), and [27](27_obs_literature_benchmark.md) already saw it on Blanco.
Campaign-wide it is 47,168 of 439,354 offshore operating days (11%) but 55% of
the offshore stations, and **846 of the 1,398 fill-from-global stations
(43,303 days) are `EL`-only too**, so the fill as sized below would skip them.
The data exist and are open: one hour of `ELZ` fetched from the DMC on
2026-10-03 for Z5.BS611 (2014), X9.BS010 (2013), 9R.OBS01 (2023), YO.201
(2014) and 1V.101 (2023) returned 1.2 to 2.9 MB each. X9 2013 is "missing" to
the S3 access survey and openly served by the DMC, so the two tiers differ.

Decision needed: whether the OBS track picks `EL` with the `obs` weight. The
2026-09-02 reasoning was about land training sets; whether PickBlue's training
corpus holds short-period OBS records is not recorded here and should be
checked before deciding. If yes, `select_channel` needs a per-weight priority
(or `EL` appended for the obs job definition only), the 767 kept stations and
846 fill stations are planned, and the empty X9, Z5, 9R, YO, YN, X6, ZU, YR,
1V shards are re-run.

### Decisions (M. Denolle, 2026-10-03)

| | decision |
|---|---|
| `NV.NSMTC.B1/B2/B3`, -8 m, Saanich Inlet, Ocean Networks Canada cabled node, 9,996 days | **keep**: cabled observatories stay in the OBS track; encoded as `CABLED_NETWORKS` and `REVIEWED` in the script |
| `YN.PARE.02`, 0 m, Punta Arenas, 343 days | return (coastal land, by the rule) |
| the 1,993 land stations' `obs`-weight picks | **archive** under `_archive/obs-land/`; nothing is deleted from the bucket |
| re-pick those 1,993 stations into `global` with `jma_wc` now | **no**: global will be relaunched with a new picker; the stations go into that plan through `global_onshore_from_obs.csv` |
| 96 shallow stations in global (201,629 days): 56 GeoNet tide gauges with no seismic channel, atoll and coastal stations at 0 m, Ross Ice Shelf | not queued; `CA` at 41.18, 1.75 (OBSEA) is the one worth a look |
| strip the 1,398 offshore stations' `jma_wc` picks out of `global/` | open; phase 2 after the fill, needs the same split over 202,468 global manifests |

---

## 3. Archive the land stations out of `obs/`

Pick objects are partitioned by network, year and month and named after the
shard, so most of the separation is a move, not a rewrite. Classified from the
manifests' per-station records on 2026-10-03 (land by elevation; the three
NV stations now kept carry no picks yet, so the counts stand):

| `obs/picks/` objects | count | size | picks (manifest) |
|---|--:|--:|--:|
| land-only partitions | 16,327 | 2.59 GB | 461.2 M land |
| sea-only partitions | 7,712 | 0.75 GB | 135.3 M sea |
| mixed partitions (a shard with both kinds of station) | 1,458 | 0.21 GB | 23.8 M land, 22.2 M sea |
| no manifest record (resumed or preempted shards) | 271 | 0.04 GB | |

Manifests: 4,349 land-only, 1,298 sea-only, 337 mixed. The pick totals here
are the manifests' sums over records; the Parquet footers are the exact count
and are what the verification uses ([measure-from-the-bucket](README.md)).

Mixed partitions are confined to twelve codes (XO, YS, Z6, 7A, YO, ZF, 3A, XZ,
XJ, YN, 1V, X9), where one shard grouped a code's land and sea stations of the
same year. Note that the mixed set includes AACSE 2018: its 30 land stations
are the benchmark's land control, and after the split their picks live under
`_archive/obs-land/`, not in `obs/`.

**Procedure, as a new script `scripts/split_obs_land.py`** (to write; the copy,
verify and delete primitives are `copy_one`, `listing` and `retry` in
[`unify_catalogue.py`](../../scripts/unify_catalogue.py)):

1. Load `offshore_stations.csv`; the land set is `action == return-to-global`.
2. For every object under `obs/picks/`, read the `tid` column only (3.6 GB
   total, about an hour from an EC2 instance in us-east-2, longer from a
   laptop) and classify it as land-only, sea-only or mixed. Do not trust the
   manifest classification for this step; the 271 unrecorded objects show why.
3. Land-only: copy to `_archive/obs-land/picks/<same relative key>`, verify by
   size and MD5 of the body (multipart ETags lie, see
   [29](29_one_prefix_per_catalogue.md)), then delete the source.
4. Mixed: split the rows by `tid`; write the sea rows back to the **same key**
   with `PICK_SCHEMA` and the land rows to the archive key; verify
   `rows_before == rows_sea + rows_land` from the footers before deleting
   nothing (the source is overwritten, so keep a copy of the original under
   `_archive/obs-land/mixed-originals/` until the campaign-wide check passes).
5. Manifests: land-only to `_archive/obs-land/manifests/`; mixed rewritten with
   the land `records` and `files` rows removed and `n_picks`, `station_days`
   recomputed; add `"split": {"land_records": n, "when": ...}` so the edit is
   visible.
6. `obs/stations.parquet`: back up to `stations.parquet.bak`, then write the
   1,396 keep rows. The daily
   `station-table.yml` check must pass afterwards: every station with picks in
   `obs/` is in the table, and no station in the table lacks a reason to be
   there.
7. Delete `obs/.dashboard/rowcount.json` so the dashboard rebuilds its cache
   from the new objects rather than serving the old count.
8. Leave `_queues/obs`, `_queues/obs-early`, `_queues/obs-dates` and the two
   repair queues as they are. They are complete or blocked and their
   `complete/` records are the only history of what ran. Their
   `stations.parquet` copies still list land stations, so **their targets must
   never be raised again**; say so in `fleet.json`'s comment.
9. `_queues/obs-2026` has no `complete/` directory, so it never ran. Delete its
   `shards.jsonl` and `stations.parquet` and re-plan it from the keep set
   (section 4).

Dry run first (`--dry-run` lists what would move, with counts and bytes, and
writes nothing). Verification after the run is in section 6.

## 4. Fill and complete the obs catalogue, same `obs/` prefix

Four pieces of work, all writing into `parquet_uri = s3://quakescope-picks-2026/obs`
with the `obs` weight (`quakescope_2026_obs:23`, image
`ghcr.io/seisscoped/quakescope:9dc5aa9`, the squash merge of PR #40 carrying the
resumed-shard fix, 8 vCPU / 16 GB, Fargate; read back from Batch with boto3 on
2026-10-03):

| queue | stations | span | planned station-days | note |
|---|--:|---|--:|---|
| `_queues/obs-fill` (new) | 1,398 from global + ~203 absent | 1993.001 to 2026.274 | ~384,000 | the planner clips to operating windows; the declared count will be close to the sum above. **846 of the 1,398 are `EL`-only and are skipped unless the `EL` decision above is yes**; without it the fill is 552 stations and ~295,000 days |
| `EL` re-run of kept stations (if the `EL` decision is yes) | 767 keep | their windows | 47,168 | the X9, Z5, 9R, YO, YN, X6, ZU, YR, 1V shards that completed empty |
| `_queues/obs-2026` (re-planned) | 1,396 keep | 2026.001 to 2026.274 | ~10,500 | today's queue has 26 sea-only shards (NV, OO) and 39 land-only |
| `_queues/obs`, blocked shards | 2F Axial 2022 and 2023 | | 7,231 | 50 sea-only shards blocked on EarthScope 403s; re-survey access and unblock when the embargo lifts. The other 285 blocked shards are land and stay blocked |
| XO 2019 and X9 2013 | | | 37,316 | recorded complete with zero picks ([obs-empty-completions](README.md)); run `python -m src.picker netyear-sweep` **from a Batch task** first, then a repair queue if the network-years exist |

Making the fill queue:

```bash
# 1. station table for the fill: the fill-from-global rows of offshore_stations.csv,
#    plus the 203 absent stations fetched at channel level (reuse plan_western_fill.fetch
#    and merge_epochs), written in the catalogue table schema
#    (id, network_code, station_code, location_code, channels, latitude, longitude,
#     elevation, start_date, end_date).
# 2. plan. --stations installs the table under the queue prefix even with --dry-run.
python -m sb_catalog.src.shard_planner \
    --campaign s3://quakescope-picks-2026/_queues/obs-fill \
    --start 1993.001 --end 2026.274 \
    --stations sb_catalog/configs/networks/obs_fill.csv --dry-run
# 3. without --dry-run writes the immutable shards.jsonl; then
#    - fleet.json: "obs-fill": {target 0, job_definition quakescope_2026_obs:23,
#      weight obs, procs 4, threads 2, queue _queues/obs-fill, parquet_uri obs}
#    - merge_station_tables.CONTRIBUTORS["obs"] += "_queues/obs-fill/stations.parquet"
#    - first Fleet run with a target starts the EarthScope access survey, not workers
```

Thresholds stay at the `obs` defaults recorded in `obs/runs/*.json`
(`p_threshold 0.2`, `s_threshold 0.2`, `components_loaded ZNE12`), so the fill
is indistinguishable from the rest of the catalogue.

**Cost.** The obs campaign to date cost about $425 (sum of `seconds` over 10,518
`complete/` records in six queues, 8 vCPU, $0.0213/vCPU-h, last attempts only)
for 1.48 M planned station-days, $0.00029 per planned station-day. The ~440,000
planned station-days above come to **$130, call it $100 to $200** with
preemption overhead. The 1,398 global-table stations are mostly deep-water
temporary deployments with short windows, so hit rates will be high and the
upper end is more likely than for the land campaigns.

## 5. Documents that now state something false

| where | what |
|---|---|
| [README.md](README.md) campaign table and [21_queues_written.md](21_queues_written.md) | "obs 3,389 stations, 6,566 shards, 996,536 station-days" counts 1,996 land stations |
| [11_launch_plan.md](11_launch_plan.md), "How onshore and offshore were split" | "Twenty-six networks qualify" is true of the codes where surveyed and false of the station table it produced |
| [27_obs_literature_benchmark.md](27_obs_literature_benchmark.md) | "All three are in the campaign's station table", true, alongside a Taiwan array; the AACSE land control's picks move to the archive |
| `reports/campaign_dashboard.html` | the obs map and headline count until the rowcount cache is rebuilt |
| `sb_catalog/configs/networks/earthscope_offshore.txt` | superseded by `offshore_stations.csv`; kept with a header saying so (done 2026-10-03) |
| `sb_catalog/configs/networks/earthscope_onshore.txt` | still "mapped minus the 26 codes"; its header now says the next global plan is per station: every `offshore_class == land` row of `offshore_stations.csv`, which adds the 1,993 stations of `global_onshore_from_obs.csv` and drops the 1,406 offshore stations hiding in the 420 codes |

## 6. Verification checklist

Each line is a command or an assertion, not a reading of a document.

- Before anything moves: footer row count over every `obs/picks/` object, and
  the same count split by land and sea `tid`. Record both in this document.
- After the split: `obs/picks/` footer total equals the sea count; the archive
  total equals the land count; the two sum to the original.
- `python scripts/merge_station_tables.py --campaign obs --check` reports zero
  stations with picks but no table row, and the table has 1,396 rows.
- No object under `obs/picks/` holds a `tid` in the return-to-global set
  (sample 500 objects after the run, read `tid` only).
- Dashboard: rebuilt, obs map shows no triangle on land.
- `global_onshore_from_obs.csv` has 1,993 rows and no id in common with the
  keep set; the next global plan's station table is checked against it.
- After the fill: `obs/runs/*.json` for the new run ids say `weight: obs`;
  `obs-fill` `complete/` records sum to the planned station-days less the
  blocked ones; `merge_station_tables --check` still passes with the
  contributor table merged.

---

## Appendix: how the numbers were measured

Station classes: `pixi run python scripts/select_offshore_stations.py --write --fdsn`
on 2026-10-03, against `obs/`, `global/` and `western/stations.parquet` as they
were that day (3,389, 53,082 and 26,377 rows). Completed station-days and
picks: every `obs/manifests/*.json` (5,984), `records` flattened to
(tid, yr, doy, npks), land by the table's elevation. Planned station-days: each
queue's `shards.jsonl`, station operating window intersected with the shard
window. Cost: every `_queues/obs*/complete/*.json` `seconds` field, attributed
to land by the shard's land share of station-days. Pick objects: a full listing
of `obs/picks/` (25,768 objects, 3.58 GB) joined to the manifest records on
(shard, network, year, month). Job definition: `boto3 batch
describe_job_definitions`, not the local `aws` CLI
([workflow_architecture_2026](README.md) on why). FDSN: station-level text
listings from IRIS (151,304 rows), NCEDC (4,763) and SCEDC (6,895) on
2026-10-03, channel level for the 507 absent candidates.
