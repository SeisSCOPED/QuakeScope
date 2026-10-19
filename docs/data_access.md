# Accessing the QuakeScope pick catalogue

Machine-learning phase picks for the western United States (`western`), for
ocean-bottom seismometers (`obs`), and the 2025 global catalogue of Ni et al.
(2025) (`quakescope-2025`), published as Parquet on S3. **Anonymous read, no
account, no database.** Read-path timings were measured on 2026-09-16; the
catalogue table in section 2 names the source and date of each number.

Two notebooks walk through this:

| | | |
|---|---|---|
| [`tutorials/read_the_catalogue.ipynb`](../tutorials/read_the_catalogue.ipynb) | start here: the three catalogues, one month read and de-duplicated, the availability table, 2025 against 2026 on the same station-days, picks on waveforms, an OBS month | runs on Colab, ~5 min |
| [`tutorials/download_the_catalogue.ipynb`](../tutorials/download_the_catalogue.ipynb) | bulk: choose a region and period, mirror it, query it locally, check coverage, export | laptop, ~10 min for a network-year |

Rendered copies with output: [read_the_catalogue.html](https://seisscoped.org/QuakeScope/read_the_catalogue.html),
[download_the_catalogue.html](https://seisscoped.org/QuakeScope/download_the_catalogue.html).

---

## 1. The short version

```bash
# one month of the Southern California network, 158 MB, 338 files (2026-10-19), ~10 s
aws s3 sync --no-sign-request \
    s3://quakescope-picks-2026/western/picks/network=CI/year=2019/month=07/ \
    western/picks/network=CI/year=2019/month=07/
```

```python
import pandas as pd
picks = pd.read_parquet("western/picks/")          # 5.5 M rows in under a second
```

That is the whole recipe: **sync the partitions you want, then read them
locally.** Reading straight from S3 works too, and the notebooks show it, but
the same month takes 86 s through `pandas` + `s3fs` against 9 s to sync plus
0.8 s to read.

## 2. What is published

Bucket **`s3://quakescope-picks-2026`**, region **us-east-2**, one prefix per
catalogue. Public read is granted on each catalogue's `picks/`, `manifests/`,
`runs/` and `stations.parquet`, and on listing. Everything else in the bucket
starts with an underscore: `_queues/` holds the per-era work queues and their
claim and completion state (not public), `_archive/` holds earlier attempts
and test runs. Every public object is also reachable over plain HTTPS at
`https://quakescope-picks-2026.s3.us-east-2.amazonaws.com/<key>`.

| catalogue | weight | years | Parquet objects | size | picks | state |
|---|---|---|--:|--:|--:|---|
| `western` | `original` | 1986 to 2026 (to September) | 580,407 | 55.3 GB | 2.13 B rows, ~2.07 B unique | complete; last write 2026-10-10 |
| `obs` | `obs` (PickBlue) | 1993 to 2026 | 16,934 | 2.9 GB | 104.4 M rows, ~103.6 M unique | complete, offshore stations only; last write 2026-10-09 |
| `global` | `jma_wc` | 2010 to 2026 | 22,460 | 15.8 GB | 583 M | paused at a few percent (2026-10-08 figures) |
| `quakescope-2025` | `instance` | 2002 to 2025-03 | 42,570 | 88.3 GB | 4.38 B | the 2025 catalogue, converted; see below |

Rows: the exact sum of `num_rows` over every Parquet footer, read on
2026-10-13 (`_archive/inventory-20261013/`: western 2,130,446,013, obs
104,420,230; global 583,482,953 on 2026-10-08). Objects, sizes and last-write
times: boto3 listing of 2026-10-19. "Unique" removes the duplicate rows
explained below. The [campaign dashboard](https://seisscoped.org/QuakeScope/campaign_dashboard.html)
shows queue progress hourly; its pick totals come from a cache that can lag
the bucket, so cite the footer count with its date.

Each catalogue was produced by several campaigns run in eras, because a work
queue is immutable once written (`western-early` for 1986 to 2009, `western`
for 2010 to 2025, `western-2026`), all with the same weight and thresholds and
all writing into the same prefix, so a reader never has to know which era a
year came from. Pick objects are named by the shard that wrote them and the
`year=` partition runs continuously across the eras.

**Western** is the stakeholder deliverable and the one to start with: 26,377
station-locations, read from the SCEDC, NCEDC and EarthScope archives. 24,008
are inside the state polygons of Washington, Oregon, California, Nevada, Idaho
and Wyoming; the rest of the stakeholder list (Utah, Montana, Arizona,
Colorado, New Mexico, British Columbia, Alberta, Baja California, Sonora) was
added by `western-fill` and `western-fill2` and merged into
`stations.parquet` on 2026-10-02. Offshore stations are in `obs`.

**Re-read, 2026-10-06 to 10-09.** About 15.6 million western station-days
inside the stations' operating epochs had never been read, mostly because the
reader chose one band per station for the whole campaign (fixed in image
d0ccf9b); `western-reread2` re-read them into the same prefix (`obs-reread` did
the same for `obs`), and the `too_big` repairs followed. The reason and the
sizing are in [32_outcomes_and_band_fallback.md](rerun_2026/32_outcomes_and_band_fallback.md).

**Duplicate rows: de-duplicate on `(tid, cha, pha, peak)` when you read.**
Spot interruptions left the partial output of interrupted shard attempts in
the bucket, and the shard was re-run from the start. On 2026-10-13, 12,125
western objects (69,690,152 rows) were listed by no manifest; in a random
sample of 200 of them, 92.2 % of rows repeated a pick in a listed object
(obs: 237 objects, 82.0 %). Estimated: ~64 M duplicate rows in `western`,
~0.8 M in `obs`. The picker is deterministic, so a repeated key is the same
pick under another `rid`. The duplicates will be removed from the bucket,
which renames objects without changing picks.

**`obs` is offshore only** since 2026-10-03: 1,996 stations selected by reused
temporary network codes were on land; their picks moved to
`_archive/obs-land/` (not public) ([31_obs_station_selection.md](rerun_2026/31_obs_station_selection.md)).

**Not compacted.** The catalogue was written by up to 1,500 concurrent
workers, so a month partition holds hundreds of files of about 120 KB each. Reads are
correct but pay per object, which is why every rule in section 4 is about
touching fewer objects. Compaction is planned and will change object names,
not content.

### The 2025 catalogue, `quakescope-2025/` (published 2026-10-14)

The picks of the 2025 run (Ni et al., 2025, *Seismica*), converted from the
retired DocumentDB database into the same layout and schema as the 2026
catalogues, so one reader works for both. Public, anonymous, no account.

| | |
|---|---|
| picks | 4,376,745,115, in 453 networks (each network's count equals the database's) |
| stored | 88.3 GB of picks in 42,570 objects (S3 listing, 2026-10-19), at `quakescope-2025/picks/network=<NET>/year=<YYYY>/month=<MM>/` |
| model | PhaseNet, `instance` weights, P and S thresholds 0.2 (run records in `quakescope-2025/runs/`) |
| station table | `quakescope-2025/stations.parquet`, codes rebuilt from `id`; `start_date` / `end_date` keep the database's zero-padded `YYYY.DDD` strings (`2019.227`, `3000.001` for open), parse with `datetime.strptime(s, "%Y.%j")` |

How it differs from 2026 (details in `docs/2025_documentdb_to_parquet.md` on [PR #56](https://github.com/SeisSCOPED/QuakeScope/pull/56), not yet merged):
`amp_vel` is empty (2025 did not measure it); `amp` used Wood-Anderson damping
h = 0.8, not 0.7; 2025 picked every band on a station-day, not one; `rid` is the
2025 database's ObjectId as text; the manifests carry `records` from the 2025
`picks_record` table but no `outcomes`, so a 2025 station-day can be "read" or
"unknown", never "no data". In 2025, `npks` = 0 also meant "more than 150
traces, skipped". The two weight sets also place arrivals differently: on CI
in July 2019, matched `original` picks are later than `instance` picks by a
median of about 50 ms (P) and 100 ms (S), consistent with their median
residuals against analyst picks ([benchmark/results/timing_full.csv](benchmark/results/timing_full.csv)).
Do not mix the two catalogues in one location without a per-phase correction.

## 3. Layout and schema

```
s3://quakescope-picks-2026/<campaign>/
    picks/network=<NET>/year=<YYYY>/month=<MM>/<shard_id>[-NNN].parquet
    manifests/<shard_id>.json     what each job wrote: object keys, per-station-day pick counts,
                                  and (image d0ccf9b on) one outcome per planned station-day
    runs/<run_id>.json            model, weight, thresholds, library versions
    stations.parquet              the station table the campaign was planned from
```

`network`, `year` and `month` are Hive partition keys: they live in the path,
not in the file, and every reader below turns them into columns.

### Pick columns

| column | type | meaning |
|---|---|---|
| `tid` | string | `NET.STA.LOC`; the location code may be empty, so `CI.CLC.` is a valid id |
| `cha` | string | band + instrument code the station-day was picked on: `HH`, `EH`, `BH`, `HN`, ... One code per station-day, the first of `constants.CHANNEL_PRIORITY` (HH > EH > SH > BH > BN > DP > HN > CN; `obs` adds EL) **present in that day's data**. Before image d0ccf9b (2026-10-06) the band was chosen once per station for the whole campaign, and days without it were not read; see [32_outcomes_and_band_fallback.md](rerun_2026/32_outcomes_and_band_fallback.md) |
| `pha` | string | `P` or `S` |
| `start`, `peak`, `end` | timestamp, ms | the pick's probability window; **`peak` is the arrival time** |
| `conf` | float32 | peak probability, 0 to 1. Everything at or above **0.2** is stored |
| `amp` | float32 | Wood-Anderson displacement, metres, mean of the horizontal peaks, response removed. **Measured only for `conf` >= 0.5**, NaN otherwise and inside the 60 s taper at trace ends |
| `amp_vel` | float32 | peak ground velocity near the pick, **m/s**: counts divided by the per-channel instrument sensitivity and high-passed, max over components (`AmplitudeExtractor.extract_velocity_amplitudes`, since ba48712 of 2026-08-30, so for every 2026 catalogue). Flat-response approximation, valid in the instrument's passband; set on 99 % of picks |
| `rid` | string | run id, resolves to `runs/<rid>.json` |

`conf` is a detection score, not a probability that the pick is correct, and
0.2 is deliberately permissive. Most uses want more; choose the value on your
own target events rather than adopting one. Amplitude conventions (IASPEI
Wood-Anderson constants, whole-day deconvolution, taper rule):
[amplitude_conventions.md](amplitude_conventions.md).

### Station columns

**Join on `id`.** `id` equals the picks' `tid` and has always been right. The
separate code columns carry CSV-parse damage that is fixed in code
(`scripts/fix_station_codes.py`, PR #49) but **not yet rewritten in the
bucket**: on 2026-10-19, 2,065 western rows still hold a float-like
`location_code` (`1.0` for `01`, `0.0` for `00`), and in `global`
`station_code` lost leading zeros (`1` for `001`) and network `NA` is empty.
Derive network, station and location from `id.split(".")`.

`id` (= `tid`), `network_code`, `station_code`, `location_code`, `channels`
(the bands the archive lists, e.g. `HH` or `DP,EH`), `latitude`, `longitude`,
`elevation` (m), `start_date`, `end_date`, `state`.

**`start_date` and `end_date` are Parquet dates** (`date32`), so they compare
directly with a `datetime.date` and need no decoding. A station still
operating carries **3000-01-01** rather than a null, so `end_date >= when` keeps
it instead of dropping it. Until 2026-09-29 both columns were a `YYYY.DDD`
float, which is still there as `start_yearday` / `end_yearday`; that float is
three zero-padded digits after the point, so `2010.21` is day **210**, and
formatting it as a string to parse it reads day 21 instead. Two of our own
planners did that, which is why the columns changed:
[30_station_dates.md](rerun_2026/30_station_dates.md). Western has 26,377 rows
since the fill stations were merged in on 2026-10-02.

### Run records

```json
{"run_id": "00009e65-...", "created": "2026-09-06T08:57:51+00:00",
 "model": "PhaseNet", "weight": "original", "p_threshold": "0.2", "s_threshold": "0.2",
 "components_loaded": "ZNE12", "seisbench_version": "0.12.5", "weight_version": "2"}
```

A campaign runs under one configuration, but each worker start creates a new
run id, so a campaign has thousands of run records that differ only in
`run_id` and `created`. Quote the configuration, not the id.

## 4. Reading it: three rules

Measured on `western`, anonymous, from a laptop in Seattle on 2026-09-16.

| operation | objects | time |
|---|--:|--:|
| list one month partition (`CI`, 2019-07) | 271 | 1 s |
| `aws s3 sync` that month, 149 MB | 271 | 9 s |
| `s3fs.get(..., recursive=True)` of the same | 271 | 12 s |
| `pandas.read_parquet` of the synced month, 5.49 M rows | 271 | 0.8 s |
| `pandas.read_parquet` of the month straight from S3, `filters=` | 271 | 86 s |
| DuckDB `count(*)` with the partition in the path | 271 | 7 s |
| DuckDB one station, `conf >= 0.5`, partition in the path | 271 | 2 s |
| DuckDB monthly aggregate with `year=2019/*` in the path, 715 MB | 3,220 | 142 s |
| DuckDB `picks/**/*.parquet` with `WHERE network= AND year= AND month=` | 276,829 | **did not return in 7 min** |

**Rule 1: put the partition in the path.** `read_parquet('.../picks/**/*.parquet') WHERE network='CI'`
lists every object in the campaign before the `WHERE` can prune anything.
`read_parquet('.../picks/network=CI/year=2019/month=07/*.parquet')` lists 271.
`pandas`' `filters=` argument prunes correctly, but still pays per-file
latency.

**Rule 2: for anything larger than a month, sync first.** The AWS CLI and
`s3fs` both fetch concurrently and land at 15 to 20 MB/s; every query after
that is local. A network-year is 0.7 GB; the whole of `western` is 55.3 GB (2026-10-19) and
580 k objects, which `aws s3 sync` handles in about two hours on a fast link.

**Rule 3: aggregate in the engine, select only the columns you need.** The
catalogue is far larger than any laptop. A count, a histogram, a per-station
summary should come back reduced. Parquet is columnar, so leaving `amp` and
`amp_vel` out of a query that wants arrival times halves the bytes read.

### AWS CLI

```bash
aws s3 ls --no-sign-request s3://quakescope-picks-2026/western/picks/network=CI/year=2019/
aws s3 sync --no-sign-request s3://quakescope-picks-2026/western/picks/network=UW/year=2019/ western/picks/network=UW/year=2019/
aws s3 cp --no-sign-request s3://quakescope-picks-2026/western/stations.parquet .
```

`--no-sign-request` is what makes it anonymous; without it the CLI looks for
credentials and fails when it finds none.

### Python

```python
import pandas as pd
ANON = {"anon": True}

stations = pd.read_parquet("s3://quakescope-picks-2026/western/stations.parquet", storage_options=ANON)

# straight from S3: fine for a month, slow beyond it
picks = pd.read_parquet("s3://quakescope-picks-2026/western/picks/",
                        filters=[("network", "=", "CI"), ("year", "=", 2019), ("month", "=", 7)],
                        storage_options=ANON)

# mirror, then read locally: the fast path
import s3fs
fs = s3fs.S3FileSystem(anon=True)
fs.get("quakescope-picks-2026/western/picks/network=CI/year=2019/", "western/picks/network=CI/year=2019/", recursive=True)
picks = pd.read_parquet("western/picks/")
```

### DuckDB

```python
import duckdb
con = duckdb.connect()
con.sql("INSTALL httpfs; LOAD httpfs; SET s3_region='us-east-2'; "
        "SET s3_access_key_id=''; SET s3_secret_access_key='';")   # anonymous
con.sql("""
    SELECT tid, count(*) AS picks, round(avg(conf), 3) AS mean_conf
    FROM read_parquet('s3://quakescope-picks-2026/western/picks/network=CI/year=2019/month=07/*.parquet',
                      hive_partitioning = true)
    WHERE conf >= 0.5
    GROUP BY tid ORDER BY picks DESC LIMIT 10
""").df()
```

Against a local mirror, replace the `s3://` path with `western/picks/**/*.parquet`;
a glob is fine on disk.

### HTTPS, no tooling

```
https://quakescope-picks-2026.s3.us-east-2.amazonaws.com/?list-type=2&prefix=western/picks/network%3DCI/year%3D2019/month%3D07/
https://quakescope-picks-2026.s3.us-east-2.amazonaws.com/western/picks/network=CI/year=2019/month=07/2019174-2019194-033f264dd09f.parquet
https://quakescope-picks-2026.s3.us-east-2.amazonaws.com/western/runs/00009e65-e358-4f89-869c-3871f5f28b69.json
```

## 5. Selecting by place and time

`stations.parquet` is the index. Filter it by `state`, by a latitude and
longitude box, or by network, take the distinct networks as
`id.str.split(".").str[0]` (not the `network_code` column, see section 3), and sync
`picks/network=<NET>/year=<YYYY>/` for each network and year in your window.
Then filter the rows on `tid` for the stations you kept, because a network
partition holds every station of that network, not only the ones in your box.

## 6. Coverage: was this station-day picked at all?

A station-day with no rows may mean the archive held no data, or that the
campaign did not read it. **Before image d0ccf9b (2026-10-06) the catalogue
recorded both cases the same way, so for those shards absence is not evidence
of an empty archive.** The mechanisms: a gap rule that dropped fragmented day
files (fixed 2026-09-10), day listings that failed silently, network-years the
archive does not hold, and, by far the largest, the reader choosing one band
per station for the whole campaign, which skipped every day that band was
absent (fixed in d0ccf9b; [32_outcomes_and_band_fallback.md](rerun_2026/32_outcomes_and_band_fallback.md)).
Of 53.6 M pickable western station-days inside FDSN epochs, 14.0 M were
recorded in manifests on 2026-10-06; `western-reread2` re-reads the rest
(NP excluded: no data in a 1,408-day sample).

**From d0ccf9b on, the manifest answers the question.** `outcomes` holds one
entry per planned station-day (`tid`, `yr`, `doy`, `status`, and `cha` or
`detail` where they apply). Final statuses: `loaded` (read and picked, see
`records` for the count), `no_data` (nothing in the archive listing),
`no_channel` (the station offers no pickable band), `empty_read` (an object
exists but holds no pickable band; `detail` names what it holds),
`not_found` (the network-year is not in the archive), `denied` (our account
may not read it), `done` (already picked by an earlier attempt). Not read:
`refused`, `throttled`, `timeout`, `read_error`, `too_big`; these are also
written to the queue's `review/` and are re-run by repair queues.

What the public objects do let you check: `manifests/<shard_id>.json` lists
the station-days the shard **processed**, with its pick count (`records`:
`tid`, `cha`, `yr`, `doy`, `npks`). A station-day present there with `npks: 0`
was read and found no arrival above 0.2. Whether a station-day *has* picks
should be read from the Parquet itself, not from the manifest: a shard that
was preempted and resumed by another worker writes a manifest covering only
the resuming attempt, and its first attempt's earliest checkpoint files are
overwritten ([28_resumed_shard_overwrite.md](rerun_2026/28_resumed_shard_overwrite.md):
308 shards and at most 11,582 station-day-channels in `western`, 0.15 % of
those processed, always at the start of the shard's 20-day window). The
download notebook builds the coverage table this way for a region and period.

### The availability table: one row per station-day

`<catalogue>/availability/network=<NET>/availability.parquet` answers "does
this station have data on this day, and was it picked?" without fetching any
manifest. Built by `scripts/build_availability.py` from every manifest
(`records` and `outcomes`), public-read, partitioned by network.

| column | meaning |
|---|---|
| `tid` | `NET.STA.LOC`, joins to picks `tid` and stations `id` |
| `date` | the day |
| `status` | `loaded` (read and picked), `no_data`, `not_found`, `no_channel`, `empty_read`, `denied`, `unread` (attempted, not read; `detail` says why, e.g. `too_big`), `unknown` (inside the station's operating epochs, no record either way) |
| `cha` | the band read, for `loaded` |
| `npks` | picks that day, for `loaded`; **0 means the station had data and no arrival above 0.2** |
| `detail` | reason for `unread`, or what an `empty_read` object held |

Rebuilt after the `too_big` repairs; counts read from the public tables on
2026-10-19 (`docs/paper_2026/figures/numbers.json`):

| status | western | obs |
|---|--:|--:|
| loaded | 16,965,900 | 449,541 |
| of which `npks` = 0 (data, no picks) | 1,368,764 | 20,009 |
| no_data | 11,627,811 | 47,481 |
| not_found | 931,317 | 143,309 |
| empty_read | 207,332 | 10,944 |
| unread | 1,030 | 0 |
| unknown | 24,621,886 | 120,556 |

Most western `unknown` is NP (about 23 M; no data in a 1,408-day sample, so not
re-read) and the networks our account cannot read (TD, EO, LH). The 1,030
`unread` are all `too_big`, 1,010 on SF and 20 on X7: day files that crash the
miniSEED decoder or exceed 4 GB.

Two cautions. The table is built from manifests, so a shard preempted and
resumed on Spot leaves its first days `unknown` even though their picks are in
the Parquet (24 station-days of CI in July 2019, for example): whether a
station-day has picks, ask the Parquet; whether a station-day without picks
had data, ask the table. And it covers days inside each station's FDSN channel
epochs, which can be narrower than the `start_date` to `end_date` window of
`stations.parquet`.

```python
import pandas as pd
a = pd.read_parquet("s3://quakescope-picks-2026/western/availability/network=UW/availability.parquet",
                    storage_options={"anon": True})
a[(a.status == "loaded") & (a.npks == 0)]        # data, no picks
```

## 7. Provenance and citation

The western catalogue is PhaseNet with the `original` weights (Zhu and Beroza,
2019, as packaged by SeisBench), P and S thresholds 0.2, components ZNE12,
SeisBench 0.12.5, weight version 2, run 2026-09-03 to 2026-09-17 on AWS Batch
Fargate Spot from image `ghcr.io/seisscoped/quakescope` at the commits pinned
in the campaign job definitions (`fleet.json`); the three eras and the
2026-09-17 repair share that configuration and are told apart by `rid`. The
re-read (`western-reread2`, 2026-10-06 to 10-09) uses the same model, weight and
thresholds on image d0ccf9b, whose reader picks the best band present each
day; its picks carry their own `rid`, and its `runs/<rid>.json` names the image. The picks reproduce exactly
when re-picked through ObsPy/FDSN on another architecture
([western_pick_validation.html](https://seisscoped.org/QuakeScope/western_pick_validation.html)).

Code and workflow: [SeisSCOPED/QuakeScope](https://github.com/SeisSCOPED/QuakeScope),
MIT, [CITATION.cff](../CITATION.cff). The previous catalogue and the method are
described in Ni et al. (2025, Seismica, doi:10.26443/seismica.v4i2.1738). A
licence statement and a DOI for the 2026 catalogue itself are still to be
issued; until then cite the repository and name the campaign prefix and the
date you read it, because the bucket is live.

## 8. Known limits

- Per-station picks, not events. A phase associator (PyOcto, GaMMA) turns them
  into a catalogue; `tutorials/seisbench_pyocto_ncedc.ipynb` shows the shape of
  that step.
- One band per station-day. A station with both `HH` and `HN` was picked on
  `HH`; the accelerometer was not run. Since d0ccf9b the band is the best one
  present that day, so one station can carry `EH` picks for early years and
  `HH` later.
- `amp` exists only above `conf` 0.5. Magnitudes from this catalogue are
  magnitudes of the confident picks.
- Duplicate rows (section 2) until they are removed: de-duplicate on
  `(tid, cha, pha, peak)`.
- 1,030 western station-days are recorded as `unread` (`detail` `too_big`),
  1,010 on SF and 20 on X7 (availability table, 2026-10-19): day files the
  reader could not open even on a 120 GB worker, because they crash the
  miniSEED decoder or exceed 4 GB.
- The bucket is live: duplicate removal and compaction will rename objects
  without changing picks. Record the date of any pull.
- The fill stations (2,369 station-locations) are merged into
  `stations.parquet` and picked; the date-parsing re-pick
  (`western-dates`, [30_station_dates.md](rerun_2026/30_station_dates.md)) is
  complete. Networks our EarthScope account may not read (TD, EO, LH in
  western; NV in obs) are not in the catalogue.
