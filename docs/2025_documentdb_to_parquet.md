# Converting the 2025 DocumentDB pick catalogue to Parquet

Specification for exporting the picks QuakeScope wrote into AWS DocumentDB in
2025 into the Parquet layout the 2026 campaigns publish, so both catalogues can
be read with the same code. Written 2026-10-08 for EarthScope engineers who
have access to the 2025 database's VPC.

**Status.** Everything below about the 2025 database comes from reading the
code that wrote it. Nobody on the QuakeScope side has connected to the
database to write this. Where the code cannot settle a fact, the text says
"unknown" and the fact appears in section 9. The exporter
(`scripts/export_documentdb_to_parquet.py`) has been tested against an
in-memory stand-in for the database (`tests/test_export_documentdb.py`, 12
tests passing on 2026-10-08), never against DocumentDB itself.

## 1. Terms

| term | meaning |
|---|---|
| pick | one phase arrival detected by the PhaseNet model (SeisBench implementation) on one station: a phase (`P` or `S`), a time, a confidence |
| station id, `tid` | `NET.STA.LOC`, the SEED network, station and location codes joined by dots. The location code may be empty, so `CI.CLC.` (trailing dot) is a valid id |
| band code, `cha` | the first two characters of a SEED channel code (`HH`, `BH`, `EH`, `HN`, ...). Picks are made on the three components of one band together, so a pick carries the band, not a full channel |
| station-day | one station, one UTC day. The unit the picker processes |
| station-day-channel | one station, one UTC day, one band code. The unit the 2025 resume table records |
| `picks_record` | 2025 collection with one document per station-day-channel that was processed, and how many picks it produced. It is how a resubmitted job knew what to skip |
| run, `rid` | one start of the picker process with one configuration. Every pick carries the id of the run that produced it; the run document holds model, weight, thresholds and library versions |
| weight | the trained parameter set loaded into PhaseNet. 2025 used `instance` |
| job (2025), shard (2026) | one AWS Batch task: in 2025, 40 stations by 20 days |
| manifest | 2026 per-job JSON listing the Parquet files the job wrote and one record per station-day-channel it processed |
| campaign | one launch of the picker over a work list. A published catalogue may be built from several campaigns |
| Hive partitioning | encoding partition keys in directory names (`network=CI/year=2019/month=07/`), which DuckDB, Athena, Spark, pandas and pyarrow all read as columns |

## 2. Where the 2025 database is

| fact | value | source |
|---|---|---|
| engine | AWS DocumentDB, MongoDB wire protocol, reached with `pymongo` | `sb_catalog/src/mongo_db.py:26` |
| region | us-east-2 | `docs/rerun_2026/archive/03_documentdb.md:28` |
| network access | only from inside the cluster's VPC, TCP 27017, TLS with the RDS `global-bundle.pem` trust store, `retryWrites=false` | `docs/rerun_2026/archive/03_documentdb.md:44-90` |
| database name | **unknown**. The notebooks that loaded stations and checked the run use `earthscope` (`notebooks/2_prepare_station_metadata.ipynb`, `notebooks/4_check_database.ipynb`); one later document uses `quakescope_2025` (`docs/rerun_2026/16_skypilot_vs_fargate.md:273`) | |
| code that wrote it | git tag `2025.05` (commit `96f1e25`, 2025-04-16). The write path is unchanged through the last 2025 commit `0a0df51` (2025-09-15): the only `picker.py` change in that span is whether the classifier loads | `git diff 2025.05 0a0df51 -- sb_catalog/src/picker.py` |
| container image | `ghcr.io/seisscoped/quakescope:latest`, so each job ran whatever `latest` was when it started | `sb_catalog/configs/job_definition_picking.yaml:5` at `2025.05` |

Line numbers below marked `@2025.05` refer to the file at that tag
(`git show 2025.05:<path>`); unmarked ones refer to `main` at `d2dbea1`.

## 3. What the 2025 database holds

### 3.1 Collections

`SeisBenchDatabase` names six collections (`mongo_db.py:38-45`) and creates a
seventh, `picks_record`, in its index setup (`mongo_db.py:69-75`).
Collections are created lazily on first write, so a collection exists only if
something wrote to it.

| collection | written by | contents | exported |
|---|---|---|---|
| `picks` | `picker.py:455-470 @2025.05` | one document per pick | yes, to `picks/` |
| `picks_record` | `picker.py:490-503 @2025.05` | one document per processed station-day-channel | yes, into `manifests/` |
| `sb_runs` | `utils.py:120-122 @2025.05` (`write_run_data`), called from `picker.py:216-230 @2025.05` | one document per run | yes, to `runs/` |
| `stations` | `notebooks/2_prepare_station_metadata.ipynb`, via `write_stations` (`utils.py:117-118 @2025.05`) | station metadata | yes, to `stations.parquet` |
| `classifies` | `picker.py:474-487 @2025.05` | QuakeXNet event-type classifications, BH and HH bands only | no (section 9, Q5) |
| `events` | `write_events`, `mongo_db.py:135-166`, only from the association step | associated events | no |
| `assignments` | same | event-to-pick links | no |

Whether `events` and `assignments` exist, and their sizes, is unknown. The
association step (`picker.py:244-293 @2025.05`) was a separate job type, and
`sb_catalog/src/plot_events_2025.py` reads `events`, which suggests it ran at
least once.

### 3.2 `picks` documents

Written at `picker.py:455-470 @2025.05`. One `insert_many` per
station-day-channel, unordered, duplicates on the unique index silently
skipped (`utils.py:157-180 @2025.05`; error code 11000 at line 175).

| field | BSON type as written | meaning | source of value |
|---|---|---|---|
| `_id` | ObjectId | assigned by the server | |
| `tid` | string | `NET.STA.LOC` | the `id` of the station in `stations` (`s3_helper.py:211,226 @2025.05`) |
| `cha` | string | band code, 2 characters | `t.stats.channel[:2]` (`picker.py:327 @2025.05`) |
| `pha` | string | `P` or `S` | SeisBench `Pick.phase` |
| `start` | date | start of the pick's probability window, UTC | `pick.start_time.datetime` |
| `peak` | date | the arrival time, UTC | `pick.peak_time.datetime` |
| `end` | date | end of the probability window, UTC | `pick.end_time.datetime` |
| `conf` | double | peak probability, 0 to 1 | `float(pick.peak_value)` |
| `amp` | double, may be NaN | Wood-Anderson amplitude (section 3.6) | `AmplitudeExtractor.extract_amplitudes` |
| `rid` | ObjectId | the run, `_id` of an `sb_runs` document | `self.run_id` |

There is no `amp_vel` field in 2025: it was added in 2026
(`picker.py:835`, commit `ba48712`). ObsPy's `UTCDateTime.datetime` is a naive
datetime in UTC and BSON stores dates as milliseconds since the epoch, so
every stored time is UTC at millisecond precision. The types above are what
the code wrote; what the server holds has not been checked (section 9, Q2).

### 3.3 `picks_record` documents

Written at `picker.py:490-503 @2025.05`, one per station-day-channel that
reached the picker, **after** that station-day-channel's picks.

| field | BSON type | meaning |
|---|---|---|
| `_id` | ObjectId | |
| `tid` | string | `NET.STA.LOC` |
| `cha` | string | band code |
| `yr` | int | year of the processed UTC day |
| `doy` | int | day of year, 1 to 366 (`strftime("%-j")`, no zero padding, cast to int) |
| `npks` | int | picks written for this station-day-channel, P and S together |
| `nclfs` | int | classifications written |
| `rid` | ObjectId | run |

### 3.4 `sb_runs` documents

Written by `write_run_data` (`utils.py:120-122 @2025.05`) with the arguments
at `picker.py:221-229 @2025.05`. A run id is created the first time a job
writes anything, so there is one document per job attempt that processed at
least one station-day.

| field | type as written | 2025 value, where the code fixes it |
|---|---|---|
| `_id` | ObjectId | the `rid` on picks and records |
| `model` | string | `PhaseNet` (default, `picker.py:71 @2025.05`; the job definition passes no `--model`) |
| `weight` | string | `instance` (default, `picker.py:77 @2025.05`; no `--weight` in the job definition) |
| `p_threshold`, `s_threshold` | double | 0.2 and 0.2 (defaults, `picker.py:81,84 @2025.05`) |
| `components_loaded` | string | `ZNE12` (default, `picker.py:66 @2025.05`) |
| `seisbench_version` | string | whatever the image had; unknown |
| `weight_version` | string or int | SeisBench's version of the weight file; unknown |
| `timestamp` | date | creation time, UTC (`utils.py:121 @2025.05`) |

The defaults are what the code would use if nothing overrode them. Whether
any job overrode them is unknown until the `sb_runs` values are read (section
9, Q3).

### 3.5 `stations` documents

Loaded from the per-network CSV files in `networks/*.zip` by notebook 2, after
two edits: `channels` reduced to the set of band codes, and `location_code`
rebuilt from `id`. CSV header (`networks/CI.zip` at `2025.05`):

`id, network_code, station_code, location_code, channels, latitude, longitude, elevation, start_date, end_date`

`start_date` and `end_date` are `YYYY.DDD` encoded as a float, and an open
epoch is `3000.001`. Because the CSV was read with plain `pd.read_csv`,
numeric-looking codes may be stored as numbers (a location code `00` as
`0.0`, a station code `1234` as an integer). The exporter rebuilds all three
code columns from `id`, which is always a string, and decodes the dates
numerically (`sb_catalog/src/utils.py:59-92`, `sb_catalog/src/s3_state.py:84-115`).

### 3.6 What `amp` measures in 2025

From `amplitude_extractor.py @2025.05`, lines 9-14 and 38-120, and
`docs/amplitude_conventions.md`:

| property | 2025 value |
|---|---|
| response removal | ObsPy `remove_response`, default output (velocity), `water_level=20`, `pre_filt=[0.02, 0.05, 40, 45]` |
| simulated instrument | Wood-Anderson, poles `-6.283 ± 4.7124j`, one zero, sensitivity 2080 (damping h = 0.8, period 0.8 s) |
| components | horizontals only (`NE12`) |
| window | 3 s before to 10 s after `peak`, cut from a window padded 10 s each side and deconvolved per pick |
| combination | mean of the per-component peak absolute values |
| missing | NaN when the response is missing or the window is empty |
| unit | metres of Wood-Anderson displacement, by construction; not checked on stored values |

The 2026 code uses h = 0.7, deconvolves the whole day once, and measures `amp`
only for `conf >= 0.5`. On five CI stations the damping change alone moves ML
by a near-uniform +0.033 (`docs/amplitude_conventions.md:26-31`). The export
copies 2025 `amp` unchanged; it is not comparable to 2026 `amp` without that
correction.

### 3.7 Indexes

Created by `_setup` (`mongo_db.py:48-75`, identical at `utils.py:37-64 @2025.05`)
the first time any `SeisBenchDatabase` connected, so before any pick was
written.

| collection | index name | keys | unique |
|---|---|---|---|
| `picks` | `pick_idx` | `tid, cha, pha, peak` | yes |
| `classifies` | `classify_idx` | `tid, cha, start` | yes |
| `stations` | `station_idx` | `id` | yes |
| `picks_record` | `picks_record_idx` | `tid, cha, yr, doy` | yes |

`docs/rerun_2026/16_skypilot_vs_fargate.md:275` says `picks` is "indexed on
(peak, tid)". No code creates such an index. Whether one was added by hand is
unknown; the dry run prints every index (section 7).

### 3.8 How the 2025 run decided what to process

This bounds what the database can say about coverage.

- **Planning.** Jobs covered 40 stations by 20 days (`submit_helper.py:44-45
  @2025.05`). Stations were filtered by operating epoch with
  `parse_year_day(str(start_date))` (`utils.py:183-192 @2025.05`). That
  formats the `YYYY.DDD` float as a string, so `2010.21` (day 210) parses as
  day 21. Every station whose start or end day of year is divisible by ten got
  the wrong epoch. The same misparse cost the 2026 western campaign 121,692
  station-days (`docs/rerun_2026/30_station_dates.md`).
- **Submission loop.** `while i < len(stations) - 1` and
  `days[min(j + 20, len(days) - 1)]` with an exclusive end
  (`submit_helper.py:86,94-98 @2025.05`) skip the last day of every submitted
  date range, and a final group holding a single station.
- **Skip on resume.** A station-day was skipped when every band listed for
  the station already had a `picks_record` document (`s3_helper.py:211-224
  @2025.05`), regardless of which run wrote it.
- **No data.** A station-day with no waveform data was not yielded
  (`s3_helper.py:265-271 @2025.05`) and left no `picks_record` document.
- **More than 150 traces.** A band whose day-long stream had more than 150
  traces (gaps) was replaced by an empty stream (`picker.py:333-338
  @2025.05`), picked as empty, and recorded with `npks: 0`
  (`picker.py:367-371 @2025.05`).
- **Files over 100 MB** were skipped as empty (`s3_helper.py:289 @2025.05`;
  raised to 200 MB on 2025-08-21, commit `c415c3c`).
- **All bands.** Every band present in the data was picked
  (`picker.py:327-344 @2025.05`), so a station with `HH` and `HN` has two sets
  of picks for the same day. The 2026 campaigns pick one band per station-day
  by a priority list (`sb_catalog/src/constants.py:509`).

## 4. Target layout

Identical to the 2026 catalogues (`sb_catalog/src/parquet_writer.py:16-20`,
`sb_catalog/src/s3_state.py:8-17`, `docs/data_access.md` section 3):

```
<root>/
    picks/network=<NET>/year=<YYYY>/month=<MM>/<job>[-NNN].parquet
    manifests/<job>.json
    runs/<rid>.json
    stations.parquet
    _export/<prefix>/...      exporter checkpoints, not part of the catalogue
```

`<job>` is `<prefix>-<NET>`, default `docdb2025-CI`. `-NNN` numbers the second
and later files of a partition, as `ParquetPickWriter._suffix` does
(`parquet_writer.py:259-261`). `year` is four digits, `month` two
(`parquet_writer.py:252-257`).

### 4.1 Pick schema

Copied from `parquet_writer.py:65-78`. The exporter imports it rather than
restating it; a test fails if its fallback copy drifts.

| column | Arrow type | note |
|---|---|---|
| `tid` | `string` | `NET.STA.LOC` |
| `cha` | `string` | band code |
| `pha` | `string` | `P` or `S` |
| `start` | `timestamp[ms]` | no time zone; values are UTC |
| `peak` | `timestamp[ms]` | arrival time |
| `end` | `timestamp[ms]` | |
| `conf` | `float32` | |
| `amp` | `float32` | NaN allowed |
| `amp_vel` | `float32` | nullable |
| `rid` | `string` | run id |

Files are written with `compression="zstd", use_dictionary=True`, as the 2026
writer does (`parquet_writer.py:339-342`). Inside a file rows are sorted by
`tid, cha, pha, peak`.

### 4.2 Manifest shape

Keys written by `ParquetPickWriter.close()` (`parquet_writer.py:467-493`),
and what the export puts in each:

| key | type | 2026 meaning | export value |
|---|---|---|---|
| `job_id` | string | the Batch job | `<prefix>-<NET>` |
| `run_id` | string | the job's single run | `null`: one network spans many 2025 runs; each record carries its own `rid` |
| `n_picks` | int | rows written | rows written for the network |
| `n_classifies` | int | classification rows | `0` |
| `station_days` | int | `len(records)` | same |
| `written_at` | ISO 8601 string | | export time |
| `files` | list of `{kind, path, rows, network, year, month}` | every object written | same |
| `records` | list of `{tid, cha, yr, doy, npks, nclfs, rid}`, all ints except the strings | one per station-day-channel | one per `picks_record` document |
| `outcomes`, `outcome_counts` | list, object | what happened to every planned station-day (2026-10-06 onward) | absent: 2025 never recorded it |
| `resumed` | object | present when a job resumed | absent |
| `export` | object | not in 2026 | `{source, database, collections, exporter, partitioned_by}` |

### 4.3 Run record shape

`S3CampaignState.write_run` (`s3_state.py:296-300`) writes
`{"run_id", "created", **meta}`, and the 2026 worker passes every value
through `str()` (`worker.py:247-250`). Example from `docs/data_access.md:158-160`:

```json
{"run_id": "00009e65-...", "created": "2026-09-06T08:57:51+00:00",
 "model": "PhaseNet", "weight": "original", "p_threshold": "0.2", "s_threshold": "0.2",
 "components_loaded": "ZNE12", "seisbench_version": "0.12.5", "weight_version": "2"}
```

### 4.4 Station table

The 2026 `stations.parquet` columns (`docs/data_access.md:130-135`, "Station
columns"): `id, network_code, station_code, location_code, channels, latitude,
longitude, elevation, start_date, end_date` and, where it came from a float,
`start_yearday, end_yearday`. Dates are Parquet `date32`, open epochs
`3000-01-01`. The 2026 western table also has `state`; the 2025 collection
does not.

## 5. Field mapping, 2025 to 2026

| 2025 (collection.field) | 2026 (object.column) | transformation |
|---|---|---|
| `picks.tid` | `picks.tid` | none; already `NET.STA.LOC` with an empty location kept as a trailing dot |
| `picks.tid` | partition `network=` | text before the first dot |
| `picks.peak` | partitions `year=`, `month=` | UTC year and month of `peak` (section 5.1) |
| `picks.cha` | `picks.cha` | none |
| `picks.pha` | `picks.pha` | none |
| `picks.start`, `.peak`, `.end` | same | BSON date to naive UTC `timestamp[ms]`. Precision is already ms, so nothing is lost. A client opened with `tz_aware=True` returns aware datetimes; they are converted to UTC and stripped |
| `picks.conf` | `picks.conf` | double to float32; relative error up to 6e-8 |
| `picks.amp` | `picks.amp` | double to float32, NaN kept. Different definition from 2026 `amp` (section 3.6) |
| (absent) | `picks.amp_vel` | null |
| `picks.rid` (ObjectId) | `picks.rid` | 24-character lowercase hex string, `str(ObjectId)` |
| `picks._id` | (dropped) | |
| `sb_runs._id` | `runs/<rid>.json` name and `run_id` | `str(ObjectId)` |
| `sb_runs.timestamp` | `created` | ISO 8601 with `+00:00` |
| other `sb_runs` fields | same keys | `str(value)`, as the 2026 worker does; `source: "documentdb-2025"` added |
| `picks_record.*` | `manifests/<job>.json` `records[]` | same seven keys; `rid` to string; `cha` null to `""` |
| `stations.*` | `stations.parquet` | codes rebuilt from `id`, dates decoded numerically, `_id` dropped |
| `classifies.*` | not exported | see Q5 |

### 5.1 Partitioning

The 2026 writer partitions a pick by the station-day it was processing
(`parquet_writer.py:188`). A 2025 document does not record that day, so the
export partitions by `peak`. The two differ only for a pick whose peak falls
after midnight of its processing day and in the next month; with day-long
streams that is a pick in the last seconds of a month. Manifest `records` keep
the processing day from `picks_record.yr/doy`, so per-day counts in the
manifest and per-month counts from the files can differ by those picks.

### 5.2 De-duplication

The key is the unique index, `(tid, cha, pha, peak)`. The index existed before
any pick was written (section 3.7), so the export expects zero duplicates and
reports the count per partition; any non-zero value is a finding to raise with
QuakeScope, not routine. Of two rows with the same key, the export keeps the
one with the smaller `rid` string, which makes reruns byte-identical.

Two 2025 runs re-picking the same station-day would produce the same key and
the second insert was dropped by the index: the stored pick is the first
run's. Picks that differ by even 1 ms in `peak` are different keys and both
survive. The export does not try to merge near-duplicates.

### 5.3 File sizing

The compaction guidance for the 2026 catalogues targets about 1 MB per file
(`sb_catalog/src/parquet_compact.py:43`, `docs/rerun_2026/22_parquet_compaction.md`).
That number was chosen to merge 49 KB fragments written by 1,500 concurrent
workers. An export has no such constraint, and the writer's own design note
puts 50 MB close to ideal (`parquet_writer.py:22-25`;
`docs/rerun_2026/archive/12_output_storage.md` section 4). The exporter
therefore defaults to **1,000,000 rows per file, about 29 to 35 MB**, split
evenly within a partition so the last file is not a runt. `--rows-per-file
30000` reproduces the 1 MB target if EarthScope prefers it (Q8).

## 6. What cannot be reconstructed

| missing in 2025 | consequence |
|---|---|
| per-station-day outcome (`loaded`, `no_data`, `denied`, ...) | a station-day with no `picks_record` document could have had no data, been outside the misparsed epoch, been the dropped last day of a range, or never been submitted. The database cannot say which |
| the reason for `npks: 0` | "picked, found nothing" and "more than 150 traces, not picked" are the same record |
| the processing day of a pick | partitioning uses `peak` (section 5.1) |
| `amp_vel` | null for every 2025 pick |
| 2026 `amp` | 2025 `amp` uses h = 0.8 and per-pick deconvolution; about +0.033 ML apart on CI stations, per-pick ill-conditioning on about 5 % of picks (`docs/amplitude_conventions.md`) |
| picks lost to a crash between the `picks` insert and the `picks_record` insert | if the job was not retried, its picks are in `picks` but the station-day has no record. Shows up as `npks` totals below row counts |
| the exact image each job ran | jobs used `:latest`; `seisbench_version` and `weight_version` in `sb_runs` are the only trace |
| classifier weight | not recorded in `sb_runs` (`docs/rerun_2026/archive/03_documentdb.md:127-130`) |

## 7. The exporter

`scripts/export_documentdb_to_parquet.py`. Read-only against the database.
Needs Python 3.10+, `pymongo`, `pyarrow`, `pandas`, `fsspec` (and `s3fs` for
an `s3://` destination). Run from a clone of the repository so it can import
`PICK_SCHEMA` and the station normalisation.

```bash
export QS_MONGO_URI='mongodb://<user>:<pass>@<cluster-endpoint>:27017/?tls=true&tlsCAFile=global-bundle.pem&retryWrites=false'

# 1. count only: per-network stations, picks, records, size estimate,
#    plus indexes and field types sampled from every collection
python scripts/export_documentdb_to_parquet.py --database <name> --dry-run --report dryrun.json

# 2. one small network end to end, then check it (section 8)
python scripts/export_documentdb_to_parquet.py --database <name> \
    --out s3://<bucket>/<prefix> --networks <NET> --staging-dir /data/staging

# 3. everything
python scripts/export_documentdb_to_parquet.py --database <name> \
    --out s3://<bucket>/<prefix> --staging-dir /data/staging
```

How it works:

1. **Station list.** Union of `stations.id` and `distinct(picks_record.tid)`,
   grouped by network.
2. **Per network, per station:** `find({"tid": tid})` with the `pick_idx`
   hint, streamed in batches of 10,000. Equality on the index's first key
   means each query touches only that station's index range. Rows are staged
   on local disk by (year, month).
3. **Per partition:** read the month back, sort, de-duplicate, write files of
   at most `--rows-per-file` rows, look `--spot-checks` random rows up again in
   the database by their unique key and compare every field.
4. **Manifest** from that network's `picks_record` documents; **run records**
   from `sb_runs`; **`stations.parquet`** from `stations`.

Resume: the unit is the network. `_export/<prefix>/network=<NET>.done.json`
marks a finished network, which a rerun skips. If `--staging-dir` survives a
crash, a network whose streaming finished is not streamed again, and each
written partition is recorded in `network=<NET>.progress.json` and not written
again. Without the staging directory the network is streamed again and its
files overwritten with identical content.

Resources: memory is bounded by the largest network-month partition, read
back whole for sorting. The 2026 western partition `CI` 2019-07 holds 5.5
million rows from one band per station-day (`docs/data_access.md`); 2025
picked every band, so the same month could be two to three times that.
Budget 16 GB of RAM and local disk of about 50 bytes per pick in the largest
network.

## 8. Validation

Built into the exporter, and stopping it on failure:

| check | where | fails when |
|---|---|---|
| streamed rows per station = `count_documents({"tid": tid})` | `_stream` | a cursor ended early or the station grew during export |
| rows written + duplicates dropped = rows streamed, per network | `export_network` | rows lost between staging and files |
| spot check: random rows looked up by `(tid, cha, pha, peak)`, all fields compared | `_spot_check` | any transformation error; recorded in `done.json` |
| total rows read = `picks.count_documents({})` | `main`, full runs only | picks whose `tid` is in neither `stations` nor `picks_record`; written to `_export/<prefix>/summary.json` |

Reported for review, not failing, in each `network=<NET>.done.json`:

| field | what to compare | expected difference |
|---|---|---|
| `rows_by_year` vs `picks_record_npks_by_year` | Parquet rows against the sum of `npks` | small: picks partitioned by `peak` across a year boundary, and section 6's crash case |
| `duplicates_dropped` | | 0 |
| `count_mismatches` | | empty |

Afterwards, from any machine that can read the destination:

```python
import duckdb
duckdb.sql("""
  SELECT network, year, count(*) AS picks, count(DISTINCT tid) AS stations,
         min(peak), max(peak), avg(conf), sum(isnan(amp)::int) AS amp_nan
  FROM read_parquet('<root>/picks/network=<NET>/*/*/*.parquet', hive_partitioning=true)
  GROUP BY ALL ORDER BY ALL
""")
```

and against the database, for the same network and year,
`count_documents({"tid": {"$in": tids}, "peak": {"$gte": Y, "$lt": Y+1}})`
should equal `picks`. One station-day worth checking by eye is `CI.CLC.`
2019-07-06, the Ridgecrest mainshock day, which the 2026 catalogue also holds.

## 9. Questions to answer before starting

1. **Which database name**, `earthscope`, `quakescope_2025` or another, and is
   the 2025 campaign in one database or several? `--dry-run` lists
   collections; `client.list_database_names()` lists databases.
2. **Stored field types.** Run `--dry-run`; its `field_types` per collection
   confirms or contradicts section 3. Particularly: is `rid` an ObjectId
   everywhere, are any times strings, are there documents with a `cha` of
   null?
3. **Run configuration.** Do all `sb_runs` documents say `weight: instance`,
   thresholds 0.2/0.2? If not, which networks or periods differ?
4. **Extra indexes.** Is there a `(peak, tid)` index as one document claims
   (section 3.7)? It changes nothing in the exporter but says the database was
   modified by hand.
5. **Classifications.** QuakeScope's 2026 catalogues are picker-only. Should
   the 2025 `classifies` collection be exported (to `classifies/` with the
   2026 `CLASSIFY_SCHEMA`, `label` null), archived as is, or dropped?
6. **Events and assignments.** Same question for the association output, if
   it exists.
7. **Destination.** Bucket, prefix, and whether it is published next to the
   2026 catalogues (`s3://quakescope-picks-2026/<catalogue>/`) or separately.
   The 2026 catalogues are one prefix per region; mixing 2025 picks into them
   would put two weights (`instance` vs `original`) in one prefix, which the
   2026 layout avoids.
8. **File size.** 1,000,000 rows (about 30 MB) per file, or the 2026
   compaction target of about 1 MB?
9. **Read load.** Can the export read from a replica instance, and is there a
   window when the cluster is otherwise idle? The exporter reads every pick
   once plus one `count_documents` per station.
10. **Snapshot.** Should the export run from a restored snapshot instead of the
    live cluster, so the database cannot change mid-export?

## 10. Size

The 2025 pick count is not recorded anywhere in this repository; the dry run
reports it. Parquet size then follows from two measured rates:

| source | bytes per pick |
|---|--:|
| published 2026 `western`: 42.0 GB over 1.47 billion picks (`docs/data_access.md`, 2026-09-18) | 28.6 |
| encoding test on an 800,000-pick batch (`docs/rerun_2026/archive/12_output_storage.md`) | 35 |

| 2025 picks | Parquet at 28.6 B | at 35 B | files at 1 M rows |
|--:|--:|--:|--:|
| 1 billion | 29 GB | 35 GB | about 1,000 plus one per small partition |
| 5 billion | 143 GB | 175 GB | about 5,000 plus |
| 10 billion | 286 GB | 350 GB | about 10,000 plus |

The 2025 `amp_vel` column is all null and costs almost nothing, so 2025 files
should sit near the low end. A cross-check without querying: the same encoding
test measured BSON at 148 bytes per pick before indexes and about 228 with
them, so the cluster's `VolumeBytesUsed` metric divided by 148 to 228 brackets
the pick count from above (other collections share the volume).
