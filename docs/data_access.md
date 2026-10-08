# Accessing the QuakeScope pick catalogue

Machine-learning phase picks for the western United States (and, as they
finish, the ocean-bottom and global campaigns), published as Parquet on S3.
**Anonymous read, no account, no database.** Everything below was verified
against the bucket on 2026-09-16; the table of campaigns names the source of
each number.

Two notebooks walk through this:

| | | |
|---|---|---|
| [`tutorials/read_the_catalogue.ipynb`](../tutorials/read_the_catalogue.ipynb) | first contact: read a month, plot it, check picks on the waveforms | runs on Colab, ~5 min |
| [`tutorials/download_the_catalogue.ipynb`](../tutorials/download_the_catalogue.ipynb) | bulk: choose a region and period, mirror it, query it locally, check coverage, export | laptop, ~10 min for a network-year |

Rendered copies with output: [read_the_catalogue.html](https://seisscoped.org/QuakeScope/read_the_catalogue.html),
[download_the_catalogue.html](https://seisscoped.org/QuakeScope/download_the_catalogue.html).

---

## 1. The short version

```bash
# one month of the Southern California network, 149 MB, 271 files, ~10 s
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
| `western` | `original` | 1986 to 2026 (to September) | 490,018 | 46.8 GB | 1.81 B | **re-read in progress** from 2026-10-08 (`western-reread2`, about 15.6 M station-days never read before); counts grow while it runs |
| `obs` | `obs` (PickBlue) | 1993 to 2026 | 14,789 | 2.6 GB | 93.1 M | offshore stations only since 2026-10-03; `obs-reread` queued |
| `global` | `jma_wc` | 2010 to 2026 | 22,460 | 15.8 GB | 583 M | paused at a few percent |

Picks: the exact sum of `num_rows` over every Parquet footer of each
catalogue, read anonymously on 2026-10-08 (western 1,810,076,434, obs
93,138,251, global 583,482,953; no footer failed). Objects and sizes: boto3
listing of the same day. Western was counted while the re-read was writing, so
it is a snapshot. The [campaign dashboard](https://seisscoped.org/QuakeScope/campaign_dashboard.html)
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

**Being re-read (from 2026-10-08).** About 15.6 million western station-days
inside the stations' operating epochs were never read, mostly because of a
reader defect fixed in image d0ccf9b; `western-reread2` is re-reading them and
writes into the same prefix (`obs-reread` does the same for `obs`). Pick counts
will grow while it runs. The reason and the sizing are in
[32_outcomes_and_band_fallback.md](rerun_2026/32_outcomes_and_band_fallback.md).

**`obs` is offshore only** since 2026-10-03: 1,996 stations selected by reused
temporary network codes were on land; their picks moved to
`_archive/obs-land/` (not public) ([31_obs_station_selection.md](rerun_2026/31_obs_station_selection.md)).

**Not compacted.** The catalogue was written by up to 1,500 concurrent
workers, so a month partition holds hundreds of files of about 120 KB each. Reads are
correct but pay per object, which is why every rule in section 4 is about
touching fewer objects. Compaction is planned and will change object names,
not content.

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

**Join on `id`.** `id` equals the picks' `tid`. Until the 2026-10 fix the
code columns carried CSV-parse damage, though `id` was always right:
`location_code` held floats such as `0.0` for `00` (2,753 western rows), and
in `global` `station_code` lost leading zeros (`1` for `001`) and network `NA`
was empty. Derive network, station and location from `id.split(".")` if you
need them, rather than trusting the columns of a table you downloaded before
the fix.

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
that is local. A network-year is 0.7 GB; the whole of `western` is 46.8 GB (2026-10-08) and
417 k objects, which `aws s3 sync` handles in about 90 minutes on a fast link.

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

## 7. Provenance and citation

The western catalogue is PhaseNet with the `original` weights (Zhu and Beroza,
2019, as packaged by SeisBench), P and S thresholds 0.2, components ZNE12,
SeisBench 0.12.5, weight version 2, run 2026-09-03 to 2026-09-17 on AWS Batch
Fargate Spot from image `ghcr.io/seisscoped/quakescope` at the commits pinned
in the campaign job definitions (`fleet.json`); the three eras and the
2026-09-17 repair share that configuration and are told apart by `rid`. The
re-read (`western-reread2`, from 2026-10-08) uses the same model, weight and
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
- Coverage before the re-read is quantified in section 6; the re-read is in
  progress from 2026-10-08 and adds picks to the same prefixes.
- The bucket is live: embargoed years fill in as EarthScope opens them, and
  compaction will rename objects. Record the date of any pull.
- The fill stations (2,369 station-locations) are merged into
  `stations.parquet` and picked; the date-parsing re-pick
  (`western-dates`, [30_station_dates.md](rerun_2026/30_station_dates.md)) is
  complete. Networks our EarthScope account may not read (TD, EO, LH in
  western; NV in obs) are not in the catalogue.
