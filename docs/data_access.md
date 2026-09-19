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
| `western` | `original` | 1986 to 2026 (to September) | 417,366 | 42.0 GB | 1.47 B | **done** for the years the archives serve; 493 shards embargoed or awaiting review |
| `obs` | `obs` (PickBlue) | 1993 to 2026 | 24,896 | 3.5 GB | 131 M | done; 335 shards blocked (EarthScope 403), 2026 not yet run |
| `global` | `jma_wc` | 2010 to 2026 | 22,460 | 15.8 GB | 584 M | paused at 3.6 % |

Object counts and sizes: S3 listing through boto3 after the eras were folded
together on 2026-09-18. Pick counts: the sum of the eras' counts on the
[campaign dashboard](https://seisscoped.org/QuakeScope/campaign_dashboard.html)
of 2026-09-18 22:20 UTC (western 1,332,145,315 + western-early 64,693,941 +
western-2026 about 77.9 M; obs 103,146,265 + obs-early 27,522,521). The
dashboard now counts per catalogue from the Parquet footers, rebuilt hourly;
queue progress per era is on the same page, as are the shards that are
embargoed, blocked or awaiting review.

Each catalogue was produced by several campaigns run in eras, because a work
queue is immutable once written (`western-early` for 1986 to 2009, `western`
for 2010 to 2025, `western-2026`), all with the same weight and thresholds and
all writing into the same prefix, so a reader never has to know which era a
year came from. Pick objects are named by the shard that wrote them and the
`year=` partition runs continuously across the eras.

**Western** is the stakeholder deliverable and the one to start with. It is
24,008 station-locations inside the state polygons of Washington, Oregon,
California, Nevada, Idaho and Wyoming, read from the SCEDC, NCEDC and
EarthScope archives. Stations in Utah, Montana, Arizona, Colorado, New Mexico,
British Columbia and Baja California are not in it (see
[29_one_prefix_per_catalogue.md](rerun_2026/29_one_prefix_per_catalogue.md)
for the count); offshore stations are in `obs`.

**Not compacted.** The catalogue was written by up to 1,500 concurrent
workers, so a month partition holds hundreds of files of about 120 KB each. Reads are
correct but pay per object, which is why every rule in section 4 is about
touching fewer objects. Compaction is planned and will change object names,
not content.

## 3. Layout and schema

```
s3://quakescope-picks-2026/<campaign>/
    picks/network=<NET>/year=<YYYY>/month=<MM>/<shard_id>[-NNN].parquet
    manifests/<shard_id>.json     what each job wrote: object keys and per-station-day pick counts
    runs/<run_id>.json            model, weight, thresholds, library versions
    stations.parquet              the station table the campaign was planned from
```

`network`, `year` and `month` are Hive partition keys: they live in the path,
not in the file, and every reader below turns them into columns.

### Pick columns

| column | type | meaning |
|---|---|---|
| `tid` | string | `NET.STA.LOC`; the location code may be empty, so `CI.CLC.` is a valid id |
| `cha` | string | band + instrument code the station-day was picked on: `HH`, `EH`, `BH`, `HN`, ... One code per station-day, chosen by `constants.CHANNEL_PRIORITY` (HH > EH > SH > BH > DP > HN > CN) |
| `pha` | string | `P` or `S` |
| `start`, `peak`, `end` | timestamp, ms | the pick's probability window; **`peak` is the arrival time** |
| `conf` | float32 | peak probability, 0 to 1. Everything at or above **0.2** is stored |
| `amp` | float32 | Wood-Anderson displacement, metres, mean of the horizontal peaks, response removed. **Measured only for `conf` >= 0.5**, NaN otherwise and inside the 60 s taper at trace ends |
| `amp_vel` | float32 | raw peak amplitude, counts, max over all components, high-passed at 1 Hz, no response removed. A detection-strength proxy, not a physical unit; set on 99 % of picks |
| `rid` | string | run id, resolves to `runs/<rid>.json` |

`conf` is a detection score, not a probability that the pick is correct, and
0.2 is deliberately permissive. Most uses want more; choose the value on your
own target events rather than adopting one. Amplitude conventions (IASPEI
Wood-Anderson constants, whole-day deconvolution, taper rule):
[amplitude_conventions.md](amplitude_conventions.md).

### Station columns

`id` (= `tid`), `network_code`, `station_code`, `location_code`, `channels`
(the bands the archive lists, e.g. `HH` or `DP,EH`), `latitude`, `longitude`,
`elevation` (m), `start_date`, `end_date` (decimal `YYYY.DDD`; `3000.001` means
still operating), `state`. Western has 24,008 rows across 119 networks; the
`state` column is filled for 23,948 of them.

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
that is local. A network-year is 0.7 GB; the whole of `western` is 42 GB and
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
longitude box, or by network, take the distinct `network_code` values, and sync
`picks/network=<NET>/year=<YYYY>/` for each network and year in your window.
Then filter the rows on `tid` for the stations you kept, because a network
partition holds every station of that network, not only the ones in your box.

## 6. Coverage: was this station-day picked at all?

A station-day with no rows may mean the archive held no data, or that the
campaign did not read it. The catalogue records both cases the same way, so
**absence is not evidence of an empty archive.** Three mechanisms are
documented in
[western_pick_validation.html](https://seisscoped.org/QuakeScope/western_pick_validation.html)
section 10: a gap rule that dropped fragmented day files (fixed 2026-09-10,
after `western` had run), day listings that failed silently, and network-years
the archive does not hold. In the validation sample, 19 of 52 targeted
station-days had no picks; at least one of those had complete data in the
archive.

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
2026-09-17 repair share that configuration and are told apart by `rid`. The picks reproduce exactly
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
  `HH`; the accelerometer was not run.
- `amp` exists only above `conf` 0.5. Magnitudes from this catalogue are
  magnitudes of the confident picks.
- Coverage gaps (section 6) are not yet quantified for the whole campaign.
- The bucket is live: embargoed years fill in as EarthScope opens them, and
  compaction will rename objects. Record the date of any pull.
