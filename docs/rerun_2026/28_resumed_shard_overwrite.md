# 28 — A resumed shard overwrites its own first checkpoint

Found 2026-09-16 while writing the data-access tutorial: the coverage grid for
the Monte Cristo box showed rectangular gaps at the *start* of shard windows on
stations that were otherwise complete. Measured the same day on every finished
campaign, from the campaign state and the Parquet, through boto3.

## The mechanism

A worker that resumes a shard another worker checkpointed reads
`progress/<shard_id>.json`, skips the station-day-channels listed there, and
builds a fresh `ParquetWriter`. That writer's `_part_seq` starts at zero, so
its first flush into a `(network, year, month)` partition writes
`<shard_id>.parquet`, its second `<shard_id>-001.parquet`, and so on: **the
same keys the first attempt wrote**, in the same order. Each flush of the
resuming attempt replaces one file of the first attempt. What the first
attempt wrote in files the resuming attempt never reached survives; what it
wrote in the files that were replaced is gone, and it is the *earliest* work
of the shard, because the sequence starts at zero on both sides.

Two consequences, one visible and one not:

- **The manifest and the completion record describe the resuming attempt
  only.** `manifests/<shard_id>.json` lists its files and its `records`;
  `complete.picks_record` counts its station-days. A shard that ran 640
  station-days shows `picks_record: 20`. Coverage read from manifests
  therefore under-counts on every resumed shard, which is why
  `tutorials/download_the_catalogue.ipynb` reads "has picks" from the Parquet
  and uses manifests only for "read, no pick".
- **The first checkpoint interval or more is lost.** The loss per shard is
  `min(files by attempt 1, files by attempt 2)` per partition, each file being
  one checkpoint of up to 40 station-day-channels.

Worked case, `western` shard `2020129-2020149-f2c81102bb75` (32 CI stations,
2020-05-08 to 05-28): worker `ip-172-31-5-164:15` checkpointed 280
station-day-channels (doys 129 to 147) by 23:58Z on 2026-09-03 in seven files;
worker `ip-172-31-19-77:14` resumed, processed 20, and at 00:06Z wrote
`<shard_id>.parquet`, replacing the first attempt's first checkpoint. Files
`-001` to `-006` survive. CI.GRA has picks from 05-11 on; 05-08 to 05-10 are
gone. The manifest lists one file and 20 records.

## How much

Two scripts: [`scripts/resumed_shards_scan.py`](../../scripts/resumed_shards_scan.py)
(for every shard with a `progress/` object, compare the checkpoint's worker
and records with the completion record and the manifest) and
[`scripts/resumed_shards_loss.py`](../../scripts/resumed_shards_loss.py) (for
each resumed shard, read every surviving Parquet object of that shard that the
manifest does not list and check which checkpointed station-day-channels have
no picks anywhere). Both need credentials, since `progress/` and `complete/`
are not public. The per-shard results of the 2026-09-16 run are committed in
[`resumed_shards/`](resumed_shards/), one row per resumed shard.

| campaign | checkpointed shards | resumed by another worker | station-day-channels in the checkpoint but not the manifest | of those, with no picks in any surviving object | share of processed station-days |
|---|--:|--:|--:|--:|--:|
| western | 39,439 | 308 | 62,120 | **11,582** | 0.15 % of 7,730,960 |
| western-2026 | 1,706 | 20 | 5,520 | **751** | 0.18 % of 411,472 |
| obs | 2,908 | 26 | 5,120 | **776** | 0.15 % of 520,114 |
| obs-early | 1,008 | 5 | 920 | **161** | 0.10 % of 155,101 |

"No picks in any surviving object" is an upper bound on the loss, because a
station-day the first attempt read and found nothing on looks the same
(about 2 to 4 % of records carry `npks: 0`). The median loss per resumed
shard is exactly 40, one checkpoint; the maximum is 120, three. Processed
station-day counts are from [24_cost_model.md](24_cost_model.md).

The lost station-day-channels are known exactly: for each shard in the tables,
the entries of `progress.done` that are in no manifest record and in no
surviving Parquet object. They are re-pickable as a small campaign of their
own, about 13,000 station-days in total.

## Two classes of resumed shard

The scan above found resumed shards by `progress.worker != complete.worker`.
That catches only the case where the resuming attempt never checkpointed. When
it did, it **overwrote the progress object** with its own records (a third
defect: `write_progress` wrote only the current attempt's records), so the
first attempt's list is gone and the shard looks like a single attempt. Those
shards are found from the bucket instead: pick files for the shard that its
manifest does not list are the first attempt's survivors. For them the lost
set cannot be enumerated from state; the repair takes the shard's plan
(stations x days) cut at the first day the manifest records, because the
loader is day-major and a resumed attempt starts on the day the first stopped,
minus every station-day with picks anywhere or a record. That over-counts by
the planned station-days that never had data, which cost nothing to re-list.

## The fix (2026-09-16, `sb_catalog/src/parquet_writer.py`, `s3_state.py`, `worker.py`)

- `ParquetPickWriter._next_seq` starts a partition's file sequence **after the
  highest suffix already present for the job id** (one LIST with the job id as
  prefix), so a resumed attempt continues `-007` rather than rewriting
  `<job>.parquet`. A fresh job finds nothing and starts at 0 as before.
- `close()` rebuilds the earlier attempt's output from the bucket: the
  partitions follow from `prior_done`, the files are the job's minus this
  attempt's, and the per-station-day pick counts and run ids are read from
  those files. The manifest lists every file the job produced and every
  station-day it covered, with a `resumed` block saying how much came from
  before.
- `write_progress` now carries the prior `done` set forward, so the progress
  object is cumulative across attempts and a third attempt cannot redo the
  first.
- `worker._run_shard` passes `done` into the bridge and the writer.

Pinned by `tests/test_resume_no_overwrite.py`; exercised read-only against the
damaged CI shard, where the writer would continue at `-007` and reconstructs
280 prior records, 55 of them (doys 129 to 130) with no surviving picks.
**Not deployed until an image is built from a commit carrying it and the
campaign job definitions are re-registered on that image.** `western-early`
runs on the old image until then and its resumed shards keep losing their
first checkpoint.

## The repair

`scripts/repair_resumed_shards.py <campaign>` finds every resumed shard of a
finished campaign, derives the lost station-days as above, groups them by
station into runs of consecutive days and writes a queue of small shards to
`<campaign>-repair/shards.jsonl` with the parent's `stations.parquet` beside
it. `--launch N` submits N workers on the parent's job definition with
`--parquet_uri` set to the **parent** prefix, so the repaired picks and their
manifests land in `western/picks/...` and `western/manifests/...` under new
shard ids, and with `--checkpoint-every 0`, so a repair shard restarts rather
than resumes if preempted and the old image's bug cannot bite. Per-shard
findings: `resumed_shards/repair_shards_<campaign>.csv`; the station-days:
`resumed_shards/repair_station_days_<campaign>.csv`.

Two things a reader of the parent prefix should know afterwards: the repaired
picks carry run ids whose records live under `<campaign>-repair/runs/` (copy
them across once the repair completes), and a repair shard's manifest lists
only the station-days it re-picked, which is what it should.
