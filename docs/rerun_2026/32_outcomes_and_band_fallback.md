# 32 — Every station-day gets an outcome; the band is chosen per day

2026-10-06. Follows the western audit prompted by Ian McBrearty's missing
station list (docs/rerun_2026/30, both addenda).

## What the audit found

Measured from `s3://quakescope-picks-2026` on 2026-10-06, completed shards of
`western-fill` and `western-fill2` only:

| | station-days |
|---|--:|
| planned, inside each station's window | 5,560,818 |
| with a manifest record (reached the picker) | 2,127,895 |
| no trace at all | 3,433,019 |
| sample of the untraced with data at EarthScope FDSN | 19 of 150 (13%) |

10,678 of 27,924 complete fill shards had no manifest. "Complete" meant only
that the worker had exited without raising.

Of the 19 data-bearing days, 10 were the same defect: the reader chose **one
band per station for the whole shard** from the table's `channels`, which is
the union over all epochs. A UU station listed `EH,EN,HH` read HH on every
day, and on the years before its HH upgrade found nothing, logged a line and
moved on. Two more offered no pickable band (EN only, LH only), as designed.
The remaining 7 had a matching band; the dry test in Batch is what explains
them.

## Changes (image)

- `S3DataSource.load_waveforms` chooses the band **per station-day**, trying
  the station's pickable bands in `CHANNEL_PRIORITY` order against what the
  archive holds that day. Still one band per station-location-day.
- Every planned station-day ends in exactly one outcome, recorded in the
  shard manifest (`outcomes`, `outcome_counts`):
  final `loaded`, `no_data`, `no_channel`, `empty_read`, `not_found`,
  `denied`, `done`; unread `refused`, `throttled`, `timeout`, `read_error`,
  `too_big`. Every failed read now carries its reason (`s3_helper._empty`).
- A manifest is written for every shard, including one that loaded nothing.
- A shard completes only if its outcomes cover every planned station-day
  (`worker.check_outcome_coverage`); otherwise it raises and is retried.
  Unread outcomes go to `review/` with kind `signal+unread`, so a repair
  queue can target exactly those days.

## Changes (planning, no image needed)

- Station tables may carry `epochs` (`scripts/add_station_epochs.py`, from
  FDSN channel epochs of pickable vertical channels); the planner plans each
  epoch rather than the start/end hull. Western: epochs found for 23,812 of
  26,377 station-locations, 1,837 with more than one.
- Every new queue writes `plan.json` with the station-table version it was
  planned from.
- `scripts/coverage_check.py`: table windows minus the union of every queue
  writing into the catalogue. Western against the published table: 84,119
  never planned on 25 station-locations; against the epoch table: 72,690 on
  514 (RE 48,030, WR 15,488, MX 4,808, UW 1,434, ZI 1,425).

## Not done

An FDSN fallback reader for networks EarthScope serves over FDSN but refuses
our account over S3 (TD 151,682 and EO 4,176 station-days). It would send
about 155,000 day-long dataselect requests to EarthScope; after the
2026-09-04 incident that needs their agreement first. Asking them to extend
the account to TD and EO gets the same data through the existing reader.
