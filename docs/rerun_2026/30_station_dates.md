# 30 — Station dates: a float that reads ten times small

Reported 2026-09-29 by a reader of the published catalogue: `start_date` and
`end_date` in `stations.parquet` are floats, so "2000.1, 2000.10 and 2000.100
will all be the same day". The encoding turned out not to be lossy, but the
report found a real defect one step further on, and it cost picks.

## The encoding is sound; reading it as a string is not

The writer produced `float(f"{year}.{doy:03d}")`, so the day-of-year is three
zero-padded digits and `2000.001`, `2000.010` and `2000.100` are three
different floats. Checked over all 80,479 rows of the three published tables:
every value decodes to a day-of-year in 1..366 with a maximum rounding error
of 2.2e-10, and no two distinct days share a value. Nothing was lost.

What is unsafe is the obvious way to decode it. `str()` drops a trailing zero:

```
str(2010.21) == "2010.21"     strptime("2010.21", "%Y.%j") -> day 21, not 210
str(2010.1)  == "2010.1"      -> day 1, not 100
str(1937.23) == "1937.23"     -> day 23, not 230
```

**Two of our own consumers did exactly that**:
`shard_planner._operating_windows` (the v3 planner) and
`utils.filter_station_by_start_end_date` (the 2025 path). Both wrapped it in
`except (ValueError, TypeError)`, and a misread never raises, so it was silent.

## What it cost

The misread day is always the smaller one, so a station's window was always
read *short*. On `start_date` that plans days the station was not yet
recording: harmless, a wasted listing. On `end_date` it stops the station
early, and those days were never planned, never read and never picked.

| catalogue | station-locations stopped early | station-days never planned, inside the campaign span |
|---|--:|--:|
| western (1986 to 2026) | 837 | 74,979 |
| obs (1993 to 2026) | 331 | 46,713 |
| global (2010 to 2026) | 1,975 | 191,481 |

Worst single cases are a full 324 days: `YK.YOU.01` really ran to 2007-12-26
and was planned to 2007-02-05; `YR.ED23/24/25` ran to 2017-12-16 and were
planned to 2017-02-04. By network the western loss is spread across NC
(7,962), 3J (6,633), NP (5,193), TA (4,941) and 50 others; the obs loss
concentrates in YR (13,083), 7D (8,151) and XO (6,300).

Global's 191,481 are notional: its queue is written but only 3.6% run, so the
days were never going to be picked yet. They are lost to that queue all the
same, because a queue is immutable once written, and need a fill queue of
their own whenever global resumes.

## The fix (2026-09-29)

**One decoder, `utils.station_date`.** Accepts a date, a datetime or
Timestamp, a `YYYY.DDD` string and the legacy float, and decodes the float
numerically (`round((v - year) * 1000)`), never through `str()`. Returns
`None` for anything unusable, and both planners then plan the station for the
whole campaign rather than dropping it. `parse_year_day` stays for the shard
queue's `%Y.%j` strings, which are always three digits.

**`stations.parquet` now holds Parquet dates.** `write_stations` converts
whatever it is handed to `date32` on the way out and keeps the original float
beside it as `start_yearday` / `end_yearday`, so the conversion stays
checkable. A station still operating carries **3000-01-01**, not null: null
would make `end_date >= when_i_care` false and silently drop exactly the
stations that are still recording, which is the same shape of bug as the one
being fixed.

All 16 station tables in the bucket were rewritten this way (3 catalogues, 13
queues), each verified row by row against the float it replaced; the
pre-conversion tables are in `_archive/station_tables_yearday/`.

Pinned by `tests/test_station_dates.py`, which asserts the three misread
values, the direction of the old error, and that a planner clips to the same
window whether the table holds floats, strings or dates.

## What still has to happen

- **Re-pick the 121,692 station-days** that western and obs never planned.
  `_queues/western-dates` and `_queues/obs-dates`, writing into the
  catalogues, the same pattern as `western-fill` and the resumed-shard repair.
  About $7 at western's measured rate.
- **Global** needs the same when its queue resumes.
- The deployed image predates this fix. Planning happens locally, so the
  repair queues can be written today; workers only need the fix if they
  re-plan, which they never do.
