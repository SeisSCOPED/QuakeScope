# 34 — Campaign playbook: what the 2026 run taught us, and how to run the next one

Written 2026-10-10, after the western and obs catalogues were closed. For
whoever runs the next QuakeScope campaign. Every number below was read from
`s3://quakescope-picks-2026`, AWS Batch or CloudWatch on the date given; the
documents cited hold the method.

**Terms.** A *station-day* is one station-location (`NET.STA.LOC`) on one day.
A *shard* is one unit of queued work, a set of stations over a run of days. A
*queue* is the immutable list of shards for one campaign
(`_queues/<name>/shards.jsonl`). A *manifest* is the JSON a shard writes
(`<catalogue>/manifests/<shard>.json`) listing what it read. An *outcome* is the
per-station-day status a manifest carries since image d0ccf9b. A *refill* is a
corrective queue that re-plans station-days an earlier queue missed.

## 1. What happened

The catalogues closed with **17,415,098 station-days analyzed** (western
16,965,557, obs 449,541; every manifest scanned 2026-10-09) and an estimated
**0.7 PB** of continuous data read (39 to 44 MB per station-day-channel,
measured 2026-09-01, OPTIMISE.md section 2; volume itself not measured). To get
there took the original queues and **twelve corrective queues**:

| queue | written | station-days | why it existed |
|---|---|--:|---|
| western-repair, western-2026-repair, obs-repair, obs-early-repair | 09-17 | 344,367 | a resumed shard overwrote its first attempt's files (doc 28) |
| western-fill | 09-21 | 5,830,943 | the station table covered six states, the stakeholder list more (doc 29 notes) |
| western-fill2 | 09-29 | 104,214 | western-fill's table had one row per FDSN epoch; 662 shards crashed on repeated ids |
| western-dates, obs-dates | 09-29 | 115,327 | station end dates misread through `str(float)` (doc 30) |
| western-dates2 | 10-06 | 25,919 | the date repair was planned from a table that did not yet hold the fill stations (doc 30 addendum) |
| obs-fill, obs-el, obs-2026 | 10-03 | 447,572 | obs was selected by reused network codes; 1,996 land stations removed (doc 31) |
| western-reread2, obs-reread | 10-06 | 15,883,014 | the reader skipped every day the station's one chosen band was absent (doc 32) |
| western-toobig, obs-toobig | 10-09 | 143,231 | day objects over the reader's 200 MB limit |

The re-read was the largest correction by far and the most avoidable: it was
almost as large as the original `western` queue (33.8 M planned station-days)
and added 2.26 M station-days with picks. The corrective runs of 10-06 to 10-09
cost **33,240 vCPU-hours, about $708** at the billed $0.0213/vCPU-h
(western-reread2 $650, western-toobig $39, obs-toobig $11; Batch attempt
durations, 2026-10-10).

## 2. Why: the defects, and the guard each one now has

| defect | cost | how it was found | fix and guard |
|---|---|---|---|
| Band chosen once per station from the union of all epochs' channels; a day without that band was not read | ~2.3 M western station-days with data never read | a collaborator's missing-station list, then an FDSN sample of unrecorded days | band chosen per day (d0ccf9b); outcome per station-day |
| libmseed **raises** when a `sourcename` selector matches no record; a catch-all turned it into an empty stream | same class, every EarthScope network | the first Batch dry test on the outcome-recording image | `_read_first_matching`: all bands against one download (d0ccf9b) |
| "Complete" meant "the worker exited", not "every planned day has a fact"; zero-pick shards wrote no manifest | made the two defects above invisible for a month | 10,678 of 27,924 complete fill shards had no manifest | `check_outcome_coverage` refuses to complete a shard with a planned day unaccounted for |
| Station end dates decoded through `str(float)` (`2010.21` read as day 21) | 121,692 + 25,919 station-days | a reader's question about float dates | real dates in the table; `utils.station_date` |
| Station codes parsed as numbers (`00` -> `0.0`, `001` -> `1`, `NA` -> NaN) | none for picking (keys on `id`); misleading for every reader | the same missing-station list | `normalize_station_codes`; a static test blocks unnormalized writers; preflight warns |
| Planned the hull of a station's epochs, not the epochs | 84,119 never-existing days planned or flagged | coverage check against FDSN epochs | `epochs` column, planned per epoch |
| A repair queue planned from an old table | 25,919 station-days | coverage check | `plan.json` records the table version; `coverage_check.py` |
| A resumed shard overwrote its own first files | 1.8% of western | coverage grid in a tutorial | sequence continues across attempts (doc 28) |
| Shards of up to 800 stations | caught before launch | reading the queue | planner caps 40 stations and 800 station-days, tested |
| Day objects over 200 MB skipped | 143,231 station-days | outcome `too_big` | `--limit-mb`, opt-in per campaign, plus a high-memory job definition |

The common pattern: **every expensive defect was a silent skip, recorded as
done.** None raised. Each was found weeks later by someone outside the run.
The fixes that matter most are the ones that make a skip impossible to record
as success: an outcome for every planned station-day, and a shard that refuses
to complete without them.

## 3. High-volume channels

What decides whether a station-day can be read is the size of the archive's day
object, not the sampling rate of the band we pick. EarthScope stores one object
per station-day holding every channel and location; dense instrumented sites
make it large.

- **Size distribution** of the objects the 200 MB limit skipped (CloudWatch,
  147,518 skip events, 2026-10-08/09): median 315 MB, 90th percentile 917 MB,
  99th 2.35 GB, maximum 15.9 GB (SF.MH020, 2008).
- **Examples**: UW.SLA (four 3-component accelerometers plus at least seven
  pressure and strain channels, all at 200 sps; about 5,100 days per location
  over the limit); PB.B093 (EH, HH, HN plus about 20 strainmeter and
  state-of-health channels).
- **Memory follows the decoded band, not the object.** The object is
  downloaded once as raw bytes (at most the limit); the libmseed selector
  decodes only the band picked. One-worker test at `--limit-mb 4096`,
  `--procs 2`, 32 GB task: peak 4.21 GB per process, back under 0.5 GB between
  shards.
- **Except** where a day file is packed with many small or overlapping records:
  SF.PH001-006 (500 sps borehole strings), X7.LB01, Z5 and YI ocean-bottom
  arrays ran out of memory at 32 GB (24 kills, exit 137). At 120 GB and one
  process most completed; 11 shards (328 station-days, SF.PH004-006 2006-07,
  X7.LB01 2008) crash the decoder (exit 139) and are recorded as unreadable.
- **What to do**: run the main campaign at the 200 MB default; send every
  `too_big` outcome to a second queue at `limit_mb` 4096 and 2 processes; send
  what dies there to a third at 120 GB and 1 process; record what still fails.
  Do not raise the limit for the whole campaign: each location code re-downloads
  its station's whole object, and at fleet scale that is several GB/s of reads
  from EarthScope.

## 4. Credentials and access

- **One EarthScope refresh token**, in Secrets Manager
  (`quakescope/earthscope-refresh-token`), injected into the job; never used
  from a laptop, because a local exchange can invalidate the copy every job
  depends on.
- **Scope every credential exchange to a network and, for temporary networks,
  a year.** An unscoped credential lists but cannot read. On 2026-09-04 a client
  that retried refused exchanges sent 351,735 rejected token requests in four
  hours and EarthScope reported it as a denial of service (incident report on
  the project site). The Fleet workflow now carries a rate breaker (500 requests
  a minute) and refuses quarantined images.
- **404 and 403 mean different things.** 404: the archive has no such
  network-year (`not_found`, final). 403: our account may not read it
  (`denied`, final until access changes). Neither is retried.
- **Run the access survey before launching** (the Fleet workflow does it on a
  campaign's first dispatch): it checks every planned network-year once and
  blocks the shards that cannot run. Western-reread2: 137 of 1,364
  network-years missing, 931,317 station-days blocked at no cost.
- **Networks refused to our account**: TD, EO, LH (western), NV (obs). EO and TD
  answer anonymous FDSN, so an FDSN reader could pick them, at a request volume
  that needs EarthScope's agreement first.
- **Automation boundaries.** The assistant that ran this could not change IAM,
  rewrite public objects, or dispatch the fleet; those stayed with the PI.
  Phone-app dispatches of the Fleet workflow did not always reach GitHub:
  confirm a run titled "<campaign> -> N workers" appears in Actions.

## 5. How to run the next campaign without refilling it

In order. `scripts/campaign_workflow.py` encodes steps 1 to 5 and 8 to 9 as
gates that stop the run when they fail.

1. **Build the station table from FDSN epochs**, codes as text, dates as dates,
   one row per station-location with an `epochs` column
   (`scripts/add_station_epochs.py`). Check every code column equals `id`.
2. **Plan per epoch**, 40 stations and 800 station-days per shard at most, never
   across networks; write `plan.json` with the table version.
3. **Coverage check before launch**: table epochs minus the union of every queue
   writing into the catalogue must be empty (`scripts/coverage_check.py`).
4. **Image gate**: the job definition's image must contain the outcome
   recording, the per-day band and the selector fix (descendant of d0ccf9b), and
   the completion rule.
5. **Yield sample**: about 3,000 station-days stratified by network, run in
   Batch, before any fleet. It sets the expected loaded fraction per network and
   finds networks with no data at all (NP: 0 of 1,408) before they cost a fleet.
6. **Access survey, then launch** through the Fleet workflow. Scale by
   EarthScope's request rate, not by our quota.
7. **Watch outcomes, not shard counts.** A loaded fraction far below the sample
   is either ordering (empty early years run first; NC 1985-94 was 0.0%) or a
   defect; check by network and year before scaling.
8. **Close-out gate**: rebuild the availability table; draw a random sample of
   station-days with no `loaded` outcome and ask FDSN dataselect; the catalogue
   closes when under 2% of them hold data we could read.
9. **Repairs from outcomes only**: `too_big` -> high-limit queue; `timeout`,
   `throttled`, `read_error` -> same image, smaller shards. Never re-plan from
   the station table a second time.
10. **Publish** the picks, manifests, runs, station table and availability
    table; extend the bucket policy for any new prefix; date every count.

## 6. What the readers asked for, and now have

- **Station-day availability**: `<catalogue>/availability/` (doc
  data_access.md), one row per station-day with status, band and pick count.
  Western 54.4 M rows (104 MB), obs 0.77 M; public. Rebuilt after repairs.
- **Station codes that join**: join picks to stations on `id` = `tid`.
- **Exact counts with dates**, from Parquet footers, never from the dashboard
  cache (which ran 28% low on 2026-10-02).

## 7. Not done

- 328 western station-days that crash the miniSEED decoder; 678 with objects
  over 4 GB.
- 673,676 western station-days in scattered short gaps, held out because as
  shards they would cost one FDSN inventory request per few days; needs shards
  that carry per-station day lists.
- TD, EO, LH, NV: access, or an agreed FDSN path.
- Bands present in an object but absent from the station table's `channels`
  are not tried (about 3 in 27 `empty_read` outcomes in a sample).
- The 2025 DocumentDB catalogue is being exported to `quakescope-2025/`
  (4,376,745,115 picks); the database retires 2026-10-12. Its availability
  table can say only "read" or "unknown": 2025 recorded no outcomes.
