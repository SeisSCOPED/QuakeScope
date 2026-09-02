# 25 — Metadata against the archive: why the hit rate is what it is

The hit rate — planned station-days that actually hold data — came out lower
than "metadata is occasionally wrong" would explain: 36.2% on SCEDC, 45.3% on
NCEDC, 32.0% on restricted EarthScope
([24_cost_model.md](24_cost_model.md)). Taking the misses apart shows they are
not one phenomenon but three, and only one of them is missing data.

**It is deterministic.** The rate is `objects that exist ÷ station-days planned`
over a fixed set of sample days. The same inputs give the same answer; the
in-container survey reproduced the standalone one exactly (434 of 1,392 on
SCEDC 2015–16), and the SCEDC calibration matched the picker on 5 of 5 shards.

---

## The three ways a planned station-day misses

One day, 2018.041, counted by cause:

| | SCEDC | NCEDC | EarthScope AK | EarthScope TA |
|---|--:|--:|--:|--:|
| **HIT** | 32.7% | 49.1% | 64.0% | 94.9% |
| **ABSENT** — station nowhere in the listing | 21.2% | 23.6% | **36.0%** | 5.1% |
| **WRONG_LOC** — station there, different location code | **35.6%** | **26.0%** | — | — |
| **WRONG_CHA** — station and location there, different band | 10.6% | 1.2% | — | — |

EarthScope has no `WRONG_*` rows because it stores one object per station-day
covering every channel, so the match is on station alone. Its misses are real
absences.

**On SCEDC, 46% of planned station-days miss on a code mismatch, not on missing
data.** That is the answer to "surely metadata is only occasionally wrong": for
the channel-per-object archives, most misses are not about whether data exists.

## Most of it is one phantom location code

`2C` is **36.1% of the SCEDC campaign — 1,484,372 planned station-days** — and
it does not exist in `scedc-pds`. Checked across three widely separated years:

```
2012.100  1,779 objects   location codes: '', 00, 10
2018.041  3,606 objects   location codes: '', 00, 10, 01, A0, B0, 30, 40
2024.200  4,374 objects   location codes: '', 10, 00, 01, 02, 30, 40, 41
'2C' present in any of them: False
```

And the `2C` entries duplicate channels the station already offers under the
blank location code:

```
CI.ADO.     HHZ,BHN,HHE,HNN,BHZ,HHN,HNZ,HNE,BHE
CI.ADO.2C                       HNN,HNZ,HNE
```

So a `.2C` row is not a second sensor. It is the same accelerometer channels
listed a second time under a location code the archive never uses.

**Consequences, in order of importance:**

1. **The SCEDC campaign is overstated by 36%.** 1.48M of its 4.1M planned
   station-days can never match an object. Removing them, the real hit rate on
   plannable SCEDC station-days is **~57%, not 36.2%**.
2. **No data is lost.** Those `HN` channels are already reachable through the
   blank-location entry, and `CHANNEL_PRIORITY` picks `HH` there in preference
   anyway. Nothing is skipped that would otherwise be picked.
3. **The waste is real but cheap** — one `s3.list`/`s3.head` per phantom
   station-day, no download and no inference.

This is why the cost model is built on seconds per **processed**
station-day-channel rather than per planned one: the planned count carries this
artefact, the processed count cannot.

## The rest is genuine metadata drift

Two smaller, real effects remain once `2C` is set aside:

- **WRONG_CHA** — the metadata lists a band the archive does not hold that day.
  `CI.AGO.` is planned for `EH` while the archive has only `HN`; `CI.AGM.` is
  planned for `HH` with only `HN` present. The station inventory describes what
  the site has been equipped with over its life, not what was recording on a
  given day.
- **ABSENT** — genuinely no data. This is the whole story on EarthScope, and it
  is where the operating windows are simply optimistic: `end_date` is frequently
  `3000.001`, an open-ended placeholder, so a station is planned to the end of
  the campaign whether or not it kept recording.

The `TA` column above is the control that makes the point: a transportable array
with well-maintained metadata hits **94.9%**. Nothing about the pipeline forces a
low hit rate — the number tracks metadata quality, network by network.

## Worth saying in the paper

Two claims are supportable and, as far as we can tell, not widely documented:

1. **A published station inventory can contain location codes that appear
   nowhere in the corresponding archive**, at a scale large enough to distort a
   campaign plan — here 36% of one network's planned work.
2. **Planned station-days are not a usable unit of work.** Availability is
   network-, era- and metadata-dependent, ranging from 32% to 95% across the
   networks measured here. Any cost or coverage estimate built on inventory
   alone, without listing the archive, will be wrong by a factor that varies
   per network.

Both are cheap to check — the listing survey behind this is
`python -m src.picker hitrate`, minutes per campaign, no picking.

## What has not been checked

- Whether `2C` is an artefact of how `stations.parquet` was built or is present
  in the upstream FDSN inventory. Worth knowing before reporting it as an
  archive-metadata discrepancy rather than a local one.
- Whether NCEDC's 26% `WRONG_LOC` has a single dominant code like SCEDC's `2C`,
  or is spread across many.
- The miss taxonomy on restricted EarthScope. It has no location dimension, so
  its misses are absences by construction, but the operating-window question
  (`end_date` 3000.001) applies there too and is untested.
