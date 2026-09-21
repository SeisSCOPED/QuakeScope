# 29 — One prefix per catalogue

Plan written 2026-09-18 and **executed the same day**; the outcome is at the
end. The three western eras
(`western-early` 1986 to 2009, `western` 2010 to 2025, `western-2026`) and the
two OBS eras are one catalogue each split across prefixes only because a
campaign's work queue is immutable once written. Readers should find every
western pick under `western/` and every OBS pick under `obs/`, and the same
for `global/` when its 2026 era runs.

## What is there now (S3 listing through boto3, 2026-09-18)

| prefix | picks objects | GB | manifests | runs | queue state (complete / shards) | in flight |
|---|--:|--:|--:|--:|--:|--:|
| `western-early` | 123,175 | 5.80 | 32,211 | 36,085 | 84,839 / 85,278 (439 blocked, 86 review) | 0 |
| `western` | 281,456 | 34.20 | 49,180 | 152,780 | 72,464 / 72,505 | 0 |
| `western-2026` | 12,735 | 1.99 | 2,177 | 2,912 | 3,224 / 3,237 | 0 |
| `obs-early` | 5,524 | 0.73 | 1,663 | 1,693 | 3,631 / 3,631 | 0 |
| `obs` | 19,372 | 2.77 | 4,219 | 5,559 | 6,231 / 6,566 (335 blocked) | 0 |
| `obs-2026` | 0 | 0 | 0 | 0 | 0 / 65 | 0 |
| `global` | 22,460 | 15.84 | 4,068 | 10,038 | 7,344 / 202,468 | 0 |
| `global-2026` | 0 | 0 | 0 | 0 | 0 / 7,795 | 0 |

Manifests are fewer than completions because a shard that loads no data never
creates a writer and writes no manifest. The four `*-repair` prefixes hold
queue state only; their picks, manifests and run records are already in the
parents. Thirty other top-level prefixes are tests, dry runs, profiles and two
abandoned attempts (`western-a`, 93,512 objects, 2.96 GB, last written
2026-09-03; `scedc`, 0.44 GB), all visible to anonymous listing.

## Target layout

```
s3://quakescope-picks-2026/
    western/     picks/ network=.../year=1986..2026/...   manifests/   runs/   stations.parquet
    obs/         same, 1993..2026
    global/      same, 2010..2026 as it fills
    _queues/     western-early/ western/ western-2026/ obs-early/ obs/ obs-2026/ global/ global-2026/ *-repair/
                 (shards.jsonl, claims/, complete/, progress/, blocked/, review/, access.json: not public)
    _archive/    western-a/ scedc/ earthscope/ ncedc/ western-b/ incident/ dry runs, tests, profiles
```

Three names a reader has to know, and everything else prefixed with an
underscore. The mechanism already exists: a worker takes its queue from
`--campaign` and its output root from `--parquet_uri`, which is how the
repairs wrote into the parents. Nothing about the layout inside a prefix
changes, so every tutorial, the dashboard and the compaction code keep working
on `<catalogue>/picks/`.

## Why a copy, not a rename

S3 has no rename. Server-side `CopyObject` inside a region moves nothing over
the network and costs $0.005 per thousand requests; deletes are free.
Partitions cannot collide: the eras hold disjoint years, shard ids embed the
date range, run ids are UUIDs. The one thing that has to be rewritten is the
`files[].path` field inside each moved manifest, which names the old prefix
(the `complete/` records carry no paths).

| move | objects | requests | wall clock at 64 threads |
|---|--:|--:|--:|
| `western-early` + `western-2026` output into `western/` | 209,295 | copies + 34,388 manifest GET/PUT | ~60 min |
| `obs-early` output into `obs/` | 8,880 | | ~3 min |
| queue state of ten campaigns into `_queues/` | ~430,000 small objects | copies | ~90 min |
| clutter into `_archive/` | ~97,000 | copies | ~30 min |
| **total** | **~745,000** | **about $4** | one afternoon, unattended |

Verification is cheaper than the move: every copied object's ETag equals its
source's (the writer uploads in one part, so the ETag is the MD5), and for
picks the row counts per partition are summed from the footers before and
after.

## Order, with a gate at each step

1. **Freeze.** Set every fleet target to 0 (they are 0 or idle already: no job
   is running and no claim is open). This is what makes the move race-free.
2. **Code first, on `main`, so the next top-up cannot write to an old root.**
   - `fleet.json`: two optional fields per campaign, `queue` (default
     `s3://bucket/<name>`, will become `s3://bucket/_queues/<name>`) and
     `parquet_uri` (the catalogue prefix). `fleet.yml` and
     `scripts/spot_governor.py` pass them as `--campaign` and `--parquet_uri`;
     the worker already accepts both.
   - `scripts/campaign_dashboard.py`: pick counts, map and time series per
     *catalogue* (`parquet_uri`), queue progress per campaign. Today it reads
     both from `<name>/`.
   - `docs/data_access.md`, both data-access tutorials, the runbook and the
     email draft: one prefix, one year range; the download notebook's `ERAS`
     table goes away.
   - `scripts/unify_catalogue.py`, four subcommands that each stop on the first
     discrepancy: `copy`, `verify`, `move-queues`, `delete`.
3. **Copy the era output into the catalogue prefixes** (`copy`): picks and
   runs by `CopyObject`; manifests by GET, path rewrite, PUT. Originals stay.
4. **Verify** (`verify`): object counts and ETags source against destination;
   row totals per partition from the footers; every `files[].path` in a moved
   manifest answers a HEAD. A count that differs stops the plan here, with
   the originals untouched.
5. **Move the queue state** (`move-queues`) for all ten campaigns and the four
   repair queues into `_queues/`, point `fleet.json` at them, run one fleet
   top-up at target 0 to prove the workflow resolves the new roots.
6. **Archive the clutter** into `_archive/<name>/` with a one-line
   `README.json` each (what it was, when, which document cites it). `_iotest2`
   and `_proctest` are cited by OPTIMISE.md for measurements already recorded
   there; the objects are not needed to keep the record.
7. **Wait seven days, then delete the originals** (`delete`), after running
   `verify` a second time. Versioning is off, so this is the only step that
   cannot be undone; the seven days of duplicated storage cost about $0.20.
8. **Unfreeze**: `western-early` back to its target, writing into `western/`.
   Its remaining shards are embargoed or need review, so this changes nothing
   until EarthScope opens the years.
9. **Bucket policy**: public `GetObject` named on `western/*`, `obs/*` and
   `global/*` (picks, manifests, runs, stations.parquet) instead of `*/picks/*`,
   so `_archive` and `_queues` are not readable anonymously. `ListBucket` stays
   public; it reveals prefix names only.
10. Re-execute the two data-access notebooks and the dashboard, republish.

## What changes for a reader

Before: `western`, `western-early` and `western-2026` had to be read as one
dataset, and the tutorial carried a table of which years live where. After:
`western/picks/network=UW/year=1998/month=03/` exists, `year` runs 1986 to
2026, and the campaign table in `docs/data_access.md` has three rows instead
of eight. Run ids still resolve under `<catalogue>/runs/`. A repair shard's
manifest, an era's manifest and an original shard's manifest sit side by side
under `<catalogue>/manifests/` and are told apart by their shard ids.

## What does not change

The shard queues stay immutable and per era; a future `western-2027` is a new
queue under `_queues/` writing into `western/`. The Parquet layout, schema and
file naming are untouched, so compaction, when it is verified, runs on the
catalogue prefix as it would have on any campaign. The resume logic lists
`<job_id>*.parquet` in the output root, which is the catalogue prefix from
step 2 on; a shard first written under an era root and resumed after the move
is covered by the copied files being in place before any worker restarts.

## Risks

- **Deleting before verifying.** Mitigated by the seven-day gap and a second
  verification; nothing is deleted by the same command that copies.
- **A worker writing to an old root during the move.** Mitigated by the freeze
  and by merging the code before copying.
- **A manifest path missed by the rewrite.** The verify step HEADs every path
  in every moved manifest.
- **The dashboard and `campaign_status.py` counting the wrong prefix.** Both
  read `<name>/picks/` today; the dashboard change is in step 2, and
  `campaign_status.py` takes the prefix as an argument.
- **Anonymous readers mid-move.** They see the catalogue grow; nothing they
  can already read disappears until step 7.

## Not in this plan

Compaction (still unverified code, [OPTIMISE.md](OPTIMISE.md) item 5) is a
separate change and should run after this one, on the unified prefixes.
Renaming the catalogues themselves (`western`, `obs`, `global`) is not
proposed; they are in every document and in the email.

## Outcome (2026-09-18 to 19)

Steps 1 to 6 and 9 done; step 7 (delete the era output under the old roots)
is due on **2026-09-25** with `scripts/unify_catalogue.py delete --era <era>
--into <catalogue> --yes` for `western-early`, `western-2026` and `obs-early`.
Logs of every step: [`unify/`](unify/).

| step | result |
|---|---|
| copy + verify | `obs-early` into `obs`: 5,524 picks, 1,693 runs, 1,663 manifests. `western-2026` into `western`: 12,735 / 2,912 / 2,177. `western-early` into `western`: 123,175 / 36,085 / 32,211. Every object has a twin with the same ETag and size; 200 sampled Parquet pairs per era hold the same rows; every rewritten manifest path answers a HEAD |
| move-queues | twelve queues, 444,000 objects, under `_queues/`; sources deleted after verification |
| archive | 26 prefixes under `_archive/`, each with a `README.json`; sources deleted after verification |
| bucket policy | public `GetObject` named on `western/`, `obs/`, `global/` (picks, manifests, runs, stations.parquet); `ListBucket` unchanged. `_queues/*` and `_archive/*` answer 403 anonymously |
| tutorials | both data-access notebooks re-executed against `western/`; the region example now also finds the repair shards' picks |

Two things the verification did not catch and the script now handles:

- **`CopyObject` of a multipart-uploaded object gets a plain MD5 ETag**, so
  every `stations.parquet` (uploaded by s3fs in parts) and one gzip log
  showed as a mismatch. Compared by MD5 of the bodies, all identical.
- **`move-queues` deleted `stations.parquet` from the catalogue-named
  campaigns** (`western`, `obs`, `global`) after copying it under `_queues/`,
  so for about ten minutes `western/stations.parquet` answered 404. Restored
  from the queue copies (MD5 identical); the script now keeps the catalogue's
  table.

Top-level listing after the move: `_archive/ _queues/ global/ obs/ western/`
plus `obs-early/ western-2026/ western-early/` holding only the era output
until 2026-09-25.

## The first new queue on the layout: `western-fill` (2026-09-21)

The before/after station plots Marine sent on 2026-09-18 showed the western
catalogue missing the interior of the stakeholder list. Of the 20,571
stations in `WestCoast_stations.txt`, 2,732 are not in the western table:
885 offshore (the `obs` catalogue's), 1,847 on land in Utah (622), Montana
(248), British Columbia (245), Alberta (203), Arizona (201), Colorado (100),
Baja California and Sonora (95), New Mexico (25), plus 107 temporary-network
stations inside the six polygons added to the repo table after the queue was
written. The specification that started the campaign (archive/09) had named
Utah and New Mexico; the launch used Wyoming instead. Decided 2026-09-21:
pick the whole list.

`scripts/plan_western_fill.py` fetched channel-level metadata from EarthScope
for the 1,847 land stations (162, nearly all `2K`, unknown to FDSN), which is
2,402 station-locations, and planned them 1986.001 to 2026.251 with the
production planner: 28,218 shards, 5,830,943 station-days, of which 1.26 M
are `NP` triggered data. The queue lives under `_queues/western-fill/` and
writes into `western/`; `fleet.json` names both. At western's measured
$0.0000556 per planned station-day the fill is about $325.
