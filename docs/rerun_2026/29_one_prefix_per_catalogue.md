# 29 — One prefix per catalogue

Plan, written 2026-09-18, not yet executed. The three western eras
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
