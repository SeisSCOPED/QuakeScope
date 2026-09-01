# 12 — Where the picks should land: DocumentDB, Parquet, or both

Written 2026-08-17, in response to the question of whether to emit Parquet on S3
instead of a provisioned DocumentDB, and whether it is cheaper to build the
database first and convert afterwards.

**Short answer.** Write Parquet directly from the jobs. Do not build DocumentDB
and convert — that pays the expensive half twice for no benefit. Keep a small
index table for the resume logic, which is the only thing the database is
genuinely needed for. The concern about dynamic Parquet is well founded but the
failure modes are avoidable by construction rather than inherent.

---

## 1. How big is this actually

Measured rather than assumed, over the launch network lists and 2010–2026:

| Campaign | Stations | Station-days |
|---|--:|--:|
| NCEDC | 1,213 | 4,551,557 |
| SCEDC | 570 | 2,467,740 |
| EarthScope offshore | 2,901 | 950,793 |
| EarthScope onshore | 42,030 | 44,127,796 |
| **Total** | **46,714** | **52,097,885** |

Station-days come from each station's actual operating window, not from
stations × span. That distinction matters enormously: the big nodal deployments
average **20 days per station**, so 42,030 onshore stations do not imply
42,030 × 5,844 station-days.

Pick rate was measured on ordinary SoCal days at production thresholds
(P 0.2 / S 0.2), deliberately away from any sequence: **778–2,648 picks per
station-day, median ~1,700**.

> **~52 million station-days × ~1,700 picks ≈ 88 billion picks.**

Treat that as an upper bound with real uncertainty. It assumes every catalogued
station-day has pickable three-component data, which it does not — many nodal
stations are single-component and will be skipped. But the order of magnitude,
**10¹⁰ picks**, is what drives the decision, and that is robust.

## 2. What a pick costs to store

Measured by encoding a realistic 800,000-row batch — one Batch job's worth — in
each format:

| Format | Bytes per pick | At 88 billion |
|---|--:|--:|
| Parquet, zstd + dictionary | **35** | **3.1 TB** |
| BSON document, the shape `picker.py` writes | 148 | 13.1 TB |
| BSON + indexes (conservative 80 B/pick) | ~228 | **20.2 TB** |

Parquet wins by about **6.5×** on the same data. Dictionary encoding is why:
`tid`, `cha`, `pha` and `rid` are enormously repetitive down a column and
compress to almost nothing, while BSON repeats every field *name* in every
document.

## 3. Why "DocumentDB first, convert later" is the expensive path

It pays for the costly half twice and discards the result:

- **Storage during the campaign.** 20 TB in DocumentDB at list price
  (~$0.10/GB-month) is **~$2,000/month**, and DocumentDB storage **does not
  shrink when you delete data** — you pay for the high-water mark until the
  cluster is dropped.
- **Instances on top.** The cluster has to stay up for the whole campaign, and
  20 TB with hundreds of concurrent writers is not a one-small-instance job.
- **The unique index is the real problem.** `picks` carries a unique index on
  `(tid, cha, pha, peak)`. Maintaining uniqueness across 88 billion rows means
  every insert does an index probe against a structure far larger than memory.
  Insert throughput degrades as the index grows, which is exactly backwards for
  a campaign that gets bigger as it runs. This, more than storage, is what makes
  the database the bottleneck at this scale.
- **Then you pay to read it all back out** to convert, and you still pay for S3.

The database earns its place when you need indexed point lookups, updates, and
transactions. A write-once pick archive that is later scanned in bulk needs none
of those.

## 4. Is dynamic Parquet writing problematic?

The instinct is right about the failure modes and wrong that they are inherent.
All three are avoidable by choosing the file boundary sensibly.

**The small-file problem — avoided by writing one file per job.** A Batch job
covers 40 stations × 20 days = 800 station-days ≈ 1.4 M picks ≈ **48 MB
compressed**, which is close to ideal Parquet sizing. The whole campaign becomes
roughly **65,000 files**, not 52 million. Writing one file per *station-day*
would produce 52 M objects of 60 KB and would indeed be a disaster — but nothing
requires that.

**Concurrency — Parquet is strictly easier than a database here.** Files are
immutable and each job writes its own key, so there is no contention, no
connection-pool ceiling, no write conflict, and no lock. Hundreds of concurrent
Fargate tasks writing separate S3 objects is the case object storage is built
for.

**Retries — deterministic keys make them idempotent for free.** Name the object
from the job's own scope, and a retried job overwrites its own file byte for
byte. That is cleaner than `insert_many_ignore_duplicates`, which exists
precisely because the database cannot express this.

**The one real piece of work is the resume logic.** `picks_record` answers "has
this station-day already been processed?", and the pipeline relies on it so a
re-submission only does what is missing. S3 cannot answer that cheaply if the
file boundary is the job rather than the station-day.

Keep a small index for exactly this and nothing else: 52 M rows of
`(tid, cha, yr, doy, npks)` at roughly 50 bytes is **~2.6 GB** — three orders of
magnitude smaller than the picks. DynamoDB is a natural fit and costs cents at
that size; a small DocumentDB would also work. The point is that the index is
tiny and the picks are huge, and only the index needs a database.

## 5. Zarr is the wrong tool for picks

Zarr stores regular N-dimensional arrays, chunked. Picks are a **table** — rows
with heterogeneous typed columns and no natural grid — which is exactly
Parquet's shape. Forcing picks into Zarr means either ragged arrays or padding,
and loses the columnar compression that produces the 35 bytes/pick above.

Zarr *is* the right choice if you later want to keep the model's **continuous
probability curves** — those are genuinely array-shaped, one regular time series
per station-channel, and would chunk and compress well. That is a different and
much larger product: at 100 Hz, three classes, float16, one station-day is
~52 MB before compression, so 52 M station-days is petabyte scale. Worth
scoping separately, and probably worth keeping only for selected windows.

## 6. Recommended layout

```
s3://<bucket>/picks/network=<NET>/year=<YYYY>/month=<MM>/<jobid>.parquet
```

Hive-style partitioning so DuckDB, Athena, Spark and pandas all prune without
being told how. Partition on network/year/month rather than day: day-level
partitioning at this span produces millions of partitions, which becomes its own
listing and metadata problem. A job spanning a month boundary simply writes two
files.

**Querying.** DuckDB reads Hive-partitioned Parquet on S3 directly and will
handle most analysis on a laptop without any service running. Athena is there
for anything larger, charged per TB scanned — and with partition pruning plus
column projection, a typical query scans a small fraction of 3 TB rather than
all of it.

**Cost, at list prices worth re-checking before committing:** 3.1 TB of S3
Standard is roughly **$70/month**, against roughly **$2,000/month** of
DocumentDB storage alone for the same picks, before instances or I/O. Moving to
S3 Infrequent Access or Glacier Instant Retrieval for older years cuts the S3
figure further, and there is nothing to keep running between campaigns.

## 7. What would need building

The pipeline already batches per job, which is the hard part. What changes:

1. **A Parquet writer** alongside the Mongo writer in
   `picker.py::_write_single_picklist_to_db` — accumulate a job's picks and
   write one file at the end rather than streaming inserts. Amplitudes and
   classifications ride along as columns.
2. **`picks_record` moves to the small index**, keeping the existing resume
   semantics unchanged.
3. **The association step reads Parquet** instead of querying Mongo.
   `SeisBenchDatabase.get_picks` becomes a partition scan, which for PyOcto's
   access pattern — all picks in a region and time window — is a better fit than
   a Mongo query anyway.
4. **Keep both paths for one campaign.** Run SCEDC, the smallest and
   best-tested, writing to both, and compare counts. That is a day of work and
   it converts this analysis into a verified claim.

## 8. Caveats

- Prices are list prices from memory and should be checked against the current
  AWS calculator for `us-east-2` before anyone commits a budget. The **ratios**
  are what this argument rests on, and those are driven by the measured 6.5×
  size difference and by DocumentDB requiring always-on instances where S3 does
  not.
- The 88-billion-pick figure is an upper bound. If the real figure is 5× lower,
  every conclusion here holds and the absolute numbers shrink together.
- Nothing above changes the picking itself. This is purely where the output
  lands.
