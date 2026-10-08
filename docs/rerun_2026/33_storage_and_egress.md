# 33 — What the catalogues cost to keep, and to give away

Measured 2026-10-07/08, read-only. Bytes: CloudWatch `AWS/S3`
`BucketSizeBytes` (StandardStorage, 2026-10-07) and boto3
`list_object_versions` per prefix (2026-10-08). Prices are AWS list prices for
us-east-2, not a bill: the CloudBank account's SCP blocks the billing APIs.

## Bytes

| prefix | current objects | current GB | noncurrent GB |
|---|--:|--:|--:|
| `western/picks` | 489,038 | 46.81 | 0.01 |
| `global/picks` | 22,460 | 15.84 | 0 |
| `obs/picks` | 14,789 | 2.57 | 1.08 |
| `western-early/`, `western-2026/`, `obs-early/` picks (era copies, see below) | 141,434 | 8.52 | 0.91 |
| all manifests, runs, station tables | about 531,000 | 2.92 | 0.24 |
| `_queues/` | 642,889 | 1.25 | 0.96 |
| `_archive/` | 122,511 | 6.74 | 0 |
| **bucket** | **1.95 M** | **84.6** | **3.2** |

CloudWatch reports 95.8 GB on 2026-10-07; the listing is lower by the
`_queues/` churn between the two readings. Every object is STANDARD. `_queues/`
also holds 3.85 M noncurrent versions and 3.8 M delete markers: almost no
bytes, but they slow every LIST of the bucket.

Picks average 110 KB per object (western: mean 96 KB, median 46 KB); 495,000
of the 667,721 pick objects are under 128 KB.

## Storage, per month

| option for the picks (73.7 GB) | $/month | why |
|---|--:|---|
| Standard ($0.023/GB) | 1.70 | |
| Standard-IA | 1.52 | 128 KB minimum billable size: 121 GB billed |
| Intelligent-Tiering, cold | 1.06 | objects under 128 KB are not tiered; $0.43 of it is the monitoring fee |
| Glacier Instant Retrieval | 0.48 | plus $0.03/GB to read, about $2.2 per full read |

The whole bucket in Standard is about $2.0 a month. Storage is not the cost to
manage; tiering saves at most $1.2 a month and only if nobody reads the picks.

## Egress, which is the real cost

The catalogues are public-read and anonymous, so the bucket owner pays for
every download: $0.09/GB to the internet after the account's first 100 GB a
month.

| download | GB | egress | GET requests |
|---|--:|--:|--:|
| `western/picks` | 46.8 | $4.2 | $0.20 |
| all picks | 73.7 | $6.6 | $0.27 |

Requester Pays cannot be combined with anonymous access: it needs signed
requests, and anonymous ones get 403. If downloads become a cost, the options
are a Requester Pays copy for heavy users or a CloudFront distribution.

## What to do

- Stay in Standard.
- Compact the western picks into files of 64 MB or more: 490,000 objects to a
  few thousand, faster anonymous downloads and DuckDB scans, and no small-object
  penalties if tiering is ever wanted. `22_parquet_compaction.md` assumes about
  30,000 objects and a 1 MB target and needs revising against these figures.
- Lifecycle rule on `_queues/`: expire delete markers and noncurrent versions
  after a few days, which clears about 7.6 M entries.
- Delete the era copies `western-early/`, `western-2026/` and `obs-early/`
  (8.5 GB of picks; their content was copied into `western/` and `obs/` on
  2026-09-18, doc 29). The deletion was due on 2026-09-25 and has not run.
  Then decide on `_archive/` (6.7 GB).
