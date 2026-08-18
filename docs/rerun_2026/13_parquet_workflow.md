# 13 — Running the campaigns with Parquet output

How to run the five launch campaigns writing Parquet to S3 instead of bulk
picks into DocumentDB. The reasoning and the measurements behind the change are
in [12_output_storage.md](12_output_storage.md); this page is the operational
half.

---

## What changed

| | Before | Now |
|---|---|---|
| Picks | `picks` collection | Parquet on S3 |
| Classifications | `classifies` collection | Parquet on S3 (when the classifier runs) |
| `picks_record` | database | **unchanged — still the database** |
| Station metadata | database | **unchanged — still the database** |
| `sb_runs` provenance | database | **unchanged — still the database** |

The database does not go away. It keeps the three things that are small and
want point lookups; only the bulk output moves. `picks_record` is what makes a
re-submission process only what is missing, and at roughly 2.6 GB against 3 TB
of picks it is not what was costing anything.

New pieces:

- [`sb_catalog/src/parquet_writer.py`](../../sb_catalog/src/parquet_writer.py) —
  `ParquetPickWriter`, which buffers a job's output and writes it partitioned on
  close.
- `--parquet_uri` on `src.picker`. Unset or empty means the old behaviour, so
  nothing changes for anyone who does not pass it.
- `parquet_uri` is a Batch job parameter, threaded through `submit_helper`.

**The job definition no longer hardcodes `--classifier`.** It previously always
passed the flag, which would have contradicted the decision to defer the
classifier. Re-register the job definition to pick this up.

## Layout

```
s3://<bucket>/<campaign>/picks/network=CI/year=2019/month=07/<jobid>.parquet
s3://<bucket>/<campaign>/classifies/network=CI/year=2019/month=07/<jobid>.parquet
s3://<bucket>/<campaign>/manifests/<jobid>.json
```

One file per job per `(network, year, month)`. A job covering 40 stations ×
20 days lands near 50 MB, which is close to ideal for Parquet, and the campaign
produces tens of thousands of objects rather than tens of millions. A job that
crosses a month boundary or spans two networks simply writes two files.

The manifest records what a job claimed — every station-day-channel it touched
and how many picks each produced — which is what makes coverage auditable after
the fact without scanning the picks.

Hive-style partition names mean DuckDB, Athena, Spark and pandas all prune
without being told the layout.

## Submitting the five campaigns

Give each campaign its own prefix so they can be read, replaced, or deleted
independently.

```bash
BUCKET=s3://quakescope-picks-2026

# 1-3: onshore, jma_wc
for CAMPAIGN in ncedc scedc earthscope_onshore; do
  NETS=$(grep -v '^#' sb_catalog/configs/networks/${CAMPAIGN}.txt | paste -sd, -)
  python -m src.submit_helper pick \
      --start 2010.001 --end 2026.001 \
      --network "$NETS" \
      --database quakescope_2026 \
      --parquet_uri "$BUCKET/${CAMPAIGN}" \
      --model PhaseNet --weight jma_wc \
      --region us-east-2
done

# 4: offshore, the OBS weights
NETS=$(grep -v '^#' sb_catalog/configs/networks/earthscope_offshore.txt | paste -sd, -)
python -m src.submit_helper pick \
    --start 2010.001 --end 2026.001 \
    --network "$NETS" \
    --database quakescope_2026 \
    --parquet_uri "$BUCKET/earthscope_offshore" \
    --model PhaseNet --weight obs \
    --region us-east-2

# 5: western states, original, isolated database and prefix
python -m src.submit_helper pick \
    --start 2010.001 --end 2026.001 \
    --extent 31.5,49.2,-125.0,-104.0 \
    --database western2026 \
    --parquet_uri "$BUCKET/western_states" \
    --model PhaseNet --weight original \
    --region us-east-2
```

The `--database` argument still matters: it is where station metadata is read
from and where `picks_record` and `sb_runs` are written. Campaigns 1–4 share
`quakescope_2026` as planned; campaign 5 stays isolated in `western2026`.

## Permissions

The Batch **job role** needs write access to the output prefix. The execution
role does not — it only pulls the image. Minimum policy on the job role:

```json
{
  "Effect": "Allow",
  "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
  "Resource": "arn:aws:s3:::quakescope-picks-2026/*"
}
```

`s3:GetObject` and `s3:ListBucket` are only needed if something later reads the
output from inside the VPC.

## Reading the output

DuckDB handles this on a laptop with no service running:

```sql
INSTALL httpfs; LOAD httpfs;

-- one network-month, the cheapest useful query
SELECT tid, pha, peak, conf
FROM read_parquet('s3://quakescope-picks-2026/scedc/picks/network=CI/year=2019/month=07/*.parquet')
WHERE conf > 0.5
ORDER BY peak
LIMIT 20;

-- pick counts per station across a campaign, pruning on the partitions
SELECT network, year, tid, count(*) AS n
FROM read_parquet('s3://quakescope-picks-2026/scedc/picks/**/*.parquet',
                  hive_partitioning = true)
WHERE year BETWEEN 2019 AND 2020
GROUP BY 1, 2, 3
ORDER BY n DESC;
```

Athena works on the same files if a query outgrows one machine — point a table
at the prefix with `PARTITIONED BY (network string, year int, month int)` and
run `MSCK REPAIR TABLE`. It is charged per TB scanned, so partition pruning and
column projection are what keep it cheap.

For the association step, read the partitions covering the region and window
and hand PyOcto a DataFrame. A partition scan suits its access pattern — all
picks in a region and time window — better than the equivalent database query.

## Verifying a campaign

Manifests are the cheap check, because they are small and complete:

```python
import json, s3fs
fs = s3fs.S3FileSystem()
total = 0
for key in fs.glob("quakescope-picks-2026/scedc/manifests/*.json"):
    with fs.open(key) as fh:
        m = json.load(fh)
    total += m["n_picks"]
print(f"{total:,} picks across {len(fs.glob('quakescope-picks-2026/scedc/manifests/*.json'))} jobs")
```

Then cross-check a sample against `picks_record` in the database, which is
written by the same code path and should agree station-day for station-day.
Disagreement means a job wrote its records but died before its Parquet flush —
re-submit it, and the deterministic key means the retry overwrites cleanly
rather than duplicating.

## Notes and limits

- **Buffering is per job.** A job holds its picks until it finishes, on the
  order of a million rows and tens of megabytes. `flush_threshold` in the
  writer caps that: past four million rows in one partition it writes and
  clears, so an unusually productive job emits a few files instead of one.
- **A killed job writes nothing.** Fargate Spot reclamation loses the buffer,
  and Batch retries the whole job. That is the intended behaviour — a whole-job
  retry is idempotent, while a partially written job would not be.
- **Parquet is immutable.** There is no update path, which is fine for a
  write-once archive but means corrections are made by rewriting a partition,
  not by editing rows.
- **The old path still works.** Omit `--parquet_uri` and everything behaves as
  before, which is what makes it safe to switch one campaign at a time.
