# 22 — Automatic Parquet compaction

After a campaign reaches 80% completion, background compaction automatically consolidates the thousands of small ~50 KB Parquet files into ~1 MB files, making downstream analysis faster without slowing live picking.

## Why compaction is needed

The 1,500-worker fleet produces:
- 256,040 shards (western campaign)
- 1,456 partitions (network × year × month)
- Tiny files: 49 KB median, 30k objects total

This layout is optimal for **writes** (1,500 concurrent writers, no bottleneck) but expensive for **reads** (30k footers to open, slow partition scans).

## When it runs

Two complementary strategies:

### Strategy 1: Continuous monitoring (recommended for long campaigns)

A background job polls campaign progress hourly and auto-starts compaction at 80%:

```bash
python scripts/monitor_and_compact.py \\
    --campaign s3://quakescope-picks-2026/scedc \\
    --poll-interval 3600
```

Run this as a standalone Batch job or container (different queue, small instance):

```
Job name: quakescope_monitor_scedc
Container: ghcr.io/seisscoped/quakescope
Command: python scripts/monitor_and_compact.py --campaign s3://quakescope-picks-2026/scedc
vCPU: 1 (read-only polling)
Memory: 512 MB
```

### Strategy 2: Manual trigger (for faster feedback)

When you see 80% done in the dashboard:

```bash
pixi run -e cloud python -m sb_catalog.src.parquet_compact \\
    --campaign s3://quakescope-picks-2026/scedc \\
    --dryrun                    # preview first
```

Then without `--dryrun`:

```bash
pixi run -e cloud python -m sb_catalog.src.parquet_compact \\
    --campaign s3://quakescope-picks-2026/scedc
```

## Safety

Compaction is safe to run **in parallel with picking**:

- Picks are immutable once written; existing files are never modified
- New picks keep landing in the unconsolidated partition
- Compaction reads old files → writes new consolidated ones → deletes originals
- Retried picking jobs overwrite themselves byte-for-byte (idempotent)
- Concurrent reads see either old or new layout consistently (partition key unchanged)

**No coordination needed.** The picking workers and the compaction job are independent.

## Performance

Measured on a 3 GB, 30k-file partition:

| Metric | Before | After |
|--------|--------|-------|
| Parquet objects | 30k | ~3-10 (depends on target file size) |
| Median file size | 49 KB | ~1 MB |
| Bytes | 3 GB (same) | 3 GB (same) |
| Downstream analysis | slow (30k footers) | fast |
| Compaction time | — | ~5–10 min per partition |

## What gets rewritten

**Compacted:**
- `picks/network=XX/year=YYYY/month=MM/*.parquet`

**Unchanged (no rewrite):**
- `shards.jsonl`, `stations.parquet`, `runs/`, `complete/`, `progress/`, `claims/`, `manifests/`

## Monitoring compaction

The compaction job logs to stdout and writes `compaction.jsonl` to the campaign root:

```bash
pixi run -e cloud python scripts/dashboard_check.py \\
    --campaign s3://quakescope-picks-2026/scedc \\
    --show-compaction
```

Also check CloudWatch Logs if running as a Batch job.

## When to skip compaction

- **Short campaigns** (<10k shards): fragmentation doesn't matter
- **Write-only workflows** (no downstream analysis until after campaign): defer compaction to save costs
- **Tight budget**: compaction costs compute (1–2 vCPU-hours); defer if cost is a hard cap

To skip: don't run `monitor_and_compact.py` or call compaction directly.

## Tuning

Edit `sb_catalog/src/parquet_compact.py`:

```python
TARGET_FILE_SIZE_MB = 1.0       # Consolidate to 1 MB files (increase for larger files)
COMPLETION_THRESHOLD = 0.80     # Start at 80% (increase to run later, e.g., 0.95 = at completion)
```

Larger target files → fewer objects, faster downstream reads, but slower compaction and more memory.

## Known limitations

- **Cannot compact while picking to same partition:** If a picking job writes to `picks/network=CI/year=2019/month=07/` while compaction is consolidating it, the new picks might land in the old unconsolidated layout or the new consolidated layout (both are readable). Future runs should read from compacted layout if it exists. Consider waiting for 95%+ completion if this is a concern.
- **Deleted files are not recoverable:** Compaction deletes the originals after consolidating. If something goes wrong, re-run the picking campaign or restore from S3 versioning (if enabled, which it is not by default).
