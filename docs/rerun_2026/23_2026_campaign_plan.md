# 23 — Campaign execution plan 2026

Living roadmap for the 2026 QuakeScope re-run. Update this as you progress; decisions made at gates update the downstream phases.

**Status:** 2026-08-31. S3 contention fixed, startup staggering live, compaction workflow ready. Ready to enter Phase 1.

---

## Executive Summary

Five campaigns (SCEDC, NCEDC, EarthScope, OBS, Western) totaling 112.9M station-days. Two critical unknowns (EarthScope I/O speed, process parallelism efficiency) can swing cost 20–50×. **Measure both in Phase 1 before scaling beyond 200 workers.** Automate Parquet compaction at 80% completion.

---

## Phase 1: Measurement (This Week)

**Goal:** Eliminate cost uncertainty. Run 1–2 hour experiments to validate Phase 2–6 assumptions.

**Timeline:** 3–4 days (run all in parallel)

### 1a. EarthScope I/O Profile

**Why:** EarthScope stores one multi-channel object per station-day. SCEDC stores one per channel. Suspected 25–30× slower per station-day.

**What:** Run `--profile` on EarthScope shards; compare `s3.get` latency and bytes to SCEDC baseline.

**Steps:**

- [ ] Pick 5 representative EarthScope shards (mix restricted-tier and open-data networks)
- [ ] Pick 5 SCEDC shards (same date ranges for fair comparison)
- [ ] Modify `launch_western_states.ipynb` notebook or write a one-off script:

```python
import json
import boto3
from sb_catalog.src.s3_state import S3CampaignState

# Submit test jobs with --profile and --max-shards 1
state = S3CampaignState("s3://quakescope-picks-2026/earthscope")
shards = state.read_shards()

# Pick shards manually: look for different networks, years
earthscope_test_shards = [shards[i] for i in [100, 500, 1000, 5000, 10000]]

# Submit to Batch with --profile flag
# Job definition already supports it; it's just not used in the dashboard
```

- [ ] **Store results:** Create `docs/rerun_2026/profile_earthscope.json`:

```json
{
  "earthscope": [
    {"shard": "...", "wall_clock": 125.3, "s3_get": 95.2, "bytes": 450_000_000},
    ...
  ],
  "scedc": [
    {"shard": "...", "wall_clock": 35.7, "s3_get": 2.1, "bytes": 50_000_000},
    ...
  ],
  "analysis": "EarthScope is X.Xx slower"
}
```

- [ ] **Expected outputs:**
  - If <10 min/shard: EarthScope is acceptable, proceed with campaigns as planned
  - If 10–30 min/shard: EarthScope is significant but manageable; watch cost drift
  - If >30 min/shard: Re-plan; consider running EarthScope last or with fewer workers

**Owner:** Marine  
**Effort:** 2–3 hrs  
**Blocks:** EarthScope campaign (campaigns 3, 4, 5)  
**Status:** 🔴 NOT STARTED

---

### 1b. Process Parallelism Sweep

**Why:** Every benchmark uses 1 process on 8 vCPU (7 cores idle). `--procs 4` could cut cost 2× or show no improvement.

**What:** Run identical shards with `--procs 1,2,4,8`; measure total wall-clock and memory.

**Steps:**

- [ ] Use the same 5 SCEDC shards from 1a
- [ ] Submit 4 runs per shard (1 per `--procs` value):

```bash
for procs in 1 2 4 8; do
  aws batch submit-job \
    --job-name "test-procs-$procs" \
    --job-queue niyiyu_earthscope_missing_station \
    --job-definition quakescope_v3_worker:2 \
    --container-overrides \
      "environment=[{name=PROCS,value=$procs}]"
done
```

- [ ] Monitor logs; extract `worker_loop` stats for each process:
  - Wall-clock time
  - Memory peak (RSS)
  - vCPU utilization (if available from CloudWatch)

- [ ] **Store results:** Create `docs/rerun_2026/profile_processes.json`:

```json
{
  "shard_id": "2019187-...",
  "results": [
    {"procs": 1, "wall_clock": 34.2, "memory_mb": 8192, "cost_model": "1x"},
    {"procs": 2, "wall_clock": 20.1, "memory_mb": 10240, "cost_model": "1.7x"},
    {"procs": 4, "wall_clock": 15.3, "memory_mb": 14336, "cost_model": "2.2x"},
    {"procs": 8, "wall_clock": 14.9, "memory_mb": 24576, "cost_model": "2.3x"}
  ],
  "recommendation": "--procs 4 gives 2.2x cost savings"
}
```

- [ ] **Expected outputs:**
  - If `--procs 4` is 2–3× cheaper per vCPU-hour: **Switch to `--procs 4` for all campaigns**
  - If diminishing returns: **Use `--procs 2` (safe, proven)**
  - If memory pressure or contention: **Stick with `--procs 1` (tested, safe)**

**Owner:** Marine  
**Effort:** 1–2 hrs  
**Blocks:** vCPU resource planning for all campaigns  
**Status:** 🔴 NOT STARTED

---

### 1c. Billing Alert + Cost Tracking Setup

**Why:** Cost Explorer is blocked by SCP. Need manual tracking to catch 2× budget overruns before month-end.

**Steps:**

- [ ] Set CloudWatch Billing Alert:
  - Go to **AWS Console → Billing → Budgets**
  - Create **Daily Budget** at $250/day threshold
  - Email alert at 100% threshold
  
- [ ] Document cost assumptions in `docs/rerun_2026/cost_estimate_2026.md`:

```markdown
## Cost Estimate (based on Phase 1 measurements)

### Inputs

Station-day counts are from [21_queues_written.md](21_queues_written.md), which
was verified by reading the written queues back from S3. Do not restate them
from memory — earlier drafts of this plan carried 2.5M for SCEDC, which came
from a superseded estimate in doc 12 and is 1.66x low.

- SCEDC: 4,106,669 station-days, ${price}/band-day (measured in Phase 1)
- NCEDC: 5,979,675 station-days, ${price}/band-day
- EarthScope: 67,983,975 station-days, ${price}/band-day (measured in Phase 1)
- OBS: 996,536 station-days, ${price}/band-day
- Western: 33,799,828 station-days, ${price}/band-day

### Totals
- Estimated vCPU-hours: XXX,XXX
- Estimated cost: $XX,XXX
- Budget headroom: $5,000
```

- [ ] Create a weekly tracking spreadsheet (or commit to `docs/rerun_2026/weekly_cost_tracking.csv`):

```
week,campaign,shards_done,vpu_hours,estimated_cost,actual_cost,variance
1,scedc,500,100,$1500,?,?
2,scedc,2000,400,$6000,?,?
...
```

**Owner:** Marine  
**Effort:** 1 hr  
**Blocks:** None (informational)  
**Status:** 🔴 NOT STARTED

---

## Phase 1 Completion Gate

**Decision:** Do we proceed to Phase 2?

| Condition | Action |
|-----------|--------|
| 1a EarthScope <10 min/shard | ✅ Proceed with all campaigns as planned |
| 1a EarthScope 10–30 min/shard | ⚠️ Proceed but monitor cost; may re-plan if 1b is unfavorable |
| 1a EarthScope >30 min/shard | 🛑 **STOP.** Re-plan; do not launch EarthScope campaigns yet |
| 1b `--procs 4` is 2–3× cheaper | ✅ Switch all remaining jobs to `--procs 4` |
| 1b `--procs 4` shows no improvement | ✅ Stick with `--procs 1–2`; accept the vCPU cost |

**Gate sign-off:** Marine reviews 1a, 1b, 1c results and decides **GO / NO-GO for Phase 2**.

**Expected date:** 2026-09-05 (Friday)

---

## Phase 2: SCEDC Smoke Test (Week 1–2, starting ~2026-09-02)

**Goal:** Validate assumptions on real infrastructure before committing to full campaign.

**Scope:** 4,106,669 station-days over 8,479 shards ([21](21_queues_written.md)).
Shards average 484 station-days and are not a fixed 40x20 grid — the sampled
shard `2015175-2015195-4cd53b5d98c6` holds 23 stations over 20 days. Station
count per shard varies; the 40x20 figure describes the 2025 run, not the v3
planner.

**Timeline:** 3–5 days

### 2.1. Launch SCEDC at 100 workers

**Steps:**

- [ ] Update job definition based on Phase 1 results:
  - If 1b chose `--procs 4`: set environment var `PROCS=4`
  - Else: leave at default `--procs 1`

- [ ] Launch the campaign:

```bash
# Notebook: launch western_states.ipynb (adapt for SCEDC)
pixi run -e cloud python notebooks/5_submit_job_parquet.ipynb \
  --campaign scedc \
  --num_workers 100
```

Or via Batch console:
```
Job name: quakescope_v3_worker_scedc_launch
Array size: 100
```

- [ ] **Milestone 1** (day 1): First shards land in Parquet
  - [ ] Check `s3://quakescope-picks-2026/scedc/picks/` — any `.parquet` objects?
  - [ ] Check dashboard: non-zero pick count?

- [ ] **Milestone 2** (day 2–3): Ramp to steady state
  - [ ] 50+ shards complete
  - [ ] Check: vCPU utilization stable? Cost per shard matches Phase 1 estimate?

- [ ] **Milestone 3** (day 5): ~2,000 shards done (25% of campaign)
  - [ ] [ ] Review weekly metrics (see Phase 3 below)
  - [ ] Cost so far vs. phase 1 estimate ±10%?
  - [ ] **Decision: scale to 200 workers, or pause to investigate?**

**Owner:** Marine  
**Effort:** Monitoring (5–10 min daily)  
**Status:** 🔴 NOT STARTED

---

### 2.2. Monitoring Checklist (daily during Phase 2)

Add this to your calendar as a recurring 10-minute daily check:

- [ ] **Dashboard** (`reports/campaign_dashboard.html`):
  - vCPU in use: should be ~100 (200 after scale-up)
  - Picks in catalogue: growing?
  - Shards complete: on track for 25% by day 5?

- [ ] **CloudWatch Logs** (sample 5 random workers):
  - Any errors? Long hangs?
  - Heartbeat failures? (should be rare)

- [ ] **Cost tracking:**
  - Tally vCPU-hours so far
  - Compare to Phase 1 per-shard estimate
  - If >20% over: investigate (archive slower? memory pressure?)

- [ ] **S3 State** (spot checks):
  ```bash
  # List active claims
  aws s3 ls s3://quakescope-picks-2026/scedc/claims/ --recursive | wc -l
  # Should show <150 active claims (normal transient)
  
  # Check completed count
  aws s3 ls s3://quakescope-picks-2026/scedc/complete/ --recursive | wc -l
  ```

**Owner:** Marine  
**Status:** 🔴 NOT STARTED

---

## Phase 2 Completion Gate

**Decision:** Do we scale to full SCEDC? Do we launch NCEDC? Do we proceed to EarthScope?

| Metric | Target | Action if Miss |
|--------|--------|--------|
| Shards complete (day 5) | >2,000 (25%) | Investigate slowness; may be archive-specific |
| vCPU-hours vs. estimate | ±10% | Adjust process count down or re-check Phase 1 |
| Cost drift | <5% | Proceed; monitor weekly |
| S3 errors | 0 | Proceed; adaptive retries are working |

**Gate sign-off:** Marine decides **SCALE / HOLD / RE-PLAN**.

**Expected date:** 2026-09-07 (Wednesday, day 6 of Phase 2)

---

## Phase 3: Campaign Execution (Weeks 2–6)

**Goal:** Run all five campaigns to completion. Monitor cost. Auto-compact at 80%.

### 3.1. Scale SCEDC (if Phase 2 gate passes)

- [ ] Scale workers to 200–400 (based on throughput from Phase 2)
- [ ] **Estimated completion:** ~10–15 days at 200 workers

**Steps:**

```bash
# Double the worker count
aws batch submit-job \
  --job-name "quakescope_v3_worker_scedc_scale" \
  --job-queue niyiyu_earthscope_missing_station \
  --array-props minIndex=0,maxIndex=199 \
  --job-definition quakescope_v3_worker:2 \
  --container-overrides \
    "environment=[{name=CAMPAIGN,value=s3://quakescope-picks-2026/scedc},{name=WEIGHT,value=jma_wc}]"
```

- [ ] Monitor: Check dashboard hourly; weekly cost tracking
- [ ] **Milestone:** SCEDC 80% complete → **Launch compaction** (see 3.4)

---

### 3.2. Launch NCEDC (once SCEDC is >50% done)

- [ ] Repeat 3.1 for NCEDC campaign (6M station-days)
- [ ] Parallel campaigns are safe (different prefixes, no state sharing)

```bash
aws batch submit-job \
  --job-name "quakescope_v3_worker_ncedc_launch" \
  --job-queue niyiyu_earthscope_missing_station \
  --array-props minIndex=0,maxIndex=199 \
  --job-definition quakescope_v3_worker:2 \
  --container-overrides \
    "environment=[{name=CAMPAIGN,value=s3://quakescope-picks-2026/ncedc},{name=WEIGHT,value=jma_wc}]"
```

---

### 3.3. Launch EarthScope Campaigns (if Phase 1a was favorable)

**Only if Phase 1a showed <10 min/station-day; else re-plan.**

Three campaigns:
- Campaign 3: EarthScope onshore (jma_wc)
- Campaign 4: OBS (obs weights)
- Campaign 5: Western states (original weights)

Each has 60+ million station-days; run sequentially or with reduced parallelism.

```bash
# Launch campaign 3 (earthscope)
aws batch submit-job \
  --job-name "quakescope_v3_worker_earthscope_launch" \
  --job-queue niyiyu_earthscope_missing_station \
  --array-props minIndex=0,maxIndex=99 \
  --job-definition quakescope_v3_worker:2 \
  --container-overrides \
    "environment=[{name=CAMPAIGN,value=s3://quakescope-picks-2026/earthscope},{name=WEIGHT,value=jma_wc}]"
```

**Stagger timing:** Don't launch all three at once. Wait for each to reach 50% before launching the next.

---

### 3.4. Automatic Compaction at 80%

Compaction runs automatically via background monitoring job. If not configured:

- [ ] **Option A (recommended):** Launch monitoring job as separate Batch job:

```bash
aws batch submit-job \
  --job-name "quakescope_monitor_scedc" \
  --job-queue niyiyu_earthscope_missing_station \
  --job-definition quakescope_v3_worker:2 \
  --container-overrides \
    "command=python,scripts/monitor_and_compact.py,--campaign,s3://quakescope-picks-2026/scedc,--poll-interval,3600"
```

- [ ] **Option B (manual):** Monitor yourself; run compaction at 80%:

```bash
pixi run -e cloud python -m sb_catalog.src.parquet_compact \
  --campaign s3://quakescope-picks-2026/scedc \
  --dryrun  # preview first

# Then without --dryrun
pixi run -e cloud python -m sb_catalog.src.parquet_compact \
  --campaign s3://quakescope-picks-2026/scedc
```

**Expected timeline:**
- SCEDC reaches 80%: day 8–10
- Compaction takes: 5–10 min per partition (runs in background)
- Dashboard refreshes: within 1 hour of compaction completing

See [22_parquet_compaction.md](22_parquet_compaction.md) for details.

---

### 3.5. Weekly Monitoring Routine

**Every Monday 9 AM UTC:**

- [ ] Check dashboard: vCPU-hours, pick counts, shards complete
- [ ] Update `docs/rerun_2026/weekly_cost_tracking.csv`:

```
week,scedc_shards,scedc_cost,ncedc_shards,ncedc_cost,earthscope_shards,earthscope_cost,total_cost,notes
1,500,$1500,0,$0,0,$0,$1500,launched scedc
2,2000,$6000,500,$1500,0,$0,$7500,launched ncedc
3,4000,$12000,1500,$4500,0,$0,$16500,launched earthscope
...
```

- [ ] Compare to Phase 1 estimate:
  - If within ±10%: no action, continue
  - If +10% to +20%: investigate (is one archive slower?)
  - If >+20%: **escalate** (email Marine, may need to re-plan)

- [ ] Check `s3://quakescope-picks-2026/*/compaction.jsonl` for compaction progress

**Owner:** You (set a calendar reminder)  
**Effort:** 15 min/week

---

## Phase 4: Campaign Completion (Week 6+)

**Goal:** All shards complete, data is consolidated and ready for analysis.

### 4.1. Finalization Checklist

Once all shards show `complete/`:

- [ ] Verify pick counts match expectations (compare to [21_queues_written.md](21_queues_written.md)):

```bash
pixi run -e cloud python notebooks/6_check_parquet.ipynb
# Should show, per 21_queues_written.md:
# - SCEDC: 4,106,669 station-days → ~X picks
# - NCEDC: 5,979,675 station-days → ~Y picks
# - etc.
```

- [ ] Verify all partitions are compacted:

```bash
# List all partitions; count objects per partition
aws s3 ls s3://quakescope-picks-2026/scedc/picks/ --recursive | grep "\.parquet$" | wc -l
# Should be much smaller than 30k (ideally <500 total objects)
```

- [ ] Generate final dashboard:

```bash
pixi run -e cloud python scripts/campaign_dashboard.py \
  --campaigns scedc ncedc earthscope obs western \
  --output reports/campaign_dashboard_final.html
```

- [ ] Stop monitoring job (if running):

```bash
# Find the monitor job ID
aws batch list-jobs --job-queue niyiyu_earthscope_missing_station --status RUNNING | grep monitor

# Cancel it
aws batch terminate-job --job-id <job-id> --reason "campaign complete"
```

---

### 4.2. Archive & Document

- [ ] Create final cost report:

```markdown
# 2026 Campaign Final Report

## By Campaign

| Campaign | Station-Days | Shards | Duration | vCPU-Hours | Cost | Cost/SD |
|----------|--------------|--------|----------|-----------|------|---------|
| SCEDC | 2.5M | 8,479 | 8 days | 800 | $12,000 | $0.005 |
| NCEDC | 6.0M | 14,941 | 15 days | 1,900 | $28,000 | $0.005 |
| ... | ... | ... | ... | ... | ... | ... |
| **TOTAL** | **112.9M** | **255,699** | **~40 days** | **~$XX,XXX** | ... | ... |

## Comparison to Estimate

- Phase 1 estimate: $XX,XXX
- Actual: $XX,XXX
- Variance: ±X%
```

- [ ] Write lessons learned in `docs/rerun_2026/LESSONS_LEARNED.md`:

```markdown
# Lessons Learned – 2026 Campaign

## What Went Well
- S3 contention fix (adaptive retries + stagger) worked perfectly
- Compaction automated and transparent
- Process parallelism (if tested) validated/rejected for future

## What To Improve
- EarthScope I/O was slower than estimated; consider [optimization X]
- Compaction should start at 70% instead of 80% for earlier feedback
- Dashboard needed [feature] to track [metric]

## Data Quality
- All 112.9M station-days processed
- Pick counts: [summary by network]
- Outliers: [notes on any suspicious campaigns]

## For Next Time
- Use Phase 1 measurements without re-testing if <6 months
- Consider pre-compaction strategy if >1M picks per partition
- Request Cost Explorer SCP exemption before campaign (not during)
```

---

## Phase 5: Long-term Improvements (Post-Campaign)

**Timeline:** 1–2 weeks after campaign completion

### 5.1. If Process Parallelism Sweep Showed a Win

- [ ] Retry `--procs` sweep on high-memory instances (if 1b was inconclusive)
- [ ] Document optimal `--procs` setting per instance type
- [ ] Add to job definition template for next campaign

### 5.2. Cost Tracking & Planning

- [ ] Export final campaign costs (manual from billing console + CloudWatch logs)
- [ ] Annualize: if annual updates are expected, use this as a baseline
- [ ] Request SCP exemption for Cost Explorer (took 1 hr this time; automate for next)

### 5.3. Optional: SQS + Distributed Work Queue

If scaling to 5k+ workers in future:

- [ ] Evaluate SQS + S3 checkpoint state (alternative to S3 LIST)
- [ ] Measure: how much does SQS reduce contention?
- [ ] Effort: 8 hrs; benefit: cleaner work distribution, no LIST operations

---

## Success Criteria

Campaign is **✅ SUCCESSFUL** when:

- ✅ All 112.9M station-days processed (zero gaps)
- ✅ Total campaign cost ≤ $50k (Phase 1 estimate ± 20%)
- ✅ Picks are queryable (Parquet compacted and consolidated)
- ✅ No data loss or catastrophic re-runs (zero >10% shard retries)
- ✅ Dashboard operational throughout (no >1 hour downtime)

Campaign **❌ NEEDS REPLANNING** if:

- ❌ EarthScope takes >15 min/station-day (cost explodes >$100k)
- ❌ Process parallelism showed OOM or contention (must use `--procs 1`)
- ❌ Cost drifts >30% above Phase 1 estimate mid-campaign
- ❌ Compaction fails and downstream analysis is blocked

---

## Rollback / Pause Procedures

**If something goes wrong, you have two options:**

### Option A: Pause & Diagnose

```bash
# Stop all workers for a campaign
aws batch terminate-job --job-id <array-job-id> --reason "pausing for investigation"

# The queue state is durably stored in S3; no work is lost.
# To resume:
aws batch submit-job \
  --job-name "quakescope_v3_worker_<campaign>_resume" \
  --job-queue niyiyu_earthscope_missing_station \
  --array-props minIndex=0,maxIndex=<new-worker-count>
```

### Option B: Revert a Campaign

If a campaign is fundamentally broken (wrong weights, bad data):

```bash
# Delete the output (if not yet compacted)
aws s3 rm s3://quakescope-picks-2026/<campaign>/picks/ --recursive
aws s3 rm s3://quakescope-picks-2026/<campaign>/complete/ --recursive

# Recreate the queue
pixi run -e cloud python -c "
from sb_catalog.src.s3_state import S3CampaignState
state = S3CampaignState('s3://quakescope-picks-2026/<campaign>')
state.write_shards(...)  # re-write from plan
"

# Relaunch
```

---

## Reference Documents

| Phase | Document | Purpose |
|-------|----------|---------|
| 1 | [19_earthscope_access.md](19_earthscope_access.md) | EarthScope credential setup |
| 1 | [16_skypilot_vs_fargate.md](16_skypilot_vs_fargate.md) | Baseline performance expectations |
| 2–3 | [15_monitoring.md](15_monitoring.md) | AWS watch, budgets, emergency stop |
| 3–4 | [22_parquet_compaction.md](22_parquet_compaction.md) | Automatic Parquet consolidation |
| 3–4 | [21_queues_written.md](21_queues_written.md) | Campaign scope and shard counts |
| 4–5 | [11_launch_plan.md](11_launch_plan.md) | Campaign definitions and networks |

---

## Contact & Escalation

| Issue | Escalation | Action |
|-------|-----------|--------|
| S3 contention (>100 SlowDown errors/min) | EarthScope support + AWS TAM | Partition claims by prefix; enable adaptive retries (already done) |
| EarthScope credential failure | EarthScope support | Check `ES_OAUTH2__REFRESH_TOKEN` in Secrets Manager; refresh if expired |
| Cost spike (>50% over estimate) | Marine | Pause campaign; investigate via Phase 1 measurements; re-plan |
| Data corruption (negative picks, invalid timestamps) | Marine | Check model weights; validate input data; may require campaign restart |
| Dashboard offline >1 hour | Marine | `scripts/campaign_dashboard.py` manual rebuild; check GitHub Actions logs |

---

## Tracking

**Last updated:** 2026-08-31  
**Next review:** 2026-09-05 (after Phase 1 measurements)  
**Campaign start:** ~2026-09-02  
**Expected completion:** ~2026-10-15  

Update this document as you progress. Each phase gate should record the decision, measurement, and date.
