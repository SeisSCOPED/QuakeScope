# Phase 1c: Billing Alert Setup & Cost Tracking

**Effort:** 1 hour  
**Goal:** Set up financial controls so cost drift is caught early, not on the final bill.

---

## Task 1c.1: Set CloudWatch Billing Alert

CloudWatch will email you daily if spend reaches $250/day threshold.

### Steps

1. **Open AWS Billing Console** (as root or billing admin):
   - https://console.aws.amazon.com/billing/

2. **Navigate to Budgets**:
   - Click **Budgets** (left sidebar)

3. **Create Daily Budget**:
   - Click **Create budget**
   - Budget name: `QuakeScope-2026-daily`
   - Budget type: **Cost budget**
   - Budgeting method: **Simple budgeting**
   - Period: **Daily**
   - Budgeted amount: `$250.00`

4. **Set Alert Thresholds**:
   - Click **Add alert threshold**
   - Alert type: **Actual**
   - Alert threshold: **100%** (i.e., when you hit the limit)
   - Email recipients: your email (mdenolle@uw.edu)
   - Click **Confirm budget**

5. **Verify**:
   - You should receive a confirmation email
   - Budget appears in the Budgets list

**Why $250/day?** 
- At $15,800 total budget for western campaign, that's ~$250/day over 60 days
- 2× overrun would be $500/day, caught in first week, not at month-end

---

## Task 1c.2: Document Phase 1 Cost Assumptions

Create `docs/rerun_2026/phase1_cost_estimate.md` based on Phase 1 measurements:

```bash
cat > docs/rerun_2026/phase1_cost_estimate.md << 'EOF'
# Phase 1 Cost Estimates (based on measurements)

**Date:** 2026-09-05  
**Status:** PENDING (Phase 1 tests not yet complete)

## Measured Inputs

From Phase 1a (EarthScope profile):
- SCEDC s3.get: X s/band-day
- EarthScope s3.get: Y s/band-day (ratio: Z×)

From Phase 1b (process parallelism):
- --procs 1: A wall-clock, B cost model
- --procs 2: C wall-clock, D cost model
- --procs 4: E wall-clock, F cost model (RECOMMENDED)

## Cost Calculations

| Campaign | Station-Days | Band-Days | $/Band-Day | Procs | vCPU-Hours | Cost |
|----------|--------------|-----------|-----------|-------|-----------|------|
| SCEDC | 2.5M | 2.5M | (from 1a) | 4 | | |
| NCEDC | 6.0M | 6.0M | (from 1a) | 4 | | |
| EarthScope | 69M | 69M | (from 1a) | 4 | | |
| OBS | 1.0M | 1.0M | (from 1a) | 4 | | |
| Western | 34M | 34M | (from 1a) | 4 | | |
| **TOTAL** | **112.9M** | **112.9M** | | | | **$XX,XXX** |

## Comparison to Original Estimate

- Original estimate: $15,800 (based on SCEDC-only measurements)
- Phase 1 revised estimate: $XX,XXX (based on all archives, all procs values)
- Variance: ±X%
- Budget headroom: $5,000

## Decision

[ ] Proceed to Phase 2 SCEDC smoke test
[ ] Modify --procs value based on Phase 1b results
[ ] Re-plan (if Phase 1a shows EarthScope is too slow)
EOF
```

---

## Task 1c.3: Create Weekly Cost Tracking Sheet

You'll fill this in **every Monday morning** during the campaign (Phase 3–4).

```bash
cat > docs/rerun_2026/weekly_cost_tracking.csv << 'EOF'
date,campaign,shards_complete,shards_total,pct_complete,vpu_hours_week,vpu_hours_total,estimated_cost_week,estimated_cost_total,actual_cost_total,variance_pct,notes
2026-09-09,scedc,200,8479,2.4,50,50,750,750,?,?,launched
2026-09-16,scedc,1000,8479,11.8,200,250,3000,3750,?,?,scaling up
EOF
```

**How to fill it in:**
1. Check dashboard for shards complete and vCPU-hours
2. Calculate: `estimated_cost = vpu_hours_total × $0.0148/vpu-hr` (use Phase 1 measured rate)
3. Calculate: `variance_pct = (actual_cost - estimated_cost) / estimated_cost`
4. Notes: any anomalies (slow archive, OOM, S3 errors, etc.)

---

## Task 1c.4: Export Billing Data Weekly

Cost Explorer is blocked by SCP. Manual tracking:

```bash
# Weekly: Export actual costs (best effort)
# Until Cost Explorer is unblocked, estimate from vCPU-hours + Parquet storage

# vCPU-hours:
aws cloudwatch get-metric-statistics \
  --namespace AWS/Batch \
  --metric-name RunningTaskCount \
  --start-time 2026-09-02T00:00:00Z \
  --end-time 2026-09-09T00:00:00Z \
  --period 3600 \
  --statistics Average \
  --region us-east-2

# Parquet storage (rough):
aws s3 ls s3://quakescope-picks-2026/scedc/picks/ --recursive --summarize | grep "Total Size"
```

---

## Reference: Cost Model

Based on Phase 1 measurements (to be filled in):

```
Cost per vCPU-hour: $0.0148 (Fargate Spot, estimated)
Typical band-day cost: X sec × (8 vCPU / 3600 sec) × $0.0148 = $Y

Examples:
- SCEDC (30 s/band-day): $Y per band-day
- EarthScope (if Z× slower): $Z×Y per band-day
```

---

## Escalation Thresholds

**If weekly variance exceeds these, investigate immediately:**

| Variance | Action |
|----------|--------|
| ±5% | None (within noise) |
| +10% | Check dashboard; is one archive slow? |
| +20% | Pause campaign; diagnose (may be S3 throttle, or Phase 1 estimate was wrong) |
| +50% | **ESCALATE.** Stop new shards; review Phase 1 measurements |

---

## Checklist

- [ ] CloudWatch Budget created at $250/day
- [ ] Confirmation email received
- [ ] Phase 1 cost estimate document created
- [ ] Weekly tracking spreadsheet created
- [ ] Budget monitor URL bookmarked for quick access
- [ ] Escalation thresholds understood and documented

**Done?** Move to Phase 1a and 1b job submission.
