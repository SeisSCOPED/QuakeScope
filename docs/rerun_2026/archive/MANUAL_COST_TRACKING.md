# Manual Cost Tracking (No CloudWatch Budgets)

Since CloudWatch Budgets is restricted on your account, use manual tracking instead.

---

## How It Works

1. **Calculate estimate** from Phase 1 results (automated)
2. **Track actual vCPU-hours** from dashboard or AWS CLI weekly
3. **Compare** to estimate; alert if >20% drift
4. **Update spreadsheet** every Monday

---

## Phase 1c.1: Calculate Cost Estimate (After Phase 1 Results Arrive)

Once Phase 1a/1b jobs complete and you have profiling data:

```bash
# Example: if Phase 1 shows
#   SCEDC: 34 seconds/band-day
#   EarthScope: 8x slower (272 seconds/band-day)
#   Best procs: 2

python scripts/calculate_phase1_cost.py \
  --scedc-seconds 34 \
  --earthscope-seconds 272 \
  --procs 2 \
  --output docs/rerun_2026/cost_estimate_calculated.md
```

**Output:**
```
Phase 1 Cost Calculation

Inputs:
  SCEDC: 34 sec/band-day
  EarthScope: 272 sec/band-day (ratio: 8.0×)
  Processes per vCPU: 2
  Spot price: $0.0148/vCPU-hr

Campaign        Station-Days    vCPU-Hours   Estimated Cost
SCEDC           4,106,669       38,460       $569
NCEDC           5,979,675       56,032       $829
EarthScope      67,983,975      637,363      $9,433
OBS             996,536         9,343        $138
Western         33,799,828      317,081      $4,693

TOTAL           112,866,083     1,058,279    $15,663

Daily burn rate (60-day campaign): $261/day
Budget recommendation: $313/day (with 20% headroom)
```

**Save this output** — it's your cost estimate for Phase 3.

---

## Phase 1c.2: Track Actual Costs Weekly (During Phase 3)

**Every Monday morning:**

### Option A: From Dashboard (Easiest)

1. Open `reports/campaign_dashboard.html`
2. Look at **"vCPU-hours"** tile
3. Record the number in your tracking spreadsheet

### Option B: From AWS CLI (More Precise)

Get exact vCPU-hours from Batch metrics:

```bash
# Get vCPU-hours for a date range
aws cloudwatch get-metric-statistics \
  --namespace AWS/Batch \
  --metric-name RunningTaskCount \
  --dimensions Name=JobQueue,Value=niyiyu_earthscope_missing_station \
  --start-time 2026-09-01T00:00:00Z \
  --end-time 2026-09-08T00:00:00Z \
  --period 3600 \
  --statistics Average \
  --region us-east-2 \
  | jq '.Datapoints | length'  # rough estimate
```

Or check **AWS Console → Batch → Dashboard** for job statistics.

### Option C: Manual Calculation

If dashboard/CLI aren't available:

```
vCPU-hours = (number of concurrent workers) × (8 vCPU per worker) × (hours running)

Example:
- 100 workers running 24 hours = 100 × 8 × 24 = 19,200 vCPU-hours
```

---

## Phase 1c.3: Update Weekly Tracking Spreadsheet

**File:** `docs/rerun_2026/weekly_cost_tracking.csv`

**Every Monday:**

```bash
# Open the spreadsheet
cat docs/rerun_2026/weekly_cost_tracking.csv
```

Add a new row:

| Column | Value | Source |
|--------|-------|--------|
| date | 2026-09-08 | Today's date |
| week | 1 | Week number |
| campaign | scedc | Active campaign |
| shards_complete | 200 | Dashboard "Shards complete" |
| shards_total | 8479 | From plan |
| pct_complete | 2.4 | shards_complete / shards_total |
| vpu_hours_this_week | 120 | Dashboard vCPU-hours minus last week |
| vpu_hours_total | 120 | Total so far |
| est_cost_this_week | 1776 | vpu_hours × $0.0148 |
| est_cost_total | 1776 | Cumulative |
| notes | launched at 25 workers | Any notes |

**Example:**
```csv
2026-09-08,1,scedc,200,8479,2.4,120,120,1776,1776,launched at 25 workers
2026-09-15,2,scedc,800,8479,9.4,400,520,5920,7696,scaling to 100 workers
2026-09-22,3,scedc,2000,8479,23.6,650,1170,9610,17306,on track
```

---

## Phase 1c.4: Cost Drift Alert Thresholds

**Compare weekly cost to estimate:**

| Variance | Action |
|----------|--------|
| ±5% | None (normal variation) |
| +10% | Note it; investigate next week if continues |
| +20% | **INVESTIGATE** (email Marine, check why slower) |
| +50% | **STOP CAMPAIGN** (cost spiraling; diagnose before continuing) |

**Example calculation:**
```
Estimated total: $15,663
Weekly estimate (60-day campaign): $15,663 / 12 weeks ≈ $1,305/week

Week 1 actual: $1,776
Variance: ($1,776 - $1,305) / $1,305 = +36% 🚨

Action: Investigate why slower than expected
- Is one archive slower than profiled?
- Are we running more workers than planned?
- Did --procs setting revert to 1?
```

---

## Phase 1c.5: What to Do If Cost Drifts

### +10–20% (Manageable)

- Check dashboard: is one campaign slower than others?
- Example: "EarthScope is 15× slower, not 8×"
- Decision: Proceed but reduce worker count, or accept higher cost

### +20–50% (Investigate)

- Stop adding new workers
- Check:
  1. Dashboard: what's the actual wall-clock per shard?
  2. CloudWatch Logs: any errors or hangs?
  3. S3: is there contention on a particular prefix?
- Decide: continue at current pace, or pause and re-plan

### >50% (Stop)

- Pause the campaign
- Don't launch more workers
- Existing shards will finish; assess damage
- Investigate root cause before resuming

---

## Tools Available (No Special Permissions Needed)

| Tool | What It Shows | How to Access |
|------|---------------|---------------|
| Dashboard | vCPU-hours, picks, shards complete | `reports/campaign_dashboard.html` (auto-updates hourly) |
| AWS Batch Console | Job status, worker logs | AWS Console → Batch → Dashboard |
| CloudWatch Logs | Worker timing and errors | AWS Console → CloudWatch → Logs → `/aws/batch/job` |
| AWS CLI | vCPU metrics | `aws batch list-jobs`, `aws cloudwatch get-metric-statistics` |

---

## Summary

✅ **Phase 1c without CloudWatch Budgets:**

1. Run cost calculator after Phase 1 results: `python scripts/calculate_phase1_cost.py --scedc-seconds X --earthscope-seconds Y --procs Z`
2. Track actual vCPU-hours weekly from dashboard
3. Update spreadsheet `weekly_cost_tracking.csv` every Monday
4. Compare to estimate; alert if >20% drift
5. Keep logs of any issues for post-campaign analysis

---

## When Phase 1 Results Arrive

```bash
# After Phase 1 jobs complete (30–60 min from now)

# 1. See what the profiling revealed
python scripts/phase1_collect_results.py

# 2. Calculate cost from those measurements
python scripts/calculate_phase1_cost.py \
  --scedc-seconds <from results> \
  --earthscope-seconds <from results> \
  --procs <from results> \
  --output docs/rerun_2026/cost_estimate_calculated.md

# 3. Review the estimate and go/no-go decision
cat docs/rerun_2026/cost_estimate_calculated.md

# Then proceed to Phase 2
```

That's it! Manual tracking is simpler and doesn't require any special AWS permissions.
