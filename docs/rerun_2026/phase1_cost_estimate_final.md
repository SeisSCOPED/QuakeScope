# Cost Estimate (Calculated from Phase 1)

**Generated:** 2026-08-31
**Method:** Manual calculation from Phase 1 profiling results

## Inputs

- SCEDC baseline: 30.0 sec/band-day
- EarthScope: 30.0 sec/band-day (1.0× slower)
- Process parallelism: --procs 1
- Spot price: $0.0148/vCPU-hr (Fargate, estimated)

## Costs by Campaign

| Campaign | Station-Days | vCPU-Hours | Cost |
|----------|--------------|-----------|------|
| scedc | 4,106,669 | 34,222 | $506 |
| ncedc | 5,979,675 | 49,831 | $737 |
| earthscope | 67,983,975 | 566,533 | $8,385 |
| obs | 996,536 | 8,304 | $123 |
| western | 33,799,828 | 281,665 | $4,169 |
| **TOTAL** | **112,866,083** | **940,555** | **$13,920** |

## Budget Recommendation

- **Estimated total cost:** $13,920
- **Daily burn rate:** $232/day (60-day campaign)
- **Safety margin:** +20% = $16,704
- **Proposed budget:** $50,000

## Tracking During Campaign

Since CloudWatch Budgets is restricted on this account, track cost manually:

1. **Weekly:** Check AWS Billing console for actual costs (or use `aws ce` CLI)
2. **Compare to estimate:** vCPU-hours × $0.0148/vCPU-hr
3. **Update spreadsheet:** docs/rerun_2026/weekly_cost_tracking.csv
4. **Alert thresholds:** If cost >20% over estimate mid-campaign, investigate

## Go/No-Go Decision

Based on this estimate:
- [ ] ✅ **GO** - Cost is acceptable (<$60k)
- [ ] ⚠️ **GO WITH CAUTION** - Cost is high but manageable ($60–100k)
- [ ] 🛑 **HOLD** - Cost is too high; need to re-plan (>$100k)
