# Phase 1 Cost Estimates

**Date:** 2026-08-31  
**Status:** PENDING (Phase 1 tests running, results expected 2026-09-01)

---

## Measured Inputs (To Be Filled In)

### From 1a: EarthScope I/O Profile

| Metric | SCEDC | EarthScope | Ratio |
|--------|-------|-----------|-------|
| s3.get seconds | _pending_ | _pending_ | _pending_ |
| Wall-clock per shard | _pending_ | _pending_ | _pending_ |

**Decision Gate:**
- [ ] If ratio <5×: ✅ EarthScope acceptable
- [ ] If ratio 5–15×: ⚠️ Manageable but monitor
- [ ] If ratio >15×: 🛑 Re-plan required

### From 1b: Process Parallelism Sweep

| Procs | SCEDC Wall-Clock | Cost Model | EarthScope Wall-Clock | Cost Model |
|-------|-----------------|------------|----------------------|------------|
| 1 | _pending_ | _pending_ | _pending_ | _pending_ |
| 2 | _pending_ | _pending_ | _pending_ | _pending_ |
| 4 | _pending_ | _pending_ | _pending_ | _pending_ |
| 8 | _pending_ | _pending_ | _pending_ | _pending_ |

**Recommendation:** (filled in after results) Use `--procs _`

---

## Cost Calculations

### Base Unit: Cost per Band-Day

From Phase 1a, assuming _X_ seconds per band-day on SCEDC:

```
Cost = (seconds/band-day) × (8 vCPU / 3600 sec) × $0.0148/vCPU-hr
     = X × 8 / 3600 × 0.0148
     = $Y per band-day
```

### Campaign Costs (using recommended --procs from 1b)

| Campaign | Station-Days | Band-Days | Cost/BD | Procs | vCPU-Hours | Cost |
|----------|--------------|-----------|---------|-------|-----------|------|
| SCEDC | 2.5M | 2.5M | $Y | — | | |
| NCEDC | 6.0M | 6.0M | $Y | — | | |
| EarthScope | 69M | 69M | $Z | — | | |
| OBS | 1.0M | 1.0M | $Y | — | | |
| Western | 34M | 34M | $Y | — | | |
| **TOTAL** | **112.9M** | **112.9M** | | | **_pending_** | **$_pending_** |

**Notes:**
- EarthScope cost ($Z) depends on Phase 1a ratio; if 10× slower than SCEDC, then $Z = 10×$Y
- All campaigns use same `--procs` value (from Phase 1b)
- Actual vCPU-hours = band-days × (seconds/band-day) / 3600

---

## Budget & Headroom

Proposed budget: **$50,000**

| Component | Cost | Budget | Headroom |
|-----------|------|--------|----------|
| Estimated total (Phase 1) | $_pending_ | $50k | +_$pending_ |
| 20% safety margin | +10% | | |
| **Budgeted amount** | | **$50k** | **_pending_** |

**Daily burn rate:** $_pending_ / 60 days ≈ $250/day (varies by campaign)

---

## Phase 1 Decision

### Measurement Results

```
[FILL IN AFTER RESULTS]

1a EarthScope ratio: ___ ×
Decision: ✅ / ⚠️ / 🛑

1b Recommended --procs: ___
Cost model: ___ vCPU-hours saved vs --procs 1
```

### Go / No-Go for Phase 2

- [ ] ✅ **GO:** All measurements favor proceeding
  - EarthScope is <15× slower (manageable)
  - --procs 2 or 4 saves cost
  - Proceed to SCEDC smoke test

- [ ] ⚠️ **GO WITH CAUTION:** Trade-offs exist
  - EarthScope slower than hoped but not blocking
  - Process parallelism shows diminishing returns
  - Proceed but monitor weekly cost drift

- [ ] 🛑 **HOLD / RE-PLAN:** Blocking issue found
  - EarthScope is >20× slower (cost explodes)
  - No process parallelism benefit (must use --procs 1)
  - Action: [describe mitigation or re-plan]

---

## Next Steps (After Results)

1. ✅ Fill in Phase 1 measurements above
2. ✅ Review go/no-go decision
3. ✅ Set CloudWatch Budget alert to $250/day (Task 1c)
4. ✅ Create weekly cost tracking sheet
5. → Proceed to Phase 2 or re-plan
