# PhaseNet Smoke Test Workflow

**Goal**: Validate the original PhaseNet v7 weights on real waveforms before deploying to production.

**Focus**: Visual inspection of picks against waveforms to ensure physical plausibility.

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Configure Event & Stations                               │
│    - Ridgecrest M7.1 (July 5, 2019)                          │
│    - 2-3 SCSN stations at varying distances                 │
│      (close: DAM; moderate: GSC; far: PAS)                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Fetch Waveforms from S3                                  │
│    - NCEDC S3 bucket: ncedc-pds/continuous_waveforms        │
│    - Network: CI (SCSN)                                     │
│    - Format: HH* (broadband, high-sample-rate)             │
│    - Date: 2019-07-05 (day of year 186)                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Load Model Weights                                       │
│    Primary: v7 (quakescope2026)                             │
│    Fallback: SeisBench 'instance' (for comparison)          │
│    Location: sb_catalog/models/phasenet/                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Run Phase Picking                                        │
│    - Model.classify() on 24h waveforms                      │
│    - P threshold: 0.3, S threshold: 0.3                     │
│    - Collect all picks in PickList                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Visual Validation                                        │
│    For each station:                                         │
│    ├─ Plot 3-component waveforms (Z, N, E)                 │
│    ├─ Overlay picks as vertical lines (blue=P, green=S)    │
│    ├─ Highlight event time window (±10s before, +60s after)│
│    └─ Inspect: Do picks align with onset features?         │
│                                                              │
│    Expected patterns:                                       │
│    • P picks on vertical (Z) or any component              │
│    • S picks after P with P-S ~ 3-10s (distance dependent)│
│    • Picks cluster near event time, not random             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Sanity Checks                                            │
│    ├─ P picks precede S picks for same event               │
│    ├─ P-S intervals are physically reasonable              │
│    ├─ Picks cluster around event time                      │
│    ├─ No spurious picks in noise windows                   │
│    └─ Consistent results across multiple stations          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ PASS/FAIL Decision                                          │
│                                                              │
│ PASS if:                                                    │
│ ✓ Picks align with waveform onsets on all channels        │
│ ✓ P consistently precedes S with 3-10s intervals          │
│ ✓ No spurious picks in noise windows                       │
│ ✓ Results consistent across stations                       │
│                                                              │
│ FAIL if:                                                    │
│ ✗ Picks miss obvious onsets                                │
│ ✗ S precedes P or P-S intervals unrealistic               │
│ ✗ High false-positive rate in noise                        │
│ ✗ Inconsistent behavior across stations                    │
└─────────────────────────────────────────────────────────────┘
```

## Running the Smoke Test

### Setup
1. Ensure NCEDC S3 credentials (anonymous access, no setup needed)
2. Verify v7 weights are in `sb_catalog/models/phasenet/`:
   ```bash
   ls sb_catalog/models/phasenet/*.v1
   ```
3. If v7 weights not present, convert from lab server:
   ```bash
   cd sb_catalog/models/phasenet
   python convert_checkpoint.py --checkpoint /path/to/best.pt --name quakescope2026 --verify
   ```

### Execute
1. Open `tutorials/phasenet_smoke_test_ridgecrest.ipynb`
2. Run cells in order:
   - **Config**: Adjust event time, stations, thresholds if needed
   - **Fetch**: Download waveforms from S3 (5-10 min depending on data availability)
   - **Load**: Load model weights
   - **Pick**: Run inference (1-5 min per station)
   - **Visualize**: Inspect plots (main validation step)
   - **Sanity Checks**: Verify statistics
3. Record findings in a cell markdown as pass/fail with notes

### What to Look For

**Good signs:**
- Clean P arrival on Z component (strong upward first motion)
- Clear S arrival on horizontal (N/E) with ~5-8s delay from P at close distance
- Picks clustered around 17:33:50 UTC (event time)
- No false positives in pre-event noise (before ~17:33:40)

**Red flags:**
- Picks scattered randomly throughout day
- S picks before P
- Picks on obvious noise transients, not seismic arrivals
- Missing obvious onsets on close stations
- Inconsistent behavior between nearby stations

## Expected Stations & Distances

| Station | Network | Lat   | Lon     | Dist (km) | Expected P-S (s) |
|---------|---------|-------|---------|-----------|------------------|
| DAM     | CI      | 35.73 | -117.52 | ~3        | ~0.5            |
| GSC     | CI      | 35.47 | -116.43 | ~130      | ~18             |
| PAS     | CI      | 34.15 | -118.17 | ~225      | ~32             |

(Assuming v_p=5.8 km/s, v_s=3.3 km/s typical for SoCal crust)

## Comparing with Baseline

If comparing v7 to "instance" weights:
1. Run notebook with both models (modify Step 3 to load both)
2. Create side-by-side plots of picks for each model
3. Document differences in pick counts, precision, false positives
4. Record in PR/commit message

## Output Artifacts

- **Waveform plots** (PNG): Visual record of picks on data
- **Pick statistics** (CSV): All picks with times, phases, confidence
- **Notebook execution log**: Terminal output showing processing steps
- **Validation checklist**: Markdown cell summarizing pass/fail decisions

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| "FileNotFoundError" on S3 | Station/channel unavailable | Skip station, try different date |
| Model fails to load | v7 weights not in right format | Convert checkpoint again, check file permissions |
| No picks returned | Waveforms too noisy or thresholds too high | Lower P_THRESHOLD, S_THRESHOLD to 0.2 |
| Picks misaligned with waveforms | Trace time/sampling mismatch | Check stats on trace, verify trim() call |
| Slow execution | Network latency or large data | Reduce time window or number of stations |

## Next Steps After Smoke Test

1. **PASS**: Proceed to full production deployment
2. **FAIL**: Debug model or revert to previous weights
3. **PARTIAL PASS**: Document caveats and acceptable error rates
4. **UNCLEAR**: Expand to additional events/stations for more evidence

## References

- **Event**: Ridgecrest earthquake sequence, July 2019
  - USGS: https://earthquake.usgs.gov/earthquakes/events/2019ca
- **Waveform data**: NCEDC public S3 bucket (Continuous California seismic data)
- **PhaseNet**: Zhu & Beroza (2019), https://doi.org/10.1038/s41467-019-09748-z
- **SeisBench**: Woollam et al. (2022), https://doi.org/10.1109/IGARSS46834.2022.9883952
