# PhaseNet v7 Model Description

## Quick Facts

| Aspect | Details |
|--------|---------|
| **Name** | `finetune_jma_wc_global_v7` |
| **Source** | Denolle-Lab/phasenet-retrain (GitHub local repo) |
| **Config** | `configs/finetune_jma_wc_global_v7.yaml` |
| **Parent Model** | SeisBench `jma_wc` (Japanese regional PhaseNet) |
| **Training Data** | Hybrid corpus: ~20 SeisBench datasets (manifests_v2) |
| **Key Innovation** | Knowledge distillation (alpha=0.3, T=4.0) + no timing loss (timing_beta=0) |
| **Benchmark** (cross-domain split) | P-MAE: 0.340s, P-recall: 0.853, MCC: 0.760 |

## What is v7?

### Design Philosophy
v7 is a **detection-first** model that prioritizes recall over precise timing:
- **Removed timing loss** (timing_beta=0, unlike v6)
- **Frozen distillation teacher** from jma_wc prevents overfitting
- **Global dataset** training → better generalization across networks

### Training Setup
```yaml
Model:
  - Pretrained from jma_wc (Japanese regional, SeisBench)
  - Unfrozen all layers (fine-tune, not frozen)

Distillation:
  - Teacher: frozen jma_wc
  - Student: fine-tuned model
  - Loss = CE(student) + 0.3 * KL(student || teacher @ T=4.0)
  - Effect: Regularizes against catastrophic forgetting, preserves jma_wc's good representations

Loss:
  - Classification loss only (CE on P/S detection)
  - NO timing loss (timing_beta=0.0) — this is the v6→v7 change
  - Rationale: v6 timing_beta=0.01 suppressed P-recall from 0.87 to 0.52

Data:
  - Train: manifests_v2/train.csv (hybrid ~20 datasets)
  - Val: manifests_v2/val.csv
  - Test: manifests_v2/test.csv
  - Window: 3001 samples (30 seconds at 100 Hz)

Optimizer:
  - AdamW, LR=5e-5 (very small—fine-tuning nudge from jma_wc)
  - Scheduler: ReduceLROnPlateau (patience=5)
  - Early stopping: patience=20 on val_loss
  - Max epochs: 150

Batch/Compute:
  - Batch size: 1024
  - Workers: 16
  - Precision: 32-bit (float32)
  - AMP: enabled
```

## v7 vs. Zhu et al. (2019) Original

### Original PhaseNet (Zhu & Beroza, 2019)

| Aspect | Original |
|--------|----------|
| **Training data** | SCSN (California regional network only) |
| **Architecture** | Vanilla ResNet + attention |
| **Distillation** | None |
| **Domain** | California seismic, high SNR |
| **Strengths** | Excellent timing on regional events, optimized for SCSN |
| **Weaknesses** | Limited to California; domain gap for other networks |

### v7 (Denolle Lab)

| Aspect | v7 |
|--------|-----|
| **Training data** | Hybrid: ~20 SeisBench datasets (global) |
| **Architecture** | Same ResNet as jma_wc (identical to original) |
| **Distillation** | Yes (knowledge distillation from jma_wc) |
| **Domain** | Global, mixed SNR (regional + teleseismic) |
| **Strengths** | Better generalization, higher recall, no domain lock-in |
| **Weaknesses** | May sacrifice timing precision; unknown behavior on SCSN due to jma_wc parent |

### Expected Behavior Differences

**On Ridgecrest (SCSN) data:**

| Metric | Original | v7 | Expectation |
|--------|----------|-----|-------------|
| P detection rate | High (SCSN-optimized) | High (global training) | v7 may differ slightly |
| P timing accuracy | Excellent (~100 ms) | Good (~300-400 ms) | v7 deprioritizes timing |
| S detection rate | High | Higher (emphasis on recall) | v7 may detect more S |
| False positives | Low (tuned to SCSN) | Moderate (global data noise mix) | v7 may flag more noise |
| Domain adaptation | Perfect (SCSN) | Unknown (trained on jma_wc parent) | **Risk: need empirical validation** |

## Critical Context: jma_wc as v7's Parent

v7 is **not a direct refinement of the original PhaseNet**—it starts from jma_wc (Japanese network) and fine-tunes on global data. This introduces a **domain shift risk**:

- **Original**: Direct SCSN lineage
- **v7**: Japanese → global path

On SCSN data, v7 may behave differently than the original because:
1. jma_wc was trained on Japanese (typically high-SNR, dense networks)
2. Global fine-tuning rebalances toward lower-SNR, sparser networks
3. No explicit California-specific tuning in v7

**Mitigation:** Smoke test validates that v7 works on SCSN *despite* this lineage shift.

## Leaderboard Context

From `paper_draft.qmd` (phasenet-retrain internal benchmarks on cross-domain split):

v7 sits in the **"good detection, solid timing"** tier:
- **v3** (early distillation baseline): comparable to v7 in performance
- **v7**: "gold standard" for detection (stopped early, epoch ~44 out of 150)
- **v12+**: Attempts to improve recall further (no distillation, alpha=0)
- **v18+**: Latest (fresh init from jma_wc, 2x teleseismic data) — may supersede v7

For QuakeScope 2026, **v7 is the chosen champion** (as documented in `sb_catalog/models/phasenet/README.md`), not v18 or later.

## Reproduction and Deployment

### Convert v7 checkpoint to SeisBench format

The checkpoint lives on the Denolle Lab server (git-ignored). Conversion command:

```bash
cd sb_catalog/models/phasenet
python convert_checkpoint.py \
    --checkpoint /path/to/phasenet-retrain/checkpoints/finetune_jma_wc_global_v7/best.pt \
    --name quakescope2026 \
    --verify
```

This produces:
- `quakescope2026.pt.v1` — PyTorch state dict (student weights only)
- `quakescope2026.json.v1` — SeisBench metadata

The `.json.v1` inherits structure from `jma_wc` (same architecture, sampling rate, input/output samples), with a docstring added:
```
"QuakeScope 2026: phasenet-retrain champion fine-tuned from jma_wc"
```

### Docker deployment

The Dockerfile in QuakeScope copies these files to `/root/.seisbench/models/v3/phasenet/` at build time. Picking jobs can then use:
```bash
quakescope submit --weight quakescope2026 ...
```

## Smoke Test Expectations

The comparison notebook (`tutorials/compare_phasenet_models.ipynb`) should reveal:

1. **Detection coherence**: Both models mark the same major events (or v7 marks more)
2. **Timing reasonableness**: P precedes S with physical intervals
3. **No spurious picks**: Both avoid marking noise as arrivals
4. **Consistency**: Results stable across nearby stations (DAM, GSC, PAS)

If all pass → v7 is production-ready on SCSN.  
If v7 misses obvious arrivals or has high false positives → investigate further or revert to original.

## References

- **phasenet-retrain**: https://github.com/Denolle-Lab/phasenet-retrain
- **Paper draft** (leaderboard, full methods): `/Users/marinedenolle/GitHub/phasenet-retrain/paper_draft.html`
- **SeisBench jma_wc**: https://seisbench.readthedocs.io/en/latest/ → Models → PhaseNet
- **Zhu & Beroza (2019)**, PhaseNet, *Geophysical Journal International* 216(1):261–273: https://doi.org/10.1093/gji/ggy423
