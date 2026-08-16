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
| **Benchmark** | P-MAE 0.340 s, P-recall 0.853, MCC 0.760 — read alongside the parent, below |

## v7 versus its own parent

This is the number that should drive the deployment decision. From the
leaderboard in `phasenet-retrain/paper_draft.qmd` (§Leaderboard, all distances):

| Model | P-recall | S-recall | P-MAE (s) | P-outlier | MCC |
|---|--:|--:|--:|--:|--:|
| `jma_wc` (parent, **not** fine-tuned) | **0.881** | **0.549** | 0.374 | 0.071 | **0.790** |
| **v7** (champion fine-tune) | 0.853 | 0.505 | **0.340** | **0.063** | 0.760 |

**Fine-tuning bought timing and paid for it in detection.** v7 improves P-MAE by
about 9% and outlier rate by about 11%, but the un-fine-tuned parent still wins
P-recall, S-recall, and MCC. The paper states it plainly: *no single fine-tuned
model dominates the baseline across all metrics.*

Choose v7 when arrival-time precision drives the science — relocation, tomography,
moment tensors. Prefer `jma_wc` when catalog completeness matters more than a few
tens of milliseconds of timing.

### Two caveats the paper itself raises

Both are flagged as blocking in that repo's own audit, and neither is resolved:

1. **Selection bias.** In-distribution validation metrics were found not to
   predict cross-domain performance, so version selection was driven by reading
   the benchmark leaderboard across 19 versions — iterated selection on the test
   set. The paper warns the v7-over-`jma_wc` timing advantage "may shrink on a
   truly held-out set."
2. **Train/test independence is unverified.** The `cross_domain` rows are
   byte-identical to `all` because `trained_on` is `None` for every fine-tune, so
   the split is currently a no-op. The fine-tunes were trained on datasets that
   also populate the benchmark, and the training manifests are not committed, so
   benchmark leakage cannot presently be ruled out.

Treat the numbers above as the best available internal comparison, not as
independently validated performance.

## What is v7?

### Design Philosophy
v7 is a **timing-first** model — it reaches the best P-MAE of any version tested
by removing the explicit timing loss rather than adding one:
- **Removed timing loss** (timing_beta=0, unlike v6, where even β=0.01 collapsed
  P-recall from 0.87 to 0.52)
- **Frozen distillation teacher** from jma_wc prevents catastrophic forgetting
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
  - AdamW, LR=5e-6 (very small - a fine-tuning nudge from jma_wc)
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
| **Strengths** | Best P-MAE and lowest outlier rate of any version tested; no single-region lock-in |
| **Weaknesses** | Lower recall and MCC than its own parent; behaviour on SCSN unvalidated |

### Expected behaviour differences

Measured against the parent on the internal benchmark, and otherwise unmeasured
— the rows marked *unmeasured* are hypotheses to test, not results:

| Metric | v7 vs `jma_wc` parent | Basis |
|---|---|---|
| P timing (MAE) | Better: 0.340 s vs 0.374 s | Leaderboard |
| P outlier rate | Better: 0.063 vs 0.071 | Leaderboard |
| P recall | Worse: 0.853 vs 0.881 | Leaderboard |
| S recall | Worse: 0.505 vs 0.549 | Leaderboard |
| MCC | Worse: 0.760 vs 0.790 | Leaderboard |
| Behaviour on SCSN | *unmeasured* | Neither model was tuned on SCSN; run the smoke test |
| False-positive rate in noise | *unmeasured* on this data | A separate noise-pool audit exists in the training repo |

## Critical Context: jma_wc as v7's Parent

v7 is **not a direct refinement of the original PhaseNet**—it starts from jma_wc (Japanese network) and fine-tunes on global data. This introduces a **domain shift risk**:

- **Original**: Direct SCSN lineage
- **v7**: Japanese → global path

On SCSN data, v7 may behave differently than the original because:
1. jma_wc was trained on Japanese (typically high-SNR, dense networks)
2. Global fine-tuning rebalances toward lower-SNR, sparser networks
3. No explicit California-specific tuning in v7

**Mitigation:** Smoke test validates that v7 works on SCSN *despite* this lineage shift.

## Why v7 and not a later version

Twenty versions were trained. The later ones were run to completion and none
displaced v7 — the recurring pattern is a recall-versus-timing seesaw, where
anything that buys recall costs P-MAE:

| Version | Change | Outcome |
|---|---|---|
| v3 | First stable KD recipe (α=0.3, T=4, LR 5e-6) | P-MAE 0.368 — the template v7 refines |
| v6 | Tiny timing loss (β=0.01) | P-recall collapsed 0.87 → 0.52; "even 0.01 is lethal" |
| **v7** | v6 with β=0 | **Best P-MAE 0.340**; stopped early around epoch 44 of 150 |
| v13 | α=0 + noise + presence loss | Best recall 0.888 and MCC 0.943, but P-MAE 0.967 |
| v18 | S-balanced + 1.5× tele + focal loss | P-MAE 0.459 — ~58% worse timing than v7 |
| v19 | Local+regional only, P-MAE-focused | P-MAE 0.381 — still did not beat v7 |
| v20 | v7's exact recipe + soft-label CE | Worse than v7 on **both** P-MAE and recall |

The team's own conclusions: knowledge distillation at α≈0.3 is the indispensable
cross-domain regularizer, explicit timing and presence losses backfire, and
in-distribution validation metrics do not predict benchmark P-MAE — which is
what forced the leaderboard-driven selection flagged as a caveat above.

## Reproduction and Deployment

### Convert v7 checkpoint to SeisBench format

**The checkpoint is not in the git repository and not on any local clone.**
`checkpoints/` and `results/` in `phasenet-retrain` contain only `.gitkeep` —
both are git-ignored, and the weights live on the Denolle Lab back-end Linux
server. Per `configs/finetune_jma_wc_global_v7.yaml`, its path there is relative
to the repo root:

```
checkpoints/finetune_jma_wc_global_v7/best.pt
```

Copy it down, then convert:

```bash
scp <labserver>:<path-to>/phasenet-retrain/checkpoints/finetune_jma_wc_global_v7/best.pt .

cd sb_catalog/models/phasenet
python convert_checkpoint.py --checkpoint best.pt --name quakescope2026 --verify
```

`--verify` installs the pair into the local SeisBench cache and reloads it
through `from_pretrained`, so a successful run means the tutorials will pick it
up automatically. The name `quakescope2026` is this repository's deployment
label for the v7 fine-tune; the training repo knows it only as v7.

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
