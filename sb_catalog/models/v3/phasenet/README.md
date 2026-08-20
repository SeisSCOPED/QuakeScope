# Custom PhaseNet weights

Place custom SeisBench PhaseNet weights here as a pair of files:

```
<weightname>.pt.v1      # torch state dict (SeisBench format)
<weightname>.json.v1    # SeisBench weight metadata (at minimum: {})
```

They are copied into `/root/.seisbench/models/v3/phasenet/` at Docker build
time and can then be selected at job submission with `--weight <weightname>`.

The default weights (`instance`, plus the set from the SeisBench model
repository baked in by the Dockerfile) remain available.

## Provenance of the 2026 re-run weights

- **Phase picker**: the champion model of
  [Denolle-Lab/phasenet-retrain](https://github.com/Denolle-Lab/phasenet-retrain)
  is experiment **v7** (fine-tuned from SeisBench `jma_wc` with knowledge
  distillation; benchmark P-MAE 0.340 s, P-recall 0.853, MCC 0.760 on the
  cross-domain split — see that repo's `paper_draft.qmd` leaderboard). Note
  that v7 wins on *timing* only: the un-fine-tuned `jma_wc` parent still scores
  higher P-recall (0.881) and MCC (0.790), so the choice is a trade, not a
  strict upgrade — see
  [`docs/phasenet_v7_model_description.md`](../../../docs/phasenet_v7_model_description.md).
  **The converted pair is already committed here** as
  `quakescope2026.pt.v1` + `quakescope2026.json.v1`, so `--weight quakescope2026`
  works without further setup.

  It came from `phasenet_jma_wc_ft_v7.pt` at the root of the `phasenet-retrain`
  repo (added there 2026-08-16 in commit `705f5a5`; note this is *not* the
  git-ignored `checkpoints/finetune_jma_wc_global_v7/best.pt` path the training
  config writes to). To regenerate after a retrain:

  ```
  python convert_checkpoint.py --checkpoint phasenet_jma_wc_ft_v7.pt --name quakescope2026 --verify
  ```

- **QuakeXNet** (classifier): `../quakexnet/base.pt.v1` carries the latest
  weights. Verified 2026-08-16 by SHA-256 against `src/models/v3/quakexnet/base.pt.v3`
  in [Akashkharita/pnw_seismic_event_detection](https://github.com/Akashkharita/pnw_seismic_event_detection)
  (published there in commit `d9af765`) — the tensors are the same model, which
  supersedes the Oct 2024 `best_model_MyCNN_2d.pth` in
  [Denolle-Lab/PNW_Seismic_Event_Classification](https://github.com/Denolle-Lab/PNW_Seismic_Event_Classification).

  **Re-saved with CPU storage**, so the file is no longer byte-identical to
  upstream while every tensor value is unchanged (63 tensors, zero mismatches).
  As published, the weights carry CUDA storage tags and `torch.load` raises
  `RuntimeError: Attempting to deserialize object on a CUDA device` on any
  machine without a GPU. The production image installs **CPU-only** PyTorch, so
  `QuakeXNet.from_pretrained("base")` failed there — the regression arrived with
  `0a0df51` (2025-09-15); the earlier weights in `070a4fd` loaded on CPU fine.
  Worth reporting upstream, since `base.pt.v3` has the same problem.
