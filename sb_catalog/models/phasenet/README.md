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
  The checkpoint lives on the lab back-end server at
  `phasenet-retrain/checkpoints/finetune_jma_wc_global_v7/best.pt`
  (checkpoints are git-ignored there). Convert it to a SeisBench pair with:

  ```
  python convert_checkpoint.py --checkpoint best.pt --name quakescope2026 --verify
  ```

- **QuakeXNet** (classifier): `../quakexnet/base.pt.v1` already carries the
  latest weights — byte-identical to `base.pt.v3` published in
  [Akashkharita/pnw_seismic_event_detection](https://github.com/Akashkharita/pnw_seismic_event_detection)
  (Dec 2025), which supersedes the Oct 2024 `best_model_MyCNN_2d.pth` in
  [Denolle-Lab/PNW_Seismic_Event_Classification](https://github.com/Denolle-Lab/PNW_Seismic_Event_Classification).
  No update needed.
