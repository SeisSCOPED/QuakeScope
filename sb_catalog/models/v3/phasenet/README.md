# Custom PhaseNet weights

Place custom SeisBench PhaseNet weights here as a pair of files:

```
<weightname>.pt.v1      # torch state dict (SeisBench format)
<weightname>.json.v1    # SeisBench weight metadata (at minimum: {})
```

They are copied into `/root/.seisbench/models/v3/phasenet/` at Docker build
time and can then be selected at job submission with `--weight <weightname>`.

## Every weight a 2026 campaign uses is committed here

`jma_wc`, `obs` and `original` are upstream SeisBench weights, committed here
deliberately rather than fetched at runtime.

They were not baked in before. The Dockerfile's other source is a tarball from
`munchmeyer.de` dated `230614` (June 2023), which predates `jma_wc`, so a
worker running campaigns 1-3 downloaded 4.13 MB from `hifis-storage.desy.de`
during startup — observed in the logs of the 2026-09-01 SCEDC smoke test:

```
Weight file jma_wc.pt.v1.partial not in cache. Downloading...
Downloading from https://hifis-storage.desy.de/.../jma_wc.pt.v1
```

At 1,500 workers that is 1,500 cold-start fetches of an external academic host
in the critical path, for a dependency the campaign does not control. Roughly
7.7 MB in the image removes it.

| weight | campaigns | files |
|---|---|---|
| `jma_wc` | 1-3 (SCEDC, NCEDC, EarthScope) | `.pt.v1`, `.json.v1` |
| `obs` | 4 (OBS) | `.pt.v1`, `.json.v1` |
| `original` | 5 (western) | `.pt.v1/.v2`, `.json.v1/.v2` |
| `quakescope2026` | none this run - v7, see below | `.pt.v1`, `.json.v1` |

`original` ships **both** v1 and v2. Which one SeisBench resolves depends on the
version pip installs at image build time, and shipping only v1 would silently
re-enable the download.

Verified to load with the network blocked, from a cache holding only these
files:

```bash
SEISBENCH_CACHE_ROOT=/tmp/sbtest python -c "
import seisbench.models as sbm, socket
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(OSError('blocked'))
for n in ['jma_wc','obs','original','quakescope2026']:
    sbm.PhaseNet.from_pretrained(n)"
```

`jma_wc` and `quakescope2026` both report 1,070,899 parameters, consistent with
v7 being the `jma_wc` fine-tune described below; `obs` and `original` are
268,499 and 268,443.

To add another upstream weight, fetch it and copy the pair in:

```bash
python -c "import seisbench.models as sbm; sbm.PhaseNet.from_pretrained('<name>')"
cp ~/.seisbench/models/v3/phasenet/<name>.{pt,json}.v* .
```

The rest of the June 2023 tarball (`instance`, `stead`, `scedc`, ...) remains
available for anything not listed above, but is not relied on by this campaign.

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
