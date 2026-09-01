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

They were not baked in before. The Dockerfile's other weight source was a
156 MB tarball from `munchmeyer.de` dated `230614` (June 2023), which predates
`jma_wc`, so a worker running campaigns 1-3 downloaded 4.13 MB from
`hifis-storage.desy.de` during startup — observed in the logs of the
2026-09-01 SCEDC smoke test:

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

## Adding a weight to the image

The image contains **only** the four weights above. The Dockerfile downloads no
weights at all — it copies this directory in — so a build is hermetic and cannot
vary with what upstream is serving that day.

To add one:

```bash
# 1. fetch it into your local SeisBench cache from the official repository
pixi run -e cloud python -c \
  "import seisbench.models as sbm; sbm.PhaseNet.from_pretrained('<name>')"

# 2. copy every version of the pair in - see the note below on why "every"
cp ~/.seisbench/models/v3/phasenet/<name>.pt.v*   .
cp ~/.seisbench/models/v3/phasenet/<name>.json.v* .

# 3. verify it loads with no network, from a cache holding only these files
rm -rf /tmp/sbtest && mkdir -p /tmp/sbtest/models/v3/phasenet
cp *.pt.v* *.json.v* /tmp/sbtest/models/v3/phasenet/
SEISBENCH_CACHE_ROOT=/tmp/sbtest pixi run -e cloud python -c "
import seisbench.models as sbm, socket
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(OSError('blocked'))
print(sum(p.numel() for p in sbm.PhaseNet.from_pretrained('<name>').parameters()), 'params')"

# 4. commit, push, let the Action rebuild, then re-register the job definition
#    against the new short-SHA tag
```

Step 3 is the one that matters. Without it you find out on 1,500 workers.

**Copy every version, not just `.v1`.** Which version SeisBench resolves depends
on its own version, and it is not always the highest-numbered or the one you
expect: seisbench 0.12.3 resolves `original` and `instance` to **`.v2`** while
`jma_wc` and `obs` have only `.v1`. Shipping `original.pt.v1` alone would put a
file in the image that SeisBench never asks for, leaving the runtime download
exactly where it was — and it would look like it had been fixed.

## What is deliberately not in the image

Everything else SeisBench offers — `instance`, `stead`, `scedc`, `ethz`,
`diting`, and the rest — still works, and downloads on first use. That is fine
for ad-hoc and notebook work and never happens on a campaign path.

One asymmetry to know about: `picker.py`'s own `--weight` default is `instance`,
which is **not** in the image, so the bare legacy entry point downloads on first
run. The `work` subcommand that campaigns use has its own parser defaulting to
`jma_wc`, and is unaffected. Add `instance` here if that ever runs at scale.

## Why the tarball was dropped

The Dockerfile previously pulled a 156 MB tarball from `munchmeyer.de` dated
`230614`, marked `TODO !!!TEMPORARY!!!`. Before removing it, its 40 PhaseNet
files were compared by SHA-256 against the current official ones:
**all 40 identical**. So nothing regressed in dropping it, and — worth ruling
out, since the tarball would have shadowed anything it supplied — the 2025
campaign was not running divergent weights.

What it lacked was everything published since June 2023: `aq2009`, `jma`,
`jma_wc`, `phasenet_sn`, `pisdl`, `volpick`. It was 156 MB serving nothing the
official host does not, while missing the one weight the campaign most needs.

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
