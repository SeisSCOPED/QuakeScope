# 02 — Installing the new weights and building the container

> **The classifier is deferred for the 2026 campaign** — see the
> [runbook](README.md). This page stays accurate for its weights and for the
> container build; just do not pass `--classifier` when submitting.

**Where the best weights come from** (verified August 2026):

- **QuakeXNet**: `sb_catalog/models/quakexnet/base.pt.v1` is the current model —
  SHA-256 verified against `base.pt.v3` in
  `Akashkharita/pnw_seismic_event_detection`. It has been **re-saved with CPU
  storage**, because as published it carries CUDA tags and cannot load under
  the image's CPU-only PyTorch. Tensor values are unchanged.
- **Phase picker**: the champion of the `Denolle-Lab/phasenet-retrain`
  project is experiment **v7** (jma_wc fine-tune; P-MAE 0.340 s, recall
  0.853, MCC 0.760 — but note it trades recall against its own parent, see
  [`../phasenet_v7_model_description.md`](../phasenet_v7_model_description.md)).
  **Already converted and committed** as
  `sb_catalog/models/phasenet/quakescope2026.{pt,json}.v1`, from
  `phasenet_jma_wc_ft_v7.pt` at the root of the phasenet-retrain repo. To
  regenerate after a retrain, convert it with
  [sb_catalog/models/phasenet/convert_checkpoint.py](../../sb_catalog/models/phasenet/convert_checkpoint.py)
  and commit the resulting pair here (step 2 below).

Two models get new weights:

1. **QuakeXNet** — the event classifier (`sb_catalog/src/classifier.py`).
2. **A SeisBench phase picker** (PhaseNet-family) — does the P/S picking.

Both are baked into the Docker image at build time, and the image is built
**automatically by GitHub Actions** every time you push to `main`
(`.github/workflows/docker.yml` → `ghcr.io/seisscoped/quakescope`). You never
run `docker push` by hand.

## 1. QuakeXNet weights (classifier)

The classifier loads `QuakeXNet.from_pretrained("base")`, which resolves to
the files shipped in this repo. To install the new weights, **replace the file
in place** (this is exactly how the September 2025 update was done):

```bash
cp /path/to/new_quakexnet_weights.pt sb_catalog/models/quakexnet/base.pt.v1
```

Keep the filename `base.pt.v1`. The companion `base.json.v1` stays as is
(it's just `{}`) unless the new training changed the architecture — if the
architecture changed, `classifier.py` must be updated to match, as was done
in commit `0a0df51`.

If you also retrained the 1-D variant, same procedure in
`sb_catalog/models/quakexnetoned/`.

## 2. Phase-picker weights (SeisBench)

Two cases:

- **The new weights are published in the SeisBench repository** (i.e.
  `sbm.PhaseNet.from_pretrained("<name>")` works on your laptop): nothing to
  add to the repo. You will just pass `--weight <name>` at submission time.
  Caveat: every container downloads the weight at startup, so baking it into
  the image (next bullet) is still kinder to the SeisBench servers when you
  run hundreds of jobs.
- **The weights are files you were handed** (the likely case): put them in
  `sb_catalog/models/phasenet/` as a SeisBench pair:

  ```bash
  cp /path/to/new_picker.pt   sb_catalog/models/phasenet/quakescope2026.pt.v1
  cp /path/to/new_picker.json sb_catalog/models/phasenet/quakescope2026.json.v1
  ```

  (If no `.json` metadata file was provided, create one containing the same
  metadata structure as other SeisBench PhaseNet weights — ask the student who
  trained it to export both files with
  `model.save(path)` / or verify with `sbm.PhaseNet.load(...)`.)

  The Dockerfile copies this folder into `/root/.seisbench/models/v3/phasenet/`
  inside the image, which makes the weight available as
  `--weight quakescope2026`.

**Name the weight something unique and descriptive** (`quakescope2026`,
`pnw2026`, …). The name is recorded in the database provenance (`sb_runs`),
so it's how you'll tell this catalog apart from the old one forever.

> If the new picker is a *different architecture* (e.g. EQTransformer instead
> of PhaseNet), the folder must match the SeisBench class name in lowercase
> (`models/eqtransformer/`) and you'd add a corresponding COPY line in the
> Dockerfile, then submit with `--model EQTransformer --weight <name>`.

## 3. Push and let GitHub build the image

```bash
cd ~/GitHub/QuakeScope
git checkout -b weights-2026
git add sb_catalog/
git commit -m "New QuakeXNet and phase picker weights for 2026 re-run"
git push origin weights-2026
```

Open a PR to `main` on <https://github.com/SeisSCOPED/QuakeScope>, merge it,
then watch the **Actions** tab — the `build-and-push` workflow takes ~5–10
minutes. When green, the image exists as:

- `ghcr.io/seisscoped/quakescope:latest`
- `ghcr.io/seisscoped/quakescope:<short-sha>` (e.g. `:a1b2c3d`)

**Write down the short-sha tag.** In Phase D you should pin the job
definition to it instead of `:latest`, so that a mid-campaign push can never
silently change what the running jobs use.

## 4. Smoke-test the image locally (recommended)

With Docker Desktop running:

```bash
docker pull ghcr.io/seisscoped/quakescope:latest
```

Check the weights are inside:

```bash
docker run --rm --entrypoint ls ghcr.io/seisscoped/quakescope:latest /root/.seisbench/models/v3/phasenet /root/.seisbench/models/v3/quakexnet
```

You should see your `quakescope2026.pt.v1` and `base.pt.v1`. A full
end-to-end local test (pick one station-day into the database) is easiest
from the EC2 controller once the database is up — see the end of
[03_documentdb.md](03_documentdb.md).

Next: [03_documentdb.md](03_documentdb.md)
