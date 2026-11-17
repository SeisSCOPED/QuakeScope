# EQTransformer weights baked into the image

| weight | campaign | files | md5 of .pt |
|---|---|---|---|
| `pnw` | pnw-eqt-fill (extends `pnw-eqt-2023/`) | `pnw.pt.v1`, `pnw.json.v1` | a92a4b7044b0... |

Upstream SeisBench `EQTransformer` `pnw` (Ni et al. 2023), copied from the
SeisBench 0.12.3 cache on 2026-11-14. The upstream `docstring` in the JSON is
copied from `original` (Mousavi et al. 2020); the tensors are not - the `.pt`
differs from `original.pt.v1`, and the JSON carries the PNW 1-45 Hz bandpass.

Committed because the first pnw-eqt-fill smoke test (2026-11-14 22:56 UTC, image
6e99850, seisbench 0.12.5) failed every shard with "No version for weight 'pnw'
available": the image had no local copy and SeisBench did not fetch one. See
../phasenet/README.md for why campaign weights live in the image.
