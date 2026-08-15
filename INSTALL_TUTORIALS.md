# Tutorial Installation

## Quick Start

```bash
git clone https://github.com/SeisSCOPED/QuakeScope.git
cd QuakeScope

# Install pixi: https://pixi.sh
curl -fsSL https://pixi.sh/install.sh | bash

# Install & run
pixi install --environment tutorials
pixi run -e tutorials install-kernel   # registers the "QuakeScope (tutorials)" Jupyter kernel
pixi run -e tutorials smoke-test
```

Select **QuakeScope (tutorials)** as the kernel when a notebook opens (Jupyter defaults to whatever kernel was last used otherwise).

Verified working (macOS, `pixi 0.76.2`): `seisbench`, `obspy`, `torch`, `pyocto`, `ipykernel` all import cleanly in the `tutorials` env, and the kernel registers and is visible to `jupyter kernelspec list`.

## Environments

| Environment | PyTorch | Platforms | Use |
|---|---|---|---|
| `tutorials` | CPU | all | **Default** |
| `tutorials-gpu` | CUDA 11.8 | Linux, Windows only | GPU acceleration — no macOS build exists |
| `dev` | CPU | all | Testing, linting, formatting |

## GPU Setup

```bash
pixi install --environment tutorials-gpu   # Linux/Windows only
pixi shell -e tutorials-gpu
jupyter notebook
```

## Alternative: pip + venv

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_tutorials.txt
```

## Tutorials

- `pixi run -e tutorials smoke-test` — PhaseNet v7 validation (Ridgecrest)
- `pixi run -e tutorials compare-models` — v7 vs original comparison
- `pixi run -e tutorials lab` — Jupyter Lab

## Troubleshooting

**"pixi: command not found"**
```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

**"could not find pixi.toml"** — run from the repo root:
```bash
cd /path/to/QuakeScope && ls pixi.toml
```

**Reinstall from scratch**
```bash
pixi install --environment tutorials --force-reinstall
```

## Full Setup

For cloud deployment, see [INSTALL.md](INSTALL.md).
