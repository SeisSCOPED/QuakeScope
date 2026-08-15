# Pixi Setup

## Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

## Environments

```bash
pixi install --environment tutorials        # CPU (default)
pixi install --environment tutorials-gpu    # NVIDIA GPU
pixi install --environment dev              # Testing/linting
```

## Commands

```bash
pixi shell -e tutorials              # Activate
pixi run smoke-test                  # Run tutorial
pixi run lab                         # Jupyter Lab
pixi run lint                        # Check code
pixi run format                      # Auto-format
```

## GPU Setup

Edit `pixi.toml` line 18 (enable pytorch-cuda) or:
```bash
pixi install --environment tutorials-gpu
```

Requires CUDA 11.8 drivers.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `pixi: command not found` | Add to PATH: `export PATH="$HOME/.pixi/bin:$PATH"` |
| `No module named 'seisbench'` | `pixi install -e tutorials --force-reinstall` |
| `could not find pixi.toml` | Run from QuakeScope root: `pwd` → check output |

## Docs

- Pixi: https://pixi.sh
- Conda-forge: https://conda-forge.org
