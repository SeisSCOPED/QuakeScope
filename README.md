# QuakeScope: Machine Learning-based Seismic Phase Picking at Cloud Scale

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A cloud-native workflow for automated seismic phase picking and earthquake detection using deep learning, deployed on AWS with containerized models and managed infrastructure.

**Authors**: Jannes Munchmeyer (munchmej@univ-grenoble-alpes.fr), Yiyu Ni (niyiyu@uw.edu), and Marine Denolle (mdenolle@uw.edu)

## Publications

If you use QuakeScope in research, please cite:

- **Ni et al. (2025a)** — *Geophysical Journal International*
  - Primary QuakeScope methodology, cloud deployment, and operational results
  - DOI: [10.1093/gji/ggxxxx](https://doi.org/10.1093/gji/ggxxxx) *(to be updated)*

- **Ni et al. (2025b)** — *Seismica Data Mine*
  - Earthquake catalog and phase picks dataset from QuakeScope 2026 re-run
  - DOI: [10.26443/seismica.xxxxx](https://doi.org/10.26443/seismica.xxxxx) *(to be updated)*

See [CITATION.cff](CITATION.cff) for BibTeX and other formats.

## Quick Start

**Local tutorials (5 min)**:
```bash
git clone https://github.com/SeisSCOPED/QuakeScope.git
cd QuakeScope
pixi install --environment tutorials
pixi run smoke-test
```

**Cloud deployment**: See [INSTALL.md](INSTALL.md)

## Documentation

| Link | Purpose |
|------|---------|
| [INSTALL_TUTORIALS.md](INSTALL_TUTORIALS.md) | Local setup (pixi + notebooks) |
| [INSTALL.md](INSTALL.md) | AWS Batch/Fargate deployment |
| [docs/rerun_2026/README.md](docs/rerun_2026/README.md) | Production runbook |
| [docs/phasenet_v7_model_description.md](docs/phasenet_v7_model_description.md) | Model architecture & benchmarks |
| [docs/smoke_test_workflow.md](docs/smoke_test_workflow.md) | Validation workflow |
| [SECURITY_AUDIT.md](SECURITY_AUDIT.md) | Security assessment |

## Key Features

- 🌩️ **Cloud-native**: AWS Batch/Fargate for elastic, serverless phase picking
- 🔬 **Production-ready models**: PhaseNet v7 fine-tuned on global seismic data
- 📊 **Scalable database**: DocumentDB for catalog storage and querying
- 🎯 **High accuracy**: PhaseNet P-recall 0.853, MAE 0.340s (cross-domain benchmark)
- 📦 **Containerized**: Docker-based deployment with custom weights and dependencies
- 🔍 **Monitored**: CloudWatch dashboards and SNS alerts for job tracking

## Architecture

```
Continuous Waveforms (SCEDC/NCEDC S3)
           ↓
┌─────────────────────────┐
│  AWS Batch/Fargate      │
│  • PhaseNet inference   │
│  • QuakeXNet classifier │
│  • PyOcto association   │
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  DocumentDB             │
│  • Picks catalog        │
│  • Event associations   │
│  • Metadata             │
└─────────────────────────┘
           ↓
   Analytics & Visualization
```

## Repository Structure

```
QuakeScope/
├── README.md                          # This file
├── CITATION.cff                       # Citation metadata
├── AUTHORS                            # Project authors
├── INSTALL.md                         # Cloud installation
├── INSTALL_TUTORIALS.md               # Tutorial setup
│
├── sb_catalog/
│   ├── models/
│   │   ├── phasenet/                 # PhaseNet weights (v7, original, etc.)
│   │   └── quakexnet/                # Event classifier weights
│   └── src/
│       ├── picking.py                # Inference pipeline
│       ├── s3_helper.py              # S3 data loading
│       └── constants.py              # Configuration
│
├── notebooks/
│   ├── 1_prepare_compute_env.ipynb   # AWS setup
│   ├── 2_prepare_station_metadata.ipynb
│   ├── 3_submit_job.ipynb            # Job submission
│   └── 4_check_database.ipynb        # Results verification
│
├── tutorials/
│   ├── phasenet_smoke_test_ridgecrest.ipynb
│   ├── compare_phasenet_models.ipynb
│   └── seisbench_pyocto_ncedc.ipynb
│
├── docs/
│   ├── rerun_2026/                   # 2026 re-run runbook
│   ├── smoke_test_workflow.md
│   ├── phasenet_v7_model_description.md
│   └── ridgecrest_2019_test_stations.md
│
└── Dockerfile                         # Container image
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- ObsPy, SeisBench, PyOcto (see [INSTALL.md](INSTALL.md) or [INSTALL_TUTORIALS.md](INSTALL_TUTORIALS.md))
- AWS account (for cloud deployment)

## License

[See LICENSE file](LICENSE)

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## Contact

- **Lead Author**: Yiyu Ni (niyiyu@uw.edu)
- **Project PI**: Marine Denolle (mdenolle@uw.edu)
- **Infrastructure**: Jannes Munchmeyer (munchmej@univ-grenoble-alpes.fr)
