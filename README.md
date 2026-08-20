# Cloud-native Machine Learning workflow for earthquake event detection and phase picking

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A cloud-native workflow for automated seismic phase picking and earthquake detection using deep learning, deployed on AWS with containerized models and managed infrastructure.

**Authors**: Jannes Munchmeyer (munchmej@univ-grenoble-alpes.fr), Yiyu Ni (niyiyu@uw.edu), and Marine Denolle (mdenolle@uw.edu)

## Publications

If you use QuakeScope in research, please cite:

- Ni, Y., Denolle, M. A., Münchmeyer, J., Wang, Y., Feng, K. F., Suarez, C. G. J.,
  Thomas, A. M., Trabant, C., Hamilton, A., Mencin, D. (2025). *A Review of Cloud
  Computing and Storage in Seismology.* Geophysical Journal International,
  243(1), ggaf322. [10.1093/gji/ggaf322](https://doi.org/10.1093/gji/ggaf322)
- Ni, Y., Denolle, M. A., Thomas, A. M., Münchmeyer, J., Hamilton, A., Wang, Y.,
  Bachelot, L., Trabant, C., Mencin, D. (2025). *A Global-scale Database of
  Seismic Phases from Cloud-based Picking at Petabyte Scale.* Seismica, 4(2).
  [10.26443/seismica.v4i2.1738](https://doi.org/10.26443/seismica.v4i2.1738)

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
| [docs/rerun_2026/15_monitoring.md](docs/rerun_2026/15_monitoring.md) | Cost alerts and emergency stop |
| [notebooks/5_submit_job_parquet.ipynb](notebooks/5_submit_job_parquet.ipynb) | Launch a Fargate campaign with Parquet output (no DocumentDB) |
| [notebooks/6_check_parquet.ipynb](notebooks/6_check_parquet.ipynb) | Query the Parquet catalogue |
| [reports/](reports/) | Rendered benchmark reports, published at [seisscoped.org/QuakeScope](https://seisscoped.org/QuakeScope/) |
| [SECURITY_AUDIT.md](SECURITY_AUDIT.md) | Security assessment |

## Cost monitoring

Campaigns run on Spot instances that are easy to leave running. Two layers watch
for that; neither is on by default.

**What is running right now**, from your laptop:

```bash
pixi run -e cloud watch      # instances, spot vs on-demand, $/hr, $/month
```

Read-only, and it prints the stop commands for whatever it finds.

**Hourly alerts by email**, via GitHub Actions
([`.github/workflows/aws-watch.yml`](.github/workflows/aws-watch.yml)). Three
things must all be true, and none is automatic:

1. **The workflow must be on the default branch.** GitHub runs `schedule:`
   triggers *only* from the default branch; on a feature branch it never fires.
2. **Credentials must be set** — repository variable `AWS_WATCH_ROLE_ARN` (OIDC,
   preferred, no stored keys) or the `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
   secrets. The IAM policy is describe-only.
3. **You must watch the repo**, since the alert is an issue comment and GitHub
   turns that into email.

Test it without waiting an hour:

```bash
gh workflow run aws-watch.yml
```

It does **not** email hourly. One issue is the dashboard, its body updates
silently, and a comment — which notifies — is posted only when the state
changes: idle → running, a new warning, or all-clear.

**Turning it off**, at the level you want:

```bash
gh workflow disable aws-watch.yml     # stop entirely (enable to resume)
```

To keep the dashboard but stop the email, unsubscribe from that one issue. To
silence all repo email, set the repo to *Participating and @mentions*. To pause,
drop the `schedule:` block and keep `workflow_dispatch`.

Add an **AWS Budgets** daily alert as an independent backstop — it fires even if
Actions, the credentials, or the repo are broken. Setup, the IAM policy and the
emergency-stop sequence are in
[docs/rerun_2026/15_monitoring.md](docs/rerun_2026/15_monitoring.md).

## Key Features

- 🌩️ **Cloud-native**: AWS Batch/Fargate for elastic, serverless phase picking
- 🔬 **Selectable weights**: ships SeisBench models; custom fine-tunes drop in via `--weight`
- 📊 **Scalable database**: DocumentDB for catalog storage and querying
- ⏱️ **Timing-tuned picker**: the v7 fine-tune improves P-MAE to 0.340 s from its
  parent's 0.374 s, trading recall to get there — see
  [the model notes](docs/phasenet_v7_model_description.md) before choosing it
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
│   ├── models/v3/
│   │   ├── phasenet/                 # PhaseNet weights (quakescope2026 = v7)
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
