# Cloud-native Machine Learning workflow for earthquake event detection and phase picking


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

## Running a campaign from your phone

The whole fleet is driven from one GitHub Actions workflow, so a phone is
enough. In the GitHub app:

**Repo → Actions → "▶ Run or stop a campaign (Fleet)" → Run workflow**, pick a
campaign and a number of workers.

That is the only button. Everything else is automatic:

| you pick | what happens |
|---|---|
| `dryrun3`, target 1 | a tiny test queue, one worker, a couple of minutes — do this first |
| any campaign, target 0 | stops launching new workers |
| a campaign for the first time | an **access survey** runs instead of the fleet; run it again afterwards |
| a campaign, target 50 | 50 workers, if the survey says the data is readable |
| ⏹ STOP | everything stops and is terminated, now |

**It refuses rather than wastes.** Before anything launches, the workflow
checks that the image is not the build that caused the 2026-09-04 incident,
that we are not already overloading EarthScope, and that we can actually read
the data this campaign plans to read. If any of those fail it explains why and
launches nothing. Read the run's summary — the reason is written in plain
English.

**To stop everything, now:** Actions → **"⏹ STOP — terminate running
workers"** → Run workflow. It defaults to `all`, sets every target to 0, and
terminates the workers that are already running.

Use it rather than setting a target to 0 if you want things to actually stop.
A target of 0 only stops *new* workers; ones already running keep going until
their queue empties or they hit the 24-hour job timeout. On 2026-09-05 that was
57 workers still burning for 50 minutes after every target read 0.

**One thing to remember.** The scheduled top-up holds a campaign at its
committed target every 15 minutes, so a campaign left at 50 stays at 50 until
somebody sets it to 0 or presses STOP.

**To check everything is healthy without launching anything:**
Actions → "✅ Preflight" → Run workflow. It answers "safe to run a campaign" or
tells you what is wrong, and it changes nothing.

Background, if a run refuses and you want to know why:
[the incident report](https://seisscoped.org/QuakeScope/earthscope_credential_audit.html).

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
- 🔬 **Selectable weights**: the image carries the four a campaign uses; any other
  SeisBench model downloads on first use, and custom fine-tunes drop in via
  `--weight` — see [updating the image's weights](#updating-the-images-weights)
- 📊 **Scalable database**: DocumentDB for catalog storage and querying
- ⏱️ **Timing-tuned picker**: the v7 fine-tune improves P-MAE to 0.340 s from its
  parent's 0.374 s, trading recall to get there — see
  [the model notes](docs/phasenet_v7_model_description.md) before choosing it
- 📦 **Containerized**: Docker-based deployment with custom weights and dependencies
- 🔍 **Monitored**: CloudWatch dashboards and SNS alerts for job tracking

## Model weights and the container image

The image carries **only** the weights a campaign actually selects. Everything
else SeisBench offers still works and downloads on first use — fine for
notebooks and ad-hoc runs, and it never happens on a campaign path.

| weight | used by | files |
|---|---|---|
| `jma_wc` | campaigns 1–3 (SCEDC, NCEDC, EarthScope) | `.pt.v1`, `.json.v1` |
| `obs` | campaign 4 (OBS) | `.pt.v1`, `.json.v1` |
| `original` | campaign 5 (western) | `.pt.v1/.v2`, `.json.v1/.v2` |
| `quakescope2026` | the v7 fine-tune; not in this run | `.pt.v1`, `.json.v1` |

They live in [`sb_catalog/models/v3/phasenet/`](sb_catalog/models/v3/phasenet/).

### A weight is a pair, and the `.json` is the architecture

Not metadata — `model_args` is what SeisBench builds the network from, so the
two files must travel together and must match:

```jsonc
// jma_wc.json.v1
"model_args": { "component_order": "ZNE", "phases": "PSN",
                "norm": "std", "filter_factor": 2 },   // 2x the filters
"seisbench_requirement": "0.9.0",
"default_args": { "P_threshold": 0.1, "S_threshold": 0.1, ... }
```

The four differ in ways that are not interchangeable:

| weight | `component_order` | `filter_factor` | needs SeisBench | params |
|---|---|---|---|---|
| `jma_wc` | `ZNE` | 2 | ≥ 0.9.0 | 1,070,899 |
| `quakescope2026` | `ZNE` | 2 | ≥ 0.9.0 | 1,070,899 |
| `original` | `ENZ` | — | ≥ 0.3.2 | 268,443 |
| `obs` | `Z12H` | — | ≥ 0.4.0 | 268,499 |

All four are the same `PhaseNet` class and the same 50-module topology.
`filter_factor: 2` widens it: every convolutional layer gets double the filters
(PhaseNetWC, [Naoi et al. 2024](https://doi.org/10.1186/s40623-024-02091-8)), so
`inc.weight` goes `(8,3,7)` → `(16,3,7)` and `down_branch.0.0.weight` goes
`(8,8,7)` → `(16,16,7)`. Doubling *both* channel dimensions quadruples each
conv, which is why 268,443 params becomes 1,070,899 — a factor of 3.99, short of
exactly 4 only because the input layer's `in_channels` is fixed by the component
count and biases scale linearly.

So the risk is not that they are different networks; it is that they are the
same network at different widths. Pairing a `.pt` with a `.json` of a different
`filter_factor` fails loudly on a state-dict shape mismatch — fine. A
`component_order` mismatch does not: `ZNE`, `ENZ` and `Z12H` are all valid, so
the model loads clean and picks on mis-ordered traces. That is the failure worth
guarding against, and why step 3 loads the model rather than just checking the
file is present.

`seisbench_requirement` is a real floor: `jma_wc` needs ≥ 0.9.0, and the
Dockerfile's `pip install seisbench` is unpinned.

`obs` wanting `Z12H` — Z, two horizontals, and a hydrophone — is worth a look
against the `--components` default of `ZNE12` before campaign 4 runs.

### Fetching and committing a weight

```bash
# 1. fetch into your local SeisBench cache from the official repository
#    (seisbench.remote_model_root -> hifis-storage.desy.de, Helmholtz-hosted)
pixi run -e cloud python -c \
  "import seisbench.models as sbm; sbm.PhaseNet.from_pretrained('<name>')"

# 2. commit EVERY version of the pair, not just .v1 - see below
cp ~/.seisbench/models/v3/phasenet/<name>.pt.v*   sb_catalog/models/v3/phasenet/
cp ~/.seisbench/models/v3/phasenet/<name>.json.v* sb_catalog/models/v3/phasenet/

# 3. verify it loads with the network blocked, from a cache holding only these
cd sb_catalog/models/v3/phasenet
rm -rf /tmp/sbtest && mkdir -p /tmp/sbtest/models/v3/phasenet
cp *.pt.v* *.json.v* /tmp/sbtest/models/v3/phasenet/
SEISBENCH_CACHE_ROOT=/tmp/sbtest pixi run -e cloud python -c "
import seisbench.models as sbm, socket
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(OSError('blocked'))
m = sbm.PhaseNet.from_pretrained('<name>')
print(sum(p.numel() for p in m.parameters()), 'params')"

# 4. commit and push - the Action rebuilds and tags the image with the short SHA
# 5. re-register the Batch job definition against that tag
```

**Step 3 is the one that matters.** Skip it and you find out on 1,500 workers.

**Copy every version, not just `.v1`.** Which one SeisBench resolves depends on
*its own* version and is not always the highest or the one you would guess:
seisbench 0.12.3 resolves `original` and `instance` to **`.v2`**, while `jma_wc`
and `obs` have only `.v1`. Committing `original.pt.v1` alone would put a file in
the image that SeisBench never asks for, leave the runtime download exactly
where it was, and look like it had been fixed.

**Step 5 is not optional.** A campaign runs whatever image the job definition
pins, and a stale pin is invisible until you read worker logs — that is how
`quakescope_v3_worker:2` spent eleven days pinning a pre-fix image.

### Why the Dockerfile is as small as it is

It does five things, and each one is load-bearing:

| step | why it cannot be dropped |
|---|---|
| `pip install torch --index-url .../cpu` | the CPU wheel; the default pulls CUDA and multiplies image size for no benefit on Fargate |
| `pip install seisbench s3fs boto3 pyarrow …` | `pyarrow` is the v3 Parquet output path — without it a job `ImportError`s before reading a byte. `pyocto` is deliberately absent: it is only used by `run_association`, which needs DocumentDB and is not part of a picking campaign |
| `wget global-bundle.pem` | RDS trust store for DocumentDB — 2025-path only, unused by v3 |
| `COPY src/` | the pipeline |
| `COPY models/v3/` | the four weights above |

It downloads **no weights**, so a build cannot vary with what upstream is
serving that day. It previously pulled a 156 MB tarball from a personal site
dated June 2023; that tarball predated `jma_wc`, so every worker fetched it at
startup instead — 1,500 cold-start requests to an external host in the critical
path. Details and the SHA-256 comparison that justified removing it are in
[`sb_catalog/models/v3/phasenet/README.md`](sb_catalog/models/v3/phasenet/README.md).

One asymmetry worth knowing: `picker.py`'s own `--weight` default is `instance`,
which is **not** in the image, so the bare legacy entry point downloads on first
run. The `work` subcommand campaigns use has its own parser defaulting to
`jma_wc` and is unaffected.

## Architecture

Two deployment paths are supported, and they differ in more than plumbing:

| | v2 — AWS Batch + DocumentDB | v3 — AWS Batch on Fargate Spot + S3 queue |
|---|---|---|
| Proven at | the 2025 petabyte campaign | two shards end to end |
| State | DocumentDB (VPC-bound) | S3 objects, no database |
| Submission from | an EC2 controller inside the VPC | anywhere |
| Notebooks | [3](notebooks/3_submit_job.ipynb), [4](notebooks/4_check_database.ipynb) | [5](notebooks/5_submit_job_parquet.ipynb), [6](notebooks/6_check_parquet.ipynb) |

Both write picks as Parquet on S3. The diagram below is the v2 path; for v3 see
[17_launch_conventions.md](docs/rerun_2026/17_launch_conventions.md), and
[16_skypilot_vs_fargate.md](docs/rerun_2026/16_skypilot_vs_fargate.md) for the
cost comparison.


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
