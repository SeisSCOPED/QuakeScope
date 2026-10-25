# QuakeScope 9× latency checkpoint

The optimized runner is [`pick_workload.py`](pick_workload.py). It uses the
round-32 implementation from commit `022c20b` (`vibesys/latency/r32`), integrated
with the newer channel fallback, station-day outcomes, and station-date handling
on `main`. The earlier 5.4× checkpoint is `551d68b` (`vibesys/latency/r09`).

## Recorded performance

These measurements are from the original round-32 snapshot, before integration
with `main`. Each optimized result is the median of three runs.

| Host | Vanilla median | Round 32 median | Speedup |
|---|---:|---:|---:|
| Local host, 8 physical cores | 124.0 s | 13.77 s | 9.0× |
| GCP `n4-highcpu-8`, 4 cores / 8 threads | 206.9 s | 22.72 s | 9.1× |

Original evaluator results: [local](results/round32-local.json) and
[GCP](results/round32-gcp.json). Their `reference_wall_seconds` fields are the
cached reference-build timings, which differ from the separately remeasured
vanilla medians above. Local measurements were rechecked on 2026-10-08; GCP
cross-evaluation was completed on 2026-10-25.

The workload is 8 stations × 3 days around Ridgecrest, PhaseNet `jma_wc`, with
8 vCPUs and a 16 GB memory limit. Both result files report recall **0.971071**
and precision **0.973362** against vanilla commit
`324e6bc5b8c865ce0e97cc91deedcebb47d05fc2`. Matching requires the same
station/band/phase, peaks within 0.5 seconds, and both amplitudes within 5%.
The campaign gate requires recall and precision of at least 0.90; it does
not require bitwise identical picks. Peak memory was 5.52 GB locally and
5.43 GB on GCP.

## Run from a clone

Use an environment with the QuakeScope runtime dependencies (the measurements
used SeisBench 0.12.5 and CPU PyTorch 2.14.0). From the repository root:

```bash
python vibesys/pick_workload.py \
  --workload vibesys/workload.json \
  --output-dir /tmp/quakescope-picks \
  --vcpus 8
```

The runner reads public SCEDC waveforms and FDSN inventory over the network,
stages the repository's model weights locally, and writes Parquet picks.
For timing comparisons, give it the same eight CPU cores and use the VibeSys
evaluator to enforce CPU/memory limits and compare output against the reference.

The optimized settings are selected by this runner; existing production CLI
defaults remain available. Its main optimizations are BF16 PhaseNet inference,
fused window processing, cached and parallel amplitude work, a compiled velocity
filter, overlapped I/O and imports, and worker thread coordination.

BF16 is used only when AMX-BF16 and the PyTorch backend support it. Otherwise,
inference falls back to FP32. `QS_AMX=off` exercises that fallback. A C compiler
enables the velocity kernel; a failed build falls back to SciPy. Import prefetch
is particularly useful on NFS; `QS_IMPORT_PREFETCH=off` disables it. Performance
depends on CPU, storage, and network conditions; 9× is the measured workload
result, not a guarantee for every deployment.

## Integration verification, 2026-10-25

The [final integrated runner check](results/integration-check.json) processed
all 24 station-days in **16.07 s**, with recall **0.971071**, precision
**0.973362**, and 5.50 GB peak memory. This is one validation run, not a new
three-run speedup claim. An earlier integration check took 21.04 s with the
same accuracy.

Reader regression tests cover channel fallback during serial and concurrent
reads, missing network-years, and station-day outcomes. Runner tests cover
both checkout layouts and the non-AMX precision fallback. The offline
EarthScope self-test passes.

The full test suite has two outstanding failures in this environment, both
also reproduced on unchanged `main` at `7dd1be7`:

- `test_memory_headroom.py::test_peak_notices_an_allocation_the_floor_forgets`
- `test_score_picks_cli.py::test_cli_numbers_are_the_module_numbers`

The original baseline run had 202 passing tests and these two failures.
The integration's other tests pass, including the added reader and runner
regressions. These existing failures are outside the latency integration.
