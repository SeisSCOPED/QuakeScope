# PR #71 (VibeSys) evaluation harness

Run on 2026-10-26. The shell scripts keep the scratch paths of that run; the Python takes paths as arguments.
Results are in `docs/benchmark/results/vibesys_pr71/`, and the page is `reports/vibesys_pr71.html`.

| step | script | what it does |
|---|---|---|
| picks | `run_arm.py`, `run_all.sh`, `run_fp16.sh` | vanilla, PR fp32, PR bf16 and fp16 on the benchmark windows, 0.02 floor |
| fused-conv emulation | `emu.py` | oneDNN's fused conv numerics (fp32 accumulate and bias, output in the input dtype); imported by `run_arm.py` and `bench_worker.py` |
| scores | `score.py` | against vanilla and against analyst picks, with `benchmark_metrics.match_picks`; the tolerance table and per-pick errors are computed in `export_results.py` |
| amplitudes | `run_amps.py`, `run_amps.sh` | main's `AmplitudeExtractor()` against the PR's fused/slim extractor on the same picks |
| laptop timing | `bench_worker.py`, `bench_run.sh` | 4 workers x 2 threads, 42 windows at 100 Hz, alternating reps |
| Graviton timing | `ec2_run.sh`, `ec2_summary.py` | the same on c6g, c7g and c8g `.2xlarge`, run from EC2 user-data |
| arm64 C kernel | `veliir_arm64.c` | the PR's `veliir.c` with its x86-only CPU probe guarded, for the arm64 amplitude check |
| tables | `export_results.py` | writes every table the page reads |
| page | `../build_vibesys_report.py` | builds `reports/vibesys_pr71.html` from those tables |
