# Archive

Kept, not deleted — these record decisions and measurements that are still worth
reading, but they describe a workflow that no longer runs, or they were wrong.
Nothing here should be followed as instructions.

Current state: [`../README.md`](../README.md). Open work:
[`../OPTIMISE.md`](../OPTIMISE.md).

## Superseded by v3

v3 dropped DocumentDB, claims work from an S3 queue instead of one Batch job per
unit, and writes Parquet. These describe the 2025 shape of the pipeline.

| document | why archived | replaced by |
|---|---|---|
| `03_documentdb.md` | v3 has no database — station metadata, resume state and provenance are S3 objects | `s3_state.py` module docstring |
| `05_submitting_jobs.md` | one-job-per-unit submission; v3 workers pull from a queue | `../README.md` |
| `06_monitoring.md` | pre-dates the hourly `aws-watch` workflow and the dashboard | `../15_monitoring.md` |
| `08_multi_picker_campaigns.md` | partitioning networks per weight | `../17_launch_conventions.md` (weight-per-campaign table) |
| `09_western_states_run.md` | specified `instance`; superseded by `original`, confirmed 2026-08-29 | `../11_launch_plan.md`, `../17_launch_conventions.md` |
| `10_tier2_smoke_test.md` | v2-era smoke test against DocumentDB | `../README.md` (smoke test section) |
| `12_output_storage.md` | the case for Parquet over DocumentDB — decision made, now history. **Its station-day estimates are superseded**; they were 1.3–1.7× low because they did not separate location codes | `../21_queues_written.md` |
| `13_parquet_workflow.md` | running campaigns against the Parquet path when it was still optional | `../README.md` |

`12_output_storage.md` is the one to be careful with: its station-day counts got
copied into later documents and were wrong. The verified counts, read back from
the written queues, are in `../21_queues_written.md`.

## Written 2026-08-31, retracted or superseded 2026-09-01

These came out of a planning session that produced a large amount of process
documentation on top of measurements that turned out to be invalid. They are
archived rather than deleted because the retraction notices are the useful part.

| document | why archived |
|---|---|
| `PHASE1_FINAL_REPORT.md` | **Retracted.** Its measurements came from an eleven-day-old container image, and its headline "30 s/band-day" divided a runtime by a shard size that was invented rather than looked up. Retraction notice retained. |
| `phase1_cost_estimate_final.md` | **Retracted.** Derived from the above, and additionally assumed EarthScope reads at SCEDC speed. |
| `phase1_cost_estimate.md` | Template of `_pending_` placeholders that was never filled in. |
| `23_2026_campaign_plan.md` | 650 lines of phase-gated process written before any of it had been run. Most of it never survived contact with the first real measurement. The parts that held are folded into `../README.md` and `../OPTIMISE.md`. |
| `SETUP_CLOUDWATCH_BUDGET.md` | Step-by-step guide to a service this account cannot use — CloudWatch Budgets is blocked, which is why it was written and then immediately worked around. |
| `MANUAL_COST_TRACKING.md` | The workaround for the above. Its weekly-tracking idea is sound and is kept in `../OPTIMISE.md`; the rest was scaffolding around numbers that were wrong. |
| `phase1c_billing_setup.md` | Same session, same scaffolding. |

The lesson from that session, recorded because it is the reason this archive
exists: **a document asserting a measurement is worth less than nothing if the
measurement was not taken.** Every number in `../README.md` names the run it
came from.
