# 07 — Troubleshooting

Symptoms ordered by how likely you are to meet them.

## Jobs stuck in RUNNABLE

- **Compute environment INVALID** (Batch → Compute environments → status
  column): usually a deleted/wrong role ARN or subnet. Read the status
  reason, fix the YAML, recreate.
- **maxvCpus exhausted**: everything is fine; jobs start as others finish.
- **Fargate Spot capacity/quota**: check Service Quotas → Fargate. Request
  an increase, or temporarily switch the compute environment type to
  `FARGATE` (on-demand, ~3× cost) if a deadline looms.

## Job fails immediately (< 2 min)

Open the job's CloudWatch log:

- `CannotPullContainerError` — image tag typo in the job definition, or the
  GHCR package is private (GitHub → org SeisSCOPED → Packages → quakescope →
  Package settings → visibility Public).
- Python traceback `ServerSelectionTimeoutError` (pymongo) — the job can't
  reach DocumentDB: wrong `db_uri`, DB stopped, or the job's security group
  isn't allowed on port 27017 of the DB's security group.
- `KeyError` on `EARTHSCOPE_S3_ACCESS_POINT` — the env var wasn't injected;
  it must be set (even to a placeholder) in `parameters.py` before
  submission, for all archives.
- Immediate `argparse` error — job definition command and `picker.py` args
  out of sync; re-register the job definition from the current YAML.

## Job fails after running a while

- Attempts show *"Your Spot Task was interrupted"* — normal; only worry if a
  job burns all 10 attempts, which happens during regional Spot droughts.
  Re-submit later; `picks_record` makes re-runs cheap.
- `MemoryError` / container OOM (exit code 137) — an unusually dense
  station-day. Note it, keep going; the 200 MB file-size guard already skips
  the worst offenders.
- Log shows repeated *"EarthScope credential client might be busy"* forever —
  refresh token expired or was revoked. Redo `es login`, update
  `parameters.py`, resubmit; running jobs on NCEDC/SCEDC are unaffected.

## No picks in the database, jobs SUCCEEDED

- Wrong database name at submission (check `sb_runs` in *all* databases:
  `db.list_database_names()`).
- All station-days were skipped as already picked — you submitted into a
  database that already has this block (see "Skip ... picks found" in logs).
  That's the resume logic; it's only a problem if you *meant* to re-pick, in
  which case use a fresh database name.
- Stations not in metadata: notebook 2 wasn't run against this database.

## Picks look wrong / too few

- Confirm `sb_runs.weight` is your new weight, not `instance` — if it says
  `instance`, the `--weight` flag didn't reach the job (job definition not
  re-registered with the `Ref::weight` command).
- Thresholds: `--p_threshold/--s_threshold` default 0.2; a differently
  calibrated model may need different values. They can be added to the job
  definition command the same way model/weight were.

## Console shows nothing anywhere

Wrong region. Top-right selector → **us-east-2 (Ohio)**.

## DocumentDB connection refused from your laptop

Expected. DocumentDB is VPC-only; use the EC2 controller (guide 03).

## Emergency stop

Guide 05 §5: disable queue + terminate jobs loop. Cost stops accruing within
minutes of jobs terminating.
