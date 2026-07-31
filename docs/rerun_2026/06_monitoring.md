# 06 — Monitoring the campaign

Three things to watch: **jobs** (Batch), **data** (DocumentDB), **money**
(Cost Explorer). A 10-minute morning routine covers all three.

## 1. Jobs — AWS Batch console

Console → **Batch** → **Jobs**, pick your queue. The status columns:

| Status | Meaning | Worry? |
|---|---|---|
| SUBMITTED / PENDING | Accepted, waiting | No |
| RUNNABLE | Waiting for compute capacity | Only if stuck > ~30 min with idle capacity — see guide 07 |
| STARTING | Pulling image | No |
| RUNNING | Working | No |
| SUCCEEDED | Done | 🎉 |
| FAILED | Exhausted its 10 retries | Investigate |

Useful habits:

- Filter by FAILED and look at the **Status reason** on a few jobs.
  "Your Spot Task was interrupted" attempts are normal (that's the Spot
  bargain; they auto-retry); a job only lands in FAILED after 10 attempts.
- Quick counts from anywhere:

  ```bash
  for s in RUNNABLE RUNNING SUCCEEDED FAILED; do
    echo -n "$s: "; aws batch list-jobs --job-queue quakescope2026_queue \
      --job-status $s --query 'length(jobSummaryList)' --output text
  done
  ```

## 2. Logs — CloudWatch

Every job streams its Python log to CloudWatch. Two ways in:

- From a job's page in the Batch console, click the **Log stream name**.
- Console → **CloudWatch** → **Log groups** → `/aws/batch/job`.

What a healthy log looks like: alternating `Load NET.STA ...`,
`Pick NET.STA ...`, `Put NET.STA ... > N phase picks` lines. Common
non-fatal chatter: "S3 might be busy, sleep 5 seconds", "credential
renewed", "Skip ... picks found" (the resume logic working).

## 3. Data — DocumentDB

On the controller, notebook
[4_check_database.ipynb](../../notebooks/4_check_database.ipynb) (point it at
`quakescope2026`). The key progress metric is `picks_record` — one document
per completed station-channel-day:

```python
db = SeisBenchDatabase(DOCDB_ENDPOINT_URI, "quakescope2026")
coll = db.database
print("picks:        ", coll["picks"].estimated_document_count())
print("classifies:   ", coll["classifies"].estimated_document_count())
print("station-days: ", coll["picks_record"].estimated_document_count())
print("runs:         ", coll["sb_runs"].estimated_document_count())

# progress by year
import pandas as pd
agg = coll["picks_record"].aggregate([{"$group": {"_id": "$yr", "n": {"$sum": 1}}}])
print(pd.DataFrame(agg).sort_values("_id"))
```

Completion check for a campaign block: (station-days recorded) vs
(stations × days submitted — from the CSV in `submissions/`). When a re-run
of the same submission command produces jobs that all exit quickly with
"Skip ... picks found", that block is done.

Also glance at Console → DocumentDB → your cluster → **Monitoring** tab:
CPU and connections. Hundreds of concurrent jobs each hold connections; if
CPU pins at 100% or connections hit the limit, lower the pace (smaller
`maxvCpus`) or scale the DB instance up a size for the campaign.

## 4. Money — Cost Explorer

Console → **Billing and Cost Management** → **Cost Explorer** → set
granularity *Daily*, group by *Service*. Expect:

- **Elastic Container Service / Fargate** — the jobs. Scales with usage;
  ballpark $0.10–0.12 per job-hour on Spot.
- **Amazon DocumentDB** — flat ~$7/day for a `db.r6g.large` while it's up.
- **EC2** — controller; cents/day if you stop it when idle.

Your budget alert (guide 01, step 5) is the safety net between check-ins.

## 5. Capping spend — what AWS can and cannot do

There is **no native "cap this Fargate/Batch job at $X" control** in AWS.
Fargate bills per vCPU-second and per GB-second with no per-task budget
parameter, and AWS Batch has no cost-aware scheduling. The bill is bounded
indirectly, with three layers that compose into a hard ceiling:

1. **Per-job cap = the job timeout.** Cost per job is `duration × rate`, and
   `attemptDurationSeconds` caps duration. With the standard picking shape
   (8 vCPU + 16 GB ≈ $0.395/hr on-demand, ~$0.12/hr on Spot), the 24 h
   timeout caps one attempt at ≈ **$2.90 on Spot** (≈ $9.50 if on-demand).
   Retries multiply this: 10 attempts is the worst case, though Spot
   interruptions rarely burn more than a fraction of the timeout.
2. **Per-hour cap = `maxvCpus` on the compute environment.** This is the burn
   -rate throttle: `maxvCpus: 256` means at most 256 vCPU running ≈
   **$3.90/hr on Spot** (≈ $13/hr on-demand) no matter how many jobs are
   queued. Scale this number to the spend rate you're comfortable with.
3. **Per-campaign cap = AWS Budgets with a budget *action*.** Beyond the
   alert-only budget from guide 01, a budget can *act* at a threshold:
   Console → Billing → Budgets → your budget → **Actions** → attach an IAM
   policy when e.g. 100% is reached. Attach a deny policy on
   `batch:SubmitJob` (blocks new submissions) — or on `ecs:RunTask` to stop
   new tasks starting. Actions can run automatically or require your
   approval. Caveat: budget data lags ~8–12 h, and already-running tasks
   are not killed — pair it with the emergency stop in guide 05.

Practical recipe: keep the 24 h timeout, set `maxvCpus` from your target
weekly spend (e.g. ≤ $10/day → 64 vCPUs on Spot), and add a budget action
denying `batch:SubmitJob` at your campaign ceiling.

## 6. End-of-campaign teardown

- [ ] Re-run each submission command once; confirm only "Skip" jobs.
- [ ] Batch: disable the job queue; set compute environment `maxvCpus` to 0
      (or leave — idle Batch objects cost nothing).
- [ ] DocumentDB: take a **manual snapshot** (Console → cluster → Actions →
      Take snapshot) named e.g. `quakescope2026-final`, then **stop** the
      cluster (or delete it after the snapshot if the campaign is fully
      done — snapshots cost ~$0.02/GB-month).
- [ ] EC2: stop the controller instance.
- [ ] Check Cost Explorer one week later to confirm spend dropped to ~$0.

Next: [07_troubleshooting.md](07_troubleshooting.md)
