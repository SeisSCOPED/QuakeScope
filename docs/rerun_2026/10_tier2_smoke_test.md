# 10 — Tier-2 smoke test: one EC2 job, or a small Fargate cluster

Tier 1 is the notebooks in [`tutorials/`](../../tutorials/): they check the
models on a laptop, against analyst picks, with no AWS involved. Tier 2 asks a
different question — **does the deployed container reproduce that same result?**

Passing means the image built correctly, the weights are inside it, S3 reads
work from the VPC, DocumentDB accepts writes, and the classifier path runs. It
is the last gate before launching a campaign that costs real money.

Budget an hour. Nothing here needs more than three stations and one day.

---

## What is being checked

| | Tier 1 (laptop) | Tier 2 (cloud) |
|---|---|---|
| Runs | `pixi run -e tutorials smoke-test` | the real container entrypoint |
| Data | direct S3 over the internet | S3 from inside the VPC |
| Output | plots and tables in a notebook | rows in DocumentDB |
| Catches | bad weights, bad picks | bad image, bad IAM, bad networking, bad writes |

The pairing matters: tier 1 establishes what the answer *should* be, so tier 2
can compare against it rather than merely checking that something ran.

## Before you start

- [ ] Steps [02](02_weights_and_container.md) through [04](04_batch_setup.md)
      complete — image pushed, DocumentDB up, Batch queue defined.
- [ ] Tier 1 passing locally, so a tier-2 failure points at infrastructure
      rather than at the models.
- [ ] A throwaway database name. Use `quakescope_smoke`, never the campaign
      database — this run will be re-executed and its rows are disposable.

## The test job

Three SCEDC stations, one day, the day of the Ridgecrest mainshock. Chosen
because it is dense enough that a silent no-op is obvious: a working job writes
tens of thousands of picks, so "it ran and wrote nothing" cannot pass unnoticed.

```
stations : CI.CLC.., CI.TOW2.., CI.SRT..
day      : 2019.187  (start 2019.187, end 2019.188)
weight   : quakescope2026
model    : PhaseNet
```

Reference values, measured locally with the same weights at the picker defaults
(`--p_threshold 0.2 --s_threshold 0.2`):

| Station | P | S | Total |
|---|--:|--:|--:|
| CI.CLC. | 4697 | 4097 | 8794 |
| CI.TOW2. | 3577 | 2810 | 6387 |
| CI.SRT. | 3860 | 2708 | 6568 |
| **All** | **12134** | **9615** | **21749** |

Roughly 65 s of inference per station-day on one CPU, plus about 10 s to fetch
each channel. A three-station job should finish in a few minutes.

These same numbers live in `REFERENCE` in
[`sb_catalog/src/verify_smoke_test.py`](../../sb_catalog/src/verify_smoke_test.py);
regenerate them if the weights or thresholds change.

---

## Option A — a single EC2 job

Best for the first attempt. You get a shell, so a failure is diagnosable
instead of just a red status in the console.

1. **Launch a controller instance** if you do not already have one from
   [03_documentdb.md](03_documentdb.md) — `t3.large` is enough, in the **same
   VPC and security group as the DocumentDB cluster**.

2. **Pull the image** you pushed in [02](02_weights_and_container.md):

   ```bash
   docker pull ghcr.io/seisscoped/quakescope:latest
   ```

3. **Confirm the weights are actually baked in.** This catches the most common
   packaging mistake before you spend time on a job:

   ```bash
   docker run --rm ghcr.io/seisscoped/quakescope:latest \
       python -c "import seisbench.models as sbm; print(sbm.PhaseNet.list_pretrained())"
   ```

   `quakescope2026` must appear. If it does not, the `COPY models/phasenet`
   layer did not take — rebuild rather than continuing.

4. **Run the picking job:**

   ```bash
   docker run --rm \
     -e EARTHSCOPE_S3_ACCESS_POINT="$EARTHSCOPE_S3_ACCESS_POINT" \
     ghcr.io/seisscoped/quakescope:latest \
     pick \
       --db_uri "$DB_URI" \
       --database quakescope_smoke \
       --stations CI.CLC..,CI.TOW2..,CI.SRT.. \
       --start 2019.187 --end 2019.188 \
       --model PhaseNet --weight quakescope2026 \
       --p_threshold 0.2 --s_threshold 0.2 \
       --classifier
   ```

   Watch for `Put CI.CLC..HH 2019.187 > NNNN phase picks` in the log. That line
   is the job confirming it wrote.

5. **Verify what landed:**

   ```bash
   docker run --rm ghcr.io/seisscoped/quakescope:latest \
     python -m src.verify_smoke_test \
       --db_uri "$DB_URI" --database quakescope_smoke \
       --stations CI.CLC.,CI.TOW2.,CI.SRT. \
       --start 2019.187 --end 2019.188
   ```

   Every check must pass. Exit status is 0 or 1, so this can gate a script.

> Station strings differ between the two commands: `--stations` for picking
> takes `NET.STA.LOC.CHA` without the component, while the verifier matches the
> `tid` field as written to the database. Copy them as printed above.

## Option B — a small Fargate cluster

Run this once Option A passes. It exercises Batch, Fargate Spot, task IAM roles,
and VPC networking — none of which Option A touches.

1. **Submit through the helper**, which is the same path a campaign uses:

   ```python
   from src.submit_helper import SubmitHelper
   from src.utils import SeisBenchDatabase
   import datetime

   db = SeisBenchDatabase(DB_URI, "quakescope_smoke")
   helper = SubmitHelper(
       start=datetime.date(2019, 7, 6),
       end=datetime.date(2019, 7, 7),
       extent=None,
       network="CI",
       db=db,
       region="us-east-2",
       station_ids=["CI.CLC..", "CI.TOW2..", "CI.SRT.."],
       station_group_size=3,      # one job, so a failure is unambiguous
       day_group_size=1,
       model="PhaseNet",
       weight="quakescope2026",
   )
   helper.submit_jobs("pick")
   ```

   `station_group_size=3` deliberately produces a single job. Splitting across
   jobs at this stage only makes it harder to tell which one failed.

2. **Watch it** with the queries in [06_monitoring.md](06_monitoring.md). A
   Fargate Spot task can be reclaimed mid-run; a reclaimed task is not a test
   failure, it is a retry.

3. **Verify** exactly as in Option A, step 5.

4. **Then scale to a handful of jobs** — raise `station_group_size` to 40 and
   add a second day — and confirm the picks-per-station-day stay in the same
   range. This is where per-job concurrency problems appear, and they do not
   show up in a single-job run.

---

## When it fails

| Symptom | Where to look |
|---|---|
| `quakescope2026` missing from `list_pretrained()` | the `COPY models/phasenet` layer; rebuild and re-push |
| `InvalidVersion: '1.partial'` | an interrupted download left `*.partial` in the SeisBench cache; `rm ~/.seisbench/models/v3/phasenet/*.partial` |
| `RuntimeError: Attempting to deserialize object on a CUDA device` | a checkpoint saved from a GPU; the image installs CPU-only PyTorch, so re-save the weights with CPU storage |
| No picks written, no error | almost always S3 or database reachability; test each from inside the container before blaming the models |
| Picks written but counts far off | compare `--weight`, `--p_threshold`, `--s_threshold` against the reference above; a threshold difference moves counts a long way |
| Classifier rows absent | `--classifier` omitted, or the channel is not `BH`/`HH` — the classifier only runs on those |
| Timeouts on EarthScope data | token expired; see [05_submitting_jobs.md](05_submitting_jobs.md). NCEDC and SCEDC are anonymous and unaffected |

## Clean up

```python
from pymongo import MongoClient
MongoClient(DB_URI).drop_database("quakescope_smoke")
```

Dropping it keeps re-runs honest — a partially populated database can make a
broken job look like it passed.

## Then what

With tier 2 green, the campaign path is [05_submitting_jobs.md](05_submitting_jobs.md)
for a full submission and [09_western_states_run.md](09_western_states_run.md)
for the scaled run. Keep `quakescope_smoke` in your pocket: any time the image
or the weights change, this is the hour that tells you whether the change was
safe.
