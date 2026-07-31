# 05 — Submitting jobs: smoke test first, then the three archives

All submission happens **on the EC2 controller** (it needs database access to
list stations). The tool is `src.submit_helper`; notebook
[3_submit_job.ipynb](../../notebooks/3_submit_job.ipynb) wraps it.

## 1. How work is split (refresher)

`submit_helper` takes a date range and a set of networks, looks up stations in
the database, and cuts the work into Batch jobs of
**40 stations × 20 days** each (`station_group_size` × `day_group_size`).
Example: 400 NCEDC stations × 1 year ≈ 10 station-groups × 19 day-groups ≈
**190 jobs**. Each job is one Fargate Spot container; the queue drains at
whatever rate `maxvCpus` allows. Every submission also writes a CSV log to
`submissions/` so you know exactly what was launched.

Which archive a station comes from is automatic: `src/constants.py` maps each
network code to `ncedc`, `scedc`, or `earthscope`, and the reader picks the
right S3 layout. So "running on the 3 archives" = submitting three groups of
networks.

## 2. EarthScope credentials (only for the EarthScope archive)

EarthScope's S3 requires a user token; NCEDC/SCEDC are anonymous.

1. On the controller: `~/miniconda/bin/pip install earthscope-cli`
2. `es login` — it prints a URL; open it on your laptop, log in with your
   EarthScope (ex-IRIS) account, approve.
3. Find the refresh token in the SDK's credential store
   (`~/.earthscope/default/tokens.json` — field `refresh_token`) and paste it
   into `parameters.py` → `ES_OAUTH2__REFRESH_TOKEN`.
4. `EARTHSCOPE_S3_ACCESS_POINT`: the S3 access-point alias EarthScope
   assigned to you (from the last campaign's notes, or ask EarthScope data
   services — it looks like `s3://es-miniseed-<...>-s3alias` without the
   `s3://`). It is account-specific and should not change between campaigns.

The submitter injects both into every job's environment automatically.

## 3. Smoke test (do not skip)

One tiny job per archive, using the new weight name (here `quakescope2026`):

```bash
cd ~/QuakeScope/notebooks
PYTHONPATH=../sb_catalog ~/miniconda/bin/python -m src.submit_helper \
    pick 2023.001 2023.003 \
    --network BK --database quakescope2026 --weight quakescope2026
```

Repeat with `--network CI` (SCEDC) and `--network UW` (EarthScope). Then:

- Watch it in the console: **Batch → Jobs** → select your queue. The job
  walks through SUBMITTED → RUNNABLE → STARTING → RUNNING. (Stuck in
  RUNNABLE > 15 min ⇒ see troubleshooting guide.)
- Click the job → **Log stream name** to see its live log in CloudWatch.
  You should see `Load NET.STA... @ ncedc`, then `Put ... > N phase picks`.
- On the controller, check the database (notebook 4 pattern):

  ```python
  from src.utils import SeisBenchDatabase
  from src.parameters import DOCDB_ENDPOINT_URI
  db = SeisBenchDatabase(DOCDB_ENDPOINT_URI, "quakescope2026")
  print(db.database["picks"].estimated_document_count())
  print(list(db.database["sb_runs"].find()))   # must show weight: quakescope2026
  ```

**Gate: do not scale up until `sb_runs` shows the new weight name and picks
are arriving for all three archives.**

## 4. The real campaigns

Submit **one year at a time per archive**, like last time — it keeps each
batch reviewable and failures contained. From the controller (a `screen` or
`tmux` session is wise):

```bash
# NCEDC year 2023 (all NCEDC networks: BG,BK,BP,NC,PG,UL,WR)
PYTHONPATH=../sb_catalog ~/miniconda/bin/python -m src.submit_helper \
    pick 2023.001 2024.001 \
    --network BG,BK,BP,NC,PG,UL,WR \
    --database quakescope2026 --weight quakescope2026
```

```bash
# SCEDC year 2023 (CI)
PYTHONPATH=../sb_catalog ~/miniconda/bin/python -m src.submit_helper \
    pick 2023.001 2024.001 \
    --network CI \
    --database quakescope2026 --weight quakescope2026
```

For EarthScope, network list = everything mapped to `earthscope` in
`constants.py`; generate it the way notebook 3 does:

```python
import glob, sys; sys.path.append("../sb_catalog")
from src.constants import NETWORK_MAPPING
net = [f.split("/")[-1].split(".")[0] for f in sorted(glob.glob("../networks/*.zip"))]
print(",".join([n for n in net if NETWORK_MAPPING[n] == "earthscope"]))
```

then pass that comma-separated list to `--network`.

Practical rhythm that worked before:

1. Submit one year for one archive.
2. Wait for the queue to mostly drain (hours to ~2 days depending on
   `maxvCpus`); check monitoring (guide 06) each morning.
3. Re-run the *same* submission command once — thanks to `picks_record`,
   already-done station-days are skipped, so this cheaply sweeps up jobs
   killed by Spot interruptions or transient failures.
4. Move to the next year / archive. Multiple archives can run concurrently
   if the queue has headroom; they share the same `maxvCpus` pool.

## 5. Several pickers for different station sets?

If different weights apply to different networks (e.g. an OBS picker, a
general picker, and a California picker), it's the same machinery — one
submission command per weight with disjoint `--network` lists. Rules and
examples: [08_multi_picker_campaigns.md](08_multi_picker_campaigns.md).

## 6. If you need to stop everything

Console → Batch → Job queues → select queue → you can **disable** the queue
(stops new jobs starting), and Jobs → select → **Terminate**. Bulk from CLI:

```bash
for s in SUBMITTED PENDING RUNNABLE STARTING RUNNING; do
  for j in $(aws batch list-jobs --job-queue quakescope2026_queue --job-status $s --query 'jobSummaryList[].jobId' --output text); do
    aws batch terminate-job --job-id $j --reason "manual stop"
  done
done
```

Next: [06_monitoring.md](06_monitoring.md)
