# 14 — Running a campaign on SkyPilot (v3)

v3 replaces AWS Batch + Fargate Spot + DocumentDB with **SkyPilot managed jobs
over an S3 work queue**. There is no job definition to register, no compute
environment, no database, and no VPC to launch inside. Submission runs from a
laptop.

## Why managed jobs and not a cluster

`sky launch` with `use_spot` gives you Spot instances but **no recovery** — when
AWS reclaims the node, the cluster stays down. Only `sky jobs launch` puts the
work under SkyPilot's jobs controller, which relaunches after a preemption.
Since the whole point here is to run on Spot, campaigns use managed jobs.

That makes preemption routine rather than exceptional, which is what the rest of
the design is built around.

## The three pieces

| Piece | What it does | Where |
|---|---|---|
| `shard_planner` | Groups stations and days into shards, writes `shards.jsonl` | run once per campaign |
| `worker` | Claims a shard, picks it, writes Parquet, writes a manifest | N loops × N nodes |
| `s3_state` | Stations, claims, manifests, provenance — all on S3 | the former database |

Grouping is unchanged from 2025: **40 stations × 20 days** per shard. At launch
scale that is 64,240 shards over 51.2M station-days.

## Running one

```bash
pixi install --environment deploy

# 1. Plan the queue (once; it is immutable afterwards)
pixi run -e deploy python -m sb_catalog.src.shard_planner \
    --campaign s3://quakescope-picks-2026/scedc \
    --stations s3://quakescope-picks-2026/stations.parquet \
    --network CI --start 2018.001 --end 2020.366

# 2. Launch the workers as a MANAGED job
pixi run -e deploy sky jobs launch -n qs-scedc skypilot/campaign.yaml \
    --env CAMPAIGN=s3://quakescope-picks-2026/scedc \
    --env WEIGHT=jma_wc

# 3. Watch
pixi run -e deploy sky jobs queue
pixi run -e deploy python -c "
from sb_catalog.src.s3_state import S3CampaignState
print(S3CampaignState('s3://quakescope-picks-2026/scedc').progress())"
```

Add `--dry-run` to the planner to see the shard count and station-day total
without writing the queue.

## How Spot preemption is survived

Three mechanisms, in order of who catches what:

1. **Atomic claims.** A worker takes a shard with an S3 conditional write
   (`IfNoneMatch: "*"`), which fails if the key exists. Two workers can never
   hold the same shard. Verified under 8 concurrent workers against 50 shards:
   50 claims, 0 double-claims.
2. **SIGTERM release.** Preemption gives about two minutes' notice. The worker
   releases its in-flight claim and exits, so that shard is available again
   immediately rather than after the lease.
3. **Lease expiry.** For a node killed outright with no warning, a claim older
   than `--lease-hours` (default 6) with no manifest is reclaimable. Set this
   above the longest expected shard runtime, or live shards will be stolen and
   done twice.

A manifest is written **only after** the shard's Parquet is durable, so a
manifest never exists for work whose picks were lost. Resume is therefore just:
skip every shard that has one.

Workers are stateless. Scale `num_nodes` up or down mid-campaign, relaunch after
a preemption, or run a second job against the same queue — nothing needs
cleaning up first.

## Terminating when the campaign is done

Managed jobs terminate their **worker** cluster when the job finishes. The
**jobs controller does not stop itself**, and `sky down --all` does not
terminate it either — it reports success and leaves an on-demand instance
running. Observed directly: after cancelling every job and running
`sky down --all`, an on-demand `m6i.xlarge` controller was still up at
$0.192/hour.

Fix it once, before launching, by copying
[`skypilot/sky-config.yaml`](../../skypilot/sky-config.yaml) to
`~/.sky/config.yaml`:

```yaml
jobs:
  controller:
    autostop:
      down: true            # terminate; the default only *stops* it
      idle_minutes: 15
      wait_for: jobs_and_ssh
```

Controller settings are read **only** from `~/.sky/config.yaml` — a task or
project YAML is ignored. `down: true` matters more than the timeout: the default
leaves a stopped instance that still bills its EBS volume.

Verify rather than trust it:

```bash
pixi run -e cloud watch      # queries EC2 directly
```

## Campaign layout on S3

```
s3://<bucket>/<campaign>/
    stations.parquet          station metadata          (was: stations)
    shards.jsonl              the work queue            (new)
    claims/<shard_id>.json    who holds what            (new)
    manifests/<shard_id>.json what finished             (was: picks_record)
    runs/<run_id>.json        provenance                (was: sb_runs)
    picks/network=/year=/month=/*.parquet
```

## The job runs in the container, not a fresh install

`resources.image_id` points at `ghcr.io/seisscoped/quakescope`, the image GitHub
Actions already builds on every push to `main`. It carries torch, SeisBench,
pyarrow and the model weights, so a node has nothing to install and `setup` is
just a readiness check.

This is not a small saving. `setup` re-runs **on every Spot recovery**, not only
at first launch, so anything expensive there is paid again at each preemption.
Measured on this campaign: a `pixi install` in `setup` took ~4m40s per node,
against a few seconds to pull the image. On a campaign sized for preemption that
is the difference between recovery being cheap and recovery being the dominant
cost.

**Pin the short-SHA tag for a real campaign.** `:latest` moves when someone
pushes to `main`, and a recovered node would then pull a different image than
the one the campaign started with:

```yaml
image_id: docker:ghcr.io/seisscoped/quakescope:a1b2c3d
```

The code is pinned with it, at `/code/src`, which is why the template has no
`workdir` and runs `python -m src.worker`. To test a branch before it is merged
and built, add a `workdir` block and run `python -m sb_catalog.src.worker` from
`~/sky_workdir` instead — the image still supplies the environment, so nothing
is installed either way.

## Sizing

`num_nodes × PROCS` is the parallelism. The default 4 nodes × 8 procs = 32
concurrent shards; a 32-vCPU instance leaves room for the picker's own threads
inside each loop. Raise `num_nodes` for throughput — the queue does not care.

Cost lever: the campaign is embarrassingly parallel and interruption-tolerant,
so Spot is close to free money here. The `any_of` block prefers Spot and falls
back to on-demand only when a region runs dry, which stops a campaign stalling
outright.

## What this replaces

| v2 (Batch) | v3 (SkyPilot) |
|---|---|
| `job_definition_picking.yaml`, re-registered per campaign | `skypilot/campaign.yaml`, no registration |
| `submit_helper.py` → `batch.submit_job` × 64,240 | `shard_planner` writes one queue file |
| `retryStrategy: attempts 10` | claims + lease + jobs-controller recovery |
| DocumentDB for stations/`picks_record`/`sb_runs` | S3 objects under the campaign prefix |
| EC2 controller inside the DocumentDB VPC | any machine with AWS credentials |

The Batch path still exists (`submit_helper.py`, the job definitions) for
reproducing the 2025 campaign, but it is not the supported route in v3 and the
DocumentDB collections it needs are no longer created by the picking path.

## Caveats

- **Association still needs a database.** `run_association` reads and writes
  `picks`, `events` and assignments through the Mongo interface, and has no S3
  equivalent yet. A v3 campaign is picking only — which matches the 2026 plan,
  where the classifier is also out.
- **Conditional writes are required.** The claim protocol needs an S3 endpoint
  that honours `IfNoneMatch`. AWS S3 does; some S3-compatible stores do not, and
  `s3_state` refuses to run rather than race silently.
- **The queue is immutable.** Completed work is keyed on shard id, so extending
  a campaign's date range means a new campaign prefix, not a rewritten
  `shards.jsonl`.
