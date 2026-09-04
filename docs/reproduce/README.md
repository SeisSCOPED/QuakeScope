# Reproducing the QuakeScope GitHub + AWS stack

Everything needed to rebuild this system in a fresh AWS account and a fresh
GitHub repository — for a person following along, or an agent working from the
machine-readable half.

**The authority is `infra/inventory.py`, not this page.** It reads the live
account and repository and prints what is actually deployed:

```bash
python infra/inventory.py            # human-readable
python infra/inventory.py --json     # for an agent, or to diff two accounts
```

A guide written from memory documents what somebody believed they built. Run
the inventory first, and treat any disagreement with this page as this page
being out of date.

---

## What the system is

Three moving parts, and the interesting design decisions are in how they meet.

```
 GitHub                             AWS
 ──────                             ───
 docker.yml ──build──▶ ghcr.io/seisscoped/quakescope:<sha>
                                      │
 fleet.yml ───────OIDC───────▶ Batch SubmitJob
   (holds N workers)                  │
                                      ▼
                            Fargate Spot workers
                            read  SCEDC / NCEDC / EarthScope / GeoNet
                            write s3://quakescope-picks-2026/<campaign>/picks/
                                      │
 campaign-status.yml ◀──OIDC──────────┘
   (renders the dashboard, commits it, dispatches pages.yml)
                                      ▲
 EventBridge Scheduler ▶ Lambda ▶ GitHub API (workflow_dispatch)
   because GitHub's own cron is best-effort
```

**Work is a durable S3 queue, not a Batch array.** A campaign is a
`shards.jsonl` plus `claims/`, `complete/` and `progress/` prefixes. Workers
claim shards with a conditional write and heartbeat a lease. This is what makes
a 99%-reclaim Spot pool survivable: a killed worker loses at most one shard's
uncommitted tail, and the shard returns to the queue when its lease expires.

**Nothing keeps state in a runner or a laptop.** Both were tried and both are
recorded below under things that do not work.

---

## Prerequisites

| | |
|---|---|
| AWS account with Batch, S3, IAM, Lambda, EventBridge Scheduler, Secrets Manager | |
| GitHub repository, public or private | Pages enabled if you want the dashboard published |
| `gh` CLI authenticated | `gh auth status` |
| **boto3, not the `aws` CLI, for Batch** | see the traps section — this one cost a day |
| A VPC with subnets that can reach S3 and the public internet | Fargate tasks pull images from ghcr.io |

---

## 1. AWS — storage and compute

### The bucket

```bash
aws s3api create-bucket --bucket <BUCKET> --region us-east-2 \
  --create-bucket-configuration LocationConstraint=us-east-2
```

Layout, which the code assumes:

```
<campaign>/shards.jsonl              the queue, immutable once written
<campaign>/stations.parquet          station metadata
<campaign>/claims/<shard>.json       who holds what, with a lease
<campaign>/complete/<shard>.json     done
<campaign>/progress/<shard>.json     checkpoints inside a shard
<campaign>/manifests/<shard>.json    what each shard wrote
<campaign>/runs/<run_id>.json        model, weight, thresholds — provenance
<campaign>/picks/network=<NET>/year=<YYYY>/month=<MM>/*.parquet
<campaign>/.dashboard/rowcount.json  the dashboard's incremental cache
```

### Public read, scoped

The catalogue is meant to be readable without an account. Grant the data, not
the operational state:

```json
{"Version": "2012-10-17", "Statement": [
  {"Sid": "PublicListForPartitionDiscovery", "Effect": "Allow",
   "Principal": "*", "Action": "s3:ListBucket",
   "Resource": "arn:aws:s3:::<BUCKET>"},
  {"Sid": "PublicReadPicksAndProvenance", "Effect": "Allow",
   "Principal": "*", "Action": "s3:GetObject",
   "Resource": ["arn:aws:s3:::<BUCKET>/*/picks/*",
                "arn:aws:s3:::<BUCKET>/*/manifests/*",
                "arn:aws:s3:::<BUCKET>/*/runs/*",
                "arn:aws:s3:::<BUCKET>/*/stations.parquet"]}]}
```

`claims/` and `progress/` stay private — they carry worker hostnames and are of
no use to a reader. `.dashboard/` stays private, and must not live under
`picks/` or the Parquet dataset scan will try to parse it.

`BlockPublicPolicy` and `RestrictPublicBuckets` must be **off** for a policy
naming `Principal: "*"` to be accepted. Leave both ACL blocks **on**: the grant
is by policy, and `BucketOwnerEnforced` disables ACLs anyway.

### Compute environment and queue

`FARGATE_SPOT`, `maxvCpus` sized to the quota you actually hold — ours is
12,000, which is 1,500 tasks at 8 vCPU. That number is the ceiling for the whole
fleet across all campaigns, and the fleet governor reads it live rather than
having it hardcoded.

### Job definitions

One per campaign, differing only in weight and memory. Register them with
**boto3**. The fields that matter:

| field | value | why |
|---|---|---|
| `platformCapabilities` | `["FARGATE"]` | |
| `resourceRequirements` | 8 vCPU / 16384 MB | 32 GB was tried and is unnecessary — see traps |
| `executionRoleArn` | the Batch role | Fargate injects `secrets` with the **execution** role |
| `jobRoleArn` | the Batch role | what the container itself uses |
| `secrets` | `ES_OAUTH2__REFRESH_TOKEN` → Secrets Manager ARN | stores the ARN, never the value |
| `retryStrategy.attempts` | 10 | Batch's hard maximum; it cannot be raised |
| `retryStrategy.evaluateOnExit` | retry on `"Your Spot Task was interrupted."`, exit otherwise | |

`evaluateOnExit` only fires on a **FAILED** attempt, so a preempted worker must
exit non-zero or Batch records a success and never replaces it. The worker exits
75 for that reason.

---

## 2. AWS — identity

Six roles. Least privilege is not decoration here: one of these can submit
unlimited compute.

| role | trusted by | may |
|---|---|---|
| `SeisBenchBatchRole` | `ecs-tasks.amazonaws.com` | read the archives, write picks, read the EarthScope secret |
| `QuakeScopeAWSWatch` | GitHub OIDC | describe Batch, read the bucket, **submit to one queue** |
| `QuakeScopeDispatchLambda` | `lambda.amazonaws.com` | read **one** secret, write its own logs |
| `QuakeScopeSchedulerInvoke` | `scheduler.amazonaws.com` | invoke **one** function |
| `QuakeScopeGovernor` | `ec2.amazonaws.com` | unused — kept only as the record of an approach that failed |
| `EC2SSMProbe` | `ec2.amazonaws.com` | SSM, for in-region diagnostics |

### GitHub OIDC trust

```json
{"Version": "2012-10-17", "Statement": [{
  "Effect": "Allow",
  "Principal": {"Federated": "arn:aws:iam::<ACCT>:oidc-provider/token.actions.githubusercontent.com"},
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": {
    "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
    "StringLike": {"token.actions.githubusercontent.com:sub": "repo:<OWNER>/<REPO>:*"}}}]}
```

Set `MaxSessionDuration` to 21600. The default 3600 is shorter than a dashboard
render over a large catalogue, and the failure is a confusing mid-run 400.

> ⚠️ `SeisBenchBatchRole` currently carries `AmazonS3FullAccess`, which is wider
> than it needs and is both the job and execution role. Known, not fixed. If you
> are building this fresh, scope it to the four archive buckets and the picks
> bucket instead.

---

## 3. GitHub

### Repository variable, not a secret

```bash
gh variable set AWS_WATCH_ROLE_ARN --body "arn:aws:iam::<ACCT>:role/QuakeScopeAWSWatch"
```

A role ARN is not confidential and a variable is visible in logs, which is what
you want when debugging OIDC. **Referencing it as `secrets.*` resolves to an
empty string, `configure-aws-credentials` skips assuming anything without
complaining, and the run fails several steps later on `NoCredentialsError`** —
which reads like a permissions problem and is a typo.

### Workflows

| file | trigger | does |
|---|---|---|
| `docker.yml` | push to `main` | builds and pushes `ghcr.io/<owner>/<repo>:<short-sha>` |
| `fleet.yml` | schedule + `workflow_dispatch` | holds each campaign at its target; the phone control surface |
| `campaign-status.yml` | schedule + dispatch | renders the dashboard, commits it, dispatches `pages.yml` |
| `pages.yml` | push to `reports/**` + dispatch | publishes |
| `aws-watch.yml` | schedule | spend and activity report |

`fleet.json` at the repository root holds the per-campaign targets. `fleet.yml`
edits and commits it, so that file is the audit trail of who scaled what.

---

## 4. The glue: a schedule AWS actually keeps

**GitHub Actions cron is best-effort, and on this repository it is far off.**
Measured 2026-09-02/04:

| asked for | actually fired |
|---|---|
| `23 * * * *` (hourly) | every **2.4–5.5 hours** |
| `*/15 * * * *` | every **~108 minutes** |

That is fine for a nightly job and useless for watching a running campaign, and
no cron expression fixes it. So the schedule lives in AWS:

```
EventBridge Scheduler ─▶ Lambda quakescope-dispatch ─▶ POST /actions/workflows/<file>/dispatches
   rate(1 hour)   -> campaign-status.yml
   rate(20 minutes) -> fleet.yml
```

`FlexibleTimeWindow: OFF` — fire on time, not "within an hour".

The token is a **fine-grained PAT** with `Actions: read and write` on that one
repository, nothing else, held in Secrets Manager and read at invocation. It is
never logged; the Lambda prints the workflow name and HTTP status only.

```bash
aws secretsmanager put-secret-value --region us-east-2 \
  --secret-id quakescope/github-dispatch-token --secret-string 'github_pat_...'
```

AWS→GitHub needs a token because GitHub's OIDC only works in the other
direction. There is no way around it; keep the scope minimal instead.

---

## 5. Verify

```bash
python infra/inventory.py                       # everything, from the account
aws batch describe-job-definitions --status ACTIVE   # ← DO NOT. see traps.
```

Then, in order:

1. `gh workflow run fleet.yml -f campaign=<c> -f target=1` — one worker end to end
2. Watch its CloudWatch log reach `Put <station> ... > N phase picks`
3. `python infra/inventory.py --json | jq .schedules` — both `ENABLED`
4. Confirm a `schedule`-triggered run appears within its interval

---

## Traps

Every one of these cost at least half a day.

**The `aws` CLI silently drops modern Batch fields.** A 2020-vintage CLI has a
service model predating `secrets`, `executionRoleArn`, `platformCapabilities`
and `evaluateOnExit`, and omits them from `describe-job-definitions` output with
no warning. An audit through it reported the secret missing when it was present,
and a day went into "fixing" wiring that was correct. **Audit and register job
definitions with boto3.**

**A successful workflow is not a delivered result.** The hourly dashboard
rebuilt correctly for a day while the published page never changed: the commit
carried `[skip ci]`, which GitHub applies to *every* workflow on that push,
including the one that deploys. Nothing was red. Check the artifact, not the
job status.

**A count that failed is not a count of zero.** The dashboard rendered `0` picks
for a campaign holding 11.7 GB of them, because the credential expired mid-count
and the failure path returned 0. A number in a table outweighs any caption
underneath it.

**One queue, several campaigns: count only your own.** The fleet governor
counted every job in the shared queue, so each campaign saw the others' workers,
concluded it was at target, and submitted nothing — reporting `deficit 0` while
doing nothing. Filter by job-name prefix.

**EarthScope credentials must be scoped.** An unscoped credential for
`s3-miniseed-v2` carries `s3:ListBucket` but not `s3:GetObject`: every listing
succeeds and every read returns `AccessDenied`. Pass `network=FDSN:<NET>`, plus
`year=` for temporary networks (codes starting with a digit or X/Y/Z). This read
as a missing grant of access for two weeks because listing never failed.

**404 and 403 from that endpoint mean opposite things.** 404 is a network-year
the archive does not hold — a correction to your plan, and it must not trigger a
retry or a scope change. 403 is a real access gap — a request to EarthScope.
The SDK raises its own `UnauthenticatedError`/`UnauthorizedError` for 401/403
*instead of* an `HTTPStatusError`, so a status-code check never sees them and
they get retried as if they were congestion.

**`aws s3api` truncates keys at `#`.** EarthScope's restricted objects carry a
`#N` version suffix, so the CLI asks for a key that does not exist and gets 404
— which points at "the data is missing" rather than "the request is malformed".
Use boto3.

**RSS ratchets upward and the OOM comes late.** Workers OOM'd after 55 minutes
and 477 station-days, not on the first big shard. glibc keeps freed blocks on
per-arena free lists and RSS never falls. `gc.collect()` + `malloc_trim(0)`
between shards and `ENV MALLOC_ARENA_MAX=2` in the image fixed it; 32 GB was not
needed. `MALLOC_ARENA_MAX` must be `ENV` — glibc reads it once at process start,
so exporting it later is a silent no-op.

**Verify a container tag exists before submitting.** `gh run list --limit 1`
right after a push returns the *previous* commit's build. Select the run by SHA,
and confirm the tag is in the registry, or Batch fails with
`CannotPullContainerError` on every job.

**Do not use the campaign's refresh token from a laptop.** `earthscope_sdk`'s
refresh grant writes the rotated token to *local* SDK state, not back to Secrets
Manager, so one local run can invalidate the credential every job depends on.

### Approaches that did not work

- **A governor loop in the agent session.** Ends when the session does.
- **A governor on a small EC2 box.** One more thing to keep alive, fails
  silently, and cannot be scaled from a phone. Its user-data died on
  `pip install --upgrade pip`, which fails on AL2023 because pip is rpm-managed.
- **Relying on GitHub cron for anything time-sensitive.** See above.

---

## For an agent

`infra/inventory.py --json` is the contract. To reproduce into a new account:

1. `python infra/inventory.py --json > reference.json` against a working account
2. Create resources until a fresh `--json` matches on: `compute_environments[].maxvCpus`,
   `job_definitions[].{vcpu,memory,secrets,retryStrategy}`,
   `bucket_config.policy`, `iam_roles[].{inline,managed,maxSessionDuration}`,
   `schedules[].{expression,state}`, `github.variables`
3. Ignore: account ids, ARNs, image SHAs, revision numbers, subnet ids

The inventory is read-only and creates nothing, so it is safe to run repeatedly
while converging.

**Do not infer state from a job's exit status.** Several failures in this
project's history were green workflows producing nothing. Read the artifact the
step was supposed to produce — the published page, the object in S3, the tag in
the registry.
