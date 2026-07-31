# 04 — AWS Batch: roles, compute environment, queue, job definition

AWS Batch is the job scheduler. Four objects, created in this order:

| Object | What it is | Config file |
|---|---|---|
| IAM roles | Permissions the containers run with | (console) |
| Compute environment | The pool of Fargate Spot capacity | `sb_catalog/configs/compute_environment.yaml` |
| Job queue | Where submitted jobs wait | `sb_catalog/configs/job_queue.yaml` |
| Job definition | The template: image, vCPU/RAM, command | `sb_catalog/configs/job_definition_picking.yaml` |

These may all still exist from the last campaign (they cost nothing while
idle). Check first: Console → **Batch** (region us-east-2) → left menu has
*Compute environments*, *Job queues*, *Job definitions*. If they're there and
`ENABLED`/`VALID`, you only need step 4 (a new job-definition revision).
Notebook [1_prepare_compute_env.ipynb](../../notebooks/1_prepare_compute_env.ipynb)
is the executable version of this page.

## 1. IAM roles (only if missing)

Console → **IAM** → **Roles**. You need two (they can be the same role, and
last time they were — e.g. `QuakeScopeBatchRole`):

- **Execution role** — lets ECS pull the image and write logs. Create role →
  Trusted entity: *AWS service* → Use case: *Elastic Container Service* →
  *Elastic Container Service Task* → attach policy
  `AmazonECSTaskExecutionRolePolicy`.
- **Job role** — what the container itself may do. Same creation path; add an
  inline policy allowing Batch `Describe*`/read (as in notebook 1). S3 reads
  of NCEDC/SCEDC are anonymous and EarthScope uses its own token, so no S3
  permissions are needed.

Note both ARNs: `arn:aws:iam::<ACCOUNT_ID>:role/<name>`.

## 2. Compute environment (only if missing)

Get your networking IDs (or read them off any console page):

```bash
aws ec2 describe-subnets | jq '.Subnets[].SubnetId'
aws ec2 describe-security-groups --filters "Name=group-name,Values=default" | jq '.SecurityGroups[0].GroupId'
```

Edit `sb_catalog/configs/compute_environment.yaml`:

- `computeEnvironmentName`: e.g. `quakescope2026_env`
- `subnets`: **all** subnet IDs from above
- `securityGroupIds`: the security group **that the DocumentDB cluster
  allows on port 27017** (see guide 03)
- `maxvCpus`: this is your **throughput knob**. Each picking job uses 8
  vCPU, so `maxvCpus: 256` = 32 jobs running at once; `2048` = 256 jobs.
  Also check Console → **Service Quotas** → AWS Fargate → *Fargate Spot vCPU*
  quota for the account, and request an increase if it's below your target.

```bash
aws batch create-compute-environment --no-cli-pager --cli-input-yaml file://sb_catalog/configs/compute_environment.yaml
```

## 3. Job queue (only if missing)

Edit `sb_catalog/configs/job_queue.yaml`: pick a `jobQueueName`
(e.g. `quakescope2026_queue`) and paste the compute environment ARN
returned by the previous command. Then:

```bash
aws batch create-job-queue --no-cli-pager --cli-input-yaml file://sb_catalog/configs/job_queue.yaml
```

## 4. Picking job definition — REQUIRED this time

The job definition **must be re-registered for this campaign** even if one
exists, because (a) the command now passes `--model`/`--weight` through, and
(b) you want to pin the new image.

Edit `sb_catalog/configs/job_definition_picking.yaml`:

- `jobDefinitionName`: e.g. `quakescope2026_picking`
- `image`: pin the SHA tag from the container build, e.g.
  `ghcr.io/seisscoped/quakescope:a1b2c3d` (avoid `:latest` for the campaign —
  a later push to main would change what new jobs run).
- `jobRoleArn` / `executionRoleArn`: the ARNs from step 1.
- Leave `parameters:` defaults; the submitter overrides `model`/`weight`.
- Resources: 8 vCPU / 16 GB and 24 h timeout worked at scale last time;
  keep them.

```bash
aws batch register-job-definition --no-cli-pager --cli-input-yaml file://sb_catalog/configs/job_definition_picking.yaml
```

(Repeat with `job_definition_association.yaml` → e.g.
`quakescope2026_association` if/when you run association.)

Registering the same name again just creates a new *revision*; jobs use the
latest revision by default.

## 5. Record the names in `parameters.py`

```python
JOB_QUEUE = "quakescope2026_queue"
JOB_DEFINITION_PICKING = "quakescope2026_picking"
JOB_DEFINITION_ASSOCIATION = "quakescope2026_association"
```

Do this in the checkout **on the EC2 controller** — that's where submission
runs.

Next: [05_submitting_jobs.md](05_submitting_jobs.md)
