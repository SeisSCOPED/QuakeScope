# Scoping AWS permissions, and proving you did

For anyone — person or agent — who needs to narrow an IAM role on this account,
or who is about to widen one. Written after the campaign role was found holding
`AmazonS3FullAccess` when it needed exactly one bucket.

## The rule that does the work

**Simulate, do not read.** `iam:SimulatePrincipalPolicy` evaluates every
attached and inline policy the way a real request would. Reading a policy
document tells you what someone intended; simulation tells you what the role can
do. The two differ whenever there is more than one policy, a `Deny`, a
permission boundary, or an SCP.

**A scoping change that cannot be shown to deny something has not been shown to
do anything.** Always simulate a *negative control*: an action on a resource the
role must not reach. If your "after" run is all green and contains no denials,
you have proved nothing.

## What this account's campaign role actually needs

`SeisBenchBatchRole` is both the Batch job role and the ECS execution role.

| need | why |
|---|---|
| `quakescope-picks-2026` read/write/delete | the shard queue, claims, progress, manifests, picks |
| `secretsmanager:GetSecretValue` on one ARN | the EarthScope refresh token |
| `AmazonECSTaskExecutionRolePolicy` | pull the image, write logs |

**Nothing else.** In particular, no IAM grant is involved in reading any
archive:

- SCEDC, NCEDC, GeoNet and EarthScope **Open Data** are opened with
  `S3FileSystem(anon=True)` — anonymous, no credentials of ours.
- EarthScope's **restricted** tier uses credentials EarthScope issues, scoped by
  network and year. Our account contributes only the refresh token.

So "the workers read five archives" does not imply "the role needs five
buckets". It needs one, the one it writes to.

## Doing it

```bash
python scripts/scope_batch_role_s3.py --check    # simulate, change nothing
python scripts/scope_batch_role_s3.py --apply    # attach scoped, then detach full
```

`--apply` attaches the scoped inline policy **before** detaching the managed
one, so there is never a window where a running job loses access, and it
re-simulates afterwards with retries because IAM is eventually consistent.

To reverse:

```bash
aws iam attach-role-policy --role-name SeisBenchBatchRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

If something turns out to need more, widen `BUCKET_ACTIONS` / `OBJECT_ACTIONS`
in the script and re-apply, rather than leaving the account on the managed
policy. The script is the record of what is granted and why.

## What it looked like

Before — the role could delete another project's data:

```
s3:DeleteObject  arn:aws:s3:::scoped-noise/anything   allowed        WRONG
```

After:

```
s3:DeleteObject  arn:aws:s3:::quakescope-picks-2026/… allowed        expected
s3:DeleteObject  arn:aws:s3:::scoped-noise/anything   implicitDeny   expected
s3:DeleteBucket  arn:aws:s3:::quakescope-picks-2026   implicitDeny   expected
```

Nine buckets live in this account, four of them another person's SkyPilot file
mounts. `s3:*` reached all of them.

## Why this was worth doing

Not because of an attacker. Because deletion is real code:

- `parquet_compact.py` calls `fs.rm()`
- `s3_state.py` calls `delete_object()`

A wrong prefix in a compaction run under `s3:*` reaches every bucket in the
account. The catalogue is ~51 GB and 85,549 completed shards — thousands of
dollars of compute — and **bucket versioning is not enabled**, so a delete is
permanent. Least privilege here is not a compliance exercise; it is the
difference between losing one prefix and losing everything the account holds.

> **Done 2026-09-05:** versioning is enabled on `quakescope-picks-2026`, with
> lifecycle rules expiring noncurrent versions after 30 days, aborting
> incomplete multipart uploads after 7, and clearing expired delete markers.
> The scoped policy grants `s3:DeleteObject` but **not**
> `s3:DeleteObjectVersion`, so a worker can create a delete marker and cannot
> destroy history.

## The recipe, for a role this document does not cover

1. **List what the code actually touches.** Grep for bucket names, secret ARNs
   and service clients. Check whether reads are anonymous before assuming they
   need a grant — that single check removed four buckets here.
2. **Write the policy as a script, not a console click.** It becomes the
   record, it is reviewable in a diff, and the next person can re-run it.
3. **Simulate before.** Capture the "wrong" rows; they are the justification.
4. **Attach the narrow policy first, detach the broad one second.**
5. **Simulate after, with a negative control**, and retry — IAM is eventually
   consistent and an immediate check can report a stale answer.
6. **Then run something real.** Simulation is a model of the evaluator. A job
   that starts, reads the queue, writes an object and exits is the evidence
   that the model matched reality. A tiny queue is enough; it does not need to
   be a campaign.

Step 6 is the one most often skipped, and it is the one that catches the action
you forgot to list.

## The role split, and the direction that catches people

`SeisBenchBatchRole` used to be both the Batch **job** role and the ECS
**execution** role, so a container's own permissions were also the platform's.
They are now two roles:

| role | holds | so that |
|---|---|---|
| `SeisBenchExecutionRole` | ECS task execution + `GetSecretValue` on one ARN | pull the image, write logs, inject the EarthScope token |
| `SeisBenchBatchRole` | `QuakeScopeCatalogueS3` only | the container reaches one bucket and nothing else |

**On Fargate the EXECUTION role resolves `secrets:`, not the job role.** This is
the part that is easy to get backwards, and getting it backwards fails at task
start with `ResourceInitializationError: unable to pull secrets`, which reads
like a Secrets Manager problem rather than a role problem.

The consequence worth having: the campaign container is handed
`ES_OAUTH2__REFRESH_TOKEN` as an environment variable but **cannot read the
secret itself**, and cannot pull or push images.

Doing it in this order matters, because the middle state is the dangerous one:

1. Create the execution role and give it what the platform needs.
2. Register new job definition revisions pointing at it — **all** of them.
   Seven definitions here still named the old role; stripping first would have
   made every one of them fail to start.
3. Run something real and confirm the secret still arrives.
4. Only then remove the execution policy and the secret read from the job role.

Between 2 and 4 both roles work, so there is no window where a job cannot
start. Skipping 3 means discovering the mistake on a campaign.

## Not covered by IAM

- **CloudWatch retention is five days** on the only record of what the fleet
  did. That is a deliberate choice; the mitigation is to export what matters
  before it ages out — see
  [`scripts/export_incident_logs.py`](../scripts/export_incident_logs.py) and
  [`incident_2026_09_04/EVIDENCE.md`](incident_2026_09_04/EVIDENCE.md).
- **Rate limiting is client-side.** The fleet workflow counts token-endpoint
  requests every 15 minutes and zeroes every campaign above
  `ES_RATE_LIMIT_PER_MIN`. That is a circuit breaker, not a quota: it stops a
  runaway within a quarter of an hour rather than preventing one.
