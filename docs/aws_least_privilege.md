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

> **Still open:** enabling versioning on `quakescope-picks-2026`, with a
> lifecycle rule expiring noncurrent versions after ~30 days, would remove the
> "permanent" from that sentence. It is independent of IAM and cheap.

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

## Not covered by IAM

Two exposures on this account that scoping a role does not touch, recorded so
they are not mistaken for solved:

- **`AmazonECSTaskExecutionRolePolicy` and the job role are the same role.** A
  compromised container has the execution role's permissions too. Splitting
  them is the standard shape and has not been done here.
- **CloudWatch retention is five days** on the only record of what the fleet
  did. That is a deliberate choice; the mitigation is to export what matters
  before it ages out — see
  [`scripts/export_incident_logs.py`](../scripts/export_incident_logs.py) and
  [`incident_2026_09_04/EVIDENCE.md`](incident_2026_09_04/EVIDENCE.md).
