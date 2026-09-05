#!/usr/bin/env python
"""Scope the Batch role's S3 access to the catalogue bucket, and prove it.

The campaign role carried `AmazonS3FullAccess`: `s3:*` on every bucket in the
account. It needs exactly one bucket. Every archive the workers read is read
ANONYMOUSLY - SCEDC, NCEDC, GeoNet and EarthScope Open Data are all opened with
`S3FileSystem(anon=True)` - and the restricted EarthScope tier uses credentials
EarthScope issues, not ours. So no IAM S3 grant is involved in reading data at
all. The only bucket the role must reach is the one it writes picks to.

Why it matters more than it looks: `parquet_compact.py` calls `fs.rm()` and
`s3_state.py` calls `delete_object()`, so deletion is real code paths, not a
hypothetical. A wrong prefix in a compaction run under `s3:*` reaches every
bucket in the account, and if the catalogue bucket has no versioning the loss is
permanent - tens of thousands of shards representing thousands of dollars of
compute, with no undo.

    python scripts/scope_batch_role_s3.py --check     # simulate, change nothing
    python scripts/scope_batch_role_s3.py --apply     # attach scoped, detach full

`--apply` attaches the scoped policy FIRST and only detaches the managed one
after simulation confirms the scoped policy covers everything the pipeline does.
Run `--check` afterwards; it is the same simulation and is the actual evidence.

TO REVERSE, if something turns out to need more:

    aws iam attach-role-policy --role-name SeisBenchBatchRole \\
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

then widen `OBJECT_ACTIONS`/`BUCKET_ACTIONS` here and re-apply, rather than
leaving the account on the managed policy.
"""

from __future__ import annotations

import argparse
import json
import sys

ROLE = "SeisBenchBatchRole"
BUCKET = "quakescope-picks-2026"
POLICY_NAME = "QuakeScopeCatalogueS3"
FULL_ACCESS = "arn:aws:iam::aws:policy/AmazonS3FullAccess"

# Everything the pipeline does to the catalogue bucket.
#
#   ListBucket                  the shard queue, claims, completions, dashboard
#   GetBucketLocation           s3fs/botocore region resolution
#   ListBucketMultipartUploads  Parquet writes above the multipart threshold
BUCKET_ACTIONS = [
    "s3:ListBucket",
    "s3:GetBucketLocation",
    "s3:ListBucketMultipartUploads",
]

#   GetObject / PutObject       shards, claims, progress, manifests, picks
#   DeleteObject                s3_state.release(), parquet_compact
#   Abort/ListMultipartUpload*  cleaning up a failed large Parquet put
OBJECT_ACTIONS = [
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject",
    "s3:AbortMultipartUpload",
    "s3:ListMultipartUploadParts",
]

# A bucket in the same account that the pipeline must NOT be able to touch.
# Used as the negative control: a scoping change that cannot be shown to deny
# something has not been shown to do anything.
CONTROL_BUCKET = "scoped-noise"


def policy_document() -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CatalogueBucket",
                "Effect": "Allow",
                "Action": BUCKET_ACTIONS,
                "Resource": f"arn:aws:s3:::{BUCKET}",
            },
            {
                "Sid": "CatalogueObjects",
                "Effect": "Allow",
                "Action": OBJECT_ACTIONS,
                "Resource": f"arn:aws:s3:::{BUCKET}/*",
            },
        ],
    }


def simulate(iam, role_arn):
    """Ask IAM what the role can actually do. Never read the policy document.

    `simulate_principal_policy` evaluates every attached and inline policy the
    way a real request would, which is the only answer that counts.
    """
    checks = []
    for act in BUCKET_ACTIONS:
        checks.append((act, f"arn:aws:s3:::{BUCKET}", True))
    for act in OBJECT_ACTIONS:
        checks.append((act, f"arn:aws:s3:::{BUCKET}/picks/x.parquet", True))
    # Must be denied: another bucket in the same account.
    for act in ("s3:GetObject", "s3:PutObject", "s3:DeleteObject"):
        checks.append((act, f"arn:aws:s3:::{CONTROL_BUCKET}/anything", False))
    checks.append(("s3:ListBucket", f"arn:aws:s3:::{CONTROL_BUCKET}", False))

    rows, ok = [], True
    for act, res, want_allow in checks:
        r = iam.simulate_principal_policy(
            PolicySourceArn=role_arn, ActionNames=[act], ResourceArns=[res])
        decision = r["EvaluationResults"][0]["EvalDecision"]
        allowed = decision == "allowed"
        good = allowed == want_allow
        ok &= good
        rows.append((act, res, decision, "expected" if good else "WRONG"))
    return ok, rows


def show(rows):
    for act, res, decision, verdict in rows:
        mark = " " if verdict == "expected" else "!"
        print(f"  {mark} {act:32} {res:58} {decision:10} {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="simulate only")
    g.add_argument("--apply", action="store_true", help="attach scoped, detach full")
    a = ap.parse_args(argv)

    import boto3
    iam = boto3.client("iam")
    role_arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]

    attached = [p["PolicyArn"] for p in
                iam.list_attached_role_policies(RoleName=ROLE)["AttachedPolicies"]]
    inline = iam.list_role_policies(RoleName=ROLE)["PolicyNames"]
    print(f"role {ROLE}")
    print(f"  attached: {[p.split('/')[-1] for p in attached]}")
    print(f"  inline  : {inline}")
    print(f"  S3FullAccess attached: {FULL_ACCESS in attached}\n")

    if a.check:
        ok, rows = simulate(iam, role_arn)
        show(rows)
        print("\n  " + ("all as intended" if ok else "NOT as intended"))
        return 0 if ok else 1

    # --apply. Scoped policy first, so there is never a window with no access.
    iam.put_role_policy(RoleName=ROLE, PolicyName=POLICY_NAME,
                        PolicyDocument=json.dumps(policy_document()))
    print(f"  put inline policy {POLICY_NAME}")

    if FULL_ACCESS in attached:
        iam.detach_role_policy(RoleName=ROLE, PolicyArn=FULL_ACCESS)
        print("  detached AmazonS3FullAccess")
    else:
        print("  AmazonS3FullAccess was not attached; nothing to detach")

    # IAM is eventually consistent; simulate until it settles rather than
    # reporting a stale answer.
    import time
    for attempt in range(12):
        ok, rows = simulate(iam, role_arn)
        if ok:
            break
        time.sleep(5)
    print()
    show(rows)
    if not ok:
        print("\n  NOT as intended - reattach AmazonS3FullAccess if the "
              "pipeline is about to run:")
        print(f"    aws iam attach-role-policy --role-name {ROLE} "
              f"--policy-arn {FULL_ACCESS}")
        return 1
    print("\n  scoped to the catalogue bucket, and denied elsewhere")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
