#!/usr/bin/env python
"""Dump every AWS and GitHub resource this campaign actually depends on.

    python infra/inventory.py            # human-readable
    python infra/inventory.py --json     # machine-readable, for an agent

Written because a reproduction guide assembled from memory documents what
somebody believed they built. This reads the account and the repository, so what
it prints is what exists - including the parts nobody wrote down.

Read-only. It creates and changes nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import boto3

REGION = "us-east-2"
BUCKET = "quakescope-picks-2026"
REPO = "SeisSCOPED/QuakeScope"


def _gh(path, jq=None):
    cmd = ["gh", "api", path]
    if jq:
        cmd += ["-q", jq]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def collect() -> dict:
    out: dict = {"region": REGION, "bucket": BUCKET, "repo": REPO}

    b = boto3.client("batch", region_name=REGION)
    out["compute_environments"] = [
        {"name": c["computeEnvironmentName"], "type": c["computeResources"]["type"],
         "maxvCpus": c["computeResources"]["maxvCpus"], "state": c["state"],
         "subnets": c["computeResources"].get("subnets", []),
         "securityGroupIds": c["computeResources"].get("securityGroupIds", [])}
        for c in b.describe_compute_environments()["computeEnvironments"]
        if "niyiyu" in c["computeEnvironmentName"] or "quakescope" in c["computeEnvironmentName"].lower()
    ]
    out["job_queues"] = [
        {"name": q["jobQueueName"], "state": q["state"], "priority": q["priority"],
         "computeEnvironments": [o["computeEnvironment"].rsplit("/", 1)[-1]
                                 for o in q["computeEnvironmentOrder"]]}
        for q in b.describe_job_queues()["jobQueues"]
    ]
    defs, tok = [], None
    while True:
        kw = {"status": "ACTIVE", "maxResults": 100}
        if tok:
            kw["nextToken"] = tok
        r = b.describe_job_definitions(**kw)
        defs += r["jobDefinitions"]
        tok = r.get("nextToken")
        if not tok:
            break
    latest = {}
    for d in defs:
        n = d["jobDefinitionName"]
        if not n.startswith("quakescope_2026"):
            continue
        if n not in latest or d["revision"] > latest[n]["revision"]:
            latest[n] = d
    out["job_definitions"] = []
    for n, d in sorted(latest.items()):
        cp = d["containerProperties"]
        rr = {x["type"]: x["value"] for x in cp.get("resourceRequirements", [])}
        out["job_definitions"].append({
            "name": n, "revision": d["revision"], "image": cp.get("image"),
            "vcpu": rr.get("VCPU"), "memory": rr.get("MEMORY"),
            "jobRoleArn": cp.get("jobRoleArn"),
            "executionRoleArn": cp.get("executionRoleArn"),
            "secrets": [s["name"] for s in cp.get("secrets", [])],
            "platformCapabilities": d.get("platformCapabilities"),
            "retryStrategy": d.get("retryStrategy"),
        })

    s3 = boto3.client("s3", region_name=REGION)
    pab = s3.get_public_access_block(Bucket=BUCKET)["PublicAccessBlockConfiguration"]
    try:
        pol = json.loads(s3.get_bucket_policy(Bucket=BUCKET)["Policy"])
    except Exception:
        pol = None
    out["bucket_config"] = {"public_access_block": pab, "policy": pol}

    iam = boto3.client("iam")
    out["iam_roles"] = []
    for name in ("SeisBenchBatchRole", "QuakeScopeAWSWatch", "QuakeScopeGovernor",
                 "QuakeScopeDispatchLambda", "QuakeScopeSchedulerInvoke", "EC2SSMProbe"):
        try:
            r = iam.get_role(RoleName=name)["Role"]
        except Exception:
            continue
        out["iam_roles"].append({
            "name": name,
            "maxSessionDuration": r.get("MaxSessionDuration"),
            "trust": r["AssumeRolePolicyDocument"],
            "inline": iam.list_role_policies(RoleName=name)["PolicyNames"],
            "managed": [p["PolicyName"] for p in
                        iam.list_attached_role_policies(RoleName=name)["AttachedPolicies"]],
        })

    sm = boto3.client("secretsmanager", region_name=REGION)
    out["secrets"] = [
        {"name": s["Name"], "description": s.get("Description", "")}
        for s in sm.list_secrets()["SecretList"]
        if "quakescope" in s["Name"].lower()
    ]  # names and descriptions only - never values

    try:
        sch = boto3.client("scheduler", region_name=REGION)
        out["schedules"] = [
            {"name": s["Name"], "state": s["State"],
             "expression": sch.get_schedule(Name=s["Name"])["ScheduleExpression"],
             "target": sch.get_schedule(Name=s["Name"])["Target"]["Arn"].rsplit(":", 1)[-1]}
            for s in sch.list_schedules()["Schedules"]
            if "quakescope" in s["Name"]
        ]
    except Exception as e:
        out["schedules"] = f"unavailable: {type(e).__name__}"

    lam = boto3.client("lambda", region_name=REGION)
    out["lambdas"] = [
        {"name": f["FunctionName"], "runtime": f["Runtime"],
         "role": f["Role"].rsplit("/", 1)[-1], "timeout": f["Timeout"],
         "env": list((f.get("Environment") or {}).get("Variables", {}))}
        for f in lam.list_functions()["Functions"]
        if "quakescope" in f["FunctionName"]
    ]

    out["github"] = {
        "workflows": [w for w in (_gh(f"repos/{REPO}/actions/workflows",
                                      ".workflows[].path") or "").split("\n") if w],
        "variables": [v for v in (_gh(f"repos/{REPO}/actions/variables",
                                      ".variables[].name") or "").split("\n") if v],
        "secrets": [s for s in (_gh(f"repos/{REPO}/actions/secrets",
                                    ".secrets[].name") or "").split("\n") if s],
        "pages": _gh(f"repos/{REPO}/pages", ".html_url"),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    inv = collect()
    if a.json:
        print(json.dumps(inv, indent=2, default=str))
        return 0
    print(f"=== QuakeScope stack, read live from {inv['region']} and {inv['repo']} ===\n")
    for ce in inv["compute_environments"]:
        print(f"compute env   {ce['name']}  {ce['type']}  maxvCpus={ce['maxvCpus']:,}  {ce['state']}")
    for q in inv["job_queues"]:
        print(f"job queue     {q['name']}  {q['state']}  -> {q['computeEnvironments']}")
    print()
    for d in inv["job_definitions"]:
        print(f"job def       {d['name']}:{d['revision']}  {str(d['image']).rsplit(':', 1)[-1]}  "
              f"{d['vcpu']}vCPU/{d['memory']}MB  secrets={d['secrets'] or '-'}")
    print()
    pab = inv["bucket_config"]["public_access_block"]
    print(f"bucket        s3://{inv['bucket']}  "
          f"BlockPublicPolicy={pab['BlockPublicPolicy']}  "
          f"policy={'yes' if inv['bucket_config']['policy'] else 'none'}")
    print()
    for r in inv["iam_roles"]:
        print(f"iam role      {r['name']}  session={r.get('maxSessionDuration')}s  "
              f"inline={r['inline']}  managed={r['managed']}")
    print()
    for s in inv["secrets"]:
        print(f"secret        {s['name']}")
    for f in inv["lambdas"]:
        print(f"lambda        {f['name']}  {f['runtime']}  role={f['role']}")
    for s in (inv["schedules"] if isinstance(inv["schedules"], list) else []):
        print(f"schedule      {s['name']}  {s['expression']}  {s['state']} -> {s['target']}")
    print()
    g = inv["github"]
    print(f"workflows     {g['workflows']}")
    print(f"vars/secrets  vars={g['variables']}  secrets={g['secrets']}")
    print(f"pages         {g['pages']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
