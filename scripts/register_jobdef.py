#!/usr/bin/env python
"""Register a job-definition revision that differs from the last one in the image only.

    # what would change, changing nothing
    python scripts/register_jobdef.py --tag 1a2b3c4 quakescope_2026_western --dry-run

    # register, verify field by field, and repoint every campaign in fleet.json
    # that was on this family's previous revision
    python scripts/register_jobdef.py --tag 1a2b3c4 quakescope_2026_western --update-fleet

WHY A SCRIPT. Repointing a campaign at a new build is a metadata call - no
compute, no image build - and it has been done by hand four times (751f206,
c4faeef, ...), each time as "clone the previous revision field by field,
change the image, read it back, diff". Doing that by hand is how revisions
drift: 751f206 found all five campaign families on an image 55 commits old
with the thread environment unset, and none of it shows in the console.

THREE THINGS THIS INSISTS ON.

1. boto3, never the `aws` CLI. The local CLI is aws-cli/2.0.34, whose service
   model predates `platformCapabilities`, `secrets`, `executionRoleArn` and
   `networkConfiguration`; it drops them silently on read AND on write, which is
   how quakescope_v3_worker:5 was registered unable to run on Fargate at all.

2. The tag must exist in ghcr before anything is registered. docker.yml builds
   on push to main only, tagged with the short sha, so a commit on a branch has
   no image and a revision pointing at it would fail every worker at pull.
   Checked with an anonymous manifest HEAD: 200 exists, 404 does not.

3. After registering, the new revision is read back and diffed against the old
   one. If anything but the image differs the script says so and exits 1 - the
   revision is already registered by then, so it also prints the deregister
   call. New revisions are additive; the previous one stays ACTIVE and a
   submission can still name it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import boto3

REGION = "us-east-2"
REGISTRY_IMAGE = "ghcr.io/seisscoped/quakescope"
# Seconds. A registry that does not answer must fail the run, not hang it.
HTTP_TIMEOUT = 30
FLEET = os.path.join(os.path.dirname(__file__), "..", "fleet.json")

# Read-only keys describe_job_definitions returns that register_job_definition
# does not accept.
READ_ONLY = {"jobDefinitionArn", "revision", "status", "containerOrchestrationType"}
# Everything else that describe returns and register accepts, passed through
# untouched when present.
PASS_THROUGH = ("jobDefinitionName", "type", "parameters", "schedulingPriority",
                "containerProperties", "nodeProperties", "retryStrategy",
                "propagateTags", "timeout", "tags", "platformCapabilities",
                "eksProperties", "ecsProperties", "consumableResourceProperties")


def ghcr_status(tag: str) -> int:
    """HTTP status of the manifest for REGISTRY_IMAGE:tag - 200 if it exists."""
    repo = REGISTRY_IMAGE.split("/", 1)[1]
    with urllib.request.urlopen(
            f"https://ghcr.io/token?scope=repository:{repo}:pull",
            timeout=HTTP_TIMEOUT) as r:
        token = json.load(r)["token"]
    req = urllib.request.Request(
        f"https://ghcr.io/v2/{repo}/manifests/{tag}", method="HEAD",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": ", ".join([
                     "application/vnd.oci.image.index.v1+json",
                     "application/vnd.docker.distribution.manifest.list.v2+json",
                     "application/vnd.docker.distribution.manifest.v2+json",
                     "application/vnd.oci.image.manifest.v1+json"])})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def latest(batch, family: str) -> dict:
    revs = batch.describe_job_definitions(
        jobDefinitionName=family, status="ACTIVE")["jobDefinitions"]
    if not revs:
        sys.exit(f"no ACTIVE revision of {family}")
    return max(revs, key=lambda d: d["revision"])


def clone(d: dict, image: str) -> dict:
    if "containerProperties" not in d:
        # Every campaign definition is type "container"; a multi-node or EKS
        # definition keeps its image elsewhere (nodeProperties, eksProperties)
        # and this script does not know how to repoint one.
        sys.exit(f"{d.get('jobDefinitionName')}:{d.get('revision')} has no "
                 f"containerProperties (type {d.get('type')!r}); this script "
                 f"only repoints container job definitions")
    body = {k: d[k] for k in PASS_THROUGH if k in d}
    body["containerProperties"] = dict(body["containerProperties"], image=image)
    return body


def flat(d, prefix=""):
    """{'a.b[0].c': value} for a nested dict, so two revisions diff line by line."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def diff(old: dict, new: dict) -> dict:
    a = {k: v for k, v in flat(old).items() if k.split(".")[0] not in READ_ONLY}
    b = {k: v for k, v in flat(new).items() if k.split(".")[0] not in READ_ONLY}
    return {k: (a.get(k, "<absent>"), b.get(k, "<absent>"))
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def update_fleet(family: str, old_rev: int, new_rev: int) -> list[str]:
    with open(FLEET) as f:
        cfg = json.load(f)
    moved = []
    for name, c in cfg["campaigns"].items():
        if c.get("job_definition") == f"{family}:{old_rev}":
            c["job_definition"] = f"{family}:{new_rev}"
            moved.append(name)
    with open(FLEET, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return moved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("families", nargs="+",
                    help="job definition names, e.g. quakescope_2026_western")
    ap.add_argument("--tag", required=True,
                    help="short sha docker.yml tagged the image with")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the change and register nothing")
    ap.add_argument("--update-fleet", action="store_true",
                    help="repoint campaigns in fleet.json that were on the "
                         "previous revision")
    args = ap.parse_args(argv)

    image = f"{REGISTRY_IMAGE}:{args.tag}"
    status = ghcr_status(args.tag)
    if status != 200:
        print(f"{image} -> HTTP {status}. docker.yml builds on push to main "
              f"only; is that commit on main, and has the build finished?")
        return 1
    print(f"{image} -> HTTP 200")

    batch = boto3.client("batch", region_name=REGION)
    rc = 0
    for family in args.families:
        old = latest(batch, family)
        old_image = old["containerProperties"]["image"]
        print(f"\n{family}:{old['revision']}  {old_image}")
        if old_image == image:
            print("  already on this image; nothing to do")
            continue
        body = clone(old, image)
        if args.dry_run:
            print(f"  would register {family}:{old['revision'] + 1} on {image}, "
                  f"all other fields as :{old['revision']}")
            continue
        r = batch.register_job_definition(**body)
        new = batch.describe_job_definitions(
            jobDefinitions=[r["jobDefinitionArn"]])["jobDefinitions"][0]
        changed = diff(old, new)
        expected = {"containerProperties.image": (old_image, image)}
        if changed != expected:
            print(f"  REGISTERED {family}:{new['revision']} but it differs from "
                  f":{old['revision']} in more than the image:")
            for k, (a, b) in changed.items():
                print(f"    {k}: {a!r} -> {b!r}")
            print(f"  to withdraw it: batch.deregister_job_definition("
                  f"jobDefinition='{r['jobDefinitionArn']}')")
            rc = 1
            continue
        print(f"  registered {family}:{new['revision']} on {image}; "
              f"read back, differs from :{old['revision']} in the image only")
        if args.update_fleet:
            moved = update_fleet(family, old["revision"], new["revision"])
            print(f"  fleet.json: {', '.join(moved) or 'no campaign'} "
                  f"-> {family}:{new['revision']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
