"""Fire a GitHub Actions workflow on a schedule AWS actually keeps.

GitHub's own cron is best-effort and says so: runs "may be delayed during
periods of high load". Measured on this repository over 2026-09-02/04, an
hourly schedule fired every 2.4-5.5 hours and a */15 schedule every ~108
minutes. That is fine for a nightly job and useless for watching a campaign.

EventBridge Scheduler fires on time. So the schedule lives in AWS and GitHub is
told what to run, rather than GitHub being asked to remember.

The token is a fine-grained PAT with Actions: read and write on this repository
only, kept in Secrets Manager and read at invocation. It is never logged: the
only things printed are the workflow name and the HTTP status.
"""

import json
import os
import urllib.error
import urllib.request

import boto3

REPO = os.environ.get("REPO", "SeisSCOPED/QuakeScope")
SECRET = os.environ.get("TOKEN_SECRET", "quakescope/github-dispatch-token")
REF = os.environ.get("REF", "main")

_sm = boto3.client("secretsmanager")
_token = None


def _get_token():
    # Cached across warm invocations: this runs every few minutes and the
    # secret does not change between them.
    global _token
    if _token is None:
        _token = _sm.get_secret_value(SecretId=SECRET)["SecretString"].strip()
    return _token


def handler(event, context):
    workflow = (event or {}).get("workflow", "campaign-status.yml")
    inputs = (event or {}).get("inputs") or {}

    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{workflow}/dispatches"
    body = {"ref": REF}
    if inputs:
        body["inputs"] = inputs

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={
            "Authorization": f"Bearer {_get_token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "quakescope-dispatch",
        })
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            # 204 No Content is success for this endpoint.
            print(f"dispatched {workflow} -> HTTP {r.status}")
            return {"workflow": workflow, "status": r.status}
    except urllib.error.HTTPError as e:
        # Print the body: GitHub explains refusals here, and without it a 403
        # is indistinguishable from a 404 on a private repo.
        detail = e.read().decode("utf-8", "replace")[:400]
        print(f"dispatch {workflow} FAILED HTTP {e.code}: {detail}")
        raise
