# Evidence guide — EarthScope credential incident, 2026-09-04

For EarthScope and for anyone reviewing the AWS side. It says where the raw
evidence is, what each part of it shows, and how to re-derive every number the
[public report](https://seisscoped.org/QuakeScope/earthscope_credential_audit.html)
quotes, without taking our word for any of it.

## Where it is

```
s3://quakescope-picks-2026/incident/2026-09-04-earthscope/
    manifest.json              provenance, window definitions, sha256 per file
    ANALYSIS.txt               every published figure, re-derived
    raw/<window>.jsonl.gz      every event, one JSON object per line
    summary/<window>.json      counts, recomputed per window
```

**This prefix is private.** Public read on that bucket is scoped to
`*/picks/*`, `*/manifests/*`, `*/runs/*` and `*/stations.parquet`; everything
here returns 403 to an anonymous request. That is deliberate: raw worker logs
are the wrong thing to publish even after redaction. Ask and we will grant
access, or send the file.

It exists because CloudWatch keeps `/aws/batch/job` for **five days**. The
incident window would have been unrecoverable after 2026-09-09.

## The four windows

| window | span (UTC) | what it shows |
|---|---|---|
| `incident` | 09-04 14:50–16:45 | the event itself: 10,797,274 events |
| `dryrun3-broken` | 09-05 01:15–01:30 | first run on the merged fix; 3 of 8 shards lost |
| `dryrun3-fixed` | 09-05 01:55–02:05 | same 8 shards after the runner fix; 8/8 |
| `dryrun4` | 09-05 02:25–04:15 | read path on 4F; 1.24 M picks, 4 mid-read renewals |

## Redaction

Every event passed through a credential-redaction pass before it was written,
and the writer re-reads its own output and refuses to upload if anything
credential-shaped survives. See `redact()` in
[`scripts/export_incident_logs.py`](../../scripts/export_incident_logs.py).

**Zero redactions were needed, across all 10,797,274 events.** That is a result,
not a formality: `earthscope_sdk` logs the refresh token and the access token at
`DEBUG`, and `worker.py --debug` sets the root logger to `DEBUG`. The only thing
standing between that and CloudWatch is one line in `s3_helper.py`:

```python
logging.getLogger("earthscope_sdk").setLevel(logging.INFO)
```

The export confirms that pin held for every build in the window. Anyone with
`logs:GetLogEvents` on the group would otherwise have been able to read our
refresh token.

## Re-deriving the numbers

```bash
python scripts/analyse_incident_logs.py <export_dir>
```

It parses the `HTTP Request:` lines — one per outbound request — rather than
grepping for substrings, and prints the token-endpoint rate, the credential
requests, the year-less requests by code, and the distinct scopes asked.

**Why that distinction matters.** The first version of the report was built with
`filter_log_events` substring counts, which match every line mentioning a scope:
the request, the warning that follows it, and each retry. That inflated the
year-less total roughly fivefold and, because the storm was measured over too
narrow a window, understated the storm. The corrected figures:

| figure | first published | corrected | how |
|---|--:|--:|---|
| year-less requests | 20,024 | **3,925** | request lines, parsed |
| distinct codes doing it | 8 | **2** (7D, 2F) | ditto |
| share of refusals | 27% | **24.4%** of 16,070 | ditto |
| token-endpoint POSTs | 260,322 / 66 min | **351,735**, 12:21–16:36 | full span, not a slice |
| peak | 5,104/min | **5,104/min at 15:55** | unchanged |

Five of the eight codes in the original table — `3J`, `3K`, `4F`, `3L`, `3Y` —
made no credential request at all in that window. `3D`, `YG`, `3H`, `YW`, `1P`,
`YU`, `YR` made requests but never a year-less one.

The ratio survived the correction because numerator and denominator were
inflated together. That is the signature of substring counting, and it is worth
knowing about before quoting any log-derived number.

## What the incident window actually contains

- **351,735** POSTs to `login.earthscope.org/oauth/token`, every one answered
  `403 invalid_grant / user is blocked`. Sustained above 100/min from 13:56,
  peaking at 5,104/min at 15:55, stopping at 16:35:57 when the fleet was
  terminated.
- **20,558** credential requests to `api.earthscope.org/beta/user/credentials/`,
  of which **16,070** were refusals: 8,607 × 403, 3,925 × 400, 3,538 × 404.
- **3,925** of those requests carried a temporary FDSN code and no year and
  could only ever be answered 400 — all of them `7D` (3,442) or `2F` (483).
- **Zero** lines from the `PermissionError` branch of the read path, which is
  how we know the code meant to bound credential refreshes after an
  `AccessDenied` had never executed.

## The AWS side

Nothing in our AWS account grants access to the archive. `s3-miniseed-v2` is an
EarthScope role alias; the credentials that read data are issued by EarthScope
and scoped by network and year. Our account contributes one thing: a
`secretsmanager:GetSecretValue` on a single secret ARN holding the refresh
token, read into the container environment.

Worth a reviewer's attention:

- The Batch job role and the ECS execution role are **the same role**
  (`SeisBenchBatchRole`), and it carries `AmazonS3FullAccess`. That grants
  nothing against the EarthScope archive, but it is broader than the account's
  own buckets need. Not yet scoped down.
- Log retention is 5 days on a group that is the only record of what the fleet
  did. That is why this export exists, and it is the wrong way round — the
  retention should outlive the incident-response window.
- There is no alarm on the token-endpoint rate. We did not detect this;
  EarthScope told us. A metric filter on the POST line with an alarm wired to
  zero the fleet is the cheapest control missing here.

## Reproducing the client behaviour without our account

```bash
python scripts/earthscope_selftest.py --offline
git show c5846f3:sb_catalog/src/s3_helper.py > /tmp/old.py
python scripts/earthscope_selftest.py --offline --baseline /tmp/old.py
```

Offline mode sends nothing and needs no EarthScope account. The `--baseline`
run puts the incident build beside the current one; all six checks fail on the
former and pass on the latter.
