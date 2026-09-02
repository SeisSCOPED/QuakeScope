# 19 — EarthScope access: two tiers, open data first

**Resolved.** The blocker recorded in [18](18_launch_readiness.md) was a
**renamed role**, not a missing entitlement.

```
$ pixi run -e cloud python scripts/check_earthscope_access.py
[1] open data   : OK, 8 networks, no credentials (AK, II, IU, N4, PB, TA, UU, UW)
[2] identity    : mdenolle@uw.edu
[3] role        : OK 's3-miniseed-v2', expires 2026-08-29 04:39:58+00:00
[4] access point: OK, earthscope-mseed-v2-4fdodyzpsz8u8u... (11 objects in ZI/2019/187)
```

## What was actually wrong

`s3-miniseed` is retired. Asking for it returns

```
UnauthorizedError: You are not allowed to assume role 's3-miniseed'
```

which reads as an entitlement the account lacks, and sent us to EarthScope
support. The role is **`s3-miniseed-v2`**, and it was available the whole time.
The regression test asserts the name for that reason.

## Two archives behind one data-centre name

Per [the SDK tutorial](https://docs.earthscope.org/sdk/s3-direct-access-tutorial):

| tier | bucket | networks | credentials |
|---|---|---|---|
| **Open Data** | `earthscope-geophysical-data` | `AK II IU N4 PB TA UU UW` | **none** |
| Repository | `earthscope-mseed-v2-…-s3alias` | everything else | `s3-miniseed-v2`, us-east-2 |

**Open Data is preferred wherever it serves the network.** A campaign over those
networks then cannot fail on an expired token, a renamed role, or a support
ticket — which is exactly how this blocker arose. It is also anonymous, so
workers need no secret in their environment.

The access-point alias is **published in that tutorial**, so it is now a default
in `s3_helper.py` rather than a value to recover from a previous campaign's
notes. Override with `EARTHSCOPE_S3_ACCESS_POINT` if EarthScope issues another.

Both tiers use the identical layout, so only the bucket differs:

```
miniseed/<NET>/<YEAR>/<DAY>/<STA>.<NET>.<YEAR>.<DAY>
```

## The naming bug this exposed

The two buckets do **not** name objects identically:

```
open data   ALCT.UW.2019.187        <- no version
restricted  ADO.CI.2019.187#2       <- version suffix
```

`get_basename` returned `f"{sta}.{net}.{year}.{day}#."`, a regexp **requiring**
the `#`. Against Open Data it matched nothing — and matched nothing *silently*,
because a station with no matching object is indistinguishable from a station
that was not recording. Every Open Data station would have been skipped with no
error at all.

It now accepts both forms, anchored so `RATT` cannot match `RATTX`:

```python
rf"{re.escape(f'{sta}.{net}.{year}.{day}')}(#.*)?$"
```

## Credentials are acquired lazily

A campaign that only touches Open Data never calls the credential exchange, so
it cannot be delayed or failed by it. `_check_archives_reachable` still fails
fast — but only when the shard actually contains restricted networks, and it now
names the role and the access point instead of raising `KeyError('earthscope')`
once per station.

## SDK version — upgraded to 1.8.0 (2026-09-02)

> **The paragraph that stood here was wrong on both counts, and it is what cost
> the two weeks.** It said the upgrade was blocked by an `aioboto3` conflict,
> and that scoping was unnecessary "since unscoped v2 credentials already read
> every network tested". Neither held. Scoping is **required** — see below —
> and the conflict was an artifact of how the environment was split, not a real
> incompatibility.

`get_aws_credentials` takes query parameters only from **1.8.0**, and those
parameters are what carry `network=FDSN:<NET>` to the token exchange. Without
them the role gets an unscoped credential that can `ListBucket` but cannot
`GetObject`.

**The "conflict" was a split-resolver artifact.** `s3fs` and `boto3` sat in
pixi's `[dependencies]` (conda) while `earthscope-sdk` sat in
`[pypi-dependencies]`. The conda solve pinned `aiobotocore==3.8.0` for its
`s3fs`; the SDK's `aioboto3` pins `aiobotocore==2.25.1` exactly; pixi cannot
satisfy both because neither resolver sees the whole graph. Moving `s3fs` and
`boto3` to `[pypi-dependencies]`, so one resolver owns the AWS stack, resolves
immediately.

The image had a second, unrelated pin: `boto3==1.35.81` caps `botocore<1.36.0`,
while `aioboto3>=13.4.0` needs `botocore>=1.36.0`. That one is genuinely
irreconcilable, so the **pin goes, not the SDK**. Resolved for cp312:

```
earthscope-sdk 1.8.0   boto3 1.40.61   botocore 1.40.61
aioboto3 15.5.0        aiobotocore 2.25.1   s3fs 2026.7.0     (91 packages)
```

`seisbench==0.12.5` is unmoved, so the weight-version resolution that defines
the picks is unchanged.

## Scoping is what the whole blocker was

An **unscoped** credential for `s3-miniseed-v2` carries `s3:ListBucket` and not
`s3:GetObject`. Every listing succeeds; every read returns `AccessDenied`.

That asymmetry is why this read as an entitlement gap for two weeks. Listing
works, so the role is plainly assumed and plainly valid — and the denial only
arrives at the read, where it looks like a permissions boundary rather than a
malformed request. The section above even cited successful listings on four
temporary networks as evidence that scoping was unnecessary. **Listing is not
read access**, and that was the wrong inference from the right observation.

```python
client.user.get_aws_credentials(
    role="s3-miniseed-v2",
    network="FDSN:AV",        # required for every restricted GET
    year=2019,                # additionally, for temporary networks only
)
```

Temporary networks — FDSN codes beginning with a digit or `X`/`Y`/`Z` — are
also year-scoped. That is most of this campaign: `XD`, `ZI`, `ZG`, `1D`, `1B`.
The [tutorial](https://docs.earthscope.org/sdk/s3-direct-access-tutorial#repository-direct-access)
is explicit, and its own example uses a temporary network:

> When requesting credentials for a temporary FDSN network (i.e. it starts with
> digits 0-9 or letters `X`, `Y` or `Z`), you must **include `year`** in the
> credential request

> Temporary FDSN networks reuse classical 2-character FDSN network codes so
> credentials must be scoped to a specific network-year prefix:
> `miniseed/<NETWORK>/<YEAR>/*`.

**Read the temporary-network section, not just the first example.** The
tutorial's opening snippet passes `network` alone, which is correct for a
permanent code and wrong for most of this campaign.

The same page states the asymmetry that made this so hard to see, and it is
worth quoting in full:

> Credentials **allow listing the entire `miniseed/` prefix** on the access
> point, including all network "subdirectories". All networks appear in
> listings but **cannot be downloaded unless your credentials were issued for
> that network**.

Every diagnostic we ran began with a successful listing. By design, that
listing would have succeeded no matter how wrong the credential was.

Confirmed interactively from EC2 in us-east-2 on 2026-09-02: the same object
that returns `AccessDenied` under an unscoped credential returns bytes under
`network=FDSN:AV`. One parameter is the entire difference.

**It was our bug.** No ticket to EarthScope was warranted, and the report should
not carry one.

## Effect on the western campaign

Of 20,902 EarthScope-routed stations, 1,392 are on Open Data networks and 19,510
are restricted. **Both tiers now work**, so the campaign is no longer limited to
the 3,211 SCEDC/NCEDC stations.

## Access confirmed for the western campaign (2026-08-29)

The `s3-miniseed-v2` role is granted to `mdenolle`, and it reaches the networks
western actually needs. Spot-checked the five largest restricted networks in the
campaign:

| network | stations | years listed |
|---|--:|--:|
| `XD` | 5,145 | 20 |
| `NP` | 3,033 | 22 |
| `ZI` | 2,183 | 6 |
| `ZG` | 1,139 | 14 |
| `1D` | 898 | 5 |

All four temporary networks (`X`, `Z`, `1` prefixes) listed without
network- or year-scoped credentials, so the scoping the tutorial describes is not
required for read access with this role.

**The remaining gap is the container, not the account.** A Fargate task has no
EarthScope login, so the restricted tier needs `ES_OAUTH2__REFRESH_TOKEN`
supplied to the job — properly via Secrets Manager and
`containerProperties.secrets`, never baked into a job definition in plaintext.
SCEDC, NCEDC and the Open Data eight are anonymous and unaffected.

## Credentials in the container (2026-08-30) — re-verified 2026-09-01

> **A retraction posted here on 2026-09-01 was itself wrong, and is withdrawn.**
> It claimed no job definition carried the secret and that none set an execution
> role. Both claims came from `aws batch describe-job-definitions` run through
> the **`aws-cli/2.0.34`** on this laptop — a 2020 build whose service model
> predates `secrets`, `executionRoleArn`, `platformCapabilities`,
> `networkConfiguration` and `evaluateOnExit`, so it drops all of them from its
> output without a warning. Re-checked through boto3 1.40.61, the wiring is
> present and complete. **Audit job definitions with boto3, not the local CLI**
> — see [OPTIMISE.md](OPTIMISE.md) item 0b.

The refresh token lives in Secrets Manager and is injected as an environment
variable by Batch, never baked into a job definition:

```
secret : quakescope/earthscope-refresh-token   (us-east-2)
policy : QuakeScopeEarthScopeSecretRead on SeisBenchBatchRole, scoped to that ARN
wiring : containerProperties.secrets -> ES_OAUTH2__REFRESH_TOKEN
```

Only `quakescope_2026_earthscope:2/:4` and `quakescope_2026_western:2` carry it.
SCEDC, NCEDC and the Open Data eight are anonymous and do not.

**Deployed state, read back through boto3 on 2026-09-01:**

```
quakescope_2026_earthscope:4
  secrets          : [{name: ES_OAUTH2__REFRESH_TOKEN,
                       valueFrom: arn:...:secret:quakescope/earthscope-refresh-token-bGo4vN}]
  executionRoleArn : arn:aws:iam::073795725844:role/SeisBenchBatchRole
  jobRoleArn       : arn:aws:iam::073795725844:role/SeisBenchBatchRole
```

Fargate injects `containerProperties.secrets` with the **execution** role, and
here the execution and job roles are the same role, so the
`QuakeScopeEarthScopeSecretRead` policy above is on the role that performs the
injection. Confirmed by simulation rather than by reading the policy:

```
$ simulate_principal_policy(SeisBenchBatchRole,
      'secretsmanager:GetSecretValue', <the token ARN>)
-> allowed
```

The role's trust policy is `ecs-tasks.amazonaws.com`, which is what both roles
need.

> ⚠️ **The "verified in a running container" evidence below does not show what
> it claims** (2026-09-02). `Load ZI.CAMP.10 @ earthscope` is printed **before**
> the read. It proves the code reached the read, not that bytes came back.
> Measured with `diag-earthscope` from Fargate in us-east-2: `LIST` returns 163
> objects in 1.4 s, then `HEAD` returns **403** and `GetObject` returns
> **AccessDenied** in 0.0 s. The role is assumed successfully and is not
> entitled to read. See [OPTIMISE.md](OPTIMISE.md) item 0g.
>
> The same section's claim that "all four temporary networks listed without
> network- or year-scoped credentials, so the scoping the tutorial describes is
> not required for read access with this role" is contradicted by that
> measurement: **listing is not read access**, and the scoping may well be
> exactly what is missing.
>
> **It was.** Resolved 2026-09-02 — see "Scoping is what the whole blocker was"
> above. The role was entitled the whole time; the request was malformed.

**Verified in a running container.** A job on `ZI` - a restricted network -
logged:

```
POST https://login.earthscope.org/oauth/token                "200 OK"
GET  .../beta/user/credentials/aws/s3-miniseed-v2            "200 OK"
Load ZI.CAMP.10  2019.187 @ earthscope
```

Rotation: `es login` on a laptop, then `put-secret-value` with the new token.
Nothing in the image or the job definition changes.

## Is it safe that the image is public? (2026-09-01)

`ghcr.io/seisscoped/quakescope` is **public** — an anonymous manifest pull
returns 200. That is fine, and the design is the reason:

- **The image carries no credential.** The Dockerfile copies `src/` and
  `models/v3/` and nothing else; there is no `ARG`, no `ENV`, no token.
- **The secret is injected at task start**, by Fargate, from Secrets Manager,
  using the execution role. `containerProperties.secrets` stores the **ARN**,
  not the value, so `describe-job-definitions` exposes nothing either.
- Pulling the public image therefore yields no way to read restricted data.

**The exposure is at runtime and inside the account, not in the registry.** The
token exists as an environment variable in a running task, so it is reachable by
anyone who can read that task's logs, exec into it, or assume the execution
role. Three specifics found on 2026-09-01:

1. **`submit_helper.py` logged the whole token at INFO** — `"EarthScope refresh
   token applied: <token>"`. On Batch that goes to CloudWatch, readable with
   `logs:GetLogEvents` on `/aws/batch/job`. **Fixed**: it now logs a length and
   a truncated SHA-256, which still answers "is one set" and "is it the one I
   rotated to". A CloudWatch search over the last 180 days found **no** matching
   event, so the token was never actually written and does not need rotating on
   this account.
2. **`--debug` would leak it.** `worker.py` calls `logging.basicConfig(level=
   DEBUG)`, which sets the **root** logger, and `earthscope_sdk`'s
   `auth_flow.py` does `logger.debug(f"Refreshed tokens: {self._tokens}")`.
   So `--debug` on an EarthScope campaign writes the refresh *and* access token
   into CloudWatch. Not fixed — the safe change is to raise
   `logging.getLogger("earthscope_sdk").setLevel(INFO)` regardless of `--debug`.
   Until then, **do not pass `--debug` to a campaign that touches restricted
   EarthScope.**
3. **`SeisBenchBatchRole` carries `AmazonS3FullAccess`** and is both the job role
   and the execution role. Unrelated to the image being public, but it means one
   role compromise reaches every bucket in the account.

**Also do not use the refresh token from a laptop.** `earthscope_sdk`'s refresh
grant saves a rotated token to *local SDK state*, not back to Secrets Manager —
so if EarthScope's Auth0 application has rotation enabled, one local run
invalidates the credential every campaign job depends on.

## Open: EarthScope reads are much slower than SCEDC

The same test then sat on `Load ZI.CAMP.10` for **25 minutes without a further
log line**, and was cancelled rather than left running. For comparison, the SCEDC
fire drill did 8 station-days in about 5 minutes.

Not yet diagnosed, and it should be before either EarthScope campaign is
planned. The likely cause is structural rather than a hang: SCEDC and NCEDC
store one object per channel, so a station-day fetches only the band it needs,
while EarthScope stores **one multi-channel object per station-day** that is
downloaded and parsed in full before `.select()` picks a band out of it. The UW
sample read earlier held 214 traces across 38 channel codes, of which the picker
uses three.

If that is the explanation, the one-channel policy saves inference on EarthScope
but no I/O at all, and the per-band-day cost model in
[16](16_skypilot_vs_fargate.md) - measured on SCEDC - understates EarthScope by
a wide margin. That would move the campaign estimates, so it needs a profile
(`--profile`, stage `s3.get` against bytes transferred) before planning.
