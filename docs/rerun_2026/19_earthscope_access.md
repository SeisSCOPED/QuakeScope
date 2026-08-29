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

## SDK version

The tutorial's `get_boto3_session(role=...)` (refreshable, and the recommended
form) and the `network=`/`year=` scoped credentials need **SDK ≥ 1.4**; the
pinned environment has **1.3.0**, whose `get_aws_credentials(role=...)` is
sufficient — it is what produced the output above.

Upgrading to 1.8.0 is **blocked**: it pulls `aioboto3`, which conflicts with the
pinned `aiobotocore==3.8.0`. Not worth destabilising the environment before
launch, since unscoped v2 credentials already read every network tested,
including temporary ones (`ZI`). Revisit after the campaign.

## Effect on the western campaign

Of 20,902 EarthScope-routed stations, 1,392 are on Open Data networks and 19,510
are restricted. **Both tiers now work**, so the campaign is no longer limited to
the 3,211 SCEDC/NCEDC stations.
