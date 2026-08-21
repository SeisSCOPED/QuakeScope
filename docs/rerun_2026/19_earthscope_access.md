# 19 — EarthScope access: what is reachable without credentials

Written after testing, not reading. Every claim below was run from a laptop with
no EarthScope role granted.

## The sponsored open-data bucket is public, and it is in our region

```bash
aws s3 ls --no-sign-request s3://earthscope-geophysical-data/miniseed/
```

**`earthscope-geophysical-data`, us-east-2, anonymous.** No `s3-miniseed` role,
no access-point alias, no credential exchange. It is also the *same region as the
Fargate quota*, so those reads are not cross-region — unlike `scedc-pds`
(us-west-2) and `ncedc-pds` (us-east-1).

Layout matches what `EarthScopeS3ObjectHelper` already builds:

```
miniseed/<NET>/<YEAR>/<DAY>/<STA>.<NET>.<YEAR>.<DAY>
```

Verified end to end: obspy reads `UW/2019/187/RATT.UW.2019.187` (52 MB,
214 traces, 100 Hz `HH?` present) straight from the anonymous filesystem.

## But it is eight networks, not all of them

```
AK  II  IU  N4  PB  TA  UU  UW
```

Against the western-states campaign:

| | stations | networks |
|---|--:|--:|
| EarthScope-routed | 20,902 | 115 |
| in the public bucket | **1,392** | 7 |
| still need credentials | **19,510** | 108 |

So this is **6.7% of the EarthScope-routed western stations**. It is a real
unblock for `UW`, `UU` and `TA` — and `UW` alone is most of the Washington set —
but it does not replace the role grant. The temporary deployments that dominate
the western list (`XD`, `ZI`, `ZG`, `1D`, …) are not in it; `miniseed/XD/` does
not exist.

**The blocker in [18](18_launch_readiness.md) stands for 93% of the affected
stations.**

## The access point could not be recovered from local notes

Searched: every commit on every branch (`""` in all of them), the working tree,
stashes, `~/Downloads`, `~/Desktop`, `~/.aws`, `~/.earthscope`, and both shell
histories. Nothing. It is not on this machine, so it has to come from the 2025
controller instance or from EarthScope directly.

## The discover API does not help S3 lookups

`GET /beta/discover/datasource/stream` returns, per the OpenAPI spec:

```
description  edid  station  station_edid  stream_type
facility  software  label  sample_interval  names
```

**No object keys, no prefixes, no time ranges, no availability.** It is a
catalogue of what streams exist, not an index of where bytes are, so it cannot
replace a LIST or a GET. `sample_interval` is the one field of interest, and the
channel policy in [17](17_launch_conventions.md) deliberately avoids needing it:
the band code already implies the rate, so looking it up per station-day would
cost a network round trip to re-derive a constant.

(The SDK's `list_stream_datasources` also returns 422 on the obvious parameter
forms; not worth debugging for a call we do not want.)

## The optimisation that does exist: skip the LIST

`s3_helper.py` lists the whole `<NET>/<YEAR>/<DAY>/` prefix and regex-matches,
because the 2025 code found a **version number** in the object name:

```python
# earthscope object name has version number
uri = list(filter(lambda v: re.match(r, v), avail_uri[net]))
```

In this bucket there is no version suffix — names are exactly
`<STA>.<NET>.<YEAR>.<DAY>`, and a `head-object` on a constructed key succeeds
without any LIST. That makes the key deterministic and the LIST unnecessary.

Two reasons not to act on it yet:

1. The LIST is **already amortised** — one per network-day, shared by every
   station of that network in that day. A 40-station shard over 20 days costs 20
   LISTs for 800 station-days. This is not where the time goes.
2. The credentialed access point may still carry version suffixes, which is
   presumably why the regex exists. Changing the reader on the evidence of the
   public bucket alone would be fixing one archive by breaking another.

Worth revisiting once the profile in
[16](16_skypilot_vs_fargate.md) attributes real time to stage 1.

## What to ask EarthScope for

> Please grant `mdenolle@uw.edu` the `s3-miniseed` role, and confirm the S3
> access-point alias for our account.

Check the state at any time with:

```bash
pixi run -e cloud python scripts/check_earthscope_access.py
```
