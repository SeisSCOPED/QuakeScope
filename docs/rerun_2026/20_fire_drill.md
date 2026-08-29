# 20 — Fire drill: one job, end to end

Run on 2026-08-29 against real infrastructure. Everything below was executed,
not designed.

## What ran

One Fargate Spot job, `quakescope_2026_firedrill:1`, image `42b44bb`:

| | |
|---|---|
| Campaign | `s3://quakescope-picks-2026/firedrill` |
| Stations | 6 CI stations around Ridgecrest |
| Days | 2019.186–187 (the M6.4/M7.1 sequence) |
| Shards | 1, 8 station-days |
| Weight | `jma_wc`, thresholds 0.2/0.2 |
| Result | SUCCEEDED in ~5 min |

## What it produced

**54,854 picks** — 29,008 P, 25,846 S — in a single 1.5 MB Parquet object.

```
firedrill/picks/network=CI/year=2019/month=07/2019186-2019188-55c7b5216b18.parquet
firedrill/manifests/2019186-2019188-55c7b5216b18.json
firedrill/complete/2019186-2019188-55c7b5216b18.json
firedrill/runs/4b3ac226-4128-459e-a287-909043ce1bd9.json
```

`channels: ['HH']` — one band per station, as
[17](17_launch_conventions.md) specifies. Amplitudes set on 54,779 of 54,854;
the 75 NaN are picks whose window fell inside the taper, which is the designed
behaviour rather than a failure.

**Four of the six stations produced picks.** `CI.GPO.` and `CI.JRC.` produced
none. Not investigated — worth a look before the real SCEDC campaign, since
"no data" and "silently skipped" look identical from the outside.

## The checks that matter

A job reporting SUCCEEDED while writing nothing is a failure mode this project
has already hit, so success was verified from the data, not the exit code:

1. **Manifest read, no LIST.** `manifest["files"][*]["path"]` are full `s3://`
   URIs; reading them returned 54,854 rows, matching `n_picks` exactly.
2. **Partition pruning.** `filters=[("network","=","CI"),("year","=",2019)]`
   returned 8,506 P picks for `CI.CLC.` without scanning the rest.
3. **Run provenance.** `runs/<rid>.json` records weight `jma_wc`, thresholds
   0.2/0.2, `seisbench 0.12.5`, components `ZNE12`.
4. **Waveform round trip.** Took the highest-confidence pick
   (`CI.CLC. HH P`, conf 0.993, amp 1.13e-2 m, 2019-07-05T06:59:03.148Z), built
   the object key with the same `SCEDCS3ObjectHelper` the picker uses, pulled the
   day file, and measured either side of the pick:

   ```
   mean|amp| 10 s before : 340
   mean|amp| 10 s after  : 9,157      ratio 26.9x
   ```

   The pick lands on an arrival.

## Watching it in the console

```
S3 -> quakescope-picks-2026 -> firedrill/ -> picks/
Batch -> Jobs -> queue niyiyu_earthscope_missing_station
CloudWatch -> /aws/batch/job -> quakescope_2026_firedrill/...
```

Objects appear per `(network, year, month)` partition as each shard closes, so
on a real campaign the count under `picks/` grows as work completes.

## Cluster state

Zero Batch jobs active in any queue. Zero SkyPilot EC2 instances and zero open
spot requests in us-east-2 or us-west-2. One unrelated instance is running,
`niyiyu-quakescope-web-service` (t2.large, us-east-2) — not part of this work and
left alone.
