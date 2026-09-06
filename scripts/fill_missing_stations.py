#!/usr/bin/env python
"""Add stations a campaign's plan names but its station table does not have.

The plan and the station table are built separately, and they can disagree. On
the 2026-09-06 western run 127 shards named 60 stations absent from
`stations.parquet`, which raised `KeyError: 'LH.HDSE.'` several frames inside
the read loop, failed the shard, and released it to the queue for the next
worker to rediscover. 122 shards cycled ~22 times each and the campaign stopped
at 99.8% complete.

The client no longer spins on that - a shard whose stations are absent is set
aside for review rather than requeued - but "set aside" is not the answer when
the stations exist and we simply did not record them. All four spot-checked
were in FDSN:

    LH.HDSE   2023-10-25 .. open     HN LC LK VE
    LH.NCBE   2023-10-25 .. open     HN LC LK VE
    YW.FACN   2007-11-02 .. 2010-10-22   BH LH LO
    YG.OR22   2012-06-08 .. 2012-07-09   EH LO

    python scripts/fill_missing_stations.py --campaign western --dry-run
    python scripts/fill_missing_stations.py --campaign western --apply

APPEND ONLY. Existing rows are never modified, so a station already in the
table keeps whatever it has. The previous table is copied beside the new one
first, and the bucket has versioning, so this is recoverable twice over.

A station FDSN does not know is REPORTED, not invented: if the plan names
something that does not exist, the plan is wrong and a person should see that
rather than have a placeholder row hide it.
"""

from __future__ import annotations

import argparse
import io
import json
import sys

BUCKET = "quakescope-picks-2026"


def missing_stations(s3, campaign):
    """Stations the plan names that the table lacks, with their shard counts."""
    import collections

    import pandas as pd

    df = pd.read_parquet(io.BytesIO(s3.get_object(
        Bucket=BUCKET, Key=f"{campaign}/stations.parquet")["Body"].read()))
    have = set(df["id"])
    body = s3.get_object(Bucket=BUCKET,
                         Key=f"{campaign}/shards.jsonl")["Body"].read().decode()
    counts = collections.Counter()
    shards = 0
    for line in body.splitlines():
        d = json.loads(line)
        miss = [t for t in d["stations"] if t not in have]
        if miss:
            shards += 1
            counts.update(miss)
    return df, counts, shards


def fetch(tids):
    """Ask FDSN for these stations. Channel level: we need the band codes."""
    import warnings

    warnings.filterwarnings("ignore")
    from obspy.clients.fdsn import Client

    cl = Client("EARTHSCOPE", timeout=120)
    by_net = {}
    for t in tids:
        net, sta = t.split(".")[0], t.split(".")[1]
        by_net.setdefault(net, set()).add(sta)

    rows, absent = [], []
    for net, stas in sorted(by_net.items()):
        try:
            # One request per network rather than per station: the same answer
            # for a fraction of the load on a service we are already sensitive
            # about.
            inv = cl.get_stations(network=net, station=",".join(sorted(stas)),
                                  level="channel")
        except Exception as exc:
            print(f"  {net}: FDSN said {type(exc).__name__}: {str(exc)[:80]}")
            absent += [t for t in tids if t.startswith(f"{net}.")]
            continue
        found = set()
        for n in inv:
            for s in n:
                found.add(s.code)
                # `channels` in this table is the set of BAND codes, comma
                # separated - the two-character prefix, not the full code.
                bands = sorted({c.code[:2] for c in s.channels})
                locs = sorted({c.location_code or "" for c in s.channels})
                for loc in locs:
                    tid = f"{n.code}.{s.code}.{loc}"
                    if tid not in tids:
                        continue
                    rows.append({
                        "id": tid,
                        "network_code": n.code,
                        "station_code": s.code,
                        "location_code": loc,
                        "channels": ",".join(bands),
                        "latitude": float(s.latitude),
                        "longitude": float(s.longitude),
                        "elevation": float(s.elevation or 0.0),
                        "start_date": _yd(s.start_date),
                        "end_date": _yd(s.end_date),
                        "state": None,
                    })
        absent += [t for t in tids
                   if t.startswith(f"{net}.") and t.split(".")[1] not in found]
    return rows, sorted(set(absent))


def _yd(t):
    """The table stores dates as YYYY.DDD floats."""
    if t is None:
        return float("nan")
    return float(f"{t.year}.{t.julday:03d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    import boto3
    import pandas as pd
    s3 = boto3.client("s3")

    df, counts, shards = missing_stations(s3, a.campaign)
    if not counts:
        print(f"  {a.campaign}: every station the plan names is in the table")
        return 0
    print(f"  {a.campaign}: {len(counts)} station(s) missing, named by "
          f"{shards:,} shard(s)")
    for t, n in counts.most_common(10):
        print(f"     {t:16} in {n} shard(s)")
    if len(counts) > 10:
        print(f"     ... and {len(counts) - 10} more")

    rows, absent = fetch(set(counts))
    print(f"\n  FDSN has {len(rows)} of them; {len(absent)} genuinely absent")
    for t in absent[:10]:
        print(f"     NOT IN FDSN: {t}  ({counts[t]} shard(s)) — the plan is "
              f"wrong about this one")

    if a.dry_run:
        print("\n  dry run: nothing written")
        return 0
    if not rows:
        print("\n  nothing to add")
        return 1

    add = pd.DataFrame(rows)
    out = pd.concat([df, add], ignore_index=True)
    # Append only: if a row somehow already existed, the original wins.
    out = out.drop_duplicates(subset=["id"], keep="first")
    print(f"\n  {len(df):,} -> {len(out):,} rows (+{len(out) - len(df):,})")

    # Keep the previous table beside the new one. Versioning would cover this,
    # but a named copy is easier to reason about than a version id.
    s3.copy_object(Bucket=BUCKET, Key=f"{a.campaign}/stations.parquet.bak",
                   CopySource={"Bucket": BUCKET,
                               "Key": f"{a.campaign}/stations.parquet"})
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=f"{a.campaign}/stations.parquet",
                  Body=buf.getvalue())
    print(f"  wrote s3://{BUCKET}/{a.campaign}/stations.parquet "
          f"(previous kept as stations.parquet.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
