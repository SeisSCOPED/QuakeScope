"""Hit rate by campaign and year, from S3 listings alone.

The cost model divides real work by PLANNED station-days, and the hit rate -
what fraction of planned station-days actually hold data - is the dominant
uncertainty (OPTIMISE item 9). It is a property of the queue, not the picker, so
it needs no inference, no GPU and no picking: one LIST per (archive, network,
day) answers it for every station in that network at once.

Prefix and basename logic is imported from the pipeline's own S3ObjectHelper
classes so the survey cannot drift from what the worker actually looks for.
"""
import argparse
import collections
import datetime
import json
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sb_catalog.src.constants import NETWORK_MAPPING, select_channel  # noqa: E402
from sb_catalog.src.s3_helper import (EarthScopeS3ObjectHelper,  # noqa: E402
                                      NCEDCS3ObjectHelper, SCEDCS3ObjectHelper)
from sb_catalog.src.shard_planner import _overlaps, _operating_windows  # noqa: E402

OPEN = {"AK", "II", "IU", "N4", "PB", "TA", "UU", "UW"}
HELPER = {"scedc": SCEDCS3ObjectHelper(), "ncedc": NCEDCS3ObjectHelper(),
          "earthscope": EarthScopeS3ObjectHelper()}

_local = threading.local()


# scedc-pds is the one cross-region archive; the rest are us-east-2.
REGION = {"scedc-pds": "us-west-2"}


def s3(bucket):
    region = REGION.get(bucket, "us-east-2")
    cache = getattr(_local, "c", None)
    if cache is None:
        cache = _local.c = {}
    if region not in cache:
        cache[region] = boto3.client(
            "s3", region_name=region,
            config=Config(signature_version=UNSIGNED,
                          retries={"max_attempts": 5, "mode": "adaptive"}))
    return cache[region]


_cache, _cache_lock = {}, threading.Lock()


def listing(prefix):
    """All keys under a bucket-qualified prefix, as basenames.

    Cached: SCEDC's prefix ignores the network - one listing covers every
    network for that day - so without this the same 30k-object listing is
    fetched once per network in the campaign."""
    with _cache_lock:
        if prefix in _cache:
            return _cache[prefix]
    out = _listing_uncached(prefix)
    with _cache_lock:
        _cache[prefix] = out
    return out


def _listing_uncached(prefix):
    bucket, _, key = prefix.partition("/")
    out, tok = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=key, MaxKeys=1000)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3(bucket).list_objects_v2(**kw)
        out.extend(o["Key"].rsplit("/", 1)[-1] for o in r.get("Contents", []))
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


def present(archive, net, year, day):
    """Set of station keys with data. SCEDC/NCEDC encode the channel in the
    object name, so those return (sta, loc, cha); EarthScope stores one object
    per station-day, so it returns (sta,)."""
    h = HELPER[archive]
    try:
        names = listing(h.get_prefix(net, year, day))
    except Exception as exc:
        return None, str(exc)
    found = set()
    if archive == "scedc":
        # NNSSSSSCCLLL YYYYDDD .ms  - net 2, sta 5, cha 2, comp 1, loc 3
        # verified against a real listing: 'AZBZN__BHE___2015100.ms', len 23
        for n in names:
            if len(n) != 23 or not n.endswith(".ms") or n[:2] != net:
                continue
            found.add((n[2:7].rstrip("_"), n[10:13].rstrip("_"), n[7:9]))
    elif archive == "ncedc":
        for n in names:
            p = n.split(".")
            if len(p) >= 6 and p[1] == net:
                found.add((p[0], p[3], p[2][:2]))
    else:
        for n in names:
            p = n.split(".")
            if len(p) >= 2 and p[1].split("#")[0] == net:
                found.add((p[0],))
    return found, None


def campaign_stations(camp, bucket, cache_dir="."):
    """Download the campaign's stations.parquet once, to `cache_dir`."""
    local = pathlib.Path(cache_dir) / f"stations_{camp}.parquet"
    if not local.exists():
        boto3.client("s3", region_name="us-east-2").download_file(
            bucket, f"{camp}/stations.parquet", str(local))
    return local


def sample_days(years, per_year):
    out = []
    for y in years:
        for k in range(per_year):
            doy = int(40 + (300 / max(per_year, 1)) * k + 37 * k) % 360 + 1
            try:
                out.append(datetime.date(y, 1, 1) + datetime.timedelta(days=doy - 1))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaigns", default="scedc,ncedc,earthscope,obs,western")
    ap.add_argument("--per-year", type=int, default=2)
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--open-only", action="store_true",
                    help="EarthScope: skip networks needing credentials")
    ap.add_argument("--bucket", default="quakescope-picks-2026",
                    help="bucket holding <campaign>/stations.parquet")
    ap.add_argument("--out", default="hitrate.json")
    a = ap.parse_args()

    days = sample_days(range(a.start_year, a.end_year + 1), a.per_year)
    print(f"{len(days)} sample days, {a.start_year}-{a.end_year}\n")
    results = {}

    for camp in a.campaigns.split(","):
        st = pd.read_parquet(campaign_stations(camp, a.bucket))
        win = _operating_windows(st)
        st = st.assign(net=[str(i).split(".")[0] for i in st["id"].astype(str)])
        st = st.assign(cha=[select_channel([c for c in str(x).split(",") if c.strip()])
                            for x in st["channels"]])
        st = st[st["cha"].notna()]

        tasks, skipped_sd = [], 0
        for d in days:
            y, doy = f"{d:%Y}", f"{d:%j}"
            live = st[[_overlaps(win.get(str(i)), d, d + datetime.timedelta(days=1))
                       for i in st["id"].astype(str)]]
            for net, g in live.groupby("net"):
                arch = NETWORK_MAPPING.get(net)
                if arch not in HELPER:
                    continue
                if arch == "earthscope" and a.open_only and net not in OPEN:
                    skipped_sd += len(g)
                    continue
                tasks.append((arch, net, y, doy, g))

        planned = hit = 0
        errs = collections.Counter()
        by_year = collections.defaultdict(lambda: [0, 0])
        by_arch = collections.defaultdict(lambda: [0, 0])

        def run(t):
            arch, net, y, doy, g = t
            found, err = present(arch, net, y, doy)
            return t, found, err

        with ThreadPoolExecutor(max_workers=16) as pool:
            for (arch, net, y, doy, g), found, err in pool.map(run, tasks):
                if found is None:
                    errs[err[:60]] += 1
                    continue
                for _, r in g.iterrows():
                    sid = str(r["id"]).split(".")
                    sta, loc = sid[1], (sid[2] if len(sid) > 2 else "")
                    key = (sta,) if arch == "earthscope" else (sta, loc, r["cha"])
                    planned += 1
                    by_year[y][0] += 1
                    by_arch[arch][0] += 1
                    if key in found:
                        hit += 1
                        by_year[y][1] += 1
                        by_arch[arch][1] += 1

        results[camp] = {"planned": planned, "hit": hit,
                         "by_year": {k: v for k, v in sorted(by_year.items())},
                         "by_archive": {k: v for k, v in sorted(by_arch.items())},
                         "skipped_needs_credentials": skipped_sd,
                         "errors": dict(errs)}
        rate = hit / planned if planned else float("nan")
        print(f"{camp:11s} sampled {planned:>7,} station-days  "
              f"hit {hit:>7,}  = {100*rate:5.1f}%"
              + (f"   [{skipped_sd:,} skipped: credentials]" if skipped_sd else ""))
        if errs:
            print(f"            errors: {dict(errs)}")

    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
