#!/usr/bin/env python
"""Plan `western-fill`: the stations of the stakeholder list the western catalogue lacks.

The western catalogue was planned from six state polygons (WA OR CA NV ID WY,
commit c1d7228). The stakeholder list, WestCoast_stations.txt (20,571 stations,
23 to 53 N), also holds Utah, Montana, Arizona, Colorado, New Mexico, British
Columbia, Alberta and Baja California, and the specification that started the
campaign (docs/rerun_2026/archive/09) named Utah and New Mexico explicitly.
Decided 2026-09-21: pick the whole list. Offshore stations are the obs
catalogue's and are left out; two Californian stations that return no data
from FDSN at all are left out too.

    python scripts/plan_western_fill.py --list ~/Downloads/WestCoast_stations.txt --dry-run
    python scripts/plan_western_fill.py --list ~/Downloads/WestCoast_stations.txt --write

`--write` installs the station table under _queues/western-fill/, plans
1986.001 to 2026.251 with the production planner (same 20-day, per-network
shards, clipped to operating windows) and writes the immutable queue. The
table is also saved as sb_catalog/configs/networks/western_fill.csv. The
campaign then needs an entry in fleet.json with `queue` under _queues/ and
`parquet_uri` set to the western catalogue, an access survey, and a target.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BUCKET = "quakescope-picks-2026"
QUEUE = f"s3://{BUCKET}/_queues/western-fill"
CATALOGUE = "western"
START, END = "1986.001", "2026.251"          # the span the western catalogue holds


def regions():
    """Natural Earth admin-1 polygons for the US, Canada and Mexico."""
    import cartopy.io.shapereader as shpreader
    from shapely.prepared import prep
    rec = shpreader.Reader(shpreader.natural_earth(resolution="10m", category="cultural",
                                                   name="admin_1_states_provinces"))
    out = []
    for r in rec.records():
        adm = r.attributes["admin"]
        if adm in ("United States of America", "Canada", "Mexico"):
            name = r.attributes["name"] + ("" if adm == "United States of America" else f" ({adm[:2]})")
            out.append((name, prep(r.geometry)))
    return out


def where(polys, lat, lon):
    from shapely.geometry import Point
    p = Point(lon, lat)
    for name, g in polys:
        if g.contains(p):
            return name
    return "offshore"


def yd(t):
    return float("nan") if t is None else float(f"{t.year}.{t.julday:03d}")


def fetch(pairs):
    """Channel-level metadata for (net, sta) pairs, one request per network."""
    warnings.filterwarnings("ignore")
    from obspy.clients.fdsn import Client
    cl = Client("EARTHSCOPE", timeout=180)
    by_net = {}
    for net, sta in pairs:
        by_net.setdefault(net, set()).add(sta)
    rows, absent = [], []
    for net, stas in sorted(by_net.items()):
        try:
            inv = cl.get_stations(network=net, station=",".join(sorted(stas)), level="channel",
                                  starttime="1986-01-01", endtime="2026-09-08")
        except Exception as exc:
            print(f"  {net}: FDSN said {type(exc).__name__}: {str(exc)[:80]}")
            absent += [(net, s) for s in stas]
            continue
        found = set()
        for n in inv:
            for s in n:
                found.add(s.code)
                # One row per location code, as the western table does: each
                # location is a separate instrument and a separate unit of work.
                by_loc = {}
                for c in s.channels:
                    by_loc.setdefault(c.location_code or "", []).append(c)
                for loc, chans in sorted(by_loc.items()):
                    bands = sorted({c.code[:2] for c in chans})
                    starts = [c.start_date for c in chans if c.start_date]
                    ends = [c.end_date for c in chans]
                    rows.append({
                        "id": f"{n.code}.{s.code}.{loc}", "network_code": n.code, "station_code": s.code,
                        "location_code": loc, "channels": ",".join(bands),
                        "latitude": float(s.latitude), "longitude": float(s.longitude),
                        "elevation": float(s.elevation or 0.0),
                        "start_date": yd(min(starts)) if starts else yd(s.start_date),
                        "end_date": 3000.001 if any(e is None for e in ends) else yd(max(ends)) if ends else yd(s.end_date),
                    })
        absent += [(net, s) for s in stas if s not in found]
    return pd.DataFrame(rows), absent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="WestCoast_stations.txt")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    wc = pd.read_csv(a.list, skipinitialspace=True)
    wc.columns = [c.strip() for c in wc.columns]
    wc["net"] = wc["Ntw.Code"].str.split(".").str[0]; wc["sta"] = wc["Ntw.Code"].str.split(".").str[1]
    wc = wc.drop_duplicates(["net", "sta"])
    western = pd.read_parquet(f"s3://{BUCKET}/{CATALOGUE}/stations.parquet")
    have = set(zip(western.network_code, western.station_code))
    absent = wc[[(n, s) not in have for n, s in zip(wc.net, wc.sta)]].copy()
    polys = regions()
    absent["region"] = [where(polys, la, lo) for la, lo in zip(absent.Lat, absent.Lon)]
    land = absent[absent.region != "offshore"]
    print(f"list {len(wc):,} stations; in western {len(wc) - len(absent):,}; absent {len(absent):,}, "
          f"of which offshore (obs) {int((absent.region == 'offshore').sum()):,}, land {len(land):,}")
    print(land.region.value_counts().to_string())

    df, missing = fetch(list(zip(land.net, land.sta)))
    region_of = {(n, s): r for n, s, r in zip(land.net, land.sta, land.region)}
    df["state"] = [region_of.get((n, s)) for n, s in zip(df.network_code, df.station_code)]
    df = df[~df.id.isin(set(western.id))]
    print(f"\nFDSN: {len(df):,} station-locations on {df.station_code.nunique():,} stations; "
          f"{len(missing)} stations unknown to FDSN: {sorted(missing)[:10]}")
    print(df.groupby("state").size().to_string())

    out = Path("sb_catalog/configs/networks/western_fill.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df):,} rows)")
    if a.dry_run:
        from sb_catalog.src.shard_planner import plan, parse_year_day
        shards = plan(df, parse_year_day(START), parse_year_day(END))
        print(f"dry run: {len(shards):,} shards, {sum(s['n_station_days'] for s in shards):,} station-days "
              f"over {START}..{END}")
        return

    from sb_catalog.src.s3_state import S3CampaignState
    from sb_catalog.src.shard_planner import plan, parse_year_day
    state = S3CampaignState(QUEUE)
    state.write_stations(df)
    shards = plan(state.get_stations(), parse_year_day(START), parse_year_day(END))
    print(f"{len(shards):,} shards, {sum(s['n_station_days'] for s in shards):,} station-days")
    print(state.write_shards(shards))
    import boto3, json
    boto3.client("s3", region_name="us-east-2").put_object(
        Bucket=BUCKET, Key="_queues/western-fill/README.json", ContentType="application/json",
        Body=json.dumps(dict(purpose="stations of WestCoast_stations.txt outside the six western state polygons "
                                     "(UT MT AZ CO NM, BC, AB, Baja California); docs/rerun_2026/29 and this script",
                             parquet_uri=f"s3://{BUCKET}/{CATALOGUE}", span=[START, END],
                             stations=int(len(df)), shards=len(shards),
                             station_days=int(sum(s["n_station_days"] for s in shards)),
                             planned=datetime.datetime.utcnow().isoformat() + "Z"), indent=1).encode())


if __name__ == "__main__":
    main()
