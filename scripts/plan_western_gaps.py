#!/usr/bin/env python
"""Plan `western-gaps`: the wc2025 station-days the western catalogue lacks and can read.

Input is the comparison Junhao Song made on 2026-11-10 between the 2025 West
Coast picks (wc2025) and western/ + obs/ (wc2026), over 134..108 W, 22..53 N,
2002-01-01..2025-03-31: `missing_station_days.parquet`, one row per station-day
with picks in wc2025 and none in wc2026, and the reason he assigned.

Two of his reasons are planned here; the others need a decision or new code
(docs/rerun_2026/35_western_gaps.md):

* "station not in wc2026" - stations of the stakeholder list that are in no
  catalogue. Both station selections classified a point outside every admin-1
  polygon as offshore and left it to the obs campaign; the obs selection then
  rejected it as land (Natural Earth land mask, elevation > -10 m). Islands and
  shoreline stations (NC.CYB, CI.SBI, UW.ALKI, Vancouver Island) fell between
  the two. Planned per epoch inside the western span, from the channel epochs
  of their pickable bands at each network's data center
  (`add_station_epochs.py`).
* "never processed" - days inside a station's epochs with no outcome
  (availability status `unknown`). Planned as windows around exactly those
  days. Excluded: networks the account cannot read or the western queues never
  carried (`--skip-networks`, default TD and NP), and station-locations with no
  band the picker uses (EP, EL, SL only; dropped 2026-09-02).

    python scripts/plan_western_gaps.py --missing missing_station_days.parquet \
        --list ~/Downloads/WestCoast_stations.txt --dry-run
    python scripts/plan_western_gaps.py ... --write

`--write` installs the station table and the queue under _queues/western-gaps/.
The campaign then needs its fleet.json entry, an access survey and a target.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_station_epochs as ase
import plan_western_fill as pwf

from sb_catalog.src.constants import channel_priority
from sb_catalog.src.shard_planner import parse_year_day, plan, shard_id
from sb_catalog.src.utils import normalize_station_codes

BUCKET = "quakescope-picks-2026"
QUEUE = f"s3://{BUCKET}/_queues/western-gaps"
CATALOGUE = "western"
START, END = pwf.START, pwf.END
WEIGHT = "original"


def pickable(channels: str) -> bool:
    bands = set(str(channels).split(","))
    return any(b in bands for b in channel_priority(WEIGHT))


def absent_stations(missing: pd.DataFrame, wc_list: str, western: pd.DataFrame) -> pd.DataFrame:
    """Station-locations of the stakeholder list that no catalogue holds."""
    want = sorted(set(missing.loc[missing.reason == "station not in wc2026", "station"]))
    wc = pd.read_csv(wc_list, skipinitialspace=True)
    wc.columns = [c.strip() for c in wc.columns]
    listed = set(wc["Ntw.Code"].str.strip())
    off_list = [s for s in want if s not in listed]
    if off_list:
        print(f"  {len(off_list)} not on the stakeholder list, left out: {off_list[:10]}")
    by_net = {}
    for s in want:
        if s in listed:
            n, st = s.split(".")
            by_net.setdefault(n, []).append(st)
    bands = set(channel_priority(WEIGHT))
    rows = []
    for net, stas in sorted(by_net.items()):
        text = ase.fetch(net, sorted(stas))
        epochs = ase.epochs_from_text(text, bands)
        meta = {}
        for line in text.splitlines():
            f = line.split("|")
            if line.startswith("#") or len(f) < 17 or f[3][:2] not in bands:
                continue
            m = meta.setdefault(f"{f[0]}.{f[1]}.{f[2]}", {"bands": set(), "lat": float(f[4]),
                                                          "lon": float(f[5]), "elev": float(f[6] or 0)})
            m["bands"].add(f[3][:2])
        for sid, w in epochs.items():
            n, st, loc = sid.split(".")
            rows.append({"id": sid, "network_code": n, "station_code": st, "location_code": loc,
                         "channels": ",".join(sorted(meta[sid]["bands"])),
                         "latitude": meta[sid]["lat"], "longitude": meta[sid]["lon"],
                         "elevation": meta[sid]["elev"],
                         "start_date": w[0][0], "end_date": w[-1][1],
                         "epochs": ";".join(f"{a}/{b}" for a, b in w)})
    df = pd.DataFrame(rows, columns=["id", "network_code", "station_code", "location_code", "channels",
                                     "latitude", "longitude", "elevation", "start_date", "end_date",
                                     "epochs"])
    got = set(zip(df.network_code, df.station_code))
    unknown = [(n, s) for n, ss in by_net.items() for s in ss if (n, s) not in got]
    if unknown:
        nets = pd.Series([n for n, _ in unknown]).value_counts().to_dict()
        print(f"  {len(unknown)} stations with no pickable vertical channel at their data center, "
              f"left out: {nets}")
    df = df[~df.id.isin(set(western.id))]
    df["source"] = "absent"
    print(f"absent: {df[['network_code', 'station_code']].drop_duplicates().shape[0]} stations, "
          f"{len(df)} station-locations")
    return df


def hole_shards(missing: pd.DataFrame, western: pd.DataFrame, skip: set[str]) -> tuple[pd.DataFrame, list[dict]]:
    """One shard per network and contiguous run of unprocessed days."""
    m = missing[missing.reason == "never processed"].copy()
    m["net"] = m.station.str.split(".").str[0]
    n0 = len(m)
    m = m[~m.net.isin(skip)]
    print(f"never processed: {n0:,} station-days; {n0 - len(m):,} in skipped networks {sorted(skip)}")
    rows = []
    for st, d, tids in zip(m.station, m.date, m.wc2026_trace_ids.fillna("")):
        for tid in filter(None, str(tids).split(",")):
            rows.append((tid, st, d))
    h = pd.DataFrame(rows, columns=["id", "station", "date"])
    tab = western.set_index("id")
    h = h[h.id.isin(tab.index)]
    h["channels"] = h.id.map(tab.channels)
    keep = h.channels.map(pickable)
    lost = h[~keep].groupby("station").size()
    print(f"  station-locations with no band the picker uses: {(~keep).sum():,} location-days "
          f"on {lost.size} stations, left out")
    h = h[keep]
    h["date"] = pd.to_datetime(h.date).dt.date
    # Contiguous runs per station-location, cut at 20 days as the production
    # planner cuts, then one shard per (network, window) of up to 40 ids.
    runs = {}
    for sid, g in h.groupby("id"):
        days = sorted(set(g.date))
        s = p = days[0]
        for d in days[1:] + [None]:
            if d is not None and (d - p).days == 1:
                p = d
                continue
            for k in range(0, (p - s).days + 1, 20):
                a0 = s + datetime.timedelta(days=k)
                a1 = min(p, a0 + datetime.timedelta(days=19))
                runs.setdefault((sid.split(".")[0], a0, a1), []).append(sid)
            if d is not None:
                s = p = d
    shards = []
    for (net, d0, last), sids in sorted(runs.items()):
        d1 = last + datetime.timedelta(days=1)                   # end exclusive
        sids = sorted(sids)
        for i in range(0, len(sids), 40):
            grp = sids[i: i + 40]
            shards.append({"shard_id": shard_id(grp, d0, d1), "stations": grp,
                           "start": f"{d0:%Y.%j}", "end": f"{d1:%Y.%j}",
                           "n_station_days": len(grp) * (d1 - d0).days})
    st = western[western.id.isin(set(h.id))].copy()
    st["source"] = "hole"
    st["epochs"] = None              # holes are planned from the day lists, not the table
    print(f"holes: {h.station.nunique():,} stations, {h[['station', 'date']].drop_duplicates().shape[0]:,} "
          f"station-days, {len(h):,} location-days, {len(shards):,} shards")
    return st, shards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missing", required=True, help="missing_station_days.parquet")
    ap.add_argument("--list", required=True, help="WestCoast_stations.txt")
    ap.add_argument("--skip-networks", default="TD,NP")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    missing = pd.read_parquet(a.missing)
    western = pd.read_parquet(f"s3://{BUCKET}/{CATALOGUE}/stations.parquet")
    absent = absent_stations(missing, a.list, western)
    holes, h_shards = hole_shards(missing, western, {n.strip() for n in a.skip_networks.split(",") if n.strip()})

    a_shards = plan(absent, parse_year_day(START), parse_year_day(END))
    print(f"absent: {len(a_shards):,} shards, {sum(s['n_station_days'] for s in a_shards):,} "
          f"planned station-location-days over {START}..{END}")
    shards = a_shards + h_shards
    ids = [s["shard_id"] for s in shards]
    assert len(ids) == len(set(ids)), "duplicate shard ids"
    table = normalize_station_codes(pd.concat([absent, holes], ignore_index=True)
                                    .drop_duplicates("id"))
    planned = sum(s["n_station_days"] for s in shards)
    print(f"TOTAL: {len(table):,} station-locations, {len(shards):,} shards, {planned:,} planned "
          f"station-location-days")
    out = Path("sb_catalog/configs/networks/western_gaps.csv")
    table.to_csv(out, index=False)
    print(f"wrote {out}")
    if a.dry_run:
        return

    from sb_catalog.src.s3_state import S3CampaignState
    state = S3CampaignState(QUEUE)
    state.write_stations(table)
    print(state.write_shards(shards))
    import boto3
    boto3.client("s3", region_name="us-east-2").put_object(
        Bucket=BUCKET, Key="_queues/western-gaps/README.json", ContentType="application/json",
        Body=json.dumps({"purpose": "wc2025 station-days missing from western: stations absent from every "
                                    "catalogue (shoreline/island) and never-processed days; "
                                    "scripts/plan_western_gaps.py, docs/rerun_2026/35_western_gaps.md",
                         "parquet_uri": f"s3://{BUCKET}/{CATALOGUE}", "span": [START, END],
                         "station_locations": len(table), "shards": len(shards),
                         "planned_station_location_days": int(planned),
                         "planned": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                        indent=1).encode())


if __name__ == "__main__":
    main()
