#!/usr/bin/env python
"""Select offshore stations per station, not per network code.

The obs campaign was planned from 26 network codes
(sb_catalog/configs/networks/earthscope_offshore.txt). FDSN temporary codes
are year-scoped and reused, so the codes swept in every land experiment that
ever carried them: 1,992 of the 3,389 stations in obs/stations.parquet are on
land (docs/rerun_2026/31_obs_station_selection.md). This script replaces the
code list with a station list.

A station is **offshore** when it is outside the Natural Earth 10 m land
polygons AND its elevation is at or below -10 m. Both halves are needed:

* elevation alone calls the Salton Trough (-60 m), Dead Sea nodes (-214 m)
  and boreholes (IU, ZL) offshore;
* the mask alone misses small islands (Johnston, Midway, Wake, Kwajalein),
  and ice shelves (Amery, Ross) are not land to Natural Earth, so an
  elevation of exactly 0 or a few metres off the mask is an island gauge or
  an ice station, not a seafloor instrument. The -10 m floor puts those on a
  review list instead of in the campaign.

Inputs are the three catalogue tables (obs/, global/, western/) read from the
bucket, and optionally the FDSN station services (--fdsn) to find offshore
stations in mapped networks that no table holds.

    pixi run python scripts/select_offshore_stations.py --write
    pixi run python scripts/select_offshore_stations.py --write --fdsn

Writes sb_catalog/configs/networks/offshore_stations.csv: one row per
station-location in any table, with `on_land`, `coast_km`, `offshore_class`
(offshore / offshore_review_shallow / land), `source_tables` and `action`
(keep-in-obs / return-to-global / fill-from-global / none), and
global_onshore_from_obs.csv: the return-to-global rows in the catalogue
table schema, to be planned with the land stations when global is next
launched. With --fdsn, also offshore_stations_absent.csv for stations in
NETWORK_MAPPING that are in no table. Needs cartopy + shapely (the pixi
environment).
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUCKET = "quakescope-picks-2026"
REGION = "us-east-2"
CATALOGUES = ("obs", "global", "western")
OUT = ROOT / "sb_catalog/configs/networks/offshore_stations.csv"
OUT_ABSENT = ROOT / "sb_catalog/configs/networks/offshore_stations_absent.csv"
OUT_RETURN = ROOT / "sb_catalog/configs/networks/global_onshore_from_obs.csv"
TABLE_COLS = ["id", "network_code", "station_code", "location_code", "channels",
              "latitude", "longitude", "elevation", "start_date", "end_date"]
OBS_SPAN = (pd.Timestamp("1993-01-01"), pd.Timestamp("2026-10-01"))
DEPTH_FLOOR = -10.0          # metres; shallower goes to review, not to the campaign

# Decisions taken by a person on the review list, keyed by station id, value
# "offshore" or "land". M. Denolle, 2026-10-03: cabled observatories (Ocean
# Networks Canada `NV`, OOI `OO`) stay in the OBS track whatever their depth,
# so the Saanich Inlet node at -8 m is offshore. YN.PARE.02 at 0 m in Punta
# Arenas is land by the rule and needs no entry.
REVIEWED: dict[str, str] = {
    "NV.NSMTC.B1": "offshore",
    "NV.NSMTC.B2": "offshore",
    "NV.NSMTC.B3": "offshore",
}
CABLED_NETWORKS = {"NV", "OO"}   # any station of these off the land mask is offshore
FDSN = {
    "iris": "https://service.iris.edu/fdsnws/station/1/query?level=station&format=text&nodata=404",
    "ncedc": "https://service.ncedc.org/fdsnws/station/1/query?level=station&format=text&nodata=404",
    "scedc": "https://service.scedc.caltech.edu/fdsnws/station/1/query?level=station&format=text&nodata=404",
}


def read_table(cat: str) -> pd.DataFrame:
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    body = s3.get_object(Bucket=BUCKET, Key=f"{cat}/stations.parquet")["Body"].read()
    d = pd.read_parquet(io.BytesIO(body))
    d["catalogue"] = cat
    return d


class LandMask:
    """Point-in-polygon against Natural Earth 10 m land, plus distance to its boundary."""

    def __init__(self):
        import cartopy.io.shapereader as shpreader
        from shapely.prepared import prep
        from shapely.strtree import STRtree
        path = shpreader.natural_earth(resolution="10m", category="physical", name="land")
        self.geoms = list(shpreader.Reader(path).geometries())
        self.tree = STRtree(self.geoms)
        self.prepared = [prep(g) for g in self.geoms]

    def classify(self, lats, lons):
        from shapely.geometry import Point
        on_land = np.zeros(len(lats), bool)
        coast_km = np.full(len(lats), np.nan)
        for i, (la, lo) in enumerate(zip(lats, lons)):
            p = Point(lo, la)
            on_land[i] = any(self.prepared[j].contains(p) for j in self.tree.query(p))
            near = self.tree.query(p.buffer(2.0))
            if len(near):
                coast_km[i] = min(self.geoms[j].boundary.distance(p) for j in near) * 111.0
        return on_land, coast_km


def offshore_class(on_land, elevation) -> np.ndarray:
    off = ~np.asarray(on_land, bool)
    elev = np.asarray(elevation, float)
    return np.where(off & (elev <= DEPTH_FLOOR), "offshore",
                    np.where(off & (elev <= 0), "offshore_review_shallow", "land"))


def days_in(df: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> pd.Series:
    s = pd.to_datetime(df["start_date"].astype(str).str[:10], errors="coerce")
    e = pd.to_datetime(df["end_date"].astype(str).str.replace(r"^3000.*", "2100-01-01", regex=True)
                       .str[:10], errors="coerce")
    return ((e.clip(upper=b) - s.clip(lower=a)).dt.days + 1).clip(lower=0).fillna(0).astype(int)


def has_vertical(channels) -> bool:
    return any(c.strip().endswith("Z") for c in str(channels).split(","))


def tables(mask: LandMask) -> pd.DataFrame:
    u = pd.concat([read_table(c) for c in CATALOGUES], ignore_index=True)
    u["source_tables"] = u.groupby("id")["catalogue"].transform(lambda s: "+".join(sorted(set(s))))
    u = u.drop_duplicates("id").drop(columns=["catalogue", "state"], errors="ignore").reset_index(drop=True)
    u["on_land"], u["coast_km"] = mask.classify(u["latitude"].values, u["longitude"].values)
    u["offshore_class"] = offshore_class(u["on_land"], u["elevation"])
    cabled = u["network_code"].isin(CABLED_NETWORKS) & ~u["on_land"]
    u.loc[cabled, "offshore_class"] = "offshore"
    decided = u["id"].map(REVIEWED)
    u.loc[decided.notna(), "offshore_class"] = decided[decided.notna()]
    u["has_vertical"] = u["channels"].map(has_vertical)
    u["days_1993_2026"] = days_in(u, *OBS_SPAN)
    in_obs = u["source_tables"].str.contains("obs")
    offshore = u["offshore_class"] == "offshore"
    # return-to-global: land stations the obs campaign picked under an offshore
    # code. Their obs picks are archived, not re-picked with jma_wc now; the
    # next global campaign (new picker) must plan them with the land stations.
    u["action"] = np.select(
        [in_obs & offshore, in_obs & ~offshore, ~in_obs & offshore & u["has_vertical"]],
        ["keep-in-obs", "return-to-global", "fill-from-global"], default="none")
    return u


def absent_from_tables(u: pd.DataFrame, mask: LandMask) -> pd.DataFrame:
    """Offshore stations in NETWORK_MAPPING networks that no catalogue table lists."""
    from sb_catalog.src.constants import NETWORK_MAPPING
    frames = []
    for name, url in FDSN.items():
        with urllib.request.urlopen(url, timeout=300) as r:
            body = r.read().decode("utf-8", "replace")
        d = pd.read_csv(io.StringIO(body), sep="|", dtype=str)
        d.columns = [c.strip().lstrip("#").strip() for c in d.columns]
        d["service"] = name
        frames.append(d)
    f = pd.concat(frames, ignore_index=True)
    for c in ("Latitude", "Longitude", "Elevation"):
        f[c] = f[c].astype(float)
    f = f[f["Network"].isin(NETWORK_MAPPING)]
    have = set(zip(u["network_code"], u["station_code"]))
    f = f[[(n, s) not in have for n, s in zip(f["Network"], f["Station"])]].copy()
    f = f[f["Elevation"] <= 0]
    f["on_land"], f["coast_km"] = mask.classify(f["Latitude"].values, f["Longitude"].values)
    f["offshore_class"] = offshore_class(f["on_land"], f["Elevation"])
    f = f.rename(columns={"StartTime": "start_date", "EndTime": "end_date"})
    f["days_1993_2026"] = days_in(f, *OBS_SPAN)
    f = f[(f["offshore_class"] != "land") & (f["days_1993_2026"] > 0)]
    f["archive"] = f["Network"].map(NETWORK_MAPPING)
    return f.sort_values(["offshore_class", "days_1993_2026"], ascending=[True, False])


def report(u: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("station-locations by class and source table (days = operating days 1993-2026):")
    print(u.groupby(["offshore_class", "source_tables"])
           .agg(n=("id", "size"), days=("days_1993_2026", "sum")).to_string())
    print("\nactions:")
    print(u.groupby("action").agg(n=("id", "size"), days=("days_1993_2026", "sum")).to_string())
    rm = u[u["action"] == "return-to-global"]
    print(f"\nreturn-to-global by network: {rm['network_code'].value_counts().to_dict()}")
    rv = u[(u["offshore_class"] == "offshore_review_shallow") & u["source_tables"].str.contains("obs")]
    if len(rv):
        print("\nobs-table stations needing a human decision (off the mask, -10 m < elevation <= 0):")
        print(rv[["id", "latitude", "longitude", "elevation", "coast_km", "channels",
                  "start_date", "end_date"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the CSVs; otherwise report only")
    ap.add_argument("--fdsn", action="store_true", help="also sweep the FDSN station services for "
                                                       "offshore stations absent from every table")
    a = ap.parse_args()
    mask = LandMask()
    u = tables(mask)
    report(u)
    if a.write:
        cols = ["id", "network_code", "station_code", "location_code", "channels", "latitude",
                "longitude", "elevation", "start_date", "end_date", "on_land", "coast_km",
                "offshore_class", "has_vertical", "days_1993_2026", "source_tables", "action"]
        u[cols].sort_values(["action", "network_code", "station_code"]).to_csv(OUT, index=False)
        print(f"\nwrote {OUT.relative_to(ROOT)} ({len(u):,} rows)")
        back = u[u["action"] == "return-to-global"]
        back[TABLE_COLS].sort_values(["network_code", "station_code"]).to_csv(OUT_RETURN, index=False)
        print(f"wrote {OUT_RETURN.relative_to(ROOT)} ({len(back):,} rows): land stations the obs "
              f"campaign picked, for the next global onshore plan")
    if a.fdsn:
        f = absent_from_tables(u, mask)
        print(f"\noffshore stations in mapped networks absent from every table: {len(f):,}, "
              f"{int(f['days_1993_2026'].sum()):,} operating days 1993-2026")
        print(f.groupby(["offshore_class", "Network"]).agg(n=("Station", "size"), days=("days_1993_2026", "sum"))
               .sort_values("days", ascending=False).head(30).to_string())
        if a.write:
            f.to_csv(OUT_ABSENT, index=False)
            print(f"wrote {OUT_ABSENT.relative_to(ROOT)} ({len(f):,} rows); channel metadata is "
                  f"not in a station-level listing, so pickability must be checked per network")


if __name__ == "__main__":
    main()
