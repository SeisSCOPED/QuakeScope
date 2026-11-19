"""Add stations that have picks but are absent from PNWStationXML.tar, from the IRIS/EarthScope FDSN station service."""
import sys, json, glob, io, datetime as dt, urllib.request, pandas as pd
out = sys.argv[1]
st = pd.read_parquet(out + "/stations_all.parquet")
tids = set()
for f in glob.glob(out + "/manifests/*.json"):
    tids.update(r["tid"] for r in json.load(open(f))["records"])
miss = sorted(tids - set(st.id))
rows, notfound = [], []
for tid in miss:
    n, s, l = tid.split(".")
    url = f"https://service.iris.edu/fdsnws/station/1/query?net={n}&sta={s}&loc={l or '--'}&level=channel&format=text"
    try:
        txt = urllib.request.urlopen(url, timeout=60).read().decode()
    except Exception as e:
        notfound.append((tid, str(e)[:40])); continue
    d = pd.read_csv(io.StringIO(txt), sep="|"); d.columns = [c.strip("# ").strip() for c in d.columns]
    sd = dt.date.fromisoformat(d.StartTime.str[:10].min())
    ed = dt.date.fromisoformat(d.EndTime.fillna("3000-01-01").replace("", "3000-01-01").str[:10].max())
    rows.append({"id": tid, "network_code": n, "station_code": s, "location_code": l,
                 "channels": ",".join(sorted(set(d.Channel.str[:2]))),
                 "latitude": float(d.Latitude.iloc[-1]), "longitude": float(d.Longitude.iloc[-1]),
                 "elevation": float(d.Elevation.iloc[-1]), "start_date": sd.isoformat(), "end_date": ed.isoformat(),
                 "state": None, "start_yearday": float(f"{sd.year}.{sd.timetuple().tm_yday:03d}"),
                 "end_yearday": float(f"{ed.year}.{ed.timetuple().tm_yday:03d}")})
filled = {r["id"] for r in rows}
st2 = pd.concat([st, pd.DataFrame(rows)], ignore_index=True).sort_values("id").reset_index(drop=True)
st2["metadata_source"] = ["fdsnws IRIS/EarthScope 2026-11-13" if i in filled else "PNWStationXML.tar 2022-01" for i in st2.id]
st2.to_parquet(out + "/stations.parquet", index=False)
print(f"missing {len(miss)}, filled {len(rows)}, not found {notfound}")
print(f"rows {len(st2)}, unique {st2.id.nunique()}, stations with picks still missing {len(tids - set(st2.id))}")
