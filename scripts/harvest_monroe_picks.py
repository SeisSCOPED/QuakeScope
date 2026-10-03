"""Monroe WA 2019 analyst arrivals, via the USGS phase-data product.

us_sequences asked SCEDC and NCEDC, which hold no arrivals for a Washington
event, and recorded zero. PNSN's picks reach ANSS and are served per event by
the phase-data product, the same path the Oklahoma and Salton Sea sequences use.
"""
import io, json, urllib.request
from collections import Counter
import pandas as pd
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn import Client

SP = "/private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8f282710-68a1-4cdd-b77e-e07856ef31af/scratchpad"
LAT, LON = 47.86, -121.93
# ComCat origin for uw61535372, in UTC. An earlier version of this script
# carried 14:51, five hours late, which put the scored window after the
# mainshock and its 128 manual arrivals and left the reference with one pick.
T0 = UTCDateTime("2019-07-12T09:51:38")
cat = Client("USGS", timeout=180).get_events(
    starttime=T0 - 600, endtime=T0 + 86400 * 7, latitude=LAT, longitude=LON,
    maxradius=0.8, minmagnitude=0.0)
print(f"{len(cat)} ComCat events")
rows = []
for ev in cat:
    eid = str(ev.resource_id).split("eventid=")[-1].split("&")[0].split("/")[-1]
    try:
        d = json.load(urllib.request.urlopen(
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={eid}&format=geojson",
            timeout=90))
        prods = d["properties"]["products"].get("phase-data", [])
        if not prods:
            continue
        cont = prods[0]["contents"]
        qml = next(k for k in cont if k.endswith("quakeml.xml"))
        full = read_events(io.BytesIO(urllib.request.urlopen(cont[qml]["url"], timeout=90).read()))
    except Exception:
        continue
    for e in full:
        o = e.preferred_origin() or (e.origins[0] if e.origins else None)
        if o is None:
            continue
        picks = {p.resource_id.id: p for p in e.picks}
        for arr in o.arrivals:
            p = picks.get(arr.pick_id.id if arr.pick_id else None)
            if p is None or str(p.evaluation_mode) != "manual":
                continue
            ph = (arr.phase or p.phase_hint or "")[:1].upper()
            if ph not in ("P", "S"):
                continue
            w = p.waveform_id
            rows.append(dict(sequence="Monroe WA", event=eid,
                             station=f"{w.network_code}.{w.station_code}",
                             phase=ph, time=p.time.datetime, origin=o.time.datetime))
df = pd.DataFrame(rows).drop_duplicates(["event", "station", "phase"])
print(f"{len(df):,} manual arrivals, P {int((df.phase=='P').sum())} S {int((df.phase=='S').sum())}, "
      f"{df.station.nunique()} stations, {df.event.nunique()} events")
print("\ntop stations:")
print(df.station.value_counts().head(8).to_string())
df.to_csv(f"{SP}/monroe_picks.csv", index=False)
