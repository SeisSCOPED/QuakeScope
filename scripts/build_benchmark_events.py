"""Event locations and station locations for every benchmark sequence.

The map needs a point per earthquake and a point per station. The held-out
build already stores a catalogue with coordinates; the swarm and western
sequences store only event ids, so those are fetched from the service that
catalogued them. Stations carry no coordinates anywhere and are resolved
against FDSN.
"""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd                                                # noqa: E402
from obspy import UTCDateTime                                      # noqa: E402
from obspy.clients.fdsn import Client, RoutingClient               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "benchmark" / "results"
RETRAIN = Path("/Users/marinedenolle/GitHub/phasenet-retrain/data/heldout_testset")
OUT = RES / "map"

TRACK_OF = {
    "Ridgecrest": "track1-western-us", "San Simeon": "track1-western-us",
    "Monte Cristo": "track1-western-us", "Mendocino 2024": "track1-western-us",
    "Monroe WA": "track1-western-us",
    "Kaikoura 2016": "track2-msas", "Norcia 2016": "track2-msas",
    "Thessaly 2021": "track2-msas",
    "Etna edifice 2022-2024": "track2-vt",
    "West Bohemia 2018": "track2-swarm", "Maurienne 2017-2019": "track2-swarm",
    "Salton Sea 2016": "track2-swarm", "Jones-Guthrie 2014-2015": "track2-swarm",
}
# which service catalogued each sequence, for the events we have only ids for
SERVICE = {"West Bohemia 2018": "ISC", "Maurienne 2017-2019": "ISC",
           "Salton Sea 2016": "USGS", "Jones-Guthrie 2014-2015": "USGS",
           "Etna edifice 2022-2024": "INGV"}
_cl: dict[str, Client] = {}


def cl(n: str) -> Client:
    if n not in _cl:
        _cl[n] = Client(n, timeout=120)
    return _cl[n]


def events_from_retrain() -> pd.DataFrame:
    rows = []
    for key, label in (("kaikoura_2016", "Kaikoura 2016"), ("norcia_2016", "Norcia 2016"),
                       ("thessaly_2021", "Thessaly 2021")):
        f = RETRAIN / key / "catalog.parquet"
        if not f.exists():
            continue
        c = pd.read_parquet(f)
        c["sequence"] = label
        rows.append(c.rename(columns={"depth_km": "depth"})[
            ["sequence", "event", "origin", "lat", "lon", "depth", "mag"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def events_by_id(sequence: str, ids: list[str]) -> pd.DataFrame:
    """One catalogue lookup per event, for sequences that stored only an id."""
    svc = SERVICE[sequence]

    # The harvest stored ids with the network prefix stripped: "10002lwf" where
    # ComCat wants "us10002lwf", "37701512" where it wants "ci37701512". Try the
    # id as stored and then the prefixes the contributing networks use.
    PREFIX = ("", "us", "ci", "nc", "ok", "nn", "uw", "av")

    def usgs_one(eid):
        """ComCat's detail endpoint. obspy's eventid path returns nothing here."""
        import json as _j
        import urllib.request as _u
        for pre in PREFIX:
            try:
                d = _j.load(_u.urlopen(
                    f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={pre}{eid}"
                    f"&format=geojson", timeout=60))
                lon, lat, dep = d["geometry"]["coordinates"]
                pr = d["properties"]
                return dict(sequence=sequence, event=f"{pre}{eid}",
                            origin=pd.to_datetime(pr["time"], unit="ms"),
                            lat=lat, lon=lon, depth=dep, mag=pr.get("mag"))
            except Exception:                                      # noqa: BLE001
                continue
        return None

    def one(eid):
        if svc == "USGS":
            return usgs_one(eid)
        try:
            ev = cl(svc).get_events(eventid=str(eid))[0]
            o = ev.preferred_origin() or ev.origins[0]
            m = ev.magnitudes[0].mag if ev.magnitudes else None
            return dict(sequence=sequence, event=eid, origin=o.time.datetime,
                        lat=o.latitude, lon=o.longitude,
                        depth=(o.depth or 0) / 1000.0, mag=m)
        except Exception:                                          # noqa: BLE001
            return None
    with ThreadPoolExecutor(8) as ex:
        got = [r for r in ex.map(one, ids) if r]
    print(f"  {sequence}: {len(got)}/{len(ids)} events located", flush=True)
    return pd.DataFrame(got)


# One region query per scoring window beats one lookup per event for every
# service here: ComCat answers 429 after a few hundred per-event requests, and
# ISC throttles sooner than that.
SWARM_REGION = {
    "West Bohemia 2018": (50.22, 12.45, 0.35),
    "Maurienne 2017-2019": (45.22, 6.50, 0.35),
    "Salton Sea 2016": (33.28, -115.60, 0.30),
    "Jones-Guthrie 2014-2015": (35.80, -97.30, 0.35),
}


def events_in_windows(sequence: str, wins: pd.DataFrame) -> pd.DataFrame:
    """Origins inside each scoring window, by region query."""
    lat, lon, rad = SWARM_REGION[sequence]
    svc = SERVICE.get(sequence, "ISC")
    rows = []
    if svc == "USGS":
        # obspy re-discovers the service on every client and ComCat answers 429
        # to that too once it is annoyed. The geojson endpoint is one plain GET.
        import json as _j
        import urllib.request as _u
        for _, w in wins.iterrows():
            url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
                   f"&starttime={w.start.isoformat()}&endtime={w.end.isoformat()}"
                   f"&latitude={lat}&longitude={lon}&maxradiuskm={rad * 111.2:.1f}")
            try:
                d = _j.load(_u.urlopen(url, timeout=120))
            except Exception as exc:                               # noqa: BLE001
                print(f"  {sequence} w{int(w.window)}: {type(exc).__name__}", flush=True)
                continue
            for f in d.get("features", []):
                lo, la, de = f["geometry"]["coordinates"]
                rows.append(dict(sequence=sequence, event=f["id"],
                                 origin=pd.to_datetime(f["properties"]["time"], unit="ms"),
                                 lat=la, lon=lo, depth=de,
                                 mag=f["properties"].get("mag")))
        print(f"  {sequence}: {len(rows)} events located from {len(wins)} windows", flush=True)
        return pd.DataFrame(rows)
    for _, w in wins.iterrows():
        try:
            cat = cl(svc).get_events(
                starttime=UTCDateTime(w.start.to_pydatetime()),
                endtime=UTCDateTime(w.end.to_pydatetime()),
                latitude=lat, longitude=lon, maxradius=rad)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {sequence} w{int(w.window)}: {type(exc).__name__}", flush=True)
            continue
        for ev in cat:
            o = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            if o is None or o.latitude is None:
                continue
            rows.append(dict(sequence=sequence, event=str(ev.resource_id).split("/")[-1],
                             origin=o.time.datetime, lat=o.latitude, lon=o.longitude,
                             depth=(o.depth or 0) / 1000.0,
                             mag=ev.magnitudes[0].mag if ev.magnitudes else None))
    print(f"  {sequence}: {len(rows)} events located from {len(wins)} windows", flush=True)
    return pd.DataFrame(rows)


def station_coords(stations: set[str]) -> pd.DataFrame:
    fed = RoutingClient("iris-federator")
    eida = RoutingClient("eida-routing")
    # GeoNet answers neither federator, so it needs its own client.
    extra = [Client(n, timeout=60) for n in ("GEONET", "INGV", "NOA")]

    def one(s):
        net, sta = s.split(".")[0], s.split(".")[1]
        for c in [fed, eida] + extra:
            try:
                inv = c.get_stations(network=net, station=sta, level="station")
                for n in inv:
                    for st in n:
                        return dict(station=s, lat=st.latitude, lon=st.longitude,
                                    elev=st.elevation, site=st.site.name or "")
            except Exception:                                      # noqa: BLE001
                continue
        return None
    with ThreadPoolExecutor(10) as ex:
        got = [r for r in ex.map(one, sorted(stations)) if r]
    print(f"  stations located: {len(got)}/{len(stations)}", flush=True)
    return pd.DataFrame(got)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else None

    print("events")
    # Cache each sequence once it lands. ComCat throttles a run that asks for
    # 158 events several times over, so a re-run must not re-ask for what it
    # already has.
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def cached(seq: str, fetch):
        f = cache / (seq.replace(" ", "_").replace("/", "_") + ".csv")
        if f.exists():
            d = pd.read_csv(f)
            if len(d):
                print(f"  {seq}: {len(d)} events from cache", flush=True)
                return d
        d = fetch()
        if d is not None and len(d):
            d.to_csv(f, index=False)
        return d if d is not None else pd.DataFrame()

    frames = [events_from_retrain()]

    raw = RES / "swarm_sequences" / "reference_picks_raw.csv"
    wins = pd.read_csv(RES / "swarm_sequences" / "windows.csv", parse_dates=["start", "end"])
    if raw.exists():
        r = pd.read_csv(raw)
        for seq, g in r.groupby("sequence"):
            if seq in SWARM_REGION:
                # 334 per-event lookups against ISC get throttled and mostly
                # fail. One region query per scoring window returns the same
                # origins in a handful of requests.
                frames.append(cached(seq, lambda s_=seq: events_in_windows(
                    s_, wins[wins.sequence == s_])))
            else:
                ids = sorted({str(x) for x in g.event.dropna().unique()})
                frames.append(cached(seq, lambda s_=seq, i_=ids: events_by_id(s_, i_)))

    # Western sequences: the mainshock is in meta.json, but the map should show
    # the sequence that was actually scored, so take every catalogued event in
    # each scoring window as well.
    meta = json.loads((RES / "us_sequences" / "meta.json").read_text())
    import json as _j
    import urllib.request as _u
    US_WIN = {"Ridgecrest": 30, "San Simeon": 120, "Monte Cristo": 120,
              "Mendocino 2024": 120, "Monroe WA": 120}
    US_OFF = {"Monroe WA": -60}

    def us_events(name, v):
        rows = [dict(sequence=name, event=f"{name} mainshock", origin=v["time"],
                     lat=v["lat"], lon=v["lon"], depth=v.get("depth", 0) or 0,
                     mag=v.get("mag"))]
        t0 = pd.Timestamp(v["time"]).tz_localize(None) + pd.Timedelta(
            seconds=US_OFF.get(name, 600))
        t1 = t0 + pd.Timedelta(minutes=US_WIN.get(name, 120))
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
               f"&starttime={t0.isoformat()}&endtime={t1.isoformat()}"
               f"&latitude={v['lat']}&longitude={v['lon']}&maxradiuskm=120")
        try:
            d = _j.load(_u.urlopen(url, timeout=120))
            for f in d.get("features", []):
                lo, la, de = f["geometry"]["coordinates"]
                rows.append(dict(sequence=name, event=f["id"],
                                 origin=pd.to_datetime(f["properties"]["time"], unit="ms"),
                                 lat=la, lon=lo, depth=de, mag=f["properties"].get("mag")))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {name}: window events {type(exc).__name__}", flush=True)
        print(f"  {name}: {len(rows)} events", flush=True)
        return pd.DataFrame(rows)

    for k, v in meta["sequences"].items():
        frames.append(cached(k, lambda k_=k, v_=v: us_events(k_, v_)))

    ev = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    ev["track"] = ev.sequence.map(TRACK_OF)
    ev = ev.dropna(subset=["lat", "lon", "track"])
    ev.to_csv(OUT / "events.csv", index=False)
    print(f"  {len(ev):,} located events -> {OUT/'events.csv'}")

    print("stations")
    stas, rows = set(), []
    for t in sorted(p.name for p in bundle.glob("track*")) if bundle else []:
        d = pd.read_csv(bundle / t / "reference_picks.csv")
        for (seq, s), g in d.groupby(["sequence", "station"]):
            rows.append(dict(track=t, sequence=seq, station=s, arrivals=len(g),
                             P=int((g.phase == "P").sum()), S=int((g.phase == "S").sum())))
            stas.add(s)
    su = pd.DataFrame(rows)
    co = station_coords(stas)
    su = su.merge(co, on="station", how="left")
    su.to_csv(OUT / "stations.csv", index=False)
    print(f"  {len(su):,} station-sequence rows -> {OUT/'stations.csv'}")


if __name__ == "__main__":
    main()
