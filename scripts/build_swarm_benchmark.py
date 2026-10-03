#!/usr/bin/env python
"""Build the fluid-driven swarm track: reference picks, waveforms, model picks.

Four sequences, chosen because each carries reviewed analyst arrivals on a dense
local network and none is a mainshock-aftershock cascade:

    west_bohemia_2018   CO2-driven swarm, Czechia, WEBNET            ISC bulletin
    maurienne_2017      fluid and slip interplay, French Alps        ISC bulletin
    salton_sea_2016     Brawley seismic zone swarm, California       USGS phase-data
    jones_guthrie_2014  injection-induced swarm, Oklahoma            USGS phase-data

The two European cases are read from the ISC bulletin rather than from the
operator, because the operator paths under-collected: the WEBNET Zenodo file
carries bare station codes that land on one station once resolved, and the
French service returned a tenth of what ISC holds for the same window.

Three stages, each restartable, writing into docs/benchmark/results/swarm_sequences/:

    picks      harvest reference arrivals        -> reference_picks.csv, provenance.csv
    waveforms  pick windows and stations, fetch  -> waveforms/, windows.csv, stations.csv
    models     run the four weight sets          -> model_picks.csv

    pixi run -e dev python scripts/build_swarm_benchmark.py --stage picks
    pixi run -e dev python scripts/build_swarm_benchmark.py --stage waveforms
    pixi run -e dev python scripts/build_swarm_benchmark.py --stage models

The output columns are the ones scripts/score_picks.py reads, so the track scores
with the same scorer as the rest of the board.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime, read_events
from obspy.geodetics import gps2dist_azimuth
from obspy.clients.fdsn import Client, RoutingClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "benchmark" / "results" / "swarm_sequences"
CACHE = OUT / "waveforms"

WEIGHTS = ["quakescope2026", "jma_wc", "original", "instance"]
DETECT_FLOOR = 0.02        # run once low, threshold offline, as the other tracks do
N_STATIONS = 6
ONLY: set[str] = set()      # --only <key>[,<key>] limits the run
TRIM_MARGIN_S = 300     # context kept either side of the outermost arrival
TRIM_IF_UNDER = 0.6     # only trim when it saves most of the window
DEDUP_TOL_S = 0.5   # the scorer's matching tolerance
PER_EVENT_FDSN = {"ingv", "geonet"}   # reject a region query with arrivals
N_WINDOWS = 2              # default; a sequence may override both
WINDOW_H = 3
CHANNEL_PREF = ["HH", "EH", "BH", "CH", "SH"]   # CH is WEBNET's 250 Hz short period

SEQUENCES = {
    "West Bohemia 2018": dict(
        key="west_bohemia_2018", source="isc", sources=["isc"], lat=50.22, lon=12.45, radius=0.35,
        t0="2018-05-10", t1="2018-06-01", min_mag=None,
        note="CO2-driven swarm, WEBNET; ISC carries the agencies' reviewed readings"),
    "Maurienne 2017-2019": dict(
        key="maurienne_2017", source="isc", sources=["franceseisme", "isc"], lat=45.22, lon=6.50, radius=0.35,
        t0="2017-08-01", t1="2019-03-31", min_mag=None, n_windows=3, window_h=6,
        note="fluid and slip interplay, French Alps; ISC over SISmalp/RESIF"),
    "Salton Sea 2016": dict(
        key="salton_sea_2016", source="usgs", sources=["usgs", "isc"], lat=33.28, lon=-115.60, radius=0.30,
        t0="2016-09-26", t1="2016-10-10", min_mag=1.0,
        note="Brawley seismic zone swarm; SCEDC analyst picks through ComCat"),
    "Jones-Guthrie 2014-2015": dict(
        key="jones_guthrie_2014", source="usgs", sources=["usgs", "isc"], lat=35.80, lon=-97.30, radius=0.35,
        t0="2014-01-01", t1="2015-06-30", min_mag=1.5, n_windows=4, window_h=6,
        note="injection-induced swarm, Oklahoma; OGS and USGS analyst picks. The rate "
             "is elevated for months rather than concentrated in a burst, so this one "
             "takes more and longer windows to reach a scorable count"),
    "Etna edifice 2022-2024": dict(
        key="etna_edifice", source="ingv", sources=["ingv"], lat=37.751, lon=14.993,
        radius=0.08, max_depth_km=8.0, t0="2022-01-01", t1="2024-12-31", min_mag=1.5,
        n_windows=4, window_h=12, n_stations=14,
        note="volcano-tectonic, Etna edifice. The held-out build harvested the busiest "
             "windows over the whole region and caught only basement and regional "
             "Sicilian earthquakes: median 22 km offset, 21.6 km depth, zero edifice "
             "events. This one is bounded to 8 km of the summit craters and 8 km depth, "
             "which is the seismicity the regime is about. INGV publishes reviewed "
             "arrivals only for the larger of these: of 269 edifice events, 78 carry "
             "arrivals, and above M1.5 it is 56 of 87, so the floor is 1.5"),
}

# Networks whose waveforms come from a known archive. Anything else is resolved
# against the federator at build time, because ISC reports some stations under a
# placeholder network code.
KNOWN_ROUTE = {
    "CI": "SCEDC", "BK": "NCEDC", "NC": "NCEDC", "NP": "NCEDC",
    "OK": "IRIS", "GS": "IRIS", "N4": "IRIS", "TA": "IRIS", "US": "IRIS", "ZD": "IRIS",
    "WB": "ORFEUS", "FR": "RESIF", "RA": "RESIF", "RD": "RESIF", "CZ": "ORFEUS",
    "GE": "GFZ", "SX": "GFZ", "TH": "GFZ", "BW": "LMU", "OE": "ORFEUS",
    "IV": "INGV", "MN": "INGV", "GU": "INGV",
}

_clients: dict[str, Client] = {}
_fed = None


def client(name: str) -> Client:
    if name not in _clients:
        if name == "franceseisme":          # BCSF-RENASS, no obspy shorthand
            _clients[name] = Client(
                base_url="https://api.franceseisme.fr", timeout=180,
                service_mappings={"event": "https://api.franceseisme.fr/fdsnws/event/1"})
        else:
            _clients[name] = Client(name, timeout=180)
    return _clients[name]


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ stage 1
def _rows_from_events(events, sequence: str, source: str) -> list[dict]:
    """One row per arrival that a human reviewed, or that a bulletin vouches for.

    ISC reports agencies' reviewed readings without an evaluation mode, so an
    unset mode counts there and only there; USGS marks its analyst picks
    `manual` and its detections `automatic`, and only the former are kept.
    """
    rows = []
    for ev in events:
        org = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        if org is None or not org.arrivals:
            continue
        eid = str(ev.resource_id).split("/")[-1].split("=")[-1]
        picks = {p.resource_id.id: p for p in ev.picks}
        for a in org.arrivals:
            p = picks.get(a.pick_id.id if a.pick_id else None)
            if p is None or p.time is None:
                continue
            mode = str(p.evaluation_mode) if p.evaluation_mode else None
            # USGS and BCSF-RENASS mark their analyst picks; ISC relays agencies'
            # reviewed readings with no mode at all, so an unset mode counts only
            # there, and an explicitly automatic one never counts anywhere.
            if mode == "automatic":
                continue
            if source in ("usgs", "franceseisme") and mode != "manual":
                continue
            phase = (a.phase or p.phase_hint or "")[:1].upper()
            if phase not in ("P", "S"):
                continue
            wid = p.waveform_id
            rows.append(dict(
                sequence=sequence, event=eid, source=source,
                station=f"{wid.network_code or '??'}.{wid.station_code}",
                station_code=wid.station_code, network_reported=wid.network_code or "",
                phase=phase, time=p.time.datetime, mode=mode or "unset",
                origin=org.time.datetime,
                mag=float(ev.preferred_magnitude().mag) if ev.preferred_magnitude() else np.nan))
    return rows


def _event_id(ev) -> str:
    """The service's own event id out of a resource identifier.

    Services disagree on spelling: USGS writes eventid, INGV writes eventId.
    A case-sensitive split left the INGV ids as "query?eventId=35112771" and
    every per-event arrival request failed.
    """
    t = str(ev.resource_id)
    low = t.lower()
    if "eventid=" in low:
        return t[low.rindex("eventid=") + len("eventid="):].split("&")[0]
    return t.split("/")[-1]


def catalogue(spec) -> pd.DataFrame:
    """Origin times over the whole span, without arrivals.

    A catalogue query is cheap and an arrival-bearing one is not, so the windows
    are chosen first and only those are harvested. ISC in particular returns a
    payload that fails to parse once a region query with arrivals covers more
    than a few hours.
    """
    src = spec["source"]
    name = {"isc": "ISC", "usgs": "USGS"}.get(src, src.upper())
    kw = dict(latitude=spec["lat"], longitude=spec["lon"], maxradius=spec["radius"])
    if spec["min_mag"] is not None:
        kw["minmagnitude"] = spec["min_mag"]
    if spec.get("max_depth_km") is not None:
        kw["maxdepth"] = spec["max_depth_km"]
    rows, t0, t1 = [], UTCDateTime(spec["t0"]), UTCDateTime(spec["t1"])
    step = 90 * 86400
    a = t0
    while a < t1:
        b = min(a + step, t1)
        try:
            cat = client(name).get_events(starttime=a, endtime=b, **kw)
        except Exception as exc:                                   # noqa: BLE001
            log(f"    catalogue {a.date}..{b.date}: {type(exc).__name__}")
            a = b
            continue
        for ev in cat:
            org = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            if org is None or org.time is None:
                continue
            if spec.get("max_depth_km") is not None and org.depth is not None:
                if org.depth / 1000.0 > spec["max_depth_km"]:
                    continue                    # the service may ignore maxdepth
            rows.append(dict(event=_event_id(ev), time=org.time.datetime))
        a = b
    df = pd.DataFrame(rows).drop_duplicates("event")
    log(f"    catalogue: {len(df):,} events")
    return df


def _isc_query(spec, a: UTCDateTime, b: UTCDateTime):
    return client("ISC").get_events(
        starttime=a, endtime=b, latitude=spec["lat"], longitude=spec["lon"],
        maxradius=spec["radius"], includearrivals=True)


def harvest_isc_window(spec, sequence, start, end, depth: int = 0) -> list[dict]:
    """ISC readings for one scoring window, splitting when the payload will not parse.

    A dense window returns more than the service will serialise, and the failure
    arrives as an XML parse error rather than an HTTP status. Halving the span
    and retrying is the only way through; the recursion bottoms out at a quarter
    hour, which has always parsed.
    """
    a = UTCDateTime(start.to_pydatetime()) - 120
    b = UTCDateTime(end.to_pydatetime()) + 120
    try:
        return _rows_from_events(_isc_query(spec, a, b), sequence, "isc")
    except Exception as exc:                                       # noqa: BLE001
        if (b - a) <= 900 or depth >= 6:
            log(f"      ISC {a} .. {b} unrecoverable: {type(exc).__name__}")
            return []
        mid = a + (b - a) / 2
        log(f"      ISC {a.strftime('%H:%M')}..{b.strftime('%H:%M')} "
            f"{type(exc).__name__}, splitting")
        rows = []
        for lo, hi in ((a, mid), (mid, b)):
            rows += harvest_isc_window(
                spec, sequence,
                pd.Timestamp(lo.datetime) + pd.Timedelta(seconds=120),
                pd.Timestamp(hi.datetime) - pd.Timedelta(seconds=120), depth + 1)
        return rows


def harvest_fdsn_window(spec, sequence, start, end, provider) -> list[dict]:
    """One region query with arrivals, for a service that serialises them."""
    try:
        cat = client(provider).get_events(
            starttime=UTCDateTime(start.to_pydatetime()) - 120,
            endtime=UTCDateTime(end.to_pydatetime()) + 120,
            latitude=spec["lat"], longitude=spec["lon"], maxradius=spec["radius"],
            includearrivals=True)
    except Exception as exc:                                       # noqa: BLE001
        log(f"      {provider} window failed: {type(exc).__name__}: {str(exc)[:60]}")
        return []
    return _rows_from_events(cat, sequence, provider)


def harvest_fdsn_per_event(spec, sequence, events, provider) -> list[dict]:
    """One arrival-bearing request per event.

    INGV and GeoNet reject a region query that asks for arrivals, so the
    catalogue is taken first and each event fetched by id.
    """
    rows, ok, fail = [], 0, 0
    for eid in events:
        try:
            cat = client(provider).get_events(eventid=eid, includearrivals=True)
            got = _rows_from_events(cat, sequence, provider)
            rows += got
            ok += 1
        except Exception:                                          # noqa: BLE001
            fail += 1
    if fail:
        log(f"      {provider}: {ok} events returned arrivals, {fail} failed")
    return rows


def harvest_usgs_window(spec, sequence, events) -> list[dict]:
    """The phase-data product for the events inside one scoring window."""
    rows = []
    for eid in events:
        try:
            d = json.load(urllib.request.urlopen(
                f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={eid}&format=geojson",
                timeout=90))
            prods = d["properties"]["products"].get("phase-data", [])
            if not prods:
                continue
            cont = prods[0]["contents"]
            qml = next(k for k in cont if k.endswith("quakeml.xml"))
            rows += _rows_from_events(read_events(io.BytesIO(
                urllib.request.urlopen(cont[qml]["url"], timeout=90).read())),
                sequence, "usgs")
        except Exception:                                          # noqa: BLE001
            continue
    return rows


def stage_picks() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, prov, win_rows = [], [], []
    for sequence, spec in SEQUENCES.items():
        if ONLY and spec['key'] not in ONLY and sequence not in ONLY:
            continue
        log(f"  {sequence} ({spec['source'].upper()})")
        cat = catalogue(spec)
        if cat.empty:
            continue
        n_win = spec.get("n_windows", N_WINDOWS)
        win_h = spec.get("window_h", WINDOW_H)
        for wi, (start, count) in enumerate(busiest_windows(cat, n_win, win_h)):
            end = start + timedelta(hours=win_h)
            rows, by_source = [], {}
            for src in spec.get("sources", [spec["source"]]):
                if src == "isc":
                    got = harvest_isc_window(spec, sequence, start, end)
                elif src == "usgs":
                    ev = cat[(cat.time >= start) & (cat.time < end)].event.tolist()
                    got = harvest_usgs_window(spec, sequence, ev)
                elif src in PER_EVENT_FDSN:
                    ev = cat[(cat.time >= start) & (cat.time < end)].event.tolist()
                    got = harvest_fdsn_per_event(spec, sequence, ev, src)
                else:
                    got = harvest_fdsn_window(spec, sequence, start, end, src)
                by_source[src] = len(got)
                rows += got
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df[(df.time >= start) & (df.time < end)]
                before = len(df)
                # The same analyst reading reaches us from more than one service.
                # Collapse on station, phase and time to a twentieth of a second,
                # keeping the source listed first, which is the one that reports a
                # real network code and an evaluation mode.
                # Rounding to a bin is not a tolerance: .620 and .660 land in
                # different bins and both survive, which left 399 arrivals inside
                # the scorer's own 0.5 s matching window of another arrival on the
                # same station and phase. Each one is a guaranteed miss under
                # one-to-one matching and inflates the reference. Scan instead, and
                # key on the alias-normalised code so a reading relayed as LBCW and
                # as LBC collapses.
                order = {src: i for i, src in enumerate(spec.get("sources", [spec["source"]]))}
                df["_pref"] = df.source.map(order).fillna(99)
                df["_sta"] = df.station_code.map(lambda c: _station_aliases(c)[-1])
                df["_t"] = pd.to_datetime(df.time)
                keep, last = [], {}
                for idx, row in df.sort_values(["_pref", "_t"]).iterrows():
                    k = (row._sta, row.phase)
                    prev = last.get(k)
                    if prev is not None and abs((row._t - prev).total_seconds()) < DEDUP_TOL_S:
                        continue
                    last[k] = row._t
                    keep.append(idx)
                df = df.loc[keep].drop(columns=["_pref", "_sta", "_t"])
                df["window"] = wi
                all_rows.append(df)
                log("      sources " + ", ".join(f"{k} {v}" for k, v in by_source.items())
                    + f" -> {len(df)} after merge (dropped {before - len(df)})")
            n_p = int((df.phase == "P").sum()) if not df.empty else 0
            n_s = int((df.phase == "S").sum()) if not df.empty else 0
            win_rows.append(dict(sequence=sequence, key=spec["key"], window=wi,
                                 start=start, end=end, catalogue_events=count,
                                 arrivals=len(df), P=n_p, S=n_s,
                                 stations=df.station_code.nunique() if not df.empty else 0))
            log(f"    w{wi} {start:%Y-%m-%d %H:%M} {count:4d} events -> "
                f"{len(df):5d} arrivals, P {n_p:4d} S {n_s:4d}, "
                f"{df.station_code.nunique() if not df.empty else 0} stations")
        sub = pd.concat([d for d in all_rows if (d.sequence == sequence).all()],
                        ignore_index=True) if all_rows else pd.DataFrame()
        if not sub.empty:
            prov.append(dict(sequence=sequence, key=spec["key"], source=spec["source"],
                             events=sub.event.nunique(), arrivals=len(sub),
                             P=int((sub.phase == "P").sum()), S=int((sub.phase == "S").sum()),
                             stations=sub.station_code.nunique(),
                             modes=json.dumps(Counter(sub["mode"]).most_common()),
                             sources=json.dumps(Counter(sub["source"]).most_common()),
                             t0=spec["t0"], t1=spec["t1"], note=spec["note"]))
    full = pd.concat(all_rows, ignore_index=True)
    # With --only, keep the sequences this run did not touch instead of
    # replacing the file with a single sequence.
    def _merge(new: pd.DataFrame, path, key="sequence") -> pd.DataFrame:
        if ONLY and path.exists() and not new.empty:
            old = pd.read_csv(path)
            if key in old.columns:
                keep = old[~old[key].isin(set(new[key]))]
                new = pd.concat([keep, new], ignore_index=True)
        return new

    _merge(full, OUT / "reference_picks_raw.csv").to_csv(OUT / "reference_picks_raw.csv", index=False)
    _merge(pd.DataFrame(prov), OUT / "provenance.csv").to_csv(OUT / "provenance.csv", index=False)
    _merge(pd.DataFrame(win_rows), OUT / "windows.csv").to_csv(OUT / "windows.csv", index=False)
    log(f"\n  {len(full):,} arrivals -> {(OUT / 'reference_picks_raw.csv').relative_to(ROOT)}")
    log(pd.DataFrame(prov)[["sequence", "events", "arrivals", "P", "S", "stations"]]
        .to_string(index=False))


# ------------------------------------------------------------------ stage 2
def _station_aliases(code: str) -> list[str]:
    """The code as the bulletin reports it, then the code the archive knows.

    ISC reports the WEBNET array with a trailing W that no FDSN service answers
    to: LBCW for LBC, KOCW for KOC, and so on for the whole dense array that
    makes West Bohemia worth scoring. Without this the resolver falls through to
    regional stations carrying four arrivals each.
    """
    out = [code]
    if len(code) == 4 and code.endswith("W"):
        out.append(code[:-1])
    return out


MAX_STATION_KM = 400.0          # a local sequence is not recorded usefully beyond this


def resolve_network(code: str, t: UTCDateTime,
                    centre: tuple[float, float] | None = None) -> tuple[str, str, str] | None:
    """Network, archive station code and band for a code the bulletin reported.

    Returns the code the archive answers to, which is not always the one the
    bulletin printed, so the caller can label the pick with a station the model
    output will also carry.

    Station codes are not unique across networks, and a query by bare code
    answers with whichever network replies first. That put Silent Canyon,
    Nevada (SN.STC) on the West Bohemia swarm, 9,097 km away, and matched its
    record against Czech analyst arrivals. When ``centre`` is given, a
    candidate more than MAX_STATION_KM from the sequence is rejected and the
    search continues, so a wrong network cannot be accepted silently.
    """
    global _fed
    if _fed is None:
        _fed = RoutingClient("iris-federator")
    for alias in _station_aliases(code):
        for router in (_fed, RoutingClient("eida-routing")):
            try:
                inv = router.get_stations(station=alias, starttime=t, endtime=t + 3600,
                                          level="channel")
            except Exception:                                      # noqa: BLE001
                continue
            for net in inv:
                for sta in net:
                    if centre is not None:
                        km = gps2dist_azimuth(centre[0], centre[1],
                                              sta.latitude, sta.longitude)[0] / 1000.0
                        if km > MAX_STATION_KM:
                            log(f"      reject {net.code}.{alias}: {km:,.0f} km from the sequence")
                            continue
                    bands = {c.code[:2] for c in sta.channels if c.code[-1] in "ZNE12"}
                    for pref in CHANNEL_PREF:
                        if pref in bands:
                            return net.code, alias, pref
    return None


def busiest_windows(df: pd.DataFrame, n: int, hours: int) -> list[tuple]:
    """The n most populated non-overlapping windows, by arrival count."""
    t = pd.to_datetime(df.time).sort_values()
    if t.empty:
        return []
    span = pd.date_range(t.min().floor("h"), t.max().ceil("h"), freq="1h")
    counts = [(s, int(((t >= s) & (t < s + timedelta(hours=hours))).sum())) for s in span]
    counts.sort(key=lambda x: -x[1])
    chosen = []
    for s, c in counts:
        if c == 0:
            break
        if all(abs((s - o).total_seconds()) >= hours * 3600 for o, _ in chosen):
            chosen.append((s, c))
        if len(chosen) == n:
            break
    return chosen


def stage_waveforms() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(OUT / "reference_picks_raw.csv", parse_dates=["time"])
    wins = pd.read_csv(OUT / "windows.csv", parse_dates=["start", "end"])
    sta_rows, keep = [], []
    for sequence, spec in SEQUENCES.items():
        if ONLY and spec['key'] not in ONLY and sequence not in ONLY:
            continue
        sub = raw[raw.sequence == sequence]
        if sub.empty:
            continue
        log(f"  {sequence}")
        for _, wrow in wins[wins.sequence == sequence].iterrows():
            wi, start, end = int(wrow.window), wrow.start, wrow.end
            inwin = sub[sub.window == wi]
            n_sta = spec.get("n_stations", N_STATIONS)
            top = inwin.station_code.value_counts().head(n_sta * 4)
            n_ok = 0
            for code in top.index:
                if n_ok >= n_sta:
                    break
                got = resolve_network(code, UTCDateTime(start.to_pydatetime()),
                                      (spec["lat"], spec["lon"]))
                if got is None:
                    continue
                net, archive_code, band = got
                station = f"{net}.{archive_code}"
                path = CACHE / f"{spec['key']}_w{wi}_{station}.mseed"
                if not path.exists():
                    try:
                        st = client(KNOWN_ROUTE.get(net, "IRIS")).get_waveforms(
                            net, archive_code, "*", f"{band}?",
                            UTCDateTime(start.to_pydatetime()) - 60,
                            UTCDateTime(end.to_pydatetime()) + 60)
                    except Exception:                              # noqa: BLE001
                        try:
                            st = RoutingClient("iris-federator").get_waveforms(
                                network=net, station=archive_code, location="*",
                                channel=f"{band}?",
                                starttime=UTCDateTime(start.to_pydatetime()) - 60,
                                endtime=UTCDateTime(end.to_pydatetime()) + 60)
                        except Exception:                          # noqa: BLE001
                            continue
                    if len(st) < 3:
                        continue
                    st.merge(fill_value=0)
                    # A long window whose arrivals occupy a small part of it is
                    # mostly bytes nobody scores: Etna's 12 h windows hold 1.75 h
                    # of arrivals, so storing them whole cost 816 MB for 337
                    # picks. Trim to the arrivals plus a margin when that saves
                    # most of the file. Every arrival is kept; only span with no
                    # arrival in it is dropped, and the stored span is recorded.
                    at = pd.to_datetime(
                        inwin[inwin.station_code == code].time, utc=True,
                        format="mixed")
                    if len(at):
                        lo = UTCDateTime(at.min().to_pydatetime()) - TRIM_MARGIN_S
                        hi = UTCDateTime(at.max().to_pydatetime()) + TRIM_MARGIN_S
                        full = st[0].stats.endtime - st[0].stats.starttime
                        a = max(lo, st[0].stats.starttime)
                        b = min(hi, st[0].stats.endtime)
                        # b <= a means the arrivals lie outside what the archive
                        # returned; keep the trace whole rather than trim to nothing
                        if b > a and (hi - lo) < TRIM_IF_UNDER * full:
                            st.trim(a, b)
                    st.write(str(path), format="MSEED")
                n_ok += 1
                _hdr = obspy.read(str(path), headonly=True)[0].stats
                _t0, _t1 = _hdr.starttime, _hdr.endtime
                sta_rows.append(dict(sequence=sequence, window=wi, station=station,
                                     station_code=code, archive_code=archive_code,
                                     network=net, band=band,
                                     arrivals=int(top[code]), file=path.name,
                                     stored_start=str(_t0), stored_end=str(_t1)))
                w = inwin[inwin.station_code == code].copy()
                w["station"] = station
                keep.append(w)
                log(f"    w{wi} {station:12s} {band} {int(top[code]):4d} arrivals")
    _st = pd.DataFrame(sta_rows)
    if ONLY and (OUT / "stations.csv").exists() and not _st.empty:
        _old = pd.read_csv(OUT / "stations.csv")
        _st = pd.concat([_old[~_old.sequence.isin(set(_st.sequence))], _st], ignore_index=True)
    _st.to_csv(OUT / "stations.csv", index=False)
    ref = pd.concat(keep, ignore_index=True)[["sequence", "station", "phase", "time"]]
    ref = ref.sort_values(["sequence", "station", "phase", "time"])
    if ONLY and (OUT / "reference_picks.csv").exists() and not ref.empty:
        _o = pd.read_csv(OUT / "reference_picks.csv")
        ref = pd.concat([_o[~_o.sequence.isin(set(ref.sequence))], ref], ignore_index=True)
    ref.to_csv(OUT / "reference_picks.csv", index=False)
    log(f"\n  {len(ref):,} scorable arrivals -> {(OUT / 'reference_picks.csv').relative_to(ROOT)}")
    log(ref.groupby(["sequence", "phase"]).size().unstack(fill_value=0).to_string())


# ------------------------------------------------------------------ stage 3
def stage_models() -> None:
    """Run each weight set over the stored windows, checkpointing as it goes.

    264 station-runs take longer than a background slot, and writing only at the
    end loses everything when one is cut short. Completed (weight, file) pairs
    are appended to a part file and skipped on restart.
    """
    import seisbench.models as sbm
    sta = pd.read_csv(OUT / "stations.csv")
    part = OUT / "model_picks.part.csv"
    done: set[tuple[str, str]] = set()
    if part.exists():
        prev = pd.read_csv(part)
        done = set(zip(prev.weights, prev.file))
        log(f"  resuming: {len(done)} (weight, window) pairs already scored, "
            f"{len(prev):,} picks")
    header = not part.exists()

    for name in WEIGHTS:
        model = None
        for _, r in sta.iterrows():
            if (name, r.file) in done:
                continue
            path = CACHE / r.file
            if not path.exists():
                continue
            if model is None:                      # load only when there is work
                log(f"  {name}")
                model = sbm.PhaseNet.from_pretrained(name)
                model.eval()
            st = obspy.read(str(path))
            picks = model.classify(st, P_threshold=DETECT_FLOOR,
                                   S_threshold=DETECT_FLOOR).picks
            rows = [dict(sequence=r.sequence, weights=name, station=r.station,
                         file=r.file, phase=p.phase[:1].upper(),
                         time=p.peak_time.datetime, conf=float(p.peak_value))
                    for p in picks]
            pd.DataFrame(rows or [dict(sequence=r.sequence, weights=name, station=r.station,
                                       file=r.file, phase="", time=pd.NaT, conf=float("nan"))]
                         ).to_csv(part, mode="a", header=header, index=False)
            header = False
            log(f"    {r.station:12s} w{r.window} {len(picks):5d} picks")

    out = pd.read_csv(part).dropna(subset=["phase"])
    out = out[out.phase != ""].drop(columns=["file"])
    out = out.sort_values(["sequence", "weights", "station", "time"])
    out.to_csv(OUT / "model_picks.csv", index=False)
    log(f"\n  {len(out):,} model picks -> {(OUT / 'model_picks.csv').relative_to(ROOT)}")
    log(out[out.conf >= 0.3].groupby(["sequence", "weights"]).size()
        .unstack(fill_value=0).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["picks", "waveforms", "models"])
    ap.add_argument("--only", default="", help="comma-separated sequence keys")
    a = ap.parse_args()
    ONLY.update(x for x in a.only.split(",") if x)
    {"picks": stage_picks, "waveforms": stage_waveforms, "models": stage_models}[a.stage]()
