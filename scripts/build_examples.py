#!/usr/bin/env python
"""Waveform examples for the picker board: what the four weight sets did, on the record.

Picks a handful of analyst arrivals from the scored sequences, fetches the same
waveform the benchmark read, and draws every model's pick against the analyst's.
Writes, per example, an SVG for the page and a MiniSEED file so anyone can pull
the record and check the figure.

    pixi run -e dev python scripts/build_examples.py

Waveforms come from the archives the benchmark itself used and need no account:
the SCEDC and NCEDC public buckets for CI, BK and NC, and the operator's FDSN
service for NZ, IV, HL and HT. Nothing here touches EarthScope.

Output lands in reports/examples/ and is read by scripts/build_leaderboard.py.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from s3fs import S3FileSystem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "docs" / "benchmark" / "results"
OUT = ROOT / "reports" / "examples"

WEIGHTS = ["quakescope2026", "jma_wc", "original", "instance"]
WCOLOR = {"quakescope2026": "#4b2e83", "jma_wc": "#c2571a",
          "original": "#1b7f79", "instance": "#2f6fb2"}
INK, STONE, LINE = "#2a1a4f", "#6f6890", "#d8d2e8"
TOL = 0.5                 # s, the board's detection tolerance
PAD_BEFORE, PAD_AFTER = 6.0, 14.0
ZOOM_S = 0.6                # s either side of the analyst pick in the zoom panel
SHARED_THR = 0.3

S3_BUCKET = {"CI": "scedc", "BK": "ncedc", "NC": "ncedc", "NP": "ncedc"}
FDSN_ROUTE = {"NZ": "GEONET", "IV": "INGV", "MN": "INGV",
              "HL": "NOA", "HT": "NOA", "HP": "NOA", "HA": "NOA"}
CHANNEL_PREF = ["HH", "EH", "BH"]

_fs = S3FileSystem(anon=True)
_clients: dict[str, Client] = {}
_listings: dict[str, set] = {}


# ---------------------------------------------------------------- waveforms
# The bucket layouts below are the ones the benchmark notebooks read: SCEDC pads
# the station to five characters and the location to three and keeps one folder
# per day; NCEDC makes the network a directory level and separates the day with
# a dot. Listing the day prefix once finds the location code instead of guessing
# it, and avoids an s3fs trap where a glob caches only the matching entries and
# a later sibling open then raises FileNotFoundError.
def _day_listing(bucket: str, net: str, year: int, doy: int):
    prefix = (f"scedc-pds/continuous_waveforms/{year}/{year}_{doy:03d}/" if bucket == "scedc"
              else f"ncedc-pds/continuous_waveforms/{net}/{year}/{year}.{doy:03d}/")
    if prefix not in _listings:
        try:
            _listings[prefix] = {k.split("/")[-1] for k in _fs.ls(prefix)}
        except Exception:
            _listings[prefix] = set()
    return prefix, _listings[prefix]


def _object_name(bucket, net, sta, cha, comp, year, doy, names):
    if bucket == "scedc":
        head, tail = f"{net}{sta.ljust(5, '_')}{cha}{comp}", f"{year}{doy:03d}.ms"
    else:
        head, tail = f"{sta}.{net}.{cha}{comp}.", f".D.{year}.{doy:03d}"
    hits = [n for n in names if n.startswith(head) and n.endswith(tail)]
    return sorted(hits)[0] if hits else None


def read_s3(net, sta, t0, t1):
    bucket = S3_BUCKET.get(net)
    if bucket is None:
        return None
    year, doy = t0.year, int(t0.strftime("%j"))
    prefix, names = _day_listing(bucket, net, year, doy)
    if not names:
        return None
    for cha in CHANNEL_PREF:
        st = obspy.Stream()
        ok = True
        for comp in "ZNE":
            name = _object_name(bucket, net, sta, cha, comp, year, doy, names)
            if name is None:
                ok = False
                break
            try:
                with _fs.open(prefix + name) as fh:
                    st += obspy.read(io.BytesIO(fh.read()))
            except Exception:
                ok = False
                break
        if ok and len(st) >= 3:
            st.merge(fill_value=0)
            st.trim(t0, t1)
            return st if len(st) >= 3 else None
    return None


def read_fdsn(net, sta, t0, t1):
    provider = FDSN_ROUTE.get(net)
    if provider is None:
        return None
    if provider not in _clients:
        _clients[provider] = Client(provider, timeout=120)
    for cha in CHANNEL_PREF:
        try:
            st = _clients[provider].get_waveforms(net, sta, "*", f"{cha}?", t0, t1)
        except Exception:
            continue
        if len(st) >= 3:
            st.merge(fill_value=0)
            st.trim(t0, t1)
            return st
    return None


def fetch(station: str, t0: UTCDateTime, t1: UTCDateTime):
    net, sta = station.split(".")
    return read_s3(net, sta, t0, t1) or read_fdsn(net, sta, t0, t1)


# ---------------------------------------------------------------- selection
def cases(picks: pd.DataFrame, ref: pd.DataFrame, study: str) -> list[dict]:
    """Arrivals chosen by what the models did, not by hand.

    One where every weight set recovered the arrival, one where at least one
    missed it while others did not, and one with the largest onset disagreement
    between weight sets. Selection is deterministic, so the examples move only
    when the picks do.
    """
    out = []
    ref = ref[ref.station.isin(set(S3_BUCKET) | set(FDSN_ROUTE)
                               | {s.split(".")[0] for s in ref.station}
                               ) | True].copy()
    ref = ref[ref.station.str.split(".").str[0].isin(set(S3_BUCKET) | set(FDSN_ROUTE))]
    rows = []
    for _, r in ref.iterrows():
        t = pd.Timestamp(r.time).timestamp()
        got, res = {}, {}
        for w in WEIGHTS:
            g = picks[(picks.sequence == r.sequence) & (picks.station == r.station)
                      & (picks.phase == r.phase) & (picks.weights == w)
                      & (picks.conf >= SHARED_THR)]
            if g.empty:
                continue
            d = (pd.to_datetime(g.time).astype("int64") / 1e9) - t
            j = d.abs().idxmin()
            if abs(d[j]) <= TOL:
                got[w] = float(d[j])
                res[w] = float(g.conf[j])
        rows.append(dict(sequence=r.sequence, station=r.station, phase=r.phase,
                         time=r.time, n_hit=len(got), spread=(max(got.values()) - min(got.values()))
                         if len(got) > 1 else 0.0, hits=got, conf=res))
    df = pd.DataFrame(rows)
    if df.empty:
        return out
    full = df[df.n_hit == len(WEIGHTS)].sort_values("spread")
    if len(full):
        out.append({**full.iloc[0].to_dict(), "why": "every weight set recovered this arrival"})
    partial = df[(df.n_hit >= 1) & (df.n_hit < len(WEIGHTS))].sort_values("n_hit")
    if len(partial):
        miss = sorted(set(WEIGHTS) - set(partial.iloc[0]["hits"]))
        out.append({**partial.iloc[0].to_dict(),
                    "why": f"{', '.join(miss)} did not recover this arrival at a 0.3 threshold"})
    wide = df[df.n_hit >= 3].sort_values("spread", ascending=False)
    if len(wide):
        out.append({**wide.iloc[0].to_dict(),
                    "why": f"the weight sets disagree by {wide.iloc[0]['spread'] * 1000:.0f} ms "
                           "on the same arrival"})
    for c in out:
        c["study"] = study
    return out


# ---------------------------------------------------------------- figure
def draw(case: dict, st, ref_picks: pd.DataFrame, picks: pd.DataFrame, path: Path) -> None:
    """Three components over the whole window, and a zoom where the picks separate.

    The weight sets often agree to within a few tens of milliseconds, which is
    invisible on a 20-second trace, so the right-hand panel spans ZOOM_S either
    side of the analyst pick. That panel is the comparison; the left is the
    context that says whether there was an arrival to pick at all.
    """
    st = st.copy().detrend("demean").taper(0.02).filter("bandpass", freqmin=1.0, freqmax=20.0)
    comps = [tr for c in "ZNE" for tr in st.select(component=c)][:3]
    if not comps:
        comps = list(st)[:3]

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(len(comps), 2, width_ratios=[2.15, 1.0], wspace=0.13, hspace=0.18)
    zoom = fig.add_subplot(gs[:, 1])

    def _bare(ax, keep_bottom=True):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_visible(keep_bottom)
        ax.spines["bottom"].set_color(LINE)
        ax.set_yticks([])
        ax.tick_params(colors=STONE, labelsize=8.5, length=3)

    for k, tr in enumerate(comps):
        ax = fig.add_subplot(gs[k, 0])
        t = tr.times() - PAD_BEFORE
        y = tr.data / (np.abs(tr.data).max() or 1)
        step = max(1, len(t) // 2200)
        ax.plot(t[::step], y[::step], color="#3b3557", lw=0.7, zorder=2)
        ax.set_ylabel(tr.stats.channel, fontsize=8.5, color=STONE)
        ax.set_xlim(-PAD_BEFORE, PAD_AFTER)
        ax.set_ylim(-1.15, 1.15)
        ax.axvline(0.0, color="#111", lw=1.4, zorder=4)
        ax.axvspan(-ZOOM_S, ZOOM_S, color="#6d5bd0", alpha=0.10, lw=0, zorder=1)
        _bare(ax, keep_bottom=(k == len(comps) - 1))
        if k < len(comps) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel(f"seconds from the analyst {case['phase']} pick",
                          fontsize=9.5, color=INK)
        if k == 0:
            ax.set_title(f"{case['sequence']} \u00b7 {case['station']} \u00b7 {case['phase']}",
                         fontsize=10.5, color=INK, loc="left")

    # The zoom component follows the phase: a P shows on the vertical, an S on a
    # horizontal. Within the allowed components, take the one with the most
    # amplitude in the shaded band.
    want = "Z" if case["phase"] == "P" else "NE12"
    pool = [x for x in comps if x.stats.channel[-1] in want] or comps

    def _band(x):
        sr = x.stats.sampling_rate
        seg = x.data[max(0, int((PAD_BEFORE - ZOOM_S) * sr)):int((PAD_BEFORE + ZOOM_S) * sr)]
        return float(np.abs(seg).max()) if len(seg) else 0.0

    tr = max(pool, key=_band)
    t = tr.times() - PAD_BEFORE
    m = (t >= -ZOOM_S) & (t <= ZOOM_S)
    y = tr.data[m] / (np.abs(tr.data[m]).max() or 1)
    zoom.plot(t[m], y, color="#3b3557", lw=1.0, zorder=2)
    zoom.axvline(0.0, color="#111", lw=1.6, zorder=5)
    for w in WEIGHTS:
        d = case["hits"].get(w)
        if d is None:
            continue
        zoom.axvline(d, color=WCOLOR[w], lw=1.5, ls="--", alpha=0.95, zorder=4)
    zoom.set_xlim(-ZOOM_S, ZOOM_S)
    zoom.set_ylim(-1.2, 1.2)
    zoom.set_xlabel("seconds", fontsize=9.5, color=INK)
    zoom.set_title(f"{tr.stats.channel}, \u00b1{ZOOM_S:g} s around the pick",
                   fontsize=9.5, color=INK, loc="left")
    zoom.set_facecolor("#f7f5fd")
    _bare(zoom)
    zoom.grid(True, color=LINE, lw=0.5, alpha=0.7, axis="x")
    zoom.set_axisbelow(True)

    handles = [plt.Line2D([], [], color="#111", lw=1.6, label=f"analyst {case['phase']}")]
    for w in WEIGHTS:
        d, c = case["hits"].get(w), case["conf"].get(w)
        lab = (f"{w}  {d * 1000:+.0f} ms, conf {c:.2f}" if d is not None
               else f"{w}  no pick within {TOL:g} s")
        handles.append(plt.Line2D([], [], color=WCOLOR[w], lw=1.4,
                                  ls="--" if d is not None else ":",
                                  alpha=0.95 if d is not None else 0.35, label=lab))
    fig.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15))
    fig.savefig(path, format=path.suffix.lstrip("."), bbox_inches="tight",
                transparent=path.suffix == ".svg", dpi=130,
                facecolor="white" if path.suffix == ".png" else "none")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for study in ("us_sequences", "global_sequences"):
        picks = pd.read_csv(RES / study / "model_picks.csv")
        ref = pd.read_csv(RES / study / "reference_picks.csv")
        for case in cases(picks, ref, study):
            t = pd.Timestamp(case["time"])
            t0 = UTCDateTime(t.to_pydatetime()) - PAD_BEFORE
            t1 = UTCDateTime(t.to_pydatetime()) + PAD_AFTER
            st = fetch(case["station"], t0, t1)
            if st is None or len(st) < 1:
                print(f"  no waveform for {case['station']} {t}, skipped")
                continue
            key = (f"{study.replace('_sequences', '')}-"
                   f"{case['sequence'].lower().replace(' ', '')}-"
                   f"{case['station'].replace('.', '')}-{case['phase']}")
            st.write(str(OUT / f"{key}.mseed"), format="MSEED")
            draw(case, st, ref, picks, OUT / f"{key}.svg")
            manifest.append({
                "id": key, "study": study.replace("_sequences", ""),
                "sequence": case["sequence"], "station": case["station"],
                "phase": case["phase"], "time": str(t), "why": case["why"],
                "channels": sorted({tr.stats.channel for tr in st}),
                "sampling_rate": float(st[0].stats.sampling_rate),
                "residuals_ms": {w: round(d * 1000, 1) for w, d in case["hits"].items()},
                "conf": {w: round(c, 3) for w, c in case["conf"].items()},
                "missed": sorted(set(WEIGHTS) - set(case["hits"])),
                "mseed": f"examples/{key}.mseed", "svg": f"examples/{key}.svg",
            })
            print(f"  {key}: {len(st)} traces, {case['n_hit']}/{len(WEIGHTS)} weight sets")
    (OUT / "examples.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} examples -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
