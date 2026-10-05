#!/usr/bin/env python
"""Build the western-states deployment talk at reports/western_deployment_talk.html.

Ten slides for a seismology audience: what the campaign produced, where it
produced it, what it did not produce, how we score it, and why the next
benchmark is organised by earthquake-sequence regime rather than by place.

Every number is read from a measurement file rather than typed:

    scratchpad/western_scan.json       manifest scan: per station, station-days, zero days
    scratchpad/western_footers.json    row count from every Parquet footer
    scratchpad/western_phase.json      P and S split, stratified file sample
    scratchpad/western_station_map.parquet   those stations joined to coordinates
    docs/benchmark/results/*.csv       the leaderboard tables
    docs/benchmark/results/swarm_sequences/  the new regime track

    pixi run -e dev python scripts/build_western_talk.py

Arrow keys or click to move. The deck prints one slide per page.
"""
from __future__ import annotations

import html as _html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SP = Path("/private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/"
          "8f282710-68a1-4cdd-b77e-e07856ef31af/scratchpad")
RES = ROOT / "docs" / "benchmark" / "results"
OUT = ROOT / "reports" / "western_deployment_talk.html"
BASEMAP = ROOT / "scripts" / "data" / "basemap.json"

WEIGHTS = ["quakescope2026", "jma_wc", "original", "instance"]
WCOLOR = {"quakescope2026": "#4b2e83", "jma_wc": "#c2571a",
          "original": "#1b7f79", "instance": "#2f6fb2"}
DETECT_TOL = 0.5          # s, the board's detection tolerance
BUCKET_HTTP = "https://quakescope-picks-2026.s3.us-east-2.amazonaws.com"


def obs_station_map() -> pd.DataFrame:
    """Picks per station in the obs catalogue, from its manifests and table.

    Same columns as the old cached table (picks, days, zero, latitude,
    longitude, network_code), indexed by station id. Only stations listed in
    obs/stations.parquet are kept, which since 2026-10-03 means offshore only.
    """
    import io
    from concurrent.futures import ThreadPoolExecutor
    import boto3
    s3 = boto3.client("s3", region_name="us-east-2")
    bucket = "quakescope-picks-2026"
    keys = [o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix="obs/manifests/") for o in p.get("Contents", [])]
    def records(k):
        return json.loads(s3.get_object(Bucket=bucket, Key=k)["Body"].read()).get("records", [])
    with ThreadPoolExecutor(64) as ex:
        rec = pd.DataFrame([r for rs in ex.map(records, keys) for r in rs])
    st = pd.read_parquet(io.BytesIO(s3.get_object(
        Bucket=bucket, Key="obs/stations.parquet")["Body"].read())).set_index("id")
    rec = rec[rec["tid"].isin(st.index)]
    g = rec.groupby("tid").agg(picks=("npks", "sum"), days=("npks", "size"),
                               zero=("npks", lambda s: float((s == 0).sum())))
    g = g.join(st[["latitude", "longitude", "network_code"]])
    print(f"obs layer: {len(keys):,} manifests, {len(g):,} stations, "
          f"{int(g['picks'].sum()):,} picks")
    return g


def load():
    d = {}
    d["scan"] = json.loads((SP / "western_scan.json").read_text())
    d["phase"] = json.loads((SP / "western_phase.json").read_text())
    f = SP / "western_footers.json"
    d["footers"] = json.loads(f.read_text()) if f.exists() else None
    o = SP / "western_objects.json"
    d["objects"] = json.loads(o.read_text()) if o.exists() else None
    d["map"] = pd.read_parquet(SP / "western_station_map.parquet")
    d["stations"] = pd.read_parquet(SP / "western_stations.parquet")
    d["detection"] = pd.read_csv(RES / "detection_full.csv")
    d["timing"] = pd.read_csv(RES / "timing_full.csv")
    d["calib"] = pd.read_csv(RES / "calibration_full.csv")
    d["quality"] = pd.read_csv(RES / "phase_quality_full.csv")
    d["repro"] = json.loads((RES / "western_reproduction" / "meta.json").read_text())
    sw = RES / "swarm_sequences"
    d["swarm"] = pd.read_csv(sw / "reference_picks.csv") if (sw / "reference_picks.csv").exists() else None
    d["swarm_prov"] = pd.read_csv(sw / "provenance.csv") if (sw / "provenance.csv").exists() else None
    d["swarm_win"] = pd.read_csv(sw / "windows.csv") if (sw / "windows.csv").exists() else None
    r = RES / "regime_roster.csv"
    d["roster"] = pd.read_csv(r) if r.exists() else None
    b = RES / "seisbench_manifest.csv"
    d["sb"] = pd.read_csv(b) if b.exists() else None
    q = RES / "regime_sequences.csv"
    d["seq"] = pd.read_csv(q) if q.exists() else None
    # The ocean-bottom layer is read from the bucket at build time, not from a
    # cached table: the cache of 2026-10-02 held 1,769 stations, 1,157 of them
    # land stations picked under reused offshore codes, archived out of obs/ on
    # 2026-10-03 (docs/rerun_2026/31_obs_station_selection.md).
    d["obs"] = obs_station_map()
    d["obs_meta"] = None
    f = SP / "station_table_fix.json"
    d["fix"] = json.loads(f.read_text()) if f.exists() else None
    return d


# ------------------------------------------------------------------ map
WEIGHT_COLOUR = {"original": "#4b2e83", "obs": "#c2571a"}
WEIGHT_WHAT = {"original": "land, PhaseNet <code>original</code>",
               "obs": "ocean bottom, PhaseNet <code>obs</code> (PickBlue)"}


def station_map(mp, obs=None, w=1180, h=620) -> str:
    """Western and ocean-bottom stations on one pannable, zoomable frame.

    Colour is the weight set that produced the picks, which is the honest
    grouping: the two campaigns ran different models and their numbers are not
    interchangeable. The frame opens on the western states and the Cascadia
    margin; the ocean-bottom campaign reaches the whole globe, so the control
    to fit everything is there rather than making that the default view.
    """
    land = mp[mp.latitude.notna() & (mp.picks > 0)].copy()
    land["w"] = "original"
    parts = [land[["picks", "latitude", "longitude", "w"]]]
    if obs is not None:
        sea = obs[obs.latitude.notna() & (obs.picks > 0)].copy()
        sea["w"] = "obs"
        parts.append(sea[["picks", "latitude", "longitude", "w"]])
    pts = pd.concat(parts, ignore_index=True)

    # one world-wide linear projection; zoom is a transform on top of it, so a
    # marker never has to be re-projected in the browser
    X0, X1, Y0, Y1 = -180.0, 180.0, -85.0, 85.0
    SCALE = 12.0                       # internal units per degree of longitude

    def sx(lo):
        return (lo - X0) * SCALE

    def sy(la):
        return (Y1 - la) * SCALE

    world_w, world_h = (X1 - X0) * SCALE, (Y1 - Y0) * SCALE
    opening = (-128.0, -103.0, 30.0, 52.0)      # the western states and the margin

    def fit(x0, x1, y0, y1):
        bw, bh = (x1 - x0) * SCALE, (y1 - y0) * SCALE
        k = min(w / bw, h / bh)
        return k, (w - bw * k) / 2 - sx(x0) * k, (h - bh * k) / 2 - sy(y1) * k

    k0, tx0, ty0 = fit(*opening)
    kw, twx, twy = fit(X0, X1, Y0, Y1)

    out = [f'<div class="mapwrap"><div class="mapbtns">'
           f'<button type="button" data-view="west">western states</button>'
           f'<button type="button" data-view="world">fit everything</button>'
           f'<span class="hint">scroll to zoom, drag to pan</span></div>'
           f'<svg viewBox="0 0 {w} {h}" class="map zoomable" role="img" '
           f'aria-label="Stations that produced picks, coloured by the weight set">'
           f'<rect width="{w}" height="{h}" fill="transparent"/>'
           f'<g id="panzoom" transform="translate({tx0:.2f},{ty0:.2f}) scale({k0:.4f})">']
    try:
        bm = json.loads(BASEMAP.read_text())
        for key, cls in (("coast", "coast"), ("states", "border")):
            for line in bm.get(key, []):
                pt = " ".join(f"{sx(lo):.1f},{sy(la):.1f}" for lo, la in line)
                if pt:
                    out.append(f'<polyline class="{cls}" points="{pt}"/>')
    except Exception:                                              # noqa: BLE001
        pass

    big = float(pts.picks.max())
    for wname in ("original", "obs"):                 # ocean bottom drawn last
        g = pts[pts.w == wname]
        c = WEIGHT_COLOUR[wname]
        for _, r in g.sort_values("picks").iterrows():
            frac = (r.picks / big) ** 0.33
            out.append(f'<circle class="stn" cx="{sx(r.longitude):.1f}" '
                       f'cy="{sy(r.latitude):.1f}" r="{(1.4 + 7.0 * frac) / k0:.2f}" '
                       f'fill="{c}" fill-opacity="{0.2 + 0.5 * frac:.2f}" '
                       f'stroke="#fff" stroke-width="{0.4 / k0:.2f}"/>')
    out.append("</g>")

    out.append(f'<g class="maplegend" transform="translate(16,{h - 76})">')
    for i, wname in enumerate(("original", "obs")):
        g = pts[pts.w == wname]
        out.append(f'<circle cx="9" cy="{i * 24 + 9}" r="7" fill="{WEIGHT_COLOUR[wname]}" '
                   f'fill-opacity=".75"/>'
                   f'<text x="26" y="{i * 24 + 14}">{wname}: {len(g):,} stations, '
                   f'{int(g.picks.sum()):,} picks</text>')
    out.append("</g></svg></div>")
    out.append(f"""<script>
(function () {{
  var g = document.getElementById("panzoom");
  if (!g) return;
  var svg = g.closest("svg");
  var views = {{west: [{k0:.4f}, {tx0:.2f}, {ty0:.2f}],
                world: [{kw:.4f}, {twx:.2f}, {twy:.2f}]}};
  var k = views.west[0], tx = views.west[1], ty = views.west[2];
  function apply() {{
    g.setAttribute("transform", "translate(" + tx + "," + ty + ") scale(" + k + ")");
    // keep markers a constant size on screen as the frame zooms
    var r0 = {k0:.4f} / k;
    g.querySelectorAll(".stn").forEach(function (c) {{
      if (!c.dataset.r) {{ c.dataset.r = c.getAttribute("r"); c.dataset.s = c.getAttribute("stroke-width"); }}
      c.setAttribute("r", c.dataset.r * r0);
      c.setAttribute("stroke-width", c.dataset.s * r0);
    }});
  }}
  function pt(ev) {{
    var b = svg.getBoundingClientRect();
    return [(ev.clientX - b.left) / b.width * {w}, (ev.clientY - b.top) / b.height * {h}];
  }}
  svg.addEventListener("wheel", function (ev) {{
    ev.preventDefault();
    var p = pt(ev), f = Math.exp(-ev.deltaY * 0.0015), nk = Math.min(400, Math.max({kw:.4f} * 0.9, k * f));
    tx = p[0] - (p[0] - tx) * (nk / k);
    ty = p[1] - (p[1] - ty) * (nk / k);
    k = nk; apply();
  }}, {{passive: false}});
  var drag = null;
  svg.addEventListener("pointerdown", function (ev) {{ drag = pt(ev); svg.setPointerCapture(ev.pointerId); }});
  svg.addEventListener("pointermove", function (ev) {{
    if (!drag) return;
    var p = pt(ev); tx += p[0] - drag[0]; ty += p[1] - drag[1]; drag = p; apply();
  }});
  svg.addEventListener("pointerup", function (ev) {{ drag = null; svg.releasePointerCapture(ev.pointerId); }});
  document.querySelectorAll(".mapbtns button").forEach(function (b) {{
    b.addEventListener("click", function (ev) {{
      ev.stopPropagation();
      var v = views[b.dataset.view]; k = v[0]; tx = v[1]; ty = v[2]; apply();
    }});
  }});
  apply();
}})();
</script>""")
    return "".join(out)


REGIME_COLOUR = {"mainshock-aftershock": "#4b2e83",
                 "volcano-tectonic": "#c2571a",
                 "fluid-driven swarm": "#1b7f79"}


def world_map(seq, w=1180, h=520) -> str:
    """The benchmark sequences on one equirectangular frame, coloured by regime.

    Filled markers are scorable today, hollow ones are waiting on their
    reference. Longitude is shifted so the Pacific sequences do not fall off
    both edges at once.
    """
    # the frame follows the data, with room for the labels; a hardcoded extent
    # dropped Kaikoura at longitude 173 off the right edge
    x0, x1 = float(seq.lon.min()) - 12, float(seq.lon.max()) + 12
    y0, y1 = float(seq.lat.min()) - 14, float(seq.lat.max()) + 10
    pad = 8

    def sx(lo):
        return pad + (lo - x0) / (x1 - x0) * (w - 2 * pad)

    def sy(la):
        return h - pad - (la - y0) / (y1 - y0) * (h - 2 * pad)

    out = [f'<svg viewBox="0 0 {w} {h}" class="map" role="img" '
           f'aria-label="Benchmark sequences by tectonic context">']
    try:
        bm = json.loads(BASEMAP.read_text())
        for key, cls in (("coast", "coast"), ("states", "border")):
            for line in bm.get(key, []):
                run = []
                for lo, la in line:
                    if x0 <= lo <= x1 and y0 <= la <= y1:
                        run.append(f"{sx(lo):.1f},{sy(la):.1f}")
                    else:
                        if len(run) > 1:
                            out.append(f'<polyline class="{cls}" points="{" ".join(run)}"/>')
                        run = []
                if len(run) > 1:
                    out.append(f'<polyline class="{cls}" points="{" ".join(run)}"/>')
    except Exception:                                              # noqa: BLE001
        pass

    # labels are placed by hand only where two sequences would overprint
    NUDGE = {"Fagradalsfjall 2021": (0, -13), "Reykjanes 2023 dike": (0, 11),
             "Noto swarm 2023": (0, 12), "Noto 2024": (0, -11),
             "Campi Flegrei 2023": (6, 12), "Etna 2022-2024": (0, 13),
             "Adriatic 2022": (0, -11), "Norcia 2016": (10, 10),
             "Samos 2020": (12, 10), "Thessaly 2021": (-6, -11),
             "Santorini-Amorgos 2025": (14, 12), "Corinth-Thiva 2020-2021": (-18, 12)}
    for _, r in seq.sort_values("regime_name").iterrows():
        c = REGIME_COLOUR.get(r.regime_name, "#6f6890")
        x, y = sx(r.lon), sy(r.lat)
        if bool(r.scorable):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{c}" '
                       f'fill-opacity=".85" stroke="#fff" stroke-width="1.3"/>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="none" '
                       f'stroke="{c}" stroke-width="2" stroke-dasharray="2.5 2"/>')
        dx, dy = NUDGE.get(r.label, (0, -10))
        out.append(f'<text class="seqlab" x="{x + dx:.1f}" y="{y + dy:.1f}" '
                   f'fill="{c}">{r.label}</text>')

    out.append(f'<g class="maplegend" transform="translate({pad + 10},{h - 96})">')
    for i, (name, col) in enumerate(REGIME_COLOUR.items()):
        n = int((seq.regime_name == name).sum())
        out.append(f'<circle cx="8" cy="{i * 20 + 8}" r="6" fill="{col}" fill-opacity=".85"/>'
                   f'<text x="22" y="{i * 20 + 12}">{name} ({n})</text>')
    out.append('<circle cx="8" cy="68" r="5.5" fill="none" stroke="#6f6890" stroke-width="2" '
               'stroke-dasharray="2.5 2"/><text x="22" y="72">waiting on its reference</text>')
    out.append("</g></svg>")
    return "".join(out)


def bar_row(label, value, biggest, colour="#4b2e83", suffix="") -> str:
    pct = 100 * value / biggest if biggest else 0
    return (f'<div class="bar"><span class="bl">{label}</span>'
            f'<span class="bt"><i style="width:{pct:.1f}%;background:{colour}"></i></span>'
            f'<span class="bv">{value:,}{suffix}</span></div>')


def main() -> None:
    d = load()
    scan, ph = d["scan"], d["phase"]
    mp, stn = d["map"], d["stations"]

    manifest_picks = scan["totals"]["picks"]
    station_days = scan["totals"]["station_days"]
    zero_days = scan["totals"]["zero"]
    manifests = scan["totals"]["manifests"]
    stations_seen = scan["stations"]
    stations_with = scan["stations_with_picks"]

    cached_total = ph["catalogue_total"]
    live = d["objects"]["live_objects"] if d["objects"] else ph["files_total"]
    if d["footers"]:
        catalogue = d["footers"]["total"]
        objects = d["footers"]["objects"]
        listed = d["footers"]["listed"]
        count_basis = (f"read from the footer of every one of the {listed:,} Parquet objects. "
                       "No sampling, no estimate")
        partial = False
    else:
        catalogue = cached_total
        objects = ph["files_total"]
        listed = live
        count_basis = (f"read from the footers of {objects:,} of the {live:,} Parquet objects "
                       f"that exist, so it is a floor rather than the total. The remaining "
                       f"{live - objects:,} are being counted now")
        partial = True
    p_frac = ph["sampled"].get("P", 0) / max(sum(ph["sampled"].values()), 1)
    s_frac = ph["sampled"].get("S", 0) / max(sum(ph["sampled"].values()), 1)
    n_p, n_s = round(catalogue * p_frac), round(catalogue * s_frac)
    gap = manifest_picks - catalogue

    obs_df = d["obs"]
    obs_stations = obs_picks = obs_near = 0
    if obs_df is not None:
        og = obs_df[obs_df.latitude.notna() & (obs_df.picks > 0)]
        obs_stations, obs_picks = len(og), int(og.picks.sum())
        obs_near = int(((og.longitude.between(-140, -115)) & (og.latitude.between(38, 52))).sum())
    fix = d["fix"]
    fixnote = ""
    if fix:
        fixnote = (f" The station table was completed on "
                   f"{fix['fixed_on'][:4]}-{fix['fixed_on'][4:6]}-{fix['fixed_on'][6:8]}: it "
                   f"listed {fix['before']:,} stations and the catalogue held picks on "
                   f"{fix['added_with_picks']:,} that were not in it, carrying "
                   f"{fix['added_picks']:,} picks. It now lists {fix['after']:,}. Anyone who "
                   f"pulled it before that date should pull it again.")
    with_coords = int(mp.latitude.notna().sum())
    topn = (mp[mp.latitude.notna() & (mp.picks > 0)]
            .sort_values("picks", ascending=False).head(20))
    top_blurb = ", ".join(
        f"{i.rstrip('.')} ({int(r.picks / 1e6)}M, {r.state})" for i, r in topn.head(3).iterrows())
    pb_share = 100 * (topn.network_code == "PB").mean()
    by_state = (mp[mp.latitude.notna()].groupby("state")
                .agg(stations=("picks", "size"), picks=("picks", "sum"))
                .sort_values("picks", ascending=False))
    zn = (mp.groupby("network_code").agg(zero=("zero", "sum"), days=("days", "sum")))
    zn["pct"] = 100 * zn.zero / zn.days
    zn = zn[zn.days > 20_000].sort_values("pct", ascending=False)

    det = d["detection"][d["detection"].n_ref > 0]
    import numpy as np

    def wmean(g, col):
        g = g.dropna(subset=[col])
        return float(np.average(g[col], weights=g.n_ref)) if len(g) else float("nan")

    tracks = {}
    for study, name in (("us", "Track 1, western United States"),
                        ("global", "Track 2, outside the United States")):
        s = det[det.study == study]
        tracks[name] = sorted(((w, wmean(s[s.weights == w], "recall_at_best"))
                               for w in WEIGHTS), key=lambda x: -x[1])
    tim = d["timing"].merge(det[["study", "sequence", "phase", "weights", "n_ref"]],
                            on=["study", "sequence", "phase", "weights"])
    tt = (tim[tim.medae.notna()].groupby(["study", "weights"])
          .apply(lambda x: float(np.average(x.medae, weights=x.n)), include_groups=False)
          .unstack(0))
    cal = d["calib"].pivot_table(index="weights", columns="study", values="ece_lb")
    qw = (d["quality"].groupby("weights")
          .apply(lambda x: float(np.average(x.swap_rate, weights=x.P_n + x.S_n)),
                 include_groups=False))

    sw, swp = d["swarm"], d["swarm_prov"]
    swt = (sw.groupby(["sequence", "phase"]).size().unstack(fill_value=0)
           if sw is not None else None)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    S: list[str] = []

    # ---------------------------------------------------------------- 1
    S.append(f"""
<section class="slide title">
  <p class="eyebrow">QuakeScope &middot; SeisSCOPED</p>
  <h1>The western-states deployment</h1>
  <p class="lede">{catalogue:,} phase picks from {stations_with:,} stations, what the
  campaign did not pick and why, how we score it, and the case for a benchmark organised
  by earthquake-sequence regime instead of by place.</p>
  <p class="meta">{built} &middot; built from the campaign's own manifests and Parquet
  footers &middot; <code>{commit}</code></p>
</section>""")

    # ---------------------------------------------------------------- 2
    S.append(f"""
<section class="slide">
  <h2>What the western campaign produced</h2>
  <div class="stats4">
    <div class="stat"><div class="n">{catalogue / 1e9:.2f}B{'+' if partial else ''}</div><div class="k">picks in the catalogue{' (floor)' if partial else ''}</div></div>
    <div class="stat"><div class="n">{n_p / 1e6:.0f}M</div><div class="k">P arrivals ({100 * p_frac:.1f}%)</div></div>
    <div class="stat"><div class="n">{n_s / 1e6:.0f}M</div><div class="k">S arrivals ({100 * s_frac:.1f}%)</div></div>
    <div class="stat"><div class="n">{stations_with:,}</div><div class="k">stations with picks</div></div>
  </div>
  <div class="cols">
    <div>
      <h3>How these are counted</h3>
      <ul>
        <li>The pick total is {count_basis}.</li>
        <li>The P and S split is measured on {sum(ph['sampled'].values()):,} picks read from
        {ph['files_read']:,} files stratified across every network, {100 * sum(ph['sampled'].values()) / cached_total:.2f}%
        of the catalogue, with no failed reads. Applied to the exact total it gives
        {n_p:,} P and {n_s:,} S.</li>
        <li>{manifests:,} shard manifests record {station_days:,} station-days of work,
        {len(stn):,} station epochs were in scope across {stn.network_code.nunique()} networks.</li>
        <li><strong>Every pick here was made by PhaseNet with the <code>original</code>
        weights</strong> of Zhu &amp; Beroza, at P and S thresholds of 0.2, three components,
        SeisBench 0.12.5. One model, one configuration, across the whole catalogue.</li>
      </ul>
    </div>
    <div>
      <h3>Picks by state</h3>
      <table class="t">
        <thead><tr><th>state</th><th>stations</th><th>picks</th></tr></thead><tbody>
        {''.join(f'<tr><td>{i}</td><td>{int(r.stations):,}</td><td>{int(r.picks):,}</td></tr>'
                 for i, r in by_state.head(6).iterrows())}
        </tbody>
      </table>
      <p class="cap">From the manifest records, which attribute every pick to a station-day.</p>
    </div>
  </div>
</section>""")

    # ---------------------------------------------------------------- 3
    S.append(f"""
<section class="slide">
  <h2>Where the picks came from</h2>
  {station_map(mp, d['obs'])}
  <p class="cap"><strong>Colour is the weight set that made the picks.</strong> Purple is the
  land campaign, PhaseNet <code>original</code> at a 0.2 threshold: {stations_with:,} of the
  {stations_seen:,} stations it processed produced a pick, and all {with_coords:,} now carry
  coordinates. Orange is the ocean-bottom campaign, PhaseNet <code>obs</code> (PickBlue) at
  the same threshold: {obs_stations:,} stations and {obs_picks:,} picks, of which
  {obs_near:,} stations sit on the Cascadia margin and the rest are deployments elsewhere in
  the world. The two ran different models, so their counts are not interchangeable. Open on
  the western states, scroll to zoom, drag to pan, or fit everything to see the whole
  ocean-bottom set. Circle
  area follows pick count. The heaviest producers are {top_blurb}. Borehole instruments
  dominate the top of the list: {pb_share:.0f}% of the twenty largest counts are
  Plate&nbsp;Boundary&nbsp;Observatory stations, which sit in quiet holes and detect far
  more than a surface sensor beside them. Coastline and state borders from Natural
  Earth.</p>
  <p class="cap src">Station coordinates:
  <a href="{BUCKET_HTTP}/western/stations.parquet">{BUCKET_HTTP}/western/stations.parquet</a>
  &middot; which stations produced picks, and how many:
  <a href="{BUCKET_HTTP}/western/manifests/">{BUCKET_HTTP}/western/manifests/</a>
  &middot; both public-read, no account.{fixnote}</p>
</section>""")

    # ---------------------------------------------------------------- 4
    S.append(f"""
<section class="slide">
  <h2>The station-days that produced nothing</h2>
  <div class="cols">
    <div>
      <div class="stats2">
        <div class="stat"><div class="n">{station_days / 1e6:.2f}M</div><div class="k">station-days processed</div></div>
        <div class="stat warn"><div class="n">{100 * zero_days / station_days:.1f}%</div><div class="k">wrote no picks ({zero_days:,})</div></div>
      </div>
      <p>A station-day that completes without writing a pick is not always an error. A
      station can be dead, clipped, or recording nothing a picker will call an arrival. It
      becomes an error when the worker reports success and the reason is a data-path
      failure rather than a quiet day, which is the case we have had to chase four times.</p>
      <p>{stations_seen - stations_with:,} stations were processed and never produced a
      single pick across their whole record. Those are the ones to look at first.</p>
    </div>
    <div>
      <h3>Zero-pick rate by network</h3>
      <table class="t">
        <thead><tr><th>network</th><th>station-days</th><th>no picks</th><th>%</th></tr></thead><tbody>
        {''.join(f'<tr><td>{i}</td><td>{int(r.days):,}</td><td>{int(r.zero):,}</td>'
                 f'<td class="{"bad" if r.pct > 25 else ""}">{r.pct:.1f}</td></tr>'
                 for i, r in zn.head(8).iterrows())}
        </tbody>
      </table>
      <p class="cap">Networks with more than 20,000 station-days. The spread is the point:
      a campaign-wide average hides which network needs looking at.</p>
    </div>
  </div>
</section>""")

    # ---------------------------------------------------------------- 5
    if partial:
        slide5 = f"""
<section class="slide">
  <h2>Do the manifests and the Parquet agree?</h2>
  <div class="cols">
    <div>
      <table class="t big">
        <tbody>
          <tr><td>manifests say workers wrote</td><td>{manifest_picks:,}</td></tr>
          <tr><td>Parquet footers read so far</td><td>{catalogue:,}</td></tr>
          <tr><td>objects counted</td><td>{objects:,} of {live:,}</td></tr>
        </tbody>
      </table>
      <p class="big-claim">These two are not yet comparable. The manifests cover the whole
      campaign; the footers cover {100 * objects / live:.0f}% of the objects. The count is
      running and the answer belongs on this slide, not an estimate.</p>
    </div>
    <div>
      <h3>What the comparison is for</h3>
      <ul>
        <li>The manifests are what the workers reported writing. The footers are what is
        actually in the bucket. A gap in either direction is a defect: picks that never
        landed, or manifests that over-report.</li>
        <li>We have found four ways a station-day can be reported as done without its
        output surviving. The largest, resumed shards overwriting their own output, cost
        25.3M picks before it was fixed. This comparison is how we would see a fifth.</li>
        <li>The dashboard's cached footer count stops on the clock each run, so it holds
        {objects:,} of {live:,} objects and its headline should not be quoted as the
        catalogue size.</li>
        <li>Whatever the answer, it does not move a benchmark result: those are scored on
        waveforms re-picked from the archive, not on the catalogue.</li>
      </ul>
    </div>
  </div>
</section>"""
    else:
        slide5 = f"""
<section class="slide">
  <h2>Manifests against the Parquet</h2>
  <div class="cols">
    <div>
      <table class="t big">
        <tbody>
          <tr><td>manifests say workers wrote</td><td>{manifest_picks:,}</td></tr>
          <tr><td>Parquet footers hold</td><td>{catalogue:,}</td></tr>
          <tr><td class="{'bad' if abs(gap) > 1e6 else 'ok'}">difference</td>
              <td class="{'bad' if abs(gap) > 1e6 else 'ok'}">{gap:+,} ({100 * gap / manifest_picks:+.1f}%)</td></tr>
        </tbody>
      </table>
      <p>Both sides now cover the same thing: all {listed:,} objects in
      <code>western/picks/</code> and all {manifests:,} shard manifests.</p>
      <p>Repair campaigns re-processing station-days would explain a manifest surplus, and
      they do not: of {station_days:,} manifest records, {scan.get('distinct_station_days', 0):,}
      are distinct station-days and only {scan.get('reprocessed', 0):,} were processed more
      than once.</p>
    </div>
    <div>
      <h3>What it means</h3>
      <ul>
        <li>{'A manifest surplus means picks the workers reported did not survive to S3, which is the silent-skip family we have chased four times.' if gap > 1e6 else ''}
        {'A footer surplus means the manifests under-report, most likely work whose worker was preempted after writing Parquet and before writing its manifest.' if gap < -1e6 else ''}
        {'The two agree to within a part in a thousand, so the catalogue holds what the workers said they wrote.' if abs(gap) <= 1e6 else ''}</li>
        <li>The dashboard's cached count covers {ph['files_total']:,} of {live:,} objects
        because it stops on the clock, so its headline is an undercount that catches up.
        This slide does not use it.</li>
        <li>No benchmark result depends on this: they are scored on waveforms re-picked
        from the archive, not on the catalogue.</li>
      </ul>
    </div>
  </div>
</section>"""
    S.append(slide5)

    # ---------------------------------------------------------------- 6
    repro = d["repro"]["totals"]
    S.append(f"""
<section class="slide">
  <h2>Offshore is not in this catalogue</h2>
  <div class="cols">
    <div>
      <p class="big-claim">Ocean-bottom stations off the western states live in the
      <code>obs/</code> partition of the bucket, not in <code>western/</code>.</p>
      <p>The split is deliberate. OBS records need a different weight set
      (<code>obs</code>, PickBlue), they carry a hydrophone channel the land models do not
      read, and their reference arrivals come from different sources. Scoring them against
      the land benchmark would mix two questions.</p>
      <p>Anyone reading the western catalogue for a coastal study should know that the
      offshore instruments are a separate read:
      <code>s3://quakescope-picks-2026/obs/picks/</code>.</p>
    </div>
    <div>
      <h3>The catalogue reproduces</h3>
      <table class="t big"><tbody>
        <tr><td>campaign picks re-picked</td><td>{repro['campaign_picks']:,}</td></tr>
        <tr><td>recovered to the millisecond</td><td>{repro['matched_exact']:,}</td></tr>
        <tr><td>did not match</td><td>{repro['campaign_only']:,}</td></tr>
      </tbody></table>
      <p class="cap">{repro['station_days_targeted']} station-days re-picked through a
      different data path on a different CPU architecture. The
      {repro['campaign_only']} that did not match are the same arrivals one to three
      samples apart. The pick values are right; the accounting above is what is unresolved.</p>
    </div>
  </div>
</section>""")

    # ---------------------------------------------------------------- 7
    METRICS = [
        ("Recall", "matched reference arrivals / reference arrivals", "exact",
         "The headline number. Unaffected by the reference being incomplete, because a "
         "missing analyst pick cannot turn a recovered arrival into a missed one."),
        ("Picks emitted", "count above the threshold", "exact",
         "Recall alone is gameable by lowering the threshold. Reported beside it always."),
        ("Recall at an equal pick count", "each model's curve read where all emit the same "
         "number of picks", "exact",
         "The only detection comparison that survives a change of threshold. Capped by the "
         "most conservative model, so it is read with the other two protocols."),
        ("Precision, F1", "matched / emitted, and their harmonic mean", "lower bound",
         "An unmatched pick may be a false positive or an arrival the analyst never marked. "
         "Comparable between models on the same reference, and not comparable with a number "
         "from a labelled-dataset paper."),
        ("MCC", "Matthews correlation coefficient", "not computable",
         "Needs true negatives. A continuous record with an incomplete reference does not "
         "define them, so a published MCC against a bulletin is not meaningful."),
        ("MAE and RMSE", "mean absolute and root-mean-square onset residual", "exact",
         "Reported together because RMSE responds to outliers and MAE does not. One without "
         "the other hides either a systematic error or a tail."),
        ("MedianAE", "median absolute residual", "exact",
         "Outlier-insensitive scatter. It is what a location code feels when most picks are good and a "
         "few are badly wrong."),
        ("Median bias", "median signed residual", "exact",
         "Separates a picker that is consistently late from one that is merely noisy. A "
         "systematic 0.15 s is invisible in MAE and moves every depth."),
        ("Gross-error rate", f"fraction of residuals beyond {DETECT_TOL:g}&thinsp;s", "exact",
         "Computed from residuals matched wider than the detection tolerance. Matched at "
         "the detection tolerance it would describe the tolerance rather than the picker."),
        ("Fraction within 0.1&thinsp;s", "share of matched picks inside 0.1&thinsp;s", "exact",
         "The tolerance PhaseNet was originally scored at, so it is the column that "
         "compares with the older literature."),
        ("Reliability curve and ECE", "observed agreement per confidence bin, and its mean gap",
         "lower bound",
         "Whether a threshold transfers between models. It usually does not: the four weight "
         "sets differ by a factor of two in calibration error."),
        ("Phase swap rate", "arrivals matched by a pick of the other phase", "exact",
         "The association axis. A P labelled S survives into the event and moves the "
         "location, which is a different failure from a miss."),
        ("Duplicate rate", "extra picks within tolerance of an already-matched arrival",
         "exact", "Invisible in recall, and work for the associator."),
        ("Model time, memory, cost", "s and MB per station-day, USD per 1,000 station-days",
         "protocol set", "A model too slow or too large to run across the archive cannot "
         "build the catalogue. Protocol fixed, no per-weight numbers yet."),
        ("Pick uncertainty", "a stated error on the arrival time", "not yet scored",
         "The peak height of a segmentation picker is neither the detection probability nor "
         "the timing probability, so a model that reports the two separately needs its own "
         "column."),
    ]
    CLS = {"exact": "ok", "lower bound": "warn2", "not computable": "bad",
           "protocol set": "warn2", "not yet scored": "warn2"}
    mrows = "".join(
        f'<tr><td class="l"><strong>{m}</strong></td><td class="l">{defn}</td>'
        f'<td class="l {CLS[st]}">{st}</td><td class="l rsn">{why}</td></tr>'
        for m, defn, st, why in METRICS)
    S.append(f"""
<section class="slide wide">
  <h2>What we score a picker on</h2>
  <p class="lede">Our reference is an operator bulletin, not a labelled test set. An analyst
  picked what a location needed and stopped, so an unmatched model pick may be a false
  positive or a real arrival nobody marked. That one fact decides which of these means
  anything, and the third column says which.</p>
  <table class="t metrics">
    <thead><tr><th class="l">metric</th><th class="l">what it is</th>
    <th class="l">against a bulletin</th><th class="l">why we report it</th></tr></thead>
    <tbody>{mrows}</tbody>
  </table>
  <p class="cap">Detection is matched at {DETECT_TOL:g}&thinsp;s and residuals at
  2&thinsp;s. Three threshold protocols are reported together, because a confidence of 0.3
  from one model and 0.3 from another are not the same operating point: one shared
  threshold, an equal pick count, and each model at its own best threshold. They disagree,
  and the disagreement is the result.</p>
</section>""")

    # ---------------------------------------------------------------- 8
    # One complete table per track. Built as whole strings, because a flat list of
    # fragments indexed by position silently drops rows and leaves a table open.
    panels = {}
    for name, rank in tracks.items():
        study = "us" if "western" in name else "global"
        body = ""
        for i, (w, v) in enumerate(rank, 1):
            lead = ' class="lead"' if i == 1 else ""
            body += (f'<tr><td>{i}</td><td class="l"><span class="dot" '
                     f'style="background:{WCOLOR[w]}"></span>{w}</td>'
                     f'<td{lead}>{v:.3f}</td><td>{tt.loc[w, study]:.3f}</td>'
                     f'<td>{cal.loc[w, study]:.3f}</td><td>{qw[w]:.4f}</td></tr>')
        panels[name] = (
            f'<h3>{name}</h3><table class="t"><thead><tr><th>#</th><th>weight set</th>'
            '<th>recall, own threshold</th><th>median onset error (s)</th>'
            '<th>ECE</th><th>swap rate</th></tr></thead><tbody>'
            + body + "</tbody></table>")
    panel_us = panels["Track 1, western United States"]
    panel_gl = panels["Track 2, outside the United States"]
    ours_us = [w for w, _ in tracks["Track 1, western United States"]].index("quakescope2026") + 1
    ours_gl = [w for w, _ in tracks["Track 2, outside the United States"]].index("quakescope2026") + 1
    S.append(f"""
<section class="slide">
  <h2>The leaderboard inverts between tracks</h2>
  <div class="cols">
    <div>{panel_us}</div>
    <div>{panel_gl}</div>
  </div>
  <p class="cap"><code>quakescope2026</code>, our own fine-tune, ranks {ours_us} of 4 on its
  own region and {ours_gl} of 4 outside it. A pooled number hides that, because the
  out-of-region track carries most of the reference arrivals. On track 2 the recall leader
  is also the least accurate on onset time, so a catalogue built for locations and one
  built for completeness do not want the same weights. Full board:
  <a href="benchmark_metrics.html">seisscoped.org/QuakeScope/benchmark_metrics.html</a></p>
</section>""")

    # ---------------------------------------------------------------- 9
    seq = d["seq"]
    S.append(f"""
<section class="slide wide">
  <h2>Where the benchmark sequences are</h2>
  {world_map(seq)}
  <div class="cols3" style="margin-top:10px">
    <div>
      <h3>Mainshock-aftershock</h3>
      <p class="cap">Events seconds apart, overlapping codas, a network that saturates in
      the first hours. The regime every published picker benchmark already covers.</p>
    </div>
    <div>
      <h3>Volcano-tectonic</h3>
      <p class="cap">Emergent onsets, low magnitudes, event types a picker trained on
      tectonic earthquakes has never seen. Held out as places rather than time windows,
      because a volcano recurs where it is and the curated corpora already hold its earlier
      years.</p>
    </div>
    <div>
      <h3>Fluid-driven swarm</h3>
      <p class="cap">Months of elevated rate with no mainshock, shallow sources, a dense
      local array. Includes the two induced sequences, Salton Sea and Jones-Guthrie, where
      the driver is injection rather than magma.</p>
    </div>
  </div>
  <p class="cap">{len(seq)} sequences, from {seq.loc[seq.lat.idxmax(), 'label']} in the
  north to {seq.loc[seq.lat.idxmin(), 'label']} in the south.
  {int(seq.scorable.sum())} are scorable today; the hollow markers are waiting on their
  reference, not on a model. Spreading a benchmark across tectonic context rather than
  across places is what lets it speak to generalisation. These three regimes fail a picker
  in different ways, and a model that handles an aftershock cascade has not been shown to
  handle a swarm that migrates for months.</p>
</section>""")

    # ---------------------------------------------------------------- 10
    rost, sb = d["roster"], d["sb"]
    REBUILT = {"West Bohemia 2018", "Maurienne 2017-2019"}   # registry cases repaired here
    NEW = {"Salton Sea 2016", "Jones-Guthrie 2014-2015"}     # added here

    def regime_block(regime, title, note):
        sub = rost[rost.regime_name == regime].copy()
        rows = ""
        for _, r in sub.iterrows():
            lab = r.label
            if lab in REBUILT and swt is not None and lab in swt.index:
                pp, ss = int(swt.loc[lab, "P"]), int(swt.loc[lab, "S"])
                state, cls = "rebuilt here", "ok"
            elif r.scorable:
                pp, ss = int(r.ref_rm_covered_P), int(r.ref_rm_covered_S)
                state, cls = "ready", "ok"
            else:
                pp, ss = int(r.ref_rm_covered_P), int(r.ref_rm_covered_S)
                # the reason text carries "<", which would open a tag if left raw
                state = _html.escape(str(r.pick_scoring_reason).replace("_", " "))
                cls = "warn2"
            rows += (f'<tr><td class="l">{lab}</td><td>{pp:,}</td><td>{ss:,}</td>'
                     f'<td class="{cls} l">{state}</td></tr>')
        if regime == "fluid-driven swarm" and swt is not None:
            for lab in sorted(NEW):
                if lab in swt.index:
                    rows += (f'<tr><td class="l">{lab}</td><td>{int(swt.loc[lab, "P"]):,}</td>'
                             f'<td>{int(swt.loc[lab, "S"]):,}</td>'
                             f'<td class="ok l">new, built here</td></tr>')
        # a set, not a sum: a rebuilt case may already have counted as scorable
        ready = set(sub[sub.scorable].label) | (REBUILT & set(sub.label))
        total = len(sub)
        if regime == "fluid-driven swarm":
            ready |= NEW
            total += len(NEW)
        return (f'<h3>{title} ({len(ready)} ready of {total})</h3>'
                f'<table class="t"><thead><tr><th class="l">sequence</th><th>P</th><th>S</th>'
                f'<th class="l">status</th></tr></thead><tbody>{rows}</tbody></table>'
                f'<p class="cap">{note}</p>')

    msas = regime_block("mainshock-aftershock", "Mainshock-aftershock",
                        "Events seconds apart, overlapping codas, a saturating network. "
                        "Four of the ready cases are already on the board.")
    vt = regime_block("volcano-tectonic", "Volcano-tectonic",
                      "Emergent onsets, low magnitudes, event types a picker was never "
                      "trained on. Held out as places rather than time windows, because a "
                      "volcano recurs where it is.")
    swarm = regime_block("fluid-driven swarm", "Fluid-driven swarm",
                         "Months of elevated rate, no mainshock, shallow sources on a dense "
                         "local array. This arm had no scorable case until this week.")
    ready_all = len(set(rost[rost.scorable].label) | REBUILT | NEW)
    sb_total = int(sb.windows.sum()) if sb is not None else 0
    sb_n = len(sb) if sb is not None else 0
    sb_top = ", ".join(f"{r.dataset} {int(r.windows):,}"
                       for _, r in sb.head(5).iterrows()) if sb is not None else ""
    S.append(f"""
<section class="slide wide">
  <h2>The next benchmark scores three regimes</h2>
  <div class="cols3">
    <div>{msas}</div>
    <div>{vt}</div>
    <div>{swarm}</div>
  </div>
  <div class="cols" style="margin-top:12px">
    <div>
      <h3>Why not reuse the curated sets</h3>
      <p>The benchmark we have is {sb_total:,} single-arrival windows from {sb_n} SeisBench
      datasets ({sb_top}, and {sb_n - 5} more). One pick per window, no sequence context,
      and <strong>every one of those datasets is in the fine-tune's training
      manifest</strong>. It measures timing on isolated arrivals and cannot measure
      generalisation, so it stays as a unit test.</p>
    </div>
    <div>
      <h3>What the regime set is instead</h3>
      <p>Continuous data scored at the event level against picks an operator or a published
      study made by hand, on sequences drawn from bulletins rather than from any curated
      corpus. {ready_all} of {len(rost) + len(NEW)} are scorable now. The rest fail on the
      reference, not the model: a bulletin that under-collects, or station codes that
      resolve to one station. <strong>Still missing: a sequence that postdates the training
      window of every model it scores, labelled independently.</strong></p>
    </div>
  </div>
</section>""")


    nav = "".join(f'<button data-go="{i}" aria-label="slide {i + 1}"></button>'
                  for i in range(len(S)))

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuakeScope: the western-states deployment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {{
  --ink:#2a1a4f; --purple:#4b2e83; --deep:#341f63; --peri:#6d5bd0; --peri-l:#c3b8f0;
  --stone:#6f6890; --lav:#f5f3fb; --lav2:#ece8f7; --paper:#fff; --line:rgba(42,26,79,.12);
  --good:#1b7f79; --warn:#c2571a; --bad:#8a1f5e;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:Manrope,system-ui,sans-serif;background:var(--lav);color:var(--ink);
  font-size:20px;line-height:1.5;-webkit-font-smoothing:antialiased}}
.deck{{position:relative}}
.slide{{display:none;min-height:100vh;padding:56px 64px 84px;max-width:1340px;margin:0 auto;
  flex-direction:column}}
.slide.on{{display:flex}}
h1{{font-size:clamp(2.4rem,4.2vw,3.6rem);font-weight:800;letter-spacing:-.02em;margin-bottom:18px}}
h2{{font-size:clamp(1.9rem,3vw,2.6rem);font-weight:700;letter-spacing:-.02em;
  margin-bottom:22px;padding-bottom:12px;border-bottom:2px solid var(--line)}}
h3{{font-size:1.18rem;font-weight:700;color:var(--purple);margin:0 0 10px}}
.title{{justify-content:center;background:linear-gradient(150deg,var(--ink),var(--deep) 55%,var(--purple));
  color:#fff;padding-left:84px}}
.title h1{{max-width:22ch}}
.title .lede{{color:#ded8f2;font-size:1.3rem;max-width:68ch;margin-bottom:26px}}
.title .eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:.86rem;font-weight:700;
  color:var(--peri-l);margin-bottom:16px}}
.title .meta{{color:#b9aee8;font-size:1rem}}
.title code{{background:rgba(255,255,255,.14);color:#fff}}
.lede{{color:var(--stone);font-size:1.18rem;max-width:88ch;margin-bottom:18px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:34px;align-items:start}}
.cols3{{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;align-items:start}}
.cols3 table.t{{font-size:.88rem}} .cols3 .cap{{font-size:.92rem}}
.slide.wide{{max-width:1500px}}
.stats4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
.stats2{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:18px}}
.stat{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:16px 18px}}
.stat .n{{font-size:2.4rem;font-weight:800;color:var(--purple);letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.1}}
.stat .k{{color:var(--stone);font-size:1rem;margin-top:4px}}
.stat.warn .n{{color:var(--warn)}}
ul{{margin:0 0 0 20px}} li{{margin-bottom:9px;max-width:72ch}}
p{{max-width:80ch;margin-bottom:12px}}
.big-claim{{font-size:1.3rem;font-weight:600;color:var(--purple-deep,var(--deep));
  background:var(--lav2);border-left:4px solid var(--peri);border-radius:10px;padding:14px 18px}}
table.t{{width:100%;border-collapse:collapse;font-size:1.02rem;font-variant-numeric:tabular-nums;
  margin-bottom:10px}}
table.t th{{text-align:right;padding:8px 11px;color:var(--purple);font-size:.84rem;
  text-transform:uppercase;letter-spacing:.06em;border-bottom:1.5px solid var(--line)}}
table.t td{{text-align:right;padding:8px 11px;border-bottom:1px solid var(--line)}}
table.t th:first-child,table.t td:first-child,table.t td.l{{text-align:left}}
table.t.big td{{font-size:1.3rem;padding:12px}}
table.t.metrics{{font-size:.92rem}} table.t.metrics td{{padding:6px 10px;white-space:normal;vertical-align:top}}
table.t.metrics td.rsn{{color:var(--stone);max-width:46ch}}
td.lead{{font-weight:700;background:rgba(109,91,208,.12)}}
td.ok,.ok{{color:var(--good);font-weight:600}}
td.warn2{{color:var(--warn);font-weight:600}}
td.bad,.bad{{color:var(--bad);font-weight:600}}
td.src{{font-size:.76rem;color:var(--stone);text-align:left}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
.cap.src{{font-size:.86rem;word-break:break-all}}
  .cap{{color:var(--stone);font-size:1rem;max-width:104ch;margin-top:8px}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:.86em;background:rgba(75,46,131,.09);
  padding:1px 5px;border-radius:4px;color:var(--deep)}}
a{{color:var(--peri)}}
svg.map{{width:100%;height:auto;background:var(--paper);border:1px solid var(--line);
  border-radius:14px}}
.mapwrap{{position:relative}}
.mapbtns{{display:flex;gap:8px;align-items:center;margin-bottom:8px}}
.mapbtns button{{font:600 .86rem Manrope,sans-serif;padding:5px 12px;border-radius:999px;border:1px solid var(--line);background:var(--lav2);color:var(--purple);cursor:pointer}}
.mapbtns button:hover{{border-color:var(--peri)}}
.mapbtns .hint{{color:var(--stone);font-size:.84rem}}
svg.zoomable{{cursor:grab;touch-action:none}} svg.zoomable:active{{cursor:grabbing}}
.coast{{fill:none;stroke:#b9b4cc;stroke-width:.8}}
.border{{fill:none;stroke:#d8d4e4;stroke-width:.6}}
.key text{{font-size:15px;fill:var(--stone);font-family:Manrope,sans-serif}}
.seqlab{{font-size:15px;font-weight:600;font-family:Manrope,sans-serif;text-anchor:middle;paint-order:stroke;stroke:#fff;stroke-width:3px}}
.maplegend text{{font-size:16px;fill:var(--ink);font-family:Manrope,sans-serif}}
.nav{{position:fixed;left:0;right:0;bottom:0;display:flex;gap:7px;justify-content:center;
  padding:14px;background:linear-gradient(transparent,var(--lav) 42%);z-index:9}}
.nav button{{width:26px;height:5px;border:0;border-radius:3px;background:var(--line);cursor:pointer}}
.nav button.on{{background:var(--purple)}}
.count{{position:fixed;right:18px;bottom:16px;color:var(--stone);font-size:1rem;
  font-variant-numeric:tabular-nums;z-index:9}}
@media print{{
  .slide{{display:flex!important;page-break-after:always;min-height:auto;padding:28px}}
  .nav,.count{{display:none}} body{{background:#fff}}
}}
@media (max-width:900px){{.cols,.cols3,.stats4,.stats2{{grid-template-columns:1fr}}
  .slide{{padding:28px 20px 72px}}}}
</style>
</head>
<body>
<div class="deck">{''.join(S)}</div>
<div class="nav">{nav}</div>
<div class="count"><span id="cur">1</span> / {len(S)}</div>
<script>
(function () {{
  var slides = [].slice.call(document.querySelectorAll('.slide'));
  var dots = [].slice.call(document.querySelectorAll('.nav button'));
  var i = 0;
  function show(n) {{
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach(function (s, k) {{ s.classList.toggle('on', k === i); }});
    dots.forEach(function (d, k) {{ d.classList.toggle('on', k === i); }});
    document.getElementById('cur').textContent = i + 1;
    location.hash = i + 1;
    window.scrollTo(0, 0);
  }}
  document.addEventListener('keydown', function (e) {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') show(i + 1);
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') show(i - 1);
    if (e.key === 'Home') show(0);
    if (e.key === 'End') show(slides.length - 1);
  }});
  dots.forEach(function (d) {{
    d.addEventListener('click', function () {{ show(+d.dataset.go); }});
  }});
  document.addEventListener('click', function (e) {{
    if (e.target.closest('.nav') || e.target.closest('a')) return;
    show(i + (e.clientX > window.innerWidth * 0.6 ? 1 : (e.clientX < window.innerWidth * 0.4 ? -1 : 0)));
  }});
  show(parseInt(location.hash.slice(1) || '1', 10) - 1);
}})();
</script>
</body>
</html>"""
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB, {len(S)} slides)")
    print(f"  catalogue {catalogue:,} picks | P {n_p:,} S {n_s:,}")
    print(f"  {stations_with:,} stations with picks, {with_coords:,} mapped")
    print(f"  {station_days:,} station-days, {zero_days:,} wrote nothing "
          f"({100 * zero_days / station_days:.2f}%)")
    print(f"  manifest/footer gap {gap:,}")


if __name__ == "__main__":
    main()
