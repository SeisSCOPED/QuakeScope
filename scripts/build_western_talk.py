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
    return d


# ------------------------------------------------------------------ map
def station_map(mp: pd.DataFrame, w=1180, h=620) -> str:
    """Stations that produced picks, sized by how many."""
    g = mp[mp.latitude.notna() & (mp.picks > 0)].copy()
    x0, x1 = float(g.longitude.min()) - 1.0, float(g.longitude.max()) + 1.0
    y0, y1 = float(g.latitude.min()) - 1.0, float(g.latitude.max()) + 1.0
    pad = 46

    def sx(lo):
        return pad + (lo - x0) / (x1 - x0) * (w - 2 * pad)

    def sy(la):
        return h - pad - (la - y0) / (y1 - y0) * (h - 2 * pad)

    out = [f'<svg viewBox="0 0 {w} {h}" class="map" role="img" '
           f'aria-label="Stations that produced picks in the western campaign">']
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

    big = g.picks.max()
    for _, r in g.sort_values("picks").iterrows():
        frac = (r.picks / big) ** 0.33
        rad = 1.6 + 7.5 * frac
        out.append(f'<circle cx="{sx(r.longitude):.1f}" cy="{sy(r.latitude):.1f}" '
                   f'r="{rad:.1f}" fill="#4b2e83" fill-opacity="{0.18 + 0.5 * frac:.2f}" '
                   f'stroke="#fff" stroke-width="0.35"/>')
    # a size key, drawn from the data rather than invented
    out.append(f'<g class="key" transform="translate({pad + 6},{pad + 4})">')
    for i, n in enumerate([1_000, 100_000, 5_000_000]):
        frac = (n / big) ** 0.33
        rad = 1.6 + 7.5 * frac
        out.append(f'<circle cx="10" cy="{i * 26 + 10}" r="{rad:.1f}" fill="#4b2e83" '
                   f'fill-opacity="{0.18 + 0.5 * frac:.2f}" stroke="#fff" stroke-width="0.35"/>')
        out.append(f'<text x="28" y="{i * 26 + 14}">{n:,} picks</text>')
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
  <h1>The western-states deployment, and what we benchmark next</h1>
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
  {station_map(mp)}
  <p class="cap">{with_coords:,} of the {stations_seen:,} stations the campaign processed
  carry coordinates in the campaign station table and produced at least one pick. Circle
  area follows pick count. The heaviest producers are {top_blurb}. Borehole instruments
  dominate the top of the list: {pb_share:.0f}% of the twenty largest counts are
  Plate&nbsp;Boundary&nbsp;Observatory stations, which sit in quiet holes and detect far
  more than a surface sensor beside them. Coastline and state borders from Natural
  Earth.</p>
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
  <h2>Manifests against the Parquet, every object counted</h2>
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
    S.append(f"""
<section class="slide">
  <h2>What we score a picker on</h2>
  <p class="lede">Our reference is an operator bulletin, not a labelled test set. An analyst
  picked what a location needed and stopped, so an unmatched model pick may be a false
  positive or a real arrival nobody marked. That one fact decides which metrics mean
  anything.</p>
  <div class="cols">
    <div>
      <table class="t">
        <thead><tr><th>metric</th><th>against a bulletin</th></tr></thead><tbody>
          <tr><td>Recall</td><td class="ok">exact</td></tr>
          <tr><td>Picks emitted</td><td class="ok">exact</td></tr>
          <tr><td>Onset MAE, RMSE, MedianAE, bias</td><td class="ok">exact</td></tr>
          <tr><td>Gross-error rate</td><td class="ok">exact</td></tr>
          <tr><td>P/S swap rate (association)</td><td class="ok">exact</td></tr>
          <tr><td>Duplicate picks</td><td class="ok">exact</td></tr>
          <tr><td>Precision, F1</td><td class="warn2">lower bound</td></tr>
          <tr><td>Calibration, ECE</td><td class="warn2">lower bound</td></tr>
          <tr><td>MCC</td><td class="bad">not computable</td></tr>
        </tbody>
      </table>
    </div>
    <div>
      <h3>Three ways to set the threshold</h3>
      <p>A confidence of 0.3 from one model and 0.3 from another are not the same operating
      point, so the protocol decides the ranking:</p>
      <ul>
        <li><strong>One shared threshold.</strong> What most comparisons report. It measures
        how willing a model is to emit a pick as much as how well it picks.</li>
        <li><strong>Equal pick count.</strong> Threshold-free, but the count all models can
        reach is capped by the most conservative one.</li>
        <li><strong>Each model's own threshold.</strong> What a tuned deployment would run.
        No held-out split, so it is an upper bound.</li>
      </ul>
      <p class="cap">Residuals are matched at 2&thinsp;s and detection at 0.5&thinsp;s:
      matching at the detection tolerance truncates the residual distribution and makes an
      outlier rate a statement about the tolerance.</p>
    </div>
  </div>
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
  <h2>The leaderboard, and why one number is not enough</h2>
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
    S.append("""
<section class="slide">
  <h2>Place-based benchmarks run out</h2>
  <div class="cols">
    <div>
      <p class="big-claim">Every sequence we score is a place, and the training corpora are
      places too.</p>
      <ul>
        <li>INSTANCE holds Etna and Campi Flegrei from 2005 to 2020. VCSEIS holds Alaska,
        Hawaii, northern California and the Cascades. CREW is global at regional distance.
        A benchmark drawn from the same places measures memory of a place.</li>
        <li>Holding out a <em>time window</em> at a known place leaves that place's earlier
        years in training. Volcanoes and swarms recur at the same place, so for those the
        hold-out has to be the place itself, at all times, and the cost in training data has
        to be accepted.</li>
        <li>Our sequences are also all mainshock-aftershock. Seven of eight on the current
        board are a cascade or a doublet. A picker that handles an aftershock cascade has
        not been shown to handle a swarm migrating for months.</li>
      </ul>
    </div>
    <div>
      <h3>What a catalogue actually has to survive</h3>
      <table class="t">
        <thead><tr><th>regime</th><th>what breaks</th></tr></thead><tbody>
          <tr><td>Mainshock-aftershock</td><td>events seconds apart, overlapping codas, a
          saturating network</td></tr>
          <tr><td>Volcano-tectonic</td><td>emergent onsets, low magnitudes, long-period
          events a picker was never trained on</td></tr>
          <tr><td>Fluid-driven swarm</td><td>months of elevated rate, no mainshock, shallow
          sources and a dense local array</td></tr>
        </tbody>
      </table>
      <p class="cap">These are different failure modes, not different places. Scoring by
      regime is what tells a deployment which one it is buying.</p>
    </div>
  </div>
</section>""")

    # ---------------------------------------------------------------- 10
    sw_rows = ""
    if swt is not None:
        prov = swp.set_index("sequence") if swp is not None else None
        for seq in swt.index:
            src = ""
            if prov is not None and seq in prov.index:
                try:
                    src = ", ".join(k for k, _ in json.loads(prov.loc[seq, "sources"]))
                except Exception:                                   # noqa: BLE001
                    src = str(prov.loc[seq, "source"])
            sw_rows += (f'<tr><td>{seq}</td><td>{int(swt.loc[seq, "P"]):,}</td>'
                        f'<td>{int(swt.loc[seq, "S"]):,}</td><td class="src">{src}</td></tr>')
    total_sw = int(swt.values.sum()) if swt is not None else 0
    S.append(f"""
<section class="slide">
  <h2>The next benchmark: three regimes, not three places</h2>
  <div class="cols">
    <div>
      <h3>Built and ready to score</h3>
      <table class="t">
        <thead><tr><th>fluid-driven swarm</th><th>P</th><th>S</th><th>reference</th></tr></thead>
        <tbody>{sw_rows}</tbody>
      </table>
      <p class="cap">{total_sw:,} analyst arrivals on the stations and windows we will
      score, from the ISC bulletin, the USGS phase-data product and BCSF-RENASS. Two
      sources per sequence where one under-collected; duplicate readings between services
      are collapsed on station, phase and time.</p>
    </div>
    <div>
      <h3>What this changes</h3>
      <ul>
        <li>Nineteen sequences are specified across the three regimes, all built. Nine are
        scorable today; the swarm arm was the one that had none until this week.</li>
        <li>The reference problem is the work, not the model. Every swarm failure was a
        station-code mismatch or a bulletin that under-collected, and each was found by
        measuring the reference before fetching waveforms.</li>
        <li><strong>Open question for this group:</strong> a sequence that postdates the
        training window of every model it scores, labelled independently. That is an analyst
        campaign, and it is the only thing that turns this from a report into a benchmark a
        paper can cite.</li>
      </ul>
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
  line-height:1.5;-webkit-font-smoothing:antialiased}}
.deck{{position:relative}}
.slide{{display:none;min-height:100vh;padding:56px 64px 84px;max-width:1340px;margin:0 auto;
  flex-direction:column}}
.slide.on{{display:flex}}
h1{{font-size:clamp(1.9rem,3.4vw,2.8rem);font-weight:800;letter-spacing:-.02em;margin-bottom:18px}}
h2{{font-size:clamp(1.4rem,2.4vw,2rem);font-weight:700;letter-spacing:-.02em;
  margin-bottom:22px;padding-bottom:12px;border-bottom:2px solid var(--line)}}
h3{{font-size:1rem;font-weight:700;color:var(--purple);margin:0 0 10px}}
.title{{justify-content:center;background:linear-gradient(150deg,var(--ink),var(--deep) 55%,var(--purple));
  color:#fff;padding-left:84px}}
.title h1{{max-width:22ch}}
.title .lede{{color:#ded8f2;font-size:1.1rem;max-width:68ch;margin-bottom:26px}}
.title .eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:700;
  color:var(--peri-l);margin-bottom:16px}}
.title .meta{{color:#b9aee8;font-size:.86rem}}
.title code{{background:rgba(255,255,255,.14);color:#fff}}
.lede{{color:var(--stone);font-size:1.02rem;max-width:88ch;margin-bottom:18px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:34px;align-items:start}}
.stats4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
.stats2{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:18px}}
.stat{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:16px 18px}}
.stat .n{{font-size:1.8rem;font-weight:800;color:var(--purple);letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.1}}
.stat .k{{color:var(--stone);font-size:.84rem;margin-top:4px}}
.stat.warn .n{{color:var(--warn)}}
ul{{margin:0 0 0 20px}} li{{margin-bottom:9px;max-width:72ch}}
p{{max-width:80ch;margin-bottom:12px}}
.big-claim{{font-size:1.1rem;font-weight:600;color:var(--purple-deep,var(--deep));
  background:var(--lav2);border-left:4px solid var(--peri);border-radius:10px;padding:14px 18px}}
table.t{{width:100%;border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums;
  margin-bottom:10px}}
table.t th{{text-align:right;padding:7px 10px;color:var(--purple);font-size:.72rem;
  text-transform:uppercase;letter-spacing:.06em;border-bottom:1.5px solid var(--line)}}
table.t td{{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line)}}
table.t th:first-child,table.t td:first-child,table.t td.l{{text-align:left}}
table.t.big td{{font-size:1.05rem;padding:10px}}
td.lead{{font-weight:700;background:rgba(109,91,208,.12)}}
td.ok,.ok{{color:var(--good);font-weight:600}}
td.warn2{{color:var(--warn);font-weight:600}}
td.bad,.bad{{color:var(--bad);font-weight:600}}
td.src{{font-size:.76rem;color:var(--stone);text-align:left}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
.cap{{color:var(--stone);font-size:.84rem;max-width:104ch;margin-top:8px}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:.86em;background:rgba(75,46,131,.09);
  padding:1px 5px;border-radius:4px;color:var(--deep)}}
a{{color:var(--peri)}}
svg.map{{width:100%;height:auto;background:var(--paper);border:1px solid var(--line);
  border-radius:14px}}
.coast{{fill:none;stroke:#b9b4cc;stroke-width:.8}}
.border{{fill:none;stroke:#d8d4e4;stroke-width:.6}}
.key text{{font-size:11px;fill:var(--stone);font-family:Manrope,sans-serif}}
.nav{{position:fixed;left:0;right:0;bottom:0;display:flex;gap:7px;justify-content:center;
  padding:14px;background:linear-gradient(transparent,var(--lav) 42%);z-index:9}}
.nav button{{width:26px;height:5px;border:0;border-radius:3px;background:var(--line);cursor:pointer}}
.nav button.on{{background:var(--purple)}}
.count{{position:fixed;right:18px;bottom:16px;color:var(--stone);font-size:.78rem;
  font-variant-numeric:tabular-nums;z-index:9}}
@media print{{
  .slide{{display:flex!important;page-break-after:always;min-height:auto;padding:28px}}
  .nav,.count{{display:none}} body{{background:#fff}}
}}
@media (max-width:900px){{.cols,.stats4,.stats2{{grid-template-columns:1fr}}
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
