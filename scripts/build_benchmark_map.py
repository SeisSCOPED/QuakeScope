"""A Leaflet map of the benchmark: every event, every scored station.

Plain grey tiles so the markers carry the page, a colour per track and a shade per
event type within it, and a contamination statement per sequence. Numbers come
from docs/benchmark/results/map/, never typed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "docs" / "benchmark" / "results" / "map"
OUT = ROOT / "reports" / "benchmark_map.html"

TRACK = {
    "track1-western-us": dict(name="Track 1 — western United States", hue="#4b2e83"),
    "track2-msas":       dict(name="Track 2a — mainshock-aftershock, outside the US", hue="#b7472a"),
    "track2-vt":         dict(name="Track 2b — volcano-tectonic", hue="#c8842a"),
    "track2-swarm":      dict(name="Track 2c — fluid-driven swarm", hue="#17776b"),
}
# shade within a track, by event type
SHADE = {"mainshock-aftershock": 1.00, "mainshock-aftershock doublet": 0.78,
         "moderate single event": 0.60, "volcano-tectonic": 1.00,
         "fluid-driven swarm": 1.00, "swarm": 0.78, "induced swarm": 0.58}
RISK = {"high": "in a curated corpus's own region and era",
        "medium": "partly overlapping a curated corpus",
        "low": "no curated corpus identified for this region"}


def shade(hexcol: str, f: float) -> str:
    """Lighten a track colour toward white by (1 - f)."""
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: int(round(c * f + 255 * (1 - f)))               # noqa: E731
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def main() -> None:
    ev = pd.read_csv(MAP / "events.csv")
    st = pd.read_csv(MAP / "stations.csv")
    cont = pd.read_csv(MAP / "contamination.csv")
    st = st[st.lat.notna()]
    ev = ev[ev.lat.notna() & ev.lon.notna()]
    cat = dict(zip(cont.sequence, cont.category))
    risk = dict(zip(cont.sequence, cont.risk))

    seqs = []
    for seq, g in ev.groupby("sequence"):
        tr = g.track.iloc[0]
        c = cont[cont.sequence == seq]
        s = st[st.sequence == seq]
        seqs.append(dict(
            sequence=seq, track=tr, track_name=TRACK[tr]["name"],
            category=cat.get(seq, ""), risk=risk.get(seq, ""),
            colour=shade(TRACK[tr]["hue"], SHADE.get(cat.get(seq, ""), 0.85)),
            n_events=int(len(g)), n_stations=int(s.station.nunique()),
            arrivals=int(s.arrivals.sum()), P=int(s.P.sum()), S=int(s.S.sum()),
            mag_max=None if g.mag.isna().all() else round(float(g.mag.max()), 1),
            depth_med=None if g.depth.isna().all() else round(float(g.depth.median()), 1),
            corpus=(c.corpus.iloc[0] if len(c) else ""),
            basis=(c.basis.iloc[0] if len(c) else ""),
            events=[[round(float(r.lat), 4), round(float(r.lon), 4),
                     None if pd.isna(r.mag) else round(float(r.mag), 1),
                     None if pd.isna(r.depth) else round(float(r.depth), 1)]
                    for r in g.itertuples()],
            stations=[[round(float(r.lat), 4), round(float(r.lon), 4), r.station,
                       int(r.arrivals), int(r.P), int(r.S)]
                      for r in s.itertuples()],
        ))
    seqs.sort(key=lambda d: (d["track"], d["sequence"]))

    tot_e = sum(d["n_events"] for d in seqs)
    tot_a = sum(d["arrivals"] for d in seqs)
    tot_s = len({x[2] for d in seqs for x in d["stations"]})

    rows = "\n".join(
        f'<tr data-seq="{d["sequence"]}"><td><span class="sw" style="background:{d["colour"]}"></span>'
        f'{d["sequence"]}</td><td>{d["category"]}</td><td class="n">{d["n_events"]:,}</td>'
        f'<td class="n">{d["n_stations"]}</td><td class="n">{d["arrivals"]:,}</td>'
        f'<td class="n">{d["P"]:,}</td><td class="n">{d["S"]:,}</td>'
        f'<td class="r {d["risk"]}">{d["risk"]}</td><td class="src">{d["corpus"]}</td></tr>'
        for d in seqs)

    legend = "\n".join(
        f'<div class="lg"><b>{v["name"]}</b>' + "".join(
            f'<span class="li"><i style="background:{d["colour"]}"></i>{d["category"]}</span>'
            for d in {x["category"]: x for x in seqs if x["track"] == k}.values()) + "</div>"
        for k, v in TRACK.items() if any(x["track"] == k for x in seqs))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark map — Seismic Phase Picking Leaderboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="quakescope-board.css">
<style>
 #map {{ height: 620px; border-radius: 12px; border: 1px solid var(--line); }}
 .sw {{ display:inline-block; width:11px; height:11px; border-radius:50%;
        margin-right:7px; vertical-align:middle; }}
 .lg {{ margin: 6px 18px 6px 0; display:inline-block; font-size:.82rem; }}
 .lg b {{ display:block; font-size:.72rem; letter-spacing:.07em;
          text-transform:uppercase; color:var(--muted); margin-bottom:3px; }}
 .li {{ display:inline-flex; align-items:center; margin-right:12px; }}
 .li i {{ width:11px; height:11px; border-radius:50%; margin-right:5px; }}
 td.r {{ font-weight:600; }} td.r.high {{ color:#b7472a; }}
 td.r.medium {{ color:#c8842a; }} td.r.low {{ color:#17776b; }}
 td.src {{ font-size:.78rem; color:var(--muted); }}
 tr.on td {{ background: rgba(75,46,131,.07); }}
 .ctl {{ margin:10px 0 14px; display:flex; gap:8px; flex-wrap:wrap; }}
 .ctl button {{ font:inherit; font-size:.82rem; padding:5px 12px; cursor:pointer;
   border:1px solid var(--line); background:var(--card); color:var(--ink);
   border-radius:999px; }}
 .ctl button.sel {{ background:var(--ink); color:var(--bg); border-color:var(--ink); }}
</style></head><body>
<main class="shell">
<p class="eyebrow">HazEvalHub · catalogue-workflow track</p>
<h1>Where the benchmark sequences are</h1>
<p class="lede">{tot_e:,} located earthquakes and {tot_s} scored stations across
{len(seqs)} sequences and four tracks, carrying {tot_a:,} analyst arrivals.
Colour is the track; shade is the event type within it. Every sequence says
whether it falls inside a curated training corpus's own region and era.</p>

<div class="ctl" id="ctl"></div>
<div id="map"></div>
<div style="margin:12px 0 26px">{legend}</div>

<h2>The sequences</h2>
<table class="t"><thead><tr><th>sequence</th><th>type</th><th>events</th>
<th>stations</th><th>arrivals</th><th>P</th><th>S</th><th>curated-corpus overlap</th>
<th>which corpus</th></tr></thead><tbody>
{rows}
</tbody></table>
<p class="note">Overlap is a reasoned assessment from each corpus's stated
region and era, not a trace-by-trace diff. <b>high</b> means the sequence sits
inside a curated corpus's own region and era, so a model trained on that corpus
has seen this place at this time. <b>low</b> means no SeisBench corpus we can
enumerate covers the region. West Bohemia is the sharpest case: a SeisBench
corpus, BohemiaSaxony, covers exactly that region and contains restricted
WEBNET data, the same array scored here, and its time span could not be checked
because the data does not auto-download.</p>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const SEQ = {json.dumps(seqs)};
const map = L.map('map', {{ worldCopyJump: true }}).setView([30, 10], 2);
// Esri's World Light Gray Canvas, which is what a plain grey reference
// basemap is for. CARTO's tiles are NOT keyless: basemaps.cartocdn.com answers
// 200 with a 2,049-byte placeholder reading "API KEY REQUIRED" at every zoom,
// so a page using them looks fine in code review and ships a watermark. That
// is also why plotly's "carto-positron" style would not have helped: it pulls
// the same tiles. Every URL below was fetched and checked for real content.
const ESRI = 'https://services.arcgisonline.com/ArcGIS/rest/services';
const ATTR = 'Tiles &copy; <a href="https://www.esri.com">Esri</a>';
const base = L.tileLayer(`${{ESRI}}/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}`,
  {{ maxZoom: 16, attribution: ATTR }}).addTo(map);
const labels = L.tileLayer(`${{ESRI}}/Canvas/World_Light_Gray_Reference/MapServer/tile/{{z}}/{{y}}/{{x}}`,
  {{ maxZoom: 16, pane: 'shadowPane', attribution: '' }}).addTo(map);
const terrain = L.tileLayer(
  'https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png',
  {{ maxZoom: 16, attribution:
     'map data &copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors, '
     + '<a href="https://viewfinderpanoramas.org">SRTM</a> | style '
     + '<a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)' }});
L.control.layers({{ 'Grey': base, 'Terrain': terrain }},
                 {{ 'Place names': labels }}).addTo(map);

const layers = {{}}, bounds = {{}};
for (const d of SEQ) {{
  const g = L.layerGroup().addTo(map);
  const pts = [];
  for (const [la, lo, mag, dep] of d.events) {{
    L.circleMarker([la, lo], {{
      radius: mag == null ? 3 : Math.max(2.5, 1.7 + mag * 1.05),
      color: d.colour, weight: 1, fillColor: d.colour, fillOpacity: .5
    }}).bindPopup(
      `<b>${{d.sequence}}</b><br>${{d.category}}<br>` +
      (mag == null ? '' : `M ${{mag}}`) + (dep == null ? '' : ` · ${{dep}} km depth`)
    ).addTo(g);
    pts.push([la, lo]);
  }}
  for (const [la, lo, sta, n, p, s] of d.stations) {{
    L.marker([la, lo], {{ icon: L.divIcon({{ className: '', iconSize: [13, 13],
      html: `<div style="width:0;height:0;border-left:6.5px solid transparent;`
          + `border-right:6.5px solid transparent;border-bottom:11px solid ${{d.colour}};`
          + `filter:drop-shadow(0 0 1px rgba(255,255,255,.9))"></div>` }}) }})
      .bindPopup(`<b>${{sta}}</b><br>${{d.sequence}}<br>${{n}} arrivals (P ${{p}}, S ${{s}})`)
      .addTo(g);
    pts.push([la, lo]);
  }}
  layers[d.sequence] = g;
  bounds[d.sequence] = L.latLngBounds(pts);
}}

const all = L.latLngBounds([].concat(...SEQ.map(d =>
  d.events.map(e => [e[0], e[1]]).concat(d.stations.map(s => [s[0], s[1]])))));
map.fitBounds(all, {{ padding: [24, 24] }});

const ctl = document.getElementById('ctl');
const mk = (label, fn, sel) => {{
  const b = document.createElement('button');
  b.textContent = label; if (sel) b.className = 'sel';
  b.onclick = () => {{ [...ctl.children].forEach(x => x.className = '');
    b.className = 'sel'; fn(); }};
  ctl.appendChild(b); return b;
}};
mk('everything', () => {{ for (const d of SEQ) map.addLayer(layers[d.sequence]);
  map.fitBounds(all, {{ padding: [24, 24] }}); }}, true);
const TRACKS = {json.dumps({k: v["name"] for k, v in TRACK.items()})};
for (const [k, name] of Object.entries(TRACKS)) {{
  const mine = SEQ.filter(d => d.track === k);
  if (!mine.length) continue;
  mk(name.replace(/^Track /, ''), () => {{
    for (const d of SEQ) (d.track === k ? map.addLayer : map.removeLayer).call(map, layers[d.sequence]);
    map.fitBounds(L.latLngBounds([].concat(...mine.map(d =>
      d.events.map(e => [e[0], e[1]])))), {{ padding: [24, 24] }});
  }});
}}
document.querySelectorAll('tbody tr').forEach(tr => {{
  tr.style.cursor = 'pointer';
  tr.onclick = () => {{
    document.querySelectorAll('tbody tr').forEach(x => x.classList.remove('on'));
    tr.classList.add('on');
    const s = tr.dataset.seq;
    for (const d of SEQ) (d.sequence === s ? map.addLayer : map.removeLayer).call(map, layers[d.sequence]);
    map.fitBounds(bounds[s], {{ padding: [40, 40] }});
    document.getElementById('map').scrollIntoView({{ behavior: 'smooth', block: 'center' }});
  }};
}});
</script>
</main></body></html>
""")
    print(f"{OUT}  {OUT.stat().st_size/1024:,.0f} kB")
    print(f"  {len(seqs)} sequences, {tot_e:,} events, {tot_s} stations, {tot_a:,} arrivals")


if __name__ == "__main__":
    main()
