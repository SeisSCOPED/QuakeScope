#!/usr/bin/env python
"""Final report pages for the 2026 catalogues: reports/western_2026.html, reports/obs_2026.html.

Every number is read from s3://quakescope-picks-2026 when the page is built:

  picks       exact, from the Parquet itself: `_archive/inventory-<date>/picks/<cat>/<net>.json.gz`,
              written by the Batch inventory job (rows, picks per station and per day,
              per object), never from the dashboard cache
  manifests   `_archive/inventory-<date>/manifests/<cat>.json.gz`, compared object by
              object and station by station with the Parquet
  outcomes    `<cat>/availability/`, one row per station-day
  stations    `<cat>/stations.parquet` (coordinates, joined on `id`)
  queues      `_queues/<name>/` for every queue fleet.json points at the catalogue

Cost is an ESTIMATE from compute hours at a rate calibrated on CloudBank's billed
costs (docs/rerun_2026/34, "What it cost"); it is passed in, not measured here.

    python scripts/build_campaign_reports.py --inventory _archive/inventory-20261013
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import html
import io
import json
import math
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "quakescope-picks-2026"
s3 = boto3.client("s3", region_name="us-east-2",
                  config=Config(retries={"max_attempts": 15, "mode": "adaptive"}, max_pool_connections=32))

# Sequential blue (dataviz reference palette), five bins of log10 picks per station.
SEQ = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

META = {
    "western": dict(
        title="QuakeScope 2026 — western catalogue",
        lede=("Machine-learning P and S picks for the western United States and its neighbours, "
              "1986 to September 2026: Washington, Oregon, California, Nevada, Idaho and Wyoming, "
              "plus Utah, Montana, Arizona, Colorado, New Mexico, British Columbia, Alberta, "
              "Baja California and Sonora."),
        model="PhaseNet, <code>original</code> weights (SeisBench 0.12.5, weight version 2), "
              "P and S thresholds 0.2, components ZNE12",
        cost="~$4,200", cost_range="$3,700 to $4,800",
        not_read=("Networks our EarthScope account may not read (TD, EO, LH); NP strong-motion "
                  "(no data in a 1,408-day sample, not re-read); 673,676 station-days in scattered "
                  "short gaps; 328 station-days whose day files crash the miniSEED decoder "
                  "(SF.PH004-006 2006-07, X7.LB01 2008) and 678 whose day object exceeds 4 GB."),
    ),
    "obs": dict(
        title="QuakeScope 2026 — ocean-bottom catalogue",
        lede=("PickBlue picks on ocean-bottom seismometers, 1993 to 2026, offshore stations only: "
              "since 2026-10-03 the catalogue holds stations selected per station by a land mask and "
              "water depth, not by network code."),
        model="PickBlue <code>obs</code> weights (PhaseNet architecture, SeisBench 0.12.5), "
              "P and S thresholds 0.2; the short-period EL band is picked for OBS only",
        cost="~$120", cost_range="$90 to $170",
        not_read=("NV (Ocean Networks Canada), refused to our account; network-years absent from "
                  "EarthScope's S3 archive (most obs `not_found`); 18,832 station-days in short gaps."),
    ),
}


def _retry(fn, tries=8):
    # The pages are often built from a laptop; a dropped connection mid-read
    # should cost a retry, not the build.
    import time
    for a in range(tries):
        try:
            return fn()
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(min(60, 2 ** a))


def get_bytes(key):
    return _retry(lambda: s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def get_json_gz(key):
    return json.loads(gzip.decompress(get_bytes(key)))


def listing(prefix):
    return [o for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
            for o in pg.get("Contents", [])]


def load_inventory(inv, cat):
    objs = []
    for o in listing(f"{inv}/picks/{cat}/"):
        objs += get_json_gz(o["Key"])
    return objs


def queue_table(cat):
    fleet = json.load(open(ROOT / "fleet.json"))["campaigns"]
    rows = []
    for name, c in fleet.items():
        if not c.get("parquet_uri", "").rstrip("/").endswith(f"/{cat}"):
            continue
        q = c["queue"].split(f"s3://{BUCKET}/", 1)[1].rstrip("/")
        try:
            body = get_bytes(f"{q}/shards.jsonl").decode()
        except Exception:
            continue
        sh = [json.loads(x) for x in body.splitlines() if x.strip()]
        n = lambda sub: sum(pg.get("KeyCount", 0) for pg in s3.get_paginator("list_objects_v2")
                            .paginate(Bucket=BUCKET, Prefix=f"{q}/{sub}/"))
        first = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{q}/shards.jsonl")["Contents"][0]["LastModified"]
        rows.append(dict(queue=name, written=first.date().isoformat(), shards=len(sh),
                         planned_sd=sum(s.get("n_station_days", 0) for s in sh),
                         complete=n("complete"), blocked=n("blocked")))
    return sorted(rows, key=lambda r: r["written"])


def audit(objs, manifests, cat):
    """Manifests against the Parquet: object by object, then station by station."""
    actual = {o["key"]: o["rows"] for o in objs}
    listed = {}
    n_mismatch_np = 0
    man_station = collections.Counter()
    for m in manifests:
        if m.get("err"):
            continue
        for path, rows in m.get("files", []):
            key = path.split(f"{BUCKET}/", 1)[-1]
            listed[key] = rows
        if (m.get("n_picks") or 0) != sum(r or 0 for _, r in m.get("files", [])):
            n_mismatch_np += 1
        man_station.update(m.get("per_station", {}))
    orphans = [k for k in actual if k not in listed]
    missing = [k for k in listed if k not in actual]
    row_diff = [k for k in listed if k in actual and actual[k] is not None and listed[k] != actual[k]]
    pq_station = collections.Counter()
    for o in objs:
        pq_station.update(o.get("per_station") or {})
    st_diff = {t: (pq_station.get(t, 0), man_station.get(t, 0))
               for t in set(pq_station) | set(man_station) if pq_station.get(t, 0) != man_station.get(t, 0)}
    return dict(
        manifests=len(manifests), unreadable=sum(1 for m in manifests if m.get("err")),
        objects=len(actual), unread_objects=sum(1 for v in actual.values() if v is None),
        orphans=len(orphans), orphan_rows=sum(actual[k] or 0 for k in orphans),
        missing=len(missing), missing_rows=sum(listed[k] or 0 for k in missing),
        row_mismatch=len(row_diff), n_picks_mismatch=n_mismatch_np,
        picks_parquet=sum(v or 0 for v in actual.values()), picks_manifests=sum(listed.values()),
        stations_differ=len(st_diff),
        worst=sorted(st_diff.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))[:10],
        orphan_examples=orphans[:5], missing_examples=missing[:5],
    ), pq_station


def outcomes(cat):
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    fs = pafs.S3FileSystem(region="us-east-2")
    t = _retry(lambda: ds.dataset(f"{BUCKET}/{cat}/availability", filesystem=fs, format="parquet",
                                  partitioning="hive").to_table(columns=["status", "npks"]).to_pandas())
    vc = t.status.value_counts().to_dict()
    vc["loaded, zero picks"] = int(((t.status == "loaded") & (t.npks == 0)).sum())
    return vc, len(t)


# ------------------------------------------------------------------ html

def fmt(n):
    return f"{n:,}"


def bars_svg(items, width=720, bar_h=22, label_w=170, value_fmt=fmt, note=""):
    """Horizontal bars, one series: magnitude by category. Hover shows the value."""
    mx = max(v for _, v in items) or 1
    h = len(items) * (bar_h + 6) + 8
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="{html.escape(note)}">']
    for i, (k, v) in enumerate(items):
        y = 4 + i * (bar_h + 6)
        w = max(2, (width - label_w - 110) * v / mx)
        out.append(f'<text x="{label_w - 8}" y="{y + bar_h * 0.7}" class="lab" text-anchor="end">{html.escape(str(k))}</text>'
                   f'<rect class="bar" x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" '
                   f'data-tip="{html.escape(str(k))}: {value_fmt(v)}"></rect>'
                   f'<text x="{label_w + w + 6:.1f}" y="{y + bar_h * 0.7}" class="val">{value_fmt(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def year_svg(per_year, width=720, height=220):
    years = sorted(per_year)
    if not years:
        return ""
    mx = max(per_year.values())
    pad_l, pad_b = 56, 22
    bw = (width - pad_l - 8) / len(years)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Picks per year">']
    for frac in (0.5, 1.0):
        y = (height - pad_b) * (1 - frac)
        out.append(f'<line x1="{pad_l}" x2="{width}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="axis" text-anchor="end">{mx * frac / 1e6:.0f} M</text>')
    for i, yr in enumerate(years):
        v = per_year[yr]
        bh = (height - pad_b) * v / mx
        x = pad_l + i * bw
        out.append(f'<rect class="bar" x="{x + 1:.1f}" y="{height - pad_b - bh:.1f}" width="{max(1, bw - 2):.1f}" '
                   f'height="{bh:.1f}" rx="2" data-tip="{yr}: {fmt(v)} picks"></rect>')
        if yr % 5 == 0:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{height - 6}" class="axis" text-anchor="middle">{yr}</text>')
    out.append("</svg>")
    return "".join(out)


CSS = """
:root{--surface:#fcfcfb;--panel:#f3f2ef;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;--rule:#e2e0db;--bar:#2a78d6;--accent:#2a78d6;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#242422;--text:#f0efec;--text2:#c3c2b7;--muted:#8f8e86;--rule:#33322f;--bar:#3987e5;--accent:#3987e5;}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#242422;--text:#f0efec;--text2:#c3c2b7;--muted:#8f8e86;--rule:#33322f;--bar:#3987e5;--accent:#3987e5;}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 64px}h1{font-size:1.9rem;margin:.2em 0}h2{font-size:1.25rem;margin:2.2em 0 .5em;border-top:1px solid var(--rule);padding-top:1em}
.lede{color:var(--text2);max-width:72ch}.meta{color:var(--muted);font-size:.9rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}
.tile{background:var(--panel);border-radius:10px;padding:14px}.tile .k{color:var(--text2);font-size:.85rem}.tile .v{font-size:1.55rem;font-weight:650;margin:2px 0}.tile .n{color:var(--muted);font-size:.8rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.6em 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}#map{height:520px;border-radius:10px;margin:10px 0}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.85rem;color:var(--text2)}
.sw{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:5px;vertical-align:-1px;border:2px solid var(--surface)}
.chart{width:100%;height:auto}.chart .bar{fill:var(--bar)}.chart .bar:hover{opacity:.8}.chart .lab,.chart .val,.chart .axis{fill:var(--text2);font-size:12px}.chart .grid{stroke:var(--rule)}
.tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:4px 8px;border-radius:6px;font-size:.8rem;display:none;z-index:9999}
.ok{color:#1a7f37}.warn{color:#b35900}code{font-size:.88em}details{margin:.6em 0}summary{cursor:pointer;color:var(--text2)}
"""


def page(cat, meta, a, pq_station, oc, n_avail, queues, stations, inv_objs, built):
    picks = a["picks_parquet"]
    size_gb = sum(o["size"] for o in inv_objs) / 1e9
    with_picks = sum(1 for v in pq_station.values() if v > 0)
    loaded = oc.get("loaded", 0)
    per_year = collections.Counter()
    for o in inv_objs:
        for k, v in (o.get("per_day") or {}).items():
            per_year[int(k.split("|")[0])] += v

    # Map points: stations with picks, binned by log10 picks.
    st = stations.set_index("id")
    pts, edges = [], [1e3, 1e4, 1e5, 1e6]
    for tid, n in pq_station.items():
        if tid in st.index and n > 0:
            r = st.loc[tid]
            b = sum(n >= e for e in edges)
            pts.append([round(float(r.latitude), 4), round(float(r.longitude), 4), tid, int(n), b])
    no_picks = [[round(float(r.latitude), 4), round(float(r.longitude), 4), i]
                for i, r in st.iterrows() if pq_station.get(i, 0) == 0 and pd.notna(r.latitude)]
    top = sorted(pq_station.items(), key=lambda kv: -kv[1])[:20]

    status_order = ["loaded", "loaded, zero picks", "no_data", "not_found", "empty_read", "no_channel",
                    "denied", "unread", "unknown"]
    explain = {"loaded": "read and picked", "loaded, zero picks": "data present, no arrival above 0.2 (subset of loaded)",
               "no_data": "the archive listing held nothing", "not_found": "the archive has no such network-year",
               "empty_read": "an object exists but holds no pickable band", "no_channel": "the station offers no pickable band",
               "denied": "our account may not read it", "unread": "attempted, not read (see Gaps)",
               "unknown": "inside the station's epochs, no record either way"}
    oc_items = [(k, oc[k]) for k in status_order if oc.get(k)]
    audit_ok = (a["orphans"] == 0 and a["missing"] == 0 and a["row_mismatch"] == 0 and a["stations_differ"] == 0)

    def tile(k, v, n):
        return f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{n}</div></div>'

    q_rows = "".join(f'<tr><td><code>{q["queue"]}</code></td><td>{q["written"]}</td><td class="num">{fmt(q["shards"])}</td>'
                     f'<td class="num">{fmt(q["planned_sd"])}</td><td class="num">{fmt(q["complete"])}</td>'
                     f'<td class="num">{fmt(q["blocked"])}</td></tr>' for q in queues)
    oc_rows = "".join(f'<tr><td>{k}</td><td>{explain.get(k, "")}</td><td class="num">{fmt(v)}</td></tr>' for k, v in oc_items)
    top_rows = "".join(f'<tr><td><code>{t}</code></td><td class="num">{fmt(n)}</td></tr>' for t, n in top)
    yr_rows = "".join(f'<tr><td>{y}</td><td class="num">{fmt(per_year[y])}</td></tr>' for y in sorted(per_year))
    worst = "".join(f'<tr><td><code>{t}</code></td><td class="num">{fmt(p)}</td><td class="num">{fmt(m)}</td></tr>'
                    for t, (p, m) in a["worst"])
    dup = DUPCHECK.get(cat)
    if dup and a["orphans"]:
        frac = dup["duplicated"] / dup["rows"]
        est = frac * a["orphan_rows"]
        verdict_dup = (f'<p class="warn"><strong>The Parquet holds duplicate picks.</strong> {fmt(a["orphans"])} objects '
                       f'({fmt(a["orphan_rows"])} picks) are listed by no manifest: files from shard attempts that Spot '
                       f'interrupted before the shard was re-run. In a random sample of {dup["sampled"]} of them, '
                       f'{100 * frac:.1f}% of rows also appear in a listed file ({dup["fully_duplicated_objects"]} objects '
                       f'fully duplicated, {dup["partly"]} partly, {dup["not_duplicated"]} not at all). Estimated unique '
                       f'picks: <strong>~{(a["picks_parquet"] - est) / 1e9:.3f} billion</strong> of {fmt(a["picks_parquet"])} '
                       f'rows. Until they are removed, de-duplicate on <code>(tid, cha, pha, peak)</code> when reading.</p>')
    else:
        verdict_dup = ""
    verdict = verdict_dup or ('<p class="ok"><strong>Manifests agree with the Parquet.</strong></p>' if audit_ok else
               '<p class="warn"><strong>Manifests and Parquet disagree somewhere; details below.</strong> '
               'The Parquet is authoritative: counts on this page come from it.</p>')
    legend = "".join(f'<span><i class="sw" style="background:{c}"></i>{lab}</span>' for c, lab in
                     zip(SEQ, ["&lt; 1,000", "1,000 to 10,000", "10,000 to 100,000", "100,000 to 1 M", "&ge; 1 M"]))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(meta['title'])}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>{CSS}</style></head>
<body><main>
<p class="meta"><a href="index.html">QuakeScope reports</a> · built {built} from <code>s3://{BUCKET}/{cat}/</code></p>
<h1>{html.escape(meta['title'])}</h1>
<p class="lede">{meta['lede']}</p>
<div class="tiles">
{tile("Picks", fmt(picks), "exact: every Parquet object read")}
{tile("Stations with picks", fmt(with_picks), f"of {fmt(len(stations))} station-locations in the table")}
{tile("Station-days analyzed", fmt(loaded), "read and run through the picker")}
{tile("Stored", f"{size_gb:,.1f} GB", f"{fmt(len(inv_objs))} Parquet objects")}
{tile("Compute cost (est.)", meta['cost'], f"range {meta['cost_range']}; rate calibrated on CloudBank billing")}
</div>

<h2>Picks per station</h2>
<p class="meta">Each dot is a station-location, shaded by its number of picks; grey crosses are stations in the table with no picks. Hover for the station and count.</p>
<div class="legend">{legend}<span><i class="sw" style="background:var(--muted)"></i>no picks</span></div>
<div id="map"></div>
<details><summary>Table: the 20 stations with the most picks</summary><div class="scroll"><table><tr><th>station</th><th class="num">picks</th></tr>{top_rows}</table></div></details>

<h2>Picks per year</h2>
{year_svg(per_year)}
<details><summary>Table</summary><div class="scroll"><table><tr><th>year</th><th class="num">picks</th></tr>{yr_rows}</table></div></details>

<h2>What happened to every station-day</h2>
<p class="meta">From <code>{cat}/availability/</code>, one row per station-day inside a station's operating epochs ({fmt(n_avail)} rows). Statuses before image d0ccf9b are coarser: those shards recorded only what they read, which is why <em>unknown</em> exists.</p>
{bars_svg(oc_items, note="Station-day outcomes")}
<div class="scroll"><table><tr><th>status</th><th>meaning</th><th class="num">station-days</th></tr>{oc_rows}</table></div>

<h2>Do the manifests match the data?</h2>
{verdict}
<div class="scroll"><table>
<tr><td>Parquet objects read</td><td class="num">{fmt(a['objects'])}</td></tr>
<tr><td>Picks in the Parquet</td><td class="num">{fmt(a['picks_parquet'])}</td></tr>
<tr><td>Picks the manifests say they wrote</td><td class="num">{fmt(a['picks_manifests'])}</td></tr>
<tr><td>Manifests read (unreadable)</td><td class="num">{fmt(a['manifests'])} ({a['unreadable']})</td></tr>
<tr><td>Objects in no manifest (rows)</td><td class="num">{fmt(a['orphans'])} ({fmt(a['orphan_rows'])})</td></tr>
<tr><td>Objects a manifest lists that are not in the bucket (rows)</td><td class="num">{fmt(a['missing'])} ({fmt(a['missing_rows'])})</td></tr>
<tr><td>Objects whose row count differs from its manifest</td><td class="num">{fmt(a['row_mismatch'])}</td></tr>
<tr><td>Manifests whose total differs from their files</td><td class="num">{fmt(a['n_picks_mismatch'])}</td></tr>
<tr><td>Stations whose pick count differs, Parquet vs manifest records</td><td class="num">{fmt(a['stations_differ'])}</td></tr>
</table></div>
<details><summary>Largest station differences (Parquet, manifests)</summary><table><tr><th>station</th><th class="num">Parquet</th><th class="num">manifests</th></tr>{worst}</table></details>

<h2>How it was made</h2>
<p>{meta['model']}. Picks are written per shard as Parquet partitioned by network, year and month; each shard writes a manifest; since image d0ccf9b every planned station-day also gets an outcome. The catalogue was built by these queues, all writing into the same prefix:</p>
<div class="scroll"><table><tr><th>queue</th><th>written</th><th class="num">shards</th><th class="num">planned station-days</th><th class="num">complete</th><th class="num">blocked</th></tr>{q_rows}</table></div>
<p>Why there are so many, and the guard each correction now has: <a href="https://github.com/SeisSCOPED/QuakeScope/blob/main/docs/rerun_2026/34_campaign_playbook.md">campaign playbook (doc 34)</a>.</p>

<h2>Gaps</h2><p>{meta['not_read']}</p>

<h2>Cost</h2>
<p><strong>Estimated</strong>, not billed: compute hours priced at $0.0213 per vCPU-hour, a rate <strong>calibrated on CloudBank's billed costs</strong> for three days of full load. The account cannot read AWS cost APIs. Storage of this catalogue is about ${size_gb * 0.023:,.2f} a month in S3 Standard; public downloads cost about $0.09 per GB once the account's free tier is used.</p>

<h2>Reading it</h2>
<p>Public, anonymous, no account: <a href="read_the_catalogue.html">reading tutorial</a> · <a href="download_the_catalogue.html">region and period</a> · <a href="https://github.com/SeisSCOPED/QuakeScope/blob/main/docs/data_access.md">data access guide</a>. Cite the prefix and the date you read it.</p>
</main>
<div class="tip" id="tip"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const PTS={json.dumps(pts)}, NONE={json.dumps(no_picks)}, SEQ={json.dumps(SEQ)};
const map=L.map('map',{{preferCanvas:true}});
const ESRI='https://services.arcgisonline.com/ArcGIS/rest/services';
L.tileLayer(ESRI+'/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}',{{attribution:'Esri',maxZoom:16}}).addTo(map);
const muted=getComputedStyle(document.documentElement).getPropertyValue('--muted').trim()||'#8a8984';
NONE.forEach(p=>L.circleMarker([p[0],p[1]],{{radius:2,color:muted,weight:1,fill:false}}).bindTooltip(p[2]+': no picks').addTo(map));
const layer=L.featureGroup();
PTS.sort((a,b)=>a[3]-b[3]).forEach(p=>L.circleMarker([p[0],p[1]],{{radius:4+p[4],color:'#ffffff',weight:1,fillColor:SEQ[p[4]],fillOpacity:.9}})
  .bindTooltip(p[2]+': '+p[3].toLocaleString()+' picks').addTo(layer));
layer.addTo(map); map.fitBounds(layer.getBounds(),{{padding:[10,10]}});
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{{
  el.addEventListener('mousemove',e=>{{tip.textContent=el.dataset.tip;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';}});
  el.addEventListener('mouseleave',()=>tip.style.display='none');}});
</script></body></html>"""


DUPCHECK: dict = {}


def main():
    global DUPCHECK
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--catalogues", default="western,obs")
    a = ap.parse_args()
    built = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    summary = {}
    try:   # orphan duplicate sample, written by the inventory's dupcheck job
        DUPCHECK = json.loads(get_bytes(f"{a.inventory}/dupcheck.json"))
    except Exception:
        DUPCHECK = {}
    for cat in a.catalogues.split(","):
        objs = load_inventory(a.inventory, cat)
        manifests = get_json_gz(f"{a.inventory}/manifests/{cat}.json.gz")
        aud, pq_station = audit(objs, manifests, cat)
        oc, n_avail = outcomes(cat)
        stations = pd.read_parquet(io.BytesIO(get_bytes(f"{cat}/stations.parquet")))
        queues = queue_table(cat)
        (ROOT / f"reports/{cat}_2026.html").write_text(
            page(cat, META[cat], aud, pq_station, oc, n_avail, queues, stations, objs, built))
        summary[cat] = {k: v for k, v in aud.items() if k not in ("worst",)} | {"outcomes": oc}
        print(cat, json.dumps(summary[cat], default=str)[:1200])
    json.dump(summary, open(ROOT / "reports/campaign_2026_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
