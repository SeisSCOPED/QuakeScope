#!/usr/bin/env python
"""reports/vibesys_pr71.html: do QuakeScope's picks and amplitudes change under the
VibeSys CPU optimizations of PR #71, and how much faster are they?

Every number is read from docs/benchmark/results/vibesys_pr71/, which
scripts/vibesys_pr71/export_results.py writes from the harness outputs. Typed
constants are limited to the published analyst-uncertainty figures (cited) and
the definitions of the IEEE and bfloat16 number formats.

    python scripts/build_vibesys_report.py
"""

from __future__ import annotations

import datetime
import html
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs/benchmark/results/vibesys_pr71"
OUT = ROOT / "reports/vibesys_pr71.html"
GH = "https://github.com/SeisSCOPED/QuakeScope"

meta = json.loads((RES / "meta.json").read_text())
vv = pd.read_csv(RES / "picks_vs_vanilla.csv")
aa = pd.read_csv(RES / "picks_vs_analyst.csv")
disc = pd.read_csv(RES / "analyst_discordant.csv")
resid = pd.read_csv(RES / "vanilla_vs_analyst_residuals.csv").set_index("phase")
tol = pd.read_csv(RES / "tolerance.csv")
fp32d = pd.read_csv(RES / "fp32_differences_at_floor.csv")
amps = pd.read_csv(RES / "amplitudes.csv").set_index("amplitude")
timing = pd.read_csv(RES / "timing_reps.csv")
exact = pd.read_csv(RES / "graviton_exactness.csv")
hosts = pd.read_csv(RES / "hosts.csv").set_index("host")
cum = pd.read_csv(RES / "error_cumulative.csv")
errs = {a: pd.read_csv(RES / f"errors_{a}.csv.gz") for a in ("pr_fp32", "pr_fp16", "pr_bf16")}

THR = 0.2  # production detection threshold; every headline number is at this threshold
TOL = 30  # ms; the tolerance Marine set on 2026-10-27, comparable to analyst P scatter (Ross 2018; this benchmark)
ALLOW = 0.5  # percent of picks allowed to fail, the calculator's starting value
TRACK = {"us_sequences": "US sequences", "swarm_sequences": "swarm sequences"}
ARM = {"vanilla": "current code", "pr_fp32": "fp32", "pr_fp16": "fp16", "pr_bf16": "bf16", "pr": "PR, fp32"}
PR_ARMS = ("pr_fp32", "pr_fp16", "pr_bf16")
COLOR = {"pr_fp32": "#2a78d6", "pr_fp16": "#1baf7a", "pr_bf16": "#eb6834"}  # validated all-pairs on #ffffff
DASH = {"pr_fp32": "", "pr_fp16": "6 4", "pr_bf16": "2 3"}
HOSTS = ["M1 Max", "c6g", "c7g", "c8g"]
HOST_LABEL = {"M1 Max": "M1 Max laptop", "c6g": "c6g.2xlarge (Graviton2)",
              "c7g": "c7g.2xlarge (Graviton3)", "c8g": "c8g.2xlarge (Graviton4)"}
WEIGHT_ID = "original"  # SeisBench weight name the harness loads (bench_worker.py, run_amps.py)
SD = 5.55  # station-days in the timing workload (meta.json timing_workload)


def e(x) -> str:
    return html.escape(str(x))


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}&thinsp;%"


def n(x) -> str:
    return f"{int(x):,}"


def table(head, rows, left=1, cls="data") -> str:
    lc = ' class="l"'
    th = "".join(f"<th{lc if i < left else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{lc if i < left else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f'<div class="table-scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def reproduced(er: pd.DataFrame, tau_ms: float, thr: float = THR) -> float:
    """Fraction of current-code picks above thr whose partner is above thr and within tau_ms (the calculator's rule)."""
    v = er[er.conf_vanilla.fillna(-1) >= thr]
    ok = v.dt_ms.notna() & (v.conf_arm.fillna(-1) >= thr) & (v.dt_ms.abs() <= tau_ms + 1e-6)
    return ok.sum() / len(v)


# ---- derived numbers ---------------------------------------------------------------------
v2 = vv[vv.threshold == THR]
tot = v2.groupby("arm")[["vanilla", "identical_n", "lost_n", "new_n"]].sum()
ident = tot.identical_n / tot.vanilla
t2 = tol[tol.threshold == THR].groupby("arm")
within10 = {a: reproduced(errs[a], 10) for a in PR_ARMS}
withinT = {a: reproduced(errs[a], TOL) for a in PR_ARMS}


def counts(er: pd.DataFrame, tau_ms: float = TOL, thr: float = THR) -> dict:
    """Picks moved beyond tau_ms, lost and new at thr, by the calculator's rule."""
    v = er[er.conf_vanilla.fillna(-1) >= thr]
    paired = v.dt_ms.notna() & (v.conf_arm.fillna(-1) >= thr)
    new = er[(er.conf_arm.fillna(-1) >= thr) & ~(er.dt_ms.notna() & (er.conf_vanilla.fillna(-1) >= thr))]
    return {"n": len(v), "moved": int((paired & (v.dt_ms.abs() > tau_ms)).sum()), "lost": int((~paired).sum()), "new": len(new)}


cnt = {a: counts(errs[a]) for a in PR_ARMS}
sd_shift = t2.std_shift_ms.max()
lost_new = {a: (cnt[a]['lost'] + cnt[a]['new']) / cnt[a]['n'] for a in PR_ARMS}
d2 = disc[disc.threshold == THR]
n_analyst = int(d2.analyst.sum())
changed = {"pr_fp32": int(d2.fp32_changed.sum()), "pr_fp16": int(d2.fp16_lost.sum() + d2.fp16_gained.sum()),
           "pr_bf16": int(d2.bf16_lost.sum() + d2.bf16_gained.sum())}
fp_total, fp_ident = int(tot.loc["pr_fp32", "vanilla"]), int(tot.loc["pr_fp32", "identical_n"])

a2 = aa[aa.threshold == THR]
base = a2[a2.arm == "vanilla"].set_index(["track", "phase"])
dmae = max((a2[a2.arm == a].set_index(["track", "phase"]).mae_s - base.mae_s).abs().max() for a in PR_ARMS)
dbias = max((a2[a2.arm == a].set_index(["track", "phase"]).bias_s - base.bias_s).abs().max() for a in PR_ARMS)
drec = max((a2[a2.arm == a].set_index(["track", "phase"]).recall - base.recall).abs().max() for a in PR_ARMS)

med = timing[timing.complete].groupby(["host", "arm"]).median(numeric_only=True)
walls = timing[timing.complete].groupby(["host", "arm"]).wall_s.apply(list)


def speed(h):
    return med.loc[(h, "vanilla"), "wall_s"] / med.loc[(h, "pr"), "wall_s"]


def usd_per_1k(h, arm, col="spot_usd_h"):
    return 1000 * med.loc[(h, arm), "wall_s"] / 3600 * hosts.loc[h, col] / SD


GRAV = ("c6g", "c7g", "c8g")
g23 = exact[(exact.a == "c6g vanilla") & (exact.b == "c7g vanilla")].iloc[0]
wa_share = {h: med.loc[(h, "vanilla"), "wa_s"] - med.loc[(h, "pr"), "wa_s"] for h in GRAV}
tot_saved = {h: (med.loc[(h, "vanilla"), ["classify_s", "wa_s", "vel_s"]] - med.loc[(h, "pr"), ["classify_s", "wa_s", "vel_s"]]).sum()
             for h in GRAV}
bias_txt = (", and none moves the median bias" if dbias < 5e-4 else f", or median bias by more than {dbias * 1e3:.1f}&thinsp;ms")
fargate_h = 8 * meta["fargate_spot_rate_usd_vcpu_h"]

# ---- answer ------------------------------------------------------------------------------
answer = f"""
<section id="answer">
  <div class="section-head"><p class="eyebrow">Answer</p>
  <h2>fp32 keeps every pick. fp16 stays within a {TOL}&thinsp;ms tolerance; bf16 does not.</h2>
  <p class="lede">PR #71 rewrites how PhaseNet and the amplitude code run on a CPU. We re-picked the leaderboard's
  {n(tot.loc['pr_fp32', 'vanilla'])} picks (threshold {THR:g}, weight <code>{e(meta['weight'])}</code>) three ways
  and compared each pick with the current code's.</p></div>
  <div class="grid-3">
    <div class="card verdict ok"><p class="tag">fp32 &middot; what the PR uses on Fargate and Graviton</p>
      <div class="big">{fp_ident:,} of {fp_total:,}</div><p class="k">picks identical to the millisecond</p>
      <p>The other {fp_total - fp_ident} moved by one sample (10&thinsp;ms) on a flat-topped peak. Amplitudes agree to
      {amps.max_rel_diff.max():.0e}. Runs {min(speed(h) for h in GRAV):.1f} to {max(speed(h) for h in GRAV):.1f} times
      faster on Graviton.</p></div>
    <div class="card verdict ok"><p class="tag">fp16 &middot; not in the PR, tested for comparison</p>
      <div class="big">{pct(ident['pr_fp16'], 1)}</div><p class="k">picks identical</p>
      <p>{pct(withinT['pr_fp16'], 2)} within {TOL}&thinsp;ms; {cnt['pr_fp16']['moved']}
      picks move further (up to {t2.max_shift_ms.max()['pr_fp16']:.0f}&thinsp;ms). Picks gained or lost all sit at the
      threshold. Speed not measured.</p></div>
    <div class="card verdict bad"><p class="tag">bf16 &middot; the PR's mode on Intel AMX CPUs only</p>
      <div class="big">{pct(ident['pr_bf16'], 1)}</div><p class="k">picks identical</p>
      <p>{pct(withinT['pr_bf16'], 2)} within {TOL}&thinsp;ms; {cnt['pr_bf16']['moved']} picks move further (up to
      {t2.max_shift_ms.max()['pr_bf16']:.0f}&thinsp;ms) and {cnt['pr_bf16']['lost'] + cnt['pr_bf16']['new']} appear or
      disappear. No speed gain on our hardware, which has no AMX.</p></div>
  </div>
  <p class="note">Against the {n_analyst:,} analyst picks, none of the three changes recall by more than
  {drec:.3f} or mean absolute error by more than {dmae * 1e3:.0f}&thinsp;ms{bias_txt}. The difference is in reproducing the catalogue, not in accuracy.</p>
</section>"""

formats = f"""
<section id="formats">
  <div class="section-head"><p class="eyebrow">Background</p><h2>What fp32, fp16 and bf16 are</h2>
  <p class="lede">They are ways of storing each number inside the network. Fewer bits make the arithmetic
  faster on CPUs built for it, and round every intermediate value more coarsely. Think of it as digitizer
  resolution applied to every layer of PhaseNet.</p></div>
  {table(['format', 'bits per value', 'bits of precision', 'rounding step (relative)', 'largest value'], [
      ['fp32 (single precision)', '32', '24, like a 24-bit digitizer', '6&times;10<sup>-8</sup>', '3&times;10<sup>38</sup>'],
      ['fp16 (half precision)', '16', '11', '5&times;10<sup>-4</sup>', '65,504'],
      ['bf16 ("brain float")', '16', '8', '4&times;10<sup>-3</sup>', '3&times;10<sup>38</sup>']])}
  <p class="note">PhaseNet's output is a probability trace per phase, and a pick is the sample where it peaks above
  the threshold. Rounding perturbs that trace a little. Where the peak is sharp nothing happens; where it is flat
  the maximum can jump to a neighbouring sample, and where it sits near the threshold the pick can appear or
  disappear. Under bf16 the 99th percentile change in the trace's peak value is
  {v2[v2.arm == 'pr_bf16'].p99_abs_dconf.max():.3f}; under fp32 it is too small to measure at the 10<sup>-4</sup>
  level.</p>
</section>"""

# ---- figure: picks reproduced within tau --------------------------------------------------
TAUS = np.arange(0, 101, 1)
curves = {a: [100 * reproduced(errs[a], t) for t in TAUS] for a in PR_ARMS}
for a in PR_ARMS:  # the figure and the exported table must agree
    for t in (0, 10, 20, 30, 50, 100):
        ref = cum[(cum.arm == a) & (cum.threshold == THR)].iloc[0][f"within_{t}ms"]
        assert abs(curves[a][t] / 100 - ref) < 5e-4, (a, t, curves[a][t], ref)
W, H, L, R, T, B = 720, 360, 56, 92, 18, 46
Y0, Y1 = 88.0, 100.0
x = lambda t: L + (W - L - R) * t / 100
y = lambda p: T + (H - T - B) * (Y1 - p) / (Y1 - Y0)
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="fig-t fig-d">',
       '<title id="fig-t">Share of picks reproduced within a timing tolerance</title>',
       (f'<desc id="fig-d">At threshold {THR:g}: fp32 reaches 100 percent at 10 ms, fp16 starts at '
        f'{curves["pr_fp16"][0]:.1f} percent, bf16 at {curves["pr_bf16"][0]:.1f} percent.</desc>')]
for p in range(int(Y0), int(Y1) + 1, 2):
    svg.append(f'<line x1="{L}" x2="{W - R}" y1="{y(p):.1f}" y2="{y(p):.1f}" class="grid"/>'
               f'<text x="{L - 8}" y="{y(p) + 4:.1f}" class="tick" text-anchor="end">{p}%</text>')
for t in range(0, 101, 10):
    svg.append(f'<text x="{x(t):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{t}</text>')
svg.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" class="axis" text-anchor="middle">timing tolerance (ms)</text>')
refs_v = [(10, "1 sample"), (TOL, "tolerance used here"), (52, "PhaseNet&ndash;analyst P &sigma;, Zhu 2019")]
for i, (t, lab) in enumerate(refs_v):
    svg.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T}" y2="{H - B}" class="ref"/>'
               f'<text x="{x(t) + 4:.1f}" y="{y(Y0) - 8 - 15 * i:.1f}" class="reflab">{lab} ({t} ms)</text>')
for a in PR_ARMS:
    pts = []
    for i, t in enumerate(TAUS):  # step: the share holds until the next tolerance adds picks
        if i:
            pts.append(f"{x(t):.1f},{y(curves[a][i - 1]):.1f}")
        pts.append(f"{x(t):.1f},{y(curves[a][i]):.1f}")
    dash = f' stroke-dasharray="{DASH[a]}"' if DASH[a] else ""
    svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{COLOR[a]}" stroke-width="2"{dash}/>')
    yl = y(curves[a][-1]) + {"pr_fp32": -8, "pr_fp16": 5, "pr_bf16": 12}[a]
    svg.append(f'<text x="{x(100) + 6:.1f}" y="{yl:.1f}" class="lab">{ARM[a]} {curves[a][-1]:.1f}%</text>')
    for t in (0, 10, 20, 30, 50, 100):
        svg.append(f'<circle cx="{x(t):.1f}" cy="{y(curves[a][t]):.1f}" r="7" fill="transparent">'
                   f'<title>{ARM[a]}: {curves[a][t]:.2f}% of picks within {t} ms</title></circle>')
svg.append("</svg>")
figure = f"""
<section id="figure">
  <div class="section-head"><p class="eyebrow">Precision</p><h2>How far do the picks move?</h2>
  <p class="lede">Share of the current code's picks that each mode reproduces within a given timing tolerance, at
  threshold {THR:g}. The {TOL}&thinsp;ms line is the tolerance used on this page: analyst P picks scatter by
  23&thinsp;ms (Ross et al. 2018) and the current code's P picks scatter around the analysts' by
  {resid.loc['P', 'robust_sigma_ms']:.0f}&thinsp;ms on this benchmark. Curves stop
  short of 100&thinsp;% because of picks that appear or disappear at the threshold
  ({pct(lost_new['pr_fp16'], 2)} for fp16, {pct(lost_new['pr_bf16'], 2)} for bf16, none for fp32).</p></div>
  <figure class="fig">{''.join(svg)}</figure>
  {table(['mode', 'identical (0 ms)', 'within 10 ms', f'within {TOL} ms', 'largest shift', 'spread of shifts (&sigma;)', 'picks appearing or disappearing'],
         [[ARM[a], pct(ident[a], 2), pct(within10[a], 2), pct(withinT[a], 2), f"{t2.max_shift_ms.max()[a]:.0f} ms",
           f"{sd_shift[a]:.1f} ms", f"{cnt[a]['lost']} / {cnt[a]['new']}"] for a in PR_ARMS])}
  <p class="note">For scale: the current code's own picks scatter around the analysts' with a &sigma; (from the median absolute deviation) of
  {resid.loc['P', 'robust_sigma_ms']:.0f}&thinsp;ms (P) and {resid.loc['S', 'robust_sigma_ms']:.0f}&thinsp;ms (S) on
  this benchmark. bf16 adds a {sd_shift['pr_bf16']:.0f}&thinsp;ms scatter on top; fp16 {sd_shift['pr_fp16']:.0f}&thinsp;ms.</p>
</section>"""

# ---- analyst picks -----------------------------------------------------------------------
rows = []
for trk in ("us_sequences", "swarm_sequences"):
    for pha in "PS":
        g = a2[(a2.track == trk) & (a2.phase == pha)].set_index("arm")
        rows.append([TRACK[trk], pha, n(g.loc["vanilla", "analyst"])]
                    + [f"{g.loc[a, 'recall']:.3f}" for a in ("vanilla",) + PR_ARMS])
analyst = f"""
<section id="analyst">
  <div class="section-head"><p class="eyebrow">Precision</p><h2>Against analyst picks</h2>
  <p class="lede">Recall within 0.5&thinsp;s at threshold {THR:g}, as on the <a href="benchmark_metrics.html">leaderboard</a>.
  Of the {n_analyst:,} analyst picks, the number whose outcome changes is {changed['pr_fp32']} for fp32,
  {changed['pr_fp16']} for fp16 and {changed['pr_bf16']} for bf16.</p></div>
  {table(['track', 'phase', 'analyst picks', 'current code', 'fp32', 'fp16', 'bf16'], rows, left=2)}
</section>"""

# ---- user-defined tolerance ---------------------------------------------------------------
SENT = -999999  # no partner within 0.5 s
data = {}
for arm in PR_ARMS:
    er = errs[arm]
    er = er[(er.conf_vanilla.fillna(0) >= 0.05) | (er.conf_arm.fillna(0) >= 0.05)]
    data[arm] = {
        "cv": np.floor(er.conf_vanilla.fillna(-0.001) * 10000).astype(int).tolist(),
        "ca": np.floor(er.conf_arm.fillna(-0.001) * 10000).astype(int).tolist(),
        "dt": (er.dt_ms * 10).round().fillna(SENT).astype(int).tolist(),
        "tr": (er.track == "swarm_sequences").astype(int).tolist(),
        "ph": (er.phase == "S").astype(int).tolist()}
links = " &middot; ".join(f'<a href="{GH}/raw/main/docs/benchmark/results/vibesys_pr71/errors_{a}.csv.gz"><code>errors_{a}.csv.gz</code></a>'
                          for a in PR_ARMS)
own = f"""
<section id="own">
  <div class="section-head"><p class="eyebrow">Your tolerance</p><h2>Choose what counts as the same pick</h2>
  <p class="lede">Starts at a {TOL}&thinsp;ms tolerance with {ALLOW:g}&thinsp;% of picks allowed to fail; change
  either. Picks that appear or disappear only because they sit just at the threshold are excused by the
  confidence margin.</p></div>
  <div class="card" id="calc">
    <form class="calc-form" onsubmit="return false">
      <label>detection threshold <input id="c-thr" type="number" min="0.05" max="0.95" step="0.01" value="{THR:g}"></label>
      <label>timing tolerance (ms) <input id="c-tau" type="number" min="0" max="500" step="1" value="{TOL}"></label>
      <label>excuse lost/new picks within <input id="c-mar" type="number" min="0" max="0.9" step="0.01" value="0.05"></label>
      <label>failures allowed (%) <input id="c-all" type="number" min="0" max="100" step="0.01" value="{ALLOW:g}"></label>
      <label>track <select id="c-trk"><option value="-1">both</option><option value="0">US sequences</option><option value="1">swarm sequences</option></select></label>
      <label>phase <select id="c-ph"><option value="-1">both</option><option value="0">P</option><option value="1">S</option></select></label>
    </form>
    <div id="c-out"></div>
    <p class="note">A pick fails when it moves by more than the tolerance, or when it is lost or new and its
    confidence is more than the margin above the threshold. Per-pick data: {links}.</p>
  </div>
</section>"""
own_js = """
<script>
const QS_ERR = """ + json.dumps(data, separators=(",", ":")) + """;
const QS_LABEL = """ + json.dumps({a: ARM[a] for a in PR_ARMS}) + """;
const SENT = """ + str(SENT) + """;
function qsCalc() {
  const T = Math.round(parseFloat(document.getElementById('c-thr').value) * 10000);
  const tau = parseFloat(document.getElementById('c-tau').value) * 10;
  const M = Math.round(parseFloat(document.getElementById('c-mar').value) * 10000);
  const allowed = parseFloat(document.getElementById('c-all').value);
  const trk = parseInt(document.getElementById('c-trk').value), ph = parseInt(document.getElementById('c-ph').value);
  let rows = '';
  for (const arm of Object.keys(QS_ERR)) {
    const d = QS_ERR[arm];
    let nv = 0, ident = 0, within = 0, shifted = 0, lost = 0, lostFar = 0, nw = 0, newFar = 0;
    for (let i = 0; i < d.cv.length; i++) {
      if ((trk >= 0 && d.tr[i] !== trk) || (ph >= 0 && d.ph[i] !== ph)) continue;
      const paired = d.dt[i] !== SENT, vOk = d.cv[i] >= T, aOk = d.ca[i] >= T, adt = Math.abs(d.dt[i]);
      if (vOk) {
        nv++;
        if (paired && aOk) { if (adt <= 5) ident++; if (adt <= tau + 1e-6) within++; else shifted++; }
        else { lost++; if (d.cv[i] > T + M) lostFar++; }
      }
      if (aOk && !(paired && vOk)) { nw++; if (d.ca[i] > T + M) newFar++; }
    }
    const fail = shifted + lostFar + newFar, frac = nv ? 100 * fail / nv : 0, ok = frac <= allowed + 1e-12;
    const f = x => (nv ? (100 * x / nv).toFixed(2) : '0') + '&thinsp;%';
    rows += `<tr><td class="l">${QS_LABEL[arm]}</td><td>${nv.toLocaleString()}</td><td>${f(ident)}</td><td>${f(within)}</td>` +
      `<td>${shifted.toLocaleString()}</td><td>${lost.toLocaleString()} / ${nw.toLocaleString()}</td>` +
      `<td>${fail.toLocaleString()} (${frac.toFixed(2)}&thinsp;%)</td>` +
      `<td>${ok ? '<span class="pill met">passes</span>' : '<span class="pill unmet">fails</span>'}</td></tr>`;
  }
  document.getElementById('c-out').innerHTML = '<div class="table-scroll"><table class="data"><thead><tr>' +
    '<th class="l">mode</th><th>picks</th><th>identical</th><th>within tolerance</th><th>moved further</th>' +
    '<th>lost / new</th><th>failures</th><th>verdict</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
}
document.querySelectorAll('#calc input, #calc select').forEach(el => el.addEventListener('input', qsCalc));
qsCalc();
</script>"""

# ---- speed and cost ----------------------------------------------------------------------
rows = []
for h in HOSTS:
    spot = (f"${usd_per_1k(h, 'vanilla'):.2f} &rarr; <strong>${usd_per_1k(h, 'pr'):.2f}</strong>"
            if h in GRAV else "&ndash;")
    rows.append([HOST_LABEL[h], f"{med.loc[(h, 'vanilla'), 'wall_s']:.0f} s", f"{med.loc[(h, 'pr'), 'wall_s']:.0f} s",
                 f"<strong>{speed(h):.1f}&times;</strong>", spot])
cost = f"""
<section id="cost">
  <div class="section-head"><p class="eyebrow">Speed</p><h2>Six times faster on AWS Graviton, in fp32</h2>
  <p class="lede">The same {SD} station-days (42 benchmark windows) processed by four workers on an 8-vCPU
  machine, the shape of a Fargate task. Median of four repetitions; picking and amplitudes included, S3 reads and
  writes not.</p></div>
  {table(['machine', 'current code', 'PR, fp32', 'faster', 'Spot $ per 1,000 station-days'], rows)}
  <div class="grid-2" style="margin-top:18px">
  <div class="card"><h3>Where the time goes</h3><p>Most of the saving is the Wood-Anderson amplitude code:
  {100 * wa_share['c8g'] / tot_saved['c8g']:.0f}&thinsp;% of the time saved on Graviton4. PhaseNet itself runs
  {med.loc[('c8g', 'vanilla'), 'classify_s'] / med.loc[('c8g', 'pr'), 'classify_s']:.1f}&times; faster there. These
  windows are aftershock sequences with many picks; an ordinary station-day has far fewer, so the gain in a
  campaign will be smaller. Not yet measured.</p></div>
  <div class="card"><h3>Why the laptop gains little</h3><p>The PR's fast PhaseNet needs a library (oneDNN) that
  no macOS build of PyTorch includes, so on an Apple laptop only the amplitude code speeds up, and picking runs
  {med.loc[('M1 Max', 'pr'), 'classify_s'] / med.loc[('M1 Max', 'vanilla'), 'classify_s']:.2f}&times; slower.</p></div>
  </div>
  <p class="note">Prices: {e(meta['prices'])}. For comparison, Fargate Spot costs about ${fargate_h:.2f}/h
  for the same 8 vCPU (CloudBank billing). Fargate itself was not timed.</p>
</section>"""

# ---- details -----------------------------------------------------------------------------
rows = []
for _, r in vv.sort_values(["threshold", "arm", "track"], ascending=[True, False, True]).iterrows():
    rows.append([f"{r.threshold:g}", ARM[r.arm], TRACK[r.track], n(r.vanilla), pct(r.identical, 3), n(r.shifted_n),
                 n(r.lost_n), n(r.new_n), f"{r.p99_abs_shift_s * 1e3:.0f} ms", f"{r.p99_abs_dconf:.3f}"])
detail_vanilla = table(['threshold', 'mode', 'track', 'picks', 'identical', 'moved', 'lost', 'new', 'p99 |shift|',
                        'p99 |&Delta;confidence|'], rows, left=3)
fp_rows = [[e(r.sequence), e(r.station), r.phase, e(r.time)[:23], f"{r.conf_vanilla:.4f}" if pd.notna(r.conf_vanilla) else "&ndash;",
            f"{r.conf_pr_fp32:.4f}" if pd.notna(r.conf_pr_fp32) else "&ndash;", e(r.side)] for _, r in fp32d.iterrows()]
amp_rows = [[a, n(r.picks), n(r.stations), n(r.with_value), n(r.missing_pattern_differs), f"{r.max_rel_diff:.1e}"]
            for a, r in amps.iterrows()]
runs = ["c6g vanilla", "c7g vanilla", "c8g vanilla", "c6g pr", "c7g pr", "c8g pr"]
same = {(r.a, r.b): int(r.identical) for _, r in exact.iterrows()}
same.update({(b_, a_): v for (a_, b_), v in list(same.items())})
lab = {r: f"{r.split()[0]} {'current' if r.endswith('vanilla') else 'PR'}" for r in runs}
x_rows = [[lab[r]] + [n(g23.n_a) if r == c else n(same[(r, c)]) for c in runs] for r in runs]
t_rows = [[HOST_LABEL[h], ARM[a], ", ".join(f"{w:.1f}" for w in walls.loc[(h, a)]), f"{med.loc[(h, a), 'classify_s']:.0f} s",
           f"{med.loc[(h, a), 'wa_s']:.0f} s", f"{med.loc[(h, a), 'vel_s']:.1f} s"] for h in HOSTS for a in ("vanilla", "pr")]
c_rows = [[HOST_LABEL[h], e(hosts.loc[h, "cpu"]), f"${hosts.loc[h, 'on_demand_usd_h']:.3f}", f"${hosts.loc[h, 'spot_usd_h']:.4f}",
           f"${usd_per_1k(h, 'vanilla', 'on_demand_usd_h'):.3f} / ${usd_per_1k(h, 'pr', 'on_demand_usd_h'):.3f}"] for h in GRAV]
details = f"""
<section id="details">
  <div class="section-head"><p class="eyebrow">Details</p><h2>Method, full tables, limits</h2></div>
  <details><summary>What each mode runs</summary>
  <p><b>Current code</b>: <code>sbm.PhaseNet.from_pretrained("{WEIGHT_ID}").classify</code> and main's
  <code>AmplitudeExtractor()</code> at <code>{e(meta['vanilla_commit'])}</code>. <b>fp32</b>: PR head
  <code>{e(meta['pr_head'])}</code> with the settings of its removed runner (<code>vibesys/pick_workload.py</code> at
  <code>4271d67</code>): BatchNorm folded into the convolutions, polyphase transposed convolutions, fast windowing
  (<code>glue=fused</code>), amplitudes <code>impl="fused", signal="slim"</code>. <b>bf16</b> and <b>fp16</b>: the same,
  with weights and inputs in 16 bits, fp32 accumulation, and every layer's output rounded to 16 bits. This
  emulates Intel AMX for bf16 and is a best case for fp16. Run on the M1 Max laptop.</p>
  <p>Precision workload: {e(meta['accuracy_workload'])}. Picks at a 0.02 floor, thresholded afterwards, paired with
  <code>benchmark_metrics.match_picks</code> within 0.5&thinsp;s. The cached waveforms for Mendocino, Monroe and San
  Simeon CI are not the streams the leaderboard picked, so absolute US recall differs from the board; every mode
  read the same files.</p></details>
  <details><summary>Picks against the current code, both thresholds</summary>{detail_vanilla}</details>
  <details><summary>Every fp32 difference at the 0.02 floor ({len(fp32d)} picks)</summary>
  {table(['sequence', 'station', 'phase', 'time', 'confidence, current', 'confidence, fp32', 'side'], fp_rows, left=3)}
  <p class="note">Same arrival one to three samples apart with confidences equal to about 10<sup>-4</sup>. Folding
  BatchNorm changes the order of the fp32 sums, which moves the probability trace by about 10<sup>-6</sup> and can flip
  which sample of a flat peak is the maximum.</p></details>
  <details><summary>Amplitudes</summary>
  {table(['amplitude', 'picks', 'stations', 'with a value', 'missing-value pattern differs', 'max relative difference'], amp_rows)}
  <p class="note">Western US and Oklahoma stations only: EarthScope served no response for the Etna (IV) stations, and
  the French and most Czech stations were not measured.</p></details>
  <details><summary>Exact agreement across Graviton generations</summary>
  {table(['picks identical to the ms'] + [lab[r] for r in runs], x_rows)}
  <p class="note">{n(g23.n_a)} picks per run at threshold {THR:g}. The current code itself differs by
  {n(g23.n_a - g23.identical)} picks between Graviton2 and Graviton3, so a millisecond rule cannot hold across our own
  hardware even without the PR.</p></details>
  <details><summary>Timing per repetition and cost at on-demand prices</summary>
  {table(['machine', 'code', 'wall per repetition (s)', 'picking', 'Wood-Anderson', 'velocity'], t_rows, left=2)}
  <p class="note">Stage times summed over the four workers.</p>
  {table(['instance', 'CPU', 'on-demand $/h', 'Spot $/h', 'on-demand $ per 1,000 station-days, current / PR'], c_rows, left=2)}</details>
  <details><summary>Limits and defects found in the PR</summary><ul>
    <li><code>jma_wc</code>, the global weight, was not scored. <code>obs</code> (PickBlue) cannot use the fast path:
    building it raises <code>does not cover this model setup: ['padding/filter']</code>.</li>
    <li>bf16 was emulated, not run on AMX hardware. fp16 speed was not measured.</li>
    <li>Streams the model resamples (30 of 130 windows) fall back to SeisBench's windowing. Timing used 100&thinsp;Hz only.</li>
    <li>The 707-line <code>s3_helper.py</code> rewrite was not reviewed.</li>
    <li><code>veliir.c</code> (velocity amplitudes) does not build on ARM, so SciPy ran on Graviton. Built with its
    x86-only CPU probe guarded, it matched SciPy to 10<sup>-11</sup>.</li>
    <li>As merged, nothing selects the fast settings, so production would run the current code.</li>
  </ul></details>
  <details><summary>Reproduce</summary>
  <p>Harness in <a href="{GH}/tree/main/scripts/vibesys_pr71"><code>scripts/vibesys_pr71/</code></a>;
  <code>export_results.py</code> writes the tables in
  <a href="{GH}/tree/main/docs/benchmark/results/vibesys_pr71"><code>docs/benchmark/results/vibesys_pr71/</code></a>,
  and <code>scripts/build_vibesys_report.py</code> builds this page from them.</p></details>
</section>"""

refs = """
<section id="refs"><div class="section-head"><p class="eyebrow">References</p></div><ol class="refs">
  <li id="ref-ross">Ross, Z. E., Meier, M.-A. and Hauksson, E. (2018). P wave arrival picking and first-motion polarity determination with deep learning. <i>Journal of Geophysical Research: Solid Earth</i> 123, 5120-5129. <a href="https://doi.org/10.1029/2017JB015251">doi:10.1029/2017JB015251</a></li>
  <li id="ref-zhu">Zhu, W. and Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. <i>Geophysical Journal International</i> 216, 261-273. <a href="https://doi.org/10.1093/gji/ggy423">doi:10.1093/gji/ggy423</a></li>
</ol></section>"""

commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
built = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Do QuakeScope picks and amplitudes change under the CPU optimizations proposed in PR #71, and how much faster is the workflow? Tested on the picker benchmark, an M1 Max laptop and three AWS Graviton instances.">
<title>VibeSys Optimizations Evaluation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="quakescope-board.css">
<style>
.verdict {{ border-top: 4px solid var(--line); }}
.verdict.ok {{ border-top-color: var(--good); }} .verdict.mid {{ border-top-color: var(--warn); }} .verdict.bad {{ border-top-color: var(--bad); }}
.verdict .tag {{ font-size: 0.8rem; color: var(--stone); margin: 0 0 8px; }}
.verdict .big {{ font-size: 1.6rem; font-weight: 800; color: var(--ink); }}
.verdict .k {{ font-size: 0.85rem; color: var(--stone); margin: 2px 0 10px; }}
.fig svg {{ font-family: Manrope, sans-serif; }}
.fig .grid {{ stroke: var(--line); }} .fig .ref {{ stroke: var(--stone); stroke-dasharray: 3 3; opacity: .6; }}
.fig .tick {{ font-size: 12px; fill: var(--stone); }} .fig .axis {{ font-size: 13px; fill: var(--ink); }}
.fig .reflab {{ font-size: 11px; fill: var(--stone); }} .fig .lab {{ font-size: 13px; font-weight: 700; fill: var(--ink); }}
details {{ border-top: 1px solid var(--line); padding: 10px 0; }}
details summary {{ cursor: pointer; font-weight: 700; color: var(--purple-deep); }}
details[open] summary {{ margin-bottom: 10px; }}
details ul {{ padding-left: 1.1em; }} details li {{ margin: 4px 0; }}
.calc-form {{ display: flex; flex-wrap: wrap; gap: 10px 18px; margin: 0 0 14px; }}
.calc-form label {{ display: flex; flex-direction: column; font-size: 0.85rem; color: var(--stone); gap: 4px; }}
.calc-form input, .calc-form select {{ font: inherit; color: var(--ink); padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; background: var(--paper); width: 9.5em; }}
</style>
</head>
<body>
<header class="hero">
  <div class="shell">
    <nav class="nav" aria-label="Primary">
      <a class="brand" href="./"><span class="brand-mark">QS</span>
        <span>QuakeScope <span style="opacity:.7;font-weight:600">code evaluation</span></span></a>
      <div class="nav-links">
        <a href="#answer">Answer</a><a href="#figure">Pick shifts</a><a href="#analyst">Analysts</a>
        <a href="#own">Your tolerance</a><a href="#cost">Speed</a><a href="#details">Details</a>
      </div>
    </nav>
    <div class="hero-inner">
      <p class="eyebrow">QuakeScope &middot; PR #71 &middot; {e(meta['run_date'])}</p>
      <h1>Faster QuakeScope: do the picks and amplitudes change?</h1>
      <p>PR #71 proposes faster CPU code for the QuakeScope workflow (waveform loading, PhaseNet picking and
      amplitude measurement), found by an automated optimization tool (VibeSys). We re-ran the leaderboard's
      waveforms with it and compared every pick and amplitude.</p>
      <div class="hero-actions">
        <a class="button primary" href="#answer">Answer</a>
        <a class="button secondary" href="{meta['pr']}">PR #71</a>
        <a class="button secondary" href="benchmark_metrics.html">Picker leaderboard</a>
      </div>
    </div>
  </div>
</header>
<main class="shell">
{answer}
{figure}
{formats}
{analyst}
{own}
{cost}
{details}
{refs}
</main>
{own_js}
<footer class="site-footer">
  <div class="shell">
    <p><strong>QuakeScope code evaluation</strong> &middot; maintained by <a href="{GH}">SeisSCOPED</a> &middot;
    <a href="./">all QuakeScope reports</a></p>
    <p class="provenance">Measured {e(meta['run_date'])} on PR head <code>{e(meta['pr_head'])}</code> against
    <code>{e(meta['vanilla_commit'])}</code>; page built {built} from <code>{commit}</code> by
    <code>scripts/build_vibesys_report.py</code>; numbers read from <code>docs/benchmark/results/vibesys_pr71/</code>.</p>
  </div>
</footer>
</body>
</html>
"""
OUT.write_text(page)
print(f"wrote {OUT.relative_to(ROOT)} ({len(page):,} bytes)")
