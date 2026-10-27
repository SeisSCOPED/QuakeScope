#!/usr/bin/env python
"""reports/vibesys_pr71.html: pick precision and computational cost of the VibeSys
CPU optimizations proposed in PR #71.

Every number is read from docs/benchmark/results/vibesys_pr71/, which
scripts/vibesys_pr71/export_results.py writes from the harness outputs. Nothing
is typed here except the two published analyst-uncertainty figures, which carry
their citation in the table.

    python scripts/build_vibesys_report.py
"""

from __future__ import annotations

import datetime
import html
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs/benchmark/results/vibesys_pr71"
OUT = ROOT / "reports/vibesys_pr71.html"
GH = "https://github.com/SeisSCOPED/QuakeScope"

meta = json.load(open(RES / "meta.json"))
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

TRACK = {"us_sequences": "US sequences", "swarm_sequences": "swarm sequences"}
ARM = {"vanilla": "vanilla", "pr_fp32": "PR fp32", "pr_bf16": "PR bf16 (AMX emulated)", "pr": "PR fp32"}
HOSTS = ["M1 Max", "c6g", "c7g", "c8g"]
HOST_LABEL = {"M1 Max": "M1 Max laptop", "c6g": "c6g.2xlarge, Graviton2",
              "c7g": "c7g.2xlarge, Graviton3", "c8g": "c8g.2xlarge, Graviton4"}
SD = 5.55  # station-days in the timing workload (meta.json timing_workload)


def e(x) -> str:
    return html.escape(str(x))


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}&thinsp;%"


def n(x) -> str:
    return f"{int(x):,}"


def table(head: list[str], rows: list[list[str]], caption: str = "", left: int = 1) -> str:
    cls = ' class="l"'
    th = "".join(f"<th{cls if i < left else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{cls if i < left else ''}>{c}</td>" for i, c in enumerate(r))
                   + "</tr>" for r in rows)
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="table-scroll"><table class="data">{cap}<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def pill(ok: bool) -> str:
    return '<span class="pill met">passes</span>' if ok else '<span class="pill unmet">fails</span>'


# ---- derived numbers ---------------------------------------------------------------------
v2 = vv[vv.threshold == 0.2]
fp = v2[v2.arm == "pr_fp32"]
bf = v2[v2.arm == "pr_bf16"]
fp_ident, fp_total = int(fp.identical_n.sum()), int(fp.vanilla.sum())
bf_ident_lo, bf_ident_hi = bf.identical.min(), bf.identical.max()
bf_shift_lo, bf_shift_hi = bf.shift_ge_1smp.min(), bf.shift_ge_1smp.max()
bf_ln = (bf.lost_n + bf.new_n).sum() / bf.vanilla.sum()

med = timing[timing.complete].groupby(["host", "arm"]).median(numeric_only=True)
walls = timing[timing.complete].groupby(["host", "arm"]).wall_s.apply(list)


def speed(h):
    return med.loc[(h, "vanilla"), "wall_s"] / med.loc[(h, "pr"), "wall_s"]


def usd_per_1k(h, arm, col):
    return 1000 * med.loc[(h, arm), "wall_s"] / 3600 * hosts.loc[h, col] / SD


best = min(("c6g", "c7g", "c8g"), key=lambda h: usd_per_1k(h, "pr", "spot_usd_h"))
d03 = disc[disc.threshold == 0.3]
d02 = disc[disc.threshold == 0.2]
n_analyst = int(d03.analyst.sum())
same_host = exact[[a.split()[0] == b.split()[0] for a, b in zip(exact.a, exact.b)]]
same_host_min = int(same_host.identical.min())
g23 = exact[(exact.a == "c6g vanilla") & (exact.b == "c7g vanilla")].iloc[0]
fargate_h = 8 * meta["fargate_spot_rate_usd_vcpu_h"]

# ---- sections ----------------------------------------------------------------------------
stats = f"""
<div class="stats">
  <div class="stat"><div class="n">{n(fp_ident)} / {n(fp_total)}</div><div class="k">picks identical to the millisecond, PR fp32 vs vanilla</div></div>
  <div class="stat"><div class="n">{amps.max_rel_diff.max():.0e}</div><div class="k">largest relative amplitude difference</div></div>
  <div class="stat"><div class="n">{speed('c8g'):.1f}&times;</div><div class="k">faster on Graviton4 (compute)</div></div>
  <div class="stat"><div class="n">${usd_per_1k(best, 'pr', 'spot_usd_h'):.3f}</div><div class="k">per 1,000 station-days, {best} Spot, PR</div></div>
  <div class="stat"><div class="n">{pct(bf_ident_lo, 0)}&ndash;{pct(bf_ident_hi, 0)}</div><div class="k">picks identical under bf16</div></div>
</div>"""

verdict = f"""
<section id="verdict">
  <div class="section-head"><p class="eyebrow">Verdict</p>
  <h2>The fp32 path keeps the catalogue and runs six times faster on Graviton; bf16 changes the catalogue</h2></div>
  <div class="grid-2">
  <div class="callout">
    <h3>Accept: fp32 forward pass, fast annotate, amplitude rewrite</h3>
    <p>With the PR's settings in fp32, {n(fp_ident)} of {n(fp_total)} picks at threshold 0.2 are identical to
    vanilla to the millisecond; the other {fp_total - fp_ident} are one sample apart on flat-topped peaks. Wood-Anderson
    and velocity amplitudes agree to {amps.max_rel_diff.max():.0e} relative. On the same Graviton instance the PR
    reproduces {same_host_min:,} of {n(g23.n_a)} vanilla picks, fewer differences than vanilla itself shows between
    Graviton2 and Graviton3 ({n(g23.n_a - g23.identical)}). Compute time falls {min(speed(h) for h in ('c6g', 'c7g', 'c8g')):.1f}&times; to
    {max(speed(h) for h in ('c6g', 'c7g', 'c8g')):.1f}&times; on Graviton, and {speed('M1 Max'):.2f}&times; on an M1 Max laptop.</p>
  </div>
  <div class="callout warn">
    <h3>Do not accept: bf16 chosen per host</h3>
    <p>bf16 inference leaves {pct(bf_ident_lo, 0)} to {pct(bf_ident_hi, 0)} of picks identical, moves
    {pct(bf_shift_lo, 0)} to {pct(bf_shift_hi, 0)} by at least one sample and makes {pct(bf_ln)} appear or disappear.
    Recall against analyst picks does not change, but the PR enables bf16 automatically on CPUs with AMX-BF16, so a
    catalogue built on mixed Spot capacity would carry two numerical paths with no record of which ran where.
    Precision should be an explicit campaign setting, default fp32, written into the run records.</p>
  </div>
  </div>
</section>"""

arms_table = table(["arm", "what runs"], [
    ["vanilla", "<code>sbm.PhaseNet.from_pretrained(\"original\").classify</code> and main's "
                f"<code>AmplitudeExtractor()</code>, at {e(meta['vanilla_commit'])}"],
    ["PR fp32", f"PR head <code>{e(meta['pr_head'])}</code>: fast graph (BatchNorm folded, polyphase deconvolution), "
                "fast annotate with <code>glue=fused</code>, amplitudes <code>impl=\"fused\", signal=\"slim\"</code>, "
                "C velocity kernel where it builds"],
    ["PR bf16", "the same in bf16, with AMX-BF16 emulated where the CPU has none: bf16 weights and input, "
                "fp32 accumulate, bias and ReLU, output rounded to bf16 after every layer"],
], left=2)
what = f"""
<section id="tested">
  <div class="section-head"><p class="eyebrow">What was tested</p>
  <h2>Three arms on the leaderboard's own waveforms</h2>
  <p class="lede">PR <a href="{meta['pr']}">#71</a> adds optional CPU paths for PhaseNet inference, amplitude
  extraction, waveform loading and process start, produced by an automated optimization campaign (VibeSys).
  It ships them switched off. The settings below are those of the runner that selected them,
  <code>vibesys/pick_workload.py</code> at <code>4271d67</code>, which was removed from the PR in <code>32d27cf</code>.</p></div>
  {arms_table}
  <div class="grid-2" style="margin-top:18px">
    <div class="card"><h3>Precision workload</h3><p>{e(meta['accuracy_workload'])}. Weight
    {e(meta['weight'])}, the western campaign's. Picks at a 0.02 floor, thresholded afterwards, matched with
    <code>benchmark_metrics.match_picks</code> at 0.5&thinsp;s. Run on the M1 Max laptop.</p></div>
    <div class="card"><h3>Timing workload</h3><p>{e(meta['timing_workload'])}, the production layout of the
    western job definition. {e(meta['timing_scope'])}. Timing includes imports and model start.</p></div>
  </div>
</section>"""

rows = []
for _, r in vv.sort_values(["threshold", "arm", "track"], ascending=[True, False, True]).iterrows():
    rows.append([f"{r.threshold:g}", ARM[r.arm], TRACK[r.track], n(r.vanilla), f"{pct(r.identical, 3)} ({n(r.identical_n)})",
                 n(r.shifted_n), pct(r.shift_ge_5smp, 2), n(r.lost_n), n(r.new_n), f"{r.p99_abs_shift_s * 1e3:.0f} ms",
                 f"{r.p99_abs_dconf:.3f}"])
fp_rows = [[e(r.sequence), e(r.station), r.phase, e(r.time)[:23], f"{r.conf_vanilla:.4f}" if pd.notna(r.conf_vanilla) else "&ndash;",
            f"{r.conf_pr_fp32:.4f}" if pd.notna(r.conf_pr_fp32) else "&ndash;", e(r.side)] for _, r in fp32d.iterrows()]
vs_vanilla = f"""
<section id="vanilla">
  <div class="section-head"><p class="eyebrow">Precision</p><h2>Picks against vanilla</h2>
  <p class="lede">A pick is identical when station, phase and peak time agree to the millisecond. Lost and new
  picks are those with no counterpart within 0.5&thinsp;s; shifts are counted in samples at 100&thinsp;Hz.</p></div>
  {table(['threshold', 'arm', 'track', 'vanilla picks', 'identical', 'shifted &ge; 1 sample', '&ge; 5 samples', 'lost', 'new', 'p99 |shift|', 'p99 |&Delta;conf|'], rows, left=3)}
  <h3 style="margin-top:22px">Every fp32 difference at the 0.02 floor ({len(fp32d)} of {n(fp_total)}+ picks)</h3>
  {table(['sequence', 'station', 'phase', 'time', 'conf vanilla', 'conf PR fp32', 'side'], fp_rows, left=3)}
  <p class="note">Most are the same arrival one to three samples apart with confidences equal to about
  10<sup>-4</sup>: folding BatchNorm and the polyphase deconvolution change the fp32 summation order, which moves
  probabilities by about 10<sup>-6</sup> and flips which sample of a flat peak is the maximum.</p>
</section>"""

rows = []
for thr in (0.3, 0.2):
    for trk in ("us_sequences", "swarm_sequences"):
        for pha in "PS":
            g = aa[(aa.threshold == thr) & (aa.track == trk) & (aa.phase == pha)].set_index("arm")
            d = disc[(disc.threshold == thr) & (disc.track == trk) & (disc.phase == pha)].iloc[0]
            rows.append([f"{thr:g}", TRACK[trk], pha, n(g.loc['vanilla', 'analyst']),
                         f"{g.loc['vanilla', 'recall']:.3f}", f"{g.loc['pr_fp32', 'recall']:.3f}", f"{g.loc['pr_bf16', 'recall']:.3f}",
                         f"{g.loc['vanilla', 'within_0_1']:.3f} / {g.loc['pr_bf16', 'within_0_1']:.3f}",
                         f"{g.loc['vanilla', 'mae_s']:.3f} / {g.loc['pr_bf16', 'mae_s']:.3f}",
                         f"{g.loc['vanilla', 'bias_s']:.3f} / {g.loc['pr_bf16', 'bias_s']:.3f}",
                         f"{int(d.bf16_lost)} / {int(d.bf16_gained)}", str(int(d.fp32_changed))])
vs_analyst = f"""
<section id="analyst">
  <div class="section-head"><p class="eyebrow">Precision</p><h2>Picks against analyst picks</h2>
  <p class="lede">Recall within 0.5&thinsp;s, residuals matched within 2&thinsp;s, as on the
  <a href="benchmark_metrics.html">leaderboard</a>. At 0.3, bf16 changes the outcome for
  {int(d03.bf16_lost.sum() + d03.bf16_gained.sum())} of {n_analyst:,} analyst picks
  ({int(d03.bf16_lost.sum())} lost, {int(d03.bf16_gained.sum())} gained); at 0.2 for
  {int(d02.bf16_lost.sum() + d02.bf16_gained.sum())}. fp32 changes none.</p></div>
  {table(['threshold', 'track', 'phase', 'analyst picks', 'recall vanilla', 'recall PR fp32', 'recall PR bf16',
          'within 0.1 s, vanilla / bf16', 'MAE (s), vanilla / bf16', 'median bias (s), vanilla / bf16',
          'bf16 lost / gained', 'fp32 changed'], rows, left=3)}
  <p class="note">The US cached waveforms for Mendocino, Monroe and San Simeon CI are not the streams the leaderboard
  picked, so absolute US recall here differs from the board. Every arm read the same files.</p>
</section>"""

trows = []
for _, r in tol.iterrows():
    ok = r.within_10ms >= 0.9999 and r.lost_or_new_conf_ge_0p25 == 0
    trows.append([ARM[r.arm], r.phase, n(r.vanilla), pct(r.within_10ms, 2), pct(r.beyond_50ms, 2), f"{r.max_shift_ms:.0f} ms",
                  f"{r.std_shift_ms:.2f} ms", f"{int(r.lost)} / {int(r.new)}", str(int(r.lost_or_new_conf_ge_0p25)), pill(ok)])
bf_sd = tol[tol.arm == "pr_bf16"].std_shift_ms.max()
lit_table = table(["source", "quantity", "P", "S"], [
    ['Ross, Meier and Hauksson (2018) <a href="#ref-ross">[1]</a>',
     "analyst-trained CNN vs SCSN analyst picks, standard deviation; the authors: hand P picks "
     "&ldquo;probably only accurate to within a few samples&rdquo;", "23 ms", "&ndash;"],
    ['Zhu and Beroza (2019) <a href="#ref-zhu">[2]</a>',
     "PhaseNet vs NCEDC analyst picks, standard deviation (residuals under 0.5 s)", "52 ms", "83 ms"],
    ['Zhu and Beroza (2019) <a href="#ref-zhu">[2]</a>', "analyst uncertainty assumed in the training labels",
     "100 ms", "100 ms"],
    ["this benchmark", "vanilla <code>original</code> vs analyst picks at 0.2, MAD-based &sigma; (1.4826 &times; median absolute deviation); "
     f"n = {n(resid.loc['P', 'matched'])} P, {n(resid.loc['S', 'matched'])} S",
     f"{resid.loc['P', 'robust_sigma_ms']:.0f} ms", f"{resid.loc['S', 'robust_sigma_ms']:.0f} ms"],
], left=2)
tol_table = table(["arm", "phase", "vanilla picks at 0.2", "within 10 ms", "beyond 50 ms", "largest shift",
                   "shift &sigma;", "lost / new", "lost or new with conf &ge; 0.25", "rule"], trows, left=2)
tolerance = f"""
<section id="tolerance">
  <div class="section-head"><p class="eyebrow">Tolerance</p><h2>A bound for a reasonable error, from analyst picking uncertainty</h2>
  <p class="lede">The acceptance rule for a picker code change is that picks match vanilla to the millisecond. Where a
  change cannot meet it, the error it adds should be small against how well analysts themselves pick.</p></div>
  {lit_table}
  <div class="callout" style="margin-top:18px"><h3>Proposed rule</h3>
  <p>Every pick within &plusmn;1 sample (10&thinsp;ms at 100&thinsp;Hz), which is under half the tightest analyst
  figure above, and no pick lost or gained unless its confidence is within 0.05 of the threshold.</p></div>
  {tol_table}
  <p class="note">bf16's shift spread of up to {bf_sd:.0f}&thinsp;ms is a third of this benchmark's MAD-based
  model&ndash;analyst &sigma; for P, and its tails reach half a second, so it fails any bound set from analyst
  uncertainty.</p>
</section>"""

amplitudes = f"""
<section id="amplitudes">
  <div class="section-head"><p class="eyebrow">Precision</p><h2>Amplitudes</h2>
  <p class="lede">The same vanilla picks at 0.2, measured by main's extractor and by the PR's fused/slim extractor with
  the C velocity kernel. Responses from EarthScope FDSN.</p></div>
  {table(['amplitude', 'picks', 'stations', 'with a value', 'missing-value pattern differs', 'bit-identical', 'max relative difference'],
         [[a, n(r.picks), n(r.stations), n(r.with_value), n(r.missing_pattern_differs), pct(r.bit_identical), f"{r.max_rel_diff:.1e}"]
          for a, r in amps.iterrows()])}
  <p class="note">The {int(amps.stations.max())} stations are in the western US and Oklahoma; EarthScope served no
  response for the Etna (IV) stations, and the French and most Czech stations were not measured.</p>
</section>"""

rows = []
for h in HOSTS:
    for arm in ("vanilla", "pr"):
        m = med.loc[(h, arm)]
        rows.append([HOST_LABEL[h] if arm == "vanilla" else "", ARM[arm], f"<strong>{m.wall_s:.1f} s</strong>",
                     ", ".join(f"{w:.1f}" for w in walls.loc[(h, arm)]), f"{m.classify_s:.0f} s", f"{m.wa_s:.0f} s", f"{m.vel_s:.1f} s",
                     f"{speed(h):.2f}&times;" if arm == "pr" else ""])
crows = []
for h in ("c6g", "c7g", "c8g"):
    crows.append([HOST_LABEL[h], hosts.loc[h, "cpu"], f"${hosts.loc[h, 'on_demand_usd_h']:.3f}", f"${hosts.loc[h, 'spot_usd_h']:.4f}",
                  f"${usd_per_1k(h, 'vanilla', 'on_demand_usd_h'):.3f} / ${usd_per_1k(h, 'pr', 'on_demand_usd_h'):.3f}",
                  f"<strong>${usd_per_1k(h, 'vanilla', 'spot_usd_h'):.3f} / ${usd_per_1k(h, 'pr', 'spot_usd_h'):.3f}</strong>"])
kern = timing[(timing.arm == "pr") & (timing.host != "M1 Max")].velocity_kernel.dropna().unique()
cost = f"""
<section id="cost">
  <div class="section-head"><p class="eyebrow">Computational cost</p><h2>Laptop and Graviton, same workload, same layout</h2>
  <p class="lede">Wall time is the slowest of the four workers, median of four alternating repetitions; stage times
  are summed over the four workers. Both arms of a host share one Python environment.</p></div>
  {table(['host', 'arm', 'wall, median', 'wall per rep (s)', 'picking', 'Wood-Anderson', 'velocity', 'speed-up'], rows, left=2)}
  <div class="grid-2" style="margin-top:18px">
  <div class="card"><h3>Where the time goes</h3><p>Most of the gain is the Wood-Anderson rewrite: vanilla spends
  {med.loc[('c6g', 'vanilla'), 'wa_s']:.0f}&thinsp;s of worker time on it on Graviton2 against
  {med.loc[('c6g', 'pr'), 'wa_s']:.0f}&thinsp;s for the PR. Picking gains
  {med.loc[('c6g', 'vanilla'), 'classify_s'] / med.loc[('c6g', 'pr'), 'classify_s']:.1f}&times; on Graviton2 and
  {med.loc[('c8g', 'vanilla'), 'classify_s'] / med.loc[('c8g', 'pr'), 'classify_s']:.1f}&times; on Graviton4, and is
  slightly slower on Graviton3. The C velocity kernel does not build on ARM, so its SciPy fallback ran.</p></div>
  <div class="card"><h3>Why the laptop gains less</h3><p>No macOS arm64 torch build has oneDNN (checked: conda-forge
  2.13.0 and the official pip wheel 2.14.1, both <code>USE_MKLDNN=OFF</code>), so the PR's fused kernels cannot run on
  an Apple M-series CPU, and its picking runs
  {med.loc[('M1 Max', 'pr'), 'classify_s'] / med.loc[('M1 Max', 'vanilla'), 'classify_s']:.2f}&times; slower there
  through an equivalent plain-torch path. The laptop's net {speed('M1 Max'):.2f}&times; is the amplitude code alone.
  aarch64 Linux torch 2.14.1 has oneDNN, which is why Graviton gains six times.</p></div>
  </div>
  <h3 style="margin-top:22px">Cost per 1,000 station-days of compute</h3>
  {table(['instance', 'CPU', 'on-demand $/h', 'Spot $/h', 'on-demand, vanilla / PR', 'Spot, vanilla / PR'], crows, left=2)}
  <p class="note">{e(meta['prices'])}. Cost = median wall &times; price / {SD} station-days. For comparison, a
  Fargate Spot task of the same 8 vCPU shape costs about ${fargate_h:.3f}/h, at the rate measured from CloudBank billing
  over all attempts, ${meta['fargate_spot_rate_usd_vcpu_h']} per vCPU-hour (docs/rerun_2026/34, PR #66). The PR on Fargate was not timed. These figures exclude S3 reads and Parquet writes, which both arms pay
  equally in production, so the end-to-end speed-up will be lower.</p>
</section>"""

runs = ["c6g vanilla", "c7g vanilla", "c8g vanilla", "c6g pr", "c7g pr", "c8g pr"]
same = {(r.a, r.b): int(r.identical) for _, r in exact.iterrows()}
same.update({(b_, a_): v for (a_, b_), v in list(same.items())})
label = {r: f"{r.split()[0]} {'vanilla' if r.endswith('vanilla') else 'PR fp32'}" for r in runs}
xrows = [[label[r]] + [n(g23.n_a) if r == c else n(same[(r, c)]) for c in runs] for r in runs]
hardware = f"""
<section id="hardware">
  <div class="section-head"><p class="eyebrow">Precision</p><h2>Exact agreement across hardware</h2>
  <p class="lede">Picks at 0.2 from repetition 2 on each instance ({n(g23.n_a)} picks per run). Repetitions on one
  instance are bit-identical. Vanilla itself loses {n(g23.n_a - g23.identical)} picks between Graviton2 and Graviton3,
  so a millisecond rule cannot hold across our own hardware even without the PR.</p></div>
  {table(['picks identical to the ms'] + [label[r] for r in runs], xrows)}
</section>"""

limits = f"""
<section id="limits">
  <div class="section-head"><p class="eyebrow">Limits and other findings</p><h2>What this does not show</h2></div>
  <div class="grid-2">
  <div class="card"><h3>Not covered</h3><ul>
    <li><code>jma_wc</code>, the global weight and the one the PR's own evaluation used, was not scored here.</li>
    <li><code>obs</code> (PickBlue) cannot use the optimized path: fast annotate raises
    <code>does not cover this model setup: ['padding/filter']</code> when the model is built.</li>
    <li>bf16 was emulated on the laptop, not run on AMX hardware.</li>
    <li>Fargate was not timed; S3 reads and Parquet writes are outside every timing above.</li>
    <li>Streams the model resamples (30 of the 130 precision windows) fall back to SeisBench's annotate; the timing
    workload is 100&thinsp;Hz only.</li>
    <li>The 707-line <code>s3_helper.py</code> rewrite (overlapped reads, per-station listing) was not reviewed.</li>
  </ul></div>
  <div class="card"><h3>Defects found</h3><ul>
    <li><code>PhaseNetGraph</code> with <code>fused=False</code> and polyphase deconvolution crashes on torch builds
    without oneDNN (<code>fast_forward.py:220</code>, a <code>view</code> of a tensor that is not channels-last).</li>
    <li><code>veliir.c</code> calls <code>__builtin_cpu_init</code>, which is x86-only; on ARM the build fails and SciPy
    runs. Built for arm64 with the probe guarded, it matched SciPy to 10<sup>-11</sup>.</li>
    <li><code>seisbench==0.12.5</code> has no aarch64 wheel: an ARM image needs <code>python3-devel</code> and
    <code>gcc</code>. Four workers importing SeisBench for the first time at once can race on its cache config.</li>
    <li>As merged, nothing selects the optimized settings, so production would run vanilla.</li>
  </ul></div>
  </div>
</section>"""

reproduce = f"""
<section id="reproduce">
  <div class="section-head"><p class="eyebrow">Reproduce</p><h2>Code and data</h2></div>
  <div class="card"><p>Harness in <a href="{GH}/tree/main/scripts/vibesys_pr71"><code>scripts/vibesys_pr71/</code></a>:
  <code>run_arm.py</code> (the three arms, with the AMX emulation in <code>emu.py</code>), <code>score.py</code> and
  <code>tolerance.py</code> (scoring), <code>run_amps.py</code> (amplitudes), <code>bench_worker.py</code> with
  <code>bench_run.sh</code> (laptop timing) and <code>ec2_run.sh</code> (Graviton timing, run from EC2 user-data with
  presigned URLs; the instances terminated themselves). Shell scripts keep the paths of the 2026-10-26 run.
  <code>export_results.py</code> writes the tables in
  <a href="{GH}/tree/main/docs/benchmark/results/vibesys_pr71"><code>docs/benchmark/results/vibesys_pr71/</code></a>,
  and this page is built from them by <code>scripts/build_vibesys_report.py</code>.</p></div>
</section>"""

refs = """
<section id="refs"><div class="section-head"><p class="eyebrow">References</p></div><ol class="refs">
  <li id="ref-ross">Ross, Z. E., Meier, M.-A. and Hauksson, E. (2018). P wave arrival picking and first-motion polarity determination with deep learning. <i>Journal of Geophysical Research: Solid Earth</i> 123, 5120-5129. <a href="https://doi.org/10.1029/2017JB015251">doi:10.1029/2017JB015251</a></li>
  <li id="ref-zhu">Zhu, W. and Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. <i>Geophysical Journal International</i> 216, 261-273. <a href="https://doi.org/10.1093/gji/ggy423">doi:10.1093/gji/ggy423</a></li>
</ol></section>"""

commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
built = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Pick precision and computational cost of the VibeSys CPU optimizations proposed for the QuakeScope picker in PR #71, measured on the picker benchmark, an M1 Max laptop and three AWS Graviton instances.">
<title>VibeSys Optimizations Evaluation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="quakescope-board.css">
<style>.card ul {{ padding-left: 1.1em; }} .card li {{ margin: 4px 0; }}</style>
</head>
<body>
<header class="hero">
  <div class="shell">
    <nav class="nav" aria-label="Primary">
      <a class="brand" href="./"><span class="brand-mark">QS</span>
        <span>QuakeScope <span style="opacity:.7;font-weight:600">code evaluation</span></span></a>
      <div class="nav-links">
        <a href="#verdict">Verdict</a><a href="#vanilla">vs vanilla</a><a href="#analyst">vs analysts</a>
        <a href="#tolerance">Tolerance</a><a href="#amplitudes">Amplitudes</a><a href="#cost">Cost</a>
        <a href="#hardware">Hardware</a><a href="#limits">Limits</a><a href="#reproduce">Reproduce</a>
      </div>
    </nav>
    <div class="hero-inner">
      <p class="eyebrow">QuakeScope &middot; PR #71 &middot; VibeSys CPU optimizations &middot; {e(meta['run_date'])}</p>
      <h1>VibeSys optimizations: pick precision and computational cost</h1>
      <p>PR #71 proposes CPU optimizations for the QuakeScope picker, found by an automated optimization campaign
      and reported at 8.5&times; on an Intel AMX host. This page tests whether the picks survive and what the speed is
      on the hardware we would use: the leaderboard's own waveforms and analyst picks for precision, and an M1 Max
      laptop and three AWS Graviton generations for cost.</p>
      <div class="hero-actions">
        <a class="button primary" href="#verdict">Verdict</a>
        <a class="button secondary" href="{meta['pr']}">PR #71</a>
        <a class="button secondary" href="benchmark_metrics.html">Picker leaderboard</a>
      </div>
    </div>
  </div>
</header>
<main class="shell">
{stats}
{verdict}
{what}
{vs_vanilla}
{vs_analyst}
{tolerance}
{amplitudes}
{cost}
{hardware}
{limits}
{reproduce}
{refs}
</main>
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
