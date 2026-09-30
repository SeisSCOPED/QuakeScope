#!/usr/bin/env python
"""Build the picker leaderboard at reports/benchmark_metrics.html.

Every number on the page is read from `docs/benchmark/results/`, which the five
benchmark notebooks write when they execute, and formatted into the template
here. Nothing is typed: if a notebook is re-run and a number moves, this script
moves it on the page. Prose that depends on the direction of a result (which
model leads, whether an ordering holds) is written against a computed value and
guarded by an assertion, so a change of result shows up as a build failure
rather than as a page that contradicts its own tables.

    pixi run -e dev python scripts/build_leaderboard.py

Styling is `reports/quakescope-board.css`, mirrored from the GAIA HazLab design
system so the board can be embedded from HazEvalHub.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sb_catalog.src.benchmark_metrics import ece, match_picks, reliability  # noqa: E402

RES = ROOT / "docs" / "benchmark" / "results"
OUT = ROOT / "reports" / "benchmark_metrics.html"

WEIGHTS = ["quakescope2026", "jma_wc", "original", "instance"]
WCOLOR = {"quakescope2026": "#4b2e83", "jma_wc": "#c2571a",
          "original": "#1b7f79", "instance": "#2f6fb2"}
INK, STONE, LINE, LAV2 = "#2a1a4f", "#6f6890", "#d8d2e8", "#ece8f7"
SHARED_THR, DETECT_TOL = 0.3, 0.5

# What each weight set is, in one line, for the board's first column.
WHAT = {
    "quakescope2026": "Fine-tune of <code>jma_wc</code> on 527k windows, this project, 2026",
    "jma_wc": "PhaseNetWC, Japan Meteorological Agency data, doubled filter width",
    "original": "Zhu &amp; Beroza (2019) weights, northern California",
    "instance": "Trained on INSTANCE, Italian national network",
}

# ---------------------------------------------------------------- references
REFS = [
    ("zhu2019", "Zhu, W. and Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based "
     "seismic arrival-time picking method. <i>Geophysical Journal International</i> 216, 261-273.",
     "https://doi.org/10.1093/gji/ggy423"),
    ("ross2018", "Ross, Z. E., Meier, M.-A., Hauksson, E. and Heaton, T. H. (2018). Generalized "
     "seismic phase detection with deep learning. <i>Bulletin of the Seismological Society of "
     "America</i> 108, 2894-2901.", "https://doi.org/10.1785/0120180080"),
    ("perol2018", "Perol, T., Gharbi, M. and Denolle, M. (2018). Convolutional neural network for "
     "earthquake detection and location. <i>Science Advances</i> 4, e1700578.",
     "https://doi.org/10.1126/sciadv.1700578"),
    ("woollam2019", "Woollam, J., Rietbrock, A., Bueno, A. and De Angelis, S. (2019). Convolutional "
     "neural network for seismic phase classification, performance demonstration over a local "
     "seismic network. <i>Seismological Research Letters</i> 90, 491-502.",
     "https://doi.org/10.1785/0220180312"),
    ("mousavi2020", "Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y. and Beroza, G. C. "
     "(2020). Earthquake transformer: an attentive deep-learning model for simultaneous earthquake "
     "detection and phase picking. <i>Nature Communications</i> 11, 3952.",
     "https://doi.org/10.1038/s41467-020-17591-w"),
    ("soto2021", "Soto, H. and Schurr, B. (2021). DeepPhasePick: a method for detecting and picking "
     "seismic phases from local earthquakes based on highly optimized convolutional and recurrent "
     "deep neural networks. <i>Geophysical Journal International</i> 227, 1268-1286.",
     "https://doi.org/10.1093/gji/ggab266"),
    ("saad2023", "Saad, O. M. et al. (2023). EQCCT: a production-ready earthquake detection and "
     "phase-picking method using the compact convolutional transformer. <i>IEEE Transactions on "
     "Geoscience and Remote Sensing</i> 61, 5910915.",
     "https://doi.org/10.1109/TGRS.2023.3319440"),
    ("feng2022", "Feng, T., Mohanna, S. and Meng, L. (2022). EdgePhase: a deep learning model for "
     "multi-station seismic phase picking. <i>Geochemistry, Geophysics, Geosystems</i> 23, "
     "e2022GC010453.", "https://doi.org/10.1029/2022GC010453"),
    ("sun2023", "Sun, H., Ross, Z. E., Zhu, W. and Azizzadenesheli, K. (2023). Phase neural operator "
     "for multi-station picking of seismic arrivals. <i>Geophysical Research Letters</i> 50, "
     "e2023GL106434.", "https://doi.org/10.1029/2023GL106434"),
    ("liu2024seislm", "Liu, T. et al. (2024). SeisLM: a foundation model for seismic waveforms. "
     "arXiv:2410.15765.", "https://arxiv.org/abs/2410.15765"),
    ("mousavi2019stead", "Mousavi, S. M., Sheng, Y., Zhu, W. and Beroza, G. C. (2019). STanford "
     "EArthquake Dataset (STEAD): a global data set of seismic signals for AI. <i>IEEE Access</i> "
     "7, 179464-179476.", "https://doi.org/10.1109/ACCESS.2019.2947848"),
    ("michelini2021", "Michelini, A. et al. (2021). INSTANCE: the Italian seismic dataset for "
     "machine learning. <i>Earth System Science Data</i> 13, 5509-5544.",
     "https://doi.org/10.5194/essd-13-5509-2021"),
    ("zhao2023", "Zhao, M. et al. (2023). DiTing: a large-scale Chinese seismic benchmark dataset "
     "for artificial intelligence in seismology. <i>Earthquake Science</i> 36, 84-94.",
     "https://doi.org/10.1016/j.eqs.2022.01.022"),
    ("ni2023", "Ni, Y., Denolle, M., Jiang, C., Bodin, T., Ulberg, C. and Shi, Q. (2023). Curated "
     "Pacific Northwest AI-ready seismic dataset. <i>Seismica</i> 2(1).",
     "https://doi.org/10.26443/seismica.v2i1.368"),
    ("aguilar2024", "Aguilar Suarez, A. L. and Beroza, G. C. (2024). Curated Regional Earthquake "
     "Waveforms (CREW) dataset. <i>Seismica</i> 3(1).",
     "https://doi.org/10.26443/seismica.v3i1.1049"),
    ("chen2024", "Chen, Y. et al. (2024). TXED: the Texas earthquake dataset for AI. "
     "<i>Seismological Research Letters</i> 95, 2013-2022.",
     "https://doi.org/10.1785/0220230327"),
    ("magrini2020", "Magrini, F., Jozinovic, D., Cammarano, F., Michelini, A. and Boschi, L. (2020). "
     "Local earthquakes detection: a benchmark dataset of 3-component seismograms built on a global "
     "scale. <i>Artificial Intelligence in Geosciences</i> 1, 1-10.",
     "https://doi.org/10.1016/j.aiig.2020.04.001"),
    ("woollam2022", "Woollam, J. et al. (2022). SeisBench: a toolbox for machine learning in "
     "seismology. <i>Seismological Research Letters</i> 93, 1695-1709.",
     "https://doi.org/10.1785/0220210324"),
    ("munchmeyer2022", "M&uuml;nchmeyer, J. et al. (2022). Which picker fits my data? A quantitative "
     "evaluation of deep learning based seismic pickers. <i>Journal of Geophysical Research: Solid "
     "Earth</i> 127, e2021JB023499.", "https://doi.org/10.1029/2021JB023499"),
    ("pita2023", "Pita-Sllim, O., Chamberlain, C. J., Townend, J. and Warren-Smith, E. (2023). "
     "Parametric testing of EQTransformer's performance against a high-quality, manually picked "
     "catalog for reliable and accurate seismic phase picking. <i>The Seismic Record</i> 3, 332-341.",
     "https://doi.org/10.1785/0320230024"),
    ("bornstein2024", "Bornstein, T. et al. (2024). PickBlue: seismic phase picking for ocean bottom "
     "seismometers with deep learning. <i>Earth and Space Science</i> 11, e2023EA003332.",
     "https://doi.org/10.1029/2023EA003332"),
    ("niksejel2024", "Niksejel, A. and Zhang, M. (2024). OBSTransformer: a deep-learning seismic "
     "phase picker for OBS data using automated labelling and transfer learning. <i>Geophysical "
     "Journal International</i> 237, 485-505.", "https://doi.org/10.1093/gji/ggae049"),
    ("yeck2020", "Yeck, W. L. et al. (2020). Leveraging deep learning in global 24/7 real-time "
     "earthquake monitoring at the National Earthquake Information Center. <i>Seismological "
     "Research Letters</i> 92, 469-480.", "https://doi.org/10.1785/0220200178"),
    ("ross2019", "Ross, Z. E., Trugman, D. T., Hauksson, E. and Shearer, P. M. (2019). Searching for "
     "hidden earthquakes in southern California. <i>Science</i> 364, 767-771.",
     "https://doi.org/10.1126/science.aaw6888"),
    ("tan2021", "Tan, Y. J. et al. (2021). Machine-learning-based high-resolution earthquake catalog "
     "reveals how complex fault structures were activated during the 2016-2017 central Italy "
     "sequence. <i>The Seismic Record</i> 1, 11-19.", "https://doi.org/10.1785/0320210001"),
    ("wilding2023", "Wilding, J. D., Zhu, W., Ross, Z. E. and Jackson, J. M. (2023). The magmatic web "
     "beneath Hawai'i. <i>Science</i> 379, 462-468.", "https://doi.org/10.1126/science.ade5755"),
    ("zhu2022", "Zhu, W., McBrearty, I. W., Mousavi, S. M., Ellsworth, W. L. and Beroza, G. C. "
     "(2022). Earthquake phase association using a Bayesian Gaussian mixture model. <i>Journal of "
     "Geophysical Research: Solid Earth</i> 127, e2021JB023249.",
     "https://doi.org/10.1029/2021JB023249"),
    ("munchmeyer2024", "M&uuml;nchmeyer, J. (2024). PyOcto: a high-throughput seismic phase "
     "associator. <i>Seismica</i> 3(1).", "https://doi.org/10.26443/seismica.v3i1.1130"),
    ("woessner2005", "Woessner, J. and Wiemer, S. (2005). Assessing the quality of earthquake "
     "catalogues: estimating the magnitude of completeness and its uncertainty. <i>Bulletin of the "
     "Seismological Society of America</i> 95, 684-698.", "https://doi.org/10.1785/0120040007"),
    ("guo2017", "Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern "
     "neural networks. <i>Proceedings of the 34th International Conference on Machine Learning</i>, "
     "PMLR 70, 1321-1330.", "https://arxiv.org/abs/1706.04599"),
    ("bekker2020", "Bekker, J. and Davis, J. (2020). Learning from positive and unlabeled data: a "
     "survey. <i>Machine Learning</i> 109, 719-760.",
     "https://doi.org/10.1007/s10994-020-05877-5"),
    ("chicco2020", "Chicco, D. and Jurman, G. (2020). The advantages of the Matthews correlation "
     "coefficient (MCC) over F1 score and accuracy in binary classification evaluation. <i>BMC "
     "Genomics</i> 21, 6.", "https://doi.org/10.1186/s12864-019-6413-7"),
    ("dehghani2021", "Dehghani, M. et al. (2021). The benchmark lottery. arXiv:2107.07002.",
     "https://arxiv.org/abs/2107.07002"),
    ("raji2021", "Raji, I. D., Bender, E. M., Paullada, A., Denton, E. and Hanna, A. (2021). AI and "
     "the everything in the whole wide world benchmark. arXiv:2111.15366.",
     "https://arxiv.org/abs/2111.15366"),
    ("mousavi2023", "Mousavi, S. M. and Beroza, G. C. (2023). Machine learning in earthquake "
     "seismology. <i>Annual Review of Earth and Planetary Sciences</i> 51, 105-129.",
     "https://doi.org/10.1146/annurev-earth-071822-100323"),
]
RKEY = {k: i + 1 for i, (k, _, _) in enumerate(REFS)}


def cite(*keys: str) -> str:
    """Superscript reference marks, e.g. cite('zhu2019', 'ross2018')."""
    for k in keys:
        if k not in RKEY:
            raise KeyError(f"no reference {k!r}")
    inner = ",&thinsp;".join(f'<a href="#r{RKEY[k]}">{RKEY[k]}</a>' for k in keys)
    return f'<sup class="c">[{inner}]</sup>'


# ---------------------------------------------------------------- data
def load():
    d = pd.read_csv(RES / "detection_full.csv")
    d = d[d.n_ref > 0].copy()
    t = pd.read_csv(RES / "timing_full.csv")
    q = pd.read_csv(RES / "phase_quality_full.csv")
    c = pd.read_csv(RES / "calibration_full.csv")
    sweep = pd.concat([pd.read_csv(RES / s / "threshold_sweep.csv")
                       .assign(study=s.replace("_sequences", "")) for s in
                       ("us_sequences", "global_sequences")], ignore_index=True)
    meta = {p.parent.name: json.loads(p.read_text()) for p in RES.glob("*/meta.json")}
    return d, t, q, c, sweep, meta


def wmean(g: pd.DataFrame, col: str, w: str = "n_ref") -> float:
    g = g.dropna(subset=[col])
    return float(np.average(g[col], weights=g[w])) if len(g) else float("nan")


def protocol_table(d: pd.DataFrame) -> pd.DataFrame:
    """Arrival-weighted recall and win counts under the three threshold protocols."""
    rows = []
    keys = ["study", "sequence", "phase"]
    for col, name in (("recall_at_03", f"Shared threshold {SHARED_THR}"),
                      ("recall_at_budget", "Matched pick budget"),
                      ("recall_at_best", "Each model's own best threshold")):
        s = d.dropna(subset=[col])
        wins = s.loc[s.groupby(keys)[col].idxmax()].weights.value_counts()
        for m in WEIGHTS:
            g = s[s.weights == m]
            rows.append({"protocol": name, "col": col, "weights": m,
                         "recall": wmean(g, col), "wins": int(wins.get(m, 0)),
                         "rows": len(g)})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("protocol").recall.rank(ascending=False, method="min").astype(int)
    return out


def timing_table(d: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    m = t.merge(d[["study", "sequence", "phase", "weights", "n_ref"]],
                on=["study", "sequence", "phase", "weights"], how="inner")
    rows = []
    for (ph, w), g in m[m.medae.notna()].groupby(["phase", "weights"]):
        rows.append({"phase": ph, "weights": w, "n": int(g.n.sum()),
                     "medae": float(np.average(g.medae, weights=g.n)),
                     "mae": float(np.average(g.mae, weights=g.n)),
                     "bias": float(np.average(g.bias, weights=g.n)),
                     "gross": float(np.average(g.gross_error_rate, weights=g.n)),
                     "w01": float(np.average(g["within_0.1"], weights=g.n))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=STONE, labelsize=8.5, length=3)
    ax.grid(True, color=LINE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK); lbl.set_fontsize(9.5)
    ax.title.set_color(INK); ax.title.set_fontsize(10)


def fig_protocols(prot: pd.DataFrame) -> str:
    """Slope chart: the ranking under each protocol, joined per model."""
    order = ["recall_at_03", "recall_at_budget", "recall_at_best"]
    labels = ["Shared\nthreshold 0.3", "Matched\npick budget", "Own best\nthreshold"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(order))
    for m in WEIGHTS:
        y = [prot[(prot.weights == m) & (prot.col == c)].recall.iloc[0] for c in order]
        ax.plot(x, y, "-o", color=WCOLOR[m], lw=2.2, ms=7, label=m, zorder=3)
        ax.annotate(f"{y[0]:.3f}", (x[0], y[0]), textcoords="offset points",
                    xytext=(-10, 0), ha="right", va="center", fontsize=8.5, color=WCOLOR[m])
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(10, 0), ha="left", va="center", fontsize=8.5, color=WCOLOR[m])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("recall, weighted by reference arrivals")
    ax.set_title("The same four weight sets, scored three ways")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower center",
              ncol=4, bbox_to_anchor=(0.5, -0.30))
    return _svg(fig)


def fig_curves(sweep: pd.DataFrame, d: pd.DataFrame) -> str:
    """Recall against picks emitted, with each model's shared-threshold point marked."""
    panels = [("global", "Kaikoura 2016", "P"), ("global", "Norcia 2016", "P"),
              ("global", "Thessaly 2021", "P"), ("us", "Ridgecrest", "P"),
              ("us", "Mendocino 2024", "S"), ("global", "Norcia 2016", "S")]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.6))
    for ax, (study, seq, ph) in zip(axes.ravel(), panels):
        sub = sweep[(sweep.study == study) & (sweep.sequence == seq) & (sweep.phase == ph)]
        for m in WEIGHTS:
            g = sub[sub.weights == m].sort_values("emitted")
            if g.empty:
                continue
            ax.plot(g.emitted, g.recall, color=WCOLOR[m], lw=1.8, zorder=3)
            row = d[(d.study == study) & (d.sequence == seq) & (d.phase == ph)
                    & (d.weights == m)]
            if len(row):
                ax.plot(row.emitted_at_03.iloc[0], row.recall_at_03.iloc[0], "o",
                        color=WCOLOR[m], ms=6, mec="white", mew=1.2, zorder=4)
        bud = d[(d.study == study) & (d.sequence == seq) & (d.phase == ph)].budget
        if len(bud) and pd.notna(bud.iloc[0]):
            ax.axvline(bud.iloc[0], color=STONE, lw=1.0, ls=":", zorder=2)
        ax.set_title(f"{seq}, {ph}", fontsize=9.5)
        ax.set_xscale("log")
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("recall")
    for ax in axes[1, :]:
        ax.set_xlabel("picks emitted (log)")
    fig.tight_layout()
    return _svg(fig)


def fig_reliability(published: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Observed agreement against stated confidence, per weight set and study.

    Every emitted pick counts, with no confidence floor: filtering at a
    threshold first would cut the low-confidence bins and leave a curve that
    only describes the top of the range. This reproduces the notebook cell that
    writes `calibration_full.csv`, and the result is checked against it.
    """
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    for ax, study in zip(axes, ("us_sequences", "global_sequences")):
        picks = pd.read_csv(RES / study / "model_picks.csv", parse_dates=["time"])
        ref = pd.read_csv(RES / study / "reference_picks.csv", parse_dates=["time"])
        ax.plot([0, 1], [0, 1], ls="--", color=STONE, lw=1.0, zorder=2)
        for m in WEIGHTS:
            conf, hit = [], []
            g = picks[picks.weights == m]
            for (seq, ph, sta), gg in g.groupby(["sequence", "phase", "station"]):
                rr = sorted(ref[(ref.sequence == seq) & (ref.phase == ph)
                                & (ref.station == sta)].time.astype("int64") / 1e9)
                gg = gg.sort_values("time")
                tt, cc = list(gg.time.astype("int64") / 1e9), list(gg.conf)
                h = np.zeros(len(tt), dtype=bool)
                for _, j, _, _ in match_picks(rr, tt, cc, tol=DETECT_TOL)["pairs"]:
                    h[j] = True
                conf += cc; hit += list(h)
            tab = reliability(conf, hit)
            e = ece(conf, hit)
            rows.append({"study": study.replace("_sequences", ""), "weights": m,
                         "ece_lb": e, "n": len(conf)})
            ax.plot(tab.mean_conf, tab.observed_lb, "-o", color=WCOLOR[m], lw=1.8,
                    ms=5, label=f"{m}  ECE {e:.3f}", zorder=3)
        ax.set_title(study.replace("_sequences", "").upper() + " sequences", fontsize=10)
        ax.set_xlabel("stated confidence")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("observed agreement with the\nreference bulletin (lower bound)")
    fig.tight_layout()
    out = pd.DataFrame(rows)
    chk = out.merge(published, on=["study", "weights"], suffixes=("", "_pub"))
    assert len(chk) == len(out), "calibration_full.csv does not cover every study and weight set"
    gap = (chk.ece_lb - chk.ece_lb_pub).abs().max()
    assert gap < 1e-9, (f"the board's calibration disagrees with the methods page by {gap:.2e}; "
                        "the two are meant to be the same computation")
    return _svg(fig), out


# ---------------------------------------------------------------- html bits
def chip(m: str) -> str:
    return f'<span class="chip-w w-{m}"><span class="sw"></span>{m}</span>'


def rank_badge(r: int) -> str:
    return f'<span class="rank r{r if r <= 3 else 0}">{r}</span>'


def fmt(v, nd=3, dash="&mdash;"):
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def main() -> None:
    d, t, q, c, sweep, meta = load()
    prot = protocol_table(d)
    tim = timing_table(d, t)
    seq_phases = d.groupby(["study", "sequence", "phase"]).ngroups
    arrivals = int(d.groupby(["study", "sequence", "phase"]).n_ref.first().sum())
    n_studies = len(meta)
    # Agencies that served the reference arrivals: the two US data centres plus one
    # per foreign sequence, read from the provenance table each notebook exports.
    prov = pd.read_csv(RES / "global_sequences" / "reference_provenance.csv")
    n_agencies = prov.sequence.nunique() + 2
    n_tests = sum(len([l for l in (ROOT / "tests" / f).read_text().splitlines()
                       if l.startswith("def test_")])
                  for f in ("test_benchmark_metrics.py", "test_score_picks_cli.py"))

    # --- the claims the prose makes, checked against the data ---------------
    p03 = prot[prot.col == "recall_at_03"].set_index("weights")
    pbud = prot[prot.col == "recall_at_budget"].set_index("weights")
    pbest = prot[prot.col == "recall_at_best"].set_index("weights")
    lead03, leadbud = p03.recall.idxmax(), pbud.recall.idxmax()
    spread03 = p03.recall.max() - p03.recall.min()
    spreadbest = pbest.recall.max() - pbest.recall.min()
    assert lead03 != leadbud, "prose says the protocols disagree at the top"
    assert spreadbest < spread03 / 3, "prose says own-threshold scoring collapses the spread"

    # which model's ceiling binds the matched budget, per sequence-phase
    mx = sweep.groupby(["study", "sequence", "phase", "weights"]).emitted.max().unstack()
    binding = mx.idxmin(axis=1).value_counts()
    bind_model, bind_n, bind_tot = binding.index[0], int(binding.iloc[0]), int(binding.sum())
    assert bind_n == bind_tot, "prose says one model's ceiling binds every row"

    # leader flips per sequence-phase
    r03 = d.pivot_table(index=["study", "sequence", "phase"], columns="weights", values="recall_at_03")
    rbud = d.pivot_table(index=["study", "sequence", "phase"], columns="weights", values="recall_at_budget")
    flips = int((r03.idxmax(axis=1) != rbud.idxmax(axis=1)).sum())

    ceilings = mx.mean().to_dict()          # mean picks each weight can emit at the floor
    fig1 = fig_protocols(prot)
    fig2 = fig_curves(sweep, d)
    fig3, cal = fig_reliability(c)
    cal_p = cal.pivot_table(index="weights", columns="study", values="ece_lb")
    best_cal = cal.groupby("weights").ece_lb.mean().idxmin()
    worst_cal = cal.groupby("weights").ece_lb.mean().idxmax()

    best_time_p = tim[tim.phase == "P"].set_index("weights").medae.idxmin()
    best_time_s = tim[tim.phase == "S"].set_index("weights").medae.idxmin()
    qw = q.groupby("weights").apply(
        lambda x: pd.Series({"swap": np.average(x.swap_rate, weights=x.P_n + x.S_n),
                             "dup": float(x.duplicate_rate.max())}), include_groups=False)
    best_swap, worst_swap = qw.swap.idxmin(), qw.swap.idxmax()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ---------------------------------------------------------------- render
    H: list[str] = []
    A = H.append

    A(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A leaderboard for deep-learning seismic phase pickers, scored against operator bulletins on {seq_phases} sequence-phases and {arrivals:,} analyst arrivals, with metrics chosen for the goal of building earthquake catalogues.">
<title>Phase-picker leaderboard &mdash; QuakeScope</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="quakescope-board.css">
</head>
<body>

<header class="hero">
  <div class="shell">
    <nav class="nav" aria-label="Primary">
      <a class="brand" href="./"><span class="brand-mark">QS</span>
        <span>QuakeScope <span style="opacity:.7;font-weight:600">picker board</span></span></a>
      <div class="nav-links">
        <a href="#why">Why</a><a href="#board">The board</a><a href="#metrics">Metrics</a>
        <a href="#standard">Standard</a><a href="#run">Score your picks</a><a href="#refs">References</a>
      </div>
    </nav>
    <div class="hero-inner">
      <p class="eyebrow">HazEvalHub &middot; catalogue-workflow track &middot; SeisSCOPED</p>
      <h1>A leaderboard for phase pickers, scored on what an earthquake catalogue needs</h1>
      <p>Deep learning replaced the first stage of earthquake catalogue production faster than
      the field agreed on how to score it. Four published sets of PhaseNet weights are ranked
      here on {seq_phases} sequence-phases and {arrivals:,} analyst arrivals from {n_agencies} agencies on
      two continents, under three threshold protocols, on detection, onset time, confidence
      calibration and phase identification.</p>
      <p>The headline result is about benchmarking, not about a winner: <strong>the ranking
      inverts</strong> depending on how the operating point is held, and when each model is
      allowed its own threshold the four are within {spreadbest:.3f} recall of each other.</p>
      <div class="hero-actions">
        <a class="button primary" href="#board">See the board</a>
        <a class="button secondary" href="#run">Score your own picks</a>
        <a class="button secondary" href="benchmark_metrics_methods.html">Methods notebook</a>
      </div>
    </div>
  </div>
</header>

<main class="shell">
<div class="stats">
  <div class="stat"><div class="n">4</div><div class="k">published weight sets scored</div></div>
  <div class="stat"><div class="n">{seq_phases}</div><div class="k">sequence-phases</div></div>
  <div class="stat"><div class="n">{arrivals:,}</div><div class="k">analyst arrivals as reference</div></div>
  <div class="stat"><div class="n">{n_studies}</div><div class="k">benchmark studies</div></div>
  <div class="stat"><div class="n">{spread03:.2f}</div><div class="k">recall spread the protocol creates</div></div>
</div>
""")

    # ---------------------------------------------------------------- why
    A(f"""
<section id="why">
  <div class="section-head">
    <p class="eyebrow">Why this board exists</p>
    <h2>The picker is now the first stage of most new earthquake catalogues</h2>
    <p class="lede">A catalogue is the product that seismology hands to everyone else: to fault
    studies, to hazard models, to the people who have to decide whether a swarm matters. Since
    2018 the stage that produces it has been a neural network, and the catalogues that followed
    changed what is known about fault structure in southern California{cite('ross2019')}, about
    the 2016-2017 central Italy sequence{cite('tan2021')} and about magma transport beneath
    Hawai'i{cite('wilding2023')}. One runs in real time at the National Earthquake Information
    Center{cite('yeck2020')}. Choosing the picker is therefore a scientific decision with a
    measurable consequence, and it is usually made by reading a number off a benchmark.</p>
  </div>

  <h3>Architectures diverged</h3>
  <p>The models being compared are not variations on one design. They make different assumptions
  about what a seismogram is, and those assumptions show up in how they fail.</p>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">Family</th><th class="l">Idea</th><th class="l">Examples</th></tr></thead>
    <tbody>
      <tr><td class="l">Fully convolutional, encoder-decoder</td>
          <td class="l">Predict a probability time series per phase over a window; picks are its peaks</td>
          <td class="l">PhaseNet{cite('zhu2019')}</td></tr>
      <tr><td class="l">Window classifier</td>
          <td class="l">Slide a short window and classify it P, S or noise</td>
          <td class="l">GPD{cite('ross2018')}, ConvNetQuake{cite('perol2018')}, CNN phase classifiers{cite('woollam2019')}</td></tr>
      <tr><td class="l">Recurrent, with attention</td>
          <td class="l">Detect the event first, then pick inside it; explicit sequence memory</td>
          <td class="l">EQTransformer{cite('mousavi2020')}, DeepPhasePick{cite('soto2021')}</td></tr>
      <tr><td class="l">Transformer</td>
          <td class="l">Self-attention over the trace in place of recurrence</td>
          <td class="l">EQCCT{cite('saad2023')}, OBSTransformer{cite('niksejel2024')}</td></tr>
      <tr><td class="l">Multi-station and operator learning</td>
          <td class="l">Pick a network jointly, so one station's noise is another's context</td>
          <td class="l">EdgePhase{cite('feng2022')}, phase neural operator{cite('sun2023')}</td></tr>
      <tr><td class="l">Pre-trained backbones</td>
          <td class="l">One self-supervised model, fine-tuned per task</td>
          <td class="l">SeisLM{cite('liu2024seislm')}</td></tr>
    </tbody>
    <caption>A picker's architecture determines what it can use as evidence. A window classifier
    cannot exploit the shape of a coda; a multi-station model can be defeated by one clock
    error. Reviews of the wider field: {cite('mousavi2023')}.</caption>
  </table></div>

  <h3>Curated datasets diverged with them</h3>
  <p>Each of these models was trained on a labelled corpus assembled by a different network, with
  its own instrumentation, noise, depth distribution and analyst conventions. That is what makes
  a single benchmark number hard to read: a picker evaluated near its training distribution is
  being asked an easier question than the same picker deployed elsewhere.</p>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">Corpus</th><th class="l">Source network</th><th class="l">Used here</th></tr></thead>
    <tbody>
      <tr><td class="l">STEAD{cite('mousavi2019stead')}</td><td class="l">Global, mostly regional distances</td><td class="l">upstream of several weight sets</td></tr>
      <tr><td class="l">INSTANCE{cite('michelini2021')}</td><td class="l">Italian national network</td><td class="l">trains <code>instance</code>; Norcia is in-domain for it</td></tr>
      <tr><td class="l">DiTing{cite('zhao2023')}</td><td class="l">China</td><td class="l">not scored here</td></tr>
      <tr><td class="l">PNW{cite('ni2023')}</td><td class="l">Pacific Northwest, incl. surface events</td><td class="l">in the <code>quakescope2026</code> fine-tune corpus</td></tr>
      <tr><td class="l">CREW{cite('aguilar2024')}</td><td class="l">Regional, continental US</td><td class="l">not scored here</td></tr>
      <tr><td class="l">TXED{cite('chen2024')}</td><td class="l">Texas</td><td class="l">not scored here</td></tr>
      <tr><td class="l">LEN-DB{cite('magrini2020')}</td><td class="l">Global, local events</td><td class="l">not scored here</td></tr>
      <tr><td class="l">OBS corpora{cite('bornstein2024','niksejel2024')}</td><td class="l">Ocean-bottom deployments</td><td class="l">offshore track only</td></tr>
    </tbody>
    <caption>SeisBench{cite('woollam2022')} made these corpora and the models interchangeable in
    code. That is what made systematic comparison possible, and also what made it easy to
    compare a model against whichever corpus was nearest to hand.</caption>
  </table></div>

  <h3>The evaluation did not keep up</h3>
  <p>The reference study is M&uuml;nchmeyer et al.{cite('munchmeyer2022')}, which scored pickers
  across regions on labelled datasets and established the cross-domain result: a picker
  transfers between regions with mild degradation, and does not transfer from regional to
  teleseismic distances. Parametric work on single models has since shown how strongly a
  reported score depends on the thresholds chosen{cite('pita2023')}.</p>
  <p>What a person deploying a picker faces differs from a labelled-dataset benchmark in three
  ways, and each one changes the answer:</p>
  <ol>
    <li><strong>The reference is an operator bulletin, not a labelled set.</strong> An analyst
    picked what a location needed and stopped. An unmatched model pick is a mixture of a false
    positive and a real arrival nobody marked, which is a positive-unlabeled learning
    problem{cite('bekker2020')}: precision, F1 and MCC{cite('chicco2020')} are not identifiable,
    and every number reported for them is a bound.</li>
    <li><strong>A threshold is not an operating point.</strong> The four weight sets put their
    probabilities on different scales, so at a shared 0.3 they emit between
    {int(d.groupby('weights').emitted_at_03.mean().min())} and
    {int(d.groupby('weights').emitted_at_03.mean().max())} picks per sequence-phase on average.
    Recall at a shared threshold measures liberality as much as skill. This is the benchmark
    lottery{cite('dehghani2021')} in a domain that can check it: the protocol, not the model,
    decides the winner.</li>
    <li><strong>The deployment is the experiment.</strong> Picking 114 million station-days
    exposes failure modes a windowed benchmark cannot: resumed jobs overwriting their own
    output, station metadata that truncates epochs, archive listings that fail silently. Each of
    those changed catalogue completeness by more than the difference between the four models
    below, and the quantity a catalogue user actually cares about is completeness{cite('woessner2005')}.</li>
  </ol>
  <div class="callout">
    <h3>What this board is, and is not</h3>
    <p>It is a scored comparison with a public, deterministic scorer, run on public bulletins,
    reported under three protocols at once. It is not yet a benchmark with a hidden test set
    that a result could be cited from. The <a href="#standard">standard below</a> scores this
    track against nine rules for a citable benchmark and records where it fails, which is the
    part of a benchmark that readers are usually not given{cite('raji2021')}.</p>
  </div>
</section>
""")

    # ---------------------------------------------------------------- board
    A(f"""
<section id="board">
  <div class="section-head">
    <p class="eyebrow">The board</p>
    <h2>Detection: three protocols, three orderings</h2>
    <p class="lede">Recall is the fraction of analyst arrivals recovered within
    {DETECT_TOL:g}&thinsp;s on the same station and phase. It is the one detection quantity that
    a non-exhaustive reference leaves identifiable, and it is weighted here by the number of
    reference arrivals, so a sequence with {int(d.n_ref.max())} arrivals counts more than one
    with {int(d.n_ref.min())}.</p>
  </div>
""")
    for col, name, note in (
        ("recall_at_03", f"Protocol A &mdash; shared threshold {SHARED_THR}",
         "What almost every published comparison reports. It also measures which model is most "
         "willing to emit a pick."),
        ("recall_at_budget", "Protocol B &mdash; matched pick budget",
         f"Each model's own curve read at the same number of emitted picks. Threshold-free, but "
         f"bounded above by the most conservative model: <code>{bind_model}</code>'s ceiling sets "
         f"the budget in every one of the {bind_tot} rows where all four ceilings are "
         f"measured, so the comparison happens where a conservative model is strongest."),
        ("recall_at_best", "Protocol C &mdash; each model's own best threshold",
         "Per-model threshold selection, as the cross-domain benchmark does on a development "
         "set. With no held-out split here this is an upper bound on a tuned deployment.")):
        sub = prot[prot.col == col].sort_values("recall", ascending=False)
        lastcol = {"recall_at_03": "mean picks emitted",
                   "recall_at_budget": "mean ceiling (picks at the floor)",
                   "recall_at_best": "mean threshold used"}[col]
        A(f'  <div class="card"><h3>{name}</h3>\n  <div class="table-scroll">'
          '<table class="board"><thead><tr><th class="l">#</th><th class="l">weight set</th>'
          '<th class="l">what it is</th><th>recall</th><th>rows won</th>'
          f'<th>{lastcol}</th></tr></thead><tbody>')
        for _, r in sub.iterrows():
            em = d[d.weights == r.weights]
            if col == "recall_at_03":
                emv = f"{em.emitted_at_03.mean():,.0f}"
            elif col == "recall_at_budget":
                emv = f"{ceilings[r.weights]:,.0f}"
            else:
                emv = f"{em.best_thr.mean():.2f}"
            A(f'    <tr><td class="l">{rank_badge(int(r["rank"]))}</td>'
              f'<td class="l">{chip(r.weights)}</td><td class="l" style="color:#6f6890;'
              f'font-size:.82rem">{WHAT[r.weights]}</td>'
              f'<td><strong>{r.recall:.3f}</strong></td><td>{r.wins} / {int(r.rows)}</td>'
              f'<td>{emv}</td></tr>')
        A(f'  </tbody><caption>{note}</caption></table></div></div>')

    A(f"""
  <figure>{fig1}
    <figcaption>Each line is one weight set. The order at the left is the order a fixed-threshold
    benchmark would publish; the order in the middle is what a threshold-free comparison gives at
    a budget the most conservative model bounds; the right is what each model reaches when it is
    allowed its own threshold. <strong>{lead03}</strong> leads protocol A and comes last in
    protocol B; <strong>{leadbud}</strong> does the reverse. Per sequence-phase, the leader
    changes between A and B in {flips} of {seq_phases} rows.</figcaption>
  </figure>

  <div class="callout warn">
    <h3>Read this before quoting a rank</h3>
    <p>The spread between best and worst weight set is {spread03:.3f} recall under protocol A
    and {spreadbest:.3f} under protocol C, so {100 * (1 - spreadbest / spread03):.0f}&thinsp;% of
    the apparent difference between these four models comes from holding the threshold fixed
    rather than from how they pick. Under protocol C the win counts are
    {' / '.join(str(int(pbest.loc[m, 'wins'])) for m in pbest.sort_values('recall', ascending=False).index)}
    across {seq_phases} rows, which is not a ranking. Detection is where these weight sets are
    hardest to separate. The axes below separate them.</p>
  </div>

  <figure>{fig2}
    <figcaption>Recall against picks emitted, six of the {seq_phases} sequence-phases, log x.
    Filled circles mark each model's shared-threshold 0.3 operating point, the dotted line the
    matched budget. Where a curve stops, the model has run out of picks with its threshold on the
    floor: that is a ceiling, not a calibration offset, and no threshold recovers it.</figcaption>
  </figure>
</section>
""")

    # ---------------------------------------------------------------- timing
    A(f"""
<section id="timing">
  <div class="section-head">
    <p class="eyebrow">The board, continued</p>
    <h2>Onset time, calibration, and phase identification</h2>
    <p class="lede">These three axes are identifiable against a bulletin, they are what a
    location and a magnitude actually consume, and unlike detection they separate the four
    weight sets.</p>
  </div>

  <div class="card"><h3>Onset time</h3>
  <div class="table-scroll"><table class="board">
    <thead><tr><th class="l">phase</th><th class="l">weight set</th><th>matched picks</th>
    <th>MedianAE (s)</th><th>MAE (s)</th><th>median bias (s)</th>
    <th>within 0.1&thinsp;s</th><th>gross error &gt;{DETECT_TOL:g}&thinsp;s</th></tr></thead><tbody>""")
    for ph in ("P", "S"):
        sub = tim[tim.phase == ph].sort_values("medae")
        for i, (_, r) in enumerate(sub.iterrows()):
            lead = ' class="lead"' if i == 0 else ""
            A(f'    <tr><td class="l">{ph if i == 0 else ""}</td><td class="l">{chip(r.weights)}</td>'
              f'<td>{r.n:,}</td><td{lead}>{r.medae:.3f}</td><td>{r.mae:.3f}</td>'
              f'<td>{r.bias:+.3f}</td><td>{r.w01:.3f}</td><td>{r.gross:.3f}</td></tr>')
    A(f"""  </tbody>
    <caption>Residuals are matched at 2&thinsp;s and detection at {DETECT_TOL:g}&thinsp;s, because
    matching at the detection tolerance truncates the residual distribution there and makes any
    outlier rate a statement about the tolerance. MedianAE and MAE are both reported because one
    is insensitive to outliers and the other is not{cite('munchmeyer2022')}; the 0.1&thinsp;s
    column is the tolerance PhaseNet was originally scored at{cite('zhu2019')}.
    <code>{best_time_p}</code> is most accurate on P and <code>{best_time_s}</code> on S.</caption>
  </table></div></div>

  <div class="card"><h3>Is the confidence a probability?</h3>
  <p>Every downstream user thresholds on <code>conf</code>, and an associator weights by
  it{cite('zhu2022','munchmeyer2024')}. Expected calibration error is the mean gap between stated
  confidence and observed agreement, weighted by bin count{cite('guo2017')}. Against a bulletin
  the observed rate is a lower bound, so this is <code>ece_lb</code>; the shape still explains why
  one threshold means different things to different weight sets.</p>
  <figure>{fig3}
    <figcaption>Perfect calibration is the dashed diagonal. Every weight set sits below it: a
    pick labelled 0.8 agrees with the bulletin less often than 80&thinsp;% of the time, partly
    because the bulletin is not exhaustive and partly because the models are overconfident.
    <code>{best_cal}</code> is the best calibrated and <code>{worst_cal}</code> the worst, by a
    factor of {cal.groupby('weights').ece_lb.mean().max() / cal.groupby('weights').ece_lb.mean().min():.1f}.
    That is the mechanism behind protocol A's ordering: the worst-calibrated model is the most
    liberal at any given threshold, so a fixed threshold flatters it.</figcaption>
  </figure>
  <div class="table-scroll"><table class="board">
    <thead><tr><th class="l">weight set</th>""")
    for st in cal_p.columns:
        A(f"<th>ECE<sub>lb</sub>, {st}</th>")
    A("<th>picks scored</th></tr></thead><tbody>")
    for m in cal.groupby("weights").ece_lb.mean().sort_values().index:
        A(f'    <tr><td class="l">{chip(m)}</td>')
        for st in cal_p.columns:
            v = cal_p.loc[m, st]
            best = ' class="lead"' if v == cal_p[st].min() else ""
            A(f"<td{best}>{v:.3f}</td>")
        A(f'<td>{int(cal[cal.weights == m].n.sum()):,}</td></tr>')
    A(f"""  </tbody></table></div></div>

  <div class="card"><h3>Phase identification and duplicate picks</h3>
  <p>A P reported where the analyst marked an S is a different failure from a miss: it survives
  association and moves a location. Both sides of the comparison carry a phase label, so unlike
  precision this is identifiable.</p>
  <div class="table-scroll"><table class="board">
    <thead><tr><th class="l">weight set</th><th>P arrivals picked as S</th>
    <th>S arrivals picked as P</th><th>swap rate</th><th>duplicate rate</th></tr></thead><tbody>""")
    qs = q.groupby("weights")[["P_swapped_to_S", "S_swapped_to_P", "P_n", "S_n"]].sum()
    for m in qw.swap.sort_values().index:
        lead = ' class="lead"' if m == best_swap else ""
        A(f'    <tr><td class="l">{chip(m)}</td><td>{int(qs.loc[m, "P_swapped_to_S"])} / '
          f'{int(qs.loc[m, "P_n"])}</td><td>{int(qs.loc[m, "S_swapped_to_P"])} / '
          f'{int(qs.loc[m, "S_n"])}</td><td{lead}>{qw.loc[m, "swap"]:.4f}</td>'
          f'<td>{qw.loc[m, "dup"]:.3f}</td></tr>')
    A(f"""  </tbody>
    <caption>Swap rates are low for every weight set and differ by a factor of
    {qw.swap.max() / qw.swap.min():.1f} between <code>{best_swap}</code> and
    <code>{worst_swap}</code>. No weight set emits duplicate picks on an arrival it already
    matched, at any threshold tested.</caption>
  </table></div></div>
</section>
""")

    # ---------------------------------------------------------------- metrics
    ident = [
        ("Recall", "matched reference arrivals / reference arrivals",
         "exact", cite('munchmeyer2022'), "The primary metric. Unaffected by the reference being incomplete."),
        ("Picks emitted", "count above the threshold", "exact", "",
         "Recall alone is gameable by lowering the threshold. Always report both."),
        ("Recall at matched budget", "each model's curve read at equal emitted picks",
         "exact", cite('munchmeyer2022'), "The comparison that survives a change of threshold, bounded by the most conservative model's ceiling."),
        ("Precision, F1", "matched / emitted, and their harmonic mean",
         "lower bound", cite('bekker2020', 'chicco2020'),
         "An unmatched pick may be an arrival the analyst never marked. Named <code>_lb</code> here; comparable between models on the same reference, not with a labelled-dataset number."),
        ("MCC", "Matthews correlation coefficient", "not computable", cite('chicco2020'),
         "Needs true negatives, which a continuous record with an incomplete reference does not define."),
        ("MAE, RMSE", "mean absolute and root-mean-square residual", "exact", cite('munchmeyer2022'),
         "Report both: RMSE responds to outliers, MAE does not."),
        ("MedianAE, median bias", "robust scatter, and systematic earliness or lateness",
         "exact", cite('munchmeyer2022', 'pita2023'),
         "Separates a picker that is consistently late from one that is noisy."),
        ("Gross-error rate", f"fraction of wide-matched residuals beyond {DETECT_TOL:g}&thinsp;s",
         "exact", cite('munchmeyer2022'),
         "Must be computed from residuals matched wider than the detection tolerance, or it describes the tolerance."),
        ("Fraction within 0.1&thinsp;s", "share of matched picks inside 0.1&thinsp;s", "exact",
         cite('zhu2019'), "The tolerance the original PhaseNet paper scored at; comparable to older literature."),
        ("Reliability curve, ECE", "observed agreement per confidence bin, and its mean gap",
         "lower bound", cite('guo2017'), "Tells you whether a threshold transfers between models. It usually does not."),
        ("Phase swap rate", "arrivals matched by a pick of the other phase", "exact",
         cite('munchmeyer2022'), "Survives association and moves a location."),
        ("Duplicate rate", "extra picks within tolerance of an already-matched arrival",
         "exact", cite('zhu2022', 'munchmeyer2024'), "Invisible in recall; costs an associator work."),
        ("Catalogue completeness", "magnitude above which the catalogue is complete",
         "downstream", cite('woessner2005'),
         "The quantity a catalogue user cares about. Not yet on this board; it needs association and location, not picks alone."),
    ]
    A("""
<section id="metrics">
  <div class="section-head">
    <p class="eyebrow">What the board measures</p>
    <h2>Every metric, what it is for, and whether a bulletin lets you compute it</h2>
    <p class="lede">The column that matters to anyone building an evaluation is the third one. A
    metric that is standard in machine learning is not automatically available here, and saying
    which are and which are not is most of the methodological work.</p>
  </div>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">metric</th><th class="l">definition</th><th class="l">against a bulletin</th>
    <th class="l">source</th><th class="l">why it earns a column</th></tr></thead><tbody>""")
    for name, defn, status, src, why in ident:
        cls = {"exact": "exact", "lower bound": "bound", "not computable": "unmet",
               "downstream": ""}[status]
        A(f'    <tr><td class="l"><strong>{name}</strong></td><td class="l">{defn}</td>'
          f'<td class="l"><span class="pill {cls}">{status}</span></td><td class="l">{src}</td>'
          f'<td class="l" style="color:#6f6890">{why}</td></tr>')
    A("""  </tbody></table></div>
</section>
""")

    # ---------------------------------------------------------------- standard
    rules = [
        ("R1", "The task is fully specified before submissions open", "partial",
         "The task, matching rule, tolerances and metrics are frozen in code and published. There is no submission process, so nobody can yet submit to a specification they did not write."),
        ("R2", "The test set is hidden, and the board says so on every row", "unmet",
         "Every reference here is a public operator bulletin. This board cannot distinguish generalisation from familiarity with a well-studied sequence."),
        ("R3", "The scorer is public, deterministic and versioned", "met",
         "<code>scripts/score_picks.py</code> and <code>sb_catalog/src/benchmark_metrics.py</code>, pinned by " + str(n_tests) + " unit tests, byte-identical output across processes, versioned in git."),
        ("R4", "A trivial baseline is published first", "unmet",
         "No STA/LTA floor is published alongside these numbers, so the recall column has no zero point."),
        ("R5", "A strong published baseline is published alongside", "met",
         "Three of the four weight sets are published models from other groups; the fourth is ours."),
        ("R6", "Contamination is addressed explicitly, in writing, per task", "partial",
         "Stated per sequence and not quantified: Norcia is in-domain for <code>instance</code>, and the <code>quakescope2026</code> fine-tuning corpus includes INSTANCE and Pacific Northwest data."),
        ("R7", "Splits are DOI-archived with a datasheet", "unmet",
         "The reference arrivals are harvested live from agency services, so a bulletin revision changes the board with no record."),
        ("R8", "The evaluation is separable from the group whose models it scores", "partial",
         "SeisSCOPED maintains the benchmark, and one of the four weight sets is ours. It does not win: <code>quakescope2026</code> ranks "
         f"{int(pbest.loc['quakescope2026', 'rank'])} of 4 under protocol C and {int(pbud.loc['quakescope2026', 'rank'])} of 4 under protocol B."),
        ("R9", "Every row carries model version, split, date and cost", "partial",
         "Weight set, SeisBench version, notebook and execution timestamp are stamped in the footer. Cost per processed station-day is measured for the campaign but is not yet on this board."),
    ]
    n_met = sum(1 for r in rules if r[2] == "met")
    n_part = sum(1 for r in rules if r[2] == "partial")
    n_unmet = sum(1 for r in rules if r[2] == "unmet")
    assert n_met + n_part + n_unmet == len(rules)
    A(f"""
<section id="standard">
  <div class="section-head">
    <p class="eyebrow">The standard</p>
    <h2>Nine rules for a citable benchmark, and where this track stands</h2>
    <p class="lede">Adapted from the questions reviewers ask on the NeurIPS and ICML
    <i>Datasets and Benchmarks</i> track, and recorded on the
    <a href="https://gaia-hazlab.github.io/hazevalhub">HazEvalHub</a> hub page for every
    evaluation the project runs. The right-hand column is a scorecard, not an aspiration:
    of {len(rules)} rules, {n_met} met, {n_part} partly met, {n_unmet} not met.</p>
  </div>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">#</th><th class="l">rule</th><th class="l">status</th>
    <th class="l">where this board stands</th></tr></thead><tbody>""")
    for num, rule, status, note in rules:
        label = {"met": "met", "partial": "partly", "unmet": "not met"}[status]
        A(f'    <tr><td class="l"><strong>{num}</strong></td><td class="l">{rule}</td>'
          f'<td class="l"><span class="pill {status}">{label}</span></td>'
          f'<td class="l" style="color:#6f6890">{note}</td></tr>')
    A("""  </tbody></table></div>
  <div class="callout">
    <h3>The one that matters most</h3>
    <p>R2. Until there is a reference set drawn from after the training window of every model it
    scores, and labelled independently, this board measures skill and familiarity together and
    cannot separate them. That is a labelling campaign, not a software task, and it is the
    critical path for turning this page into a benchmark a paper could cite.</p>
  </div>
</section>
""")

    # ---------------------------------------------------------------- run it
    A(f"""
<section id="run">
  <div class="section-head">
    <p class="eyebrow">Score your own picks</p>
    <h2>One script, two CSV files</h2>
    <p class="lede">The scorer behind every number above is public and takes anybody's picks. It
    imports nothing but <code>numpy</code> and <code>pandas</code>.</p>
  </div>
  <div class="card">
<pre><code># the arrivals you trust:   station,phase,time
# what your model emitted:  station,phase,time,conf
python scripts/score_picks.py --reference arrivals.csv --picks mypicks.csv --out scores/

# see it work on synthetic picks with known properties, no data needed
python scripts/score_picks.py --demo</code></pre>
    <p style="margin-top:14px">Run your model <strong>once at a low confidence floor</strong> and
    keep every pick, so each threshold above is a filter over one file rather than another pass
    over the waveforms. Optional <code>dataset</code> and <code>model</code> columns score several
    sequences or models in one run; the column names our exports and SeisBench already use
    (<code>sequence</code>, <code>weights</code>, <code>probability</code>,
    <code>pick_time</code>) are accepted without renaming. Output is
    <code>detection.csv</code>, <code>timing.csv</code>, <code>calibration.csv</code>,
    <code>phase_quality.csv</code> and <code>sweep.csv</code>.</p>
    <p>If your reference marks <em>every</em> arrival, pass <code>--exhaustive</code> and
    precision and F1 are reported under those names instead of as bounds.</p>
  </div>
  <div class="grid-2" style="margin-top:18px">
    <div class="card">
      <h3>Reproducing this board</h3>
      <p>The <a href="benchmark_metrics_methods.html">methods notebook</a> computes every table
      here from the exported picks and checks, on every row of both
      sequence studies, that the standalone scorer reproduces it bit for bit. Source and data:
      <a href="https://github.com/SeisSCOPED/QuakeScope">SeisSCOPED/QuakeScope</a>.</p>
    </div>
    <div class="card">
      <h3>What would move a rank</h3>
      <p>A held-out sequence nobody has looked at; an STA/LTA baseline for the recall floor;
      association and location, so completeness{cite('woessner2005')} can replace recall as the
      headline; and cost per station-day on every row.</p>
    </div>
  </div>
</section>

<section id="limits">
  <div class="section-head">
    <p class="eyebrow">Limits</p>
    <h2>What this board cannot claim</h2>
  </div>
  <ul>
    <li>Sample sizes differ by an order of magnitude between sequences, from
    {int(d.n_ref.min())} to {int(d.n_ref.max())} reference arrivals. The board weights by
    arrivals; a per-sequence reading is in the methods notebook.</li>
    <li>Precision and F1 are bounds, so a model that finds real arrivals the analyst skipped is
    penalised exactly like one that hallucinates. Only association can separate those two, and
    no association step runs here.</li>
    <li>Protocol B is evaluated at a budget that <code>{bind_model}</code>'s ceiling bounds in
    all {bind_tot} rows where every ceiling is measured. Protocol C has no held-out split, so it is an upper bound on a tuned deployment.</li>
    <li>Contamination is stated, not measured. Two of the four weight sets have plausible
    exposure to data from the regions scored here.</li>
    <li>Model runtime differs measurably between these weight sets and is not on the board. A
    model that cannot be run across a fifteen-year archive cannot be deployed, whatever its
    recall.</li>
  </ul>
</section>

<section id="refs">
  <div class="section-head">
    <p class="eyebrow">References</p>
    <h2>Sources</h2>
    <p class="lede">Every entry was checked against Crossref or arXiv at build time.</p>
  </div>
  <ol class="refs">""")
    for i, (key, text, url) in enumerate(REFS, start=1):
        A(f'    <li id="r{i}">{text} <a href="{url}">{url.replace("https://doi.org/", "doi:").replace("https://arxiv.org/abs/", "arXiv:")}</a></li>')
    A("""  </ol>
</section>
</main>

<footer class="site-footer">
  <div class="shell">
    <p><strong>QuakeScope picker board</strong> &middot; the catalogue-workflow track of
    <a href="https://gaia-hazlab.github.io/hazevalhub">HazEvalHub</a> &middot;
    maintained by <a href="https://github.com/SeisSCOPED/QuakeScope">SeisSCOPED</a> &middot;
    <a href="./">all QuakeScope reports</a></p>
    <p class="provenance">""")
    for study in sorted(meta):
        m = meta[study]
        A(f'      {study}: <code>{m.get("notebook", "?")}</code>, executed '
          f'{m.get("executed", "?")}, seisbench {m.get("seisbench", "?")}<br>')
    A(f"""      page built {built} from <code>{commit}</code> by
      <code>scripts/build_leaderboard.py</code>; numbers read from
      <code>docs/benchmark/results/</code>, never typed.</p>
  </div>
</footer>
</body>
</html>""")

    OUT.write_text("\n".join(H))
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  {seq_phases} sequence-phases, {arrivals:,} arrivals, {len(REFS)} references")
    print(f"  protocol A leader {lead03} ({p03.recall.max():.3f}), "
          f"B leader {leadbud} ({pbud.recall.max():.3f}), "
          f"C spread {spreadbest:.3f}")
    print(f"  {bind_model} bounds the budget in {bind_n}/{bind_tot} rows; "
          f"leader flips in {flips}/{seq_phases}")


if __name__ == "__main__":
    main()
