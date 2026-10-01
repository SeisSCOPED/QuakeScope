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
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sb_catalog.src.benchmark_metrics import ece, match_picks, reliability  # noqa: E402

RES = ROOT / "docs" / "benchmark" / "results"
OUT = ROOT / "reports" / "benchmark_metrics.html"
# The figures are set in the same face as the page. The file is vendored under
# reports/fonts/ (SIL Open Font License) so the build does not depend on what
# happens to be installed.
FONT = ROOT / "reports" / "fonts" / "Manrope[wght].ttf"


def _use_page_font() -> None:
    """Set the figures in Manrope, the face the page uses.

    The vendored file is a variable font whose default axis position is wght
    200, so registering it as it stands gives ExtraLight axis labels. Instance
    it at Medium and Bold instead, into a temporary directory, so no extra
    binaries sit in the repository and the result is derived from the one file
    that is versioned.
    """
    if not FONT.exists():
        print(f"warning: {FONT.name} missing, figures keep the default face")
        return
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
    except ModuleNotFoundError:
        print("warning: fonttools missing, figures keep the default face")
        return
    tmp = Path(tempfile.mkdtemp(prefix="board-fonts-"))
    # A distinct family name, because a variable Manrope registered anywhere in
    # matplotlib's font cache also answers to "Manrope" and would win at its
    # default axis position, which is ExtraLight.
    # One instance, at Medium, declared as the family's regular face. A single
    # font in the family leaves matplotlib nothing to mis-resolve, and no figure
    # text here is set bold.
    family = "Manrope Board"
    inst = instancer.instantiateVariableFont(TTFont(str(FONT)), {"wght": 500},
                                             updateFontNames=True)
    name = inst["name"]
    # Drop the typographic family and subfamily records: they say "Manrope" and
    # take precedence over name ID 1 in the parser matplotlib reads through.
    for nid in (16, 17, 21, 22):
        name.removeNames(nameID=nid)
    for nid, value in ((1, family), (2, "Regular"), (4, f"{family} Regular"),
                       (6, "ManropeBoard-Regular")):
        name.setName(value, nid, 3, 1, 0x409)
        name.setName(value, nid, 1, 0, 0)
    inst["OS/2"].usWeightClass = 400
    out = tmp / "ManropeBoard-Regular.ttf"
    inst.save(str(out))
    font_manager.fontManager.addfont(str(out))
    plt.rcParams["font.family"] = family
    got = Path(font_manager.findfont(font_manager.FontProperties(family=family))).name
    if got != out.name:
        raise RuntimeError(f"figures would not use the page face: matplotlib resolved "
                           f"{family!r} to {got}, not {out.name}")


_use_page_font()

WEIGHTS = ["quakescope2026", "jma_wc", "original", "instance"]
WCOLOR = {"quakescope2026": "#4b2e83", "jma_wc": "#c2571a",
          "original": "#1b7f79", "instance": "#2f6fb2"}
INK, STONE, LINE, LAV2 = "#2a1a4f", "#6f6890", "#d8d2e8", "#ece8f7"
SHARED_THR, DETECT_TOL = 0.3, 0.5
DETECT_FLOOR = 0.02        # the confidence floor the exported inference runs were kept at

# The two benchmark tracks. They answer different questions, and the ranking
# differs between them, so the board reports them separately.
TRACKS = {
    "us": dict(
        title="Track 1: western United States",
        lede="The region the QuakeScope campaign catalogues. Mainshock-aftershock "
             "sequences chosen to vary network, magnitude and era, with analyst arrivals "
             "pulled from ANSS through the SCEDC and NCEDC event services. This track asks "
             "whether a weight set serves the catalogue we are building.",
        studies=("us_sequences", "ridgecrest_aftershocks", "western_reproduction")),
    "global": dict(
        title="Track 2: outside the United States",
        lede="Sequences on other networks, none of them in the curated corpora the weight "
             "sets were trained on, each with the operator's own reviewed arrivals. This "
             "track asks which scientific use cases a weight set serves when the region is "
             "not its own.",
        studies=("global_sequences",)),
}

# Sequence settings as the two benchmark notebooks record them. Descriptions, not
# measurements: every number on the board is read from the result tables.
SEQ_SETTING = {
    "Ridgecrest": ("2019-07-06", "7.1", "Eastern California, aftershocks seconds apart"),
    "San Simeon": ("2003-12-22", "6.5", "Central Coast, 2003 network and instrumentation"),
    "Monte Cristo": ("2020-05-15", "6.5", "Nevada, Basin and Range, different network"),
    "Mendocino 2024": ("2024-12-05", "7.0", "Offshore, one-sided geometry, every station 55 km or more"),
    "Monroe WA": ("2019-07-12", "4.6", "Cascadia, moderate magnitude"),
    "Kaikoura 2016": ("2016-11-13", "7.8", "New Zealand, GeoNet. Sparse permanent network, most reviewed picks 80 to 120 km out"),
    "Norcia 2016": ("2016-10-30", "6.5", "Central Italy, INGV. Apennine normal faulting, permanent plus post-Amatrice temporary stations"),
    "Thessaly 2021": ("2021-03-03", "6.3", "Central Greece, NOA. Normal-faulting doublet, a station 5 km from the epicentre"),
}

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
    ("aguilar2025", "Aguilar Suarez, A. L. and Beroza, G. C. (2025). Picking regional seismic "
     "phase arrival times with deep learning. <i>Seismica</i> 4(1).",
     "https://doi.org/10.26443/seismica.v4i1.1431"),
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
    ("park2025", "Park, Y., Armstrong, A. D., Yeck, W. L., Shelly, D. R. and Beroza, G. C. "
     "(2025). Divide and conquer: separating the two probabilities in seismic phase picking. "
     "<i>Geophysical Journal International</i> 243, ggaf333.",
     "https://doi.org/10.1093/gji/ggaf333"),
    ("yuan2023", "Yuan, C., Denolle, M. A., Ni, Y., Chen, Y. and Zhu, W. (2023). Better "
     "together: ensemble learning for earthquake detection and phase picking. <i>IEEE "
     "Transactions on Geoscience and Remote Sensing</i> 61, 5914213.",
     "https://doi.org/10.1109/TGRS.2023.3320148"),
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
# Published for download next to the board. The copy under reports/data/ is made
# from docs/benchmark/results/ on every build, so the files served are the files
# the numbers were computed from.
DATA_FILES = [
    ("us_sequences/model_picks.csv",
     "Every pick the four weight sets emitted on the western US sequences, with confidences"),
    ("us_sequences/reference_picks.csv", "The analyst arrivals those were scored against"),
    ("global_sequences/model_picks.csv",
     "The same, for the sequences outside the United States"),
    ("global_sequences/reference_picks.csv", "The operators' reviewed arrivals"),
    ("detection_full.csv", "Recall, precision and F1 bounds per sequence, phase and weight set"),
    ("timing_full.csv", "Onset-time statistics per sequence, phase and weight set"),
    ("calibration_full.csv", "Expected calibration error per study and weight set"),
    ("phase_quality_full.csv", "Phase swaps and duplicate picks per sequence"),
    ("summary_matched_budget.csv", "Recall read at an equal pick count"),
]


def publish_data() -> None:
    """Copy the result tables the board reads into the published directory."""
    dest = ROOT / "reports" / "data"
    for rel, _ in DATA_FILES:
        src = RES / rel
        if not src.exists():
            print(f"  warning: {rel} missing, not published")
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(src.read_bytes())


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


def track_table(d: pd.DataFrame, study: str) -> pd.DataFrame:
    """One track's ranking, ordered by recall at each weight set's own threshold."""
    s = d[d.study == study]
    keys = ["sequence", "phase"]
    rows = []
    wins = s.loc[s.groupby(keys).recall_at_best.idxmax()].weights.value_counts()
    for m in WEIGHTS:
        g = s[s.weights == m]
        rows.append({"weights": m, "own": wmean(g, "recall_at_best"),
                     "shared": wmean(g, "recall_at_03"),
                     "equal": wmean(g, "recall_at_budget"),
                     "thr": float(g.best_thr.mean()),
                     "wins": int(wins.get(m, 0)), "rows": len(g)})
    out = pd.DataFrame(rows).sort_values("own", ascending=False).reset_index(drop=True)
    out["rank"] = out.own.rank(ascending=False, method="min").astype(int)
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
    labels = ["One shared\nthreshold, 0.3", "Equal\npick count", "Each model's\nown threshold"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(order))
    ys = {m: [prot[(prot.weights == m) & (prot.col == c)].recall.iloc[0] for c in order]
          for m in WEIGHTS}
    for m in WEIGHTS:
        ax.plot(x, ys[m], "-o", color=WCOLOR[m], lw=2.2, ms=7, label=m, zorder=3)
        ax.annotate(f"{ys[m][0]:.3f}", (x[0], ys[m][0]), textcoords="offset points",
                    xytext=(-10, 0), ha="right", va="center", fontsize=8.5, color=WCOLOR[m])

    # Protocol C bunches the four within a few points, so the right-hand labels
    # would overprint. Lay them out at a fixed spacing around their own mean.
    span = max(max(v) for v in ys.values()) - min(min(v) for v in ys.values())
    gap = 0.035 * span
    right = sorted(WEIGHTS, key=lambda m: ys[m][-1], reverse=True)
    place = [ys[m][-1] for m in right]
    for i in range(1, len(place)):
        place[i] = min(place[i], place[i - 1] - gap)
    for m, yt in zip(right, place):
        ax.annotate(f"{ys[m][-1]:.3f}", (x[-1], ys[m][-1]), xytext=(x[-1] + 0.10, yt),
                    ha="left", va="center", fontsize=8.5, color=WCOLOR[m],
                    arrowprops=dict(arrowstyle="-", color=WCOLOR[m], lw=0.7,
                                    shrinkA=2, shrinkB=2, alpha=0.55))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, 2.75)
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
    handles = [plt.Line2D([], [], color=WCOLOR[m], lw=2.0, marker="o", ms=5,
                          mec="white", mew=1.0, label=m) for m in WEIGHTS]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK,
               loc="lower center", ncol=len(WEIGHTS), bbox_to_anchor=(0.5, -0.035))
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
        ax.set_title(TRACKS[study.replace("_sequences", "")]["title"], fontsize=10)
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
    publish_data()
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

    # per-track boards, and the claim that separating them changes the reading
    tracks = {k: track_table(d, k) for k in TRACKS}
    us_lead = tracks["us"].weights.iloc[0]
    gl_lead = tracks["global"].weights.iloc[0]
    ours = "quakescope2026"
    us_ours = int(tracks["us"].set_index("weights").loc[ours, "rank"])
    gl_ours = int(tracks["global"].set_index("weights").loc[ours, "rank"])
    assert us_ours < gl_ours, ("the prose says our own fine-tune ranks higher on its own "
                               "region than abroad")
    track_arrivals = {k: int(d[d.study == k].groupby(["sequence", "phase"]).n_ref.first().sum())
                      for k in TRACKS}
    track_rows = {k: d[d.study == k].groupby(["sequence", "phase"]).ngroups for k in TRACKS}
    # sequences that were picked but carry no manual arrivals, so they cannot be scored
    all_seq = set(pd.read_csv(RES / "us_sequences" / "model_picks.csv", usecols=["sequence"]).sequence)
    unscorable = sorted(all_seq - set(d[d.study == "us"].sequence))
    missing_note = ("" if not unscorable else
                    (", ".join(unscorable) + (" was" if len(unscorable) == 1 else " were") +
                     " picked as well, and the operator publishes no manual arrivals for "
                     + ("it" if len(unscorable) == 1 else "them") + ", so "
                     + ("it cannot" if len(unscorable) == 1 else "they cannot") + " be scored."))
    tim_track_raw = (t[t.medae.notna()].merge(
        d[["study", "sequence", "phase", "weights", "n_ref"]],
        on=["study", "sequence", "phase", "weights"])
        .groupby(["study", "weights"])
        .apply(lambda x: float(np.average(x.medae, weights=x.n)), include_groups=False)
        .unstack(0))
    tim_track = tim_track_raw
    # on track 2 the recall leader is also the least accurate in onset time
    gl_time_worst = tim_track["global"].idxmax()
    gl_time_best = tim_track["global"].idxmin()
    assert gl_time_worst == gl_lead, ("the callout says the out-of-region recall leader is the "
                                      "least accurate on onset time")
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
<meta name="description" content="A public leaderboard for phase picking and association for P and S waves from seismic waveform data, with a United States benchmark track and a global track, and the data and code to reproduce every number.">
<title>Seismic Phase Picking Leaderboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="quakescope-board.css">
</head>
<body>

<header class="hero">
  <div class="shell">
    <nav class="nav" aria-label="Primary">
      <a class="brand" href="./"><span class="brand-mark">QS</span>
        <span>QuakeScope <span style="opacity:.7;font-weight:600">picker board</span></span></a>
      <div class="nav-links">
        <a href="#board">The board</a><a href="#tracks">Tracks</a>
        <a href="#protocols">Thresholds</a><a href="#examples">Examples</a>
        <a href="#metrics">Metrics</a><a href="#data">Data</a>
        <a href="#standard">Standard</a><a href="#run">Score your picks</a>
      </div>
    </nav>
    <div class="hero-inner">
      <p class="eyebrow">HazEvalHub &middot; catalogue-workflow track &middot; SeisSCOPED</p>
      <h1>Seismic Phase Picking Leaderboard</h1>
      <p>A public leaderboard for phase picking and association for P and S waves from seismic
      waveform data. The evaluation gathers the metrics the community has used to judge a
      picker. The benchmark runs on two tracks of data sets: a United States benchmark set, and
      a global benchmark set covering several types of earthquake sequence.</p>
      <p>Every pick on both sides of every comparison downloads from
      <a href="#data">the data section</a>, with the code to fetch the matching waveforms and
      plot them.</p>
      <div class="hero-actions">
        <a class="button primary" href="#board">See the board</a>
        <a class="button secondary" href="#data">Download the data</a>
        <a class="button secondary" href="benchmark_data.html">Code to fetch waveforms</a>
      </div>
    </div>
  </div>
</header>

<main class="shell">
<div class="stats">
  <div class="stat"><div class="n">4</div><div class="k">published weight sets scored</div></div>
  <div class="stat"><div class="n">{seq_phases}</div><div class="k">sequence-phases</div></div>
  <div class="stat"><div class="n">{arrivals:,}</div><div class="k">analyst arrivals as reference</div></div>
  <div class="stat"><div class="n">{n_studies}</div><div class="k">studies feeding the tracks</div></div>
  <div class="stat"><div class="n">2</div><div class="k">tracks, with different winners</div></div>
</div>
""")

    # ---------------------------------------------------------------- board
    A(f"""
<section id="board">
  <div class="section-head">
    <p class="eyebrow">The board</p>
    <h2>Recall per track at each weight set's own threshold</h2>
    <p class="lede">Recall is the fraction of analyst arrivals recovered within
    {DETECT_TOL:g}&thinsp;s on the same station and phase, weighted by the number of reference
    arrivals, so a sequence with {int(d.n_ref.max())} arrivals counts more than one with
    {int(d.n_ref.min())}. Each weight set is read at the threshold that maximises its own score
    on the track, because a confidence of 0.3 from one model and 0.3 from another do not put
    two models at the same operating point. The columns to the right give the same weight sets
    under the other two protocols, both defined below the tables.</p>
  </div>
""")
    for study, spec in TRACKS.items():
        tt = tracks[study]
        A(f'  <div class="card"><h3>{spec["title"]}</h3>')
        A(f'  <p style="color:#6f6890;font-size:.94rem">{spec["lede"]}</p>')
        A(f'  <p style="color:#6f6890;font-size:.88rem">{track_rows[study]} sequence-phases, '
          f'{track_arrivals[study]:,} reference arrivals.</p>')
        A('  <div class="table-scroll"><table class="board"><thead><tr>'
          '<th class="l">#</th><th class="l">weight set</th>'
          '<th>recall at its own threshold</th><th>threshold used</th>'
          '<th>sequence-phases led</th><th>recall at a shared 0.3</th>'
          '<th>recall at an equal pick count</th><th>median onset error (s)</th>'
          '</tr></thead><tbody>')
        for _, r in tt.iterrows():
            lead = ' class="lead"' if r["rank"] == 1 else ""
            A(f'    <tr><td class="l">{rank_badge(int(r["rank"]))}</td>'
              f'<td class="l">{chip(r.weights)}</td>'
              f'<td{lead}><strong>{r.own:.3f}</strong></td><td>{r.thr:.2f}</td>'
              f'<td>{r.wins} / {int(r.rows)}</td><td>{r.shared:.3f}</td>'
              f'<td>{r.equal:.3f}</td>'
              f'<td>{tim_track.loc[r.weights, study]:.3f}</td></tr>')
        A('  </tbody></table></div></div>')

    # ---------------------------------------------------------------- tracks
    A(f"""
<section id="tracks">
  <div class="section-head">
    <p class="eyebrow">What is scored</p>
    <h2>Two tracks answer two different questions</h2>
    <p class="lede">A picker is chosen for a purpose. One purpose is a regional catalogue, where
    the network, the instrumentation and the analyst conventions are the ones the picker will
    meet every day. The other is a sequence somewhere the picker has never been trained, where
    the question is which scientific use cases it still serves. Those are separate tests and
    they return different rankings.</p>
  </div>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">track</th><th class="l">sequence</th><th class="l">date</th>
    <th>M</th><th class="l">setting</th><th>P arrivals</th><th>S arrivals</th></tr></thead>
    <tbody>""")
    for study, spec in TRACKS.items():
        seqs = [s for s in d[d.study == study].sequence.unique()]
        for k, seq in enumerate(sorted(seqs)):
            date, mag, setting = SEQ_SETTING.get(seq, ("", "", ""))
            nref = d[(d.study == study) & (d.sequence == seq)].groupby("phase").n_ref.first()
            A(f'      <tr><td class="l">{spec["title"].split(":")[0] if k == 0 else ""}</td>'
              f'<td class="l"><strong>{seq}</strong></td><td class="l">{date}</td>'
              f'<td>{mag}</td><td class="l" style="color:#6f6890;font-size:.85rem">{setting}</td>'
              f'<td>{int(nref.get("P", 0))}</td><td>{int(nref.get("S", 0))}</td></tr>')
    A(f"""    </tbody>
    <caption>Reference arrivals are manual picks only, from ANSS through the SCEDC and NCEDC
    event services on track 1 and from the operating network's own event service on track 2.
    Both tracks score the aftershock window rather than the mainshock, which is where a
    catalogue is made and lost. Two further studies feed the tracks without appearing in this
    table: a 30-minute dense-aftershock window at Ridgecrest, and a reproduction of
    {int(meta['western_reproduction']['totals']['campaign_picks']):,} campaign picks through a
    second data path. {missing_note}</caption>
  </table></div>
  <div class="grid-2" style="margin-top:18px">
    <div class="card">
      <h3>Why these sequences</h3>
      <p>None of them is in the curated corpora the weight sets were trained on. They were
      selected to vary the network, the magnitude and the era of the recording, and to put the
      pickers on the sequence types a catalogue has to handle: a mainshock-aftershock cascade
      with events seconds apart, a doublet, a 2003 network with 2003 instrumentation, and an
      offshore geometry where every station is more than 55 km away.</p>
    </div>
    <div class="card">
      <h3>What the reference is and is not</h3>
      <p>An analyst picked what the location needed and stopped. Recall against that reference
      is exact. A model pick with no analyst counterpart may be a false positive or a real
      arrival nobody marked, so precision and everything derived from it is a bound, marked
      <code>_lb</code> throughout{cite('bekker2020')}.</p>
    </div>
  </div>
</section>
""")

    A(f"""
  <p style="color:#6f6890;font-size:.93rem;max-width:88ch">
    <code>{ours}</code>, the fine-tune this project trained, ranks {us_ours} of {len(WEIGHTS)}
    in the western United States and {gl_ours} of {len(WEIGHTS)} outside it, so the track
    matters more than the pooled average: track 2 carries {track_arrivals['global']:,} of the
    {arrivals:,} reference arrivals and would set a single number on its own. On track 2 the
    recall leader <code>{gl_lead}</code> is also the least accurate on onset time,
    {tim_track['global'][gl_time_worst]:.3f}&thinsp;s median error against
    {tim_track['global'][gl_time_best]:.3f}&thinsp;s for <code>{gl_time_best}</code>.</p>
</section>

<section id="protocols">
  <div class="section-head">
    <p class="eyebrow">The board</p>
    <h2>Three ways to set the threshold give three orderings</h2>
    <p class="lede">A picker emits a pick when its confidence passes a threshold, and the four
    weight sets put their confidences on different scales. How that threshold is set therefore
    decides the ranking. All three settings are reported here, pooled across both tracks.</p>
  </div>
""")
    for col, name, note in (
        ("recall_at_03", f"Protocol A: one threshold for every model, {SHARED_THR}",
         "Every model read at confidence &ge; 0.3, which is what most published comparisons "
         "report. It measures how willing a model is to emit a pick as much as how well it "
         "picks."),
        ("recall_at_budget", "Protocol B: every model emitting the same number of picks",
         f"Each model's threshold is moved until it emits the same number of picks as the "
         f"others, and recall is read there. No threshold is assumed, but the count all four "
         f"can reach is capped by the most conservative model. <code>{bind_model}</code> sets "
         f"that cap in every one of the {bind_tot} sequence-phases where all four limits are "
         f"measured, so the comparison sits at the low-pick-count end of every curve, where a "
         f"conservative model looks best."),
        ("recall_at_best", "Protocol C: each model at its own best threshold",
         "The threshold that maximises each model's own score on this reference, as the "
         "cross-domain benchmark does on a development set. There is no held-out split here, "
         "so the number is an upper bound on what a tuned deployment reaches.")):
        sub = prot[prot.col == col].sort_values("recall", ascending=False)
        lastcol = {"recall_at_03": "mean picks emitted",
                   "recall_at_budget": "most picks it can emit",
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
    <figcaption>Each line is one weight set. The left column is what a single shared threshold
    publishes. The middle holds every model to the same number of picks, a count the most
    conservative model caps. The right is what each model reaches on its own threshold.
    <strong>{lead03}</strong> leads protocol A and comes last in protocol B, and
    <strong>{leadbud}</strong> does the reverse. Per sequence-phase the leader changes between A
    and B in {flips} of {seq_phases} rows.</figcaption>
  </figure>

  <div class="callout warn">
    <h3>Most of the spread is the protocol</h3>
    <p>The spread between best and worst weight set is {spread03:.3f} recall under protocol A
    and {spreadbest:.3f} under protocol C, so {100 * (1 - spreadbest / spread03):.0f}&thinsp;% of
    the apparent difference between these four models comes from holding the threshold fixed
    rather than from how they pick. Under protocol C the win counts are
    {' / '.join(str(int(pbest.loc[m, 'wins'])) for m in pbest.sort_values('recall', ascending=False).index)}
    across {seq_phases} rows, which is not a ranking. Detection is the axis on which these four
    weight sets are hardest to separate. The three axes below separate them.</p>
  </div>

  <figure>{fig2}
    <figcaption>Recall against picks emitted, six of the {seq_phases} sequence-phases, log x.
    Filled circles mark each model's operating point at a shared 0.3, the dotted line the
    equal pick count. Where a curve stops, the model has run out of picks with its threshold on
    the floor. That is a ceiling rather than a calibration offset, and no threshold recovers
    it.</figcaption>
  </figure>
</section>
""")

    # ---------------------------------------------------------------- timing
    A(f"""
<section id="timing">
  <div class="section-head">
    <p class="eyebrow">The board</p>
    <h2>Onset time, calibration and phase identification</h2>
    <p class="lede">A bulletin leaves all three of these identifiable. They are also what a
    location and a magnitude consume, and they separate the four weight sets where detection
    does not.</p>
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
    is insensitive to outliers and the other is not{cite('munchmeyer2022')}. The 0.1&thinsp;s
    column is the tolerance PhaseNet was originally scored at{cite('zhu2019')}.
    <code>{best_time_p}</code> is most accurate on P and <code>{best_time_s}</code> on S.</caption>
  </table></div></div>

  <div class="card"><h3>Confidence calibration</h3>
  <p>Every downstream user thresholds on <code>conf</code>, and an associator weights by
  it{cite('zhu2022','munchmeyer2024')}. Expected calibration error is the mean gap between
  stated confidence and observed agreement, weighted by bin count{cite('guo2017')}. Against a
  bulletin the observed rate is a lower bound, so the quantity here is <code>ece_lb</code>. The
  shape of the curve is what makes one threshold mean different things to different weight
  sets.</p>
  <p>There is a reason beyond training for the gap. A segmentation picker is trained against a
  kernel placed on the labelled arrival time, and the height of the output peak it is read at
  is neither the probability that a phase exists nor the probability attached to the arrival
  time{cite('park2025')}. The curves below measure a quantity that the architecture does not
  define as a probability in the first place.</p>
  <figure>{fig3}
    <figcaption>Perfect calibration is the dashed diagonal. Every weight set sits below it: a
    pick labelled 0.8 agrees with the bulletin less often than 80&thinsp;% of the time, partly
    because the bulletin is incomplete and partly because the models are overconfident.
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
  association and moves a location. Both sides carry a phase label, so this is identifiable
  where precision is not.</p>
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
""")
    A(f"""
  <div class="card"><h3>Inference cost</h3>
  <p>A picker that cannot be run across the archive cannot build the catalogue, so cost belongs
  on the board next to recall. It is a scoring axis here with a fixed protocol and no numbers
  yet, which is why rule R9 below is only partly met.</p>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">quantity</th><th class="l">unit</th><th class="l">how to report it</th>
    <th class="l">status</th></tr></thead><tbody>
      <tr><td class="l"><strong>Model time</strong></td><td class="l">s per station-day</td>
      <td class="l">Wall clock for the forward pass alone, at a stated batch size and window
      overlap, excluding data read. Architecture drives this: the doubled filter width of
      <code>jma_wc</code> and its fine-tune costs more per window than
      <code>original</code>.</td>
      <td class="l"><span class="pill unmet">not measured here</span></td></tr>
      <tr><td class="l"><strong>End-to-end time</strong></td><td class="l">s per station-day</td>
      <td class="l">Wall clock including the archive read, which dominates at campaign scale
      and depends on the store rather than the model.</td>
      <td class="l"><span class="pill unmet">not measured here</span></td></tr>
      <tr><td class="l"><strong>Peak memory</strong></td><td class="l">MB resident</td>
      <td class="l">Peak resident set for one worker at the stated batch size. It sets the task
      size, and a model that does not fit the smallest task is more expensive than its
      per-window time suggests.</td>
      <td class="l"><span class="pill unmet">not measured here</span></td></tr>
      <tr><td class="l"><strong>Cost</strong></td><td class="l">USD per 1,000 station-days</td>
      <td class="l">vCPU-hours times the stated instance price, summed over attempts rather than
      jobs, so retries and preemptions are counted. The QuakeScope campaign measures this per
      campaign and not per weight set.</td>
      <td class="l"><span class="pill partial">campaign only</span></td></tr>
  </tbody>
  <caption>Reporting recall without cost ranks a model nobody can afford to run first. The four
  quantities are separated because they have different causes: model time follows the
  architecture, end-to-end time follows the data path, memory sets the task size, and cost
  follows the platform.</caption>
  </table></div></div>

  <div class="card"><h3>Pick uncertainty comes next</h3>
  <p>Every number on this board treats a pick as a time and a scalar confidence. Two lines of
  work make that assumption avoidable, and neither has a column here yet.</p>
  <p>Ensemble pickers report the spread across models as well as the pick, so a disagreement
  between architectures becomes a usable uncertainty rather than a hidden
  one{cite('yuan2023')}. Separately, the detection probability and the arrival-time
  probability can be estimated as the distinct quantities they are, instead of being read off
  one output peak{cite('park2025')}. A board that scored those would replace the calibration
  curve above with something a location code could consume directly: an arrival time with a
  standard error, scored by whether the stated error matches the observed residual
  distribution. That is the next axis to add.</p>
  </div>
""")
    A("""
</section>
""")

    # ---------------------------------------------------------------- context
    A(f"""
<section id="why">
  <div class="section-head">
    <p class="eyebrow">Why this board exists</p>
    <h2>The picker is the first stage of most new earthquake catalogues</h2>
    <p class="lede">A catalogue is what seismology hands to fault studies, to hazard models and
    to whoever has to decide whether a swarm matters. Neural-network pickers built the
    catalogues that revised fault structure in southern California{cite('ross2019')}, the
    2016-2017 central Italy sequence{cite('tan2021')} and magma transport beneath
    Hawai'i{cite('wilding2023')}, and one runs in real time at the National Earthquake
    Information Center{cite('yeck2020')}. The choice of picker therefore has a measurable
    consequence for the catalogue, and it is usually made by reading a number off a
    benchmark.</p>
  </div>

  <h3>Architecture families</h3>
  <p>These models are not variations on one design. Each treats a different property of the
  seismogram as evidence, and that choice sets how it fails.</p>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">Family</th><th class="l">Idea</th><th class="l">Examples</th></tr></thead>
    <tbody>
      <tr><td class="l">Fully convolutional, encoder-decoder</td>
          <td class="l">Predict a probability time series per phase over a window. Picks are its peaks</td>
          <td class="l">PhaseNet{cite('zhu2019')}</td></tr>
      <tr><td class="l">Window classifier</td>
          <td class="l">Slide a short window and classify it P, S or noise</td>
          <td class="l">GPD{cite('ross2018')}, ConvNetQuake{cite('perol2018')}, CNN phase classifiers{cite('woollam2019')}</td></tr>
      <tr><td class="l">Recurrent, with attention</td>
          <td class="l">Detect the event first, then pick inside it, with explicit sequence memory</td>
          <td class="l">EQTransformer{cite('mousavi2020')}, DeepPhasePick{cite('soto2021')}</td></tr>
      <tr><td class="l">Transformer</td>
          <td class="l">Self-attention over the trace in place of recurrence</td>
          <td class="l">EQCCT{cite('saad2023')}, OBSTransformer{cite('niksejel2024')}</td></tr>
      <tr><td class="l">Multi-station and operator learning</td>
          <td class="l">Pick a network jointly, so one station's noise is another's context</td>
          <td class="l">EdgePhase{cite('feng2022')}, phase neural operator{cite('sun2023')}</td></tr>
      <tr><td class="l">Pretrained backbones</td>
          <td class="l">One self-supervised model, fine-tuned per task</td>
          <td class="l">SeisLM{cite('liu2024seislm')}</td></tr>
    </tbody>
    <caption>A window classifier cannot use the shape of a coda. A multi-station model can be
    defeated by one clock error. Review of the wider field: {cite('mousavi2023')}.</caption>
  </table></div>

  <h3>Training corpora</h3>
  <p>Each model is trained on a labelled corpus from a different network, with its own
  instrumentation, noise, depth distribution and analyst conventions. A picker evaluated near
  its training distribution is answering an easier question than the same picker deployed
  elsewhere, which is what makes a single benchmark number hard to read.</p>
  <p>The corpora also carry artefacts. Labels inherited from an operator pipeline include
  mislabelled and duplicated arrivals, windows where the marked phase is not the first
  arrival, and traces whose metadata does not describe the recording. CREW was assembled with
  semi-supervised quality control for that reason{cite('aguilar2024')}, and the regional
  pickers trained on it are the current demonstration that a cleaner corpus changes what a
  model learns{cite('aguilar2025')}. Excluding artefact-contaminated windows from both the
  training and the test set is the design of the retraining this project runs, and it is why a
  benchmark assembled from the same curated corpora a model was trained on cannot settle
  whether the model generalises.</p>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">Corpus</th><th class="l">Source network</th><th class="l">Used here</th></tr></thead>
    <tbody>
      <tr><td class="l">STEAD{cite('mousavi2019stead')}</td><td class="l">Global, mostly regional distances</td><td class="l">upstream of several weight sets</td></tr>
      <tr><td class="l">INSTANCE{cite('michelini2021')}</td><td class="l">Italian national network</td><td class="l">trains <code>instance</code>. Norcia is in-domain for it</td></tr>
      <tr><td class="l">DiTing{cite('zhao2023')}</td><td class="l">China</td><td class="l">not scored here</td></tr>
      <tr><td class="l">PNW{cite('ni2023')}</td><td class="l">Pacific Northwest, incl. surface events</td><td class="l">in the <code>quakescope2026</code> fine-tune corpus</td></tr>
      <tr><td class="l">CREW{cite('aguilar2024')}</td><td class="l">Regional, continental US</td><td class="l">not scored here</td></tr>
      <tr><td class="l">TXED{cite('chen2024')}</td><td class="l">Texas</td><td class="l">not scored here</td></tr>
      <tr><td class="l">LEN-DB{cite('magrini2020')}</td><td class="l">Global, local events</td><td class="l">not scored here</td></tr>
      <tr><td class="l">OBS corpora{cite('bornstein2024','niksejel2024')}</td><td class="l">Ocean-bottom deployments</td><td class="l">offshore track only</td></tr>
    </tbody>
    <caption>SeisBench{cite('woollam2022')} exposes these corpora and models through one
    interface. That is what allows a comparison across them, and also what makes it easy to
    score a model against the corpus nearest to hand.</caption>
  </table></div>

  <h3>Scoring against an operator bulletin</h3>
  <p>M&uuml;nchmeyer et al.{cite('munchmeyer2022')} score pickers across regions on labelled
  datasets and report the cross-domain result: a picker transfers between regions with mild
  degradation and does not transfer from regional to teleseismic distances. Parametric tests of
  a single model show how strongly a reported score depends on the thresholds
  chosen{cite('pita2023')}.</p>
  <p>A deployment differs from a labelled-dataset benchmark in three ways, each of which changes
  the answer:</p>
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
    Recall at a shared threshold measures liberality as much as skill. The protocol decides the
    winner, which is the benchmark lottery{cite('dehghani2021')} in a domain where it can be
    measured.</li>
    <li><strong>The deployment is the experiment.</strong> Picking 114 million station-days
    exposes failure modes a windowed benchmark cannot: resumed jobs overwriting their own
    output, station metadata that truncates epochs, archive listings that fail silently. Each of
    those changed catalogue completeness by more than the difference between the four models
    below, and the quantity a catalogue user actually cares about is completeness{cite('woessner2005')}.</li>
  </ol>
  <div class="callout">
    <h3>Scope</h3>
    <p>This is a scored comparison on public bulletins with a public, deterministic scorer,
    reported under three protocols at once. It has no hidden test set, so a result here is not
    yet citable as a benchmark result. The <a href="#standard">standard below</a> scores the
    track against nine rules and records where it fails{cite('raji2021')}.</p>
  </div>
</section>
""")

    # ---------------------------------------------------------------- examples
    ex_path = ROOT / "reports" / "examples" / "examples.json"
    examples = json.loads(ex_path.read_text()) if ex_path.exists() else []
    if examples:
        A(f"""
<section id="examples">
  <div class="section-head">
    <p class="eyebrow">On the record</p>
    <h2>What the weight sets did on individual arrivals</h2>
    <p class="lede">Every number above is a count over thousands of arrivals. These are
    {len(examples)} of them, one record at a time: the waveform the benchmark read, the analyst
    pick in black, and each weight set's pick where it made one. The cases are chosen by what
    the models did rather than by hand, so they move only when the picks move. Each record
    downloads as MiniSEED.</p>
  </div>
  <div class="card">
    <div class="filters" id="example-tabs">""")
        for k, e in enumerate(examples):
            A(f'      <button class="chip" type="button" data-ex="{e["id"]}" '
              f'aria-pressed="{"true" if k == 0 else "false"}">{e["sequence"]} '
              f'{e["station"].split(".")[1]} {e["phase"]}</button>')
        A("    </div>")
        for k, e in enumerate(examples):
            hid = "" if k == 0 else ' hidden'
            res = e["residuals_ms"]
            got = " &middot; ".join(
                f'<span style="color:{WCOLOR[w]}">{w} {res[w]:+.0f}&thinsp;ms, conf {e["conf"][w]:.2f}</span>'
                for w in WEIGHTS if w in res)
            missed = (" &middot; ".join(f'<span style="color:#6f6890">{w} no pick</span>'
                                        for w in e["missed"]))
            A(f'    <div class="example" id="ex-{e["id"]}"{hid}>')
            A(f'      <img src="{e["svg"]}" alt="{e["sequence"]} {e["station"]} {e["phase"]}: '
              f'waveform with the analyst pick and each weight set\'s pick" loading="lazy">')
            A(f'      <p class="ex-note"><strong>{e["sequence"]} &middot; {e["station"]} &middot; '
              f'{e["phase"]}</strong>, {e["time"][:19]} UTC &middot; chosen because {e["why"]}.</p>')
            A(f'      <p class="ex-note">{got}{" &middot; " + missed if missed else ""}</p>')
            A(f'      <p class="ex-note"><a href="{e["mseed"]}">download this record</a> '
              f'({"/".join(e["channels"])}, {e["sampling_rate"]:g}&thinsp;Hz, MiniSEED) &middot; '
              f'picks for the whole sequence are in the data below</p>')
            A("    </div>")
        A("""  </div>
</section>
<script>
(function () {
  var tabs = document.getElementById("example-tabs");
  if (!tabs) return;
  tabs.addEventListener("click", function (ev) {
    var b = ev.target.closest("button[data-ex]");
    if (!b) return;
    tabs.querySelectorAll("button[data-ex]").forEach(function (o) {
      o.setAttribute("aria-pressed", String(o === b));
    });
    document.querySelectorAll(".example").forEach(function (d) {
      d.hidden = d.id !== "ex-" + b.dataset.ex;
    });
  });
})();
</script>""")

    # ---------------------------------------------------------------- metrics
    ident = [
        ("Recall", "matched reference arrivals / reference arrivals",
         "exact", cite('munchmeyer2022'), "Reported first. Unaffected by the reference being incomplete."),
        ("Picks emitted", "count above the threshold", "exact", "",
         "Recall alone is gameable by lowering the threshold. Always report both."),
        ("Recall at an equal pick count", "each model's curve read where all emit the same number of picks",
         "exact", cite('munchmeyer2022'), "Survives a change of threshold. Capped by the most conservative model."),
        ("Precision, F1", "matched / emitted, and their harmonic mean",
         "lower bound", cite('bekker2020', 'chicco2020'),
         "An unmatched pick may be an arrival the analyst never marked. Named <code>_lb</code> here. Comparable between models on the same reference, not against a labelled-dataset number."),
        ("MCC", "Matthews correlation coefficient", "not computable", cite('chicco2020'),
         "Needs true negatives, which a continuous record with an incomplete reference does not define."),
        ("MAE, RMSE", "mean absolute and root-mean-square residual", "exact", cite('munchmeyer2022'),
         "Report both: RMSE responds to outliers, MAE does not."),
        ("MedianAE, median bias", "outlier-insensitive scatter, and systematic earliness or lateness",
         "exact", cite('munchmeyer2022', 'pita2023'),
         "Separates a picker that is consistently late from one that is noisy."),
        ("Gross-error rate", f"fraction of wide-matched residuals beyond {DETECT_TOL:g}&thinsp;s",
         "exact", cite('munchmeyer2022'),
         "Must be computed from residuals matched wider than the detection tolerance, or it describes the tolerance."),
        ("Fraction within 0.1&thinsp;s", "share of matched picks inside 0.1&thinsp;s", "exact",
         cite('zhu2019'), "The tolerance the original PhaseNet paper scored at, so it is comparable to older literature."),
        ("Reliability curve, ECE", "observed agreement per confidence bin, and its mean gap",
         "lower bound", cite('guo2017'), "Tells you whether a threshold transfers between models. It usually does not."),
        ("Phase swap rate", "arrivals matched by a pick of the other phase", "exact",
         cite('munchmeyer2022'), "Survives association and moves a location."),
        ("Duplicate rate", "extra picks within tolerance of an already-matched arrival",
         "exact", cite('zhu2022', 'munchmeyer2024'), "Invisible in recall. Costs an associator work."),
        ("Model time, memory, cost", "seconds and MB per station-day, and USD per 1,000 "
         "station-days", "protocol set", "",
         "A model too slow or too large to run across the archive cannot build the catalogue. "
         "The reporting protocol is fixed above. No per-weight numbers yet."),
        ("Pick uncertainty", "a stated error on the arrival time, scored against the observed "
         "residual spread", "not yet scored", cite('yuan2023', 'park2025'),
         "The peak height of a segmentation picker is neither the detection probability nor the "
         "timing probability, so a model that reports the two separately needs its own column."),
        ("Catalogue completeness", "magnitude above which the catalogue is complete",
         "downstream", cite('woessner2005'),
         "The quantity a catalogue user cares about. Absent from this board, which needs association and location as well as picks."),
    ]
    A("""
<section id="metrics">
  <div class="section-head">
    <p class="eyebrow">What the board measures</p>
    <h2>The metrics on this board</h2>
    <p class="lede">The third column is the one that matters to anyone building an evaluation. A
    metric that is standard in machine learning is not automatically available against an
    incomplete reference, and marking which are and which are not is most of the method.</p>
  </div>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">metric</th><th class="l">definition</th><th class="l">against a bulletin</th>
    <th class="l">source</th><th class="l">reason</th></tr></thead><tbody>""")
    for name, defn, status, src, why in ident:
        cls = {"exact": "exact", "lower bound": "bound", "not computable": "unmet",
               "downstream": "", "protocol set": "partial", "not yet scored": "partial"}[status]
        A(f'    <tr><td class="l"><strong>{name}</strong></td><td class="l">{defn}</td>'
          f'<td class="l"><span class="pill {cls}">{status}</span></td><td class="l">{src}</td>'
          f'<td class="l" style="color:#6f6890">{why}</td></tr>')
    A("""  </tbody></table></div>
</section>
""")

    # ---------------------------------------------------------------- standard
    rules = [
        ("R1", "The scored task is fixed before anyone runs it", "partial",
         f"Phase, station, a {DETECT_TOL:g}&thinsp;s matching tolerance, greedy nearest one-to-one "
         "matching and a separate 2&thinsp;s residual tolerance are fixed in code and published. "
         "Nobody outside the project can submit a run, so the specification currently binds only us."),
        ("R2", "The reference arrivals are withheld from the models being scored", "unmet",
         "Every bulletin here was public before these weight sets were trained. A weight set may "
         "have seen these very arrivals, so the board measures picking skill and familiarity with "
         "the sequence together and cannot separate them."),
        ("R3", "The scorer is public, deterministic and versioned", "met",
         "<code>scripts/score_picks.py</code> and <code>sb_catalog/src/benchmark_metrics.py</code>, "
         f"pinned by {n_tests} unit tests, byte-identical output across processes, versioned in git. "
         "Anyone can rerun the board on their own picks."),
        ("R4", "The standard the field already uses is on the board", "met",
         "The baseline for phase picking is the analyst. Recall is measured directly against "
         "reviewed analyst arrivals, pulled from ANSS on track 1 and from the operating network on "
         "track 2, so every row is scored against what a human picker produced for the same events "
         "on the same stations."),
        ("R5", "A published model from another group is scored alongside", "met",
         f"Three of the four weight sets are published models from other groups: "
         f"<code>original</code>{cite('zhu2019')}, <code>jma_wc</code> and <code>instance</code>, "
         f"the last trained on INSTANCE{cite('michelini2021')}. The fourth is ours."),
        ("R6", "Training overlap with the scored regions is stated per sequence", "partial",
         "Stated and not quantified. Norcia is in-domain for <code>instance</code>, and the "
         f"<code>{ours}</code> fine-tuning corpus includes INSTANCE and Pacific Northwest "
         f"data{cite('ni2023')}. Neither overlap has been measured against the scored arrivals."),
        ("R7", "The reference set is archived so a score can be reproduced", "unmet",
         "Arrivals are pulled live from the SCEDC, NCEDC, GeoNet, INGV and NOA event services. "
         "Operators revise bulletins, so a number on this board is not reproducible after a "
         "revision. The fix is a DOI-archived arrival set with the station list and time windows."),
        ("R8", "The benchmark is run by people other than those whose model it ranks", "partial",
         f"SeisSCOPED maintains the board and one of the four weight sets is ours. It does not win "
         f"outside its own region: <code>{ours}</code> ranks {us_ours} of {len(WEIGHTS)} on track 1 "
         f"and {gl_ours} of {len(WEIGHTS)} on track 2."),
        ("R9", "Every row states the model version, the data and the cost", "partial",
         "Weight set, SeisBench version, notebook and execution time are stamped in the footer, and "
         "the reference agency is stated per track. Model time, memory and cost per station-day "
         "have a fixed reporting protocol on the board and no per-weight numbers yet."),
    ]
    n_met = sum(1 for r in rules if r[2] == "met")
    n_part = sum(1 for r in rules if r[2] == "partial")
    n_unmet = sum(1 for r in rules if r[2] == "unmet")
    assert n_met + n_part + n_unmet == len(rules)
    A(f"""
<section id="standard">
  <div class="section-head">
    <p class="eyebrow">The standard</p>
    <h2>Nine rules for a benchmark a paper can cite</h2>
    <p class="lede">The standard
    <a href="https://gaia-hazlab.github.io/hazevalhub">HazEvalHub</a> applies to every evaluation
    it collects. A picker comparison that fails any of these still tells you something, but it
    does not settle a question. Of {len(rules)} rules this board meets {n_met}, meets {n_part}
    in part, and fails {n_unmet}.</p>
  </div>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">#</th><th class="l">rule</th><th class="l">status</th>
    <th class="l">this board</th></tr></thead><tbody>""")
    for num, rule, status, note in rules:
        label = {"met": "met", "partial": "partly", "unmet": "not met"}[status]
        A(f'    <tr><td class="l"><strong>{num}</strong></td><td class="l">{rule}</td>'
          f'<td class="l"><span class="pill {status}">{label}</span></td>'
          f'<td class="l" style="color:#6f6890">{note}</td></tr>')
    A("""  </tbody></table></div>
  <div class="callout">
    <h3>R2 is the binding limit</h3>
    <p>The reference this board needs is a set of arrivals from sequences that postdate the
    training window of every weight set it ranks, picked by at least two analysts with a third
    adjudicating disagreements. Relabelling existing curated data would be cheaper and would
    measure memorisation. That is an analyst campaign rather than a software task, and it is the
    critical path to a number a paper can cite.</p>
  </div>
</section>
""")

    # ---------------------------------------------------------------- data
    A(f"""
<section id="data">
  <div class="section-head">
    <p class="eyebrow">Data</p>
    <h2>Download what the board is computed from</h2>
    <p class="lede">Every pick on both sides of every comparison, as CSV, plus the metric tables
    the page reads. Nothing here needs an account.</p>
  </div>
  <div class="table-scroll"><table class="data">
    <thead><tr><th class="l">file</th><th class="l">what is in it</th><th>rows</th>
    <th>size</th></tr></thead><tbody>""")
    for rel, what in DATA_FILES:
        f = ROOT / "reports" / "data" / rel
        if not f.exists():
            continue
        rows = sum(1 for _ in f.open()) - 1
        n_bytes = f.stat().st_size
        size = (f"{n_bytes / 1e6:.1f}&thinsp;MB" if n_bytes >= 1e6
                else f"{n_bytes / 1e3:.0f}&thinsp;kB")
        A(f'      <tr><td class="l"><a href="data/{rel}">{rel}</a></td>'
          f'<td class="l" style="color:#6f6890">{what}</td><td>{rows:,}</td>'
          f'<td>{size}</td></tr>')
    A(f"""    </tbody>
    <caption>The model-pick files are the whole inference run at a {DETECT_FLOOR} confidence
    floor, not the picks above a threshold, which is what lets any threshold on this page be
    recomputed without running a model. Column names are the ones
    <code>scripts/score_picks.py</code> reads.</caption>
  </table></div>
  <div class="grid-2" style="margin-top:18px">
    <div class="card">
      <h3>Waveforms</h3>
      <p>The archives serve the waveforms openly, so they are fetched rather than republished:
      the SCEDC and NCEDC public buckets for the western United States track, the operator's
      FDSN service for the global track. <a href="benchmark_data.html">Downloading the benchmark
      data</a> is an executed notebook that pulls the picks, fetches a window from each archive
      and plots the record with every weight set's pick on it. The records behind the examples
      above also download individually as MiniSEED.</p>
<pre><code>fs = S3FileSystem(anon=True)   # western US, no account
prefix = f"scedc-pds/continuous_waveforms/{{year}}/{{year}}_{{doy:03d}}/"

Client("GEONET").get_waveforms("NZ", "KHZ", "*", "HH?", t0, t1)   # global track</code></pre>
    </div>
    <div class="card">
      <h3>Citing a number from this page</h3>
      <p>Quote the weight set, the track, the protocol and the tolerance, because a recall
      without those four is not reproducible. The footer carries the notebook, the execution
      time and the SeisBench version behind every table.</p>
    </div>
  </div>
</section>
""")

    # ---------------------------------------------------------------- run it
    A(f"""
<section id="run">
  <div class="section-head">
    <p class="eyebrow">Score your own picks</p>
    <h2>The scorer takes two CSV files</h2>
    <p class="lede">The scorer behind every number above is public and runs on anybody's picks.
    It imports nothing but <code>numpy</code> and <code>pandas</code>.</p>
  </div>
  <div class="card">
<pre><code># the arrivals you trust:   station,phase,time
# what your model emitted:  station,phase,time,conf
python scripts/score_picks.py --reference arrivals.csv --picks mypicks.csv --out scores/

# see it work on synthetic picks with known properties, no data needed
python scripts/score_picks.py --demo</code></pre>
    <p style="margin-top:14px">Run your model <strong>once at a low confidence floor</strong>
    and keep every pick. Each threshold above is then a filter over one file rather than another
    pass over the waveforms. Optional <code>dataset</code> and <code>model</code> columns score
    several sequences or models in one run, and the column names our exports and SeisBench use
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
      <p>A held-out sequence nobody has looked at, from the embargoed acceptance suite. Model
      time, memory and cost per station-day on every row. Association and location, so that
      completeness{cite('woessner2005')} replaces recall as the headline metric. A column for
      models that report an arrival time with an error rather than a
      confidence{cite('yuan2023', 'park2025')}.</p>
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
    arrivals. The per-sequence reading is in the methods notebook.</li>
    <li>Precision and F1 are bounds, so a model that finds real arrivals the analyst skipped is
    penalised the same as one that invents them. Separating the two requires association, and no
    association step runs here.</li>
    <li>Protocol B compares the models at a pick count that <code>{bind_model}</code> caps in
    all {bind_tot} sequence-phases where every limit is measured. Protocol C has no held-out
    split, so it is an upper bound on a tuned deployment.</li>
    <li>Contamination is stated, not measured. Two of the four weight sets have plausible
    exposure to data from the regions scored here.</li>
    <li>Model runtime differs measurably between these weight sets and is absent from the
    board. A model too slow to run across a fifteen-year archive cannot be deployed, whatever
    its recall.</li>
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
