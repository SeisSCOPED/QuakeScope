"""Metrics for scoring a phase picker against a reference catalogue of arrivals.

The metric set follows the benchmark literature, with one adaptation that
matters for everything here and is stated first.

**Our reference is not exhaustive.** Münchmeyer et al. (2022,
doi:10.1029/2021JB023499), Zhu & Beroza (2019) and the SeisBench benchmarks
score against labelled datasets where every arrival in the scored window is
known, so an unmatched model pick is a false positive and precision, F1 and
MCC are identifiable. Ours are operator bulletins: an analyst picked what a
location needed, and in a dense aftershock sequence most unmatched model picks
are real arrivals nobody had time to mark. Counting them as false positives
therefore *understates* precision by an unknown amount.

So every quantity here that needs a false-positive count is named `_lb`, a
**lower bound**: `precision_lb`, `f1_lb`. They are comparable between models on
the same reference, which is what a model choice needs, and they are not
comparable with a number from a labelled-dataset benchmark. Recall is
unaffected and is the primary metric, as it is in the OBS picker literature
(Bornstein et al., 2024; Niksejel & Zhang, 2024), where the same limitation
applies.

What is computed, and why each one earns its place:

*Detection*, at a threshold and against a pick budget
    `recall`, `precision_lb`, `f1_lb`, and the count of unmatched model picks
    per reference pick (`extra_rate`). A threshold is not an operating point:
    two weight sets at 0.3 emit different numbers of picks, so `recall` at a
    shared threshold conflates skill with liberality. `sweep()` gives the curve
    and `recall_at_budget()` reads it at equal emitted picks - the comparison
    that survives a change of threshold. `best_threshold()` is the other
    standard answer, per-model threshold selection, as Münchmeyer et al. do on
    a development set.

*Onset time*
    `mae`, `rmse`, `medae`, `bias` (median residual), `std`, `p10`/`p90`, and
    the fraction within 0.1, 0.25 and 0.5 s. MAE and RMSE together because one
    is insensitive to outliers and the other is not, which is the argument
    Münchmeyer et al. make for reporting both; the median because it separates
    a systematic early or late pick from scatter.

    **Residuals are truncated by the matching tolerance.** Match at 0.5 s and
    no residual can exceed 0.5 s, so an outlier rate computed there is a
    statement about the tolerance, not the picker. Score residuals with a wide
    tolerance (`residual_tol`, default 2 s) and detection with the strict one,
    and report `gross_error_rate`, the fraction of wide matches beyond the
    strict tolerance. That is the quantity Münchmeyer et al. call the fraction
    of high residuals (>0.45 s regional), made honest for a matched reference.

*Confidence calibration*
    `reliability()` and `ece()`. A pick's `conf` is used as a threshold by
    every downstream user, so whether it behaves like a probability of being a
    real arrival is a practical question, and it is why a threshold carried
    across weight sets moves the operating point. Reported as observed
    agreement with the reference per confidence bin; where the reference is
    sparse this is a lower bound like precision, and labelled as such.

*Phase identification*
    `phase_confusion()`: analyst P matched by a model S and the reverse. The
    phase label exists on both sides, so unlike precision this is identifiable.
    Münchmeyer et al. score it with MCC on a labelled set; on a bulletin the
    honest form is the confusion count and the swap rate.

*Multiplicity*
    `multiplicity()`: model picks within the tolerance of one reference pick
    beyond the one that matched. Duplicate picks on a single arrival inflate a
    pick count without adding information and cost an associator work.

All functions take plain sequences of floats (seconds, any epoch) and return
plain dicts or DataFrames, so they can be used on any study's exported picks.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Fractions of matched picks inside these residuals are reported as
# `within_0.1` and so on. 0.1 s is the tolerance Zhu & Beroza (2019) score
# PhaseNet at; 0.5 s is the usual regional operating tolerance.
WITHIN = (0.1, 0.25, 0.5)


def match_picks(reference: Sequence[float], candidate: Sequence[float],
                conf: Optional[Sequence[float]] = None, tol: float = 0.5) -> dict:
    """Greedy nearest one-to-one matching of model picks to reference picks.

    Each reference pick takes the nearest unused candidate within `tol`; each
    candidate is consumed once. Greedy-nearest is what the picker benchmarks
    use and is stable for arrivals separated by more than the tolerance; in a
    sequence with arrivals closer together than `tol` it can differ from the
    optimal assignment, which is why Ridgecrest is scored at 0.5 s rather than
    wider (aftershocks there are 0.35 s apart).

    Returns the matched pairs (reference index, candidate index, signed
    residual candidate - reference, candidate confidence), and the indices left
    over on each side.
    """
    reference = list(reference)
    candidate = list(candidate)
    conf = list(conf) if conf is not None else [float("nan")] * len(candidate)
    used: set[int] = set()
    pairs = []
    for i, a in enumerate(reference):
        best_j = best_d = None
        for j, m in enumerate(candidate):
            if j in used:
                continue
            d = m - a
            if abs(d) <= tol and (best_d is None or abs(d) < abs(best_d)):
                best_j, best_d = j, d
        if best_j is not None:
            used.add(best_j)
            pairs.append((i, best_j, best_d, conf[best_j]))
    return {
        "pairs": pairs,
        "missed": [i for i in range(len(reference)) if i not in {p[0] for p in pairs}],
        "extra": [j for j in range(len(candidate)) if j not in used],
    }


def detection_scores(n_reference: int, n_matched: int, n_extra: int) -> dict:
    """Recall, and the lower bounds on precision and F1.

    `precision_lb` counts every unmatched model pick as a false positive. On an
    operator bulletin that is wrong for an unknown share of them - they are
    arrivals the analyst did not need - so the true precision is at least this.
    """
    recall = n_matched / n_reference if n_reference else float("nan")
    precision_lb = n_matched / (n_matched + n_extra) if (n_matched + n_extra) else float("nan")
    f1_lb = (2 * recall * precision_lb / (recall + precision_lb)
             if recall and precision_lb and (recall + precision_lb) else 0.0)
    return {
        "n_reference": n_reference, "n_matched": n_matched, "n_extra": n_extra,
        "recall": recall, "precision_lb": precision_lb, "f1_lb": f1_lb,
        "extra_rate": n_extra / n_reference if n_reference else float("nan"),
    }


def residual_stats(residuals: Iterable[float], strict_tol: float = 0.5) -> dict:
    """Onset-time statistics of signed residuals (model minus reference).

    Feed residuals matched at a WIDE tolerance: matching at the strict
    tolerance truncates the distribution there and makes any outlier measure a
    property of the tolerance. `gross_error_rate` is the fraction beyond
    `strict_tol`, the honest form of the high-residual fraction in
    Münchmeyer et al. (2022).
    """
    r = np.asarray([x for x in residuals if x is not None and not math.isnan(x)], dtype=float)
    if r.size == 0:
        return {"n": 0, **{k: float("nan") for k in
                           ("mae", "rmse", "medae", "bias", "mean", "std", "p10", "p90",
                            "iqr", "gross_error_rate", *[f"within_{w:g}" for w in WITHIN])}}
    out = {
        "n": int(r.size),
        "mae": float(np.mean(np.abs(r))),
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "medae": float(np.median(np.abs(r))),      # robust, reported by recent benchmarks
        "bias": float(np.median(r)),               # systematic early (<0) or late (>0)
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)) if r.size > 1 else float("nan"),
        "p10": float(np.percentile(r, 10)),
        "p90": float(np.percentile(r, 90)),
        "iqr": float(np.percentile(r, 75) - np.percentile(r, 25)),
        "gross_error_rate": float(np.mean(np.abs(r) > strict_tol)),
    }
    for w in WITHIN:
        out[f"within_{w:g}"] = float(np.mean(np.abs(r) <= w))
    return out


def sweep(reference: Sequence[float], candidate_times: Sequence[float],
          candidate_conf: Sequence[float], thresholds: Iterable[float],
          tol: float = 0.5) -> pd.DataFrame:
    """Detection scores across confidence thresholds, from one inference run.

    Run the model once at a low floor and keep each pick's peak confidence;
    every threshold is then a filter, not another pass over the waveforms.
    """
    times = np.asarray(candidate_times, dtype=float)
    conf = np.asarray(candidate_conf, dtype=float)
    rows = []
    for thr in thresholds:
        keep = conf >= thr
        m = match_picks(reference, times[keep], conf[keep], tol=tol)
        s = detection_scores(len(reference), len(m["pairs"]), len(m["extra"]))
        rows.append({"threshold": float(thr), "emitted": int(keep.sum()), **s})
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame, by: str = "f1_lb") -> dict:
    """The row of a sweep that maximises `by` - per-model threshold selection.

    Münchmeyer et al. (2022) choose each model's threshold on a development
    set rather than sharing one, because the probability scales differ. With no
    development set here, this reports the threshold each model would need to
    be seen at its best on this reference, which is an upper bound on what a
    tuned deployment would achieve and is labelled that way.
    """
    if curve.empty or curve[by].isna().all():
        return {}
    row = curve.loc[curve[by].idxmax()]
    return {f"best_{by}": float(row[by]), "best_threshold": float(row["threshold"]),
            "recall_at_best": float(row["recall"]), "precision_lb_at_best": float(row["precision_lb"]),
            "emitted_at_best": int(row["emitted"])}


def recall_at_budget(curves: dict[str, pd.DataFrame], budget: Optional[float] = None,
                     n_points: int = 1) -> pd.DataFrame:
    """Recall of several models at equal numbers of emitted picks.

    The comparison that survives a change of threshold. The budget range is the
    overlap of the models' emitted counts; a model that cannot reach the others
    at all has a ceiling rather than a calibration offset, and is returned as
    NaN so the difference is visible instead of silently dropped.
    """
    names = list(curves)
    lo = max(float(c["emitted"].min()) for c in curves.values())
    hi = min(float(c["emitted"].max()) for c in curves.values())
    rows = []
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return pd.DataFrame([{"budget": float("nan"), **{n: float("nan") for n in names}}])
    targets = [budget] if budget is not None else list(np.linspace(lo, hi, n_points + 2)[1:-1]) or [0.5 * (lo + hi)]
    for t in targets:
        row = {"budget": int(round(t))}
        for n, c in curves.items():
            d = c.sort_values("emitted")
            row[n] = (float(np.interp(t, d["emitted"], d["recall"]))
                      if d["emitted"].min() <= t <= d["emitted"].max() else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def reliability(conf: Sequence[float], hit: Sequence[bool], bins: int = 10) -> pd.DataFrame:
    """Observed agreement with the reference, per confidence bin.

    `hit` is whether each emitted pick matched a reference pick. Where the
    reference is not exhaustive this is a lower bound on correctness, the same
    caveat as `precision_lb`, and the curve is still the right shape to compare
    weight sets: it is why one threshold means different things to each.
    """
    c = np.asarray(conf, dtype=float)
    h = np.asarray(hit, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(c, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_conf": float(c[m].mean()), "observed_lb": float(h[m].mean())})
    return pd.DataFrame(rows)


def ece(conf: Sequence[float], hit: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error: mean |confidence - observed|, weighted by bin count."""
    tab = reliability(conf, hit, bins)
    if tab.empty:
        return float("nan")
    w = tab["n"] / tab["n"].sum()
    return float((w * (tab["mean_conf"] - tab["observed_lb"]).abs()).sum())


def phase_confusion(reference: dict[str, Sequence[float]],
                    candidate: dict[str, Sequence[float]], tol: float = 0.5) -> dict:
    """Reference arrivals matched by a model pick of the other phase.

    Both sides carry a phase label, so this is identifiable where precision is
    not. A P reported where the analyst marked S is a different failure from a
    missed arrival: it survives association and moves a location.
    """
    out = {}
    for pha, other in (("P", "S"), ("S", "P")):
        ref = list(reference.get(pha, []))
        same = match_picks(ref, candidate.get(pha, []), tol=tol)
        missed_times = [ref[i] for i in same["missed"]]
        cross = match_picks(missed_times, candidate.get(other, []), tol=tol)
        out[f"{pha}_n"] = len(ref)
        out[f"{pha}_matched_same"] = len(same["pairs"])
        out[f"{pha}_matched_as_{other}"] = len(cross["pairs"])
        out[f"{pha}_swap_rate"] = len(cross["pairs"]) / len(ref) if ref else float("nan")
    return out


def multiplicity(reference: Sequence[float], candidate: Sequence[float],
                 tol: float = 0.5) -> dict:
    """Extra model picks sitting within `tol` of a reference pick that matched.

    One arrival, several picks: no new information, and an associator has to
    resolve them. Counted only around matched reference picks, so it does not
    conflate duplicates with detections of arrivals the analyst never marked.
    """
    m = match_picks(reference, candidate, tol=tol)
    matched_ref = [reference[i] for i, *_ in m["pairs"]]
    extra_times = [candidate[j] for j in m["extra"]]
    dup = sum(1 for a in matched_ref for t in extra_times if abs(t - a) <= tol)
    return {"n_matched": len(matched_ref), "duplicate_picks": dup,
            "duplicate_rate": dup / len(matched_ref) if matched_ref else float("nan")}


def score(reference: Sequence[float], candidate_times: Sequence[float],
          candidate_conf: Sequence[float], threshold: float = 0.3,
          tol: float = 0.5, residual_tol: float = 2.0) -> dict:
    """Every scalar metric for one (reference, model, phase) at one threshold.

    Detection is scored at `tol`; residuals at `residual_tol`, wide enough not
    to truncate the distribution, with `gross_error_rate` reporting how much of
    it falls outside `tol`.
    """
    times = np.asarray(candidate_times, dtype=float)
    conf = np.asarray(candidate_conf, dtype=float)
    keep = conf >= threshold
    strict = match_picks(reference, times[keep], conf[keep], tol=tol)
    wide = match_picks(reference, times[keep], conf[keep], tol=residual_tol)
    det = detection_scores(len(reference), len(strict["pairs"]), len(strict["extra"]))
    res = residual_stats([p[2] for p in wide["pairs"]], strict_tol=tol)
    hit = np.zeros(int(keep.sum()), dtype=bool)
    for _, j, _, _ in strict["pairs"]:
        hit[j] = True
    return {"threshold": threshold, "emitted": int(keep.sum()), **det,
            **{f"res_{k}": v for k, v in res.items()},
            "ece_lb": ece(conf[keep], hit) if keep.any() else float("nan"),
            **multiplicity(reference, times[keep], tol=tol)}
