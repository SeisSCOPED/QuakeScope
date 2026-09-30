"""The picker metrics, pinned on cases where the right answer is known by hand.

Written against the definitions in Münchmeyer et al. (2022,
doi:10.1029/2021JB023499) and the adaptation this project needs: an operator
bulletin is not exhaustive, so anything that counts a false positive is a lower
bound and is named `_lb`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sb_catalog.src.benchmark_metrics import (best_threshold, detection_scores, ece,
                                              match_picks, multiplicity, phase_confusion,
                                              recall_at_budget, reliability, residual_stats,
                                              score, sweep)


def test_matching_is_nearest_and_one_to_one():
    ref = [10.0, 20.0, 30.0]
    cand = [10.2, 19.9, 19.95, 60.0]          # two near 20, one spurious, none near 30
    m = match_picks(ref, cand, conf=[0.9, 0.8, 0.7, 0.6], tol=0.5)
    assert [(i, j) for i, j, _, _ in m["pairs"]] == [(0, 0), (1, 2)], m["pairs"]
    assert [round(d, 3) for _, _, d, _ in m["pairs"]] == [0.2, -0.05]
    assert m["missed"] == [2] and sorted(m["extra"]) == [1, 3]
    print("PASS  nearest match, each model pick consumed once, leftovers reported")


def test_precision_and_f1_are_lower_bounds():
    s = detection_scores(n_reference=10, n_matched=8, n_extra=8)
    assert s["recall"] == 0.8
    assert s["precision_lb"] == 0.5                       # 8 of 16 emitted matched
    assert abs(s["f1_lb"] - 2 * 0.8 * 0.5 / 1.3) < 1e-12
    assert s["extra_rate"] == 0.8
    print("PASS  recall exact; precision and F1 are the lower bounds the naming claims")


def test_residuals_report_both_outlier_sensitive_and_not():
    r = [0.0, 0.1, -0.1, 0.2, 3.0]                        # one gross error
    st = residual_stats(r, strict_tol=0.5)
    assert st["n"] == 5
    assert abs(st["medae"] - 0.1) < 1e-12                 # robust
    assert st["rmse"] > st["mae"]                         # the 3 s pick moves RMSE more
    assert abs(st["bias"] - 0.1) < 1e-12                  # median, not dragged by the outlier
    assert abs(st["gross_error_rate"] - 0.2) < 1e-12      # 1 of 5 beyond 0.5 s
    assert abs(st["within_0.25"] - 0.8) < 1e-12
    print("PASS  MAE, RMSE, MedianAE, median bias and the gross-error rate agree by hand")


def test_a_strict_tolerance_truncates_the_residual_distribution():
    """Why residuals are matched wide and detection strict."""
    ref = [100.0]
    cand = [100.9]                                        # 0.9 s late
    strict = match_picks(ref, cand, tol=0.5)
    wide = match_picks(ref, cand, tol=2.0)
    assert strict["pairs"] == [] and len(wide["pairs"]) == 1
    assert residual_stats([p[2] for p in wide["pairs"]], strict_tol=0.5)["gross_error_rate"] == 1.0
    print("PASS  a 0.9 s error is invisible at tol 0.5 and counted as gross at tol 2")


def test_sweep_and_per_model_threshold():
    ref = list(np.arange(0.0, 100.0, 10.0))               # 10 reference picks
    times = ref + [5.0, 15.0, 25.0]                       # 3 spurious
    conf = [0.9] * 10 + [0.4] * 3
    c = sweep(ref, times, conf, thresholds=[0.2, 0.5, 0.95])
    assert list(c["emitted"]) == [13, 10, 0]
    assert c.loc[0, "recall"] == 1.0 and abs(c.loc[0, "precision_lb"] - 10 / 13) < 1e-12
    assert c.loc[1, "recall"] == 1.0 and c.loc[1, "precision_lb"] == 1.0
    b = best_threshold(c)
    assert b["best_threshold"] == 0.5 and b["best_f1_lb"] == 1.0
    print("PASS  one inference run scores every threshold; the best is the one that drops the noise")


def test_recall_at_equal_pick_budgets():
    """A liberal model and a careful one, compared where they emit the same count."""
    liberal = pd.DataFrame({"threshold": [0.1, 0.5], "emitted": [200, 100], "recall": [0.9, 0.6],
                            "precision_lb": [0.4, 0.6], "f1_lb": [0.5, 0.6]})
    careful = pd.DataFrame({"threshold": [0.1, 0.5], "emitted": [150, 50], "recall": [0.8, 0.5],
                            "precision_lb": [0.5, 0.8], "f1_lb": [0.6, 0.6]})
    tab = recall_at_budget({"liberal": liberal, "careful": careful}, budget=150)
    assert tab.loc[0, "careful"] == 0.8
    assert abs(tab.loc[0, "liberal"] - 0.75) < 1e-9      # interpolated between 100 and 200
    print("PASS  recall read at a common budget, interpolated on each model's own curve")

    # A model that cannot reach the others' counts is NaN, not silently dropped.
    tiny = pd.DataFrame({"threshold": [0.1], "emitted": [20], "recall": [0.3],
                         "precision_lb": [0.9], "f1_lb": [0.45]})
    assert np.isnan(recall_at_budget({"liberal": liberal, "tiny": tiny}, budget=150).loc[0, "tiny"])
    print("PASS  a model with a ceiling below the budget reports NaN rather than a number")


def test_calibration_says_when_confidence_is_not_a_probability():
    conf = [0.95] * 10 + [0.55] * 10
    hit = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5     # 50% in both bins
    tab = reliability(conf, hit, bins=10)
    assert set(tab["observed_lb"]) == {0.5}
    # 0.95 claimed against 0.5 observed, 0.55 against 0.5: weighted mean |diff|
    assert abs(ece(conf, hit, bins=10) - (0.45 + 0.05) / 2) < 1e-9
    print("PASS  a model claiming 0.95 and agreeing half the time is reported as miscalibrated")


def test_phase_swaps_are_identifiable_where_precision_is_not():
    reference = {"P": [10.0, 20.0], "S": [15.0]}
    candidate = {"P": [10.1, 15.05], "S": []}         # the analyst's S picked as a P
    c = phase_confusion(reference, candidate, tol=0.5)
    assert c["P_matched_same"] == 1 and c["P_n"] == 2
    assert c["S_matched_same"] == 0 and c["S_matched_as_P"] == 1
    assert c["S_swap_rate"] == 1.0
    print("PASS  an S reported as a P is counted as a swap, not as a miss")


def test_duplicate_picks_on_one_arrival():
    ref = [10.0, 50.0]
    cand = [10.05, 10.3, 50.1]                        # two picks on the first arrival
    m = multiplicity(ref, cand, tol=0.5)
    assert m["n_matched"] == 2 and m["duplicate_picks"] == 1 and m["duplicate_rate"] == 0.5
    print("PASS  a second pick on a matched arrival is counted as a duplicate")


def test_score_bundles_the_lot_consistently():
    ref = [10.0, 20.0, 30.0]
    times = [10.1, 20.2, 40.0, 30.9]                  # 2 matched, 1 spurious, 1 gross-late
    conf = [0.9, 0.8, 0.7, 0.6]
    s = score(ref, times, conf, threshold=0.5, tol=0.5, residual_tol=2.0)
    assert s["emitted"] == 4 and s["n_matched"] == 2 and s["n_extra"] == 2
    assert abs(s["recall"] - 2 / 3) < 1e-12
    assert s["res_n"] == 3                            # the 0.9 s pick matches at tol 2
    assert abs(s["res_gross_error_rate"] - 1 / 3) < 1e-12
    print("PASS  detection at the strict tolerance, residuals at the wide one, in one call")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\nall benchmark-metric checks passed")
