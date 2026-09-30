"""The scorer colleagues run on their own picks, pinned end to end.

`scripts/score_picks.py` is the published entry point to the metrics in
`sb_catalog/src/benchmark_metrics.py` (see section 8 of
https://seisscoped.org/QuakeScope/benchmark_metrics.html). These tests check
the two things a user cannot check for themselves: that the numbers the CLI
writes are the numbers the module computes, and that the failures which would
otherwise report a working picker as recall 0 stop the run instead.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sb_catalog.src.benchmark_metrics import detection_scores, match_picks, residual_stats

REPO = Path(__file__).resolve().parents[1]
SCORER = [sys.executable, str(REPO / "scripts" / "score_picks.py")]

# Two stations, arrivals 10 s apart so greedy matching is unambiguous. Picks:
# accurate on AAA, 0.12 s late on BBB, one arrival missed, one spurious pick
# below the threshold and one above it.
REF = [("XX.AAA", "P", 100.0), ("XX.AAA", "P", 110.0), ("XX.AAA", "S", 105.0),
       ("XX.BBB", "P", 100.0), ("XX.BBB", "P", 110.0), ("XX.BBB", "P", 120.0)]
PICKS = [("XX.AAA", "P", 100.04, 0.91), ("XX.AAA", "P", 110.02, 0.62),
         ("XX.AAA", "P", 130.00, 0.55), ("XX.AAA", "P", 140.00, 0.08),
         ("XX.AAA", "S", 105.03, 0.77),
         ("XX.BBB", "P", 100.12, 0.84), ("XX.BBB", "P", 110.12, 0.41),
         ("XX.BBB", "P", 119.10, 0.66)]
T0 = pd.Timestamp("2024-06-01", tz="UTC")
THR = 0.3


def write(tmp, ref_cols=("station", "phase", "time"), rename=None):
    ref = pd.DataFrame(REF, columns=["station", "phase", "sec"])
    picks = pd.DataFrame(PICKS, columns=["station", "phase", "sec", "conf"])
    for d in (ref, picks):
        d["time"] = T0 + pd.to_timedelta(d.sec, unit="s")
    rp, pp = Path(tmp) / "ref.csv", Path(tmp) / "picks.csv"
    ref[list(ref_cols)].rename(columns=rename or {}).to_csv(rp, index=False)
    picks[["station", "phase", "time", "conf"]].rename(columns=rename or {}).to_csv(pp, index=False)
    return rp, pp


def run(rp, pp, out, *extra, env=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run(SCORER + ["--reference", str(rp), "--picks", str(pp),
                                    "--out", str(out), "--threshold", str(THR), *extra],
                          capture_output=True, text=True, env=e)


def test_cli_numbers_are_the_module_numbers():
    """End to end against the metric functions, computed here by hand-grouped lists."""
    with tempfile.TemporaryDirectory() as tmp:
        rp, pp = write(tmp)
        out = Path(tmp) / "scores"
        r = run(rp, pp, out)
        assert r.returncode == 0, r.stderr
        det = pd.read_csv(out / "detection.csv")
        tim = pd.read_csv(out / "timing.csv")

        for phase in ("P", "S"):
            ref_t = [s for _, ph, s in REF if ph == phase]
            matched = extra = 0
            res = []
            for sta in sorted({s for s, *_ in REF} | {s for s, *_ in PICKS}):
                rr = sorted(s for st, ph, s in REF if ph == phase and st == sta)
                cc = sorted(s for st, ph, s, c in PICKS if ph == phase and st == sta and c >= THR)
                m = match_picks(rr, cc, tol=0.5)
                matched += len(m["pairs"]); extra += len(m["extra"])
                res += [p[2] for p in match_picks(rr, cc, tol=2.0)["pairs"]]
            want = detection_scores(len(ref_t), matched, extra)
            got = det[det.phase == phase].iloc[0]
            for k in ("n_reference", "recall", "precision_lb", "f1_lb", "extra_rate"):
                assert abs(float(got[k]) - want[k]) < 1e-12, (phase, k, got[k], want[k])
            # The CLI differences absolute epoch seconds (~1.7e9), where float64
            # spacing is 5e-7 s, so its residuals are quantised at about half a
            # microsecond. This test builds them from small numbers instead and so
            # cannot agree more closely than that. Half a microsecond is four
            # orders of magnitude below the sample interval at 100 Hz; if that ever
            # matters, give the CLI a per-group time origin.
            wt = residual_stats(res, strict_tol=0.5)
            gt = tim[tim.phase == phase].iloc[0]
            for k in ("n", "mae", "rmse", "medae", "bias", "gross_error_rate", "within_0.1"):
                assert abs(float(gt[k]) - wt[k]) < 1e-6, (phase, k, gt[k], wt[k])
        # The 0.08 pick is below the threshold, the 0.55 one is not.
        assert int(det[det.phase == "P"].emitted.iloc[0]) == 6
        print("PASS  detection and onset-time columns equal the module's own output")


def test_cli_stops_on_station_names_that_do_not_match():
    """The failure that would otherwise be published as recall 0."""
    with tempfile.TemporaryDirectory() as tmp:
        rp, pp = write(tmp)
        bad = pd.read_csv(rp)
        bad["station"] = bad.station + "..HHZ"
        bad.to_csv(rp, index=False)
        r = run(rp, pp, Path(tmp) / "scores")
        assert r.returncode != 0
        assert "No station name appears in both files" in r.stdout + r.stderr
        assert "XX.AAA..HHZ" in r.stdout + r.stderr, "the message has to show both spellings"
        print("PASS  a station-name mismatch is fatal and names a spelling from each side")


def test_cli_stops_on_disjoint_time_windows():
    with tempfile.TemporaryDirectory() as tmp:
        rp, pp = write(tmp)
        bad = pd.read_csv(rp)
        bad["time"] = pd.to_datetime(bad.time) + pd.Timedelta(days=400)
        bad.to_csv(rp, index=False)
        r = run(rp, pp, Path(tmp) / "scores")
        assert r.returncode != 0 and "disjoint time windows" in r.stdout + r.stderr
        print("PASS  a time-zone or epoch-unit error is caught before any metric is written")


def test_cli_takes_the_column_names_people_already_use():
    """`sequence`/`weights`/`probability` are what our own exports and SeisBench emit."""
    with tempfile.TemporaryDirectory() as tmp:
        rp, pp = write(tmp, rename={"conf": "probability", "time": "pick_time"})
        out = Path(tmp) / "scores"
        r = run(rp, pp, out)
        assert r.returncode == 0, r.stdout + r.stderr
        assert len(pd.read_csv(out / "detection.csv")) == 2      # P and S
        print("PASS  aliased column names are accepted without renaming the file")


def test_cli_output_is_byte_reproducible():
    """Stations are visited in sorted order, so the last digit does not move.

    Set iteration order varies with PYTHONHASHSEED between processes, which
    changes how floating-point sums accumulate. Two runs under different seeds
    have to produce identical files, or a number quoted from this script is not
    a number anyone can reproduce.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rp, pp = write(tmp)
        outs = []
        for seed in ("1", "424242"):
            out = Path(tmp) / f"scores_{seed}"
            assert run(rp, pp, out, env={"PYTHONHASHSEED": seed}).returncode == 0
            outs.append(out)
        for name in ("detection.csv", "timing.csv", "calibration.csv",
                     "phase_quality.csv", "sweep.csv"):
            a, b = [(o / name).read_bytes() for o in outs]
            assert a == b, f"{name} differs between hash seeds"
        print("PASS  identical bytes under two hash seeds, all five tables")


def test_demo_runs_without_any_data():
    """`--demo` is what somebody runs before they have a file to score."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(SCORER + ["--demo", "--out", tmp],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        det = pd.read_csv(Path(tmp) / "detection.csv")
        assert set(det.model) == {"sharp", "late", "liberal"}
        # The properties the page describes the demo by, checked rather than asserted.
        tim = pd.read_csv(Path(tmp) / "timing.csv")
        late = tim[(tim.model == "late") & (tim.phase == "P")].iloc[0]
        sharp = tim[(tim.model == "sharp") & (tim.phase == "P")].iloc[0]
        assert abs(late.bias - 0.15) < 0.03, "the injected 0.15 s delay must show up as bias"
        assert abs(sharp.bias) < 0.03 and sharp["within_0.1"] > late["within_0.1"]
        lib = det[(det.model == "liberal") & (det.phase == "P")].iloc[0]
        shp = det[(det.model == "sharp") & (det.phase == "P")].iloc[0]
        assert lib.recall > shp.recall and lib.emitted > 2 * shp.emitted
        assert abs(lib.recall_at_budget - shp.recall_at_budget) < 0.05, \
            "at a matched budget the liberal model's recall advantage has to vanish"
        print("PASS  demo reproduces the three properties section 8 reads it by")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\nall score_picks CLI checks passed")
