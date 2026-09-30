#!/usr/bin/env python
"""Score phase picks against a reference catalogue of arrivals.

Give it two CSV files - the arrivals you trust, and the picks a model made -
and it writes the metric set described in
https://seisscoped.org/QuakeScope/benchmark_metrics.html: detection at three
threshold treatments, onset-time statistics, confidence calibration, phase
swaps and duplicate picks.

Nothing here is specific to QuakeScope. It needs `numpy` and `pandas`, and the
metric definitions in `sb_catalog/src/benchmark_metrics.py`; to use it outside
this repository, copy those two files next to each other.

    # see it work on synthetic picks, no data needed
    python scripts/score_picks.py --demo

    # your own data
    python scripts/score_picks.py --reference arrivals.csv --picks mypicks.csv --out scores/

INPUT

`--reference` one row per arrival you are scoring against:

    station,phase,time
    NZ.KHZ,P,2016-11-13T11:12:58.400
    NZ.KHZ,S,2016-11-13T11:13:04.100

`--picks` one row per pick the model emitted, with its confidence. Run the
model ONCE at a low confidence floor and keep every pick: each threshold is
then a filter over this file rather than another pass over the waveforms.

    station,phase,time,conf
    NZ.KHZ,P,2016-11-13T11:12:58.512,0.83

Optional columns in either file: `dataset` (to score several sequences,
regions or experiments in one run) and, in `--picks`, `model` (to compare
several models). Missing ones default to a single unnamed group. Times are
anything `pandas.to_datetime` accepts; UTC is assumed.

WHAT IT PRINTS AND WRITES

  detection.csv    recall, precision_lb, f1_lb, emitted, extra_rate, at the
                   shared threshold, at a matched pick budget, and at each
                   model's own best threshold
  timing.csv       MAE, RMSE, MedianAE, median bias, std, percentiles, the
                   fraction within 0.1/0.25/0.5 s, and the gross-error rate
  calibration.csv  reliability bins and the expected calibration error
  phase_quality.csv  P/S swap counts and rates, duplicate picks per arrival
  sweep.csv        the full threshold sweep behind all of the above

READ THE NAMES. Anything ending `_lb` is a LOWER BOUND, because a reference
catalogue of arrivals is usually not exhaustive: an analyst picked what a
location needed and stopped, so a model pick with no reference counterpart may
be a false positive or a real arrival nobody marked. Recall, phase swaps,
residual statistics and duplicate rate are unaffected. If your reference IS
exhaustive - a labelled test set with every arrival marked - then `precision_lb`
and `f1_lb` are precision and F1: pass `--exhaustive` and they are reported
under those names.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]   # the repo, then this directory
try:                                            # inside QuakeScope
    from sb_catalog.src.benchmark_metrics import (WITHIN, best_threshold,  # noqa: E402
                                                  detection_scores, ece, match_picks,
                                                  multiplicity, phase_confusion,
                                                  recall_at_budget, reliability, residual_stats)
except ModuleNotFoundError:                     # benchmark_metrics.py copied next to this file
    from benchmark_metrics import (WITHIN, best_threshold, detection_scores,  # noqa: E402
                                   ece, match_picks, multiplicity, phase_confusion,
                                   recall_at_budget, reliability, residual_stats)

REQUIRED = {"reference": ("station", "phase", "time"), "picks": ("station", "phase", "time", "conf")}

# Whatever you already call these columns. The first name is what the code uses.
ALIAS = {"dataset": ("sequence", "event", "region", "experiment", "study", "catalogue"),
         "model": ("weights", "picker", "method", "run", "version"),
         "conf": ("confidence", "probability", "prob", "peak_value", "score"),
         "station": ("sta", "station_id", "trace_id", "id", "nsta"),
         "phase": ("phase_hint", "phase_type", "label", "type"),
         "time": ("pick_time", "arrival_time", "onset", "timestamp", "peak_time")}


def read(path: str, kind: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith((".parquet", ".pq")) else pd.read_csv(path)
    for canon, others in ALIAS.items():
        if canon not in df.columns:
            for alt in others:
                if alt in df.columns:
                    df[canon] = df[alt]
                    break
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing column(s) {missing}. Needs {list(REQUIRED[kind])}; "
                 f"found {list(df.columns)}. Accepted aliases: "
                 + "; ".join(f"{k} <- {'/'.join(v)}" for k, v in ALIAS.items() if k in missing))
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    df["epoch"] = df["time"].astype("int64") / 1e9
    for optional, default in (("dataset", "all"), ("model", "model")):
        if optional not in df.columns:
            df[optional] = default
    df["phase"] = df["phase"].astype(str).str.upper().str[0]
    return df.dropna(subset=["time"])


def demo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic picks with known properties, so the output can be read against truth.

    Three models on one station: `sharp` is accurate and calibrated, `late` has
    a 0.15 s systematic delay, `liberal` finds more arrivals but emits twice
    the picks at any threshold - the case a shared threshold hides.
    """
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    n = 300
    arrivals = np.sort(rng.uniform(0, 7200, n))
    ref = pd.DataFrame({"station": "XX.AAA", "phase": np.where(np.arange(n) % 2, "S", "P"),
                        "time": t0 + pd.to_timedelta(arrivals, unit="s")})
    rows = []
    for name, found, shift, scatter, noise in (("sharp", 0.72, 0.00, 0.04, 200),
                                               ("late", 0.72, 0.15, 0.04, 200),
                                               ("liberal", 0.88, 0.00, 0.09, 900)):
        keep = rng.random(n) < found
        t = arrivals[keep] + rng.normal(shift, scatter, keep.sum())
        rows.append(pd.DataFrame({"model": name, "station": "XX.AAA",
                                  "phase": ref.phase.values[keep],
                                  "time": t0 + pd.to_timedelta(t, unit="s"),
                                  "conf": np.clip(rng.beta(5, 2, keep.sum()), 0.02, 0.99)}))
        junk = np.sort(rng.uniform(0, 7200, noise))
        rows.append(pd.DataFrame({"model": name, "station": "XX.AAA",
                                  "phase": rng.choice(["P", "S"], noise),
                                  "time": t0 + pd.to_timedelta(junk, unit="s"),
                                  "conf": np.clip(rng.beta(2, 5, noise), 0.02, 0.99)}))
    return ref, pd.concat(rows, ignore_index=True)


def preflight(ref: pd.DataFrame, picks: pd.DataFrame) -> None:
    """Catch the mismatches that would otherwise report a working picker as recall 0."""
    rs, ps = set(ref.station), set(picks.station)
    shared = rs & ps
    def n(k, word="station"):
        return f"{k} {word}" + ("" if k == 1 else "s")
    print(f"{len(ref):,} reference arrivals on {n(len(rs))}, "
          f"{len(picks):,} picks on {n(len(ps))}, {n(len(shared))} in both")
    if not shared:
        sys.exit("No station name appears in both files, so nothing can match. "
                 f"Reference uses e.g. {sorted(rs)[:3]}; picks use e.g. {sorted(ps)[:3]}. "
                 "Make the two spellings the same (network.station is the usual choice) and rerun.")
    if len(shared) < min(len(rs), len(ps)) / 2:
        print(f"  warning: only {len(shared)} station names match. Reference-only e.g. "
              f"{sorted(rs - ps)[:3]}, picks-only e.g. {sorted(ps - rs)[:3]}")
    lo, hi = max(ref.epoch.min(), picks.epoch.min()), min(ref.epoch.max(), picks.epoch.max())
    if hi < lo:
        sys.exit("The two files cover disjoint time windows: reference "
                 f"{ref.time.min()} to {ref.time.max()}, picks {picks.time.min()} to {picks.time.max()}. "
                 "Check the time zone and the epoch units.")
    for ph in sorted(set(picks.phase) - set(ref.phase)):
        print(f"  note: picks contain phase {ph!r} with no reference arrivals of that phase; not scored")
    print()


def sweep_group(ref: pd.DataFrame, picks: pd.DataFrame, thresholds, tol: float) -> pd.DataFrame:
    """Detection scores across thresholds, matching station by station.

    Stations are visited in sorted order throughout this script, not in set
    order, so floating-point sums accumulate the same way on every run and the
    output is reproducible to the last digit.
    """
    rows = []
    n_ref = len(ref)
    for thr in thresholds:
        keep = picks[picks.conf >= thr]
        tp = extra = 0
        for sta in sorted(set(ref.station) | set(keep.station)):
            m = match_picks(sorted(ref[ref.station == sta].epoch),
                            sorted(keep[keep.station == sta].epoch), tol=tol)
            tp += len(m["pairs"]); extra += len(m["extra"])
        rows.append({"threshold": float(thr), "emitted": len(keep),
                     **detection_scores(n_ref, tp, extra)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", help="CSV of arrivals to score against")
    ap.add_argument("--picks", help="CSV of model picks with confidences")
    ap.add_argument("--demo", action="store_true", help="run on synthetic picks instead")
    ap.add_argument("--out", default="pick_scores", help="directory for the CSVs (default: pick_scores)")
    ap.add_argument("--threshold", type=float, default=0.3, help="the shared threshold to report at")
    ap.add_argument("--tol", type=float, default=0.5, help="seconds; a reference arrival counts as recovered within this")
    ap.add_argument("--residual-tol", type=float, default=2.0,
                    help="seconds; residuals are matched this wide so the distribution is not truncated by --tol")
    ap.add_argument("--exhaustive", action="store_true",
                    help="the reference marks EVERY arrival, so precision and F1 are exact, not lower bounds")
    a = ap.parse_args()

    if a.demo:
        ref, picks = demo()
        ref["epoch"] = ref.time.astype("int64") / 1e9
        picks["epoch"] = picks.time.astype("int64") / 1e9
        ref["dataset"] = picks["dataset"] = "demo"
        ref["model"] = "-"
        print("demo: 300 synthetic arrivals on one station, three models - "
              "`sharp` accurate, `late` 0.15 s delayed, `liberal` finds more but emits 2x the picks\n")
    elif a.reference and a.picks:
        ref, picks = read(a.reference, "reference"), read(a.picks, "picks")
    else:
        ap.error("give --reference and --picks, or --demo")

    preflight(ref, picks)
    thresholds = np.round(np.arange(0.02, 0.96, 0.02), 3)
    suffix = "" if a.exhaustive else "_lb"
    det_rows, tim_rows, cal_rows, qual_rows, sweeps = [], [], [], [], []

    for dataset in sorted(set(ref.dataset) | set(picks.dataset)):
        r_all = ref[ref.dataset == dataset]
        p_all = picks[picks.dataset == dataset]
        if r_all.empty:
            print(f"note: {dataset!r} has no reference arrivals; {len(p_all):,} picks not scored")
            continue
        for phase in sorted(set(r_all.phase)):
            r = r_all[r_all.phase == phase]
            if r.empty:
                continue
            curves = {}
            for model, p in p_all[p_all.phase == phase].groupby("model"):
                c = sweep_group(r, p, thresholds, a.tol)
                curves[model] = c
                sweeps.append(c.assign(dataset=dataset, phase=phase, model=model))
            if not curves:
                continue
            budget = recall_at_budget(curves, n_points=1)
            for model, c in curves.items():
                at = c.iloc[(c.threshold - a.threshold).abs().argmin()]
                b = best_threshold(c, by="f1_lb")
                det_rows.append(dict(
                    dataset=dataset, phase=phase, model=model, n_reference=int(at.n_reference),
                    emitted=int(at.emitted), recall=at.recall,
                    **{f"precision{suffix}": at.precision_lb, f"f1{suffix}": at.f1_lb},
                    extra_rate=at.extra_rate,
                    recall_at_budget=float(budget[model].iloc[0]) if model in budget else np.nan,
                    budget=int(budget.budget.iloc[0]) if budget.budget.notna().iloc[0] else np.nan,
                    best_threshold=b.get("best_threshold"), recall_at_best=b.get("recall_at_best")))

                # Residuals wide, detection strict; calibration from the strict match.
                p = p_all[(p_all.phase == phase) & (p_all.model == model) & (p_all.conf >= a.threshold)]
                res, conf, hit = [], [], []
                for sta in sorted(set(r.station) | set(p.station)):
                    rr = sorted(r[r.station == sta].epoch)
                    ps = p[p.station == sta].sort_values("epoch")
                    tt, cc = list(ps.epoch), list(ps.conf)
                    res += [x[2] for x in match_picks(rr, tt, cc, tol=a.residual_tol)["pairs"]]
                    h = np.zeros(len(tt), dtype=bool)
                    for _, j, _, _ in match_picks(rr, tt, cc, tol=a.tol)["pairs"]:
                        h[j] = True
                    conf += cc; hit += list(h)
                tim_rows.append(dict(dataset=dataset, phase=phase, model=model,
                                     **residual_stats(res, strict_tol=a.tol)))
                if conf:
                    cal = reliability(conf, hit)
                    if a.exhaustive:   # observed agreement is exact, so drop the _lb name
                        cal = cal.rename(columns={"observed_lb": "observed"})
                    cal_rows.append(cal.assign(dataset=dataset, phase=phase, model=model,
                                               **{f"ece{suffix}": ece(conf, hit)}))

        # Phase swaps and duplicates need both phases at once.
        for model, p in p_all[p_all.conf >= a.threshold].groupby("model"):
            tot = {"P_n": 0, "S_n": 0, "P_matched_as_S": 0, "S_matched_as_P": 0}
            dup = {"n_matched": 0, "duplicate_picks": 0}
            for sta in sorted(set(r_all.station) | set(p.station)):
                r_by = {ph: sorted(r_all[(r_all.phase == ph) & (r_all.station == sta)].epoch) for ph in "PS"}
                c_by = {ph: sorted(p[(p.phase == ph) & (p.station == sta)].epoch) for ph in "PS"}
                c = phase_confusion(r_by, c_by, tol=a.tol)
                for k in tot:
                    tot[k] += c[k]
                for ph in "PS":
                    m = multiplicity(r_by[ph], c_by[ph], tol=a.tol)
                    dup["n_matched"] += m["n_matched"]; dup["duplicate_picks"] += m["duplicate_picks"]
            qual_rows.append(dict(dataset=dataset, model=model,
                                  P_n=tot["P_n"], P_matched_as_S=tot["P_matched_as_S"],
                                  S_n=tot["S_n"], S_matched_as_P=tot["S_matched_as_P"],
                                  swap_rate=(tot["P_matched_as_S"] + tot["S_matched_as_P"])
                                            / max(tot["P_n"] + tot["S_n"], 1),
                                  duplicate_rate=dup["duplicate_picks"] / max(dup["n_matched"], 1)))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    detection = pd.DataFrame(det_rows); timing = pd.DataFrame(tim_rows)
    quality = pd.DataFrame(qual_rows)
    calibration = pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame()
    detection.to_csv(out / "detection.csv", index=False)
    timing.to_csv(out / "timing.csv", index=False)
    quality.to_csv(out / "phase_quality.csv", index=False)
    calibration.to_csv(out / "calibration.csv", index=False)
    pd.concat(sweeps, ignore_index=True).to_csv(out / "sweep.csv", index=False)

    pd.set_option("display.width", 200)
    print("DETECTION  " + ("(--exhaustive given, so precision and f1 are exact)" if a.exhaustive
                            else "(recall is exact; precision and f1 are LOWER BOUNDS - "
                                 "see the docstring at the top of this script)"))
    cols = ["dataset", "phase", "model", "n_reference", "emitted", "recall",
            f"precision{suffix}", f"f1{suffix}", "recall_at_budget", "budget",
            "best_threshold", "recall_at_best"]
    print(detection[cols].round(3).to_string(index=False))
    print(f"\nONSET TIME  (residuals matched at {a.residual_tol:g} s; gross_error_rate is the fraction beyond {a.tol:g} s)")
    tcols = ["dataset", "phase", "model", "n", "mae", "rmse", "medae", "bias", "std",
             "gross_error_rate"] + [f"within_{w:g}" for w in WITHIN]
    print(timing[tcols].round(3).to_string(index=False))
    if not calibration.empty:
        e = calibration.groupby(["dataset", "phase", "model"])[f"ece{suffix}"].first().round(3)
        print(f"\nCALIBRATION  expected calibration error ({'exact' if a.exhaustive else 'lower bound'})")
        print(e.to_string())
    print("\nPHASE QUALITY  (a swap is an arrival matched by a pick of the other phase)")
    print(quality.round(3).to_string(index=False))
    print(f"\nwrote detection.csv, timing.csv, calibration.csv, phase_quality.csv, sweep.csv to {out}/")


if __name__ == "__main__":
    main()
