#!/usr/bin/env python
"""Regenerate the result tables inside docs/benchmark/README.md from
docs/benchmark/results/<study>/*.csv.

The README is prose with marked blocks::

    <!-- table:recall_at_threshold -->
    ...anything here is replaced...
    <!-- /table -->

Each block is rebuilt from the CSVs the benchmark notebooks export, so the
numbers in the document are never typed. Run after re-executing any benchmark
notebook; prose outside the markers is untouched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "benchmark" / "results"
README = ROOT / "docs" / "benchmark" / "README.md"
WEIGHTS = ["quakescope2026", "jma_wc", "original", "instance"]
ORDER = ["Ridgecrest", "San Simeon", "Monte Cristo", "Mendocino 2024",
         "Kaikoura 2016", "Norcia 2016", "Thessaly 2021"]


def load(study: str, name: str) -> pd.DataFrame | None:
    p = RES / study / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def meta(study: str) -> dict:
    p = RES / study / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Markdown table; NaN renders blank, floats with floatfmt, ints plain."""
    df = df.copy()
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else (f"{int(v):,}" if float(v).is_integer() and abs(v) >= 10 else floatfmt.format(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bench() -> pd.DataFrame:
    us, gl = load("us_sequences", "recall_at_threshold"), load("global_sequences", "recall_at_threshold")
    parts = [d.assign(region=r) for d, r in ((us, "US"), (gl, "abroad")) if d is not None]
    b = pd.concat(parts, ignore_index=True)
    return b[b.sequence.isin(ORDER)]


def sweep() -> pd.DataFrame:
    parts = [d for d in (load("us_sequences", "threshold_sweep"), load("global_sequences", "threshold_sweep")) if d is not None]
    return pd.concat(parts, ignore_index=True)


def matched(sw: pd.DataFrame, phase: str, sequence: str, names=WEIGHTS) -> dict | None:
    sub = sw[(sw.phase == phase) & (sw.sequence == sequence)]
    present = [n for n in names if n in set(sub.weights)]
    if len(present) < 2:
        return None
    lo = max(sub[sub.weights == n].emitted.min() for n in present)
    hi = min(sub[sub.weights == n].emitted.max() for n in present)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return None
    mid = 0.5 * (lo + hi)
    row = {"budget": int(round(mid))}
    for n in present:
        d = sub[sub.weights == n].sort_values("emitted")
        row[n] = float(np.interp(mid, d.emitted, d.recall))
    return row


def budget_table(b: pd.DataFrame, sw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase in ("P", "S"):
        for s in [s for s in ORDER if s in set(sw.sequence)]:
            r = matched(sw, phase, s)
            if r is None:
                r = matched(sw, phase, s, [n for n in WEIGHTS if n != "instance"])
                if r is None:
                    continue
                r["instance"] = np.nan
            n = int(b[(b.phase == phase) & (b.sequence == s)].analyst.iloc[0])
            rows.append(dict(phase=phase, sequence=s, analyst=n, **r))
    return pd.DataFrame(rows)


def build_tables() -> dict[str, str]:
    t: dict[str, str] = {}
    b, sw = bench(), sweep()

    # meta
    ms = {s: meta(s) for s in ["us_sequences", "global_sequences", "ridgecrest_aftershocks", "obs_offshore", "western_reproduction"]}
    t["meta"] = md_table(pd.DataFrame([dict(study=s, notebook=m.get("notebook", ""), executed=m.get("executed", "not run"),
                                             seisbench=m.get("seisbench", "")) for s, m in ms.items()]))

    # recall at threshold
    tab = b.pivot_table(index=["phase", "sequence"], columns="weights", values="recall").reindex(columns=WEIGHTS)
    tab = tab.reindex([(p, s) for p in ("P", "S") for s in ORDER if s in set(b.sequence)])
    tab.insert(0, "analyst", [int(b[(b.phase == p) & (b.sequence == s)].analyst.iloc[0]) for p, s in tab.index])
    t["recall_at_threshold"] = md_table(tab.reset_index())

    # matched budget
    bt = budget_table(b, sw)
    t["matched_budget"] = md_table(bt)

    # pair
    at = b.pivot_table(index=["phase", "sequence"], columns="weights", values="recall")
    pair = bt.set_index(["phase", "sequence"])[["budget", "quakescope2026", "jma_wc"]].copy()
    pair["difference at budget"] = pair.quakescope2026 - pair.jma_wc
    pair["difference at 0.3"] = (at.quakescope2026 - at.jma_wc).reindex(pair.index)
    pair = pair.reset_index()
    n_pos = int((pair["difference at budget"] > 0).sum())
    mean_p = pair[pair.phase == "P"]["difference at budget"].mean()
    mean_s = pair[pair.phase == "S"]["difference at budget"].mean()
    t["pair"] = md_table(pair) + (f"\n\nMean difference at matched budget: P {mean_p:+.3f}, S {mean_s:+.3f}; "
                                  f"the fine-tune is ahead in {n_pos} of {len(pair)} sequence-phases.")

    # MAE
    mae = b.pivot_table(index=["phase", "sequence"], columns="weights", values="MAE").reindex(columns=WEIGHTS)
    mae = mae.reindex([(p, s) for p in ("P", "S") for s in ORDER if s in set(b.sequence)]) * 1000
    t["mae"] = md_table(mae.round(0).reset_index(), floatfmt="{:.0f}")

    # OBS
    det = load("obs_offshore", "detection_vs_iasp91")
    if det is not None:
        piv = det.pivot(index="weights", columns="experiment", values="rate").reindex(
            ["pickblue_phasenet", "pickblue_eqt", "obstransformer", "quakescope2026", "original"])
        t["obs_detection"] = md_table(piv.reset_index())
    aacse = load("obs_offshore", "aacse_campaign_vs_analyst")
    if aacse is not None:
        a = aacse.rename(columns={"recall_0.25s": "0.25 s", "recall_0.5s": "0.5 s", "recall_1s": "1 s", "recall_2s": "2 s",
                                  "median_residual_s": "median residual (s), 1 s hits"})
        t["aacse"] = md_table(a)
    p = RES / "obs_offshore" / "aacse_rescoring.csv"
    if p.exists():
        r = pd.read_csv(p, header=[0, 1])
        r.columns = ["weights"] + [f"{a} {bb}" if not bb.startswith("P_median") else "median P residual (s)" for a, bb in r.columns[1:]]
        t["aacse_rescore"] = md_table(r)
    ax = load("obs_offshore", "axial_event_detection_by_magnitude")
    if ax is not None:
        t["axial"] = md_table(ax.rename(columns={"detected": "P on >= 3 stations", "detected_conf05": "same, conf >= 0.5", "n": "events"}))

    # reproduction
    m = ms.get("western_reproduction", {})
    if m.get("totals"):
        tt = m["totals"]
        rows = [("targeted station-days", tt["station_days_targeted"]),
                ("of which with campaign picks", tt["station_days_with_picks"]),
                ("campaign picks on those", tt["campaign_picks"]),
                ("re-picked via FDSN", tt["repicked"]),
                ("matched exactly on (station, band, phase, peak)", tt["matched_exact"]),
                ("fraction of the campaign's picks reproduced exactly", f"{tt['matched_exact'] / tt['campaign_picks']:.4%}"),
                ("campaign-only", tt["campaign_only"]), ("re-pick-only", tt["repick_only"])]
        t["reproduction"] = md_table(pd.DataFrame(rows, columns=["", "value"]).astype({"value": str}))
    else:
        t["reproduction"] = "_western_reproduction/ not present: execute tutorials/western_pick_validation.ipynb._"

    # aftershocks
    ra = load("ridgecrest_aftershocks", "recall_at_threshold")
    if ra is not None:
        w = ra.pivot(index="weights", columns="phase", values=["analyst", "recall", "MAE"]).reindex(WEIGHTS)
        w.columns = [f"{a} {b}" for a, b in w.columns]
        t["aftershocks"] = md_table(w.reset_index())
    return t


def main() -> None:
    text = README.read_text()
    tables = build_tables()
    for name, body in tables.items():
        pat = re.compile(rf"(<!-- table:{name} -->\n)(.*?)(<!-- /table -->)", re.S)
        if not pat.search(text):
            print(f"no block for {name} in README; skipped")
            continue
        text = pat.sub(lambda m: f"{m.group(1)}{body}\n{m.group(3)}", text)
    README.write_text(text)
    print(f"rewrote {len(tables)} table blocks in {README.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
