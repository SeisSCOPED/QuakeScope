#!/usr/bin/env python
"""Write the PR #71 (VibeSys) evaluation tables to docs/benchmark/results/vibesys_pr71/.

Reads the outputs of the harness in this directory, as they were left by the
2026-10-26 run, and writes one CSV per table that scripts/build_vibesys_report.py
renders. No number on the report page is typed; each comes from these files.

    python scripts/vibesys_pr71/export_results.py RUN_DIR

RUN_DIR holds out/ (laptop picks, scores, amplitudes, timing) and ec2_results2/
(the three Graviton instances' results_<tag>/out/).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sb_catalog.src.benchmark_metrics import match_picks  # noqa: E402

RUN = Path(sys.argv[1])
OUT = ROOT / "docs/benchmark/results/vibesys_pr71"
OUT.mkdir(parents=True, exist_ok=True)
LAP = RUN / "out"
EC2 = RUN / "ec2_results2"
TRACKS = ("us_sequences", "swarm_sequences")
THR = 0.2  # production threshold of the western campaign


def write(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}.csv  {len(df)} rows")


def fast_match(a, b, tol=0.5):
    """Greedy one-to-one match by increasing |b - a| within tol: (pairs, missed a, extra b)."""
    order = np.argsort(b)
    bs = b[order]
    cand = []
    for i, x in enumerate(a):
        lo, hi = np.searchsorted(bs, x - tol), np.searchsorted(bs, x + tol, side="right")
        cand += [(abs(bs[k] - x), i, order[k], bs[k] - x) for k in range(lo, hi)]
    cand.sort()
    ua, ub, pairs = set(), set(), []
    for _, i, j, d in cand:
        if i not in ua and j not in ub:
            ua.add(i)
            ub.add(j)
            pairs.append((i, j, d))
    return pairs, [i for i in range(len(a)) if i not in ua], [j for j in range(len(b)) if j not in ub]


# --- 1. picks against vanilla and against analysts (laptop run, 130 windows) ---------------
for thr in (0.2, 0.3):
    p = pd.read_csv(LAP / f"paired_{thr}.csv")
    p.insert(0, "threshold", thr)
    for col in ("identical", "lost", "new"):
        p[f"{col}_n"] = (p[col] * p.vanilla).round().astype(int)
    p["shifted_n"] = (p.shift_ge_1smp * p.paired * p.vanilla).round().astype(int)
    a = pd.read_csv(LAP / f"analyst_{thr}.csv")
    a.insert(0, "threshold", thr)
    if thr == 0.2:
        vv, aa = p, a
    else:
        vv, aa = pd.concat([vv, p]), pd.concat([aa, a])
write("picks_vs_vanilla", vv)
write("picks_vs_analyst", aa)

rows = []
for thr in (0.2, 0.3):
    h = pd.read_csv(LAP / f"hits_{thr}.csv")
    h = h.pivot_table(index=["track", "seq", "sta", "phase", "i"], columns="arm", values="hit").reset_index()
    for arm in ("vanilla", "pr_fp32", "pr_bf16"):
        h[arm] = h[arm].astype(bool)
    for (trk, pha), g in h.groupby(["track", "phase"]):
        rows.append(dict(threshold=thr, track=trk, phase=pha, analyst=len(g),
                         bf16_lost=int((g.vanilla & ~g.pr_bf16).sum()),
                         bf16_gained=int((~g.vanilla & g.pr_bf16).sum()),
                         fp32_changed=int((g.vanilla != g.pr_fp32).sum())))
write("analyst_discordant", pd.DataFrame(rows))

# --- 2. perturbation on the scale of analyst uncertainty ------------------------------------
picks = {a: pd.read_csv(LAP / f"picks_original_{a}.csv") for a in ("vanilla", "pr_fp32", "pr_bf16")}
for d in picks.values():
    d["t"] = pd.to_datetime(d.time, format="mixed").astype("int64").to_numpy() / 1e9
groups = {a: {k: (g.t.to_numpy(), g.conf.to_numpy()) for k, g in d.groupby(["sequence", "station", "phase"])}
          for a, d in picks.items()}
empty = (np.array([]), np.array([]))
ref = []
for t in TRACKS:
    r = pd.read_csv(ROOT / f"docs/benchmark/results/{t}/reference_picks.csv")
    r["track"] = t
    ref.append(r)
ref = pd.concat(ref)
ref["t"] = pd.to_datetime(ref.time, format="mixed").astype("int64").to_numpy() / 1e9
covered = set(groups["vanilla"])
rows = []
for pha in "PS":
    res = []
    for key, g in ref[ref.phase == pha].groupby(["sequence", "station", "phase"]):
        if key not in covered:
            continue
        t, q = groups["vanilla"][key]
        res += [d for _, _, d in fast_match(g.t.to_numpy(), t[q >= THR])[0]]
    r = np.array(res)
    rows.append(dict(phase=pha, matched=len(r), median_ms=1e3 * np.median(r), std_ms=1e3 * r.std(),
                     robust_sigma_ms=1e3 * 1.4826 * np.median(np.abs(r - np.median(r)))))
write("vanilla_vs_analyst_residuals", pd.DataFrame(rows))

rows = []
for arm in ("pr_fp32", "pr_bf16"):
    for pha in "PS":
        sh, lost, new = [], [], []
        for key in sorted(k for k in covered if k[2] == pha):
            vt, vq = groups["vanilla"].get(key, empty)
            ct, cq = groups[arm].get(key, empty)
            vk, ck = vq >= THR, cq >= THR
            pairs, mi, ex = fast_match(vt[vk], ct[ck])
            sh += [d for _, _, d in pairs]
            lost += list(vq[vk][mi])
            new += list(cq[ck][ex])
        a = np.abs(np.array(sh))
        rows.append(dict(arm=arm, phase=pha, threshold=THR, vanilla=len(sh) + len(lost),
                         within_10ms=np.mean(a <= 0.0105), within_20ms=np.mean(a <= 0.0205),
                         beyond_50ms=np.mean(a > 0.0505), max_shift_ms=1e3 * a.max(),
                         std_shift_ms=1e3 * np.std(sh), lost=len(lost), new=len(new),
                         lost_or_new_conf_ge_0p25=int(np.sum(np.array(lost) >= THR + 0.05)
                                                      + np.sum(np.array(new) >= THR + 0.05))))
write("tolerance", pd.DataFrame(rows))

# fp32 differences from vanilla at the 0.02 floor, every one of them
k = ["sequence", "station", "phase", "time"]
m = picks["vanilla"][k + ["conf"]].merge(picks["pr_fp32"][k + ["conf"]], on=k, how="outer",
                                         indicator=True, suffixes=("_vanilla", "_pr_fp32"))
write("fp32_differences_at_floor", m[m._merge != "both"].rename(columns={"_merge": "side"})
      .replace({"side": {"left_only": "vanilla only", "right_only": "pr_fp32 only"}}))

# --- 3. amplitudes --------------------------------------------------------------------------
a, b = pd.read_csv(LAP / "amp_all_main.csv"), pd.read_csv(LAP / "amp_all_pr.csv")
m = a.merge(b, on=["sequence", "station", "phase", "time", "conf"], suffixes=("_main", "_pr"))
rows = []
for c, name in (("wa", "Wood-Anderson"), ("vel", "velocity")):
    x, y = m[c + "_main"], m[c + "_pr"]
    both = x.notna() & y.notna()
    r = (y[both] / x[both] - 1).abs()
    rows.append(dict(amplitude=name, picks=len(m), stations=m.station.nunique(), with_value=int(both.sum()),
                     missing_pattern_differs=int((x.isna() != y.isna()).sum()),
                     bit_identical=float((x[both] == y[both]).mean()), max_rel_diff=float(r.max()),
                     p999_rel_diff=float(r.quantile(0.999))))
write("amplitudes", pd.DataFrame(rows))

# --- 4. timing ------------------------------------------------------------------------------
rows = []
for host, out in [("M1 Max", LAP)] + [(t, EC2 / f"results_{t}" / "out") for t in ("c6g", "c7g", "c8g")]:
    for arm in ("vanilla", "pr"):
        for rep in (1, 2, 3, 4):
            fs = [f for f in sorted(glob.glob(str(out / f"bench_{arm}_{rep}_g*.json"))) if not f.endswith(".picks.json")]
            d = [json.load(open(f)) for f in fs]
            rows.append(dict(host=host, arm=arm, rep=rep, workers=len(d), complete=len(d) == 4,
                             wall_s=max(x["total"] for x in d) if d else np.nan,
                             **{f"{s}_s": sum(x[s] for x in d) if d else np.nan
                                for s in ("startup", "read", "classify", "wa", "vel")},
                             picks=sum(x["picks"] for x in d) if d else 0,
                             emulated=d[0].get("emulated") if d else None,
                             velocity_kernel=(d[0].get("kernel") or "").split(" isa")[0] if d else None))
write("timing_reps", pd.DataFrame(rows))

# --- 5. exact agreement between hosts (rep 2: every host has a complete rep 2) --------------
sets = {}
for t in ("c6g", "c7g", "c8g"):
    for arm in ("vanilla", "pr"):
        s = set()
        for f in sorted(glob.glob(str(EC2 / f"results_{t}/out/bench_{arm}_2_g*.picks.json"))):
            s |= {(sta, pha, tt[:23]) for sta, pha, tt, _ in json.load(open(f))}
        sets[(t, arm)] = s
rows = [dict(a=f"{a[0]} {a[1]}", b=f"{b[0]} {b[1]}", n_a=len(sets[a]), n_b=len(sets[b]),
             identical=len(sets[a] & sets[b])) for a in sets for b in sets if a < b]
write("graviton_exactness", pd.DataFrame(rows))

# --- 6. hosts and prices --------------------------------------------------------------------
hosts = [dict(host="M1 Max", kind="laptop", cpu="Apple M1 Max (8 performance + 2 efficiency cores), 64 GB",
              vcpu=10, memory_gb=64, python="3.10", torch="2.13.0 (conda-forge, no oneDNN)",
              seisbench="0.12.3", on_demand_usd_h=np.nan, spot_usd_h=np.nan)]
for t, g, od, sp in (("c6g", "Graviton2 (Neoverse-N1)", 0.272, 0.0802), ("c7g", "Graviton3", 0.289, 0.0824),
                     ("c8g", "Graviton4", 0.31792, 0.0772)):
    env = (EC2 / f"results_{t}/out/env.txt").read_text()
    torch_line = next(line for line in env.splitlines() if "mkldnn" in line and line[:1].isdigit())
    hosts.append(dict(host=t, kind=f"EC2 {t}.2xlarge on-demand, us-east-2", cpu=f"AWS {g}", vcpu=8, memory_gb=16,
                      python="3.12.15", torch=torch_line, seisbench="0.12.5", on_demand_usd_h=od, spot_usd_h=sp))
write("hosts", pd.DataFrame(hosts))

json.dump(dict(
    pr="https://github.com/SeisSCOPED/QuakeScope/pull/71", pr_head="a5d4d43", vanilla_commit="7dd1be7",
    vanilla_note="sb_catalog/src identical at 887366d, origin/main 7dd1be7 and the PR's parent",
    optimized_settings_from="4271d67 vibesys/pick_workload.py (removed from the PR in 32d27cf)",
    weight="PhaseNet original, weight version 2", run_date="2026-10-26",
    accuracy_workload="130 cached station-windows of docs/benchmark/results/{us,swarm}_sequences, 7,593 analyst picks",
    timing_workload="42 station-windows at 100 Hz with a cached response, 133.3 station-hours = 5.55 station-days; "
                    "4 worker processes x 2 threads; 4 alternating reps per arm",
    timing_scope="compute only: local miniSEED and StationXML; no S3 reads, no Parquet writes",
    prices="on-demand from the AWS Pricing API, cheapest-AZ Spot from describe_spot_price_history, both 2026-10-26",
    fargate_spot_rate_usd_vcpu_h=0.0213,
    fargate_spot_rate_source="CloudBank billed costs over attempts, campaign closeout (docs/rerun_2026/34, PR #66)",
), open(OUT / "meta.json", "w"), indent=1)
print("meta.json")
