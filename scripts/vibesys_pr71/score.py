"""Score the PR #71 arms against the analyst reference and against vanilla.

usage: score.py REPO OUT_DIR
Reads OUT_DIR/picks_<weight>_<arm>.csv, writes OUT_DIR/*.csv tables and prints them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REPO))
from sb_catalog.src.benchmark_metrics import match_picks, residual_stats  # noqa: E402

R = REPO / "docs/benchmark/results"
TRACK = {}
ref = []
for t in ("us_sequences", "swarm_sequences"):
    r = pd.read_csv(R / t / "reference_picks.csv", parse_dates=["time"])
    r["track"] = t
    ref.append(r)
    for s in r.sequence.unique():
        TRACK[s] = t
ref = pd.concat(ref)
ref["time"] = pd.to_datetime(ref.time, format="mixed")
# Only stations whose waveforms were picked (the reference covers more stations).
ARMS = ("vanilla", "pr_fp32", "pr_bf16", "pr_fp16")
WEIGHTS = ("original",)
picks = pd.concat([pd.read_csv(OUT / f"picks_{w}_{a}.csv", parse_dates=["time"])
                   for w in WEIGHTS for a in ARMS])
picks["time"] = pd.to_datetime(picks.time, format="mixed")
picks["track"] = picks.sequence.map(lambda s: TRACK.get(s, "us_sequences"))
covered = set(zip(picks.sequence, picks.station))
ref = ref[[k in covered for k in zip(ref.sequence, ref.station)]]
sec = lambda s: s.astype("int64").to_numpy() / 1e9  # noqa: E731
picks["t"] = sec(picks.time)
GROUPS = {k: (g.t.to_numpy(), g.conf.to_numpy()) for k, g in
          picks.groupby(["weights", "arm", "sequence", "station", "phase"])}
EMPTY = (np.array([]), np.array([]))


def fast_match(a: np.ndarray, b: np.ndarray, tol: float = 0.5):
    """Greedy one-to-one match by increasing |b - a| within tol; (pairs, missed a, extra b)."""
    order = np.argsort(b)
    bs = b[order]
    cand = []
    for i, x in enumerate(a):
        lo, hi = np.searchsorted(bs, x - tol), np.searchsorted(bs, x + tol, side="right")
        for k in range(lo, hi):
            cand.append((abs(bs[k] - x), i, order[k], bs[k] - x))
    cand.sort()
    ua, ub, pairs = set(), set(), []
    for _, i, j, d in cand:
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j); pairs.append((i, j, d))
    return pairs, [i for i in range(len(a)) if i not in ua], [j for j in range(len(b)) if j not in ub]
pd.set_option("display.width", 200, "display.max_columns", 30)


def analyst(threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall and residuals per (track, weight, arm, phase); per-reference hits."""
    rows, hits = [], []
    for (seq, sta, pha), rg in ref.groupby(["sequence", "station", "phase"]):
        rt = sec(rg.time)
        for w in WEIGHTS:
            for a in ARMS:
                t, q = GROUPS.get((w, a, seq, sta, pha), EMPTY)
                ct = t[q >= threshold]
                strict = match_picks(rt, ct, tol=0.5)
                wide = match_picks(rt, ct, tol=2.0)
                rows.append(dict(track=TRACK[seq], weights=w, arm=a, phase=pha, n_ref=len(rt),
                                 matched=len(strict["pairs"]), emitted=len(ct),
                                 resid=[p[2] for p in wide["pairs"]]))
                got = {p[0]: p[2] for p in strict["pairs"]}
                for i in range(len(rt)):
                    hits.append(dict(track=TRACK[seq], seq=seq, sta=sta, phase=pha, i=i,
                                     weights=w, arm=a, hit=i in got, resid=got.get(i, np.nan)))
    df = pd.DataFrame(rows)
    out = []
    for key, g in df.groupby(["track", "weights", "phase", "arm"]):
        res = residual_stats([x for r in g.resid for x in r])
        out.append(dict(zip(["track", "weights", "phase", "arm"], key),
                        analyst=int(g.n_ref.sum()), emitted=int(g.emitted.sum()),
                        recall=g.matched.sum() / g.n_ref.sum(), mae_s=res["mae"],
                        bias_s=res["bias"], within_0_1=res["within_0.1"],
                        gross=res["gross_error_rate"]))
    return pd.DataFrame(out), pd.DataFrame(hits)


def paired(threshold: float) -> pd.DataFrame:
    """Each PR arm against vanilla, same weight, picks above `threshold`."""
    out = []
    keys = sorted({k[2:] for k in GROUPS})
    for w in WEIGHTS:
        for a in ARMS[1:]:
            for trk in ("us_sequences", "swarm_sequences"):
                n_v = n_c = same = 0
                shifts, dconf, lost_conf, new_conf = [], [], [], []
                for key in keys:
                    if TRACK.get(key[0], "us_sequences") != trk:
                        continue
                    vt, vq = GROUPS.get((w, "vanilla", *key), EMPTY)
                    ct, cq = GROUPS.get((w, a, *key), EMPTY)
                    vk, ck = vq >= threshold, cq >= threshold
                    vt, vq, ct, cq = vt[vk], vq[vk], ct[ck], cq[ck]
                    n_v += len(vt)
                    n_c += len(ct)
                    pairs, missed, extra = fast_match(vt, ct)
                    for i, j, d in pairs:
                        shifts.append(d)
                        dconf.append(cq[j] - vq[i])
                        same += abs(d) < 5e-4
                    lost_conf += list(vq[missed])
                    new_conf += list(cq[extra])
                s = np.round(np.asarray(shifts) * 100)  # samples at 100 Hz
                out.append(dict(weights=w, arm=a, track=trk, vanilla=n_v, arm_picks=n_c,
                                identical=same / n_v, paired=len(shifts) / n_v,
                                lost=len(lost_conf) / n_v, new=len(new_conf) / n_v,
                                shift_ge_1smp=np.mean(np.abs(s) >= 1),
                                shift_ge_5smp=np.mean(np.abs(s) >= 5),
                                shift_ge_10smp=np.mean(np.abs(s) >= 10),
                                p99_abs_shift_s=np.percentile(np.abs(shifts), 99),
                                mean_shift_ms=1e3 * np.mean(shifts),
                                p99_abs_dconf=np.percentile(np.abs(dconf), 99),
                                lost_median_conf=np.median(lost_conf) if lost_conf else np.nan,
                                lost_max_conf=max(lost_conf) if lost_conf else np.nan,
                                new_max_conf=max(new_conf) if new_conf else np.nan))
    return pd.DataFrame(out)


for thr in (0.2, 0.3):
    tab, hits = analyst(thr)
    tab.to_csv(OUT / f"analyst_{thr}.csv", index=False)
    hits.to_csv(OUT / f"hits_{thr}.csv", index=False)
    print(f"\n=== against analyst picks, threshold {thr}")
    print(tab.round(4).to_string(index=False))
    # Discordant analyst picks: recovered by one arm and not the other.
    h = hits.pivot_table(index=["track", "seq", "sta", "phase", "i", "weights"], columns="arm",
                         values="hit").reset_index()
    for _a in ARMS:
        h[_a] = h[_a].astype(bool)
    print("discordant (vanilla hit, bf16 miss) / (vanilla miss, bf16 hit):")
    print(h.groupby(["track", "weights", "phase"]).apply(
        lambda g: pd.Series(dict(v_only=int((g.vanilla & ~g.pr_bf16).sum()),
                                 bf16_only=int((~g.vanilla & g.pr_bf16).sum()),
                                 fp32_differs=int((g.vanilla != g.pr_fp32).sum())))).to_string())
    p = paired(thr)
    p.to_csv(OUT / f"paired_{thr}.csv", index=False)
    print(f"\n=== against vanilla, threshold {thr}")
    print(p.round(4).to_string(index=False))
