"""PR arms' perturbation of picks vs vanilla, on the scale of analyst pick uncertainty."""
import sys
import numpy as np, pandas as pd
S = sys.argv[1]
exec(open(f"{S}/scripts/score.py").read().split("def analyst")[0].replace("REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])",
     "REPO, OUT = Path('/Users/marinedenolle/GitHub/QuakeScope'), Path(S + '/out')"))
THR = 0.2
# model - analyst residual spread for vanilla (matched within 0.5 s), per phase
res = {"P": [], "S": []}
for (seq, sta, pha), rg in ref.groupby(["sequence", "station", "phase"]):
    t, q = GROUPS.get(("original", "vanilla", seq, sta, pha), EMPTY)
    pairs, _, _ = fast_match(sec(rg.time), t[q >= THR])
    res[pha] += [d for _, _, d in pairs]
for p in "PS":
    r = np.array(res[p]); print(f"vanilla - analyst {p}: n={len(r)} median={np.median(r)*1e3:.0f} ms std={r.std()*1e3:.0f} ms MAD-sigma={1.4826*np.median(np.abs(r-np.median(r)))*1e3:.0f} ms")
rows = []
for arm in ("pr_fp32", "pr_bf16"):
    for pha in "PS":
        sh, dq, lost, new = [], [], [], []
        for key in sorted({k[2:] for k in GROUPS if k[4] == pha}):
            vt, vq = GROUPS.get(("original", "vanilla", *key), EMPTY)
            ct, cq = GROUPS.get(("original", arm, *key), EMPTY)
            vk, ck = vq >= THR, cq >= THR
            pairs, mi, ex = fast_match(vt[vk], ct[ck])
            sh += [d for _, _, d in pairs]; dq += [cq[ck][j] - vq[vk][i] for i, j, _ in pairs]
            lost += list(vq[vk][mi]); new += list(cq[ck][ex])
        a = np.abs(np.array(sh)); n = len(sh) + len(lost)
        rows.append(dict(arm=arm, phase=pha, vanilla=n, std_shift_ms=np.std(sh)*1e3,
                         within_10ms=np.mean(a <= 0.0105), within_20ms=np.mean(a <= 0.0205),
                         within_50ms=np.mean(a <= 0.0505), max_shift_ms=a.max()*1e3,
                         max_dconf=np.max(np.abs(dq)), lost=len(lost), new=len(new),
                         lost_above_thr_plus_0p05=int(np.sum(np.array(lost) >= THR + 0.05)),
                         new_above_thr_plus_0p05=int(np.sum(np.array(new) >= THR + 0.05))))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(4).to_string(index=False))
