"""Summarize the Graviton runs: timing per rep, medians, $/station-day, exact pick agreement.

usage: ec2_summary.py RESULTS_DIR LAPTOP_OUT_DIR
RESULTS_DIR holds results_<tag>/out/ (extracted); LAPTOP_OUT_DIR holds bench_*_g*.json
from the M1 Max run (no pick dumps there; laptop vanilla picks are re-derived below).
"""

import glob
import json
import statistics as st
import sys
from pathlib import Path

R, LAP = Path(sys.argv[1]), Path(sys.argv[2])
PRICE = {"c6g": 0.272, "c7g": 0.289, "c8g": 0.31792}       # on-demand $/h, us-east-2, AWS Pricing API 2026-10-26
SPOT = {"c6g": 0.0802, "c7g": 0.0824, "c8g": 0.0772}        # cheapest-AZ spot $/h, same day
STATION_DAYS = 5.55


def reps(out: Path, arm: str):
    rows = []
    for r in (1, 2, 3, 4):
        d = [json.load(open(f)) for f in sorted(out.glob(f"bench_{arm}_{r}_g*.json"))
             if not f.name.endswith(".picks.json")]
        if len(d) != 4:
            continue
        rows.append(dict(rep=r, wall=max(x["total"] for x in d),
                         **{k: sum(x[k] for x in d) for k in ("classify", "wa", "vel", "read")},
                         picks=sum(x["picks"] for x in d), extra={k: d[0].get(k) for k in ("emulated", "fused_ops", "kernel")}))
    return rows


def med(rows, k):
    return st.median(x[k] for x in rows)


def picks(out: Path, arm: str, rep: int = 1) -> set:
    s = set()
    for f in sorted(out.glob(f"bench_{arm}_{rep}_g*.picks.json")):
        for sta, pha, t, q in json.load(open(f)):
            s.add((sta, pha, t[:23]))  # ms
    return s


lap = {a: reps(LAP, a) for a in ("vanilla", "pr")}
print(f"{'host':6s} {'arm':8s} {'wall med':>9s} {'walls':>28s} {'classify':>9s} {'wa':>7s} {'vel':>6s}")
for a in ("vanilla", "pr"):
    w = [round(x["wall"], 1) for x in lap[a]]
    print(f"{'M1Max':6s} {a:8s} {med(lap[a], 'wall'):9.1f} {str(w):>28s} {med(lap[a], 'classify'):9.1f} "
          f"{med(lap[a], 'wa'):7.1f} {med(lap[a], 'vel'):6.1f}")
summary = {}
for d in sorted(R.glob("results_*")):
    tag = d.name.split("_")[1]
    out = d / "out"
    for a in ("vanilla", "pr"):
        rows = reps(out, a)
        if not rows:
            print(tag, a, "no complete reps")
            continue
        w = [round(x["wall"], 1) for x in rows]
        summary[(tag, a)] = med(rows, "wall")
        print(f"{tag:6s} {a:8s} {med(rows, 'wall'):9.1f} {str(w):>28s} {med(rows, 'classify'):9.1f} "
              f"{med(rows, 'wa'):7.1f} {med(rows, 'vel'):6.1f}  {rows[0]['extra'] if a == 'pr' else ''}")
    v, p = picks(out, "vanilla"), picks(out, "pr")
    print(f"       exact picks (rep 1): vanilla {len(v)}, pr {len(p)}, common {len(v & p)}, "
          f"vanilla-only {len(v - p)}, pr-only {len(p - v)}")
    for r in (2, 3, 4):
        if picks(out, "pr", r) != p or picks(out, "vanilla", r) != v:
            print(f"       rep {r}: picks differ from rep 1 (non-deterministic)")
    summary[(tag, "vanilla_picks")] = v

print("\n$ per 1,000 station-days (wall x price / 5.55 station-days)")
for tag in ("c6g", "c7g", "c8g"):
    if (tag, "vanilla") in summary and (tag, "pr") in summary:
        for a in ("vanilla", "pr"):
            h = summary[(tag, a)] / 3600
            print(f"  {tag} {a:8s} on-demand ${1000 * h * PRICE[tag] / STATION_DAYS:6.3f}  spot ${1000 * h * SPOT[tag] / STATION_DAYS:6.3f}"
                  f"  speed-up {summary[(tag, 'vanilla')] / summary[(tag, 'pr')]:.2f}x")
tags = [t for t in ("c6g", "c7g", "c8g") if (t, "vanilla_picks") in summary]
for i, a in enumerate(tags):
    for b in tags[i + 1:]:
        x, y = summary[(a, "vanilla_picks")], summary[(b, "vanilla_picks")]
        print(f"vanilla {a} vs {b}: common {len(x & y)} of {len(x)} / {len(y)}")
