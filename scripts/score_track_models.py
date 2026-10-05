"""Score every picker against every track's reference arrivals.

Reads the raw picks score_tracks.py produced (one row per pick, at a 0.02
floor) and the bundle's reference_picks.csv, and writes the metric tables the
board reads. Three protocols, because no single one orders the models: a shared
threshold, each model's own best threshold, and an equal pick count.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sb_catalog.src.benchmark_metrics import (                     # noqa: E402
    ece, match_picks, phase_confusion, residual_stats,
)

SHARED_THR = 0.3
TOL = 0.5
RES_TOL = 2.0
THRESHOLDS = [round(x, 3) for x in np.arange(0.02, 0.96, 0.02)]

# Each weight set ships a threshold its authors chose, and they disagree by two
# orders of magnitude: EQTransformer instance 0.005, volpick 0.39, scedc 0.41.
# Scoring everything at a shared 0.3 is not what any of their authors intended,
# so recall at the published threshold is reported alongside.
PUBLISHED = {}
try:
    _pt = pd.read_csv(ROOT / "docs" / "benchmark" / "results" / "tracks" /
                      "published_thresholds.csv")
    PUBLISHED = {r.model: {"P": r.P_thr, "S": r.S_thr} for r in _pt.itertuples()}
except Exception:                                                  # noqa: BLE001
    pass


def sweep_stations(rt: dict, pt: dict, pc: dict, thresholds, tol: float) -> pd.DataFrame:
    """Recall and picks emitted across every station, at each threshold.

    benchmark_metrics.sweep works on one station's arrays. A pick only answers
    for its own station, so matching has to stay per station and only the
    counts are pooled.
    """
    n_ref = sum(len(v) for v in rt.values())
    rows = []
    for th in thresholds:
        n_m = n_e = 0
        for s in set(rt) | set(pt):
            c = pc.get(s, np.array([]))
            keep = c >= th
            cand = pt.get(s, np.array([]))[keep]
            n_e += len(cand)
            n_m += len(match_picks(rt.get(s, np.array([])), cand,
                                   c[keep], tol=tol)["pairs"])
        rec = n_m / n_ref if n_ref else np.nan
        prec = n_m / n_e if n_e else np.nan
        rows.append(dict(threshold=th, emitted=n_e, matched=n_m, recall=rec,
                         precision_lb=prec,
                         f1_lb=(2 * rec * prec / (rec + prec)) if rec and prec else np.nan))
    return pd.DataFrame(rows)


def file_to_sequence(bundle: Path) -> dict[str, str]:
    """Which sequence each waveform file belongs to.

    stations.csv carries it where the swarm builder wrote the track; elsewhere
    the filename prefix is the sequence key, so map that and assert nothing is
    left unresolved rather than silently scoring a file against the wrong
    reference.
    """
    m: dict[str, str] = {}
    for t in sorted(p.name for p in bundle.glob("track*")):
        st = bundle / t / "stations.csv"
        if st.exists():
            d = pd.read_csv(st)
            if {"file", "sequence"} <= set(d.columns):
                m.update(dict(zip(d.file, d.sequence)))
    PREFIX = {
        "ridgecrest": "Ridgecrest", "sansimeon": "San Simeon",
        "montecristo": "Monte Cristo", "mendocino2024": "Mendocino 2024",
        "monroewa": "Monroe WA", "kaikoura_2016": "Kaikoura 2016",
        "norcia_2016": "Norcia 2016", "thessaly_2021": "Thessaly 2021",
        "etna_edifice": "Etna edifice 2022-2024",
        "west_bohemia_2018": "West Bohemia 2018", "maurienne_2017": "Maurienne 2017-2019",
        "salton_sea_2016": "Salton Sea 2016", "jones_guthrie_2014": "Jones-Guthrie 2014-2015",
    }
    for t in sorted(p.name for p in bundle.glob("track*")):
        for wf in (bundle / t / "waveforms").glob("*.mseed"):
            if wf.name in m:
                continue
            for pre, seq in PREFIX.items():
                if wf.name.startswith(pre):
                    m[wf.name] = seq
                    break
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs" / "benchmark" / "results" / "tracks"))
    a = ap.parse_args()
    bundle, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pk = pd.read_csv(a.picks)
    pk["time"] = pd.to_datetime(pk.time, utc=True, format="mixed").dt.tz_localize(None)
    f2s = file_to_sequence(bundle)
    pk["sequence"] = pk.file.map(f2s)
    unresolved = pk[pk.sequence.isna()].file.nunique()
    if unresolved:
        raise SystemExit(f"{unresolved} waveform files could not be mapped to a sequence")
    print(f"{len(pk):,} picks, {pk.model.nunique()} models, {pk.sequence.nunique()} sequences")

    ref = []
    for t in sorted(p.name for p in bundle.glob("track*")):
        d = pd.read_csv(bundle / t / "reference_picks.csv")
        d["track"] = t
        ref.append(d)
    ref = pd.concat(ref, ignore_index=True)
    ref["time"] = pd.to_datetime(ref.time, format="mixed")

    det, tim, cal, qual, curves = [], [], [], [], {}
    for (track, seq, phase), r in ref.groupby(["track", "sequence", "phase"]):
        rt = {s: np.sort((g.time.values.astype("int64") / 1e9)) for s, g in r.groupby("station")}
        n_ref = len(r)
        for model, p in pk[(pk.track == track) & (pk.sequence == seq)
                           & (pk.phase == phase)].groupby("model"):
            pt, pc = {}, {}
            for s, g in p.groupby("station"):
                g = g.sort_values("time")
                pt[s] = g.time.values.astype("int64") / 1e9
                pc[s] = g.conf.values
            cur = sweep_stations(rt, pt, pc, THRESHOLDS, tol=TOL)
            curves[(track, seq, phase, model)] = cur
            bt = (cur.loc[cur.f1_lb.idxmax()].to_dict()
                  if cur.f1_lb.notna().any() else {})

            res, hits, confs, n_m, n_x, n_e = [], [], [], 0, 0, 0
            for s in sorted(set(rt) | set(pt)):
                keep = pc.get(s, np.array([])) >= SHARED_THR
                cand = pt.get(s, np.array([]))[keep]
                cf = pc.get(s, np.array([]))[keep]
                mm = match_picks(rt.get(s, np.array([])), cand, cf, tol=RES_TOL)
                n_e += len(cand)
                for _i, _j, dt, _c in mm["pairs"]:
                    res.append(dt)
                m2 = match_picks(rt.get(s, np.array([])), cand, cf, tol=TOL)
                n_m += len(m2["pairs"])
                n_x += len(m2["extra"])
                paired = {j for _, j, _, _ in m2["pairs"]}
                for k, c in enumerate(cf):
                    confs.append(c)
                    hits.append(1 if k in paired else 0)
            # the same count again at the model's own published threshold
            pub = PUBLISHED.get(model, {}).get(phase)
            pub_rec = pub_emit = np.nan
            if pub is not None and not pd.isna(pub):
                _m = _e = 0
                for s_ in sorted(set(rt) | set(pt)):
                    k = pc.get(s_, np.array([])) >= pub
                    cd = pt.get(s_, np.array([]))[k]
                    _e += len(cd)
                    _m += len(match_picks(rt.get(s_, np.array([])), cd,
                                          pc.get(s_, np.array([]))[k], tol=TOL)["pairs"])
                pub_rec = _m / n_ref if n_ref else np.nan
                pub_emit = _e

            rec = n_m / n_ref if n_ref else np.nan
            prec = n_m / n_e if n_e else np.nan
            det.append(dict(track=track, sequence=seq, phase=phase, model=model,
                            n_ref=n_ref, emitted_at_03=n_e, recall_at_03=rec,
                            precision_lb_at_03=prec,
                            f1_lb_at_03=(2 * rec * prec / (rec + prec)) if rec and prec else np.nan,
                            best_thr=bt.get("threshold"), recall_at_best=bt.get("recall"),
                            f1_lb_at_best=bt.get("f1_lb"),
                            published_thr=pub, recall_at_published=pub_rec,
                            emitted_at_published=pub_emit))
            if res:
                st = residual_stats(np.array(res), strict_tol=TOL)
                tim.append(dict(track=track, sequence=seq, phase=phase, model=model,
                                n=len(res), **{k: st[k] for k in
                                ("mae", "rmse", "medae", "bias", "p10", "p90",
                                 "gross_error_rate", "within_0.1", "within_0.25", "within_0.5")}))
            if confs:
                cal.append(dict(track=track, sequence=seq, phase=phase, model=model,
                                n=len(confs), ece_lb=ece(np.array(confs), np.array(hits))))

    # phase swaps and duplicates, per track and model
    for track, r in ref.groupby("track"):
        for model, p in pk[pk.track == track].groupby("model"):
            tot_sw = tot_n = tot_dup = tot_pk = 0
            for seq in sorted(set(r.sequence)):
                rr = r[r.sequence == seq]
                pp = p[(p.sequence == seq) & (p.conf >= SHARED_THR)]
                # phase_confusion takes {phase: times} for one station: a pick
                # only answers for the station it was made on, so swaps are
                # counted per station and pooled.
                for sta in sorted(set(rr.station) | set(pp.station)):
                    R = {ph: np.sort(g.time.values.astype("int64") / 1e9)
                         for ph, g in rr[rr.station == sta].groupby("phase")}
                    P = {ph: np.sort(g.time.values.astype("int64") / 1e9)
                         for ph, g in pp[pp.station == sta].groupby("phase")}
                    if not R:
                        continue
                    cf = phase_confusion(R, P, tol=TOL)
                    tot_sw += cf.get("P_as_S", 0) + cf.get("S_as_P", 0)
                    tot_n += sum(len(v) for v in R.values())
                tot_pk += len(pp)
            qual.append(dict(track=track, model=model, n_ref=tot_n, emitted=tot_pk,
                             swaps=tot_sw, swap_rate=tot_sw / max(tot_n, 1)))

    pd.DataFrame(det).to_csv(out / "detection.csv", index=False)
    pd.DataFrame(tim).to_csv(out / "timing.csv", index=False)
    pd.DataFrame(cal).to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(qual).to_csv(out / "phase_quality.csv", index=False)

    # equal pick count: the budget every model can reach on this sequence-phase
    rows = []
    keys = sorted({k[:3] for k in curves})
    for k in keys:
        cs = {m: c for (t, s, p, m), c in curves.items() if (t, s, p) == k}
        if len(cs) < 2:
            continue
        lo = max(c.emitted.min() for c in cs.values())
        hi = min(c.emitted.max() for c in cs.values())
        if hi <= lo:
            continue
        b = int(round((lo + hi) / 2))
        for m, c in cs.items():
            o = c.sort_values("emitted")
            rows.append(dict(track=k[0], sequence=k[1], phase=k[2], model=m, budget=b,
                             recall_at_budget=float(np.interp(b, o.emitted, o.recall)),
                             ceiling=int(c.emitted.max())))
    pd.DataFrame(rows).to_csv(out / "equal_count.csv", index=False)
    print(f"  detection {len(det)}  timing {len(tim)}  calibration {len(cal)}  "
          f"quality {len(qual)}  equal-count {len(rows)}  -> {out}")


if __name__ == "__main__":
    main()
