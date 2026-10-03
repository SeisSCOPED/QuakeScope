"""Run several published pickers over every benchmark window, once per window.

The windows are long and reading them is not free, so a stream is loaded once
and every model is applied to it before moving on. Each (model, window) result
is appended as it finishes, so a run that is interrupted resumes instead of
starting over -- this job takes hours and has been killed repeatedly.

  python scripts/score_tracks.py --bundle <dir> [--tracks t1,t2] [--models a,b]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import obspy                                                       # noqa: E402
import pandas as pd                                                # noqa: E402
import seisbench.models as sbm                                     # noqa: E402

# PhaseNet weight sets plus one other architecture. `diting` is the large
# Chinese corpus, `volpick` is trained on volcanic seismicity and is the one
# with a prior claim on the volcano-tectonic track, `stead` and `scedc` are the
# most-used public baselines, and quakescope2026 is ours.
MODELS = [
    ("PhaseNet", "original"), ("PhaseNet", "instance"), ("PhaseNet", "jma_wc"),
    ("PhaseNet", "quakescope2026"), ("PhaseNet", "diting"), ("PhaseNet", "stead"),
    ("PhaseNet", "volpick"), ("PhaseNet", "scedc"),
    ("EQTransformer", "original"), ("EQTransformer", "instance"),
]
FLOOR = 0.02        # keep everything above this; a threshold is a filter later


def load(arch: str, weight: str):
    return getattr(sbm, arch).from_pretrained(weight)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--tracks", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    bundle = Path(a.bundle)
    tracks = [t for t in (a.tracks.split(",") if a.tracks else
                          sorted(p.name for p in bundle.glob("track*"))) if t]
    want = set(a.models.split(",")) if a.models else None
    models = [(ar, w) for ar, w in MODELS if want is None or w in want or f"{ar}:{w}" in want]
    out = Path(a.out) if a.out else bundle.parent / "track_model_picks.csv"

    done: set[tuple] = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.track, prev.model, prev.file))
        print(f"resuming: {len(prev):,} picks, {len(done):,} (model, window) pairs done",
              flush=True)

    loaded: dict[tuple, object] = {}
    for track in tracks:
        wfs = sorted((bundle / track / "waveforms").glob("*.mseed"))
        print(f"\n{track}: {len(wfs)} windows x {len(models)} models", flush=True)
        for wf in wfs:
            todo = [(ar, w) for ar, w in models if (track, f"{ar}:{w}", wf.name) not in done]
            if not todo:
                continue
            try:
                st = obspy.read(str(wf))
            except Exception as exc:                               # noqa: BLE001
                print(f"  {wf.name}: unreadable, {type(exc).__name__}", flush=True)
                continue
            rows = []
            for ar, w in todo:
                key = (ar, w)
                if key not in loaded:
                    loaded[key] = load(ar, w)
                try:
                    ann = loaded[key].classify(st).picks
                except Exception as exc:                           # noqa: BLE001
                    print(f"  {wf.name} {ar}:{w}: {type(exc).__name__}", flush=True)
                    continue
                for p in ann:
                    if p.peak_value is None or p.peak_value < FLOOR:
                        continue
                    rows.append(dict(track=track, model=f"{ar}:{w}", file=wf.name,
                                     station=p.trace_id.rsplit(".", 1)[0] if p.trace_id.count(".") > 1
                                             else p.trace_id,
                                     phase=str(p.phase).upper()[0],
                                     time=p.peak_time.datetime,
                                     conf=round(float(p.peak_value), 5)))
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"  {wf.name:52s} {len(rows):6,} picks from {len(todo)} models", flush=True)

    if out.exists():
        d = pd.read_csv(out)
        print(f"\n{len(d):,} picks  ->  {out}")
        print(d.groupby(["track", "model"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
