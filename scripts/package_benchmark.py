#!/usr/bin/env python
"""Assemble a downloadable evaluation set: waveforms, arrivals, scorer, task spec.

Built for someone who does not work in seismology. The task is stated in one
JSON file, the waveforms are MiniSEED, the arrivals to recover are one CSV, and
the scorer that produced our published numbers is in the bundle rather than
somewhere else. A baseline result is included so a newcomer can tell whether
their run is working before they trust it.

    pixi run -e dev python scripts/package_benchmark.py --out dist/
    pixi run -e dev python scripts/package_benchmark.py --out dist/ --push --private

The push needs a Hugging Face login with write access to the organisation:
`hf auth login`, or HF_TOKEN in the environment. Nothing in this script asks for
a token or stores one.

What is deliberately NOT here: the held-out split. See HOLDOUT in this file and
the datasheet it writes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import obspy
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "docs" / "benchmark" / "results"
BUCKET = "quakescope-picks-2026"
REGION = "us-east-2"
HTTP = f"https://{BUCKET}.s3.{REGION}.amazonaws.com"

SCRATCH = Path("/private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/8f282710-68a1-4cdd-b77e-e07856ef31af/scratchpad")

SEQ_LABEL = {                      # held-out registry key -> published label
    "kaikoura_2016": "Kaikoura 2016", "norcia_2016": "Norcia 2016",
    "thessaly_2021": "Thessaly 2021", "adriatic_2022": "Adriatic 2022",
    "etna_2022_2024": "Etna 2022-2024",
}

RETRAIN = Path("/Users/marinedenolle/GitHub/phasenet-retrain/data/heldout_testset")

TRACKS = {
    "track1-western-us": dict(
        title="Track 1: western United States",
        regime="mainshock-aftershock and moderate events",
        question="Does the picker serve the catalogue this project builds?",
        src=RES / "us_sequences",
        blurb="Five sequences in the region the QuakeScope campaign catalogues, "
              "chosen to vary network, magnitude and recording era. Reference "
              "arrivals are manual picks published through ANSS by SCEDC, NCEDC "
              "and PNSN.",
        extra_refs=[("Monroe WA", SCRATCH / "monroe_picks.csv")],
    ),
    "track2-msas": dict(
        title="Track 2a: mainshock-aftershock, outside the United States",
        regime="mainshock-aftershock",
        question="Does it hold up on another continent's network and analysts?",
        src=RES / "global_sequences",
        blurb="Three large sequences with the operating network's own reviewed "
              "arrivals: GeoNet, INGV and NOA. Events seconds apart, overlapping "
              "codas, a network that saturates in the first hours.",
        retrain=["kaikoura_2016", "norcia_2016", "thessaly_2021"],
    ),
    "track2-vt": dict(
        title="Track 2b: volcano-tectonic",
        regime="volcano-tectonic",
        question="Does it pick emergent, low-magnitude, volcanic seismicity?",
        src=RES / "swarm_sequences",
        only=["Etna edifice 2022-2024"],
        blurb="Etna 2022-2024 on the volcanic edifice, from the INGV catalogue: "
              "within 8 km of the summit craters and shallower than 8 km. "
              "Emergent onsets, low magnitudes and event types a picker trained "
              "on tectonic earthquakes has not seen. Held out as a place rather "
              "than a time window, because a volcano recurs where it is and the "
              "curated corpora hold its earlier years. An earlier build of this "
              "track took the busiest windows over the whole region and caught "
              "only basement and regional Sicilian earthquakes, median 22 km "
              "offset and 21.6 km depth with no edifice event in it; this one "
              "has median 4.6 km offset and 5.3 km depth.",
    ),
    "track2-swarm": dict(
        title="Track 2c: fluid-driven swarm",
        regime="fluid-driven swarm",
        exclude=["Etna edifice 2022-2024"],
        question="Does it pick a swarm that migrates for months with no mainshock?",
        src=RES / "swarm_sequences",
        blurb="Four swarms, two driven by crustal fluids and two by wastewater "
              "injection. Months of elevated rate, no mainshock, shallow sources "
              "on dense local arrays. Reference arrivals from the ISC bulletin, "
              "the USGS phase-data product and BCSF-RENASS.",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_spec(src: Path, spec: dict) -> dict:
    """The task in a form a program can read: inputs, target, metric, tolerance."""
    ref = pd.read_csv(src / "reference_picks.csv")
    sta = pd.read_csv(src / "stations.csv")
    win = pd.read_csv(src / "windows.csv")
    return {
        "name": f"quakescope-{spec['regime'].replace(' ', '-')}-v1",
        "title": spec["title"],
        "task": ("Given three-component seismic waveforms, output the arrival time of "
                 "every P and S phase, each with a confidence in [0, 1]."),
        "inputs": {"waveforms": "waveforms/*.mseed",
                   "format": "MiniSEED, 3 components per station-window",
                   "stations": "stations.csv"},
        "target": {"file": "reference_picks.csv",
                   "columns": ["sequence", "station", "phase", "time"],
                   "what": "arrival times a human analyst picked and an operator published"},
        "submission": {"columns": ["sequence", "station", "phase", "time", "conf"],
                       "note": "Run once at a low confidence floor and keep every pick. "
                               "Each threshold is then a filter over one file rather than "
                               "another pass over the waveforms."},
        "scoring": {"entrypoint": "score/score_picks.py",
                    "command": "python score/score_picks.py --reference reference_picks.csv "
                               "--picks your_picks.csv --out scores/",
                    "detection_tolerance_s": 0.5,
                    "residual_tolerance_s": 2.0,
                    "matching": "greedy nearest, one-to-one, per station and phase",
                    "primary_metric": "recall",
                    "protocols": ["one shared threshold",
                                  "equal pick count across models",
                                  "each model at its own best threshold"]},
        "identifiability": {
            "exact": ["recall", "picks emitted", "MAE", "RMSE", "MedianAE", "median bias",
                      "gross-error rate", "phase swap rate", "duplicate rate"],
            "lower_bound": ["precision", "F1", "calibration", "ECE"],
            "not_computable": ["MCC"],
            "why": ("The reference is an operator bulletin, not a labelled test set. An "
                    "analyst picked what a location needed and stopped, so a model pick "
                    "with no analyst counterpart may be a false positive or a real arrival "
                    "nobody marked. Anything needing a false-positive count is a bound, and "
                    "MCC needs true negatives that a continuous record does not define."),
        },
        "counts": {"sequences": int(ref.sequence.nunique()),
                   "stations": int(sta.station.nunique()),
                   "windows": int(len(win)),
                   "reference_arrivals": int(len(ref)),
                   "P": int((ref.phase == "P").sum()),
                   "S": int((ref.phase == "S").sum())},
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
    }


DATASHEET = """# {title}

{blurb}

## What is in here

| file | what |
|---|---|
| `task.json` | the task, the scoring rule and the tolerances, machine-readable |
| `reference_picks.csv` | the arrivals to recover: `sequence, station, phase, time` |
| `model_picks.csv` | what four published PhaseNet weight sets emitted, for reproducing our numbers |
| `stations.csv` | station, network, band, window, and how many arrivals it carries |
| `windows.csv` | the scoring windows, chosen as the busiest in each sequence |
| `sequences.csv` | the four sequences with coordinates, period and reference source |
| `waveforms/` | MiniSEED, three components per station-window |
| `score/` | the scorer that produced our published numbers, and its metric definitions |
| `baseline.csv` | what the four weight sets score, so you can tell a working run from a broken one |

## The reference is not a labelled test set

This is the single thing to understand before reading any number from it.

The arrivals come from operator bulletins and published analyst catalogues. An
analyst picked what a location needed and then stopped. In a swarm with
hundreds of events a day, most real arrivals are never marked. So a model pick
with no analyst counterpart is a mixture of a false positive and a real arrival
nobody had time to write down.

Recall is therefore exact and is the primary metric. Precision, F1 and
calibration are **lower bounds**, comparable between models on this reference
and not comparable with a number from a labelled-dataset paper. MCC is not
computable at all, because a continuous record with an incomplete reference does
not define true negatives. `task.json` carries this as a field so a program can
respect it.

## How the set was built

{how}

## Known limitations

- **Sample sizes differ** between sequences by a factor of {spread:.0f}. Weight by
  arrivals or report per sequence; do not average the four.
- **The reference is incomplete by construction**, as above.
- **Contamination is stated, not measured.** None of these four sequences is in
  the curated corpora we know about, but we have not diffed them trace by trace
  against every training set a submitter might use.
- **No association step.** Picks are scored as picks. Turning extra detections
  into a precision estimate needs an associator, which is not in this bundle.
- **These are development sequences.** See the holdout note below.

## Held-out evaluation

{holdout}

## Licence and provenance

Waveforms are redistributed from open archives and remain the property of their
operators: {operators}. Reference arrivals come from {sources}. Both are
redistributed here under the terms those services publish, with no additional
restriction. The code in `score/` is MIT, as the rest of
[SeisSCOPED/QuakeScope](https://github.com/SeisSCOPED/QuakeScope).

Cite the operators when you use the waveforms. If you report a number from this
set, state the sequence, the protocol and the tolerance, because a recall
without those three is not reproducible.

Built {built} from `{commit}`.
"""

HOLDOUT = """**This bundle is a development set. It is not the whole evaluation.**

A benchmark whose every sequence is public stops measuring generalisation the
moment people tune on it. In seismology you cannot hide the underlying data —
the bulletins are public and anyone can re-harvest them — so what is withheld
here is the **selection**, which is the part that actually carries the
evaluation: which sequences, which time windows, which stations, and the
reference set assembled for them.

Twelve further sequences across three regimes (mainshock-aftershock,
volcano-tectonic, fluid-driven swarm) are held back and scored only on
submission. Their existence, their regimes and the scoring rule are public; the
selection is not. That is the only way a score on this board can mean
"generalises" rather than "was tuned on".

Use this bundle to build and debug. Submit to be scored on the rest.
"""


def assemble_track(key: str, spec: dict, out: Path) -> dict:
    """One track's directory: arrivals, waveforms, stations, task spec."""
    d = out / key
    (d / "waveforms").mkdir(parents=True, exist_ok=True)
    refs, stations, n_wf = [], [], 0

    if spec.get("src"):
        src = spec["src"]
        # Two tracks can share one build directory: the swarm builder also
        # produces the Etna edifice sequence, which belongs to the volcano
        # -tectonic track. `only` keeps those sequences, `exclude` drops them,
        # and both have to reach the station table and the waveform files or a
        # track ships records it does not score.
        def _sel(df):
            if "sequence" not in df.columns:
                return df
            if spec.get("only"):
                return df[df.sequence.isin(spec["only"])]
            if spec.get("exclude"):
                return df[~df.sequence.isin(spec["exclude"])]
            return df

        if (src / "reference_picks.csv").exists():
            refs.append(_sel(pd.read_csv(src / "reference_picks.csv")))
        for f in ("model_picks.csv", "windows.csv", "provenance.csv"):
            if (src / f).exists():
                _sel(pd.read_csv(src / f)).to_csv(
                    d / (f if f != "provenance.csv" else "sequences.csv"), index=False)
        keys = None
        if (src / "stations.csv").exists():
            st = _sel(pd.read_csv(src / "stations.csv"))
            stations.append(st)
            if "file" in st.columns:
                keys = set(st.file)
        for wf in sorted((src / "waveforms").glob("*.mseed")) if (src / "waveforms").exists() else []:
            if keys is not None and wf.name not in keys:
                continue
            shutil.copy2(wf, d / "waveforms" / wf.name)
            n_wf += 1

    for seq, path in spec.get("extra_refs", []):
        if Path(path).exists():
            e = pd.read_csv(path)
            refs.append(e[["sequence", "station", "phase", "time"]])

    # sequences whose waveforms and picks were built in the held-out set
    for key_r in spec.get("retrain", []):
        rd = RETRAIN / key_r
        if not rd.exists():
            print(f"    {key_r}: not built, skipped")
            continue
        pk = rd / "picks.parquet"
        if pk.exists():
            t = pd.read_parquet(pk)
            cols = {c.lower(): c for c in t.columns}
            # Keep analyst work only. The QuakeScope harvest filters on
            # evaluation_mode before it writes reference_picks.csv; this branch
            # reads the held-out parquet directly and has to do it here. Without
            # it, 1,884 automatic AIC picks from Kaikoura reach the reference
            # under a card that says nothing here was produced by a model.
            if "mode" in cols:
                before = len(t)
                t = t[t[cols["mode"]].astype(str).str.lower() == "manual"]
                if before != len(t):
                    print(f"    {key_r}: dropped {before - len(t):,} picks whose "
                          f"evaluation_mode is not manual")
            lab = t[cols.get("label", "label")] if "label" in cols else key_r
            frame = pd.DataFrame({
                "sequence": lab if not isinstance(lab, str) else key_r,
                "station": t[cols["station"]] if "station" in cols else t.get("trace_id"),
                "phase": t[cols["phase"]].astype(str).str[0].str.upper() if "phase" in cols else None,
                "time": t[cols["time"]] if "time" in cols else t.get("pick_time")})
            refs.append(frame.dropna(subset=["station", "phase", "time"]))
        for wf in sorted((rd / "waveforms").rglob("*")):
            if wf.is_file():
                shutil.copy2(wf, d / "waveforms" / f"{key_r}_{wf.name}")
                n_wf += 1

    ref = pd.concat(refs, ignore_index=True)[["sequence", "station", "phase", "time"]] if refs \
        else pd.DataFrame(columns=["sequence", "station", "phase", "time"])

    # The same sequence reaches a track under two labels: QuakeScope writes
    # "Kaikoura 2016" and the held-out build writes its registry key
    # "kaikoura_2016". Deduplicating on the tuple as-is keeps both and doubles
    # the reference, which is what happened in the first push. Normalise the
    # label first, then collapse on station, phase and time to a tenth of a
    # second, which is well inside the 0.5 s matching tolerance.
    # One timestamp convention across every file: naive UTC with microseconds.
    # Sources hand back a mix of tz-aware and naive, and with and without a
    # fractional part, which made two of the four files unreadable by a plain
    # pd.to_datetime and only the shipped scorer (format="mixed") could load them.
    ref["time"] = (pd.to_datetime(ref.time, utc=True, format="mixed")
                     .dt.tz_localize(None)
                     .dt.strftime("%Y-%m-%d %H:%M:%S.%f"))
    ref["sequence"] = ref.sequence.map(lambda v: SEQ_LABEL.get(str(v), str(v)))
    before = len(ref)
    t = pd.to_datetime(ref.time, utc=True, format="mixed").dt.tz_localize(None)
    ref = (ref.assign(_k=t.dt.round("100ms"))
              .sort_values(["sequence", "station", "phase", "_k"])
              .drop_duplicates(["sequence", "station", "phase", "_k"])
              .drop(columns=["_k"]))
    if before != len(ref):
        print(f"    {key}: collapsed {before - len(ref):,} duplicate arrivals reported under "
              f"more than one label")

    # Only keep arrivals on stations whose waveforms are in the bundle. A
    # reference listing 714 stations against 18 waveform files is not a task
    # anyone can attempt; the rest of the bulletin is context, not target.
    # An arrival is only a target if the bundle carries the record it is on, in
    # the window it falls in. Filtering on station alone is not enough: the
    # Monroe reference spans a week of aftershocks while its waveforms cover two
    # hours, and scoring it unfiltered gave every model a recall of zero.
    spans = {}
    for wf in (d / "waveforms").glob("*.mseed"):
        try:
            for tr in obspy.read(str(wf), headonly=True):
                spans.setdefault(f"{tr.stats.network}.{tr.stats.station}", []).append(
                    (pd.Timestamp(tr.stats.starttime.datetime),
                     pd.Timestamp(tr.stats.endtime.datetime)))
        except Exception:                                          # noqa: BLE001
            continue
    if spans:
        before = len(ref)
        # some sources carry a timezone and some do not; compare in UTC
        t = pd.to_datetime(ref.time, utc=True, format="mixed").dt.tz_localize(None)
        keep = pd.Series(False, index=ref.index)
        for station, windows in spans.items():
            on = ref.station == station
            for a, b in windows:
                keep |= on & (t >= a) & (t <= b)
        ref = ref[keep]
        off_station = int((~ref.index.isin(ref.index)).sum())
        print(f"    {key}: {before:,} arrivals -> {len(ref):,} that fall on a station and "
              f"inside a window the bundle carries")
    ref.to_csv(d / "reference_picks.csv", index=False)
    if stations:
        pd.concat(stations, ignore_index=True).to_csv(d / "stations.csv", index=False)

    task = {
        "track": key, "title": spec["title"], "regime": spec["regime"],
        "question": spec["question"],
        "task": ("Given three-component seismic waveforms, output the arrival time of every "
                 "P and S phase, each with a confidence in [0, 1]."),
        "inputs": {"waveforms": f"{key}/waveforms/*.mseed",
                   "format": "MiniSEED, three components per station-window",
                   "stations": f"{key}/stations.csv"},
        "target": {"file": f"{key}/reference_picks.csv",
                   "columns": ["sequence", "station", "phase", "time"],
                   "what": "arrival times a human analyst picked and an operator published"},
        "submission": {"columns": ["sequence", "station", "phase", "time", "conf"],
                       "note": "Run once at a low confidence floor and keep every pick, so "
                               "each threshold is a filter over one file."},
        "scoring": {"entrypoint": "score/score_picks.py",
                    "command": f"python score/score_picks.py --reference {key}/reference_picks.csv "
                               f"--picks your_picks.csv --out scores/",
                    "detection_tolerance_s": 0.5, "residual_tolerance_s": 2.0,
                    "matching": "greedy nearest, one-to-one, per station and phase",
                    "primary_metric": "recall",
                    "protocols": ["one shared threshold", "equal pick count",
                                  "each model at its own best threshold"]},
        "identifiability": IDENTIFIABILITY,
        "counts": {"sequences": int(ref.sequence.nunique()), "arrivals": int(len(ref)),
                   "P": int((ref.phase == "P").sum()), "S": int((ref.phase == "S").sum()),
                   "stations": int(ref.station.nunique()), "waveform_files": n_wf},
    }
    (d / "task.json").write_text(json.dumps(task, indent=1))
    print(f"  {key:20s} {len(ref):6,} arrivals  P {task['counts']['P']:5,} S "
          f"{task['counts']['S']:5,}  {task['counts']['stations']:3d} stations  "
          f"{n_wf:3d} waveform files")
    return task


IDENTIFIABILITY = {
    "exact": ["recall", "picks_emitted", "MAE", "RMSE", "MedianAE", "median_bias",
              "gross_error_rate", "phase_swap_rate", "duplicate_rate"],
    "lower_bound": ["precision", "F1", "calibration", "ECE"],
    "not_computable": ["MCC"],
    "why": ("The reference is an operator bulletin, not a labelled test set. An analyst "
            "picked what a location needed and stopped, so a model pick with no analyst "
            "counterpart may be a false positive or a real arrival nobody marked. Anything "
            "needing a false-positive count is a bound; MCC needs true negatives that a "
            "continuous record with an incomplete reference does not define."),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--push", action="store_true", help="upload to Hugging Face")
    ap.add_argument("--repo", default="gaia-hazlab/quakescope-eval-v1")
    ap.add_argument("--private", action="store_true",
                    help="create the dataset private; make it public when you are ready")
    a = ap.parse_args()

    name = "quakescope-eval-v1"
    out = Path(a.out) / name
    if out.exists():
        shutil.rmtree(out)
    (out / "score").mkdir(parents=True)

    tasks = [assemble_track(k, v, out) for k, v in TRACKS.items()]

    shutil.copy2(ROOT / "sb_catalog" / "src" / "benchmark_metrics.py", out / "score")
    shutil.copy2(ROOT / "scripts" / "score_picks.py", out / "score")
    (out / "score" / "requirements.txt").write_text("numpy\npandas\n")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "tasks.json").write_text(json.dumps(
        {"name": name, "built": built, "commit": commit,
         "scorer": "score/score_picks.py", "identifiability": IDENTIFIABILITY,
         "tracks": tasks}, indent=1))
    prov = pd.read_csv(RES / "label_provenance.csv")
    nets = json.loads((RES / "network_doi.json").read_text())
    # the card must describe what the bundle actually contains, so keep only the
    # sequences and networks that made it through station and window filtering
    seqs = set()
    for t in tasks:
        r = pd.read_csv(out / t["track"] / "reference_picks.csv")
        seqs |= set(r.sequence.unique())
    missing = seqs - set(prov.sequence)
    if missing:
        raise SystemExit(f"no recorded label provenance for: {sorted(missing)}")
    prov = prov[prov.sequence.isin(seqs)]
    (out / "README.md").write_text(readme(tasks, built, commit, prov, nets))
    prov.to_csv(out / "label_provenance.csv", index=False)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    manifest = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
                 "sha256": sha256(p)} for p in files]
    total = sum(m["bytes"] for m in manifest)
    (out / "MANIFEST.json").write_text(json.dumps(
        {"name": name, "files": len(manifest), "bytes": total, "entries": manifest}, indent=1))
    print(f"\n{out}: {len(manifest):,} files, {total / 1e6:.0f} MB")

    if not a.push:
        print("  not pushed. Pass --push to upload to Hugging Face.")
        print(f"  repo would be: https://huggingface.co/datasets/{a.repo}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        who = api.whoami()
    except Exception:                                              # noqa: BLE001
        raise SystemExit(
            "Not logged in to Hugging Face. Run `hf auth login` (or set HF_TOKEN) with an\n"
            "account that can write to the target organisation, then run this again.")
    print(f"  logged in as {who.get('name')}")
    api.create_repo(a.repo, repo_type="dataset", exist_ok=True, private=a.private)
    api.upload_folder(folder_path=str(out), repo_id=a.repo, repo_type="dataset",
                      commit_message=f"QuakeScope evaluation set v1, built from {commit}")
    print(f"  https://huggingface.co/datasets/{a.repo}")


def readme(tasks: list, built: str, commit: str, prov, nets: dict) -> str:
    """A Hugging Face dataset card: YAML frontmatter, then the prose.

    The `configs` block makes every track's arrivals loadable with
    `load_dataset(repo, "track2-swarm")`, which is the two-line start a
    newcomer needs. The waveforms stay as MiniSEED files alongside, because
    there is no honest way to put a three-component seismogram in a table.
    """
    cfgs = "\n".join(
        f"  - config_name: {t['track']}\n"
        f"    data_files:\n"
        f"      - split: reference\n"
        f"        path: {t['track']}/reference_picks.csv" for t in tasks)
    rows = "\n".join(
        f"| `{t['track']}` | {t['regime']} | {t['counts']['sequences']} | "
        f"{t['counts']['arrivals']:,} | {t['counts']['P']:,} | {t['counts']['S']:,} | "
        f"{t['counts']['waveform_files']} |" for t in tasks)
    total = sum(t["counts"]["arrivals"] for t in tasks)
    prov_rows = "\n".join(
        f"| {r.sequence} | {r.agency_long} | {r.service} | {r.mode_filter} | "
        f"[{r.doi_label}](https://doi.org/{r.doi}) |" for r in prov.itertuples())
    net_rows = "\n".join(
        f"| `{c}` | {i['n']:,} | {i['years']} | {i['name'][:46]} | "
        + (f"[{i['doi']}](https://doi.org/{i['doi']}) |" if i.get("doi") else "not registered |")
        for c, i in sorted(nets.items()))
    return f"""---
pretty_name: QuakeScope phase-picking evaluation set
license: cc-by-4.0
language:
  - en
tags:
  - seismology
  - earthquake
  - phase-picking
  - time-series
  - benchmark
  - geoscience
task_categories:
  - time-series-forecasting
size_categories:
  - 10K<n<100K
configs:
{cfgs}
---

# QuakeScope phase-picking evaluation set, v1

Seismic waveforms, the phase arrivals a human analyst picked on them, and the
scorer that turns one into a number. {total:,} arrivals across four tracks.

Built for people who build models rather than catalogues. Nothing here assumes
you know what an operator bulletin is, and the one thing you do have to
understand is two sections down.

## The task

Given three-component seismic waveforms, output the arrival time of every P and
S phase, each with a confidence in [0, 1]. You are scored on how many of the
analyst's arrivals you recover within 0.5 s on the same station and phase.

```python
from datasets import load_dataset
arrivals = load_dataset("gaia-hazlab/quakescope-eval-v1", "track2-swarm")["reference"]
```

Waveforms are MiniSEED beside each track, readable with
[ObsPy](https://docs.obspy.org): `obspy.read("track2-swarm/waveforms/*.mseed")`.

`tasks.json` states the task, the inputs, the target, the scoring command, both
tolerances and the matching rule in a form your code can read rather than your
reader having to notice.

## The tracks

| track | regime | sequences | arrivals | P | S | waveform files |
|---|---|--:|--:|--:|--:|--:|
{rows}

Track 1 asks whether a picker serves the catalogue this project builds. Track 2
asks whether it generalises, split by what actually breaks a picker:
mainshock-aftershock cascades, volcano-tectonic seismicity, fluid-driven swarms.
Those are different failure modes, not different places, which is the point of
organising a benchmark this way.

## The reference is not a labelled test set

**Read this before taking any number from the set.**

The arrivals come from operator bulletins and published analyst catalogues. An
analyst picked what a location needed and then stopped. In a dense sequence most
real arrivals are never marked, so a model pick with no analyst counterpart is a
mixture of a false positive and a real arrival nobody had time to write down.

| metric | against this reference |
|---|---|
| recall, picks emitted, MAE, RMSE, MedianAE, median bias, gross-error rate, phase swap rate, duplicate rate | **exact** |
| precision, F1, calibration, ECE | **lower bound** |
| MCC | **not computable** |

MCC needs true negatives, and a continuous record with an incomplete reference
does not define them. A published MCC against a bulletin is not meaningful.
`tasks.json` carries this as a field so your code can branch on it.

## Score your picks

```sh
pip install -r score/requirements.txt
python score/score_picks.py --demo                     # synthetic, no data needed
python score/score_picks.py \\
    --reference track2-swarm/reference_picks.csv \\
    --picks     your_picks.csv \\
    --out       scores/
```

Your picks need `station, phase, time, conf`, with optional `sequence`. Run your
model **once at a low confidence floor** and keep every pick: each threshold is
then a filter over one file rather than another pass over the waveforms.

`model_picks.csv`, where present, is what four published PhaseNet weight sets
emitted on the same windows, so you can reproduce our numbers before trusting
your own.

## What is withheld

This is a development set. Twelve further sequences across the same three
regimes are held back and scored on submission.

The underlying data cannot be hidden: every arrival here comes from a public
service and anyone can re-harvest it. What is withheld is the **selection** —
which sequences, which windows, which stations, and the assembled reference for
them. That is the part that carries the evaluation and the part a model would
otherwise be tuned against.

## Caveats that change how you read a score

- **Sample sizes differ by an order of magnitude.** Weight by arrivals or report
  per sequence; do not average the tracks.
- **Some sequences are deliberately thin and their recall is not
  interpretable.** Monte Cristo has 16 P and 12 S on one station. Monroe WA is a
  normal moderate earthquake whose 784 arrivals spread over seven days of small
  aftershocks, so almost none fall in the scored window. Both are in for network
  and era coverage. A recall of zero on either is an empty reference, not a model
  failure.
- **Station selection is constrained by instrument.** A three-component picker
  cannot use a vertical-only short-period station, and regional networks are full
  of them, so these are not simply the stations with the most arrivals.
- **Contamination is stated, not measured.** None of these sequences is in the
  curated corpora we know of, but we have not diffed them trace by trace against
  every training set a submitter might use.
- **No association step.** Picks are scored as picks.

## Where the labels come from

Every arrival was picked by an analyst at a named agency and retrieved from a
named service. Nothing here was produced by a model.

| sequence | analysed by | retrieved from | which picks | cite |
|---|---|---|---|---|
{prov_rows}

Tracks 1 and 2a/2b keep **only** arrivals whose `evaluation_mode` is `manual`,
so preliminary automatic picks the catalogues also carry are dropped before the
reference is built. That filter removed 1,826 automatic AIC picks from Kaikoura
alone.

The swarm track is the exception and the difference matters. The ISC Bulletin
relays its contributing agencies' **reviewed** readings but does not populate an
evaluation-mode field, so for ISC-sourced arrivals "analyst" follows from what
the ISC Bulletin is, not from a flag on the record. Where a sequence has a
second source — BCSF-RENASS for Maurienne, ANSS for Salton Sea and
Jones-Guthrie — those arrivals do carry `manual` and are deduplicated against
the ISC ones at 100 ms.

## Seismic networks

The waveforms belong to the networks that recorded them. Each has its own DOI
and its own citation; cite the ones whose data you use.

| network | arrivals | years | operator | DOI |
|---|--:|---|---|---|
{net_rows}

Station codes are **not unique across networks**. Resolving a bare code against
a federated catalogue returns whichever network answers first, which put a
Nevada Test Site station on a Czech swarm 9,097 km away in our first build. Every
station in this set is checked to lie within 400 km of its sequence.

## Reference catalogues and bulletins

| source | DOI |
|---|---|
| ISC Bulletin, International Seismological Centre | [10.31905/D808B830](https://doi.org/10.31905/D808B830) |
| ANSS Comprehensive Catalog, U.S. Geological Survey | [10.5066/F7MS3QZH](https://doi.org/10.5066/F7MS3QZH) |
| Southern California Earthquake Data Center | [10.7909/C3WD3xH1](https://doi.org/10.7909/C3WD3xH1) |
| Northern California Earthquake Data Center | [10.7932/NCEDC](https://doi.org/10.7932/NCEDC) |
| GeoNet Aotearoa New Zealand, GNS Science | [10.21420/G19Y-9D40](https://doi.org/10.21420/G19Y-9D40) |
| Rete Sismica Nazionale, INGV | [10.13127/sd/x0fxnh7qfy](https://doi.org/10.13127/sd/x0fxnh7qfy) |
| Hellenic Unified Seismological Network | [10.7914/SN/HL](https://doi.org/10.7914/SN/HL) |
| EPOS-France / RESIF | [10.15778/RESIF.FR](https://doi.org/10.15778/RESIF.FR) |

Every DOI above was resolved against doi.org when this card was generated.

## Licence

Code in `score/` is MIT, from
[SeisSCOPED/QuakeScope](https://github.com/SeisSCOPED/QuakeScope). The board
these numbers feed is at
[seisscoped.org/QuakeScope/benchmark_metrics.html](https://seisscoped.org/QuakeScope/benchmark_metrics.html).

If you report a number, state the track, the protocol and the tolerance. A
recall without those three is not reproducible.

Built {built} from `{commit}`.
"""


if __name__ == "__main__":
    main()
