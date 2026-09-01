# 08 — Running several pickers in one campaign (OBS / general / California)

Plan: three different phase-picker weights, each applied to its own set of
stations:

1. **OBS picker** — offshore / ocean-bottom-seismometer networks.
2. **General picker** — the newly trained default, for everything else.
3. **California picker** — a named list of remaining California networks.

This needs **no new infrastructure**. One image, one job definition, one
queue, one database — the weight is just a per-submission argument
(`--weight`), and each job's `sb_runs` document records which weight made
which picks.

## 1. Bake all three weights into the image

Put all three weight pairs in `sb_catalog/models/v3/phasenet/`:

```
sb_catalog/models/v3/phasenet/
├── obs2026.pt.v1          + obs2026.json.v1
├── general2026.pt.v1      + general2026.json.v1
└── california2026.pt.v1   + california2026.json.v1
```

Push to `main`, one image build, done (guide 02). If any of the three is a
different architecture (not PhaseNet), see the architecture note in guide 02.

## 2. Partition the stations — this is the one real design rule

Within one database, the resume logic (`picks_record`) is **weight-agnostic**:
once a station-day is picked by *any* picker, later jobs skip it. Two
consequences:

- **If the three station sets are disjoint** (the normal case: a station is
  offshore OR in the CA list OR neither), use **one database**
  (`quakescope2026`) for all three campaigns. Clean, and association later
  sees all picks together.
- **If you want the same station picked by two different weights** (e.g. to
  compare pickers on overlap stations), those runs must go into **separate
  databases** (`quakescope2026_obs`, `quakescope2026_ca`, ...), because the
  second picker would otherwise be skipped. Each database then needs the
  station metadata loaded (notebook 2) — a few minutes each.

Make the partition explicit before submitting: three network lists with no
network appearing twice. Keep them in a small text file next to the
submission log, e.g.:

```
OBS_NETS = "OO,7D,X9,..."          # the OBS deployments you target
CA_NETS  = "BG,BP,PG,WR,NC,CI,..." # the agreed California list
GEN_NETS = everything else from networks/*.zip minus the two lists above
```

Generate `GEN_NETS` programmatically so nothing is dropped or duplicated:

```python
import glob
obs = set("OO,7D,X9".split(","))          # paste your OBS list
ca  = set("BG,BP,PG,WR,NC,CI".split(","))  # paste your CA list
allnets = {f.split("/")[-1].split(".")[0] for f in glob.glob("../networks/*.zip")}
assert not (obs & ca), f"overlap: {obs & ca}"
print(",".join(sorted(allnets - obs - ca)))
```

Note the archive is orthogonal to the picker: `constants.py` still routes
each network to NCEDC/SCEDC/EarthScope automatically, so an OBS network on
EarthScope and a CA network on NCEDC both "just work".

## 3. Submit — one command per picker (× year blocks)

```bash
# 1) OBS networks with the OBS weights
PYTHONPATH=../sb_catalog ~/miniconda/bin/python -m src.submit_helper \
    pick 2023.001 2024.001 \
    --network "$OBS_NETS" \
    --database quakescope2026 --weight obs2026
```

```bash
# 2) General picker for the rest of the world
PYTHONPATH=../sb_catalog ~/miniconda/bin/python -m src.submit_helper \
    pick 2023.001 2024.001 \
    --network "$GEN_NETS" \
    --database quakescope2026 --weight general2026
```

```bash
# 3) California picker for the CA list
PYTHONPATH=../sb_catalog ~/miniconda/bin/python -m src.submit_helper \
    pick 2023.001 2024.001 \
    --network "$CA_NETS" \
    --database quakescope2026 --weight california2026
```

You can also add `--extent minlat,maxlat,minlon,maxlon` to any of these to
clip a network list geographically (e.g. bound the CA campaign to
`32,42,-125,-114`); `--network` and `--extent` combine as AND.

All three can be in the queue simultaneously — they share the Fargate pool
and drain in submission order-ish. Smoke-test each weight with a 2-day
1-network submission first (guide 05 §3) and check `sb_runs` shows three
distinct weight names.

## 4. Provenance and later analysis

Each pick carries `rid` → `sb_runs` → `weight`, so per-picker catalogs are a
join away:

```python
runs = {r["_id"]: r["weight"] for r in db.database["sb_runs"].find()}
obs_run_ids = [k for k, w in runs.items() if w == "obs2026"]
n = db.database["picks"].count_documents({"rid": {"$in": obs_run_ids}})
```

## 5. OBS caveat worth checking before launch

The workflow fetches components `ZNE12` (seismometer channels listed in the
station metadata). **Hydrophone channels (e.g. `HDH`/`EDH`) are not fetched.**
If the OBS weights were trained with a hydrophone component, the pipeline
needs a small extension (`--components` plus channel metadata) — check with
whoever trained the OBS model whether it expects 3-C seismometer data only;
if 3-C only, nothing to do.
