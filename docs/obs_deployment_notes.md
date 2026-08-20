# Deploying QuakeScope on ocean-bottom data

Notes for running the picker over temporary OBS experiments and the OOI cabled
network (`OO`). Companion benchmark:
[`tutorials/phasenet_obs_offshore_benchmark.ipynb`](../tutorials/phasenet_obs_offshore_benchmark.ipynb),
rendered at
[seisscoped.org/QuakeScope](https://seisscoped.org/QuakeScope/phasenet_obs_offshore_benchmark.html).

---

## 1. The permanent picking offset is the catalog, not the picker

The benchmark reports picks landing 1–3 s **before** the arrival predicted from
the catalog, consistently, across every weight set. That is worth explaining
because it looks like a bug and is not one.

Three candidate causes, tested:

**Resampling — ruled out.** SeisBench resamples everything to 100 Hz, so a
resampling artefact was the obvious suspect. AACSE (`XO`) records at a native
100 Hz on every channel, seismometer and hydrophone alike, so no resampling is
performed there at all. The offset persists at −1.65 s median. Whatever causes
it survives the absence of resampling.

**Travel-time phase selection — ruled out.** The prediction originally asked
taup for `["p", "P"]`, which risks missing a faster refracted branch at regional
distance. Re-predicting with the full first-arriving set (`ttp`) changed nothing:
the gap between the two predictions is **0.00 s** on every event, and the first
arrival is variously named `P`, `Pn` or `p` without affecting the time.

**Catalog location and origin time — this is it.** The decisive test is whether
the offset belongs to the *event* or to the *station*. Twelve events recorded at
two or three OBS stations each:

| | |
|---|---|
| Mean spread **within** an event (same quake, different stations) | **0.55 s** |
| Spread **between** events | **1.55 s** |
| Ratio | **2.8** |

Each earthquake carries its own offset, and every station sees nearly the same
one. Per-event means run from +0.74 s to −5.14 s while the within-event scatter
stays under a second. That is the signature of an error in the origin time and
hypocentre, which shifts all predicted arrivals for that event together, and not
of anything happening at the instrument or in the model.

The cause is unsurprising: these are offshore earthquakes located almost
entirely by **onshore** networks, with all the stations on one side. Depths in
this set range from 8 to 40 km and several sit on round numbers. Correlation
between offset and epicentral distance is weak (+0.24), which also argues
against a simple velocity-model scaling error — a wrong 1-D model would grow
with distance.

### What follows for the deployment

- **Do not calibrate or threshold-tune against catalog-predicted arrivals.**
  The reference is noisier than the thing being measured.
- **Do not treat the offset as a picker bias to correct.** There is nothing to
  correct; applying a constant shift would corrupt good picks.
- **The picks are the better product.** Recovering arrivals at ocean-bottom
  stations is exactly what improves these locations — the association step
  solves for origin time and absorbs the discrepancy.
- For validation, prefer **inter-model agreement** and **cross-station
  moveout coherence** over agreement with a catalog that has no OBS constraint
  in it.

---

## 2. Which model to start with

The benchmark covers three ocean-bottom pickers across Cascadia, the Alaska
Peninsula, and the Blanco transform:

| Detection rate | pickblue_phasenet | pickblue_eqt | obstransformer | quakescope2026 | original |
|---|--:|--:|--:|--:|--:|
| Cascadia (7D) | 0.74 | **0.79** | 0.61 | 0.72 | 0.65 |
| AACSE (XO) | 0.78 | 0.72 | **0.89** | 0.78 | 0.67 |
| Blanco (X9) | 0.53 | 0.53 | 0.51 | 0.53 | 0.49 |

The ocean-bottom models lead the land models by roughly 5–15 points, and no
single one leads everywhere.

**Starting with a PhaseNet model is a sound choice**, and the one to use is
`PickBlue(base="phasenet")` — the SeisBench `obs` weights. It is a PhaseNet, so
it drops into the existing pipeline unchanged; it is trained on ocean-bottom
data; and it is competitive everywhere without winning outright anywhere. The
EQTransformer variants are worth revisiting once the pipeline is running, but
they change the model class as well as the weights.

**The hydrophone is not the reason these models are better.** Withholding it on
94 detected windows moved mean P confidence by **+0.0002**, helping 32 windows
and hurting 35. Its sampling rate spans 100 Hz to 10 Hz across these
deployments without changing that conclusion, and `OBSTransformer` competes
strongly using no hydrophone at all. The advantage comes from training on
ocean-bottom data. Practically: **stations with a dead or missing hydrophone are
not disqualified.**

---

## 3. The `OO` cabled network

Surveyed 2026-08 for 2024–2026. Thirteen stations, of which six carry a
pressure channel.

**Components are `ENZ`, not `1`/`2`.** OOI instruments are cabled and installed
with known orientation, unlike free-fall OBS. This is the opposite of every
temporary deployment in the benchmark.

That raised an obvious concern — `PickBlue` declares `Z12H` — and it turns out
not to be a problem. Running `PickBlue` on `AXCC1` with channels named
`HHZ/HHN/HHE/HDH` and again with the horizontals renamed to `HH1/HH2` gives
**byte-identical picks**, so SeisBench already maps the horizontal slots
equivalently. **No channel renaming is required.**

Two things that do need handling:

- **Band codes vary and many stations are short-period.** `AXBA1`, `AXCC1`,
  `AXEC2` and `HYS14` carry `BH`/`HH`; the rest are `EH`/`SH`. Any station list
  should select the band per station rather than assume `HH`, exactly as the
  land pipeline had to for 2003 SCEDC data.
- **Sampling rates are high.** `AXCC1` delivers 200 Hz, double the model rate.
  Resampling is handled by SeisBench, and per section 1 it is not a source of
  timing error, but it is worth knowing for throughput estimates.

Pressure channel band codes also vary (`BD`, `HD`, `LD`, `UD`), so a hydrophone
selector has to be permissive. Given the ablation result, it is reasonable to
run `OO` without the pressure channel entirely and keep the station list
simple.

---

## 4. Suggested first campaign

1. **Weight:** `obs` via `PickBlue(base="phasenet")`, submitted as
   `--weight obs`. Confirm it is baked into the image the same way
   `quakescope2026` was.
2. **Thresholds:** do not inherit the land defaults. `obs` ships P 0.2 / S 0.1,
   already asymmetric, and the land benchmark showed a shared threshold
   silently changes the operating point between weight sets. Set the OBS
   threshold from a pick-budget target on a held-out week.
3. **Scope:** start with `OO`, which is permanent, cabled, well-characterised
   and small enough to inspect by hand. Then extend to the temporary
   experiments, which are larger and messier.
4. **Validation:** inter-model agreement and moveout coherence across stations,
   not agreement with catalog arrival times — see section 1.
