# QuakeXNet outside the Pacific Northwest: what we measured, and what to do

Written 2026-08-16, after testing the classifier on Alaska and on Southern
California. Companion notebook:
[`tutorials/quakexnet_alaska_test.ipynb`](../tutorials/quakexnet_alaska_test.ipynb).

**Short version.** The model is not broken and the classes do transfer. Two
things degrade it away from home, and only one of them needs a retrain:

1. **Window placement.** Accuracy on a fixed set of Alaska events swings from
   16% to 78% depending only on where the arrival sits in the 100 s window.
   Nothing about the data changes. This is free to fix in the pipeline today.
2. **PNW-only training.** What remains after fixing placement is a genuine
   domain limit, worst for surface events. That needs labels from elsewhere.

Our first Alaska result — 38%, which prompted the alarm — was an artifact of
(1). We had centred the arrival. Re-cutting the same waveforms to the training
convention took it to 78%.

---

## 1. What we measured

Thirty-two Alaska events at two stations, spanning all three interesting
classes, scored against Alaska Earthquake Center labels.

At a 15 s lead, matching training:

| Catalog type | Agreement |
|---|---|
| explosion / quarry blast | 11/12 (92%) |
| earthquake | 9/12 (75%) |
| surface event | 5/8 (62%) |
| **overall** | **25/32 (78%)** |

Sweeping the window placement across the same events and waveforms:

| lead before arrival | overall | eq | px | su |
|--:|--:|--:|--:|--:|
| 5 s | 56% | 83% | 25% | 62% |
| 10 s | 69% | 75% | 67% | 62% |
| **15 s** | **78%** | 75% | **92%** | 62% |
| 20 s | 66% | 75% | 75% | 38% |
| 30 s | 72% | 75% | 83% | 50% |
| 40 s | 66% | 67% | 75% | 50% |
| 50 s | 38% | 42% | 17% | 62% |
| 60 s | 28% | 25% | 8% | 62% |
| 75 s | 16% | 8% | 0% | 50% |

Two patterns worth noting. **Explosions are the most placement-sensitive class**
(0% to 92%), which makes sense — an explosion is defined by a sharp impulsive
onset, so where that onset sits dominates the spectrogram the model sees.
**Surface events are the least sensitive** (38–62% throughout), equally sensible
for an emergent signal with no onset to key on. That also means surface events
are the class where placement is *not* the problem.

## 2. Was the training wrong?

No — and the paper is straighter about the limitations than the summary
statistics suggest.

[Kharita, Denolle, Hutko, Hartog & Malone (2026), *Exploration of Machine
Learning Methods to Seismic Event Discrimination in the Pacific
Northwest*, Seismica](https://seismica.library.mcgill.ca/article/view/2068)
reports 92.4% in-domain accuracy for QuakeXNet 2D, on ~200,000 waveforms from
>70,000 events, with **event-level train/test separation** — they verified no
event identifier appears in both splits, which is the right way to do it and
avoids the leakage this kind of study often has.

On generalization the paper states plainly that both CNNs "struggled when
applied to out-of-domain datasets", that QuakeXNet generalized better than the
alternative but "a slight confusion between surface events and explosions
persisted", and that on the global ESEC catalog surface events were frequently
misclassified as explosions. It also notes the "surface event" class is
"broad, encompassing a diverse set of mass-movement and volcanic processes",
and estimates **0.2–8% of training labels per class may be wrong**.

Our Alaska numbers sit comfortably inside those stated caveats. Surface events
are our weakest class; the paper says surface events are the hardest and the
most confusable with explosions. Nothing here contradicts the published work.

The one thing the paper does not quantify is the placement dependence. It
documents the convention — "a 100-second window, with the window start chosen
randomly between 5 and 20 seconds before the analyst's pick time" — but that
15 s of jitter is the *entire* range of arrival positions the model has ever
seen. It is enough augmentation to avoid memorising one exact offset and not
nearly enough to make the model position-invariant. The sweep above is, as far
as we can tell, the first measurement of how sharply that matters.

## 3. Fix the pipeline before retraining anything

QuakeScope classifies continuous data by sliding a 100 s window with a stride of
2500 samples — 50 s. Events land wherever they land. From the table above, that
means a substantial fraction of events are classified in the regime where the
model is worst, and the same event appearing in two consecutive windows gets two
different placements.

This is visible in production output. One M4.6 Ridgecrest aftershock, crossing
six consecutive windows, came back `eq`, `px`, `su`, `px`, `eq`, `eq`. That is
not a model that cannot decide; it is one arrival presented six different ways.

**The pipeline already picks phases before it classifies.** Cutting each
classification window at a fixed lead before a pick, instead of on a blind
slide, puts every event in the trained band. It costs nothing, needs no
retraining, and should be done regardless of what follows.

Two smaller items in the same category:

- **Keep the probabilities.** The `classifies` collection stores per-class
  probabilities, and as of this branch the classifier actually populates them.
  A margin of 0.92 versus 0.34 is the difference between a usable label and a
  coin flip, and argmax alone throws that away.
- **Do not use it outside its band.** On a Southern California M4.6 the model
  returns `px` at 0.91. Until there are Californian labels in training, the
  classifier should not be writing class labels into a Californian catalog.

## 4. Labels worth pulling

Counts below are from the USGS catalog, 2015 to present, M≥1.0, queried
2026-08-16. The 2000 entries are query caps, not totals.

| Region | quarry blast | explosion | ice quake | landslide | volcanic |
|---|--:|--:|--:|--:|--:|
| Alaska | 94 | 2000+ | **2000+** | 31 | — |
| Nevada | **2000+** | 579 | — | — | — |
| Wyoming / Montana | 707 | — | — | — | — |
| Utah | 164 | 8 | — | — | — |
| Cascades (PNW) | 20 | 2000+ | — | — | — |
| Hawaii | 1 | — | — | — | 62 |

**Explosions are not the bottleneck.** Nevada alone roughly doubles the
available blast population, and Wyoming/Montana adds another 700 from different
mines, different rock, different network. Utah's Bingham Canyon blasts are
valuable precisely because they are large, repeated, and well catalogued. This
is the cheapest class to broaden and, from the sweep, the one whose performance
moves most.

**Surface events are the binding constraint, and Alaska is the unlock.**
Two thousand-plus catalogued ice quakes around Columbia Glacier and Prince
William Sound is more surface-event labels than the original PNW training set
contained in total, from a genuinely different setting. Landslides stay scarce
everywhere (31 in Alaska), so the realistic path to a broad `su` class is
glacial sources plus:

- **ESEC**, the Exotic Seismic Event Catalog — 245 globally distributed events
  verified by non-seismic means. Small, but it is the closest thing to
  ground truth for this class, and the paper already uses it as an
  out-of-domain test. Keep it as a test set, do not train on it.
- **Volcano observatories** — AVO, CVO, HVO carry rockfall, lahar, and
  pyroclastic-flow catalogues that never reach the USGS national feed. Hawaii
  shows 62 volcanic events nationally, which is certainly an undercount of
  what HVO holds locally.
- **Mt Rainier**, via the 114,775 surface events in Akash's own 15-year
  catalog. These are model-derived rather than analyst-labelled, so they are
  bootstrapping rather than ground truth — usable with care, for instance to
  mine hard negatives, not as clean labels.

**On DiTing.** It is large and high quality but it is a phase-picking dataset —
earthquakes with P and S annotations, not source-type labels. It would help a
picker and does nothing for a four-class discriminator. Not worth the effort
for this problem.

### 4a. Where the surface events actually are

Surveyed 2026-08-16 against the providers' own services and the literature.
Counts are analyst-assigned labels, not model output, unless stated.

| Source | Access | SU-relevant content | Verdict |
|---|---|---|---|
| **Alaska (AEC/USGS)** | FDSN, open | 2000+ ice quakes; 31 landslides | **Best single source.** Glacial, well catalogued, already tested |
| **ESEC** (EarthScope SPUD) | Open, + USGS ScienceBase SQLite | ~245 events: landslides, debris flows, avalanches, lahars, outburst floods, mine collapses, a submarine landslide, a volcanic flank collapse. Global. **Includes pointers to waveforms at the DMC** | **Keep as the test set.** Too small and too precious to train on |
| **Piton de la Fournaise** (OVPF/IPGP) | Published catalogs; FDSN carries events but the text service omits event type | ~7,000 volcano-seismic events, 2014–2021, labelled across 7 classes including **rockfall** | **Largest labelled rockfall set found.** Worth a direct request to OVPF |
| **Swiss Alps (SED/ETH)** | FDSN, open | Landslides rising 7/yr (2016) → **39/yr (2024)**; ~30 ice quakes; occasional rockslides; plus 2000+ quarry blasts | **Yes.** Modest volume but clean labels and a different tectonic and climatic setting |
| **Illgraben** (WSL + SED) | Published; instrumented catchment | Manually labelled **debris flows**; roughly four weeks of dense labels, dozens of slope failures | **Yes, for a class we have nothing else for.** Small but it is the real thing |
| **New Zealand** (GNS/GeoNet + universities) | FDSN catalog is earthquakes only; the useful material is in published work and institutional databases | Lahars at Ruapehu; volcano-seismic classes at Whakaari; two large landslide *inventories* | **Partly** — see 4c |
| **Italy** (INGV + universities) | Open data portal has nothing under landslide or *frana*; the material is in published work | Stromboli flank landslides; instrumented rockslides; nanoseismic collapse catalogs | **Partly** — see 4c |
| **Japan (NIED/JMA)** | Hi-net registration; no open labelled corpus found | Volcanic and tremor classification exists per study, not as a curated multi-class corpus | **Not now.** Access friction plus label assembly; revisit only if a collaborator supplies labels |
| **Volcano observatories** (AVO, CVO, HVO) | Local catalogs, request | Rockfall, lahar, pyroclastic flow | Worth asking; these never reach the national feed |

### 4c. New Zealand and Italy, looked at properly

Both were dismissed too quickly on the strength of an empty FDSN query. The
labels exist in both countries; they are simply not in the event web service.

**New Zealand.** Three distinct things, only one of which is what we need.

- **NZ Landslide Database** and the **2016 Kaikōura Landslide Inventory (v3)**
  are large — hundreds of thousands of features in the former, tens of
  thousands from Kaikōura alone — but they are *geomorphic inventories* mapped
  from LiDAR and aerial imagery. Most entries are rainfall-triggered with no
  known origin time, and the Kaikōura ones are coseismic, so their signals sit
  inside a Mw 7.8 coda. Without a timestamp there is no window to cut. **Not
  trainable as they stand**, though the database is the right place to look if
  we ever want to match known failures against continuous data.
- **Ruapehu lahars** are the genuinely useful piece. GNS runs two operational
  seismo-acoustic detection systems on the mountain — ERLAWS and the Eruption
  Detection System — and lahars are debris flows, the class we have almost
  nothing for. Cole et al. (2009, GRL) analyse the seismic signature of the
  2007 snow-slurry lahars directly. Small, operational, and squarely on target.
- **Whakaari/White Island** has a rich classified record — VT, LP, VLP, and
  tremor, with recent unsupervised classification work (Steinke et al., 2023).
  Valuable, but see the taxonomy problem below.

**Italy.** The open portal is genuinely empty on this topic — searches for both
*rockfall landslide* and *frana* return nothing — but the published record is
not.

- **Stromboli** is the strongest candidate in the country. Landslides down the
  Sciara del Fuoco produce a documented and distinctive signature: broader band
  and higher frequency than explosion quakes or tremor, with a cigar-shaped
  amplitude envelope. INGV has run a 13-station broadband network there since
  2003, so the record is long, continuous, operational, and includes the
  tsunamigenic 30 December 2002 collapse. There is published work on the
  seismic signals of these landslides and on precursors to crater collapses.
  **This is the Italian ask.**
- **Instrumented slopes** — the Torgiovannetto quarry rockslide (temporary
  network, 2012–13) and the Peschiera Springs system, where nanoseismic arrays
  located 397 slope-instability events separated into 16 failures and 381
  collapses, plus later sequences of 500+ underground collapses. These are
  real labelled catalogs, but they are **nanoseismic**: metres to hundreds of
  metres, recorded on dedicated arrays. Useful for understanding failure
  physics, not directly transferable to regional stations at tens of km.
- **Etna's** Valle del Bove flank instability is monitored but the published
  classification work is volcanic-activity oriented — lava fountains, tremor
  regimes — rather than mass movement.

### 4d. The taxonomy problem nobody has solved

The volcano observatories have the largest classified archives, and their
ontology does not match ours. They work in **VT / LP / VLP / tremor /
rockfall / eruption**; QuakeXNet works in **eq / px / no / su**.

Rockfall maps onto `su` cleanly. VT maps onto `eq` cleanly. **LP, VLP and
tremor map onto nothing** — they are volcanic source processes, not surface
mass movement, not tectonic, and certainly not noise. Forcing them into `su`
would teach the model that `su` means "anything unusual near a volcano", which
is precisely the vagueness the paper already identifies as making `su` hard.

This has to be decided before ingesting any observatory data, and the honest
options are:

1. **Take only the mappable classes** — rockfall and VT — and discard the rest.
   Simple, wasteful, and safe.
2. **Add volcanic classes** and move to six or seven, accepting that the model
   then needs volcanic examples everywhere it is deployed or it will apply
   those labels off-volcano.
3. **Keep four classes but treat volcanic sources as an explicit reject class**,
   so the model can say "not one of mine" rather than guessing.

Option 1 is the right first move; it gets us Piton de la Fournaise rockfalls
and Stromboli landslides without committing to an ontology change. Option 3 is
where this probably needs to end up, and it pairs naturally with the
calibration work in section 5.

Two things this survey changes about the plan.

**Rockfall is reachable at volume.** The Piton de la Fournaise catalog is roughly
seven thousand labelled volcano-seismic events with rockfall as an explicit
class — comparable in size to the entire exotic-event portion of the PNW
dataset, from an island volcano with nothing in common with the Cascades. If
one collaboration is worth pursuing for this project, it is that one.

**Debris flow is a genuine hole.** Nothing in the American catalogs covers it
and Illgraben is the only well-labelled source found. That matters because
debris flows are long-duration emergent signals, which is exactly the corner of
the `su` class the model is weakest in.

### 4b. A warning from the rockfall literature

Hibert and colleagues classified rockfalls against volcano-tectonic events at
Piton de la Fournaise with up to 99% accuracy — and then reported that a
classifier trained on 2009–2011 data **collapsed** when applied to 2014–2015
data from the same volcano, attributed to a change in the physical mechanism of
the rockfalls themselves.

Same instruments, same site, same class label, five years apart, and the model
stopped working. That is a stronger warning than anything in our own results:
for surface events, the label names a process that can itself change. It argues
for held-out splits in **time** as well as region, and for periodic
re-validation rather than a train-once-deploy-forever posture.

**Noise** should be sampled per region rather than reused from the PNW. Noise
is the most site-specific class there is, and a model that learns PNW noise will
call Alaskan noise something else.

## 5. Building a classifier that travels

Ordered by expected value per unit of effort.

**Make position irrelevant.** This is the single highest-value change and it
does not need a single new label. Widen the training jitter from 15 s to the
full window, so an arrival can appear anywhere, and let the model see that
position carries no information. Architecturally, the current model flattens the
convolutional stack into a dense layer, which is what makes it position-aware in
the first place; global pooling over the time axis before the dense layers would
remove the dependence structurally instead of statistically. Either path is
cheap. Do this first, then re-run the sweep above — a flat curve is the success
criterion.

**Then add regional labels**, in the order the counts justify: Nevada and
Wyoming blasts, Alaska ice quakes, Utah blasts. Hold out entire regions rather
than random events. In-domain accuracy is already 92% and is not the number that
matters; the number that matters is accuracy on a region the model has never
seen, and only a region-level split measures it.

**Reconsider the surface-event class.** The paper flags that `su` lumps together
mass movement and volcanic processes; our results show it is the weakest and
least placement-sensitive class, which suggests the model is not finding a
shared signature because there may not be one. Two options worth testing: split
`su` into sub-classes with enough labels to support them (icequake, rockfall,
lahar, debris flow), or keep it single but train it on a much wider variety so
it becomes a genuine "not eq, not px" catch-all. The clustering work already in
Akash's repo — UMAP plus HDBSCAN on QuakeXNet embeddings — is the natural way to
decide which, since it will show whether the sub-types separate in the model's
own representation.

**Fix labels before adding more.** With 0.2–8% of labels wrong by the paper's
own estimate, and quarry-blast labels often assigned from time of day and known
source location rather than waveform, some of the residual confusion is catalog
error rather than model error. Cross-checking blast labels against operator
schedules where those exist would tighten the class more cheaply than new data.

**Keep calibration, not just argmax.** For a catalog, a well-calibrated
probability with an abstain option beats a confident wrong label. Several of our
Alaska errors were made at above 0.9, which is the failure mode a downstream
catalog cannot detect. Temperature scaling on a held-out region is a small
addition with a large effect on usability.

## 6. Sequencing

1. **Now, no retraining.** Cut classification windows relative to picks. Re-run
   the Alaska notebook to confirm the gain in place. Stop writing class labels
   for regions with no representation in training.
2. **Next, one retrain, no new labels.** Full-window position augmentation, or
   global pooling. Success is a flat placement curve.
3. **Then, labels.** Nevada and Wyoming blasts and Alaska ice quakes first —
   all open, all FDSN, no permission needed. Region-level held-out splits, and
   for surface events hold out in time as well. ESEC stays a test set.
4. **In parallel, ask.** Four conversations, none of which is a download, all
   covering classes nothing open fills. Start them early because they run on
   other people's calendars:
   - **OVPF / IPGP** — Piton de la Fournaise rockfalls, ~7,000 labelled events.
     The single largest prize.
   - **INGV Osservatorio Vesuviano** — Stromboli Sciara del Fuoco landslides,
     a 20-year operational broadband record with a documented signature.
   - **WSL / SED** — Illgraben debris flows.
   - **GNS** — Ruapehu lahars via ERLAWS.

   Take only the classes that map onto ours to begin with (§4d), so none of
   these commits us to an ontology change.
5. **Then, the `su` question.** Cluster the embeddings, decide split-or-broaden,
   and size the label collection from that answer.

Steps 1 and 2 are days of work and address the larger measured effect. Step 3 is
where the real cost sits, and it is worth knowing how much of the gap step 2
closes before paying it.

## References

- Kharita, Denolle, Hutko, Hartog & Malone (2026), *Exploration of Machine
  Learning Methods to Seismic Event Discrimination in the Pacific Northwest*,
  Seismica — <https://seismica.library.mcgill.ca/article/view/2068>
  ([preprint](https://arxiv.org/html/2510.23795))
- Ni et al. (2023), *Curated Pacific Northwest AI-ready Seismic Dataset*,
  Seismica — <https://seismica.library.mcgill.ca/article/view/368>
- Training code: <https://github.com/Denolle-Lab/PNW_Seismic_Event_Classification>
- Deployment and Mt Rainier catalog:
  <https://github.com/Akashkharita/pnw_seismic_event_detection>

Surface-event sources:

- ESEC, Exotic Seismic Events Catalog — <https://ds.iris.edu/ds/products/esec/>,
  searchable at <http://ds.iris.edu/spud/esec>; SQLite release *Seismogenic
  Landslides and other Mass Movements* (v3.0, May 2025) on USGS ScienceBase
- Hibert et al., *Automatic identification of rockfalls and volcano-tectonic
  earthquakes at Piton de la Fournaise using a Random Forest algorithm*, JVGR —
  <https://www.sciencedirect.com/science/article/abs/pii/S0377027316303948>
  (source of the 2009–2011 → 2014–2015 degradation)
- *Deep Learning and Machine Learning Applied to the Detection and
  Classification of Volcano-Seismic Events at Piton de la Fournaise*, Pure
  Appl. Geophys. (2025) — <https://link.springer.com/article/10.1007/s00024-025-03809-9>
  (~7,000 events, seven classes, 2014–2021)
- Chmiel et al., *Near-real-time automated classification of seismic signals of
  slope failures with continuous random forests*, NHESS —
  <https://nhess.copernicus.org/articles/21/339/2021/> (Illgraben)
- Swiss Seismological Service event service — <https://eida.ethz.ch>

New Zealand:

- NZ Landslide Database (GNS) —
  <https://www.gns.cri.nz/data-and-resources/new-zealand-landslide-database/>
  (geomorphic inventory; most entries have no origin time)
- Massey et al. (2018), *Landslides Triggered by the 14 November 2016 Mw 7.8
  Kaikōura Earthquake*, BSSA — inventory behind the Kaikōura dataset
- Cole et al. (2009), *Seismic signals of snow-slurry lahars in motion:
  25 September 2007, Mt Ruapehu*, GRL —
  <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009GL038030>
- Eastern Ruapehu Lahar Alarm and Warning System (ERLAWS) — operational
  seismo-acoustic lahar detection run by GNS
- Steinke et al. (2023), *Identification of Seismo-Volcanic Regimes at
  Whakaari/White Island via Systematic Tuning of an Unsupervised Classifier*,
  JGR — <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JB026221>

Italy:

- INGV open data portal — <https://data.ingv.it/en/> (no landslide or *frana*
  datasets as of 2026-08; the material is in the literature)
- *Seismic Signals Associated with Landslides and with Tsunami at Stromboli
  Volcano* — signature of Sciara del Fuoco failures
- *Seismic and thermal precursors of crater collapses and overflows at
  Stromboli*, Sci. Rep. (2023) —
  <https://www.nature.com/articles/s41598-023-38205-7>
- *Seismic Monitoring of a Rockslide: The Torgiovannetto Quarry* —
  <https://link.springer.com/chapter/10.1007/978-3-319-09057-3_272>
- *Seismic monitoring system for landslide hazard assessment at the Peschiera
  Springs* — nanoseismic arrays, 397 located slope-instability events
