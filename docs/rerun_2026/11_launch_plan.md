# 11 — Launch plan: five campaigns

The 2026 production run, split into five independent Batch campaigns. Each has
its own network set, its own weight, and its own database, so a problem in one
never contaminates another and any single campaign can be re-run alone.

Network lists live in [`sb_catalog/configs/networks/`](../../sb_catalog/configs/networks/)
as one code per line, and as `networks.json` for programmatic use.

---

## The five campaigns

| # | Campaign | Archive | Networks | Weight | Database |
|---|---|---|--:|---|---|
| 1 | **NCEDC onshore** | `ncedc-pds` | 7 | `jma_wc` | `quakescope_2026` |
| 2 | **SCEDC onshore** | `scedc-pds` | 1 | `jma_wc` | `quakescope_2026` |
| 3 | **EarthScope onshore** | EarthScope S3 | 420 | `jma_wc` | `quakescope_2026` |
| 4 | **EarthScope offshore** | EarthScope S3 | 26 | `obs` | `quakescope_2026` |
| 5 | **Western states** | NCEDC + SCEDC + EarthScope | by extent | `original` | `western2026` |

Campaigns 1–4 are the science run and share a database; `sb_runs` records which
weight produced which picks, so they remain separable. Campaign 5 is the
stakeholder deliverable and is isolated in its own database — see
[09_western_states_run.md](archive/09_western_states_run.md), whose weight and state
list this plan supersedes.

**The classifier is not part of any of these.** Submit without `--classifier`;
see the [runbook](README.md) and
[`../quakexnet_generalization_plan.md`](../quakexnet_generalization_plan.md).

---

## How onshore and offshore were split

The distinction matters because the ocean-bottom weight is trained on data the
land weights have never seen, and vice versa. The split was made from **station
elevation**, not from memory: a network counts as offshore when it has at least
three stations more than 50 m below sea level and at least half its stations are
submarine, surveyed across the Cascadia, California, Alaska, Atlantic, Gulf and
Hawaii margins.

Twenty-six networks qualify:

```
1V 2D 2F 3A 7A 7D 7K 7S 9A 9R NV OO X6 X9 XJ XO XZ YN YO YR YS Z5 Z6 ZF ZS ZU
```

That is `OO` (the cabled OOI observatory), `NV` (Ocean Networks Canada), and
twenty-four temporary OBS experiments including Cascadia Initiative (`7D`, 259
stations), AACSE (`XO`), ENAM (`YO`) and Blanco (`X9`). The deepest stations run
past 5,700 m.

**The split is by network, not by station.** A few onshore networks contain
island or lake-bottom stations that will be picked with a land weight. That is
an accepted approximation; doing it per station would mean a station-level
routing layer the pipeline does not currently have.

---

## Weight choices, and how confident to be about them

**Onshore — `jma_wc`.** Worth knowing this is close to a coin flip. Our own
`quakescope2026` is fine-tuned *from* `jma_wc`, and across four sequences,
matched on pick budget, the two are within a few points of each other with no
consistent ordering — see
[the sequence comparison](https://seisscoped.org/QuakeScope/phasenet_sequence_comparison.html).
`jma_wc` is the better-documented published parent; `quakescope2026` is ours and
has slightly better timing. Either is defensible. **What is not defensible is
switching between them mid-campaign**, because the two sit at different points
on the same recall curve at any shared threshold.

**Offshore — `obs`, i.e. `PickBlue(base="phasenet")`.** It is a PhaseNet, so it
drops into the pipeline unchanged, and it is trained on ocean-bottom data. In
[the OBS benchmark](https://seisscoped.org/QuakeScope/phasenet_obs_offshore_benchmark.html)
the ocean-bottom models lead the land models by 5–15 points of detection, with
no single one ahead everywhere. The hydrophone is *not* where the gain comes
from, so stations with a dead pressure channel are still worth picking.

**Western states — `original`.** This changes
[09_western_states_run.md](archive/09_western_states_run.md), which specifies
`instance`. The reason to change: `instance` has a genuine **ceiling** on dense
near-field aftershock sequences — at Ridgecrest it emits 246 S picks with its
threshold on the floor where the others reach 684 and 832, and no threshold
recovers that. For a stakeholder catalog over seismically active western states
that is the wrong failure mode. Confirm the substitution before submitting, as
the deliverable may have been specified against `instance`.

---

## Thresholds

Do **not** carry one threshold across campaigns. The land benchmark showed a
shared cutoff silently moves the operating point between weight sets and changes
catalog completeness with it, and `obs` ships different defaults again
(P 0.2 / S 0.1, already asymmetric).

Set each campaign's threshold from a **pick-budget target on a held-out week**
before launching the full range. The
[tier-2 smoke test](archive/10_tier2_smoke_test.md) is the right place to do it.

| Campaign | Starting point | Notes |
|---|---|---|
| 1–3 onshore | P 0.2 / S 0.2 | the picker defaults; tune from a held-out week |
| 4 offshore | P 0.2 / S 0.1 | the `obs` defaults, asymmetric by design |
| 5 western | P 0.2 / S 0.2 | match whatever the stakeholder catalog expects |

---

## Submitting

Each campaign is one `submit_helper` invocation. Networks come from the list
files:

```bash
NETS=$(grep -v '^#' sb_catalog/configs/networks/ncedc.txt | paste -sd, -)

python -m src.submit_helper pick \
    --start 2010.001 --end 2026.001 \
    --network "$NETS" \
    --database quakescope_2026 \
    --model PhaseNet --weight jma_wc \
    --region us-east-2
```

Repeat with `scedc.txt` and `earthscope_onshore.txt` for campaigns 2 and 3, and
for campaign 4:

```bash
NETS=$(grep -v '^#' sb_catalog/configs/networks/earthscope_offshore.txt | paste -sd, -)

python -m src.submit_helper pick \
    --start 2010.001 --end 2026.001 \
    --network "$NETS" \
    --database quakescope_2026 \
    --model PhaseNet --weight obs \
    --region us-east-2
```

Campaign 5 is selected geographically rather than by network. A single bounding
box over WA, OR, CA, NV, ID and WY also captures parts of AZ, UT, MT and CO:

```bash
python -m src.submit_helper pick \
    --start 2010.001 --end 2026.001 \
    --extent 31.5,49.2,-125.0,-104.0 \
    --database western2026 \
    --model PhaseNet --weight original \
    --region us-east-2
```

If the deliverable requires strict state membership, build a station list
instead and pass `--station_file`; `read_station_file` accepts a plain list of
`NET.STA.LOC` ids or a CSV with an `id` column.

---

## Order, and why

1. **Tier-2 smoke test first**, on three stations for one day, for every weight
   the launch uses — `jma_wc`, `obs` and `original`. It is an hour and it
   separates infrastructure faults from model faults.
   → [10_tier2_smoke_test.md](archive/10_tier2_smoke_test.md)
2. **Campaign 2 (SCEDC)** next. One network, one bucket, the archive we have
   tested most. If anything is wrong with the image or the database it shows up
   here cheapest.
3. **Campaign 1 (NCEDC)**, which adds a second bucket layout and the location-code
   discovery that layout needs.
4. **Campaign 4 (offshore)** before campaign 3, despite being smaller: it uses a
   different weight and a component convention the pipeline has never run in
   production, so it deserves attention rather than being buried behind 420
   networks.
5. **Campaign 3 (EarthScope onshore)**, the largest, once everything else is
   proven.
6. **Campaign 5 (western states)** whenever the stakeholder needs it; it is
   independent of the rest.

---

## Before pressing go

- [ ] `jma_wc`, `obs` and `original` all resolve inside the image
      (`sbm.PhaseNet.list_pretrained()` from within the container).
- [ ] Tier-2 smoke test green for each of the three weights.
- [ ] Thresholds set per campaign from a held-out week, not inherited.
- [ ] `--classifier` omitted everywhere.
- [ ] EarthScope token fresh — campaigns 3 and 4 need it; NCEDC and SCEDC are
      anonymous. → [05_submitting_jobs.md](archive/05_submitting_jobs.md)
- [ ] Databases exist: `quakescope_2026` and `western2026`.
- [ ] Spend caps and monitoring in place → [06_monitoring.md](archive/06_monitoring.md)

## A note on older data

If any campaign reaches back before roughly 2010, be aware the public buckets
are **not** a mirror of the FDSN archives for that period — on San Simeon's day
in 2003 the SCEDC bucket carries only `BH` and `LH` for stations whose `HH`
data the web service still serves. A campaign reading S3 over an older window
will quietly see less than a web-service query would. This is not a failure
mode that raises an error.
