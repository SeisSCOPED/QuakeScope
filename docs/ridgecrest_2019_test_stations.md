# Ridgecrest 2019 smoke-test data

Reference for the stations and events used by
[`tutorials/phasenet_smoke_test_ridgecrest.ipynb`](../tutorials/phasenet_smoke_test_ridgecrest.ipynb)
and [`tutorials/compare_phasenet_models.ipynb`](../tutorials/compare_phasenet_models.ipynb).

Everything below was verified against the live SCEDC bucket and the USGS
catalog — station codes, coordinates, and object keys all resolve.

## Events

Both fall on 2019 day-of-year **187**, so they share the same SCEDC day files.

| Event | Origin (UTC) | Lat | Lon | Depth |
|---|---|---|---|---|
| M7.1 mainshock | 2019-07-06 03:19:53.04 | 35.770 | −117.599 | 8.0 km |
| M4.6 aftershock | 2019-07-06 08:32:57.55 | 35.639 | −117.491 | 3.1 km |

The M6.4 foreshock (2019-07-04 17:33:49, DOY **185**) is a different day and is
not used.

**Why two events.** A magnitude 7 ruptures for tens of seconds, so at these
distances its S arrival is buried inside ongoing rupture radiation and pickers
routinely miss it — observed here as P-only picks at every station. That makes
the mainshock good for the record section but useless for validating S. The
moderate aftershock has a short, impulsive source and yields clean P and S, so
it carries the S−P timing check.

## Stations

Network **CI** (SCSN), channel **HH** (broadband, 100 Hz). Distances are
epicentral, in km.

| Station | Lat | Lon | → mainshock | → aftershock |
|---|---|---|---|---|
| CLC | 35.8157 | −117.5975 | 5.1 | 21.8 |
| TOW2 | 35.8086 | −117.7649 | 15.6 | 31.1 |
| SRT | 35.6923 | −117.7505 | 16.2 | 24.2 |
| WRC2 | 35.9479 | −117.6504 | 20.3 | 37.2 |
| JRC2 | 35.9825 | −117.8089 | 30.2 | 47.7 |

WRC2 records the mainshock but yields no picks for the aftershock — expected
for the most distant station and a moderate event, and reported as such rather
than silently dropped.

## SCEDC object layout

SCEDC differs from NCEDC: the network is **not** a directory level, the day
folder uses an underscore, and the station code is padded to five characters.

```
scedc-pds/continuous_waveforms/<year>/<year>_<doy>/
    <net><sta padded to 5><cha><comp><loc padded to 3><year><doy>.ms
```

For example `CICLC__HHZ___2019187.ms`. Files are whole-day, roughly 20 MB per
channel, and readable anonymously — no AWS credentials needed.

Using the NCEDC pattern (`ncedc-pds/continuous_waveforms/<net>/...`) against
this bucket returns `FileNotFoundError` for every request.

## Expected result

Observed S−P should track the interval implied by hypocentral distance. From a
run with the published `original` weights:

| Station | Observed S−P | Predicted | Difference |
|---|---|---|---|
| CLC | 3.05 s | 2.63 s | +0.42 |
| TOW2 | 4.87 s | 3.72 s | +1.15 |
| SRT | 4.11 s | 2.91 s | +1.20 |
| JRC2 | 6.10 s | 5.69 s | +0.41 |

Observed running slightly long is normal: the prediction assumes a single layer
at Vp 6.0 / Vs 3.5 km/s (Vp/Vs = 1.71), while the real crust here is a little
higher.

## References

- USGS event pages: <https://earthquake.usgs.gov/earthquakes/eventpage/ci38457511/executive>
- SCEDC open data: <https://scedc.caltech.edu/data/cloud.html>
