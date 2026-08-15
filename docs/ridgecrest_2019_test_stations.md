# Ridgecrest 2019 Smoke Test Stations

**Event**: M7.1 Ridgecrest earthquake, July 5, 2019, 17:33:50 UTC  
**Epicenter**: 35.705°N, 117.504°W (Kern County, California)  
**Depth**: ~6-7 km  
**Network**: CI (Southern California Seismic Network)  
**Data Source**: SCEDC S3 bucket (`scedc-pds/continuous_waveforms/`)

## Selected Stations (All Within 50 km)

Five SCSN stations selected for close-field smoke testing. All stations are within 50 km of the epicenter, providing strong, clear signals for robust model validation.

| # | Code | Network | Latitude | Longitude | Distance (km) | Azimuth | Description |
|---|------|---------|----------|-----------|---------------|---------|-------------|
| 1 | **DAM** | CI | 35.740 | -117.560 | **6.4** | **Very close** — Furnace Creek area, nearest to epicenter |
| 2 | **BOR** | CI | 35.650 | -117.330 | **16.9** | **Close** — Boron area, northern Kern County |
| 3 | **SYC** | CI | 35.810 | -117.860 | **34.2** | **Moderate-close** — Sycamore area, Sierra Nevada foothills |
| 4 | **TNP** | CI | 35.500 | -117.850 | **38.7** | **Moderate-close** — Timberlake Peak area, mountains |
| 5 | **PIG** | CI | 35.280 | -117.400 | **48.2** | **Moderate** — Pisgah/Mojave area, desert |

## Expected Phase Arrivals

Using typical Southern California crustal velocities (Vp = 5.8 km/s, Vs = 3.3 km/s):

### P-Wave Arrival Times (approx from event time)

| Station | Distance | P arrival time | Comments |
|---------|----------|-----------------|----------|
| DAM | 6.4 km | ~1.1s | Strong emergent motion |
| BOR | 16.9 km | ~2.9s | Clear, distinct P |
| SYC | 34.2 km | ~5.9s | Regional P |
| TNP | 38.7 km | ~6.7s | Clear phase |
| PIG | 48.2 km | ~8.3s | Moderate amplitude |

### S-Wave Arrival Times & P-S Intervals

| Station | Distance | P-S Interval | S arrival (from event) | Comments |
|---------|----------|--------------|------------------------|----------|
| DAM | 6.4 km | 0.7-1.5s | ~1.8-2.6s | Strong S phase, short duration |
| BOR | 16.9 km | 2.0-3.2s | ~5.0-6.1s | Clear S |
| SYC | 34.2 km | 4.0-6.0s | ~9.9-11.9s | Distinct S arrival |
| TNP | 38.7 km | 4.6-7.0s | ~11.3-13.7s | Moderate S amplitude |
| PIG | 48.2 km | 5.7-8.6s | ~14.0-16.9s | Clear S phase |

**All stations within close range** → all P-S intervals are short, all picks should cluster tightly around the event time.

## Validation Checklist

### Physical Plausibility
- [ ] **DAM**: P arrives before S, P-S ≈ 0.5-1.5s (very close → short interval)
- [ ] **ISA**: P arrives before S, P-S ≈ 6-9s (moderate distance)
- [ ] **SBC**: P arrives before S, P-S ≈ 8-12s (comparable to ISA—slightly farther)
- [ ] **GSC**: P arrives before S, P-S ≈ 18-22s (far distance → long interval)
- [ ] **PAS**: P arrives before S, P-S ≈ 30-40s (teleseismic → very long interval)

### Detection Consistency
- [ ] All 5 stations detect the M7.1 main event
- [ ] Picks cluster within ±5s of true event time (17:33:50 UTC)
- [ ] Both P and S picks present for main event on all stations
- [ ] No false positives in pre-event noise (before 17:33:40 UTC)

### Model Comparison (if running compare_phasenet_models.ipynb)
- [ ] v7 and original both detect main event
- [ ] v7 detects additional events (higher recall)?
- [ ] v7 and original agree on timing of main event (within ±1s)?
- [ ] No systematic biases in v7 picks vs. original

## Data Availability & Known Issues

### DAM (Furnace Creek area)
- **Status**: Excellent data for Ridgecrest (very close station)
- **SNR**: Very high (close to event)
- **Known issues**: None typical

### ISA (Anza array)
- **Status**: Reliable, part of long-running array
- **SNR**: High (mountain site)
- **Known issues**: Possible gaps in digital archive pre-2010

### SBC (Santa Barbara)
- **Status**: Good, coastal station
- **SNR**: Moderate (marine noise)
- **Known issues**: Occasional data gaps, tilt noise at long periods

### GSC (Goldstone)
- **Status**: Excellent, located at NASA tracking station
- **SNR**: High (remote desert site)
- **Known issues**: Instrument sensitivity changed c. 2006

### PAS (Pasadena/Caltech)
- **Status**: Excellent historical data
- **SNR**: Moderate in urban environment
- **Known issues**: Urban noise, especially at short periods

## Ridgecrest Earthquake Sequence Context

The 2019 Ridgecrest sequence included two major events:

1. **M6.4 foreshock** — July 4, 2019, 10:33 UTC (day 185)
   - Depth: ~8 km
   - ~35 km away from main epicenter

2. **M7.1 main shock** — July 5, 2019, 17:33 UTC (day 186) ← **Smoke test target**
   - Depth: ~6 km
   - This is the event we test on

**Note**: This notebook fetches data from July 5 (day 186) to capture the main event. If you want to include the foreshock, you'd need to also fetch July 4 data (day 185).

## References

- **USGS Event Page**: https://earthquake.usgs.gov/earthquakes/events/2019ca/
- **SCSN Network**: Southern California Seismic Network (CI), operated by Caltech
- **SCSN Station Map**: https://scsn.caltech.edu/
- **SCEDC Data**: Southern California Earthquake Data Center — public S3 bucket `scedc-pds/continuous_waveforms/`
- **Typical Southern California velocities**: Hadley & Kanamori (1977), Vp ≈ 5.8 km/s, Vs ≈ 3.3 km/s
