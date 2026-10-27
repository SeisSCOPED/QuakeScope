"""Wood-Anderson and velocity amplitudes on vanilla original picks (threshold 0.2),
from main's AmplitudeExtractor or from PR #71's fused/slim/C-kernel arm.

usage: run_amps.py ROOT ARM INV_DIR OUT_CSV FILE [FILE ...]
ROOT is the checkout whose sb_catalog is imported; ARM is "main" or "pr".
Each FILE is "sequence|station|path". Inventories are fetched once from
EarthScope FDSN (what the campaign reads) and cached as StationXML in INV_DIR.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT, ARM, INV_DIR, OUT = sys.argv[1:5]
ITEMS = [a.split("|") for a in sys.argv[5:]]
sys.path.insert(0, ROOT)

import obspy  # noqa: E402
import pandas as pd  # noqa: E402
import seisbench.models as sbm  # noqa: E402
from obspy.clients.fdsn import Client  # noqa: E402

from sb_catalog.src.amplitude_extractor import AmplitudeExtractor  # noqa: E402

if ARM == "pr":
    from sb_catalog.src import veliir

    import os
    if os.environ.get("VELIIR_SOURCE"):
        veliir.SOURCE = os.environ["VELIIR_SOURCE"]
    build = veliir.start_build()
    build.join(10)
    ext = AmplitudeExtractor(impl="fused", signal="slim")
    ext.warm_up()
else:
    ext = AmplitudeExtractor()

model = sbm.PhaseNet.from_pretrained("original")
model.default_args["P_threshold"] = model.default_args["S_threshold"] = 0.2


def inventory(sta: str, st: obspy.Stream) -> obspy.Inventory | None:
    path = Path(INV_DIR) / f"{sta}.xml"
    if not path.exists():
        net, code = sta.split(".")
        t0, t1 = min(tr.stats.starttime for tr in st), max(tr.stats.endtime for tr in st)
        try:
            inv = Client("EARTHSCOPE").get_stations(network=net, station=code, starttime=t0,
                                                    endtime=t1, level="response")
        except Exception as exc:  # noqa: BLE001
            print(f"{sta}: no inventory ({type(exc).__name__})", flush=True)
            path.write_text("")
            return None
        inv.write(str(path), format="STATIONXML")
    if path.stat().st_size == 0:
        return None
    return obspy.read_inventory(str(path))


rows = []
for seq, sta, fname in ITEMS:
    st = obspy.read(fname)
    inv = inventory(sta, st)
    if inv is None:
        continue
    picks = model.classify(st).picks
    t0 = time.perf_counter()
    wa = ext.extract_amplitudes(st, picks, inv)
    vel = ext.extract_velocity_amplitudes(st, picks, inv)
    dt = time.perf_counter() - t0
    for p, a, v in zip(picks, wa, vel):
        rows.append(dict(sequence=seq, station=sta, phase=p.phase,
                         time=pd.Timestamp(p.peak_time.datetime), conf=float(p.peak_value),
                         wa=a, vel=v))
    print(f"{seq}|{sta} {len(picks)} picks, amps {dt:.1f}s", flush=True)

if ARM == "pr":
    print("vel_counters", dict(ext.vel_counters), "slim", ext.slim_counters,
          "kernel", veliir.kernel().summary(), flush=True)
ext.close() if hasattr(ext, "close") else None
pd.DataFrame(rows).to_csv(OUT, index=False)
print("done", len(rows))
