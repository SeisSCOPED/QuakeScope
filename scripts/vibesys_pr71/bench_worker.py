"""One timing worker: pick (threshold 0.2) and measure amplitudes on local windows.

usage: bench_worker.py ROOT ARM INV_DIR OUT_JSON FILE [FILE ...]
ROOT: checkout to import sb_catalog from. ARM: "vanilla" (main's code) or "pr"
(PR #71 fp32 fast graph + fast annotate + fused/slim amplitudes + C kernel).
Each FILE is "sequence|station|path"; the station's StationXML is INV_DIR/<station>.xml.
"""

import time

T0 = time.perf_counter()
WALL0 = time.time()
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

ROOT, ARM, INV_DIR, OUT = sys.argv[1:5]
ITEMS = [a.split("|") for a in sys.argv[5:]]
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import obspy  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from sb_catalog.src.amplitude_extractor import AmplitudeExtractor  # noqa: E402

if ARM == "vanilla":
    import seisbench.models as sbm

    model = sbm.PhaseNet.from_pretrained("original")
    model.default_args["P_threshold"] = model.default_args["S_threshold"] = 0.2
    ext = AmplitudeExtractor()
else:
    from sb_catalog.src import fast_forward, picker, veliir
    from emu import EmuMKLDNN  # fp32: oneDNN's fused conv semantics, no oneDNN here

    if os.environ.get("VELIIR_SOURCE"):
        veliir.SOURCE = os.environ["VELIIR_SOURCE"]
    build = veliir.start_build()
    EMULATED = not fast_forward.FUSED_OPS or os.environ.get("BENCH_EMU") == "1"
    if EMULATED:
        fast_forward._MKLDNN = EmuMKLDNN
        d = list(fast_forward.PhaseNetGraph.__init__.__defaults__)
        d[1] = True
        fast_forward.PhaseNetGraph.__init__.__defaults__ = tuple(d)
    model = picker.S3MongoSBBridge.create_model("PhaseNet", "original", 0.2, 0.2, lean=True)
    model = picker.apply_classify_impl(model, "fp32", "fast", "polyphase")
    model = picker.apply_annotate_impl(model, "fast", "fused")
    build.join(10)
    ext = AmplitudeExtractor(impl="fused", signal="slim")
    ext.warm_up()

t_ready = time.perf_counter()
PICKS = []
stages = {"read": 0.0, "classify": 0.0, "wa": 0.0, "vel": 0.0}
picks_n = 0
samples = 0
for seq, sta, path in ITEMS:
    t = time.perf_counter()
    st = obspy.read(path)
    inv = obspy.read_inventory(os.path.join(INV_DIR, f"{sta}.xml"))
    stages["read"] += time.perf_counter() - t
    samples += sum(tr.stats.npts for tr in st)
    t = time.perf_counter()
    picks = model.classify(st).picks
    PICKS += [(sta, p.phase, str(p.peak_time), float(p.peak_value)) for p in picks]
    stages["classify"] += time.perf_counter() - t
    t = time.perf_counter()
    ext.extract_amplitudes(st, picks, inv)
    stages["wa"] += time.perf_counter() - t
    t = time.perf_counter()
    ext.extract_velocity_amplitudes(st, picks, inv)
    stages["vel"] += time.perf_counter() - t
    picks_n += len(picks)
json.dump(PICKS, open(OUT.replace(".json", ".picks.json"), "w"))
extra = {}
if ARM == "pr":
    extra = dict(emulated=EMULATED, fused_ops=fast_forward.FUSED_OPS, kernel=veliir.kernel().summary())
if hasattr(ext, "close"):
    ext.close()
json.dump(dict(arm=ARM, wall_start=WALL0, wall_end=time.time(), startup=t_ready - T0, total=time.perf_counter() - T0, picks=picks_n,
               samples=samples, items=len(ITEMS), **stages, **extra), open(OUT, "w"))
