"""Pick benchmark waveforms with one arm of PR #71 and write the picks as CSV.

Arms
  vanilla     sbm.PhaseNet.from_pretrained(w).classify, as the benchmark notebooks run it
  pr_fp32     PR #71 runner defaults (lean PhaseNet, fast graph, polyphase deconv,
              fast annotate, glue=fused) in fp32: what a CPU without AMX-BF16 runs
  pr_fp16     the same fast graph built in float16 (not offered by the PR), same emulation
  pr_bf16     the same in bf16, with oneDNN's fused conv emulated exactly:
              bf16 weights and input, fp32 accumulate + fp32 bias + ReLU, output
              rounded to bf16 after every layer (what AMX-BF16 computes, up to the
              fp32 summation order)

usage: run_arm.py PR_ROOT ARM WEIGHT OUT_CSV FILE [FILE ...]
Each FILE is "sequence|station|path".
"""

from __future__ import annotations

import os
import sys
import time

PR_ROOT, ARM, WEIGHT, OUT = sys.argv[1:5]
ITEMS = [a.split("|") for a in sys.argv[5:]]
FLOOR = 0.02

import obspy  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, PR_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emu import EmuMKLDNN  # noqa: E402  (oneDNN fused-conv numerics; shared with bench_worker.py)


def build(arm: str, weight: str):
    if arm == "vanilla":
        import seisbench.models as sbm

        return sbm.PhaseNet.from_pretrained(weight), None
    from sb_catalog.src import fast_forward, picker

    precision = "bf16" if arm == "pr_bf16" else "fp32"
    picker.amx_bf16_available = lambda: True
    # oneDNN's fused path on every arm: this torch build has no mkldnn, and the
    # graph's unfused fallback breaks the polyphase interleave (non channels-last).
    fast_forward._MKLDNN = EmuMKLDNN
    d = list(fast_forward.PhaseNetGraph.__init__.__defaults__)
    d[1] = True  # fused: fp32 bias, as on oneDNN
    fast_forward.PhaseNetGraph.__init__.__defaults__ = tuple(d)
    model = picker.S3MongoSBBridge.create_model("PhaseNet", weight, FLOOR, FLOOR, lean=True)
    lean = type(model).__name__
    if arm == "pr_fp16":
        # The PR offers fp32 and bf16 only; fp16 is the same fast graph built in
        # float16 (the graph takes any dtype), with the same fused-op emulation.
        model.eval()
        model.inference_precision = "fp16"
        graph = fast_forward.PhaseNetGraph(model, torch.float16, deconv="polyphase").eval()
        graph.batches = graph.windows = graph.station_days = 0
        model = picker.attach_fast_graph(model, graph)
    else:
        model = picker.apply_classify_impl(model, precision, "fast", "polyphase")
    model = picker.apply_annotate_impl(model, "fast", "fused")
    if isinstance(model, picker.LeanPhaseNet):
        import functools
        model.seisbench_factory = functools.partial(
            picker.S3MongoSBBridge._seisbench_twin, "PhaseNet", weight, FLOOR, FLOOR,
            model.fast_graph)
    info = dict(model_class=lean, precision=model.inference_precision,
                graph_dtype=str(model.fast_graph.dtype), fused=model.fast_graph.fused)
    return model, info


model, info = build(ARM, WEIGHT)
print(ARM, WEIGHT, info, "threads", torch.get_num_threads(), flush=True)

rows = []
t_all = time.perf_counter()
for seq, sta, path in ITEMS:
    st = obspy.read(path)
    t0 = time.perf_counter()
    out = model.classify(st, P_threshold=FLOOR, S_threshold=FLOOR)
    dt = time.perf_counter() - t0
    for p in out.picks:
        rows.append(dict(sequence=seq, weights=WEIGHT, arm=ARM, station=sta, phase=p.phase,
                         time=pd.Timestamp(p.peak_time.datetime), conf=float(p.peak_value),
                         start=pd.Timestamp(p.start_time.datetime),
                         end=pd.Timestamp(p.end_time.datetime)))
    print(f"{seq}|{sta} {len(out.picks)} picks {dt:.1f}s", flush=True)

fa = getattr(model, "fast_annotator", None)
if fa is not None:
    print("fast_annotator:", fa.summary(), flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"done {len(rows)} picks in {time.perf_counter() - t_all:.0f}s", flush=True)
