"""
A vendored `classify` path for SeisBench's PhaseNet on the fast graph.

With the forward on `fast_forward.PhaseNetGraph`, about 40% of an isolated
PhaseNet `classify` is no longer the network: it is SeisBench's generic
annotate machinery around it. Per station-day that is a float64 copy of the
whole day (`stream_to_array`), a list of window views copied into every
batch (`np.array`), a list of per-window predictions copied again for the C
stacker, three strided channel copies to build the annotation traces, and a
`Trace.times()` array over 8.6M samples per phase, all on one thread.

For PhaseNet, whose windows are fixed (3001 samples, overlap 1500, blinding,
avg stacking), this module does the same work without those copies:

- the stream goes through SeisBench's own merge, validation and grouping
  (`annotate_stream_pre`, `annotate_stream_validate`, `group_stream`), on a
  copy that shares the trace data (none of those steps write samples);
- each group becomes one contiguous float32 array, filled the way
  `stream_to_array` fills its float64 one (the float32 cast the batch would
  get later is applied once, and gives the same values);
- windows are SeisBench's offsets, copied straight from the array into
  a torch-allocated batch; batches are the same 64-window chunks of the same
  window sequence, normalized by `annotate_batch_pre` itself;
- blinding and avg stacking are one overlap-add. Windows two apart never
  overlap (2 x stride >= the unblinded length), so the even and the odd
  windows of a run are each added with one strided view. Every sample then
  holds the same float32 sum, in the same order, that `stack_windows` forms,
  and is divided by the same count in float32;
- picks are taken from the stacked arrays with obspy's `trigger_onset` (the
  copy in lean_phasenet, so this module never imports obspy.signal) and
  the time arithmetic of `picks_from_annotations`, per phase, sorted as
  `classify_aggregate` sorts them. The noise channel is never stacked.

With glue="fused" (QS_GLUE in pick_workload.py) the same stacking runs with
fewer full passes over each batch:

- input: a batch that is one regular run of one block is normalized straight
  from the block (a strided view, no batch copy), with annotate_batch_pre's
  statistics - the float32 mean, then the unbiased std of the centred window
  (its sum of squares in float64) plus 1e-10 - and written once, in the
  compute dtype, into the channels-last layout the first conv reads. A batch
  with a tail window or windows of several blocks, a model without std
  normalization, and debug mode take the off arm's batch copy and
  annotate_batch_pre, and are counted by reason;
- output: the softmax is taken in float32 over the unblinded samples of the
  out conv's logits only, and a NaN check of each batch's P/S part replaces
  the scan of the stacked arrays;
- averaging: each sample's sum is divided by its window count per run of
  equal count (one strided division for the evenly spaced overlaps), instead
  of building, cumulating and dividing by a count array over the whole day.
  The quotients are the same float32 divisions, and a sum over one window
  is left as it is (x / 1 == x).

A stream this path does not reproduce exactly goes to SeisBench's own
`classify` and is counted with the reason: masked data, a rate the model
resamples, several stations in one group, an unexpected dtype, a duplicate
block, or a NaN from the network. Gapped days are several blocks, as in
SeisBench, and stay on this path.
"""

from __future__ import annotations

import collections
import math
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import numpy as np
import obspy
import seisbench
import seisbench.util as sbu
import torch
from numpy.lib.stride_tricks import as_strided

from .lean_phasenet import is_phasenet, trigger_onset

if TYPE_CHECKING:
    import seisbench.models as sbm

#: Sample dtypes whose float32 cast equals SeisBench's float64-then-float32 cast.
EXACT_DTYPES = (np.int16, np.int32, np.float32, np.float64)

#: How the passes around the graph run (QS_GLUE in pick_workload.py): "off" is
#: the round-22 path; "fused" normalizes from the block into the first conv's
#: layout, takes the softmax of the unblinded samples only and averages per run.
GLUE_ARMS = ("off", "fused")


class Fallback(Exception):
    """This stream goes to SeisBench's own classify; the message is the reason."""


class Block(NamedTuple):
    """One group of the stream as SeisBench's `stream_to_array` would build it."""

    data: np.ndarray  # (components, samples) float32
    start_time: obspy.UTCDateTime
    stations: list[str]
    offsets: np.ndarray  # window offsets, SeisBench's `_cut_fragments_array`
    n_regular: int  # offsets[:n_regular] are k * stride; the rest is the tail window


class FastAnnotator:
    """PhaseNet `classify` on `graph`, with SeisBench's results and fewer copies."""

    def __init__(self, model: sbm.WaveformModel, graph: torch.nn.Module, glue: str = "off") -> None:
        if not is_phasenet(model):
            raise ValueError(f"fast annotate is PhaseNet's; got {type(model).__name__}")
        if glue not in GLUE_ARMS:
            raise ValueError(f"glue arm {glue!r} is not one of {GLUE_ARMS}")
        argdict = model.default_args
        self.model = model
        self.graph = graph
        self.seisbench_classify = model.classify
        self.in_samples = model.in_samples
        pred_sample = model._get_in_pred_samples(None)[1]
        stacking = model._argdict_get_with_default(argdict, "stacking").lower()
        self.overlap = model._get_overlap(self.in_samples, argdict)
        self.stride = self.in_samples - self.overlap
        prenan, postnan = argdict.get("blinding", model._annotate_args.get("blinding")[1])
        self.lo = int(prenan)
        self.hi = self.in_samples - int(postnan) if postnan > 0 else self.in_samples
        problems = []
        if model.output_type != "array" or model._grouping.grouping != "instrument":
            problems.append("output type/grouping")
        if model.allow_padding or model.filter_args is not None or model.filter_kwargs is not None:
            problems.append("padding/filter")
        if tuple(pred_sample) != (0, self.in_samples):
            problems.append(f"pred_sample {pred_sample}")
        if stacking != "avg":
            problems.append(f"stacking {stacking}")
        if not 0 <= self.lo < self.hi or self.stride < 1 or 2 * self.stride < self.hi - self.lo:
            problems.append(f"windows (stride {self.stride}, unblinded {self.lo}:{self.hi})")
        if problems:
            raise ValueError(f"fast annotate does not cover this model setup: {problems}")
        self.labels = list(model.labels)
        self.pick_channels = [i for i, phase in enumerate(self.labels) if phase != "N"]
        self.counters = {"station_days": 0, "windows": 0, "batches": 0,
                         "fallbacks": collections.Counter()}
        self.glue = glue
        #: The glue paths fused_ops can take (the graph's logits, the model's std norm).
        self.glue_paths = ([] if glue == "off" else
                           (["input"] if getattr(model, "norm", None) == "std"
                            and not getattr(model, "norm_detrend", False)
                            and not getattr(model, "norm_amp_per_comp", False) else [])
                           + (["output", "average"] if hasattr(graph, "logits") else []))
        #: Batches through each fused path, and the batches that took the off arm's
        #: pass instead, by reason (QS_GLUE=fused; all zero with "off").
        self.glue_counters = {"input_fused": 0, "output_fused": 0, "average_fused_blocks": 0,
                              "input_fallbacks": collections.Counter(),
                              "output_fallbacks": collections.Counter()}
        #: Set to a dict to keep the stage intermediates of the next call (tests only).
        self.debug: Optional[dict] = None

    # ---- entry point ----------------------------------------------------------

    def classify(self, stream: obspy.Stream, **kwargs) -> sbu.ClassifyOutput:
        """`model.classify(stream)`, SeisBench's result."""
        argdict = self.model.default_args.copy()
        argdict.update(kwargs)
        try:
            output = self._classify(stream, argdict)
        except Fallback as reason:
            self.counters["fallbacks"][str(reason)] += 1
            return self.seisbench_classify(stream, **kwargs)
        self.counters["station_days"] += 1
        return output

    def summary(self) -> str:
        """One line for the group log."""
        c = self.counters
        fallbacks = ", ".join(f"{n} {reason}" for reason, n in sorted(c["fallbacks"].items()))
        return (f"{c['station_days']} station-days through the fast annotate, "
                f"{sum(c['fallbacks'].values())} fallbacks ({fallbacks or 'none'}), "
                f"{c['windows']} windows cut in {c['batches']} batches")

    def glue_summary(self) -> str:
        """The glue line for the group log: arm, active paths, per-path counters."""
        g = self.glue_counters

        def reasons(counter):
            return ",".join(f"{reason.replace(' ', '_')}:{n}"
                            for reason, n in sorted(counter.items())) or "-"
        return (f"glue={self.glue} paths={','.join(self.glue_paths) or '-'} "
                f"batches={self.counters['batches']} input_fused={g['input_fused']} "
                f"input_fallbacks={sum(g['input_fallbacks'].values())} ({reasons(g['input_fallbacks'])}) "
                f"output_fused={g['output_fused']} "
                f"output_fallbacks={sum(g['output_fallbacks'].values())} ({reasons(g['output_fallbacks'])}) "
                f"average_fused_blocks={g['average_fused_blocks']}")

    # ---- stream -> blocks -------------------------------------------------------

    def _classify(self, stream: obspy.Stream, argdict: dict) -> sbu.ClassifyOutput:
        model = self.model
        if len(stream) == 0:
            raise Fallback("empty stream")
        for trace in stream:
            if isinstance(trace.data, np.ma.MaskedArray):
                raise Fallback("masked data")
            if trace.stats.sampling_rate != model.sampling_rate:
                raise Fallback("resampled rate")
            if trace.data.dtype.type not in EXACT_DTYPES:
                raise Fallback(f"dtype {trace.data.dtype}")

        # annotate_async(copy=True): the copy only has to protect the caller's
        # traces, and the steps below change headers, never samples (merge(-1)
        # joins records into new arrays; grouping slices views).
        st = obspy.Stream([obspy.Trace(data=trace.data, header=trace.stats.copy())
                           for trace in stream])
        st.merge(-1)
        model.annotate_stream_pre(st, argdict)
        st = model.annotate_stream_validate(st, argdict)
        if len(st) == 0:
            raise Fallback("empty after validation")
        sampling_rate = st[0].stats.sampling_rate
        argdict["sampling_rate"] = sampling_rate
        flexible = model._argdict_get_with_default(argdict, "flexible_horizontal_components")
        comp_dict, _ = model._build_comp_dict(st, flexible)
        groups = model._grouping.group_stream(
            st,
            strict=model._argdict_get_with_default(argdict, "strict"),
            min_length_s=(self.in_samples - 1) / sampling_rate,
            comp_dict=comp_dict,
        )

        blocks, keys = [], set()
        for group in groups:
            block = self._block(group, argdict, flexible)
            if block is None:
                continue
            key = (block.start_time.timestamp, "__".join(block.stations))
            if key in keys:
                raise Fallback("duplicate block key")
            keys.add(key)
            blocks.append(block)
        if self.debug is not None:
            self.debug["blocks"] = blocks

        stacked = self._predict_and_stack(blocks, argdict)
        return self._picks(blocks, stacked, argdict)

    def _block(self, traces: list[obspy.Trace], argdict: dict, flexible: bool
               ) -> Optional[Block]:
        """`stream_to_array` in float32, then `_cut_fragments_array`'s offsets."""
        model = self.model
        comp_dict, comp_order = model._build_comp_dict(traces, flexible)
        key_of = model._grouping.trace_id_without_component
        stations = np.unique([key_of(trace) for trace in traces]).tolist()
        if len(stations) != 1:
            raise Fallback("several stations in a group")

        sampling_rate = traces[0].stats.sampling_rate
        t_start = min(trace.stats.starttime for trace in traces)
        t_end = max(trace.stats.endtime for trace in traces)
        n_samples = int((t_end - t_start) * sampling_rate) + 2  # avoid fractional error
        data = np.zeros((len(comp_order), n_samples), dtype=np.float32)
        t_offsets = []
        for trace in traces:
            offset = int((trace.stats.starttime - t_start) * sampling_rate)
            if trace.id[-1] not in comp_dict:
                continue
            data[comp_dict[trace.id[-1]], offset: offset + trace.data.size] = trace.data
            t_offsets.append(trace.stats.get("t_offset", 0.0))
        if not t_offsets:
            raise Fallback("group without model components")
        t_start += np.mean(t_offsets)
        data = data[:, :-1]  # Remove fractional error +1

        n = data.shape[-1]
        offsets = np.arange(0, n - self.in_samples + 1, self.stride)
        if len(offsets) == 0:
            seisbench.logger.warning(
                "Parts of the input stream consist of fragments shorter than the number "
                "of input samples. Output might be empty."
            )
            return None
        n_regular = len(offsets)
        if offsets[-1] + self.in_samples < n:
            offsets = np.concatenate([offsets, [n - self.in_samples]])
        return Block(data, t_start, stations, offsets, n_regular)

    # ---- windows -> stacked probabilities ------------------------------------------

    def _segments(self, blocks: list[Block], batch_size: int):
        """Yield each batch as [(block index, first window, end window)], in
        SeisBench's order: the windows of every block in turn, cut into
        batch_size chunks that run across block boundaries."""
        batch, filled = [], 0
        for b, block in enumerate(blocks):
            j, n = 0, len(block.offsets)
            while j < n:
                take = min(batch_size - filled, n - j)
                batch.append((b, j, j + take))
                filled += take
                j += take
                if filled == batch_size:
                    yield batch
                    batch, filled = [], 0
        if batch:
            yield batch

    def _runs(self, block: Block, j0: int, j1: int):
        """Split windows j0:j1 into (first, end, regular) runs: regular runs are
        spaced by stride, the tail window stands alone."""
        if j0 < block.n_regular:
            yield j0, min(j1, block.n_regular), True
        if j1 > block.n_regular:
            yield max(j0, block.n_regular), j1, False

    def _predict_and_stack(self, blocks: list[Block], argdict: dict) -> list[np.ndarray]:
        if self.glue == "fused":
            if self.debug is None and "output" in self.glue_paths:
                return self._predict_and_stack_fused(blocks, argdict)
            reason = "debug" if self.debug is not None else "no logits"
            n_batches = -(-sum(len(block.offsets) for block in blocks)
                          // self.model._argdict_get_with_default(argdict, "batch_size"))
            self.glue_counters["input_fallbacks"][reason] += n_batches
            self.glue_counters["output_fallbacks"][reason] += n_batches
        model, graph = self.model, self.graph
        W, stride, lo, hi = self.in_samples, self.stride, self.lo, self.hi
        V = hi - lo
        chans = (range(len(self.labels)) if self.debug is not None else self.pick_channels)
        c0, c1 = chans[0], chans[-1] + 1
        if list(chans) != list(range(c0, c1)):
            raise Fallback("non-contiguous pick channels")
        batch_size = model._argdict_get_with_default(argdict, "batch_size")
        sums = [np.zeros((c1 - c0, blk.offsets[-1] + W), dtype=np.float32) for blk in blocks]
        itemsize = np.dtype(np.float32).itemsize
        first_batch = True
        # Tail widening (tail_share): called before every batch with (batches
        # run, batches in this call), in this thread, so it can change this
        # thread's intra-op thread count between forwards. None when static.
        hook = getattr(graph, "tail_hook", None)
        if hook is not None:
            n_batches = -(-sum(len(block.offsets) for block in blocks) // batch_size)

        with torch.inference_mode():
            for batch_index, segments in enumerate(self._segments(blocks, batch_size)):
                if hook is not None:
                    hook(batch_index, n_batches)
                n = sum(j1 - j0 for _, j0, j1 in segments)
                # torch's allocator, as torch.as_tensor gives SeisBench's batch
                batch = torch.empty((n, blocks[segments[0][0]].data.shape[0], W),
                                    dtype=torch.float32)
                buf = batch.numpy()
                i = 0
                for b, j0, j1 in segments:
                    data = blocks[b].data
                    for r0, r1, regular in self._runs(blocks[b], j0, j1):
                        m = r1 - r0
                        if regular:
                            src = as_strided(data[:, r0 * stride:],
                                             shape=(m, data.shape[0], W),
                                             strides=(stride * itemsize, data.strides[0], itemsize),
                                             writeable=False)
                            buf[i:i + m] = src
                        else:
                            for k in range(r0, r1):
                                off = int(blocks[b].offsets[k])
                                buf[i + k - r0] = data[:, off:off + W]
                        i += m
                preprocessed = model.annotate_batch_pre(batch, argdict=argdict)
                if self.debug is not None and first_batch:
                    self.debug["batch0_input"] = batch.clone()
                    self.debug["batch0_normalized"] = preprocessed.clone()
                first_batch = False
                probs = graph(preprocessed).numpy()  # (n, labels, W) float32
                graph.batches += 1
                graph.windows += n
                self.counters["batches"] += 1
                self.counters["windows"] += n

                # blinding + avg stacking as an overlap-add of the unblinded part
                i = 0
                for b, j0, j1 in segments:
                    total = sums[b]
                    for r0, r1, regular in self._runs(blocks[b], j0, j1):
                        m = r1 - r0
                        if regular:
                            for parity in (0, 1):
                                part = probs[i + parity:i + m:2, c0:c1, lo:hi]
                                k = part.shape[0]
                                if k == 0:
                                    continue
                                base = (r0 + parity) * stride + lo
                                view = as_strided(total[:, base:], shape=(c1 - c0, k, V),
                                                  strides=(total.strides[0],
                                                           2 * stride * itemsize, itemsize))
                                view += part.transpose(1, 0, 2)
                        else:
                            for k in range(r0, r1):
                                off = int(blocks[b].offsets[k])
                                total[:, off + lo:off + hi] += probs[i + k - r0, c0:c1, lo:hi]
                        i += m

        for block, total in zip(blocks, sums):
            # windows covering each sample (small integers: exact in float32)
            count = np.zeros(total.shape[-1] + 1, dtype=np.float32)
            np.add.at(count, block.offsets + lo, 1)
            np.add.at(count, block.offsets + hi, -1)
            count = np.cumsum(count[:-1], dtype=np.float32)
            covered = count > 0
            np.divide(total, count, out=total, where=covered)
            total[:, ~covered] = np.nan
            if np.count_nonzero(np.isnan(total)) != total.shape[0] * np.count_nonzero(~covered):
                raise Fallback("NaN in model output")
        if self.debug is not None:
            self.debug["stacked"] = sums
        return sums

    def _normalized(self, src: torch.Tensor) -> torch.Tensor:
        """annotate_batch_pre(src) in the graph's dtype, channels-last (n, C, 1, W).

        The mean is the float32 mean annotate_batch_pre takes; its std is the
        unbiased std of the centred window, here sqrt(sum of squares / (W - 1))
        with the sum of squares accumulated in float64 (torch.std accumulates
        in double too), rounded to float32, plus 1e-10. The quotient is written
        once, in the compute dtype, in the layout the first conv reads.
        """
        n, channels, width = src.shape
        centred = src - src.mean(axis=-1, keepdims=True)
        std = (torch.linalg.vector_norm(centred, dim=-1, keepdim=True, dtype=torch.float64)
               / math.sqrt(width - 1)).float()
        out = torch.empty((n, 1, width, channels), dtype=self.graph.dtype)
        torch.div(centred, std + 1e-10, out=out.squeeze(1).permute(0, 2, 1))
        return out.permute(0, 3, 1, 2)

    def _predict_and_stack_fused(self, blocks: list[Block], argdict: dict) -> list[np.ndarray]:
        """_predict_and_stack with QS_GLUE=fused: the same windows, batches and
        overlap-add, with the input, softmax and averaging passes fused."""
        model, graph = self.model, self.graph
        W, stride, lo, hi = self.in_samples, self.stride, self.lo, self.hi
        V = hi - lo
        c0, c1 = self.pick_channels[0], self.pick_channels[-1] + 1
        if self.pick_channels != list(range(c0, c1)):
            raise Fallback("non-contiguous pick channels")
        batch_size = model._argdict_get_with_default(argdict, "batch_size")
        sums = [np.zeros((c1 - c0, blk.offsets[-1] + W), dtype=np.float32) for blk in blocks]
        tdata = [torch.from_numpy(blk.data) for blk in blocks]
        itemsize = np.dtype(np.float32).itemsize
        fuse_input = "input" in self.glue_paths
        g = self.glue_counters
        hook = getattr(graph, "tail_hook", None)
        if hook is not None:
            n_batches = -(-sum(len(block.offsets) for block in blocks) // batch_size)

        with torch.inference_mode():
            for batch_index, segments in enumerate(self._segments(blocks, batch_size)):
                if hook is not None:
                    hook(batch_index, n_batches)
                n = sum(j1 - j0 for _, j0, j1 in segments)
                runs = [(b, run) for b, j0, j1 in segments for run in self._runs(blocks[b], j0, j1)]
                if fuse_input and len(runs) == 1 and runs[0][1][2]:
                    b, (r0, _, _) = runs[0]
                    data = tdata[b]
                    x = self._normalized(data.as_strided((n, data.shape[0], W),
                                                         (stride, data.stride(0), 1), r0 * stride))
                    g["input_fused"] += 1
                else:
                    g["input_fallbacks"]["no std norm" if not fuse_input else
                                         "tail window" if any(not run[2] for _, run in runs)
                                         else "several blocks"] += 1
                    batch = torch.empty((n, blocks[segments[0][0]].data.shape[0], W),
                                        dtype=torch.float32)
                    buf = batch.numpy()
                    i = 0
                    for b, (r0, r1, regular) in runs:
                        data = blocks[b].data
                        m = r1 - r0
                        if regular:
                            buf[i:i + m] = as_strided(data[:, r0 * stride:],
                                                      shape=(m, data.shape[0], W),
                                                      strides=(stride * itemsize, data.strides[0],
                                                               itemsize),
                                                      writeable=False)
                        else:
                            for k in range(r0, r1):
                                off = int(blocks[b].offsets[k])
                                buf[i + k - r0] = data[:, off:off + W]
                        i += m
                    x = model.annotate_batch_pre(batch, argdict=argdict).to(graph.dtype)
                    x = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)
                # (n, labels, 1, V) float32: the unblinded samples only
                probs_t = torch.softmax(graph.logits(x)[..., lo:hi].float(), dim=1)
                if not math.isfinite(float(probs_t[:, c0:c1].sum())):
                    raise Fallback("NaN in model output")
                probs = probs_t.numpy()
                g["output_fused"] += 1
                graph.batches += 1
                graph.windows += n
                self.counters["batches"] += 1
                self.counters["windows"] += n

                i = 0
                for b, (r0, r1, regular) in runs:
                    total = sums[b]
                    m = r1 - r0
                    if regular:
                        for parity in (0, 1):
                            part = probs[i + parity:i + m:2, c0:c1, 0]
                            k = part.shape[0]
                            if k == 0:
                                continue
                            base = (r0 + parity) * stride + lo
                            view = as_strided(total[:, base:], shape=(c1 - c0, k, V),
                                              strides=(total.strides[0],
                                                       2 * stride * itemsize, itemsize))
                            view += part.transpose(1, 0, 2)
                    else:
                        for k in range(r0, r1):
                            off = int(blocks[b].offsets[k])
                            total[:, off + lo:off + hi] += probs[i + k - r0, c0:c1, 0]
                    i += m

        for block, total in zip(blocks, sums):
            self._average(block, total)
            g["average_fused_blocks"] += 1
        return sums

    def _average(self, block: Block, total: np.ndarray) -> None:
        """Divide each sample of `total` by the windows covering it, NaN where none
        does, in place: the off arm's float32 quotients, per run of equal count."""
        lo, hi = self.lo, self.hi
        n_win = len(block.offsets)
        # the count changes by +1 at each window's lo and by -1 at its hi
        edges = np.concatenate([block.offsets + lo, block.offsets + hi])
        steps = np.concatenate([np.ones(n_win, dtype=np.int64), -np.ones(n_win, dtype=np.int64)])
        order = np.argsort(edges, kind="stable")
        edges, level = edges[order], np.cumsum(steps[order])
        last = np.r_[edges[1:] != edges[:-1], True]  # the count after all changes at an edge
        starts = np.r_[0, edges[last]]
        ends = np.r_[edges[last], total.shape[-1]]
        counts = np.r_[0, level[last]]
        keep = ends > starts
        starts, ends, counts = starts[keep], ends[keep], counts[keep]
        for a, b in zip(starts[counts == 0], ends[counts == 0]):
            total[:, a:b] = np.nan
        sel = counts >= 2
        starts, lengths, counts = starts[sel], (ends - starts)[sel], counts[sel]
        if len(starts) == 0:
            return
        # runs of segments with one count and length, evenly spaced: one strided division each
        gaps = np.diff(starts)
        cont = np.zeros(len(starts), dtype=bool)
        cont[1:] = (lengths[1:] == lengths[:-1]) & (counts[1:] == counts[:-1])
        cont[2:] &= gaps[1:] == gaps[:-1]
        first = np.flatnonzero(~cont)
        itemsize = total.itemsize
        for r, j0 in enumerate(first):
            j1 = first[r + 1] if r + 1 < len(first) else len(starts)
            step = int(gaps[j0]) if j1 - j0 > 1 else 0
            view = as_strided(total[:, int(starts[j0]):],
                              shape=(total.shape[0], j1 - j0, int(lengths[j0])),
                              strides=(total.strides[0], step * itemsize, itemsize))
            np.divide(view, np.float32(counts[j0]), out=view)

    # ---- stacked probabilities -> picks ---------------------------------------------

    def _picks(self, blocks: list[Block], stacked: list[np.ndarray], argdict: dict
               ) -> sbu.ClassifyOutput:
        """`_predictions_to_stream` + `classify_aggregate`, without the traces."""
        model = self.model
        sampling_rate = argdict["sampling_rate"]
        pred_rate = sampling_rate * (self.in_samples / self.in_samples)
        c0 = 0 if self.debug is not None else self.pick_channels[0]
        annotations = []  # (block index, trimmed start, trimmed data per label index)
        for block, total in zip(blocks, stacked):
            valid = ~np.isnan(total[0])
            begin = int(np.argmax(valid))
            end = int(total.shape[-1] - np.argmax(valid[::-1]))
            pred_time = block.start_time + 0 / sampling_rate
            # the trace header's copy of starttime (obspy Stats.__setitem__)
            trimmed_start = obspy.UTCDateTime(ns=(pred_time + begin / pred_rate).ns)
            annotations.append((block, trimmed_start, total[:, begin:end]))
        if self.debug is not None:
            self.debug["annotations"] = [(start, pred_rate, data) for _, start, data in annotations]

        picks = sbu.PickList()
        for index, phase in enumerate(self.labels):
            if phase == "N":
                continue
            threshold = argdict.get(f"{phase}_threshold",
                                    model._annotate_args.get("*_threshold")[1])
            phase_picks = []
            for block, start, data in annotations:
                network, station, location = block.stations[0].split(".")[:3]
                trace_id = f"{network}.{station}.{location}"
                series = data[index - c0]
                for s0, s1 in trigger_onset(series, threshold, threshold / 2):
                    window = series[s0:s1 + 1]
                    s_peak = s0 + np.argmax(window)
                    phase_picks.append(sbu.Pick(
                        trace_id=trace_id,
                        start_time=start + np.float64(s0) / pred_rate,
                        end_time=start + np.float64(s1) / pred_rate,
                        peak_time=start + np.float64(s_peak) / pred_rate,
                        peak_value=np.max(window),
                        phase=phase,
                    ))
            picks += sbu.PickList(sorted(phase_picks))
        return sbu.ClassifyOutput(model.name, picks=sbu.PickList(sorted(picks)))


def apply(model: sbm.WaveformModel, graph: torch.nn.Module, glue: str = "off") -> FastAnnotator:
    """Route `model.classify` through a FastAnnotator on `graph`, in place."""
    annotator = FastAnnotator(model, graph, glue)
    model.classify = annotator.classify
    model.fast_annotator = annotator
    return annotator
