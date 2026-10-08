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

A stream this path does not reproduce exactly goes to SeisBench's own
`classify` and is counted with the reason: masked data, a rate the model
resamples, several stations in one group, an unexpected dtype, a duplicate
block, or a NaN from the network. Gapped days are several blocks, as in
SeisBench, and stay on this path.
"""

from __future__ import annotations

import collections
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

    def __init__(self, model: sbm.WaveformModel, graph: torch.nn.Module) -> None:
        if not is_phasenet(model):
            raise ValueError(f"fast annotate is PhaseNet's; got {type(model).__name__}")
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

        with torch.inference_mode():
            for segments in self._segments(blocks, batch_size):
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


def apply(model: sbm.WaveformModel, graph: torch.nn.Module) -> FastAnnotator:
    """Route `model.classify` through a FastAnnotator on `graph`, in place."""
    annotator = FastAnnotator(model, graph)
    model.classify = annotator.classify
    model.fast_annotator = annotator
    return annotator
