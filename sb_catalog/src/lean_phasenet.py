"""
SeisBench's PhaseNet, as much of it as the fast classify path uses, without
importing `seisbench.models`.

`seisbench.models` itself is cheap (about 10 ms once torch is loaded), but
`seisbench.models.base` imports `obspy.signal.trigger`, and importing any
`obspy.signal` module runs `obspy.signal.__init__`. That pulls in scipy.signal,
scipy.stats and matplotlib, about 0.7 s on one core. A worker paid that before
its first classify, holding the GIL while its prefetch thread parsed the first
station-day. The fast path needs none of it: the fast graph
(`fast_forward.PhaseNetGraph`) and the fast annotate (`fast_annotate`) only
read PhaseNet's layers and call a handful of `WaveformModel` methods on the
stream and the batch. The amplitude passes do need scipy.signal and
obspy.signal, but only after the first classify. In lean mode they import them
then, on a thread of their own (`AmplitudeExtractor.start_warm_up`).

This module provides those pieces:

- `LeanPhaseNet`: PhaseNet's layers, under the same names, so its state_dict
  is SeisBench's. `from_cache` loads it from the same staged weight and JSON
  files that `PhaseNet.from_pretrained` reads.
- The `WaveformModel` / `PhaseNet` methods the fast annotate calls, copied from
  SeisBench 0.12.5 (`seisbench/models/base.py`, `seisbench/models/phasenet.py`).
- `GroupingHelper`, also copied from SeisBench 0.12.5.
- `trigger_onset`, copied from ObsPy 1.5.1 (`obspy/signal/trigger.py`). It is
  pure numpy.

The copies are verbatim, apart from names that point here. A stream the fast
annotate does not take goes to `classify`, which builds SeisBench's own
PhaseNet from the same files (importing `seisbench.models` then) and runs its
`classify` on the shared fast graph, as the full path does.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Callable, Optional

import numpy as np
import obspy
from .sb_lite import seisbench, stream_slice  # QS_SBLITE: the SeisBench originals under off
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version


def is_phasenet(model: Any) -> bool:
    """`type(model) is seisbench.models.PhaseNet` (or a LeanPhaseNet), without
    importing `seisbench.models`."""
    cls = type(model)
    return cls is LeanPhaseNet or (
        cls.__module__ == "seisbench.models.phasenet" and cls.__qualname__ == "PhaseNet")


# ---- obspy/signal/trigger.py (ObsPy 1.5.1) ------------------------------------------


def trigger_onset(charfct, thres1, thres2, max_len=9e99, max_len_delete=False):
    """
    Calculate trigger on and off times.

    Given thres1 and thres2 calculate trigger on and off times from
    characteristic function.

    This method is written in pure Python and gets slow as soon as there
    are more then 1e6 triggerings ("on" AND "off") in charfct --- normally
    this does not happen.

    Copied from ObsPy 1.5.1 `obspy.signal.trigger.trigger_onset`.
    """
    # 1) find indices of samples greater than threshold
    # 2) calculate trigger "of" times by the gap in trigger indices
    #    above the threshold i.e. the difference of two following indices
    #    in ind is greater than 1
    # 3) in principle the same as for "of" just add one to the index to get
    #    start times, this operation is not supported on the compact
    #    syntax
    # 4) as long as there is a on time greater than the actual of time find
    #    trigger on states which are greater than last of state an the
    #    corresponding of state which is greater than current on state
    # 5) if the signal stays above thres2 longer than max_len an event
    #    is triggered and following a new event can be triggered as soon as
    #    the signal is above thres1
    ind1 = np.where(charfct >= thres1)[0]
    if len(ind1) == 0:
        return []
    ind2 = np.where(charfct >= thres2)[0]
    #
    on = deque([ind1[0]])
    of = deque([-1])
    # determine the indices where charfct falls below off-threshold
    ind2_ = np.empty_like(ind2, dtype=bool)
    ind2_[:-1] = np.diff(ind2) > 1
    # last occurence is missed by the diff, add it manually
    ind2_[-1] = True
    of.extend(ind2[ind2_].tolist())
    on.extend(ind1[np.where(np.diff(ind1) > 1)[0] + 1].tolist())
    # include last pick if trigger is on or drop it
    if max_len_delete:
        # drop it
        of.extend([1e99])
        on.extend([on[-1]])
    else:
        # include it
        of.extend([ind2[-1]])

    # add last sample to ensure trigger gets switched off if ctf does not fall
    # below off-threshold before hitting the end
    of.append(len(charfct))
    #
    pick = []
    while on[-1] > of[0]:
        while on[0] <= of[0]:
            on.popleft()
        while of[0] < on[0]:
            of.popleft()
        if of[0] - on[0] > max_len:
            if max_len_delete:
                on.popleft()
                continue
            of.appendleft(on[0] + max_len)
        pick.append([on[0], of[0]])
    return np.array(pick, dtype=np.int64)


# ---- seisbench/models/base.py (SeisBench 0.12.5) ----------------------------------------


class GroupingHelper:
    """
    A helper class for grouping streams for the annotate function.
    Copied from SeisBench 0.12.5 `seisbench.models.base.GroupingHelper`.
    """

    def __init__(self, grouping: str) -> None:
        self._grouping = grouping

        self._grouping_functions = {
            "instrument": self._group_instrument,
            "channel": self._group_channel,
            "full": self._group_full,
        }

        if grouping not in self._grouping_functions:
            raise ValueError(f"Unknown grouping parameter '{self.grouping}'.")

    @property
    def grouping(self):
        return self._grouping

    def group_stream(
        self,
        stream: obspy.Stream,
        strict: bool,
        min_length_s: float,
        comp_dict: dict[str, int],
    ) -> list[list[obspy.Trace]]:
        return self._grouping_functions[self.grouping](
            stream, strict, min_length_s, comp_dict
        )

    @staticmethod
    def trace_id_without_component(trace: obspy.Trace):
        return f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"

    def _group_instrument(
        self, stream: obspy.Stream, *args, **kwargs
    ) -> list[list[obspy.Trace]]:
        pre_groups = defaultdict(list)
        for trace in stream:
            pre_groups[self.trace_id_without_component(trace)].append(trace)

        groups = []
        for group in pre_groups.values():
            groups.extend(self._group_full(obspy.Stream(group), *args, **kwargs))

        return groups

    def _group_channel(
        self, stream: obspy.Stream, *args, **kwargs
    ) -> list[list[obspy.Trace]]:
        pre_groups = defaultdict(list)
        for trace in stream:
            pre_groups[trace.id].append(trace)

        groups = []
        for group in pre_groups.values():
            groups.extend(
                self._group_full(obspy.Stream(group), *args, channel=True, **kwargs)
            )

        return groups

    def _group_full(
        self,
        stream: obspy.Stream,
        strict: bool,
        min_length_s: float,
        comp_dict: dict[str, int],
        channel: bool = False,
    ) -> list[list[obspy.Trace]]:
        intervals = self._get_intervals(
            stream, strict, min_length_s, comp_dict, channel=channel
        )

        return self._assemble_groups(stream, intervals)

    @staticmethod
    def _bin_search_idx(coords: list[obspy.UTCDateTime], t: obspy.UTCDateTime) -> int:
        mini = 0
        maxi = len(coords)

        while (maxi - mini) != 1:
            m = (maxi + mini) // 2
            if coords[m] > t:
                maxi = m
            else:
                mini = m
        return mini

    @staticmethod
    def _align_fractional_samples(stream: obspy.Stream) -> None:
        for trace in stream:
            ts = trace.stats.starttime.timestamp
            ts *= trace.stats.sampling_rate
            ts = np.round(ts) / trace.stats.sampling_rate

            trace.stats.t_offset = (
                trace.stats.starttime.timestamp - ts
            )  # How much we offset the data

            trace.stats.starttime = obspy.UTCDateTime(ts)

    def _get_intervals(
        self,
        stream: obspy.Stream,
        strict: bool,
        min_length_s: float,
        comp_dict: dict[str, int],
        channel: bool = False,
    ) -> list[tuple[list[str], obspy.UTCDateTime, obspy.UTCDateTime]]:
        if channel:
            strict = False

        self._align_fractional_samples(stream)

        # Do coordinate compression
        coords = np.unique(
            [trace.stats.starttime for trace in stream]
            + [trace.stats.endtime for trace in stream]
        )
        coords = sorted(list(coords))
        if len(coords) <= 1:
            return []

        if channel:
            n_comp = 1
            stations = sorted(list(set(trace.id for trace in stream)))
        else:
            n_comp = max(comp_dict.values()) + 1
            stations = sorted(
                list(set(self.trace_id_without_component(trace) for trace in stream))
            )

        sta_dict = {sta: i for i, sta in enumerate(stations)}

        covered = np.zeros((len(stations), n_comp, len(coords) - 1), dtype=bool)

        for trace in stream:
            p0 = self._bin_search_idx(coords, trace.stats.starttime)
            p1 = self._bin_search_idx(coords, trace.stats.endtime)

            if channel:
                comp_idx = 0
                sta_idx = sta_dict[trace.id]
            else:
                if trace.id[-1] not in comp_dict:
                    continue

                comp_idx = comp_dict[trace.id[-1]]
                sta_idx = sta_dict[self.trace_id_without_component(trace)]

            covered[sta_idx, comp_idx, p0:p1] = True

        covered = covered.all(axis=1) if strict else covered.any(axis=1)
        covered, coords = self._merge_intervals(covered, coords, min_length_s)

        intervals = []

        act = covered[:, 0]
        start_i = 0
        for i in range(1, covered.shape[1]):
            if np.all(act == covered[:, i]):
                # Same station coverage in both blocks, merge the intervals
                continue
            else:
                if act.any():
                    interval_stations = [
                        sta for sta, m in zip(stations, act) if m
                    ]  # Active stations in interval
                    t0 = coords[start_i]
                    t1 = coords[i]

                    intervals.append((interval_stations, t0, t1))

                start_i = i
                act = covered[:, i]

        if act.any():
            interval_stations = [
                sta for sta, m in zip(stations, act) if m
            ]  # Active stations in interval
            t0 = coords[start_i]
            t1 = coords[covered.shape[1]]
            intervals.append((interval_stations, t0, t1))

        return intervals

    def _merge_intervals(
        self, covered: np.ndarray, coords: list[obspy.UTCDateTime], min_length_s: float
    ) -> tuple[np.ndarray, list[obspy.UTCDateTime]]:
        # Goal: Maximize "stations * time" while ensuring no segment is too short
        # Use a greedy approach for maximizing which will not always lead to the globally optimal results
        # but usually to reasonably good results.
        has_warned = np.zeros(1, dtype=bool)

        def encompassing_interval(t0: obspy.UTCDateTime, t1: obspy.UTCDateTime):
            p0 = self._bin_search_idx(coords, t0)
            # Move index to actual left border of this segment
            while p0 > 0:
                if (covered[:, p0 - 1] == covered[:, p0]).all():
                    p0 -= 1
                else:
                    break

            p1 = self._bin_search_idx(coords, t1)
            if coords[p1] != t1:
                # This ensures that coords[p1] is greater or equal than t1
                p1 += 1
            # Move index to actual right border of this segment
            while p1 < covered.shape[-1] - 1:
                if (covered[:, p1 + 1] == covered[:, p1]).all():
                    p1 += 1
                else:
                    break

            return p0, p1

        def merge_costs(t0: obspy.UTCDateTime, t1: obspy.UTCDateTime) -> float:
            if t0 < coords[0] or t1 > coords[-1]:
                # The interval is not actually fully covered
                return np.inf

            p0, p1 = encompassing_interval(t0, t1)

            cost = 0

            merged_cover = np.all(
                covered[:, p0:p1], axis=1
            )  # Stations present in the whole interval
            if (
                not merged_cover.any()
            ):  # This is never better than just deleting the center interval
                return np.inf

            for p in range(p0, p1):
                if t0 - coords[p] > min_length_s:  # Left border profits from splitting
                    delta_t = t0 - (coords[p + 1] - min_length_s)
                elif (
                    coords[p + 1] - t1 > min_length_s
                ):  # Right border profits from splitting
                    delta_t = t1 - (coords[p] + min_length_s)
                else:
                    delta_t = coords[p + 1] - coords[p]
                delta_sta = np.sum(covered[:, p]) - np.sum(merged_cover)
                cost += delta_t * delta_sta

            return cost

        def merge_interval(t0: obspy.UTCDateTime, t1: obspy.UTCDateTime) -> None:
            p0, p1 = encompassing_interval(t0, t1)
            merged_cover = np.all(covered[:, p0:p1], axis=1)
            for p in range(p0, p1):
                if t0 - coords[p] > min_length_s:  # Left border profits from splitting
                    coords[p + 1] = t0  # End interval earlier
                elif (
                    coords[p + 1] - t1 > min_length_s
                ):  # Right border profits from splitting
                    coords[p] = t1  # Start interval later
                else:
                    covered[:, p] = merged_cover

        def fix_interval_if_necessary(act: np.ndarray, p0: int, p1: int):
            t0 = coords[p0]
            t1 = coords[p1]

            if act.any() and t1 - t0 < min_length_s:
                # Fixing required
                # Iterate over all reasonable merging times and find the cheapest
                # Reasonable merge intervals:
                # - every interval covering the target, starting either at a coord to the left
                #   or ending at a coord to the right
                # - if corner interval has at least min_length_s left, make new coord
                # - else, just merge intervals

                if not has_warned[0]:
                    has_warned[0] = True
                    seisbench.logger.warning(
                        "Parts of the input stream consist of fragments shorter than the number "
                        "of input samples or misaligned traces. Output might be empty."
                    )

                candidate_starts = []
                for p in range(p0, -1, -1):
                    if t1 - coords[p] > min_length_s:
                        break
                    candidate_starts.append(coords[p])
                for p in range(p0 + 1, len(coords)):
                    if coords[p] - t0 > min_length_s:
                        break
                    candidate_starts.append(coords[p] - min_length_s)

                candidate_starts = np.unique(candidate_starts)

                candidate_merge_costs = [
                    merge_costs(t, t + min_length_s) for t in candidate_starts
                ]

                if np.isinf(np.min(candidate_merge_costs)):
                    # Only delete if there is no other option.
                    # This improves coverage over time
                    act &= False
                else:
                    best_merge = np.argmin(candidate_merge_costs)
                    merge_interval(
                        candidate_starts[best_merge],
                        candidate_starts[best_merge] + min_length_s,
                    )

        act = covered[:, 0]
        start_i = 0

        for i in range(1, covered.shape[1]):
            if np.all(act == covered[:, i]):
                # Same station coverage in both blocks, merge the intervals
                continue
            else:
                fix_interval_if_necessary(act, start_i, i)

                start_i = i
                act = covered[:, i]

        fix_interval_if_necessary(act, start_i, covered.shape[1])

        return covered, coords

    @staticmethod
    def _assemble_groups(
        stream: obspy.Stream,
        intervals: list[tuple[list[str], obspy.UTCDateTime, obspy.UTCDateTime]],
    ) -> list[list[obspy.Trace]]:
        groups = []
        for stations, t0, t1 in intervals:
            sliced_stream = stream_slice(stream, t0, t1)
            group = []

            for station in stations:
                for trace in sliced_stream.select(id=station + "*"):
                    group.append(trace)
            groups.append(group)

        return groups


# ---- the model --------------------------------------------------------------------------


class LeanPhaseNet(nn.Module):
    """SeisBench's PhaseNet (layers, metadata and the annotate helpers the fast
    path calls), loaded from SeisBench's weight cache without `seisbench.models`."""

    #: WaveformModel._annotate_args with PhaseNet's overrides (SeisBench 0.12.5).
    _annotate_args = {
        "batch_size": ("Batch size for the model", 256),
        "overlap": (
            "Overlap between prediction windows. Values between 0 and 1 are treated as fractions of the window length. "
            "Values above 1 a sample counts. (only for window prediction models)",
            1500,
        ),
        "stacking": (
            "Stacking method for overlapping windows (only for window prediction models). "
            "Options are 'max' and 'avg'. ",
            "avg",
        ),
        "stride": ("Stride in samples (only for point prediction models)", 1),
        "strict": (
            "If true, only annotate if recordings for all components are available, "
            "otherwise impute missing data with zeros.",
            False,
        ),
        "flexible_horizontal_components": (
            "If true, accepts traces with Z12 components as ZNE and vice versa. "
            "This is usually acceptable for rotationally invariant models, "
            "e.g., most picking models.",
            True,
        ),
        "zerophase_resample": (
            "If true, the filter applied before resampling for anti-aliasing is zero-phase. Otherwise, uses causal "
            "filter. Note that using a different filter in application than in training might cause small out of "
            "distribution issues",
            True,
        ),
        "*_threshold": ("Detection threshold for the provided phase", 0.3),
        "blinding": (
            "Number of prediction samples to discard on each side of each window prediction",
            (0, 0),
        ),
    }

    #: PhaseNet._weight_warnings (SeisBench 0.12.5).
    _weight_warnings = [
        (
            "ethz|geofon|instance|iquique|lendb|neic|scedc|stead",
            "1",
            "The normalization for this weight version is incorrect and will lead to degraded performance. "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/seisbench/seisbench/pull/188 .",
        ),
        (
            "diting",
            "1",
            "This version of the Diting picker uses an incorrect sampling rate (100 Hz instead of 50 Hz). "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/JUNZHU-SEIS/USTC-Pickers/issues/1 .",
        ),
    ]

    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases="NPS",
        sampling_rate=100,
        norm="std",
        filter_factor: int = 1,
        component_order: Optional[list | str] = None,
        **kwargs,
    ):
        super().__init__()
        # PickBlue options
        for option in ("norm_amp_per_comp", "norm_detrend"):
            if option in kwargs:
                setattr(self, option, kwargs[option])
                del kwargs[option]
            else:
                setattr(self, option, False)
        if kwargs:
            # A model_args key this copy does not mirror: let the caller use SeisBench.
            raise ValueError(f"LeanPhaseNet does not mirror model args {sorted(kwargs)}")
        if self.norm_detrend or self.norm_amp_per_comp:
            raise ValueError("LeanPhaseNet does not mirror PickBlue normalization")

        # SeisBenchModel / WaveformModel state, as PhaseNet.__init__ sets it
        self._weights_docstring = None
        self._weights_version = None
        self._weights_metadata = None
        if component_order is None:
            self._component_order = seisbench.config["component_order"]
        else:
            self._component_order = component_order
        self.sampling_rate = sampling_rate
        self.output_type = "array"
        self.default_args = {}
        self.in_samples = 3001
        self.pred_sample = (0, 3001)
        self.filter_args = None
        self.filter_kwargs = None
        self._grouping = GroupingHelper("instrument")
        self.allow_padding = False
        self.labels = phases

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.filter_factor = filter_factor
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        self.inc = nn.Conv1d(
            self.in_channels,
            self.filters_root * filter_factor,
            self.kernel_size,
            padding="same",
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root * filter_factor, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root * filter_factor
        for i in range(self.depth):
            filters = int(2**i * self.filters_root) * filter_factor
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        for i in range(self.depth - 1):
            filters = int(2 ** (3 - i) * self.filters_root) * filter_factor
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

        #: Set by the picker: builds SeisBench's PhaseNet for the streams the fast
        #: annotate hands back (see `classify`).
        self.seisbench_factory: Optional[Callable[[], Any]] = None
        self._seisbench_model = None

    # ---- SeisBenchModel ----------------------------------------------------------------

    @property
    def name(self):
        return "PhaseNet"

    @property
    def weights_docstring(self):
        return self._weights_docstring

    @property
    def weights_version(self):
        return self._weights_version

    @property
    def component_order(self):
        return self._component_order

    @classmethod
    def from_cache(cls, name: str) -> Optional["LeanPhaseNet"]:
        """`PhaseNet.from_pretrained(name)` for weights already in SeisBench's cache.

        Returns None whenever from_pretrained would do more than read the cached
        files: there is no model directory, a cache file without a version
        suffix would be renamed, the v0 cache would be migrated, no local version
        exists, or the weight or JSON file is missing. The caller then uses
        SeisBench itself.
        """
        model_path = Path(seisbench.cache_model_root, "phasenet")
        if not seisbench.cache_model_root.is_dir() or not model_path.is_dir():
            return None
        files = [x.name for x in model_path.iterdir()]
        if any(f.endswith(".json") for f in files):
            return None  # _cleanup_local_repository would rename it
        configs = [x for x in files if x.startswith(f"{name}.json")]
        prefix_len = len(f"{name}.json.v")
        versions = sorted([config[prefix_len:] for config in configs])
        if len(versions) == 0:
            return None
        version_str = max(versions, key=version.parse)

        for name_regex, weight_version, warning_str in cls._weight_warnings:
            if not re.fullmatch(name_regex, name):
                continue
            if not weight_version == version_str:
                continue
            seisbench.logger.warning(f"Weight version warning: {warning_str}")

        path_json = model_path / f"{name}.json.v{version_str}"
        path_pt = model_path / f"{name}.pt.v{version_str}"
        if not (path_json.is_file() and path_pt.is_file()):
            return None

        # WaveformModel.load
        with open(path_json, "r") as f:
            weights_metadata = json.load(f)
        model_weights = torch.load(f"{path_pt}")

        model_args = weights_metadata.get("model_args", {})
        seisbench_requirement = weights_metadata.get("seisbench_requirement", None)
        if seisbench_requirement is not None:
            if version.parse(seisbench_requirement) > version.parse(
                seisbench.__version__
            ):
                raise ValueError(
                    f"Weights require seisbench version at least {seisbench_requirement}, "
                    f"but the installed version is {seisbench.__version__}."
                )
        try:
            model = cls(**model_args)
        except ValueError:
            return None
        model._weights_metadata = weights_metadata
        model._weights_docstring = weights_metadata.get("docstring", "")
        model._weights_version = weights_metadata.get("version", "1")
        default_args = weights_metadata.get("default_args", {})
        model.default_args.update(default_args)

        model.load_state_dict(model_weights)

        return model

    # ---- PhaseNet ------------------------------------------------------------------------

    def forward(self, x, logits=False):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(axis=-1, keepdims=True)
        if self.norm == "std":
            std = batch.std(axis=-1, keepdims=True)
            batch = batch / (std + 1e-10)
        elif self.norm == "peak":
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)

        return batch

    # ---- WaveformModel -------------------------------------------------------------------

    def _argdict_get_with_default(self, argdict, key):
        return argdict.get(key, self._annotate_args.get(key)[1])

    def _get_overlap(self, in_samples: int, argdict: dict) -> int:
        overlap = self._argdict_get_with_default(argdict, "overlap")
        if overlap < 1:
            return int(overlap * in_samples)
        else:
            return overlap

    def _get_in_pred_samples(self, group) -> tuple[int, tuple[int, int]]:
        pred_sample = (
            (0, self.pred_sample)
            if isinstance(self.pred_sample, int)
            else self.pred_sample
        )
        return self.in_samples, pred_sample

    def annotate_stream_pre(self, stream, argdict):
        # _filter_stream: filter_args and filter_kwargs are None for PhaseNet.
        if self.sampling_rate is not None:
            self.resample(
                stream,
                self.sampling_rate,
                zerophase=self._argdict_get_with_default(argdict, "zerophase_resample"),
            )
        return stream

    def annotate_stream_validate(self, stream, argdict):
        if self.sampling_rate is not None:
            if any(trace.stats.sampling_rate != self.sampling_rate for trace in stream):
                raise ValueError(
                    f"Detected traces with mismatching sampling rate. "
                    f"Expected {self.sampling_rate} Hz for all traces."
                )

        return self.sanitize_mismatching_overlapping_records(stream)

    @staticmethod
    def resample(stream: obspy.Stream, sampling_rate: float, zerophase: bool = True):
        del_list = []
        for i, trace in enumerate(stream):
            if trace.stats.sampling_rate == sampling_rate:
                continue
            if trace.stats.sampling_rate % sampling_rate == 0:
                trace.filter("lowpass", freq=sampling_rate * 0.5, zerophase=zerophase)
                trace.decimate(
                    int(trace.stats.sampling_rate / sampling_rate), no_filter=True
                )
            else:
                trace.resample(sampling_rate, no_filter=True)

        for i in del_list:
            del stream[i]

    @staticmethod
    def sanitize_mismatching_overlapping_records(stream):
        stream.merge(-1)  # Ensures overlapping matching traces are merged
        original_num_traces = len(stream)

        ids = defaultdict(list)
        for trace_idx, trace in enumerate(stream):
            ids[trace.id].append((trace_idx, trace))

        del_idx = []
        for traces in ids.values():
            # Go through all traces in order, keep stack of active elements.
            # If more than one element is active, all events are discarded until the stack is empty again.
            start_times = sorted(
                [
                    (trace.stats.starttime, trace.stats.endtime, trace_idx)
                    for trace_idx, trace in traces
                ],
                key=lambda x: x[0],
            )

            p = 0
            conflict = False
            active_traces = PriorityQueue()
            while p < len(start_times):
                # Note that as active_traces.queue is a heap, the first element is always the smallest one
                act_start_time, act_end_time, act_trace_idx = start_times[p]
                while (
                    not active_traces.empty()
                    and active_traces.queue[0][0] <= act_start_time
                ):
                    _, trace_idx = active_traces.get()
                    if conflict:
                        del_idx.append(trace_idx)

                if active_traces.qsize() == 0:
                    conflict = False

                active_traces.put((act_end_time, act_trace_idx))
                p += 1

                if active_traces.qsize() > 1:
                    conflict = True

            if conflict:
                while not active_traces.empty():
                    _, trace_idx = active_traces.get()
                    del_idx.append(trace_idx)

        for idx in sorted(del_idx, reverse=True):
            # Reverse order to ensure that the different deletions do not interfere with each other
            del stream[idx]

        if not original_num_traces == len(stream):
            seisbench.logger.warning(
                "Detected multiple records for the same time and component that did not agree. "
                "All mismatching traces will be ignored."
            )

        return stream

    def _build_comp_dict(
        self, stream: obspy.Stream, flexible_horizontal_components: bool
    ):
        if self._component_order is None:
            # Use the provided component
            component_order = stream[0].id[-1]
        else:
            component_order = self._component_order

        comp_dict = {c: i for i, c in enumerate(component_order)}
        matches = [
            ("1", "N"),
            ("2", "E"),
        ]  # Component regarded as identical if flexible_horizontal_components is True.
        if flexible_horizontal_components:
            for a, b in matches:
                if a in comp_dict:
                    comp_dict[b] = comp_dict[a]
                elif b in comp_dict:
                    comp_dict[a] = comp_dict[b]

            # Maps traces to the components existing. Allows to warn for mixed use of ZNE and Z12.
            existing_trace_components = defaultdict(list)

            for trace in stream:
                if trace.id[-1] in comp_dict and len(trace.data) > 0:
                    existing_trace_components[
                        self._grouping.trace_id_without_component(trace)
                    ].append(trace.id[-1])

            for trace, components in existing_trace_components.items():
                for a, b in matches:
                    if a in components and b in components:
                        seisbench.logger.warning(
                            f"Station {trace} has both {a} and {b} components. "
                            f"This might lead to undefined behavior. "
                            f"Please preselect the relevant components "
                            f"or set flexible_horizontal_components=False."
                        )

        return comp_dict, component_order

    # ---- the streams the fast annotate hands back -------------------------------------------

    def classify(self, stream: obspy.Stream, **kwargs):
        """SeisBench's own `classify`, on a SeisBench PhaseNet built (once, on
        first use) from the same files and sharing this model's fast graph."""
        if self._seisbench_model is None:
            if self.seisbench_factory is None:
                raise RuntimeError("LeanPhaseNet.classify needs a seisbench_factory")
            self._seisbench_model = self.seisbench_factory()
        return self._seisbench_model.classify(stream, **kwargs)
