"""The SeisBench pieces a lean picking worker uses, without importing the `seisbench` package
(pick_workload QS_SBLITE).

`import seisbench.util` runs `seisbench/util/annotations.py`, which imports pandas at module level:
about 0.35 s of a worker's startup on NFS, for a picking path that never builds a DataFrame. The
worker modules (picker, amplitude_extractor, fast_annotate, lean_phasenet) take the names they use
from here instead:

- `seisbench`: `cache_root`, `cache_model_root`, `config`, `__version__` and `logger`, computed as
  `seisbench/__init__.py` (SeisBench 0.12.5) computes them, config.json included;
- `sbu`: `ClassifyOutput`, `PickList` and `Pick`, copied from `seisbench/util/annotations.py`
  (pandas imported only inside `PickList.to_dataframe`);
- `stream_slice`, copied from `seisbench/util/trace_ops.py` with its `_round_py2`.

`StationFrame` is the station table's rows for the two reads `S3DataSource` makes of it, without a
DataFrame. The copies are verbatim apart from that one lazy import.

ARM is `QS_SBLITE` at import: "on" serves the copies; "off" (the default for every other entry
point; pick_workload sets the variable for its workers) re-exports the SeisBench originals.
"""

from __future__ import annotations

import json as _json
import logging as _logging
import math
import os as _os
import re
from pathlib import Path as _Path
from types import SimpleNamespace

import obspy

ARMS = ("off", "on")
ARM = _os.environ.get("QS_SBLITE", "off")

MAX_ENTRIES_STR = 10


# ---- seisbench/util/annotations.py (SeisBench 0.12.5) ------------------------------------

class ClassifyOutput(SimpleNamespace):
    """
    A general container to hold the outputs of the classify function of SeisBench models.
    This allows each model to provide a different set of outputs while keeping a consistent output type.
    For example, EQTransformer can output picks and detections, while PhaseNet only provides detections.

    :param creator: The model creating the output.
    :param kwargs: All outputs of the model
    """

    def __init__(self, creator: str, **kwargs):
        self.creator = creator
        super().__init__(**kwargs)

    @staticmethod
    def _raise_breaking_change_error():
        raise NotImplementedError(
            "This method is not implemented. "
            "Most likely you are trying to use an old interface to access the list of picks. "
            "To access the list of picks, just use `classify_output.picks` instead."
        )

    def __iter__(self):
        self._raise_breaking_change_error()

    def __getitem__(self, item):
        self._raise_breaking_change_error()

    def __reduce__(self):
        """
        Overwrite reduce function to allow pickling of the output.
        See https://stackoverflow.com/questions/51519351/typeerror-on-pickle-load-with-class-derived-from-simplenamespace
        for details on the underlying issue.
        """
        out = super().__reduce__()
        return out[0], (self.creator,), out[2]


class PickList(list):
    """
    A list of Pick objects with convenience functions for selecting and printing
    """

    def __str__(self) -> str:
        return f"PickList with {len(self)} entries:\n\n" + self._rep_entries()

    def _rep_entries(self):
        if len(self) <= MAX_ENTRIES_STR:
            return "\n".join([str(pick) for pick in self])
        else:
            return "\n".join(
                [str(pick) for pick in self[:3]]
                + ["..."]
                + [str(pick) for pick in self[-3:]]
            )

    def __repr__(self):
        return str(self)

    def select(
        self, trace_id: str = None, min_confidence: float = None, phase: str = None
    ):
        """
        Select specific picks. Only arguments provided will be used to filter.

        :param trace_id: A regular expression to match against the trace id.
                         The string is directly passed to the `re` module in Python, i.e.,
                         characters like dots need to be escapes and wildcards are represented
                         using `.*`.
        :param min_confidence: The minimum confidence values.
                               Picks without confidence value are discarded.
        :param phase: The phase of the pick. Only exact matches will be returned.
                      Picks without phase information are discarded.
        """
        filtered: list[Pick] = [x for x in self]
        if trace_id is not None:
            exp = re.compile(trace_id)
            filtered = [x for x in filtered if exp.fullmatch(x.trace_id)]

        if min_confidence is not None:
            filtered = [
                x
                for x in filtered
                if x.peak_value is not None and x.peak_value >= min_confidence
            ]

        if phase is not None:
            filtered = [x for x in filtered if x.phase == phase]

        return PickList(filtered)

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Converts the pick list to a Pandas DataFrame. This is useful to export the picks.

        For example, to export the picks as a csv, use ``picks.to_dataframe().to_csv("picks.csv")``.
        """
        import pandas as pd

        pick_df = []
        for p in self:
            pick_df.append(
                {
                    "station": p.trace_id,
                    "time": p.peak_time.datetime,
                    "start_time": p.start_time.datetime,
                    "end_time": p.end_time.datetime,
                    "probability": p.peak_value,
                    "phase": p.phase,
                    "polarity": p.polarity,
                    "polarity_probability": p.polarity_value,
                }
            )
        pick_df = pd.DataFrame(pick_df)

        if len(pick_df) > 0:
            pick_df.sort_values("time", inplace=True)
            pick_df.reset_index(inplace=True)

        return pick_df


class Pick:
    """
    This class serves as container for storing pick information.
    Defines an ordering based on start time, end time and trace id.

    :param trace_id: Id of the trace the pick was generated from
    :type trace_id: str
    :param start_time: Onset time of the pick
    :type start_time: UTCDateTime
    :param end_time: End time of the pick
    :type end_time: UTCDateTime
    :param peak_time: Peak time of the characteristic function for the pick
    :type peak_time: UTCDateTime
    :param peak_value: Peak value of the characteristic function for the pick
    :type peak_value: float
    :param phase: Phase hint
    :type phase: str
    :param polarity: Polarity hint
    :type polarity: str
    :param pol_value: Polarity value of the characteristic function for the pick
    :type pol_value: float
    """

    def __init__(
        self,
        trace_id,
        start_time,
        end_time=None,
        peak_time=None,
        peak_value=None,
        phase=None,
        polarity=None,
        polarity_value=None,
    ):
        self.trace_id = trace_id
        self.start_time = start_time
        self.end_time = end_time
        self.peak_time = peak_time
        self.peak_value = peak_value
        self.phase = phase
        self.polarity = polarity
        self.polarity_value = polarity_value

        if end_time is not None and peak_time is not None:
            if not start_time <= peak_time <= end_time:
                raise ValueError("peak_time must be between start_time and end_time.")

    def __lt__(self, other):
        """
        Compares start time, end time and trace id in this order.
        """
        if self.start_time < other.start_time:
            return True
        if self.end_time < other.end_time:
            return True
        if self.trace_id < other.trace_id:
            return True
        return False

    def __str__(self):
        parts = [self.trace_id]
        if self.peak_time is None:
            parts.append(str(self.start_time))
        else:
            parts.append(str(self.peak_time))

        if self.phase is not None:
            parts.append(str(self.phase))

        if self.polarity is not None:
            parts.append(str(self.polarity))

        return "\t".join(parts)


# ---- seisbench/util/trace_ops.py (SeisBench 0.12.5) --------------------------------------

def _round_py2(x: float) -> float:
    if x > 0:
        return float(math.floor((x) + 0.5))
    else:
        return float(math.ceil((x) - 0.5))


def _stream_slice(
    stream: obspy.Stream,
    starttime: obspy.UTCDateTime,
    endtime: obspy.UTCDateTime,
) -> obspy.Stream:
    """Slice an ObsPy Stream object between starttime and endtime.

    This is a lightweight implementation of `obspy.Stream.slice` that ensures with
    default paramters `Stream.slice(..., keep_empty_traces=False, nearest_sample=True)`.

    The performance gain of ~7x is achieved by avoiding deepcopies of traces and
    combining `Trace._ltrim` and `Trace._rtrim` into a single operation.

    Difference for edge cases compared to `Stream.slice` is that times are rounded
    with Python3 rounding (round half to even) instead of ObsPy's custom rounding
    (round half away from zero). This can lead to differences of 1 sample in edge
    cases.

    :param stream: The input ObsPy Stream.
    :type stream: obspy.Stream
    :param starttime: The start time for slicing.
    :type starttime: obspy.UTCDateTime
    :param endtime: The end time for slicing.
    :type endtime: obspy.UTCDateTime

    :return: The sliced ObsPy Stream.
    :rtype: obspy.Stream
    """
    if len(stream) == 0:
        raise ValueError("Stream is empty and cannot be sliced.")

    if starttime >= endtime:
        raise ValueError("starttime must be earlier than endtime.")

    first = stream[0].stats

    # select start/end time fitting to a sample point on the first trace
    delta = _round_py2((starttime - first.starttime) * first.sampling_rate)
    starttime = first.starttime + delta * first.delta

    delta = _round_py2((endtime - first.endtime) * first.sampling_rate)
    endtime = first.endtime + delta * first.delta  # delta is negative!

    out = obspy.Stream()
    for tr in stream:
        stats = tr.stats
        sampling_rate = stats.sampling_rate

        samples_start = int(_round_py2((starttime - stats.starttime) * sampling_rate))
        samples_start = max(samples_start, 0)
        if samples_start >= stats.npts - 1:
            continue
        slice_starttime = stats.starttime + samples_start * stats.delta

        samples_end = (
            int(_round_py2((endtime - slice_starttime) * sampling_rate))
            + samples_start
            + 1
        )
        samples_end = min(samples_end, stats.npts)

        n_samples = samples_end - samples_start
        if n_samples <= 0:
            continue

        sliced_trace = obspy.Trace(
            tr.data[samples_start:samples_end],
            {
                "network": stats.network,
                "station": stats.station,
                "location": stats.location,
                "channel": stats.channel,
                "starttime": slice_starttime,
                "sampling_rate": sampling_rate,
                "t_offset": stats.get("t_offset", 0.0),
            },
        )
        out.append(sliced_trace)
    return out


# ---- seisbench/__init__.py (SeisBench 0.12.5) --------------------------------------------

def _package_namespace() -> SimpleNamespace:
    """The `seisbench` attributes the worker modules read, with `seisbench/__init__.py`'s side
    effects (cache root created, config.json written when absent, the logger's handler)."""
    from importlib.metadata import version as _version

    cache_root = _Path(
        _os.getenv("SEISBENCH_CACHE_ROOT", _Path(_Path.home(), ".seisbench"))
    )
    if not cache_root.is_dir():
        cache_root.mkdir(parents=True, exist_ok=True)

    _config_path = cache_root / "config.json"
    if not _config_path.is_file():
        config = {"dimension_order": "NCW", "component_order": "ZNE"}
        with open(_config_path, "w") as _fconfig:
            _json.dump(config, _fconfig, indent=4, sort_keys=True)
    else:
        with open(_config_path, "r") as _fconfig:
            config = _json.load(_fconfig)

    logger = _logging.getLogger("seisbench")
    _ch = _logging.StreamHandler()
    _ch.setFormatter(
        _logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_ch)

    return SimpleNamespace(
        cache_root=cache_root,
        cache_data_root=cache_root / "datasets",
        cache_model_root=cache_root / "models" / "v3",
        cache_aux_root=cache_root / "auxiliary",
        config=config,
        __version__=_version("seisbench"),
        logger=logger,
    )


class StationFrame:
    """The station table's records, without pandas: `select(ids)` is the rows whose id is in `ids`,
    in table order (`frame[frame["id"].isin(ids)].reset_index(drop=True)`); after `set_index("id")`,
    `index` is the ids, `channels` (or `["channels"]`) the column's values in row order and
    `loc[id, "channels"]` one row's value - the reads S3DataSource makes of its metadata."""

    def __init__(self, records: list[dict], index: str | None = None):
        self.records = list(records)
        self._index = index

    def __len__(self) -> int:
        return len(self.records)

    def select(self, ids) -> "StationFrame":
        wanted = set(ids)
        return StationFrame([r for r in self.records if r["id"] in wanted])

    def set_index(self, column: str) -> "StationFrame":
        return StationFrame(self.records, index=column)

    @property
    def index(self) -> list:
        return [r[self._index] for r in self.records] if self._index else list(range(len(self.records)))

    @property
    def loc(self):
        rows = {r[self._index]: r for r in self.records}

        class _Loc:
            def __getitem__(self, key):
                row, column = key
                return rows[row][column]

        return _Loc()

    def __getitem__(self, column: str) -> list:
        return [r[column] for r in self.records]

    def __getattr__(self, column: str) -> list:
        if column.startswith("_") or column == "records":
            raise AttributeError(column)
        return [r[column] for r in self.records]


if ARM == "on":
    seisbench = _package_namespace()
    sbu = SimpleNamespace(ClassifyOutput=ClassifyOutput, PickList=PickList, Pick=Pick)
    stream_slice = _stream_slice
else:
    import seisbench  # noqa: F811
    import seisbench.util as sbu  # noqa: F811
    from seisbench.util.trace_ops import stream_slice  # noqa: F811
