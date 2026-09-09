"""obspy reports an unmergeable stream as TypeError, and the guard must know.

The per-station-day guard in `picker._pick_data` exists so that one bad trace
costs one station-day and not the shard around it. Its first version caught
ValueError, ArithmeticError, IndexError and KeyError - and its own comment
named "Sampling rate differs" as the case it was written for. obspy raises
that as TypeError, from `Trace.__add__`, and SeisBench calls `merge(-1)`
before it annotates, so the error comes out of the model.

The guard therefore missed exactly its target. The last 16 western shards and
5 western-2026 shards each hold such a station-day; every worker sent to them
on 2026-09-08/09 failed all of them and exited, and the scheduled top-up sent
the next batch to do the same - 155 workers over 30 hours for zero shards.

Why a ONE-sample trace below. obspy's own `_merge_checks` refuses a stream
whose traces disagree on rate or dtype and `merge` then does nothing, which is
why the plain two-trace case never reaches `Trace.__add__`. A trace of a single
sample has no rate to check and slips through, and that is what the archives
serve: the last record of a day at a different rate. These tests pin the path
the container actually took, so the guard cannot drift from what obspy raises.
"""

import os
import sys

import numpy as np
import obspy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.picker import SIGNAL_FAULTS, is_signal_fault


def _tr(rate, dtype, npts=1000, t0=0.0):
    t = obspy.Trace(data=np.arange(npts, dtype=dtype))
    t.stats.sampling_rate = rate
    t.stats.network, t.stats.station, t.stats.channel = "NN", "TEST", "HHZ"
    t.stats.starttime += t0
    return t


def test_mixed_sampling_rates_come_out_of_merge_as_typeerror():
    """The western failure: 5 Hz beside 100 Hz on one channel id."""
    st = obspy.Stream([_tr(100.0, np.int32), _tr(5.0, np.int32, npts=1, t0=10.0)])
    with pytest.raises(TypeError, match="Sampling rate differs") as ei:
        st.merge(-1)                       # what SeisBench does before annotate
    assert is_signal_fault(ei.value), (
        "obspy raised TypeError; the guard must treat it as a bad trace")


def test_mixed_dtypes_come_out_of_merge_as_typeerror():
    """The western-2026 failure: float64 beside int32 at the same rate."""
    st = obspy.Stream([_tr(100.0, np.int32), _tr(100.0, np.float64, npts=1, t0=10.0)])
    with pytest.raises(TypeError, match="Data type differs") as ei:
        st.merge(-1)
    assert is_signal_fault(ei.value)


def test_the_except_clause_can_reach_typeerror():
    # is_signal_fault decides; the except tuple has to catch it first.
    assert TypeError in SIGNAL_FAULTS


def test_the_original_types_are_still_signal_faults():
    for exc in (ValueError("Selected corner frequency is above Nyquist."),
                ZeroDivisionError("float division by zero"),
                IndexError("index 0 is out of bounds"),
                KeyError("HHZ")):
        assert is_signal_fault(exc)


def test_a_typeerror_from_our_own_code_stays_loud():
    """A code bug must fail the shard and be seen, not be filed in review/."""
    for exc in (TypeError("unsupported operand type(s) for +: 'int' and 'str'"),
                TypeError("'NoneType' object is not subscriptable"),
                TypeError("classify() missing 1 required positional argument")):
        assert not is_signal_fault(exc)
