"""A stream may hold both resampled and untouched traces, and obspy will not
merge across dtypes.

`downsample_to_target` resamples only traces above the target, and resampling
returns float64. A station whose sampling rate changes during the day therefore
ends up with float64 and int32 in one stream, and obspy raises

    TypeError: Data type differs: int32 vs float64

from inside SeisBench's annotate, where it reads as a model problem. It failed
693 western shards on 2026-09-04 - all in the tail of the campaign, because the
stations with a mid-day rate change are the ones left when the easy work is
done.
"""

import os
import sys

import numpy as np
import obspy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.s3_helper import TARGET_SAMPLING_RATE, downsample_to_target


def _tr(rate, dtype, npts=1000, cha="HHZ"):
    t = obspy.Trace(data=np.arange(npts, dtype=dtype))
    t.stats.sampling_rate = rate
    t.stats.channel = cha
    t.stats.station = "TEST"
    return t


def test_mixed_rates_end_up_with_one_dtype():
    # 200 Hz gets resampled to float64; 100 Hz is left alone as int32.
    st = obspy.Stream([_tr(200.0, np.int32), _tr(100.0, np.int32)])
    out = downsample_to_target(st, TARGET_SAMPLING_RATE)
    assert len({tr.data.dtype for tr in out}) == 1, (
        "a mixed-dtype stream is what obspy refuses to merge"
    )


def test_the_merge_that_used_to_raise_now_succeeds():
    """The actual failure, reproduced end to end."""
    a = _tr(200.0, np.int32, npts=2000)
    b = _tr(100.0, np.int32, npts=1000)
    b.stats.starttime = a.stats.starttime + 10
    st = downsample_to_target(obspy.Stream([a, b]), TARGET_SAMPLING_RATE)
    st.merge(method=0)          # raised TypeError before the fix


def test_untouched_streams_are_left_alone():
    # Nothing above the target means nothing to resample and nothing to
    # promote - the dtype must not change for its own sake, since that would
    # double memory on every ordinary station-day.
    st = obspy.Stream([_tr(100.0, np.int32), _tr(40.0, np.int32)])
    out = downsample_to_target(st, TARGET_SAMPLING_RATE)
    assert all(tr.data.dtype == np.int32 for tr in out)


def test_empty_traces_are_dropped():
    # A zero-length trace surfaces much later as "cannot reshape array of size
    # 0 into shape (0)", which names nothing useful.
    st = obspy.Stream([_tr(100.0, np.int32), _tr(100.0, np.int32, npts=0)])
    out = downsample_to_target(st, TARGET_SAMPLING_RATE)
    assert len(out) == 1 and out[0].stats.npts > 0
