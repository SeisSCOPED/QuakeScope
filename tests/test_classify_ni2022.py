"""picker.classify_ni2022 reproduces the 2022 PNW EQTransformer procedure
(single_station_detection.py, Yiyu Ni 2022-10-05) without touching the
caller's stream, which the amplitude pass still needs."""

import datetime
import os
import sys

import numpy as np
import obspy
import seisbench.util as sbu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.picker import classify_ni2022


class _Model:
    name = "fake"

    def __init__(self):
        self.calls = []

    def classify(self, s, **kw):
        self.calls.append((s.copy(), kw))
        pick = sbu.Pick("UW.TEST.", s[0].stats.starttime + 1, phase="P")
        return sbu.ClassifyOutput("fake", picks=sbu.PickList([pick]))


def _day():
    t0 = obspy.UTCDateTime(2023, 3, 2)
    rng = np.random.default_rng(0)
    n = 86400 * 100
    trend = np.linspace(0, 5000, n)
    return obspy.Stream([obspy.Trace((rng.standard_normal(n) * (10 ** k) + trend).astype("float64"),
                                     header=dict(network="UW", station="TEST", channel=f"HH{c}",
                                                 sampling_rate=100, starttime=t0))
                         for k, c in enumerate("ZNE")])


def test_hourly_detrend_normalize_and_args():
    st = _day()
    before = [tr.data.copy() for tr in st]
    m = _Model()
    out = classify_ni2022(m, st, datetime.date(2023, 3, 2), {"overlap": 5400})
    assert len(m.calls) == 24
    for s, kw in m.calls:
        assert kw == {"strict": False, "overlap": 5400}
        assert s[0].stats.endtime - s[0].stats.starttime <= 3600
        for tr in s:
            assert np.isclose(np.abs(tr.data).max(), 1.0)      # normalize()
            assert abs(tr.data[0]) < 1e-9 and abs(tr.data[-1]) < 1e-9         # detrend('simple'), ObsPy default
    assert len(out.picks) == 24
    for tr, b in zip(st, before):                               # caller's data untouched
        assert np.array_equal(tr.data, b)


def test_daily_mode_is_one_call():
    m = _Model()
    classify_ni2022(m, _day(), datetime.date(2023, 3, 2), hourly=False)
    assert len(m.calls) == 1
