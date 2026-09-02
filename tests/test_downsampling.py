"""Downsampling at read time must not change the picks.

`annotate` resamples its own copy to the model's 100 Hz regardless, so doing it
earlier should be invisible to the model - but only if the method matches.
`downsample_to_target` therefore calls SeisBench's own resampler rather than
reimplementing it, and this pins that: if a future change swaps in a different
filter, or drops `zerophase=True`, the picks move and this fails.

Both branches of that resampler are covered: an integer ratio (200 -> 100,
lowpass then decimate) and a non-integer one (250 -> 100, FFT resample).
"""

import os
import sys

import numpy as np
import obspy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.s3_helper import (TARGET_SAMPLING_RATE,  # noqa: E402
                                      downsample_to_target)


def _synth(rate, seconds=300, seed=0):
    """Noise with a few impulsive arrivals, so there is something to pick."""
    rng = np.random.default_rng(seed)
    n = int(rate * seconds)
    base = rng.standard_normal(n) * 50
    for k in (0.2, 0.45, 0.7):
        i, w = int(n * k), int(rate * 2)
        base[i:i + w] += (np.exp(-np.linspace(0, 6, w))
                          * np.sin(np.linspace(0, 90, w)) * 4000)
    st = obspy.Stream()
    for c in "ZNE":
        tr = obspy.Trace((base + rng.standard_normal(n) * 20).astype(np.float32))
        tr.stats.network, tr.stats.station, tr.stats.location = "XX", "TEST", ""
        tr.stats.channel, tr.stats.sampling_rate = f"HH{c}", float(rate)
        st += tr
    return st


@pytest.mark.parametrize("rate", [200, 250, 500])
def test_downsample_reaches_target(rate):
    st = downsample_to_target(_synth(rate, seconds=60))
    assert {t.stats.sampling_rate for t in st} == {TARGET_SAMPLING_RATE}


@pytest.mark.parametrize("rate", [40, 50, 100])
def test_at_or_below_target_is_untouched(rate):
    """Never upsample. That would inflate the queue that item 0d is about, for
    data `annotate` will resample on its own copy anyway."""
    st = _synth(rate, seconds=60)
    before = [(t.stats.sampling_rate, len(t.data)) for t in st]
    downsample_to_target(st)
    assert [(t.stats.sampling_rate, len(t.data)) for t in st] == before


@pytest.mark.parametrize("rate", [200, 250])
def test_picks_are_unchanged(rate):
    sbm = pytest.importorskip("seisbench.models")
    model = sbm.PhaseNet.from_pretrained("jma_wc")

    def picks(stream):
        out = model.classify(stream, P_threshold=0.2, S_threshold=0.2).picks
        return sorted((p.phase, round(p.peak_time.timestamp, 3)) for p in out)

    native = _synth(rate)
    assert picks(native) == picks(downsample_to_target(native.copy()))
