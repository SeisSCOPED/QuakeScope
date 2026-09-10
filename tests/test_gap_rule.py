"""A station-day is skipped only when nothing in it is long enough to pick.

The picker used to drop any channel set with more than 150 traces as "too
many gaps", counted on the stream as read from the archive's day objects, and
recorded the day as processed with zero picks. Two real SCEDC days show what
that counted:

  CI.SRT  2019-06-25  137 traces per component, every boundary an overlap of
                      duplicated records; obspy's non-destructive merge makes
                      it ONE trace of 8.67M samples. Skipped: 187 picks lost.
  CI.WRC2 2019-07-06  the Ridgecrest mainshock day, 2,445 genuine telemetry
                      gaps per component, median segment 2 s - and one segment
                      ten hours long. Skipped: thousands of picks lost.

SeisBench discards any trace shorter than one model window and picks the rest,
so the only day that is truly worthless is one with no trace that long. The
rule is now: merge first, skip only then, and write the skip to review/.
"""

import os
import sys

import numpy as np
import obspy
from obspy import UTCDateTime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.picker import fragmentation_note, merge_record_runs

WINDOW = 3001 / 100.0          # PhaseNet: 3001 samples at 100 Hz
DAY = 8_640_000                # samples in a day at 100 Hz


def _segments(bounds, rate=100.0, data=None, cha="HHZ"):
    """Traces covering [a, b) sample ranges of one day, sharing one array."""
    data = data if data is not None else np.arange(DAY, dtype=np.int32)
    st = obspy.Stream()
    for a, b in bounds:
        tr = obspy.Trace(data[a:b].copy())
        tr.stats.sampling_rate = rate
        tr.stats.starttime = UTCDateTime(2019, 7, 6) + a / rate
        tr.stats.network, tr.stats.station, tr.stats.channel = "CI", "TEST", cha
        st += tr
    return st


def test_duplicate_overlapping_records_merge_to_one_trace():
    """The CI.SRT shape: 137 record runs, each overlapping the last by a few
    identical samples. That is packaging, not data, and must not be counted."""
    step, ov = 63_000, 200
    bounds = [(max(0, i * step - ov), min(DAY, (i + 1) * step)) for i in range(138)]
    st = _segments(bounds)
    assert len(st) == 138
    merged = merge_record_runs(st)
    assert len(merged) == 1
    assert merged[0].stats.npts == DAY
    assert fragmentation_note(merged, WINDOW) is None


def test_exactly_adjacent_record_runs_merge_to_one_trace():
    """200 adjacent runs used to trip the >150 rule on a perfectly complete day."""
    edges = np.linspace(0, DAY, 201).astype(int)
    st = _segments(list(zip(edges[:-1], edges[1:])))
    assert len(st) == 200
    merged = merge_record_runs(st)
    assert len(merged) == 1 and merged[0].stats.npts == DAY
    assert fragmentation_note(merged, WINDOW) is None


def test_a_gappy_day_with_one_long_segment_is_picked():
    """The CI.WRC2 mainshock shape: thousands of 2 s fragments and a ten-hour
    segment. The fragments are useless; the ten hours are the mainshock."""
    bounds = [(i * 400, i * 400 + 200) for i in range(2_400)]      # 2 s on, 2 s off
    bounds.append((1_000_000, 1_000_000 + 3_600_000))                # ten hours
    st = merge_record_runs(_segments(bounds))
    assert len(st) == 2_401, "real gaps must survive the merge untouched"
    assert fragmentation_note(st, WINDOW) is None


def test_nothing_as_long_as_a_window_is_skipped_and_says_why():
    bounds = [(i * 400, i * 400 + 200) for i in range(3_000)]
    st = merge_record_runs(_segments(bounds))
    note = fragmentation_note(st, WINDOW)
    assert note is not None
    assert "3000 segments" in note and "2.0 s" in note and "30 s" in note


def test_the_window_is_compared_in_seconds_not_samples():
    """A 40 Hz BH trace of 1,300 samples is 32.5 s: one window, pickable,
    even though 1,300 < 3001."""
    st = _segments([(0, 1_300)], rate=40.0, cha="BHZ")
    assert fragmentation_note(st, WINDOW) is None
    st = _segments([(0, 1_100)], rate=40.0, cha="BHZ")          # 27.5 s
    assert fragmentation_note(st, WINDOW) is not None


def test_an_empty_stream_is_a_skip_not_a_crash():
    assert fragmentation_note(obspy.Stream(), WINDOW) == "no data"


def test_the_merge_changes_no_sample():
    step, ov = 63_000, 200
    bounds = [(max(0, i * step - ov), min(DAY, (i + 1) * step)) for i in range(138)]
    merged = merge_record_runs(_segments(bounds))
    assert np.array_equal(merged[0].data, np.arange(DAY, dtype=np.int32))
