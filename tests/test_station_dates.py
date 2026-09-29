"""A station's operating window must survive the table it is stored in.

Station tables written before 2026-09-29 hold `start_date`/`end_date` as a
float `YYYY.DDD` with the day-of-year zero-padded to three digits. The float
itself is not lossy - every value in the three published tables round-trips -
but `str()` drops a trailing zero, so `strptime(str(2010.21), "%Y.%j")` reads
day 21 instead of day 210. `shard_planner._operating_windows` and
`utils.filter_station_by_start_end_date` both did that, and every station whose
day-of-year was divisible by ten was planned with the wrong window: always
short, because the misread day is always the smaller one. 837 western and 331
obs station-locations stopped early, 121,692 station-days never picked.

These checks pin the decoder, the direction of the old error, and the fact that
a planner now clips to the true window whichever encoding the table uses.
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sb_catalog.src.shard_planner import plan
from sb_catalog.src.utils import station_date


def test_float_day_of_year_is_decoded_numerically():
    # The three cases the old str() path got wrong, and the ones it got right.
    assert station_date(2010.21) == datetime.date(2010, 7, 29)    # day 210, not 21
    assert station_date(2010.1) == datetime.date(2010, 4, 10)     # day 100, not 1
    assert station_date(1937.23) == datetime.date(1937, 8, 18)    # day 230, not 23
    assert station_date(2010.021) == datetime.date(2010, 1, 21)
    assert station_date(2010.002) == datetime.date(2010, 1, 2)
    assert station_date(3000.001) == datetime.date(3000, 1, 1)    # still operating
    print("PASS  a day-of-year divisible by ten decodes to the right day")

    # The old path, kept here so the regression is visible rather than asserted
    # about in a comment.
    old = datetime.datetime.strptime(str(2010.21), "%Y.%j").date()
    assert old == datetime.date(2010, 1, 21) and old < station_date(2010.21)
    print("PASS  the old str()+strptime path is shown reading day 21, 189 days early")


def test_every_other_encoding_a_table_may_hold():
    assert station_date("2010.210") == datetime.date(2010, 7, 29)
    assert station_date(datetime.date(2010, 7, 29)) == datetime.date(2010, 7, 29)
    assert station_date(pd.Timestamp("2010-07-29")) == datetime.date(2010, 7, 29)
    assert station_date("2010-07-29") == datetime.date(2010, 7, 29)
    for junk in (None, float("nan"), pd.NaT, "", "not a date", 2010.400, 2010.0):
        assert station_date(junk) is None, junk
    print("PASS  dates, timestamps, strings and the float decode; junk and "
          "out-of-range day-of-year return None")


def _stations(start, end):
    return pd.DataFrame([{"id": "XX.AAA.", "network_code": "XX", "station_code": "AAA",
                          "location_code": "", "channels": "HH", "latitude": 0.0, "longitude": 0.0,
                          "elevation": 0.0, "start_date": start, "end_date": end}])


def test_planner_clips_to_the_true_window_in_either_encoding():
    c0, c1 = datetime.date(2010, 1, 1), datetime.date(2011, 1, 1)
    # Operating to day 210 of 2010. The old parse stopped at day 21.
    for start, end in ((2010.001, 2010.21), ("2010.001", "2010.210"),
                       (datetime.date(2010, 1, 1), datetime.date(2010, 7, 29))):
        sd = sum(s["n_station_days"] for s in plan(_stations(start, end), c0, c1))
        assert sd == 210, (start, end, sd)
    print("PASS  210 station-days planned from the float, the string and a real date")

    # A station with no usable window is planned for the whole campaign rather
    # than dropped: missing metadata costs a listing, never a missing station.
    sd = sum(s["n_station_days"] for s in plan(_stations(None, None), c0, c1))
    assert sd == 366, sd
    print("PASS  a station with no dates is still planned for the whole campaign")


def test_the_stored_table_keeps_the_still_operating_sentinel():
    """The conversion write_stations applies, which is where the sentinel dies.

    `pd.to_datetime` bounds a Timestamp to 1677..2262, so routing the decoded
    dates through it turns the year-3000 sentinel into NaT - the null this
    design exists to avoid, and the same "silently drops what is still
    recording" failure as the float bug. Caught in review on PR #42 before any
    table was written that way; the published tables were converted by a
    standalone script that assigns dates directly.
    """
    import tempfile

    import pyarrow.parquet as pq

    from sb_catalog.src.s3_state import OPEN_ENDED, prepare_station_dates

    df = pd.concat([_stations(2010.001, 3000.001),                    # still operating
                    _stations(2023.298, float("nan")).assign(id="XX.BBB."),  # no end epoch
                    _stations(2010.001, 2010.21).assign(id="XX.CCC.")],      # the misread value
                   ignore_index=True)
    out = prepare_station_dates(df)

    assert out.end_date.tolist() == [OPEN_ENDED, OPEN_ENDED, datetime.date(2010, 7, 29)]
    assert out.start_date.tolist()[0] == datetime.date(2010, 1, 1)
    print("PASS  year 3000 survives, a missing end epoch becomes it, day 210 is day 210")

    assert out.start_yearday.iloc[0] == 2010.001 and out.end_yearday.iloc[2] == 2010.21
    print("PASS  the original float is kept beside the date")

    path = tempfile.mkdtemp() + "/stations.parquet"
    out.to_parquet(path, index=False)
    schema = pq.read_schema(path)
    for c in ("start_date", "end_date"):
        assert str(schema.field(c).type) == "date32[day]", (c, schema.field(c).type)
    back = pd.read_parquet(path)
    assert back.end_date.iloc[0] == OPEN_ENDED and back.end_date.notna().all()
    print("PASS  written as date32 and read back with the sentinel intact")


if __name__ == "__main__":
    test_float_day_of_year_is_decoded_numerically()
    test_every_other_encoding_a_table_may_hold()
    test_planner_clips_to_the_true_window_in_either_encoding()
    test_the_stored_table_keeps_the_still_operating_sentinel()
    print("\nall station-date checks passed")
