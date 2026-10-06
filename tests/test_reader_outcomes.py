"""Every planned station-day ends in an outcome, and the band is chosen per day.

Found 2026-10-06 auditing western-fill: 3.4 million planned station-days had
no trace in any manifest, and 13% of a sample had data at EarthScope. Two
causes, both in `S3DataSource.load_waveforms`:

* the band was chosen once per station from the table's `channels`, the union
  over all epochs, so a UU station listed `EH,EN,HH` read HH on every day and
  found nothing on the years it only recorded EH;
* a station-day that produced no stream - for that reason or any other - was
  logged and dropped, so "not read" looked exactly like "nothing there".
"""

import asyncio
import datetime
import os
import sys

import numpy as np
import obspy
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src import s3_helper
from sb_catalog.src.s3_helper import S3DataSource, _empty
from sb_catalog.src.worker import ShardIncomplete, check_outcome_coverage

D0 = datetime.date(2015, 1, 26)


def _trace(net, sta, loc, cha):
    t = obspy.Trace(np.zeros(100, dtype=np.int32))
    t.stats.update(dict(network=net, station=sta, location=loc, channel=cha,
                        sampling_rate=100.0))
    return t


class FakeHelper:
    """An EarthScope-shaped archive: one object per station-day, all bands in it."""

    def __init__(self, archive, dc="earthscope", list_error=None):
        self.archive, self.dc, self.list_error = archive, dc, list_error

    def get_prefix(self, net, year, doy):
        return f"{net}/{year}/{doy}"

    def get_filesystem(self, net, year):
        return None

    def list_day(self, net, fs, prefix):
        if self.list_error:
            raise self.list_error
        return [k for k in self.archive if k.startswith(prefix)]

    def get_data_center(self, net):
        return self.dc

    def get_s3_path(self, net, sta, loc, cha, year, day, c):
        return f"{net}/{year}/{day}/{sta}.{loc}.{cha}{c}"


class FakeDB:
    def __init__(self, done=()):
        self.done = set(done)

    def get_picks_record(self, station, day, channel, key=None):
        return {"_id": 1} if (station, day.year, int(day.strftime("%j")), channel) in self.done else None


def _source(stations, channels, archive, days=1, dc="earthscope", done=(),
            list_error=None, read=None):
    src = object.__new__(S3DataSource)
    src.start, src.end = D0, D0 + datetime.timedelta(days=days)
    src.stations = stations
    src.networks = sorted({s.split(".")[0] for s in stations})
    src.components, src.weight, src.limit_mb = "ZNE", "original", None
    src.db, src._logged, src.outcomes = FakeDB(done), set(), []
    src.s3helper = FakeHelper(archive, dc, list_error)
    src.meta = pd.DataFrame({"id": stations, "channels": channels}).set_index("id")

    async def _read(uri, net, station, day, sourcename=None):
        if read is not None:
            return read(uri, sourcename)
        st = archive.get(uri, obspy.Stream()).copy()
        if sourcename:
            # Same contract as S3DataSource._read_first_matching: the first
            # selector that matches wins and is named in `.band`; none
            # matching is an empty_read that lists the bands present.
            names = [sourcename] if isinstance(sourcename, str) else sourcename
            for name in names:
                band = name.split(".")[-1][:2]
                sel = st.select(channel=f"{band}?")
                if len(sel):
                    sel.band = band
                    return sel
            out = _empty("empty_read")
            out.present = sorted({t.stats.channel[:2] for t in st})
            return out
        return st
    src._read_with_timeout = _read
    if dc == "earthscope":
        src._generate_waveform_uris = lambda net, sta, loc, cha, day: [
            rf"{net}/{day:%Y}/{day:%j}/{sta}\.{loc}\.mseed"]
    return src


def _run(src):
    async def go():
        return [x async for x in src.load_waveforms()]
    return asyncio.run(go())


def _es_object(net, sta, loc, bands):
    return obspy.Stream([_trace(net, sta, loc, f"{b}Z") for b in bands])


def test_falls_back_to_the_band_that_existed_that_day():
    # UU.BMUT.01 2015-01-26: the table says EH,EN,HH; only EH was recording.
    key = f"UU/{D0:%Y}/{D0:%j}/BMUT.01.mseed"
    src = _source(["UU.BMUT.01"], ["EH,EN,HH"], {key: _es_object("UU", "BMUT", "01", ["EH"])})
    out = _run(src)
    assert len(out) == 1 and out[0][0][0].stats.channel == "EHZ"
    assert src.outcomes == [{"tid": "UU.BMUT.01", "yr": 2015, "doy": 26,
                             "status": "loaded", "cha": "EH"}]


def test_still_one_band_when_several_exist():
    key = f"UU/{D0:%Y}/{D0:%j}/BMUT.01.mseed"
    src = _source(["UU.BMUT.01"], ["EH,HH"], {key: _es_object("UU", "BMUT", "01", ["EH", "HH"])})
    out = _run(src)
    assert {t.stats.channel[:2] for t in out[0][0]} == {"HH"}


def test_every_station_day_gets_an_outcome():
    key = f"UU/{D0:%Y}/{D0:%j}/AAA..mseed"
    archive = {key: _es_object("UU", "AAA", "", ["HH"])}
    src = _source(["UU.AAA.", "UU.BBB.", "UU.CCC."], ["HH", "HH", "LH"], archive, days=2)
    _run(src)
    got = {(o["tid"], o["doy"]): o["status"] for o in src.outcomes}
    assert got == {("UU.AAA.", 26): "loaded", ("UU.AAA.", 27): "no_data",
                   ("UU.BBB.", 26): "no_data", ("UU.BBB.", 27): "no_data",
                   ("UU.CCC.", 26): "no_channel", ("UU.CCC.", 27): "no_channel"}


def test_object_present_but_no_pickable_band_is_empty_read():
    key = f"UU/{D0:%Y}/{D0:%j}/AAA..mseed"
    src = _source(["UU.AAA."], ["HH"], {key: _es_object("UU", "AAA", "", ["LH"])})
    _run(src)
    assert src.outcomes[0]["status"] == "empty_read"


def test_a_failed_read_is_recorded_with_its_reason():
    key = f"UU/{D0:%Y}/{D0:%j}/AAA..mseed"
    src = _source(["UU.AAA."], ["EH,HH"], {key: obspy.Stream()},
                  read=lambda uri, sel: _empty("timeout"))
    _run(src)
    # And it stops after the first band: a timed-out object will not answer
    # for the second band either.
    assert src.outcomes[0]["status"] == "timeout"


def test_a_denied_listing_is_recorded_per_station():
    src = _source(["TD.T1.", "TD.T2."], ["HH", "HH"], {},
                  list_error=s3_helper.EarthScopeNoAccess("403"))
    _run(src)
    assert [o["status"] for o in src.outcomes] == ["denied", "denied"]


def test_resume_counts_any_band_as_done():
    src = _source(["UU.AAA."], ["EH,HH"], {}, done={("UU.AAA.", 2015, 26, "EH")})
    _run(src)
    assert src.outcomes[0] == {"tid": "UU.AAA.", "yr": 2015, "doy": 26,
                               "status": "done", "cha": "EH"}


def test_per_channel_archives_fall_back_through_the_listing():
    # SCEDC/NCEDC: one object per channel, so the listing names the bands.
    def uri(cha, c):
        return f"CI/{D0:%Y}/{D0:%j}/ABC..{cha}{c}"
    archive = {uri("EH", c): obspy.Stream([_trace("CI", "ABC", "", f"EH{c}")]) for c in "ZNE"}
    src = _source(["CI.ABC."], ["EH,HH"], archive, dc="scedc")
    out = _run(src)
    assert len(out[0][0]) == 3 and src.outcomes[0]["cha"] == "EH"


def test_a_shard_with_a_day_unaccounted_for_does_not_complete():
    shard = {"shard_id": "s", "stations": ["UU.AAA.", "UU.BBB."],
             "start": "2015.026", "end": "2015.028"}
    outs = [{"tid": t, "yr": 2015, "doy": d, "status": "no_data"}
            for t in shard["stations"] for d in (26, 27)]
    assert check_outcome_coverage(shard, outs) == {"no_data": 4}
    with pytest.raises(ShardIncomplete, match="1 of 4"):
        check_outcome_coverage(shard, outs[:-1])


def test_no_failed_read_returns_an_untagged_empty_stream():
    # Static: a bare `return obspy.Stream()` in the read path is a failure that
    # would be recorded as "empty_read", i.e. as a fact about the archive.
    import inspect
    for fn in (S3DataSource._read_with_timeout, S3DataSource._read_waveform_from_s3):
        assert "return obspy.Stream()" not in inspect.getsource(fn), fn.__name__


def _mseed(bands, loc="01"):
    import io
    st = obspy.Stream([_trace("UU", "BMUT", loc, f"{b}Z") for b in bands])
    buf = io.BytesIO()
    st.write(buf, format="MSEED")
    return buf.getvalue()


def test_libmseed_raises_on_a_selector_that_matches_nothing():
    # The behaviour the 2026-10-06 dry test exposed: in production this raise
    # was swallowed into an empty stream and the day disappeared.
    import io
    raw = _mseed(["EH"])
    with pytest.raises(Exception):
        obspy.read(io.BytesIO(raw), format="MSEED", sourcename="UU.BMUT.01.HH?")


def test_first_matching_selector_is_decoded_from_the_same_bytes():
    raw = _mseed(["EH", "EN"])
    st = S3DataSource._read_first_matching(raw, ["UU.BMUT.01.HH?", "UU.BMUT.01.EH?"])
    assert st.band == "EH" and {t.stats.channel for t in st} == {"EHZ"}


def test_no_matching_selector_says_what_the_object_holds():
    raw = _mseed(["EN", "LH"])
    st = S3DataSource._read_first_matching(raw, ["UU.BMUT.01.HH?", "UU.BMUT.01.EH?"])
    assert len(st) == 0 and st.fault == "empty_read" and st.present == ["EN", "LH"]


def test_empty_read_detail_is_recorded():
    key = f"UU/{D0:%Y}/{D0:%j}/AAA..mseed"
    src = _source(["UU.AAA."], ["HH"], {key: _es_object("UU", "AAA", "", ["LH"])})
    _run(src)
    assert src.outcomes[0]["detail"] == "object holds LH"
