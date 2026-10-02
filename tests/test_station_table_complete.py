"""The station table has to be complete with respect to the picks.

A pick names a station by `tid`. A reader selects stations from
`<campaign>/stations.parquet` and then fetches their partitions. If the
catalogue holds picks on a station the table does not list, those picks are
unreachable by anyone following the documented workflow, and they disappear
silently rather than raising.

That is not hypothetical. The `western-fill` campaigns built their own station
table, wrote picks into `western/picks/`, and their station rows never reached
`western/stations.parquet`: 1,383 stations and 178,416,797 picks, 10.3% of the
catalogue, invisible. Two outside groups rediscovered it from maps with a hole
over Utah before anyone here noticed.

These tests pin the two rules that keep it from recurring. The live check
against the bucket is `scripts/merge_station_tables.py --check`, which the
station-table workflow runs per campaign; it needs credentials and minutes, so
it does not belong here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from scripts.merge_station_tables import merge_epochs
from sb_catalog.src.s3_state import S3CampaignState


def _epochs() -> pd.DataFrame:
    """A station reconfigured twice, as FDSN returns it, plus a plain one."""
    return pd.DataFrame([
        dict(id="UU.ASI1.01", network_code="UU", station_code="ASI1", location_code="01",
             channels="HH", latitude=40.1, longitude=-111.2, elevation=1500.0,
             start_date="2010-01-01", end_date="2014-06-30"),
        dict(id="UU.ASI1.01", network_code="UU", station_code="ASI1", location_code="01",
             channels="EH,HN", latitude=40.1, longitude=-111.2, elevation=1500.0,
             start_date="2014-07-01", end_date="2021-12-31"),
        dict(id="TA.M12A.", network_code="TA", station_code="M12A", location_code="",
             channels="BH", latitude=44.0, longitude=-120.0, elevation=900.0,
             start_date="2008-01-01", end_date="2012-01-01"),
    ])


def test_epochs_collapse_to_one_row_per_station():
    """FDSN returns a row per epoch; the reader indexes by id and wants one.

    With a repeated id, `meta.loc[station, "channels"]` is a Series and the
    shard dies on `'Series' object has no attribute 'split'` deep in the read
    loop. It is released, the next worker rediscovers it, and the queue spins.
    On 2026-09-25 that cost 662 shards, four days and $51.
    """
    out = merge_epochs(_epochs())
    assert len(out) == 2, out
    assert out.id.is_unique
    row = out.set_index("id").loc["UU.ASI1.01"]
    assert set(row.channels.split(",")) == {"EH", "HH", "HN"}, row.channels
    assert str(row.start_date) == "2010-01-01"      # earliest epoch
    assert str(row.end_date) == "2021-12-31"        # latest epoch
    print("PASS  epochs union their bands and widen the window to one row per id")


def test_merging_two_identical_contributor_tables_is_idempotent():
    """western-fill and western-fill2 publish byte-identical station tables.

    Concatenating them without merging writes every id twice, which is the
    exact defect the write guard refuses. The merge has to absorb that.
    """
    fill = _epochs().drop_duplicates("id")
    doubled = pd.concat([fill, fill], ignore_index=True)
    out = merge_epochs(doubled)
    assert out.id.is_unique
    assert len(out) == len(fill)
    print("PASS  two identical contributor tables merge to one row per station")


def test_write_stations_refuses_a_repeated_id():
    """The guard fires at the producer, where it can still be fixed."""
    class _Fake(S3CampaignState):
        def __init__(self):
            pass

        def uri(self, name):
            return name

    with pytest.raises(ValueError, match="more than once"):
        _Fake().write_stations(_epochs())
    print("PASS  a table with a repeated id is refused before it is published")


def test_the_invariant_is_a_subset_relation():
    """Every station with picks must appear in the table; the reverse is fine.

    The table is an inventory of what was in scope, so it is legitimately
    larger than the set that produced picks. Only the other direction is a
    defect.
    """
    table = {"UU.ASI1.01", "TA.M12A.", "CI.PASC."}
    with_picks = {"UU.ASI1.01", "TA.M12A."}
    assert with_picks <= table

    orphaned = with_picks | {"UU.NEW."}
    assert not orphaned <= table
    assert orphaned - table == {"UU.NEW."}
    print("PASS  completeness is `stations_with_picks <= table`, not equality")


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\nall station-table completeness checks passed")
