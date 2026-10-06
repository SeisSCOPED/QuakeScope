"""The planner plans each FDSN epoch, not the hull of them.

XA.AZ01 has a 1993 deployment (Sep to Dec) and a 2017-2022 one. The station
table's start/end hull is 1993-09-19 to 2022-12-31, and planning the hull
invented 23 years of station-days the archive never held (2026-10-06 audit:
8,642 per station, every sampled day 404).
"""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.shard_planner import _operating_windows, parse_epochs, plan


def _table(epochs):
    return pd.DataFrame({"id": ["XA.AZ01."], "channels": ["BH,HH"],
                         "start_date": [dt.date(1993, 9, 19)],
                         "end_date": [dt.date(2022, 12, 31)], "epochs": [epochs]})


def test_parse_epochs():
    assert parse_epochs("1993-09-19/1993-12-31;2017-05-27/2022-12-31") == [
        (dt.date(1993, 9, 19), dt.date(1993, 12, 31)),
        (dt.date(2017, 5, 27), dt.date(2022, 12, 31))]
    assert parse_epochs(None) is None and parse_epochs("") is None
    assert parse_epochs(float("nan")) is None


def test_the_gap_between_epochs_is_not_planned():
    t = _table("1993-09-19/1993-12-31;2017-05-27/2022-12-31")
    shards = plan(t, dt.date(1993, 1, 1), dt.date(2023, 1, 1))
    sd = sum(s["n_station_days"] for s in shards)
    assert sd == 104 + (dt.date(2022, 12, 31) - dt.date(2017, 5, 27)).days + 1
    assert not any(s["start"].startswith("2000") for s in shards)


def test_without_epochs_the_hull_is_planned_as_before():
    t = _table(None)
    sd = sum(s["n_station_days"] for s in plan(t, dt.date(1993, 1, 1), dt.date(2023, 1, 1)))
    assert sd == (dt.date(2022, 12, 31) - dt.date(1993, 9, 19)).days + 1


def test_a_shard_straddling_an_epoch_boundary_counts_only_epoch_days():
    t = _table("2010-01-05/2010-01-10")
    shards = plan(t, dt.date(2010, 1, 1), dt.date(2010, 1, 31), day_group_size=20)
    assert [s["n_station_days"] for s in shards] == [6]
