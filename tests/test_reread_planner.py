"""The re-read planner keeps production shard sizes: <= 40 stations, <= 800 station-days."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan_unread_repair import make_shards  # noqa: E402

LO = dt.date(1986, 1, 1)


def test_a_one_day_run_on_800_stations_is_split_at_40():
    runs = {("UW", 100, 101): [f"UW.S{i:03d}." for i in range(800)]}
    shards, holes = make_shards(runs, LO)
    assert not holes and len(shards) == 20
    assert max(len(s["stations"]) for s in shards) == 40


def test_station_day_cap_still_applies():
    runs = {("UW", 0, 20): [f"UW.S{i:03d}." for i in range(100)]}
    shards, _ = make_shards(runs, LO)
    assert max(s["n_station_days"] for s in shards) <= 800
    assert sum(s["n_station_days"] for s in shards) == 2000


def test_small_runs_are_held_out():
    shards, holes = make_shards({("UW", 5, 7): ["UW.A.", "UW.B."]}, LO)
    assert shards == [] and len(holes) == 2
