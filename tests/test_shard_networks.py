"""A shard covers exactly one network.

Restricted EarthScope credentials are scoped per network, so a shard that
straddles a boundary pays a second token exchange for the sake of whatever few
stations spilled over. Sorting station ids nearly achieves single-network
shards on its own - ids lead with the network code - but "nearly" leaves one
straddling shard per boundary, and the western campaign has 90+ networks.

This also protects the shard-id contract: `plan` must stay deterministic, or a
re-plan stops recognising work that is already done and the campaign repeats it.
"""

import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.shard_planner import plan

START = datetime.date(2019, 1, 1)
END = datetime.date(2019, 1, 20)


def _stations(spec):
    """spec: {network: n_stations} -> a metadata frame covering all of 2019."""
    rows = []
    for net, n in spec.items():
        for i in range(n):
            rows.append({
                "id": f"{net}.STA{i:03d}",
                "start_date": pd.Timestamp("2018-01-01"),
                "end_date": pd.Timestamp("2026-01-01"),
            })
    return pd.DataFrame(rows)


def _networks_of(shard):
    return {s.split(".")[0] for s in shard["stations"]}


def test_no_shard_mixes_networks():
    # Counts chosen so several networks fall short of a full group of 40 and
    # would otherwise be packed together with their neighbours.
    stations = _stations({"XD": 45, "ZI": 12, "ZG": 7, "1D": 3, "NP": 60})
    shards = plan(stations, START, END)
    assert shards
    for shard in shards:
        assert len(_networks_of(shard)) == 1, _networks_of(shard)


def test_every_station_is_still_planned():
    # Grouping per network must not drop anyone: the partial trailing group of
    # each network has to become its own shard rather than be discarded.
    spec = {"XD": 45, "ZI": 12, "ZG": 7, "1D": 3, "NP": 60}
    stations = _stations(spec)
    shards = plan(stations, START, END)
    planned = {s for shard in shards for s in shard["stations"]}
    assert planned == set(stations["id"])
    assert len(planned) == sum(spec.values())


def test_plan_is_deterministic():
    stations = _stations({"XD": 45, "ZI": 12, "NP": 60})
    a = plan(stations, START, END)
    # Same stations, different row order: ids are sorted within each network,
    # so the shard ids must not move.
    b = plan(stations.sample(frac=1, random_state=0), START, END)
    assert [s["shard_id"] for s in a] == [s["shard_id"] for s in b]


def test_group_size_is_respected_within_a_network():
    stations = _stations({"XD": 45})
    shards = plan(stations, START, END, station_group_size=40)
    sizes = sorted(len(s["stations"]) for s in shards)
    assert sizes == [5, 40]
