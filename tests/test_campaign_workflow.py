"""The workflow's gates, decided without AWS.

Each gate encodes a defect the 2026 campaigns paid for (doc 34): code columns
that disagreed with id, the hull of epochs planned, shards of 800 stations,
whole networks with no data planned at fleet scale, and a catalogue declared
complete while FDSN still served data for its unloaded days.
"""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from campaign_workflow import (image_is_current, residual_verdict, shard_problems,  # noqa: E402
                               table_problems, yield_verdict)


def _table(**over):
    d = dict(id=["SB.DBR1.00"], network_code=["SB"], station_code=["DBR1"], location_code=["00"],
             start_date=[dt.date(2018, 12, 6)], end_date=[dt.date(3000, 1, 1)],
             epochs=["2018-12-06/2599-12-31"])
    d.update(over)
    return pd.DataFrame(d)


def test_a_clean_table_passes():
    assert table_problems(_table()) == []


def test_float_damaged_codes_are_caught():
    assert any("disagree with id" in p for p in table_problems(_table(location_code=["0.0"])))


def test_missing_epochs_and_float_dates_are_caught():
    p = table_problems(_table(start_date=[2018.340]).drop(columns="epochs"))
    assert any("epochs" in x for x in p) and any("start_date" in x for x in p)


def test_shard_caps():
    ok = [dict(shard_id="a", stations=[f"UW.S{i}." for i in range(40)], n_station_days=800)]
    assert shard_problems(ok) == []
    big = [dict(shard_id="b", stations=[f"UW.S{i}." for i in range(800)], n_station_days=800)]
    mixed = [dict(shard_id="c", stations=["UW.A.", "CI.B."], n_station_days=40)]
    assert shard_problems(big) and shard_problems(mixed)


def test_yield_leaves_out_only_networks_with_enough_evidence():
    outs = ([dict(tid="NP.1.", status="no_data")] * 40 + [dict(tid="UU.A.", status="no_data")] * 5
            + [dict(tid="NC.X.", status="loaded")] * 3 + [dict(tid="NC.Y.", status="no_data")] * 7)
    v = yield_verdict(outs)
    assert v["exclude"] == ["NP"]             # UU has 5 samples: too few to drop
    assert round(v["loaded_fraction"], 3) == round(3 / 55, 3)


def test_residual_gate():
    assert residual_verdict([False] * 99 + [True])["pass"]          # 1%
    assert not residual_verdict([False] * 87 + [True] * 13)["pass"]  # 13%, the 2026-10-06 audit
    assert not residual_verdict([])["pass"]                          # no evidence is not a pass


def test_image_gate_knows_the_fix():
    import subprocess

    import pytest
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for c in ("6e99850", "9dc5aa9", "d0ccf9b"):
        if subprocess.run(["git", "-C", root, "cat-file", "-e", c], capture_output=True).returncode:
            pytest.skip("shallow checkout: history not available")
    assert image_is_current("6e99850")        # the too_big image descends from d0ccf9b
    assert not image_is_current("9dc5aa9")    # the image western ran on before the audit
