import datetime
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "plan_pnw_band_repair",
    Path(__file__).resolve().parents[1] / "scripts" / "plan_pnw_band_repair.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

D = datetime.date


def o(tid, doy, status, detail=None):
    return {"tid": tid, "yr": 2025, "doy": doy, "status": status, "detail": detail}


def test_held_bands():
    assert m.held_bands("object holds BH,LH,VH") == {"BH", "LH", "VH"}
    assert m.held_bands(None) == set()
    assert m.held_bands("something else") == set()


def test_lost_days_rules():
    outs = [
        o("UW.VLM.", 1, "empty_read", "object holds EN,HH"),       # lost: HH in file, nothing picked
        o("IW.MFID.", 1, "empty_read", "object holds BH,LH"),      # sibling picked: not lost
        o("IW.MFID.00", 1, "loaded"),
        o("RE.JRDN.30", 1, "empty_read", "object holds EN,HN"),    # strong motion only: not lost
        o("UW.SOS.", 1, "no_data"),                                 # no file: not lost
        o("UU.MLI.", 2, "empty_read", "object holds EH,EN"),
        o("UU.MLI.01", 2, "done"),                                  # picked on an earlier attempt
    ]
    assert m.lost_days(outs) == {"UW.VLM": {D(2025, 1, 1)}}


def test_runs_split_on_gaps_and_length():
    days = {D(2025, 1, 1), D(2025, 1, 2), D(2025, 1, 3), D(2025, 1, 10)}
    assert m.runs(days, 20) == [(D(2025, 1, 1), D(2025, 1, 4)), (D(2025, 1, 10), D(2025, 1, 11))]
    assert m.runs(days, 2) == [(D(2025, 1, 1), D(2025, 1, 3)), (D(2025, 1, 3), D(2025, 1, 4)),
                               (D(2025, 1, 10), D(2025, 1, 11))]
