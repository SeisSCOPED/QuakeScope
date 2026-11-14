"""convert_pnw_eqt_picks: decode a seisbench-style pick pickle without seisbench, refuse anything else."""
import io
import os
import pickle
import sys
import tarfile
import types

import numpy as np
import pyarrow.dataset as ds
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import convert_pnw_eqt_picks as cv  # noqa: E402


def _fake_modules():
    """Classes with the same module paths as seisbench Pick and obspy UTCDateTime."""
    sb = types.ModuleType("seisbench.util.annotations")
    ob = types.ModuleType("obspy.core.utcdatetime")

    class Pick:
        pass

    class UTCDateTime:
        pass

    Pick.__module__, Pick.__qualname__ = sb.__name__, "Pick"
    UTCDateTime.__module__, UTCDateTime.__qualname__ = ob.__name__, "UTCDateTime"
    sb.Pick, ob.UTCDateTime = Pick, UTCDateTime
    return {sb.__name__: sb, ob.__name__: ob}, Pick, UTCDateTime


def _pickle_picks(monkeypatch, peaks_ns):
    mods, Pick, UTC = _fake_modules()
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)

    def utc(ns):
        u = UTC.__new__(UTC)
        u.__dict__.update({"_UTCDateTime__precision": 6, "_UTCDateTime__ns": ns})
        return u

    picks = []
    for i, ns in enumerate(peaks_ns):
        p = Pick.__new__(Pick)
        p.__dict__.update({"trace_id": "UW.ASR.", "start_time": utc(ns - 10**8), "end_time": utc(ns + 10**8),
                           "peak_time": utc(ns), "peak_value": np.float32(0.25 + 0.1 * i), "phase": "PS"[i % 2]})
        picks.append(p)
    return pickle.dumps(picks)


def _tar(path, members):
    with tarfile.open(path, "w") as tf:
        for name, raw in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))


def test_round_trip(tmp_path, monkeypatch):
    t0 = 1609459200 * 10**9 + 3600 * 10**9          # 2021-01-01 01:00 UTC
    raw = _pickle_picks(monkeypatch, [t0, t0 + 60 * 10**9])
    _tar(tmp_path / "in.tar", {"001/UW.ASR.EH.001": raw})
    m = cv.convert(tmp_path / "in.tar", tmp_path / "out", 2021)
    assert (m["n_picks"], m["station_days"], m["n_problems"]) == (2, 1, 0)
    t = ds.dataset(tmp_path / "out" / "picks", format="parquet", partitioning="hive").to_table().to_pandas()
    assert list(t.pha) == ["P", "S"] and set(t.cha) == {"EH"} and set(t.tid) == {"UW.ASR."}
    assert str(t.peak.iloc[0]) == "2021-01-01 01:00:00"
    assert t.amp.isna().all() and t.amp_vel.isna().all()
    assert (t.start < t.peak).all() and (t.peak < t.end).all()


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_refuses_other_classes(tmp_path):
    _tar(tmp_path / "in.tar", {"001/UW.ASR.EH.001": pickle.dumps([_Evil()])})
    m = cv.convert(tmp_path / "in.tar", tmp_path / "out", 2021)
    assert m["n_picks"] == 0 and m["n_problems"] == 1
    assert m["problems"][0][0] == "unpickle" and "refused" in m["problems"][0][2]


def test_unpickler_rejects_builtins():
    with pytest.raises(pickle.UnpicklingError):
        cv.SafeUnpickler(io.BytesIO(pickle.dumps(_Evil()))).load()
