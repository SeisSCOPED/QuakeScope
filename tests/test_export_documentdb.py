"""The 2025 DocumentDB -> 2026 Parquet export must be lossless and land in
the same layout, with the same schema, as the 2026 writer.

mongomock is not in the dev environment, so the database is a small in-memory
stand-in implementing only the calls the exporter makes. The transformation
functions are tested directly, without it.
"""

import datetime
import importlib.util
import json
import math
import os
import pathlib
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from bson import ObjectId

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.parquet_writer import PICK_SCHEMA  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "export_documentdb_to_parquet",
    pathlib.Path(__file__).parent.parent / "scripts/export_documentdb_to_parquet.py",
)
ex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ex)


# ------------------------------------------------------------ a tiny database

def _match(doc, filt):
    return all(doc.get(k) == v for k, v in filt.items())


def _project(doc, projection):
    if not projection:
        return dict(doc)
    if not any(projection.values()):                    # exclusion only
        return {k: v for k, v in doc.items() if projection.get(k, 1)}
    keep = [k for k, v in projection.items() if v and k != "_id"]
    out = {k: doc[k] for k in keep if k in doc}
    if projection.get("_id", 1) and "_id" in doc:
        out["_id"] = doc["_id"]
    return out


class FakeCursor(list):
    def hint(self, name):
        return self

    def limit(self, n):
        return FakeCursor(self[:n])


class FakeCollection:
    def __init__(self, docs=(), indexes=None):
        self.docs = [dict(d) for d in docs]
        self.indexes = indexes or {}

    def find(self, filt=None, projection=None, batch_size=None):
        return FakeCursor(_project(d, projection) for d in self.docs if _match(d, filt or {}))

    def find_one(self, filt, projection=None):
        hits = self.find(filt, projection)
        return hits[0] if hits else None

    def count_documents(self, filt):
        return sum(_match(d, filt) for d in self.docs)

    def estimated_document_count(self):
        return len(self.docs)

    def distinct(self, field):
        return sorted({d.get(field) for d in self.docs if d.get(field) is not None})

    def index_information(self):
        return self.indexes

    def aggregate(self, pipeline):
        raise NotImplementedError


class FakeDB(dict):
    def __getitem__(self, name):
        return self.setdefault(name, FakeCollection())

    def list_collection_names(self):
        return list(self)


T0 = datetime.datetime(2019, 7, 31, 23, 59, 58, 120000)   # ms precision, as BSON stores
RID = ObjectId("65f0a1b2c3d4e5f601234567")


def _pick(tid, peak, pha="P", cha="HH", conf=0.73, amp=1.5e-6, rid=RID):
    return {
        "_id": ObjectId(), "tid": tid, "cha": cha, "pha": pha,
        "start": peak - datetime.timedelta(seconds=0.4), "peak": peak,
        "end": peak + datetime.timedelta(seconds=0.4),
        "conf": conf, "amp": amp, "rid": rid,
    }


def _db():
    picks = [
        _pick("CI.CLC.", T0),                                          # July
        _pick("CI.CLC.", T0 + datetime.timedelta(seconds=5), pha="S"),  # August
        _pick("CI.CLC.", T0 + datetime.timedelta(days=1)),              # August
        _pick("CI.WRC2.", T0, amp=float("nan")),                        # no response
        _pick("CI.WRC2.", T0, cha="HN"),                                # second band, same day
        _pick("NC.KCPB.00", T0, conf=0.21),
    ]
    db = FakeDB()
    db["picks"] = FakeCollection(picks, {"_id_": {}, "pick_idx": {}})
    db["picks_record"] = FakeCollection(
        [
            {"_id": ObjectId(), "tid": "CI.CLC.", "cha": "HH", "yr": 2019, "doy": 212,
             "npks": 1, "nclfs": 0, "rid": RID},
            {"_id": ObjectId(), "tid": "CI.CLC.", "cha": "HH", "yr": 2019, "doy": 213,
             "npks": 2, "nclfs": 0, "rid": RID},
            {"_id": ObjectId(), "tid": "CI.WRC2.", "cha": "HH", "yr": 2019, "doy": 212,
             "npks": 1, "nclfs": 0, "rid": RID},
            {"_id": ObjectId(), "tid": "CI.WRC2.", "cha": "HN", "yr": 2019, "doy": 212,
             "npks": 1, "nclfs": 0, "rid": RID},
            {"_id": ObjectId(), "tid": "CI.ZERO.", "cha": "HH", "yr": 2019, "doy": 212,
             "npks": 0, "nclfs": 0, "rid": RID},
            {"_id": ObjectId(), "tid": "NC.KCPB.00", "cha": "HH", "yr": 2019, "doy": 212,
             "npks": 1, "nclfs": 0, "rid": RID},
        ],
        {"picks_record_idx": {}},
    )
    db["stations"] = FakeCollection([
        {"_id": ObjectId(), "id": "CI.CLC.", "network_code": "CI", "station_code": "CLC",
         "location_code": "", "channels": "HH,HN", "latitude": 35.8, "longitude": -117.6,
         "elevation": 775.0, "start_date": 2010.21, "end_date": 3000.001},
        {"_id": ObjectId(), "id": "NC.KCPB.00", "network_code": "NC", "station_code": "KCPB",
         "location_code": 0.0, "channels": "HH", "latitude": 39.7, "longitude": -123.6,
         "elevation": 900.0, "start_date": 2001.001, "end_date": 3000.001},
    ])
    db["sb_runs"] = FakeCollection([{
        "_id": RID, "model": "PhaseNet", "weight": "instance", "p_threshold": 0.2,
        "s_threshold": 0.2, "components_loaded": "ZNE12", "seisbench_version": "0.8",
        "weight_version": 1,
        "timestamp": datetime.datetime(2025, 4, 10, 12, 0, 0),
    }])
    return db


# ------------------------------------------------------------ transformations

def test_fallback_schema_is_the_writer_schema():
    """The exporter carries a copy of PICK_SCHEMA for hosts without s3fs.
    It must be the writer's schema, field for field."""
    src = pathlib.Path(ex.__file__).read_text()
    for field in PICK_SCHEMA:
        assert f'("{field.name}", pa.' in src
    assert ex.PICK_SCHEMA.equals(PICK_SCHEMA)


def test_pick_doc_to_row_keeps_every_field_and_types_it_for_parquet():
    doc = _pick("CI.CLC.", T0)
    row = ex.pick_doc_to_row(doc)
    assert row["tid"] == "CI.CLC." and row["cha"] == "HH" and row["pha"] == "P"
    assert row["peak"] == T0 and row["peak"].tzinfo is None
    assert row["rid"] == "65f0a1b2c3d4e5f601234567"
    assert row["amp_vel"] is None                       # did not exist in 2025
    table = pa.Table.from_pylist([row], schema=PICK_SCHEMA)
    back = table.to_pylist()[0]
    assert back["peak"] == T0 and back["start"] == doc["start"]
    assert math.isclose(back["conf"], 0.73, rel_tol=1e-6)


def test_tz_aware_and_naive_bson_dates_land_on_the_same_instant():
    aware = T0.replace(tzinfo=datetime.timezone.utc)
    pdt = aware.astimezone(datetime.timezone(datetime.timedelta(hours=-7)))
    assert ex.to_naive_utc(aware) == ex.to_naive_utc(pdt) == ex.to_naive_utc(T0) == T0


def test_partition_follows_the_peak_across_a_month_boundary():
    july = ex.pick_doc_to_row(_pick("CI.CLC.", T0))
    aug = ex.pick_doc_to_row(_pick("CI.CLC.", T0 + datetime.timedelta(seconds=5)))
    assert ex.partition_of(july) == ("CI", 2019, 7)
    assert ex.partition_of(aug) == ("CI", 2019, 8)


def test_run_record_matches_the_2026_shape():
    rec = ex.run_doc_to_record(_db()["sb_runs"].docs[0])
    assert rec["run_id"] == str(RID)
    assert rec["created"] == "2025-04-10T12:00:00+00:00"
    # The 2026 worker stringifies every value (worker.py write_run_data).
    assert rec["p_threshold"] == "0.2" and rec["weight_version"] == "1"
    assert rec["weight"] == "instance" and rec["source"] == "documentdb-2025"


def test_sort_and_dedupe_drops_only_exact_key_duplicates():
    rows = [ex.pick_doc_to_row(_pick("CI.CLC.", T0, conf=c)) for c in (0.5, 0.9)]
    rows.append(ex.pick_doc_to_row(_pick("CI.CLC.", T0, pha="S")))
    table, dropped = ex.sort_and_dedupe(pa.Table.from_pylist(rows, schema=PICK_SCHEMA))
    assert dropped == 1 and table.num_rows == 2


def test_file_slices_cover_every_row_once():
    for n, per in [(0, 10), (1, 10), (10, 10), (11, 10), (2_500_001, 1_000_000)]:
        slices = ex.file_slices(n, per)
        assert sum(length for _, length in slices) == n
        assert all(length <= per for _, length in slices)
        assert [o for o, _ in slices] == sorted(o for o, _ in slices)


# ------------------------------------------------------------------- export

def _export(tmp_path, db=None, **kw):
    db = db or _db()
    out = ex.Output(str(tmp_path / "cat"))
    exporter = ex.Exporter(db, out, "docdb2025", str(tmp_path / "staging"),
                           database_name="earthscope", **kw)
    by_net = ex.station_ids(db, ["stations", "picks_record"])
    return db, out, exporter, by_net


def test_export_writes_the_2026_layout_losslessly(tmp_path):
    db, out, exporter, by_net = _export(tmp_path)
    assert by_net == {"CI": ["CI.CLC.", "CI.WRC2.", "CI.ZERO."], "NC": ["NC.KCPB.00"]}
    exporter.export_runs()
    exporter.export_stations()
    summaries = {n: exporter.export_network(n, t) for n, t in by_net.items()}

    root = tmp_path / "cat"
    july = root / "picks/network=CI/year=2019/month=07/docdb2025-CI.parquet"
    aug = root / "picks/network=CI/year=2019/month=08/docdb2025-CI.parquet"
    assert july.exists() and aug.exists()
    assert pq.read_schema(july).equals(PICK_SCHEMA)

    # Every pick, once, through the Hive layout the 2026 readers use.
    import pandas as pd
    picks = pd.read_parquet(root / "picks")
    assert len(picks) == db["picks"].estimated_document_count() == 6
    assert sorted(picks["network"].astype(str).unique()) == ["CI", "NC"]
    wrc2 = picks[picks["tid"] == "CI.WRC2."]
    assert sorted(wrc2["cha"]) == ["HH", "HN"]       # 2025 kept every band
    assert wrc2[wrc2["cha"] == "HH"]["amp"].isna().all()

    # Sorted by the unique key inside a file.
    t = pq.read_table(july).to_pandas()
    assert list(t["tid"]) == sorted(t["tid"])

    assert summaries["CI"]["rows_written"] == 5 and summaries["CI"]["duplicates_dropped"] == 0
    assert summaries["CI"]["spot_check_failures"] == []
    assert summaries["CI"]["picks_record_npks_by_year"] == {"2019": 5}
    assert summaries["CI"]["rows_by_year"] == {"2019": 5}

    manifest = json.loads((root / "manifests/docdb2025-CI.json").read_text())
    assert set(manifest) >= {"job_id", "run_id", "n_picks", "station_days", "files", "records"}
    assert manifest["n_picks"] == 5 and manifest["station_days"] == 5
    zero = [r for r in manifest["records"] if r["tid"] == "CI.ZERO."]
    assert zero == [{"tid": "CI.ZERO.", "cha": "HH", "yr": 2019, "doy": 212,
                     "npks": 0, "nclfs": 0, "rid": str(RID)}]
    assert sum(f["rows"] for f in manifest["files"]) == 5

    run = json.loads((root / f"runs/{RID}.json").read_text())
    assert run["weight"] == "instance"

    stations = pd.read_parquet(root / "stations.parquet")
    kcpb = stations[stations["id"] == "NC.KCPB.00"].iloc[0]
    assert kcpb["location_code"] == "00"              # rebuilt from id, not 0.0
    clc = stations[stations["id"] == "CI.CLC."].iloc[0]
    assert clc["start_date"] == datetime.date(2010, 7, 29)   # day 210, not 21


def test_rerun_skips_done_networks_and_rewrites_nothing(tmp_path):
    db, out, exporter, by_net = _export(tmp_path)
    exporter.export_network("CI", by_net["CI"])
    f = tmp_path / "cat/picks/network=CI/year=2019/month=07/docdb2025-CI.parquet"
    mtime = f.stat().st_mtime_ns
    db["picks"].docs.clear()          # a re-run that queried would now find nothing
    again = exporter.export_network("CI", by_net["CI"])
    assert again["rows_written"] == 5 and f.stat().st_mtime_ns == mtime


def test_resume_from_staging_writes_only_unfinished_partitions(tmp_path):
    """Streaming done, July written, crash before August: the rerun must not
    query the database again and must not rewrite July."""
    db, out, exporter, by_net = _export(tmp_path, spot_checks=0)
    calls = {"n": 0}
    real = exporter._finalise_partition

    def crash_on_august(key, stager, job_id):
        if key[2] == 8:
            raise RuntimeError("simulated crash")
        calls["n"] += 1
        return real(key, stager, job_id)

    exporter._finalise_partition = crash_on_august
    with pytest.raises(RuntimeError):
        exporter.export_network("CI", by_net["CI"])
    july = tmp_path / "cat/picks/network=CI/year=2019/month=07/docdb2025-CI.parquet"
    mtime = july.stat().st_mtime_ns

    exporter._finalise_partition = real
    db["picks"].docs.clear()          # proves the rerun read from staging
    summary = exporter.export_network("CI", by_net["CI"])
    assert summary["rows_written"] == 5
    assert july.stat().st_mtime_ns == mtime


def test_a_short_cursor_stops_the_network(tmp_path):
    """A cursor that ends early must not be written and marked done."""
    db, out, exporter, by_net = _export(tmp_path)
    real_find = db["picks"].find
    db["picks"].find = lambda *a, **k: ex_short(real_find(*a, **k))
    with pytest.raises(RuntimeError, match="different count"):
        exporter.export_network("CI", by_net["CI"])
    assert not (tmp_path / "cat/_export/docdb2025/network=CI.done.json").exists()


def ex_short(cursor):
    return FakeCursor(cursor[:-1]) if len(cursor) > 1 else cursor


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    db = _db()
    by_net = ex.station_ids(db, ["stations", "picks_record"])
    report = ex.dry_run(db, by_net, sample=10)
    assert report["networks"]["CI"]["picks"] == 5
    assert report["picks_total_over_listed_stations"] == 6
    assert "pick_idx" in report["collections"]["picks"]["indexes"]
    assert report["collections"]["picks"]["field_types"]["rid"] == {"ObjectId": 6}
    assert not any(tmp_path.iterdir())
