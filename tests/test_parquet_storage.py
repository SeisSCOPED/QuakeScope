"""Local output stays independent of s3fs; S3 output keeps its storage contract."""

import datetime
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import uuid

import fsspec
import pyarrow.parquet as pq

from sb_catalog.src.parquet_writer import PICK_SCHEMA, ParquetPickWriter


def test_local_writer_works_without_importing_s3fs(tmp_path):
    # A fresh interpreter catches eager imports even if another test loaded s3fs.
    script = textwrap.dedent("""
        import json
        from pathlib import Path
        import sys

        sys.modules["s3fs"] = None
        from sb_catalog.src.parquet_writer import ParquetPickWriter

        root = Path(sys.argv[1])
        writer = ParquetPickWriter(str(root), "run", job_id="job")
        summary = writer.close()
        assert json.loads((root / "manifests/job.json").read_text()) == summary
        assert sys.modules["s3fs"] is None
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_s3_writer_preserves_options_and_writes_parquet(monkeypatch):
    import s3fs

    # Exercise S3 backend selection and real Parquet encoding without AWS calls.
    fs = fsspec.filesystem("memory")
    calls = []

    def s3_filesystem(**options):
        calls.append(options)
        return fs

    monkeypatch.setattr(s3fs, "S3FileSystem", s3_filesystem)
    options = {"anon": True, "client_kwargs": {"region_name": "us-east-2"}}
    root = f"s3://writer-test/{uuid.uuid4().hex}"
    writer = ParquetPickWriter(root, "run", job_id="job", storage_options=options)
    day = datetime.datetime(2019, 7, 6)
    at = SimpleNamespace(datetime=day)
    pick = SimpleNamespace(
        start_time=at, peak_time=at, end_time=at, peak_value=0.75, phase="P")
    writer.add([pick], [1.0], [2.0], [], "CI.CLC.", day, "HH")
    summary = writer.close()

    assert calls == [options]
    assert summary["n_picks"] == 1
    assert len(summary["files"]) == 1
    path = summary["files"][0]["path"]
    assert path.startswith(root + "/picks/")
    with fs.open(path, "rb") as stream:
        table = pq.read_table(stream)
    assert table.schema == PICK_SCHEMA
    assert table.to_pylist() == [{
        "tid": "CI.CLC.", "cha": "HH", "pha": "P",
        "start": day, "peak": day, "end": day,
        "conf": 0.75, "amp": 1.0, "amp_vel": 2.0, "rid": "run",
    }]
    assert json.loads(fs.cat(f"{root}/manifests/job.json")) == summary
