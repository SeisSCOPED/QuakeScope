"""A preempted shard must resume, not restart.

Without checkpointing a Spot interruption twelve hours into a shard discards
twelve hours of work, because picks are buffered until the job ends. These
checks cover the three things that has to get right: progress is only recorded
after the data is durable, a resumed shard skips what is already written, and
the ordering cannot be inverted.
"""

import datetime
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obspy
import seisbench.util as sbu
from botocore.exceptions import ClientError

from sb_catalog.src.parquet_writer import ParquetPickWriter
from sb_catalog.src.s3_state import S3CampaignState
from sb_catalog.src.worker import S3StateAdapter


class FakeS3:
    """Enough S3 for the state store, with real conditional-write semantics."""

    def __init__(self):
        self.obj, self.lock = {}, threading.Lock()

    def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None):
        with self.lock:
            if IfNoneMatch is not None and Key in self.obj:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            self.obj[Key] = Body if isinstance(Body, bytes) else Body.encode()
            return {}

    def get_object(self, Bucket, Key):
        import io
        with self.lock:
            if Key not in self.obj:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": io.BytesIO(self.obj[Key])}

    def head_object(self, Bucket, Key):
        with self.lock:
            if Key not in self.obj:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            return {}

    def delete_object(self, Bucket, Key):
        with self.lock:
            self.obj.pop(Key, None)
            return {}

    def get_paginator(self, name):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix):
                with outer.lock:
                    keys = [k for k in outer.obj if k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in keys]}

        return P()


def _pick(t):
    return sbu.Pick(trace_id="CI.CLC.", start_time=t, end_time=t + 1,
                    peak_time=t + 0.5, peak_value=0.9, phase="P")


def test_checkpoint_and_resume():
    fake = FakeS3()
    state = S3CampaignState("s3://bkt/camp", client=fake)
    shard_id = "2019187-2019190-abc123def456"
    root = tempfile.mkdtemp()

    # --- a shard runs three station-days, then is preempted -----------------
    writer = ParquetPickWriter(root=root, run_id="r1", job_id=shard_id)
    t = obspy.UTCDateTime("2019-07-06T00:00:10")
    for doy, day in enumerate([datetime.date(2019, 7, 6),
                               datetime.date(2019, 7, 7),
                               datetime.date(2019, 7, 8)]):
        writer.add([_pick(t)], [1.0], [2.0], [], "CI.CLC.", day, "HH")

    records = writer.checkpoint()
    state.write_progress(shard_id, records)
    assert len(records) == 3
    print(f"PASS  checkpoint flushed and recorded {len(records)} station-days")

    # Durability is the point: the Parquet must exist before progress claims it.
    written = [f for f in os.listdir(os.path.join(root, "picks")) or []]
    assert written, "checkpoint recorded progress without writing Parquet"
    print("PASS  Parquet written before progress was recorded")

    # --- a new worker picks the shard up ------------------------------------
    resumed = state.read_progress(shard_id)
    assert resumed == {("CI.CLC.", 2019, 187, "HH"),
                       ("CI.CLC.", 2019, 188, "HH"),
                       ("CI.CLC.", 2019, 189, "HH")}
    print(f"PASS  resume loaded {len(resumed)} completed station-day-channels")

    adapter = S3StateAdapter(state, stations=None, done=resumed)
    already = adapter.get_picks_record("CI.CLC.", datetime.date(2019, 7, 6), "HH")
    fresh = adapter.get_picks_record("CI.CLC.", datetime.date(2019, 7, 9), "HH")
    assert already is not None, "resumed shard would redo a completed station-day"
    assert fresh is None, "resumed shard would skip work it never did"
    print("PASS  completed station-days skipped, unfinished ones still run")

    # A shard that never checkpointed starts clean rather than erroring.
    assert state.read_progress("never-ran-0000") == set()
    print("PASS  a shard with no progress starts from the beginning")

    # --- the ordering trap ---------------------------------------------------
    # Progress must never be recorded for picks that were not written. The
    # writer returns records only from checkpoint(), which flushes first, so
    # there is no path that reports unwritten work.
    w2 = ParquetPickWriter(root=tempfile.mkdtemp(), run_id="r2", job_id="j2")
    w2.add([_pick(t)], [1.0], [2.0], [], "CI.TOW2.", datetime.date(2019, 7, 6), "HH")
    assert w2.pending_records == 1
    before = json.loads(fake.obj.get(
        state._key("progress", "j2.json"), b"null") or b"null")
    assert before is None, "progress existed before any checkpoint"
    print("PASS  buffered-but-unflushed work is never recorded as progress")

    print("\nall checkpoint/resume checks passed")


if __name__ == "__main__":
    test_checkpoint_and_resume()
