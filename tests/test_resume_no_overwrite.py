"""A resumed shard must add to what the first attempt wrote, never replace it.

Found 2026-09-16 on the western campaign (docs/rerun_2026/28_resumed_shard_overwrite.md):
a worker resuming a checkpointed shard built a fresh writer whose file sequence
started at zero, so its first flush landed on ``<job>.parquet`` and replaced the
first attempt's first checkpoint. Its manifest then described only the resuming
attempt, and its progress object dropped the first attempt's records, so a
third attempt would have redone them and written the picks twice.

Three checks: the resumed writer continues the sequence, the manifest covers
both attempts with the first attempt's pick counts read back from its files,
and progress stays cumulative.
"""

import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obspy
import pyarrow.parquet as pq
import seisbench.util as sbu

from sb_catalog.src.parquet_writer import ParquetPickWriter
from sb_catalog.src.s3_state import S3CampaignState
from tests.test_checkpoint_resume import FakeS3


def _pick(t, sta="CI.CLC."):
    return sbu.Pick(trace_id=sta, start_time=t, end_time=t + 1,
                    peak_time=t + 0.5, peak_value=0.9, phase="P")


def _files(root):
    part = os.path.join(root, "picks", "network=CI", "year=2019", "month=07")
    return set(os.listdir(part)) if os.path.isdir(part) else set()


def test_resumed_writer_continues_sequence_and_merges_manifest():
    fake = FakeS3()
    state = S3CampaignState("s3://bkt/camp", client=fake)
    shard_id = "2019187-2019207-abc123def456"
    root = tempfile.mkdtemp()
    day = lambda d: datetime.date(2019, 7, d)
    at = lambda d, s=10: obspy.UTCDateTime(f"2019-07-{d:02d}T00:00:{s:02d}")   # a pick on that day

    # --- attempt 1: two checkpoints, then preempted --------------------------
    w1 = ParquetPickWriter(root=root, run_id="r1", job_id=shard_id)
    w1.add([_pick(at(6)), _pick(at(6, 15))], [1.0, 1.0], [2.0, 2.0], [], "CI.CLC.", day(6), "HH")
    w1.add([_pick(at(7))], [1.0], [2.0], [], "CI.CLC.", day(7), "HH")
    records = w1.checkpoint()
    state.write_progress(shard_id, records)
    w1.add([_pick(at(8))], [1.0], [2.0], [], "CI.CLC.", day(8), "HH")
    records = w1.checkpoint()
    state.write_progress(shard_id, records, prior=state.read_progress(shard_id))
    assert _files(root) == {f"{shard_id}.parquet", f"{shard_id}-001.parquet"}
    first_file = os.path.join(root, "picks", "network=CI", "year=2019", "month=07", f"{shard_id}.parquet")
    first_bytes = open(first_file, "rb").read()
    print("PASS  attempt 1 left two checkpoint files")

    # --- attempt 2 resumes on another worker --------------------------------
    done = state.read_progress(shard_id)
    assert done == {("CI.CLC.", 2019, 187, "HH"), ("CI.CLC.", 2019, 188, "HH"),
                    ("CI.CLC.", 2019, 189, "HH")}
    w2 = ParquetPickWriter(root=root, run_id="r2", job_id=shard_id, prior_done=done)
    w2.add([_pick(at(9))], [1.0], [2.0], [], "CI.CLC.", day(9), "HH")
    records2 = w2.checkpoint()
    state.write_progress(shard_id, records2, prior=done)
    summary = w2.close()

    # The first attempt's files are untouched and the new one continues the series.
    assert _files(root) == {f"{shard_id}.parquet", f"{shard_id}-001.parquet",
                            f"{shard_id}-002.parquet"}, _files(root)
    assert open(first_file, "rb").read() == first_bytes, "resume overwrote the first checkpoint"
    print("PASS  resumed attempt wrote -002 and left <job>.parquet and -001 alone")

    # The manifest describes the whole job: four station-days, three files, four picks
    # from attempt 1 plus one from attempt 2, with attempt 1's counts read from its files.
    assert summary["station_days"] == 4
    assert [f["path"].rsplit("/", 1)[1] for f in summary["files"]] == [
        f"{shard_id}.parquet", f"{shard_id}-001.parquet", f"{shard_id}-002.parquet"]
    assert summary["n_picks"] == 5
    by_day = {(r["tid"], r["doy"]): r for r in summary["records"]}
    assert by_day[("CI.CLC.", 187)]["npks"] == 2 and by_day[("CI.CLC.", 187)]["rid"] == "r1"
    assert by_day[("CI.CLC.", 188)]["npks"] == 1
    assert by_day[("CI.CLC.", 189)]["npks"] == 1
    assert by_day[("CI.CLC.", 190)]["npks"] == 1 and by_day[("CI.CLC.", 190)]["rid"] == "r2"
    assert summary["resumed"] == {"prior_station_days": 3, "prior_files": 2, "prior_picks": 4}
    print("PASS  manifest covers both attempts with the first attempt's counts read back")

    # Every row the job produced is in exactly one file the manifest lists.
    rows = sum(pq.read_metadata(f["path"]).num_rows for f in summary["files"])
    assert rows == 5
    print("PASS  files listed in the manifest hold every pick once")

    # --- progress stayed cumulative, so a third attempt skips everything ------
    prog = json.loads(fake.obj[state._key("progress", f"{shard_id}.json")])
    assert sorted(tuple(e) for e in prog["done"]) == sorted(done | {("CI.CLC.", 2019, 190, "HH")})
    print("PASS  progress carries attempt 1's records after attempt 2's checkpoint")

    # --- a fresh job in an empty partition still starts at zero ---------------
    w3 = ParquetPickWriter(root=tempfile.mkdtemp(), run_id="r3", job_id="fresh-job")
    w3.add([_pick(at(6))], [1.0], [2.0], [], "CI.CLC.", day(6), "HH")
    s3 = w3.close()
    assert s3["files"][0]["path"].endswith("fresh-job.parquet") and "resumed" not in s3
    print("PASS  a fresh job is unchanged: <job>.parquet, no resumed block")

    print("\nall resume/no-overwrite checks passed")


if __name__ == "__main__":
    test_resumed_writer_continues_sequence_and_merges_manifest()
