"""A run record belongs with the picks it describes, not with the queue.

Every pick carries `rid`, and a reader resolves it under the prefix the pick is
in: `<catalogue>/runs/<rid>.json`. A worker takes its queue from `--campaign`
and its output root from `--parquet_uri`, and since 2026-09-17 those differ for
every repair and fill queue. The record used to be written to the queue, so
published picks carried an rid that resolved to nothing - 112,549 records for
`western-fill` alone, found on 2026-09-29, nine days after its picks went into
the western catalogue.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.s3_state import S3CampaignState
from sb_catalog.src.worker import S3StateAdapter
from tests.test_checkpoint_resume import FakeS3


def _state(fake, uri):
    return S3CampaignState(uri, client=fake)


def test_run_record_follows_the_picks_not_the_queue():
    fake = FakeS3()
    queue = _state(fake, "s3://bkt/_queues/western-fill2")
    catalogue = _state(fake, "s3://bkt/western")

    db = S3StateAdapter(queue, stations=None, output=catalogue)
    rid = db.write_run_data(model="PhaseNet", weight="original", p_threshold=0.2)

    assert f"western/runs/{rid}.json" in fake.obj, sorted(fake.obj)[:3]
    assert f"_queues/western-fill2/runs/{rid}.json" not in fake.obj
    print("PASS  the record lands in the catalogue the picks go to")

    # An ordinary campaign, where queue and output are the same prefix, is
    # unchanged: no `output` given, so the record goes to the campaign state.
    fake2 = FakeS3()
    only = _state(fake2, "s3://bkt/global")
    rid2 = S3StateAdapter(only, stations=None).write_run_data(model="PhaseNet", weight="jma_wc")
    assert f"global/runs/{rid2}.json" in fake2.obj
    print("PASS  a campaign that writes into its own prefix is unchanged")


if __name__ == "__main__":
    test_run_record_follows_the_picks_not_the_queue()
    print("\nall run-record checks passed")
