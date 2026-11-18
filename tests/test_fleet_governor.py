"""One queue, several campaigns: each governor must count only its own workers.

On 2026-09-03 obs was set to a target of 59 while western held 101 workers in
the same queue. alive_count() counted every job in the queue, so obs read
"alive 101 >= 59", reported deficit 0, submitted nothing, and never started -
while looking healthy. The soak test could not catch it because only one
campaign was running at the time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import spot_governor
from spot_governor import alive_count


class _Batch:
    """Two campaigns' workers in one queue, as Batch would return them."""

    def __init__(self):
        self.jobs = ([{"jobName": f"western-{i}"} for i in range(101)] +
                     [{"jobName": f"obs-{i}"} for i in range(3)])

    def list_jobs(self, jobQueue, jobStatus, maxResults=100, nextToken=None):
        # Everything lands in RUNNING; the other statuses are empty.
        if jobStatus != "RUNNING":
            return {"jobSummaryList": []}
        return {"jobSummaryList": self.jobs}


def test_counts_only_this_campaign():
    b = _Batch()
    assert sum(alive_count(b, "q", "western").values()) == 101
    assert sum(alive_count(b, "q", "obs").values()) == 3


def test_a_busy_neighbour_does_not_satisfy_this_target():
    """The exact failure: obs must still see a deficit."""
    b = _Batch()
    obs_alive = sum(alive_count(b, "q", "obs").values())
    target = 59
    assert max(0, target - obs_alive) == 56, (
        "obs has 3 workers against a target of 59; western's 101 are not obs's"
    )


def test_prefix_is_required():
    # Making it optional invites exactly the bug back. It has no default.
    import inspect
    p = inspect.signature(alive_count).parameters["prefix"]
    assert p.default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# When is a campaign finished?
#
# On 2026-09-08/09 western and western-2026 were complete except for embargoed
# LH shards, which are blocked - not complete, not claimable. The governor's
# only exit was `complete >= total`, so both read as unfinished, and every
# scheduled top-up submitted the full target into a queue with nothing to
# claim: 155 workers in 30 hours for zero shards.

from spot_governor import finished


def test_all_complete_is_finished():
    assert finished({"total": 4, "complete": 4, "blocked": 0, "remaining": 0})


def test_blocked_tail_counts_as_finished():
    """The 2026-09-08 shape: everything either done or embargoed."""
    assert finished({"total": 72505, "complete": 72464, "blocked": 41,
                     "in_flight": 0, "remaining": 0})


def test_leftover_shards_are_not_finished():
    """The 2026-09-09 shape: 16 shards neither done nor blocked. Still work."""
    assert not finished({"total": 72505, "complete": 72448, "blocked": 41,
                         "in_flight": 0, "remaining": 16})


def test_unreadable_progress_is_not_finished():
    # A failed S3 read reports total 0; that must not stop a fleet.
    assert not finished({})
    assert not finished({"total": 0, "complete": 0, "blocked": 0})


def test_unblocking_reopens_the_campaign():
    # Blocked is reversible, so finished must be too: once the embargo lifts
    # and the shards are unblocked, the governor has to resume.
    before = {"total": 10, "complete": 8, "blocked": 2}
    after = {"total": 10, "complete": 8, "blocked": 0}
    assert finished(before) and not finished(after)


class _SubmitBatch:
    def __init__(self):
        self.cmds = []

    def submit_job(self, **kw):
        self.cmds.append(kw["containerOverrides"]["command"])
        return {"jobId": "j"}


def _args(argv):
    import argparse
    ap = argparse.ArgumentParser()
    for flag, kw in [("--campaign", {}), ("--weight", {"default": "jma_wc"}),
                     ("--model", {"default": ""}),
                     ("--p-threshold", {"type": float, "default": None}),
                     ("--s-threshold", {"type": float, "default": None})]:
        ap.add_argument(flag, **kw)
    a = ap.parse_args(argv)
    a.procs, a.threads, a.checkpoint_every, a.lease_hours = 4, 2, 40, 1.0
    a.max_hours, a.limit_mb, a.parquet_uri = 0, 0, ""
    a.name_prefix, a.queue, a.job_definition = "t", "q", "jd"
    return a


def test_model_and_thresholds_are_omitted_by_default():
    b = _SubmitBatch()
    spot_governor.submit(b, _args(["--campaign", "s3://b/c"]), 1)
    assert not {"--model", "--p_threshold", "--s_threshold"} & set(b.cmds[0])


def test_model_and_thresholds_are_forwarded_when_set():
    b = _SubmitBatch()
    spot_governor.submit(b, _args(["--campaign", "s3://b/c", "--weight", "pnw",
                                   "--model", "EQTransformer",
                                   "--p-threshold", "0.1", "--s-threshold", "0.1"]), 1)
    c = b.cmds[0]
    assert c[c.index("--model") + 1] == "EQTransformer"
    assert c[c.index("--p_threshold") + 1] == "0.1"
    assert c[c.index("--s_threshold") + 1] == "0.1"


def test_preprocess_and_classify_args_are_forwarded_when_set():
    b = _SubmitBatch()
    a = _args(["--campaign", "s3://b/c"])
    a.preprocess, a.classify_args = "ni2022", '{"overlap": 5400}'
    spot_governor.submit(b, a, 1)
    c = b.cmds[0]
    assert c[c.index("--preprocess") + 1] == "ni2022"
    assert c[c.index("--classify-args") + 1] == '{"overlap": 5400}'
