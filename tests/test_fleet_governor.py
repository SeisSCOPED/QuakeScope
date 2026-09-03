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
