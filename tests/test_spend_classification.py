"""The spend split must not impute a category it cannot determine.

The first version of campaign_spend.py read each job's log to decide whether
it completed a shard, and fell back to the Batch status when the log was gone.
The log group keeps 5 days. That fallback counted every Spot-reclaimed job as
"spinning" and every old job that completed nothing as "productive" - wrong in
both directions, and silently, on the one figure the whole point of the script
is to get right.

These tests pin the three distinctions that fallback destroyed:
  - a completion is proof at any age, so it outranks the retention window
  - a job outside the window is UNKNOWABLE, not spinning
  - Spot taking the task back is not the same as our code failing to finish
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from campaign_spend import classify, is_reclaim  # noqa: E402

PLAIN = {"statusReason": "Essential container in task exited"}
SPOT = {"statusReason": "Host EC2 (instance i-0abc) terminated."}


def test_a_completion_outranks_an_expired_log():
    # The job ran before the retention window, but a "Completed " line for it
    # survives inside the window. Proof does not expire.
    assert classify("western-3", True, False, PLAIN) == "productive"


def test_unknowable_is_named_not_guessed():
    # No completion found AND the log cannot answer for this job. The old
    # fallback called this "spinning" and charged it to our bugs.
    assert classify("western-3", False, False, PLAIN) == "log expired"


def test_spot_reclaim_is_not_our_waste():
    # Reclaim is the price of the 70% discount, and the part-done shard is
    # checkpointed under progress/. Counting it as waste overstates the bill
    # we could have avoided.
    assert classify("western-3", False, True, SPOT) == "spot reclaim"


def test_a_worker_that_finished_nothing_on_its_own_is_spinning():
    assert classify("western-3", False, True, PLAIN) == "spinning"


def test_dry_runs_and_surveys_are_never_science():
    # By name, before any log or age question - these are deliberate and their
    # cost belongs in its own line however they ended.
    assert classify("dryrun4-1", False, False, PLAIN) == "dry run"
    assert classify("survey-obs", False, False, PLAIN) == "survey"


def test_reclaim_is_recognised_from_either_field():
    assert is_reclaim({"statusReason": "Your Spot Task was interrupted"})
    assert is_reclaim({"container": {"reason": "Host EC2 terminated"}})
    assert not is_reclaim(PLAIN)
    assert not is_reclaim({})
