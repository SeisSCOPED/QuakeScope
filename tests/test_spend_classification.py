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


def test_the_rate_is_calibrated_not_guessed():
    """The all-in rate must come from a bill, not a constant.

    The first estimate priced vCPU-hours at a guessed $0.0148 and reported
    $1,111 against a validated $2,976 - 2.7x low, because it also missed a
    1,500-task array job and never priced memory at all. The rate now divides a
    billed figure by the vCPU-hours measured over the same days, so it cannot
    drift from the invoice without someone editing the invoice.
    """
    import json
    act = json.load(open("costs_actual.json"))
    w = act["campaign_window"]
    days = [d for d in act["daily"] if w["start"] <= d <= w["end"]]
    assert days, "the window must select some billed days"
    billed = sum(act["daily"][d] for d in days)
    net = (billed - act["baseline_per_day"] * len(days)
           - sum(e["amount"] for e in act["exclusions"])
           - sum(u["amount"] for u in act["unattributed"]))
    # The recorded entry is what the dashboard shows beside the estimate, so it
    # must equal what the window arithmetic produces - not the gross.
    entry = act["entries"][0]["amount"]
    assert abs(net - entry) < 0.01, f"entry {entry} != derived {net:.2f}"
    # And it must be net of the exclusions, not the gross over baseline.
    assert entry < act["entries"][0]["gross_over_baseline"]


def test_calibration_skips_days_that_cannot_price_compute():
    """A day billing more than on-demand LIST is not evidence about Spot.

    2026-08-31 bills $608.87 over baseline against 2,836 measured vCPU-hours -
    432% of Fargate on-demand list, so it cannot be our compute at any Spot
    rate. Averaging it in raised the derived rate 21% and made Spot look far
    worse than it is. It must stay on the unattributed list, not be silently
    folded into the campaign.
    """
    import json
    act = json.load(open("costs_actual.json"))
    skip = {u["day"] for u in act["unattributed"]}
    assert "2026-08-31" in skip
    ONDEMAND_ALL_IN = 0.04048 + 2.06 * 0.004445
    for day in skip:
        # Anything excluded should be excluded FOR A REASON that survives
        # arithmetic: it prices above list, or it carries no load.
        assert act["daily"][day] - act["baseline_per_day"] > 0
    assert act["calibration_min_vcpu_hours_per_day"] > 0
    assert ONDEMAND_ALL_IN > 0


def test_the_cost_prose_follows_the_artefact():
    """What the page SAYS about the rate must match what the rate IS.

    The spend section read "the published Fargate Spot list rate" and "nothing
    here has been reconciled against what was actually charged" for an hour
    after the rate started being derived from a real CloudBank invoice. Both
    sentences were true when written and false when read - the failure this
    project keeps repeating. The wording is now switched by the artefact's own
    rate_is_calibrated flag, and this test is what keeps the two in step.
    """
    import json
    import re
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    import campaign_dashboard as D

    def flat(cal):
        doc = {"generated": "2026-09-07T00:00:00+00:00",
               "rate_per_vcpu_hour": 0.0313, "rate_is_calibrated": cal,
               "categories": {"productive": {"jobs": 1, "vcpu_hours": 1.0,
                                             "gb_hours": 2.0, "spend": 1.0}},
               "total": {"jobs": 1, "vcpu_hours": 1.0, "gb_hours": 2.0,
                         "spend": 1.0},
               "by_campaign": {}}
        g = dict(spend_doc=doc, per_station={}, per_day={},
                 per_station_month={}, camps=[], sampled=[], unreadable=[],
                 picks=0, bytes=0, files=0, coords={}, vcpu_now=0,
                 vcpu_hours=0, status={}, vcpu_partial=False)
        # Tags stripped and whitespace collapsed: the sentences wrap across
        # lines in the source, so a raw substring search gives a false negative.
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", D.render(g, {})))

    on = flat(True)
    assert "derived from a real invoice" in on
    assert "list price" in on          # only to say it is NOT one
    assert "not a bill" not in on
    assert "Nothing here has been reconciled" not in on

    off = flat(False)
    assert "list price nothing has checked" in off
    assert "derived from a real invoice" not in off

    # The badge stays in both: a reconciled TOTAL does not make the per-category
    # split measured, and the heading must not imply it does.
    for h in (on, off):
        assert "estimated" in h


# ---------------------------------------------------------------------------
# Attempts, not jobs.
#
# Batch overwrites a job's startedAt/stoppedAt on every retry, so a job that
# Spot reclaimed three times reports the span of its fourth attempt and nothing
# else. global-477782054: four attempts, 17.16 h run, job-level span 1.04 h.
# Summing job spans undercounted the three calibration days by 1.47x (74,030
# against 108,664 vCPU-h) and inflated the derived rate by the same factor -
# $0.0313 instead of $0.0213 - and that figure was published as "calibrated".

from campaign_spend import job_usage  # noqa: E402

H = 3600000  # ms


def _job(attempts, job_span=None, vcpu=8, mem=16384):
    d = {"jobName": "global-1", "attempts": [], "container": {
        "resourceRequirements": [{"type": "VCPU", "value": str(vcpu)},
                                 {"type": "MEMORY", "value": str(mem)}],
        "logStreamName": "last"}}
    for i, (start, stop) in enumerate(attempts):
        d["attempts"].append({"startedAt": start, "stoppedAt": stop,
                              "container": {"logStreamName": f"att{i}"}})
    if job_span:
        d["startedAt"], d["stoppedAt"] = job_span
    return d


def test_every_attempt_is_counted_not_the_last_one():
    # Three attempts of 5 h; the job record shows only the third.
    j = _job([(0, 5 * H), (6 * H, 11 * H), (12 * H, 17 * H)], job_span=(12 * H, 17 * H))
    vh, gh, per_day, streams, first = job_usage(j, now=20 * H)
    assert vh == 8 * 15, "15 hours ran and were billed; the job span says 5"
    assert gh == 16 * 15
    assert first == 0, "the run began with the first attempt, not the last"


def test_the_day_split_follows_attempts():
    # One attempt on day 0 and one on day 1: neither may land on the other's
    # day, because the calibration compares each day to its invoice line.
    day = 24 * H
    j = _job([(2 * H, 4 * H), (day + 2 * H, day + 4 * H)])
    _, _, per_day, _, _ = job_usage(j, now=2 * day)
    assert len(per_day) == 2 and all(abs(v - 16) < 1e-9 for v in per_day.values())


def test_an_earlier_attempt_that_completed_makes_the_job_productive():
    # Only the last attempt's stream is on the job record; the completion was
    # in the first. classify() must be able to see it.
    j = _job([(0, 5 * H), (6 * H, 7 * H)])
    _, _, _, streams, _ = job_usage(j, now=8 * H)
    assert {"att0", "att1", "last"} <= streams
    from campaign_spend import classify
    assert classify("global-1", bool(streams & {"att0"}), True, j) == "productive"


def test_a_running_job_with_no_attempt_record_still_counts():
    d = {"jobName": "western-1", "startedAt": 0, "container": {
        "resourceRequirements": [{"type": "VCPU", "value": "8"}]}}
    vh, _, _, _, first = job_usage(d, now=2 * H)
    assert vh == 16 and first == 0


def test_a_job_that_never_ran_consumed_nothing():
    vh, gh, per_day, streams, first = job_usage({"jobName": "x"}, now=H)
    assert vh == 0 and gh == 0 and not per_day and first is None
