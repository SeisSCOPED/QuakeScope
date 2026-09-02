"""A preemption must reach the handler that releases the claim.

Found 2026-09-02 on a dry-run arm. The SIGTERM landed inside an FDSN metadata
request, whose retry loop is a broad `except Exception ... sleep(5)`, so all
four worker loops logged

    FDSN request failed (1/8): Preempted. Sleeping 5 s.

and kept working after being told to stop. Docker SIGKILLed the container ~120 s
later: exit 137, four claims stranded for the full six-hour lease - the exact
failure the SIGTERM forwarding fix was written to end, reintroduced by a handler
in a different module.

`Preempted` is now a BaseException, like KeyboardInterrupt and SystemExit, so no
`except Exception` can absorb it - including the ones inside obspy, boto3 and
seisbench, which this package does not control.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.worker import Preempted  # noqa: E402


def test_preempted_is_a_baseexception_not_an_exception():
    assert issubclass(Preempted, BaseException)
    assert not issubclass(Preempted, Exception), (
        "as an Exception it is swallowed by every broad retry loop in the "
        "package - there are 19 of them"
    )


def test_a_broad_retry_loop_cannot_absorb_it():
    """The shape of the FDSN loop that swallowed it."""
    attempts = []

    def flaky():
        attempts.append(1)
        raise Preempted()

    with pytest.raises(Preempted):
        for _ in range(8):
            try:
                flaky()
            except Exception:          # the loop as written in s3_helper
                continue
    assert len(attempts) == 1, "it must escape on the first raise, not retry"


def test_the_worker_handler_still_catches_it():
    """Making it a BaseException must not stop the intended handler working."""
    released = []
    try:
        try:
            raise Preempted()
        except Preempted:
            released.append("claim")
            raise
    except Preempted:
        pass
    assert released == ["claim"]


def test_ordinary_failures_are_still_retryable():
    """The change must not make real errors escape their retry loops."""
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("transient")
        return "ok"

    for _ in range(8):
        try:
            result = flaky()
            break
        except Exception:
            continue
    assert result == "ok" and len(attempts) == 3
