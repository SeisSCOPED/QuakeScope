"""Shared fixtures.

The EarthScope credential state is process-wide by design - a worker runs many
shards in one process and must not re-ask EarthScope a question it has already
had answered - so it has to be cleared between tests, or a verdict recorded by
one test leaks into the next.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _clean_earthscope_state():
    from sb_catalog.src.s3_helper import reset_earthscope_state
    reset_earthscope_state()
    yield
    reset_earthscope_state()
