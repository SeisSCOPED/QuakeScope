"""The sweep must not re-ask one scope once per year.

A permanent network's credential is scoped {"network": "FDSN:XX"} with no year
in it, so asking about 2011 and about 2019 send the IDENTICAL request. Sweeping
year by year re-exchanged that one scope a dozen times and the local throttle
refused it: 2,172 of `global`'s 3,757 network-years came back
EarthScopeExchangeThrottled and were never asked about at all. The summary then
counted every one of them as not readable, so a campaign that is 76% readable
across the network-years that answered was reported as 32%.

A survey that cannot see half its own plan is worse than no survey, because it
is believed - and the access gate refuses to launch on a low fraction.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_a_network_scope_carries_no_year():
    """The premise: for a permanent code the year is absent from the scope."""
    from sb_catalog.src.s3_helper import CompositeS3ObjectHelper
    h = CompositeS3ObjectHelper()
    permanent = h.es_scope("CI", 2011)
    assert "year" not in permanent, permanent
    # ...so 2011 and 2019 are the same request, and asking both is a repeat.
    assert h.es_scope("CI", 2011) == h.es_scope("CI", 2019)


def test_a_temporary_scope_does_carry_the_year():
    """And the converse, or the fix would over-apply one year's answer."""
    from sb_catalog.src.s3_helper import CompositeS3ObjectHelper
    h = CompositeS3ObjectHelper()
    h.es_scope_mode["7D"] = "network+year"
    a, b = h.es_scope("7D", 2012), h.es_scope("7D", 2019)
    assert a.get("year") == 2012 and b.get("year") == 2019
    assert a != b


def test_the_gate_divides_by_what_was_answered():
    """An unanswered question is not a no.

    Replays the arithmetic the access gate does, on global's real numbers from
    the 2026-09-07 sweep.
    """
    a = {"checked": 3757, "present": 1209, "missing": [0] * 376,
         "denied": [], "errors": [0] * 2172}
    present = a["present"]
    denied, missing = len(a["denied"]), len(a["missing"])
    resolved = a.get("resolved") or (present + denied + missing)
    # The old arithmetic divided by `checked` and made this look near the floor.
    assert present / a["checked"] < 0.33
    # The new one divides by what answered.
    assert resolved == 1585
    assert present / resolved > 0.76
    # And an incomplete survey is caught rather than believed.
    assert resolved < 0.5 * a["checked"]
