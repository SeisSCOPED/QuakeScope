"""Restricted EarthScope credentials must be scoped, and must be cached.

Two independent failures are pinned here, because each one on its own is
invisible until a campaign is running:

1. An UNSCOPED credential for `s3-miniseed-v2` carries `s3:ListBucket` but not
   `s3:GetObject`. Every listing succeeds and every read returns AccessDenied.
   Nothing in the logs distinguishes that from a missing entitlement, and for
   two weeks we read it as one - see docs/rerun_2026/26.

2. Fetching a credential per object would turn every GET into an OAuth round
   trip. The campaign is ~113M station-days behind a few dozen network-years,
   so the exchange has to happen once per scope and be reused.
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.s3_helper import CompositeS3ObjectHelper


class _Cred:
    """Stands in for AwsTemporaryCredentials, including the SecretStr wrapper
    that SDK >= 1.4.1 puts around the secret and the session token."""

    class _Secret:
        def __init__(self, v):
            self._v = v

        def get_secret_value(self):
            return self._v

        def __str__(self):
            return "**********"

    def __init__(self, ttl_minutes=60):
        self.aws_access_key_id = "AKIAEXAMPLE"
        self.aws_secret_access_key = self._Secret("secret-value")
        self.aws_session_token = self._Secret("token-value")
        self.expiration = datetime.datetime.now(
            tz=datetime.timezone.utc
        ) + datetime.timedelta(minutes=ttl_minutes)


def _helper(monkeypatch_calls):
    """A helper whose credential exchange is counted, never performed."""
    h = CompositeS3ObjectHelper()

    def fake(net, year=None):
        monkeypatch_calls.append((net, year))
        return _Cred()

    h.get_es_credential = fake
    return h


def test_scope_carries_the_network():
    h = CompositeS3ObjectHelper()
    assert h.es_scope("AV", 2019)["network"] == "FDSN:AV"


def test_permanent_networks_are_not_year_scoped_by_default():
    # A permanent code means one deployment for all time, so a year would
    # fragment the cache into one credential per year for no benefit.
    h = CompositeS3ObjectHelper()
    for net in ("AV", "CC", "NP", "BK", "CI"):
        assert h.es_scope(net, 2019) == {"network": f"FDSN:{net}"}


def test_temporary_networks_are_year_scoped_by_default():
    # FDSN reassigns codes beginning with a digit or X/Y/Z, so the same code is
    # a different experiment in a different year.
    h = CompositeS3ObjectHelper()
    for net in ("XD", "ZI", "ZG", "1D", "7D", "YW"):
        assert h.es_scope(net, 2018) == {
            "network": f"FDSN:{net}", "year": 2018}


def test_year_omitted_when_unknown():
    assert "year" not in CompositeS3ObjectHelper().es_scope("XD")


def test_one_exchange_per_scope_not_per_object():
    calls = []
    h = _helper(calls)

    # A thousand reads on one network-year: the shard's whole working set.
    for _ in range(1000):
        h.get_filesystem("ZI", 2019)
    assert len(calls) == 1, f"one credential per shard, got {len(calls)}"

    # The same filesystem object, not an equivalent one rebuilt each time.
    assert h.get_filesystem("ZI", 2019) is h.get_filesystem("ZI", 2019)


def test_scopes_do_not_collide():
    calls = []
    h = _helper(calls)

    h.get_filesystem("ZI", 2019)
    h.get_filesystem("ZI", 2020)      # temporary: a distinct experiment
    h.get_filesystem("XD", 2019)      # distinct network
    assert calls == [("ZI", 2019), ("ZI", 2020), ("XD", 2019)]

    # Permanent networks share one credential across years, because their
    # scope carries no year to differ on.
    calls.clear()
    h.get_filesystem("NP", 2011)
    h.get_filesystem("NP", 2024)
    assert len(calls) == 1


def test_open_data_never_asks_for_a_credential():
    calls = []
    h = _helper(calls)
    for net in ("UW", "AK", "TA", "UU", "II", "IU", "N4", "PB"):
        h.get_filesystem(net, 2019)
    assert calls == []


def test_secretstr_is_unwrapped_for_s3fs():
    # Handing s3fs a SecretStr signs the request with the literal string
    # "**********". That fails as a signature mismatch, not a type error, so it
    # surfaces as a puzzling 403 rather than a traceback.
    calls = []
    h = _helper(calls)
    fs = h.get_filesystem("ZI", 2019)
    assert fs.storage_options.get("secret") == "secret-value"
    assert fs.storage_options.get("token") == "token-value"


def test_expired_credential_is_refetched():
    h = CompositeS3ObjectHelper()
    calls = []

    def fake(net, year=None):
        calls.append((net, year))
        # Inside the 5-minute TTL threshold, so it is due for renewal.
        return _Cred(ttl_minutes=1)

    h.get_es_credential = fake
    h.get_filesystem("ZI", 2019)
    h.get_filesystem("ZI", 2019)
    assert len(calls) == 2


def test_update_forces_a_new_credential():
    calls = []
    h = _helper(calls)
    h.get_filesystem("ZI", 2019)
    h.update_es_filesystem("ZI", 2019)
    assert len(calls) == 2

    # ... but not for a network that never had one.
    h.update_es_filesystem("UW", 2019)
    assert len(calls) == 2


# --- the scoping is a guess, so it has to be correctable -------------------
#
# `default_scope_mode` encodes what EarthScope's docs say: temporary networks
# want a year, permanent ones do not. That has been verified for exactly one
# network (AV, permanent). If it is wrong for any other, the campaign must not
# stall on it - a denial has to teach the worker the right scoping instead.


def test_escalation_flips_a_temporary_network_to_network_only():
    h = _helper([])
    assert "year" in h.es_scope("XD", 2018)
    assert h.escalate_scope_mode("XD") is True
    assert h.es_scope("XD", 2018) == {"network": "FDSN:XD"}


def test_escalation_flips_a_permanent_network_to_year_scoped():
    # The guess can be wrong in the other direction too.
    h = _helper([])
    assert "year" not in h.es_scope("NP", 2018)
    assert h.escalate_scope_mode("NP") is True
    assert h.es_scope("NP", 2018) == {"network": "FDSN:NP", "year": 2018}


def test_escalation_gives_up_once_both_are_tried():
    h = _helper([])
    assert h.escalate_scope_mode("XD") is True
    assert h.escalate_scope_mode("XD") is False     # nothing left to try
    # And it stays on the second mode rather than oscillating.
    assert h.es_scope("XD", 2018) == {"network": "FDSN:XD"}


def test_escalation_is_learned_once_per_network():
    calls = []
    h = _helper(calls)
    h.get_filesystem("XD", 2018)
    h.update_es_filesystem("XD", 2018, escalate=True)
    calls.clear()
    # Every later read on that network uses the corrected scoping, with no
    # further exchange - the flip is remembered, not rediscovered per object.
    for _ in range(500):
        h.get_filesystem("XD", 2018)
    assert calls == []
    assert h.es_scope("XD", 2018) == {"network": "FDSN:XD"}


def test_escalation_discards_every_year_of_that_network():
    # Credentials for 2018 and 2019 were both built under the old mode, so
    # both are stale once the mode changes.
    calls = []
    h = _helper(calls)
    h.get_filesystem("XD", 2018)
    h.get_filesystem("XD", 2019)
    assert len(calls) == 2
    h.escalate_scope_mode("XD")
    assert h.credentials == {} and h.es_fs == {}


def test_escalation_leaves_other_networks_alone():
    calls = []
    h = _helper(calls)
    h.get_filesystem("XD", 2018)
    h.get_filesystem("ZI", 2018)
    h.escalate_scope_mode("XD")
    # ZI keeps its credential and its default scoping.
    assert len(h.credentials) == 1
    assert h.es_scope("ZI", 2018) == {"network": "FDSN:ZI", "year": 2018}


def test_a_refused_exchange_escalates_too():
    # A wrong scope can fail at either end. ZI (temporary) was refused at the
    # exchange with an HTTP error, never reaching S3 - so escalation cannot be
    # driven by the GET alone.
    h = CompositeS3ObjectHelper()
    seen = []

    def fake(net, year=None):
        seen.append(h.es_scope(net, year))
        if "year" in h.es_scope(net, year):
            raise RuntimeError("EarthScope refused credentials for scope ...")
        return _Cred()

    h.get_es_credential = fake
    fs = h.get_es_filesystem("ZI", 2019)
    assert fs is not None
    assert seen == [{"network": "FDSN:ZI", "year": 2019},
                    {"network": "FDSN:ZI"}]


def test_a_refused_exchange_still_raises_when_both_fail():
    h = CompositeS3ObjectHelper()

    def fake(net, year=None):
        raise RuntimeError("refused")

    h.get_es_credential = fake
    try:
        h.get_es_filesystem("ZI", 2019)
    except RuntimeError:
        pass
    else:
        raise AssertionError("should raise once both scopings are refused")


def test_a_missing_network_year_does_not_poison_the_network():
    # ZI 2019 does not exist in the archive (404). ZI 2011 reads at 92 MB/s.
    # Escalating on the 404 would flip ZI to network-only, get a 400 from the
    # next request, mark both scopings spent - and take 2011 down with it.
    from sb_catalog.src.s3_helper import EarthScopeNetworkYearNotFound

    h = CompositeS3ObjectHelper()
    calls = []

    def fake(net, year=None):
        calls.append(year)
        if year == 2019:
            raise EarthScopeNetworkYearNotFound("no such network-year")
        return _Cred()

    h.get_es_credential = fake

    try:
        h.get_es_filesystem("ZI", 2019)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing network-year should surface as one")

    # The scoping is untouched, so the years that DO exist still work.
    assert h.es_scope("ZI", 2011) == {"network": "FDSN:ZI", "year": 2011}
    assert h.get_es_filesystem("ZI", 2011) is not None
    assert h.es_scope_tried.get("ZI", set()) == set()


def test_denial_budget_covers_both_scopings():
    # ES_DENIED_ATTEMPTS bounds the retries in `_read_waveform_from_s3`: the
    # first denial re-requests the same scope (an expiry), the second flips it.
    # Anything less than 2 would abandon the station-day before the alternative
    # scoping was ever tried.
    from sb_catalog.src.s3_helper import ES_DENIED_ATTEMPTS
    assert ES_DENIED_ATTEMPTS >= 2


def test_a_missing_network_year_does_not_fail_the_shard():
    """The whole point of making the 404 a FileNotFoundError.

    In the 2026-09-02 dry run the preflight caught it and continued, then the
    listing loop re-raised it from OUTSIDE its own try block and killed the
    shard anyway - 16 of 48, all 5A/2018, a network-year that does not exist.
    Assert the acquisition sits inside the handler.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "sb_catalog/src/s3_helper.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "load_waveforms")

    guarded = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Try):
            continue
        catches_fnf = any(
            h.type is not None
            and "FileNotFoundError" in ast.dump(h.type)
            for h in node.handlers
        )
        calls_get_fs = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "get_filesystem"
            for stmt in node.body for c in ast.walk(stmt)
        )
        if catches_fnf and calls_get_fs:
            guarded = True
    assert guarded, (
        "get_filesystem must be called INSIDE the try that handles "
        "FileNotFoundError - a 404 network-year otherwise fails the shard"
    )


def test_401_and_403_are_verdicts_not_congestion():
    """The SDK raises its own types for 401/403, bypassing the 4xx fast-fail.

    `UnauthenticatedError` (401) and `UnauthorizedError` (403) are raised
    INSTEAD of an HTTPStatusError, so neither carries `.response` and the
    status-code branch never sees them. Before this was handled, network `LH`
    burned 25 s per year - five attempts with five-second sleeps - re-asking a
    question EarthScope had already answered 403.
    """
    import pathlib
    src = (pathlib.Path(__file__).parent.parent
           / "sb_catalog/src/s3_helper.py").read_text()
    assert "UnauthorizedError" in src and "UnauthenticatedError" in src
    # Both must be handled BEFORE the generic retry/sleep.
    i403 = src.index("isinstance(exc, UnauthorizedError)")
    i401 = src.index("isinstance(exc, UnauthenticatedError)")
    isleep = src.index("time.sleep(5)")
    assert i403 < isleep and i401 < isleep, (
        "401/403 must fail fast, before the retry sleep"
    )


def test_not_entitled_is_distinct_from_missing():
    # 404 (no such network-year) is a plan correction; 403 (not entitled) is a
    # request to EarthScope. Conflating them sends the wrong ticket.
    from sb_catalog.src.s3_helper import (EarthScopeNetworkYearNotFound,
                                          EarthScopeNotEntitled)
    assert not issubclass(EarthScopeNotEntitled, EarthScopeNetworkYearNotFound)
    assert not issubclass(EarthScopeNotEntitled, FileNotFoundError), (
        "403 must not be swallowed by the listing loop's FileNotFoundError "
        "handler - an entitlement gap has to be visible"
    )
