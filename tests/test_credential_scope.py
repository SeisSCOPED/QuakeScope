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
    scope = CompositeS3ObjectHelper.es_scope("AV", 2019)
    assert scope["network"] == "FDSN:AV"


def test_permanent_networks_are_not_year_scoped():
    # A permanent code means one deployment for all time, so a year would
    # fragment the cache into one credential per year for no benefit.
    for net in ("AV", "CC", "NP", "BK", "CI"):
        assert CompositeS3ObjectHelper.es_scope(net, 2019) == {
            "network": f"FDSN:{net}"}


def test_temporary_networks_are_year_scoped():
    # FDSN reassigns codes beginning with a digit or X/Y/Z, so the same code is
    # a different experiment in a different year.
    for net in ("XD", "ZI", "ZG", "1D", "7D", "YW"):
        assert CompositeS3ObjectHelper.es_scope(net, 2018) == {
            "network": f"FDSN:{net}", "year": 2018}


def test_year_omitted_when_unknown():
    assert "year" not in CompositeS3ObjectHelper.es_scope("XD")


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
