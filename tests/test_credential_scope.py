"""Restricted EarthScope credentials must be scoped, cached, and asked for once.

Three independent failures are pinned here, because each one on its own is
invisible until a campaign is running:

1. An UNSCOPED credential for `s3-miniseed-v2` carries `s3:ListBucket` but not
   `s3:GetObject`. Every listing succeeds and every read returns AccessDenied.
   Nothing in the logs distinguishes that from a missing grant of access, and for
   two weeks we read it as one - see docs/rerun_2026/26.

2. Fetching a credential per object would turn every GET into an OAuth round
   trip. The campaign is ~113M station-days behind a few dozen network-years,
   so the exchange has to happen once per scope and be reused.

3. A REFUSAL has to be cached as hard as a success. EarthScope reported on
   2026-09-04 that this fleet was effectively DOSing their credentials
   endpoint: 400/403/404 were being retried, and some requests asked for a
   temporary network with no year - which can never succeed, because temporary
   FDSN codes are reused between experiments and are authorised per year.
   `get_filesystem` is called once per day per network by the listing loop and
   once per object by the read path, so an uncached verdict became hundreds of
   requests per shard per worker.
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


def test_escalation_never_drops_the_year_from_a_temporary_network():
    """The request EarthScope told us never to send.

    Temporary FDSN codes are reassigned between experiments, so EarthScope
    authorises them at the year level and a year-less request for one is a 400
    by construction. The old fallback generated exactly that - it flipped a
    temporary network to `network` scoping after a denial and asked again - and
    it was reachable from a 403, the one status where retrying is guaranteed
    pointless. Escalation may only ADD scope.
    """
    h = _helper([])
    assert "year" in h.es_scope("XD", 2018)
    assert h.escalate_scope_mode("XD") is False
    assert h.es_scope("XD", 2018) == {"network": "FDSN:XD", "year": 2018}


def test_escalation_flips_a_permanent_network_to_year_scoped():
    # The guess can be wrong in the other direction too.
    h = _helper([])
    assert "year" not in h.es_scope("NP", 2018)
    assert h.escalate_scope_mode("NP") is True
    assert h.es_scope("NP", 2018) == {"network": "FDSN:NP", "year": 2018}


def test_escalation_gives_up_once_the_year_has_been_added():
    h = _helper([])
    assert h.escalate_scope_mode("NP") is True      # network -> network+year
    assert h.escalate_scope_mode("NP") is False     # nothing left to try
    # And it stays on the second mode rather than oscillating.
    assert h.es_scope("NP", 2018) == {"network": "FDSN:NP", "year": 2018}


def test_escalation_is_learned_once_per_network():
    calls = []
    h = _helper(calls)
    h.get_filesystem("NP", 2018)
    h.update_es_filesystem("NP", 2018, escalate=True)
    calls.clear()
    # Every later read on that network uses the corrected scoping, with no
    # further exchange - the flip is remembered, not rediscovered per object.
    for _ in range(500):
        h.get_filesystem("NP", 2018)
    assert calls == []
    assert h.es_scope("NP", 2018) == {"network": "FDSN:NP", "year": 2018}


def test_escalation_discards_what_was_cached_under_the_old_mode():
    # The credential was issued for a year-less scope, so it is stale the
    # moment the network moves to year scoping.
    calls = []
    h = _helper(calls)
    h.get_filesystem("NP", 2018)
    assert len(calls) == 1
    h.escalate_scope_mode("NP")
    assert h.credentials == {} and h.es_fs == {}


def test_escalation_leaves_other_networks_alone():
    calls = []
    h = _helper(calls)
    h.get_filesystem("NP", 2018)
    h.get_filesystem("ZI", 2018)
    h.escalate_scope_mode("NP")
    # ZI keeps its credential and its default scoping.
    assert len(h.credentials) == 1
    assert h.es_scope("ZI", 2018) == {"network": "FDSN:ZI", "year": 2018}


def test_a_400_on_a_yearless_scope_adds_the_year_once():
    # A wrong scope can fail at either end: refused at the exchange, or
    # accepted there and denied at the GET. `NP` is guessed permanent, so the
    # first request carries no year; a 400 says that guess was wrong, and the
    # ONE legal correction is to add the year.
    from sb_catalog.src.s3_helper import EarthScopeScopeRefused

    h = CompositeS3ObjectHelper()
    seen = []

    def fake(net, year=None):
        scope = h.es_scope(net, year)
        seen.append(scope)
        if "year" not in scope:
            raise EarthScopeScopeRefused("400: year required for this network")
        return _Cred()

    h.get_es_credential = fake
    fs = h.get_es_filesystem("NP", 2019)
    assert fs is not None
    assert seen == [{"network": "FDSN:NP"},
                    {"network": "FDSN:NP", "year": 2019}]


def test_the_learned_year_scoping_survives_the_next_shard():
    """Reported by EarthScope's developers, 2026-09-04.

    A worker builds a fresh `CompositeS3ObjectHelper` per shard but runs many
    shards in ONE process, so `_ES_STATE["refused"]` outlives the helper. The
    learned scope mode did not, and the two disagreed:

        shard 1  NP defaults to year-less -> 400 -> recorded under the
                 year-less key -> escalate -> asks with the year -> succeeds
        shard 2  fresh helper, mode defaults back to year-less, builds the
                 SAME year-less key, finds shard 1's cached 400 and raises it
                 without ever asking for the year

    The refusal was remembered; the escalation that made it survivable was not.
    So the DoS fix turned "one wasted request per network" into "this network
    is unreadable for the rest of the process", which is the worse failure: it
    loses data rather than bandwidth.
    """
    from sb_catalog.src.s3_helper import EarthScopeScopeRefused

    def shard():
        """What the worker does at the top of each shard."""
        h = CompositeS3ObjectHelper()
        seen = []

        def fake(net, year=None):
            scope = h.es_scope(net, year)
            seen.append(scope)
            if "year" not in scope:
                raise EarthScopeScopeRefused(
                    "400: temporary networks require a year")
            return _Cred()

        h.get_es_credential = fake
        return h, seen

    h1, seen1 = shard()
    assert h1.get_es_filesystem("NP", 2019) is not None
    assert seen1 == [{"network": "FDSN:NP"},
                     {"network": "FDSN:NP", "year": 2019}]

    h2, seen2 = shard()
    assert h2.get_es_filesystem("NP", 2019) is not None, (
        "shard 2 raised shard 1's cached year-less 400 instead of asking "
        "with the year it had already learned to send")
    # And it costs nothing to know: the year-less question is not re-asked.
    assert seen2 == [{"network": "FDSN:NP", "year": 2019}], seen2


def test_the_scope_mode_shares_the_lifetime_of_the_verdicts():
    """The invariant behind the test above, pinned so it cannot drift back.

    `es_scope_key` is built from `es_scope_mode`, and the four scope-keyed
    stores - refusals, the throttle's stamps, the denial counters and the
    credentials - are all looked up with it. If the mode's lifetime is shorter
    than theirs, keys written under one mode are read under another. So this
    asserts identity, not equality: the helper must share the process-wide
    dict rather than hold a copy of it.
    """
    from sb_catalog.src.s3_helper import _ES_STATE

    h = CompositeS3ObjectHelper()
    assert h.es_scope_mode is _ES_STATE["scope_mode"]
    assert h.es_scope_tried is _ES_STATE["scope_tried"]
    assert h.es_refused is _ES_STATE["refused"]

    # And a second helper in the same process sees what the first learned.
    h.escalate_scope_mode("NP")
    assert CompositeS3ObjectHelper().es_scope("NP", 2019) == {
        "network": "FDSN:NP", "year": 2019}


def test_the_sdk_client_can_exchange_from_inside_an_event_loop():
    """Found by the 2026-09-05 dry run; 3 of 8 shards lost to it.

    `SdkContext` picks a sync runner the FIRST time anything is syncified and
    caches it for the life of the context: no running loop in this thread gets
    a SimpleSyncRunner, a running loop gets a BackgroundSyncRunner. One client
    per process - which is what stops the SDK re-bootstrapping OAuth on every
    call - means that single choice is made by whichever caller exchanges
    first and is then wrong for every other caller.

    A shard's first exchange comes from `_check_archives_reachable`, which is
    synchronous. Every later one comes from inside `load_waveforms`, which is
    `async def`. So the context cached the runner that cannot work there, and
    the second year of any shard that crosses a year boundary failed with a
    bare RuntimeError before a request was ever sent - retried five times,
    because a bare RuntimeError carries no status and is not terminal.

    Any shard outliving the one-hour credential TTL would have failed the same
    way at its first renewal, which is most of the campaign.
    """
    import asyncio
    import gc
    import inspect
    import warnings

    import pytest

    pytest.importorskip("earthscope_sdk.common._sync_runner")
    from sb_catalog.src.s3_helper import _ES_STATE

    # Whatever es_client() builds must carry a runner that survives being
    # called from inside a loop. Assert on the runner the SDK would use.
    from earthscope_sdk.common._sync_runner import (BackgroundSyncRunner,
                                                    SimpleSyncRunner)

    async def _work():
        return "ok"

    # The runner we pin: usable from both contexts.
    good = BackgroundSyncRunner().syncify(_work)
    assert good() == "ok"                       # sync caller

    async def _from_loop(fn):
        return fn()
    assert asyncio.run(_from_loop(good)) == "ok"   # caller inside a loop

    # The runner the SDK would have cached from a synchronous first call, and
    # which is what broke in production. Pinned here so that if a future change
    # lets it back in, this test says why it matters.
    bad = SimpleSyncRunner().syncify(_work)
    assert bad() == "ok"
    # The failing call abandons its coroutine, which is the point - it raises
    # before anything is awaited - so silence the warning that proves it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError, match="another loop is running"):
            asyncio.run(_from_loop(bad))
        # Collect it here rather than letting it surface as a warning against
        # whichever unrelated test happens to trigger the next collection.
        gc.collect()

    # And our client is built with the safe one rather than by accident.
    src = inspect.getsource(CompositeS3ObjectHelper.es_client)
    assert "BackgroundSyncRunner" in src, (
        "es_client must pin the runner; letting SdkContext choose makes the "
        "choice depend on whether the first exchange happened to be async")
    assert _ES_STATE["client"] is None          # nothing was really built


def _denying_source(heads, cap=60):
    """An S3DataSource whose every HEAD is refused, the way s3fs refuses one.

    s3fs maps S3 AccessDenied to PermissionError with a message and NO errno,
    which is the whole reason D5 hid: `except OSError` sat first and both
    PermissionError and FileNotFoundError subclass it.
    """
    from sb_catalog.src.s3_helper import S3DataSource

    helper = CompositeS3ObjectHelper()

    class _FS:
        def info(self, uri):
            heads.append(uri)
            if len(heads) > cap:
                raise AssertionError(
                    f"the read loop is spinning: {len(heads)} HEADs for a "
                    f"single denied object, with no way out. That is D5, and "
                    f"in production it ran at socket speed against "
                    f"EarthScope's access point until the 900 s station-day "
                    f"timeout killed it.")
            raise PermissionError("Forbidden")

    helper.get_filesystem = lambda net, year=None: _FS()
    refreshes = []
    helper.update_es_filesystem = (
        lambda net, year=None, escalate=False: refreshes.append(escalate))

    src = object.__new__(S3DataSource)
    src.s3helper = helper
    src.limit_mb = None
    return src, refreshes


def test_a_denied_object_gives_up_instead_of_spinning():
    """D5: the handler that bounds denials had never once executed.

    `PermissionError` and `FileNotFoundError` both subclass `OSError`, and in
    c5846f3 an `except OSError` sat before both, so neither could run. A denied
    object fell out of that branch WITHOUT returning, the enclosing `while True`
    went round again, and the object was re-HEADed as fast as the network
    allowed until STATION_DAY_TIMEOUT killed the station-day 900 s later. One
    profiling shard sat on a single denied GET for 447 minutes.

    The audit could not find a single log line from the PermissionError branch
    across the full five-day CloudWatch retention, which is how we know the
    ES_DENIED_ATTEMPTS budget was dead code. Two live dry runs since have not
    reached it either - we hold credentials that work - so this is the only
    place the branch is actually exercised.
    """
    from sb_catalog.src.s3_helper import ES_DENIED_ATTEMPTS
    import obspy

    heads = []
    src, refreshes = _denying_source(heads)
    st = src._read_waveform_from_s3("s3://es/AV.X.2019.001", "AV", year=2019)

    assert isinstance(st, obspy.Stream) and len(st) == 0, "must end the object"
    # Bounded: one HEAD per refresh attempt, not thousands.
    assert len(heads) <= ES_DENIED_ATTEMPTS + 1, heads
    # And the branch really ran - a refresh was attempted, escalating only on
    # the second denial, never dropping scope.
    assert refreshes == [False, True][:len(refreshes)] and refreshes, refreshes


def test_the_denial_budget_is_spent_per_scope_not_per_object():
    """The budget lived in a local that reset on every object.

    A station-day is thousands of objects; re-running the whole refresh dance
    for each of them is the same tight loop one level up. `note_access_denied`
    counts per scope, so the second denied object on a scope we have already
    given up on costs nothing.
    """
    heads = []
    src, refreshes = _denying_source(heads)
    src._read_waveform_from_s3("s3://es/AV.X.2019.001", "AV", year=2019)
    first_heads, first_refreshes = len(heads), len(refreshes)

    src._read_waveform_from_s3("s3://es/AV.X.2019.002", "AV", year=2019)
    assert len(refreshes) == first_refreshes, (
        "the second object on a scope already given up on re-ran the refresh")
    assert len(heads) - first_heads <= 1, (
        "the second object should cost at most one HEAD, not another budget")


def test_a_403_never_changes_the_scope():
    """403 means "no access", so a differently shaped request cannot help.

    Escalating on it was how a denied temporary network ended up being asked
    for without a year - the request EarthScope singled out.
    """
    from sb_catalog.src.s3_helper import EarthScopeNoAccess

    h = CompositeS3ObjectHelper()
    seen = []

    def fake(net, year=None):
        seen.append(h.es_scope(net, year))
        raise EarthScopeNoAccess("403")

    h.get_es_credential = fake
    try:
        h.get_es_filesystem("ZI", 2019)
    except EarthScopeNoAccess:
        pass
    else:
        raise AssertionError("403 must surface")
    assert seen == [{"network": "FDSN:ZI", "year": 2019}], seen
    assert h.es_scope("ZI", 2019) == {"network": "FDSN:ZI", "year": 2019}


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


def test_no_access_is_distinct_from_missing():
    # 404 (no such network-year) is a plan correction; 403 (no access) is a
    # request to EarthScope. Conflating them sends the wrong ticket.
    from sb_catalog.src.s3_helper import (EarthScopeNetworkYearNotFound,
                                          EarthScopeNoAccess)
    assert not issubclass(EarthScopeNoAccess, EarthScopeNetworkYearNotFound)
    assert not issubclass(EarthScopeNoAccess, FileNotFoundError), (
        "403 must not be swallowed by the listing loop's FileNotFoundError "
        "handler - an access gap has to be visible"
    )


def test_a_denied_network_does_not_kill_the_shard():
    """403 must not propagate out of the listing loop.

    EarthScopeNoAccess is a RuntimeError - deliberately NOT a
    FileNotFoundError, so it stays visible - which means the loop's
    FileNotFoundError handler does not catch it. Unhandled, a shard on a denied
    network burns ten Batch retries, fails permanently, requeues and fails
    again: a network we are simply not allowed to read would look like a broken
    fleet. 127 western shards were in that position before launch.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "sb_catalog/src/s3_helper.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "load_waveforms")
    handlers = [h for node in ast.walk(fn) if isinstance(node, ast.Try)
                for h in node.handlers]
    names = {ast.dump(h.type) for h in handlers if h.type is not None}
    assert any("EarthScopeNoAccess" in n for n in names), (
        "the listing loop must handle a denied network"
    )


# --- a refusal is cached as hard as a success ------------------------------
#
# EarthScope, 2026-09-04: "We were effectively getting DOSed by your workers...
# when it gets a 400/403/404 back from our creds endpoint, it is retrying
# aggressively. 400/403/404s should not be retried."


def test_a_403_is_asked_once_and_never_again():
    """The listing loop calls get_filesystem once per DAY per network.

    Uncached, a single 403 became one credential request per day of the shard -
    ~366 for a year-long shard - repeated by every worker and every requeue.
    """
    from sb_catalog.src.s3_helper import EarthScopeNoAccess

    h = CompositeS3ObjectHelper()
    calls = []

    def fake(net, year=None):
        calls.append((net, year))
        raise EarthScopeNoAccess("403 no access")

    h.get_es_credential = fake

    for _ in range(366):                       # a year-long shard, day by day
        try:
            h.get_filesystem("ZI", 2019)
        except EarthScopeNoAccess:
            pass
    assert len(calls) == 1, f"403 must be asked once, asked {len(calls)}x"


def test_a_404_is_asked_once_and_never_again():
    from sb_catalog.src.s3_helper import EarthScopeNetworkYearNotFound

    h = CompositeS3ObjectHelper()
    calls = []

    def fake(net, year=None):
        calls.append((net, year))
        raise EarthScopeNetworkYearNotFound("404 no such network-year")

    h.get_es_credential = fake

    for _ in range(366):
        try:
            h.get_filesystem("5A", 2018)
        except FileNotFoundError:
            pass
    assert len(calls) == 1, f"404 must be asked once, asked {len(calls)}x"


def test_a_400_is_asked_once_per_scope_including_the_escalation():
    """A 400 may be corrected once, by adding the year. Then it is final."""
    from sb_catalog.src.s3_helper import EarthScopeScopeRefused

    h = CompositeS3ObjectHelper()
    calls = []

    def fake(net, year=None):
        calls.append(h.es_scope(net, year))
        raise EarthScopeScopeRefused("400 bad request")

    h.get_es_credential = fake

    for _ in range(366):
        try:
            h.get_filesystem("NP", 2019)
        except EarthScopeScopeRefused:
            pass
    # Once year-less, once with the year, then never again.
    assert calls == [{"network": "FDSN:NP"},
                     {"network": "FDSN:NP", "year": 2019}], calls


def test_a_401_stops_every_network_not_just_one():
    """A rejected token is not a fact about a network.

    Caching it per scope would let a bad refresh token re-present itself once
    per network per day of the shard.
    """
    from sb_catalog.src.s3_helper import EarthScopeAuthFailed

    h = CompositeS3ObjectHelper()
    calls = []

    def fake(net, year=None):
        calls.append((net, year))
        raise EarthScopeAuthFailed("401 token rejected")

    h.get_es_credential = fake

    for net, year in [("ZI", 2019), ("XD", 2018), ("NP", 2020), ("ZI", 2011)]:
        try:
            h.get_filesystem(net, year)
        except EarthScopeAuthFailed:
            pass
    assert len(calls) == 1, f"401 must stop everything, asked {len(calls)}x"


def test_a_verdict_survives_the_next_shard():
    """A worker claims shard after shard in ONE process.

    Per-instance caching made every shard re-learn the same refusals, so the
    saving was undone every few minutes.
    """
    from sb_catalog.src.s3_helper import EarthScopeNoAccess

    calls = []

    def fake(net, year=None):
        calls.append((net, year))
        raise EarthScopeNoAccess("403 no access")

    for _ in range(20):                        # twenty shards, one process
        h = CompositeS3ObjectHelper()
        h.get_es_credential = fake
        try:
            h.get_filesystem("ZI", 2019)
        except EarthScopeNoAccess:
            pass
    assert len(calls) == 1, f"asked {len(calls)}x across 20 shards"


def test_a_yearless_request_for_a_temporary_network_is_never_sent():
    """The request EarthScope singled out, answered without asking.

    Their words: i see requests for a temporary network without a year - this
    will never succeed.
    """
    from sb_catalog.src.s3_helper import EarthScopeScopeIncomplete

    h = CompositeS3ObjectHelper()
    sent = []

    def fake_client():
        raise AssertionError(f"a request was sent: {sent}")

    h.es_client = fake_client
    try:
        h.get_es_credential("ZI")              # temporary code, no year
    except EarthScopeScopeIncomplete:
        pass
    else:
        raise AssertionError("a year-less temporary scope must be refused")


def test_the_exchange_is_rate_limited_per_scope():
    """A refresh loop must hit our own limiter, not EarthScope's."""
    from sb_catalog.src.s3_helper import (ES_REFRESH_BUDGET,
                                          EarthScopeExchangeThrottled)

    h = CompositeS3ObjectHelper()
    calls = []

    class _Client:
        class user:
            @staticmethod
            def get_aws_credentials(**kw):
                calls.append(kw)
                return _Cred()

    h.es_client = lambda: _Client
    for _ in range(ES_REFRESH_BUDGET):
        h.get_es_credential("ZI", 2019)
    try:
        h.get_es_credential("ZI", 2019)
    except EarthScopeExchangeThrottled:
        pass
    else:
        raise AssertionError("the budget must stop the next exchange")
    assert len(calls) == ES_REFRESH_BUDGET


def test_the_sdk_client_is_built_once_per_process():
    """earthscope-sdk 1.8.0 caches issued credentials in memory on the service
    object and nowhere else - the disk cache 1.3.x had is gone. A client per
    call threw that cache away every time and re-bootstrapped OAuth."""
    import sb_catalog.src.s3_helper as sh

    built = []

    class _FakeClient:
        def __init__(self):
            built.append(1)

        def close(self):
            pass

    sh._ES_STATE["client"] = None
    try:
        h1, h2 = CompositeS3ObjectHelper(), CompositeS3ObjectHelper()
        sh._ES_STATE["client"] = _FakeClient()   # stand in for the real one
        for _ in range(50):
            assert h1.es_client() is h2.es_client()
        assert len(built) == 1
    finally:
        sh._ES_STATE["client"] = None


def test_denials_are_counted_per_scope_not_per_object():
    """`_read_waveform_from_s3` resets its own counter on every call.

    On its own that re-ran the refresh for each of the thousands of objects in
    a station-day we cannot read.
    """
    h = CompositeS3ObjectHelper()
    for i in range(1, 6):
        assert h.note_access_denied("ZI", 2019) == i
    # A read that succeeds says the scope is fine after all.
    h.clear_access_denied("ZI", 2019)
    assert h.note_access_denied("ZI", 2019) == 1
    # And another scope is unaffected.
    assert h.note_access_denied("ZI", 2011) == 1


def test_permission_and_filenotfound_are_handled_before_oserror():
    """Both subclass OSError, so an OSError handler placed first shadows them.

    It did, since 070a4fd. A denied object never refreshed and never returned:
    the `while True` loop re-HEADed it as fast as the socket allowed until the
    900 s station-day timeout. Tight request loop, per denied object - and the
    ES_DENIED_ATTEMPTS budget below it had never once executed.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "sb_catalog/src/s3_helper.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_read_waveform_from_s3")
    order = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                if h.type is None:
                    continue
                order += [n.id for n in ast.walk(h.type)
                          if isinstance(n, ast.Name)]
    for narrow in ("PermissionError", "FileNotFoundError"):
        assert narrow in order and "OSError" in order
        assert order.index(narrow) < order.index("OSError"), (
            f"{narrow} subclasses OSError; handled after it, it is dead code"
        )


def test_an_auth_flow_error_is_terminal_and_global():
    """A bad refresh token is not congestion and is not about a scope.

    `InvalidRefreshTokenError` and friends carry no `.response`, so the
    status-code branch never saw them and they fell into the retry loop - and
    every attempt re-ran the refresh grant against login.earthscope.org. Five
    token-endpoint hits per scope, per shard, per worker, for a token that had
    already been rejected.
    """
    import sys
    import types

    from sb_catalog.src.s3_helper import EarthScopeAuthFailed

    class _AuthFlowError(Exception):
        pass

    class _InvalidRefreshToken(_AuthFlowError):
        pass

    err = types.ModuleType("earthscope_sdk.auth.error")
    err.AuthFlowError = _AuthFlowError
    err.UnauthorizedError = type("UnauthorizedError", (_AuthFlowError,), {})
    err.UnauthenticatedError = type(
        "UnauthenticatedError", (_AuthFlowError,), {})
    saved = sys.modules.get("earthscope_sdk.auth.error")
    sys.modules["earthscope_sdk.auth.error"] = err

    attempts = []
    h = CompositeS3ObjectHelper()

    class _Client:
        class user:
            @staticmethod
            def get_aws_credentials(**kw):
                attempts.append(kw)
                raise _InvalidRefreshToken("refresh token exchange failed")

    h.es_client = lambda: _Client
    try:
        for net, year in [("ZI", 2019), ("XD", 2018), ("NP", 2020)]:
            try:
                h.get_es_credential(net, year)
            except EarthScopeAuthFailed:
                pass
            else:
                raise AssertionError("an auth-flow error must be terminal")
        assert len(attempts) == 1, (
            f"a rejected token must be presented once, not {len(attempts)}x"
        )
    finally:
        if saved is not None:
            sys.modules["earthscope_sdk.auth.error"] = saved
        else:
            del sys.modules["earthscope_sdk.auth.error"]


def test_every_4xx_is_remembered_not_just_the_three_we_named():
    """A status nobody had seen yet must not slip past the verdict store.

    400/403/404 each had a type; everything else raised a bare RuntimeError,
    which `ES_TERMINAL` does not cover - so a 409 or a 422 would have been
    re-asked on every one of the 11,315 calls a shard makes, which is the
    whole bug. Found by review on PR #32.
    """
    from sb_catalog.src.s3_helper import EarthScopeRequestRefused

    class _Resp:
        def __init__(self, code):
            self.status_code, self.text, self.headers = code, "nope", {}

    class _HTTPError(Exception):
        def __init__(self, code):
            super().__init__(f"HTTP {code}")
            self.response = _Resp(code)

    for code in (402, 405, 409, 410, 422, 451):
        h = CompositeS3ObjectHelper()
        from sb_catalog.src.s3_helper import reset_earthscope_state
        reset_earthscope_state()
        calls = []

        class _Client:
            class user:
                @staticmethod
                def get_aws_credentials(**kw):
                    calls.append(kw)
                    raise _HTTPError(code)

        h.es_client = lambda: _Client
        for _ in range(500):
            try:
                h.get_filesystem("AV", 2019)
            except EarthScopeRequestRefused:
                pass
            except Exception as exc:
                raise AssertionError(
                    f"HTTP {code} surfaced as {type(exc).__name__}, which is "
                    f"not in ES_TERMINAL and so is never remembered") from exc
        assert len(calls) == 1, f"HTTP {code} asked {len(calls)}x, expected 1"


def test_a_cached_verdict_carries_no_traceback_and_no_context_chain():
    """We store the class and its message, never the caught instance.

    An instance holds __traceback__, and through it every frame it passed -
    the helper, the shard, whatever those reference - pinned for the life of a
    process-wide store. And `raise verdict` happens inside an `except` block,
    so Python would assign __context__ on every raise: one shared object would
    grow a chain thousands deep and eventually a cycle. Found by review on
    PR #32.
    """
    from sb_catalog.src.s3_helper import EarthScopeNoAccess

    h = CompositeS3ObjectHelper()

    def fake(net, year=None):
        raise EarthScopeNoAccess("403 no access")

    h.get_es_credential = fake

    seen = []
    for _ in range(200):
        try:
            h.get_filesystem("ZI", 2019)
        except EarthScopeNoAccess as exc:
            seen.append(exc)

    # A fresh object each time, not one shared instance handed back repeatedly.
    assert len(set(id(e) for e in seen)) == len(seen), (
        "the same exception instance is being re-raised; __context__ will "
        "accumulate on it")
    for exc in seen[:5]:
        assert exc.__traceback__ is not None   # set by the raise, not retained
    # Nothing chained: depth stays 1 however many times it is raised.
    def depth(e):
        n, seen_ids = 0, set()
        while e.__context__ is not None and id(e) not in seen_ids:
            seen_ids.add(id(e)); e = e.__context__; n += 1
            if n > 50: break
        return n
    assert depth(seen[-1]) == 0, (
        f"__context__ chain is {depth(seen[-1])} deep after 200 raises")

    # And what is stored is a description, not a caught exception.
    record = list(h.es_refused.values())[0]
    assert isinstance(record, tuple) and record[0] is EarthScopeNoAccess
