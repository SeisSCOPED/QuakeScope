#!/usr/bin/env python
"""Prove that this client does not retry EarthScope's refusals.

Written for EarthScope after the 2026-09-04 report that our workers were
effectively DOSing the credentials endpoint. It is meant to be run BY
EarthScope, with an EarthScope account of their own, and it is deliberately
cheap: the offline mode sends nothing at all, and the live mode sends a
counted, capped handful of requests.

WHAT IT CHECKS

    1. A 403 is asked once and never again, however many times the caller asks.
    2. A 404 is asked once and never again.
    3. A 400 is corrected at most once - by ADDING the year, never by removing
       it - and then never asked again.
    4. A rejected refresh token ("user is blocked") stops every network at
       once, rather than being re-presented per network per day.
    5. No request for a temporary FDSN network is ever sent without a year.

The caller in each case is a realistic shard: 365 days x 30 stations, which is
how many times `CompositeS3ObjectHelper.get_filesystem` is invoked for one
network in one shard. That is the multiplier that turned single refusals into
thousands of requests in the deployed version.

USAGE

    # No EarthScope account needed, no network traffic at all.
    python scripts/earthscope_selftest.py --offline

    # Same checks, but against the real endpoint with YOUR OWN account.
    # Sends at most --max-requests requests (default 12) and refuses to exceed
    # it. Every outbound request is printed with its full query string.
    export ES_OAUTH2__REFRESH_TOKEN=...        # or: earthscope-cli login
    python scripts/earthscope_selftest.py --live \
        --allowed-network AV --allowed-year 2019 \
        --denied-network   LH --missing-network 5A --missing-year 2018

    # Side by side with the version that caused the incident.
    git show c5846f3:sb_catalog/src/s3_helper.py > /tmp/old_s3_helper.py
    python scripts/earthscope_selftest.py --offline --baseline /tmp/old_s3_helper.py

EXIT CODE is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# One shard's worth of calls on ONE network. The listing loop in
# `S3DataSource.load_waveforms` calls get_filesystem once per day per network;
# `_read_waveform_from_s3` calls it once per object.
SHARD_DAYS = 365
SHARD_STATIONS = 30
SHARD_CALLS = SHARD_DAYS + SHARD_DAYS * SHARD_STATIONS      # 11,315

GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = BOLD = OFF = ""


# --------------------------------------------------------------------------
# A fake earthscope_sdk. Installed only in --offline mode, so the checks can
# run with no account, no token and no network.
# --------------------------------------------------------------------------

class _FakeAuthFlowError(Exception): ...
class _FakeUnauthorized(_FakeAuthFlowError): ...        # 403
class _FakeUnauthenticated(_FakeAuthFlowError): ...     # 401
class _FakeInvalidRefreshToken(_FakeAuthFlowError): ...  # blocked account


class _FakeResponse:
    """Enough of an httpx.Response for the status-code branch."""

    def __init__(self, status, body=""):
        self.status_code, self.text, self.headers = status, body, {}


class _FakeHTTPStatusError(Exception):
    def __init__(self, status, body=""):
        super().__init__(f"HTTP {status}")
        self.response = _FakeResponse(status, body)


def install_fake_sdk(behaviour, log):
    """behaviour(params) -> credential, or raises. `log` records every call."""

    class _User:
        @staticmethod
        def get_aws_credentials(**kw):
            log.append(kw)
            return behaviour(kw)

    class EarthScopeClient:
        user = _User()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def close(self):
            pass

    sdk = types.ModuleType("earthscope_sdk")
    sdk.EarthScopeClient = EarthScopeClient
    err = types.ModuleType("earthscope_sdk.auth.error")
    err.AuthFlowError = _FakeAuthFlowError
    err.UnauthorizedError = _FakeUnauthorized
    err.UnauthenticatedError = _FakeUnauthenticated
    auth = types.ModuleType("earthscope_sdk.auth")
    auth.error = err
    sys.modules.update({"earthscope_sdk": sdk, "earthscope_sdk.auth": auth,
                        "earthscope_sdk.auth.error": err})


def _cred(minutes=60):
    import datetime

    class _S:
        def __init__(self, v):
            self._v = v

        def get_secret_value(self):
            return self._v

    class _C:
        aws_access_key_id = "AKIAEXAMPLE"
        aws_secret_access_key = _S("secret")
        aws_session_token = _S("token")
        expiration = (datetime.datetime.now(datetime.timezone.utc)
                      + datetime.timedelta(minutes=minutes))

    return _C()


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

class Check:
    def __init__(self, name, why):
        self.name, self.why = name, why
        self.sent, self.expected, self.ok, self.note = 0, 0, False, ""


class _NoSleep:
    """Stands in for the `time` module inside the client under test.

    The deployed version sleeps 5 s between credential retries, so replaying a
    shard against it in real time takes days. Accounting for the sleep instead
    of serving it makes the run finish AND turns the delay into a number: how
    long that shard would have spent waiting to retry a question that had
    already been answered.
    """

    def __init__(self, real):
        self._real, self.slept, self.calls = real, 0.0, 0

    def sleep(self, seconds):
        self.slept += seconds
        self.calls += 1

    def __getattr__(self, name):
        return getattr(self._real, name)


def run_offline(mod, label):
    """Every check, against `mod` (a loaded s3_helper module)."""
    import time as _real_time
    clock = _NoSleep(_real_time)
    mod.time = clock                       # the client's own `import time`
    results = []
    reset = getattr(mod, "reset_earthscope_state", lambda: None)

    def shard(helper, net, year):
        """One shard's worth of get_filesystem calls on one network."""
        for _ in range(SHARD_CALLS):
            try:
                helper.get_filesystem(net, year)
            except Exception:
                pass

    # -- 1. 403 -----------------------------------------------------------
    c = Check("403 is asked once",
              "EarthScope: 403 means you do not have access to the network or "
              "network+year. Retrying cannot change that answer.")
    log = []
    install_fake_sdk(lambda kw: (_ for _ in ()).throw(_FakeUnauthorized("403")), log)
    reset()
    shard(mod.CompositeS3ObjectHelper(), "ZI", 2019)
    c.sent, c.expected = len(log), 1
    c.ok = c.sent == 1
    results.append(c)

    # -- 2. 404 -----------------------------------------------------------
    c = Check("404 is asked once",
              "EarthScope: 404 means the FDSN code was not found. The campaign "
              "plan legitimately contains network-years the archive never held.")
    log = []
    install_fake_sdk(lambda kw: (_ for _ in ()).throw(_FakeHTTPStatusError(404, "not found")), log)
    reset()
    shard(mod.CompositeS3ObjectHelper(), "5A", 2018)
    c.sent, c.expected = len(log), 1
    c.ok = c.sent == 1
    results.append(c)

    # -- 3. 400, corrected once by ADDING the year -------------------------
    c = Check("400 is corrected once, by adding the year",
              "A network we guessed permanent may in fact be authorised per "
              "year. That is worth exactly one more request, and it must ADD "
              "scope, never remove it.")
    log = []
    install_fake_sdk(lambda kw: (_ for _ in ()).throw(_FakeHTTPStatusError(400, "year required")), log)
    reset()
    shard(mod.CompositeS3ObjectHelper(), "NP", 2019)   # NP: guessed permanent
    c.sent, c.expected = len(log), 2
    c.ok = c.sent == 2
    c.note = " -> ".join(
        ("no year" if "year" not in k else f"year={k['year']}") for k in log[:4])
    results.append(c)

    # -- 4. blocked refresh token ------------------------------------------
    c = Check("a blocked account stops every network at once",
              "This is the 2026-09-04 incident. InvalidRefreshTokenError has "
              "no .response, so it fell past the status-code check into the "
              "generic retry - 5 POSTs to login.earthscope.org per scope.")
    log = []
    install_fake_sdk(
        lambda kw: (_ for _ in ()).throw(_FakeInvalidRefreshToken(
            '{"error":"invalid_grant","error_description":"user is blocked"}')), log)
    reset()
    h = mod.CompositeS3ObjectHelper()
    for net, year in (("ZI", 2019), ("XD", 2018), ("NP", 2020),
                      ("AV", 2011), ("4F", 2019), ("AX", 2016)):
        for _ in range(SHARD_DAYS):
            try:
                h.get_filesystem(net, year)
            except Exception:
                pass
    c.sent, c.expected = len(log), 1
    c.ok = c.sent == 1
    results.append(c)

    # -- 5. never a yearless request for a temporary network ---------------
    c = Check("no yearless request for a temporary network",
              'EarthScope: "i see requests for a temporary network without a '
              'year - this will never succeed". Temporary FDSN codes are '
              "reused, so they are authorised per year.")
    log = []
    install_fake_sdk(lambda kw: (_ for _ in ()).throw(_FakeUnauthorized("403")), log)
    reset()
    h = mod.CompositeS3ObjectHelper()
    # Temporary codes: a leading digit, or X/Y/Z.
    for net in ("ZI", "XD", "YW", "1D", "7D", "ZG"):
        for _ in range(200):
            try:
                h.get_filesystem(net, 2019)
            except Exception:
                pass
            try:
                # The denial path, which is what used to drop the year.
                h.update_es_filesystem(net, 2019, escalate=True)
            except Exception:
                pass
    bad = [k for k in log
           if k.get("network", "")[5:6] in "0123456789XYZ" and "year" not in k]
    c.sent, c.expected = len(bad), 0
    c.ok = not bad
    c.note = (f"{len(log)} requests total, {len(bad)} of them yearless"
              if log else "no requests")
    results.append(c)

    if clock.calls:
        c = Check("no retry sleeps at all",
                  "A verdict needs no backoff. Time spent sleeping between "
                  "retries is time spent preparing to re-ask a question "
                  "EarthScope has already answered.")
        c.sent, c.expected = clock.calls, 0
        c.ok = False
        c.note = (f"{clock.calls:,} sleeps totalling {clock.slept:,.0f} s "
                  f"= {clock.slept/3600:,.1f} h per shard, per worker")
        results.append(c)
    else:
        c = Check("no retry sleeps at all",
                  "A verdict needs no backoff. Time spent sleeping between "
                  "retries is time spent preparing to re-ask a question "
                  "EarthScope has already answered.")
        c.sent, c.expected, c.ok = 0, 0, True
        results.append(c)

    return results


# --------------------------------------------------------------------------
# Live mode: the same client, the real endpoint, every request printed.
# --------------------------------------------------------------------------

def run_live(args):
    """A counted, capped set of real requests using the caller's own account."""
    import httpx

    sent = []
    original = httpx.AsyncClient.send

    async def counting_send(self, request, *a, **kw):
        sent.append(str(request.url))
        print(f"    {DIM}-> {request.method} {request.url}{OFF}")
        if len(sent) > args.max_requests:
            raise SystemExit(
                f"{RED}ABORTED: exceeded --max-requests={args.max_requests}. "
                f"That is the whole point of this script - it should never "
                f"get here.{OFF}")
        return await original(self, request, *a, **kw)

    # try/finally, because the cap above aborts with SystemExit: on that path
    # the restore at the end of the function is never reached and every later
    # httpx request in the process would keep going through `counting_send`.
    # The abort is the one exit this script most expects to take.
    httpx.AsyncClient.send = counting_send
    try:
        return _run_live_cases(args, _live_module(), sent)
    finally:
        httpx.AsyncClient.send = original


def _live_module():
    os.environ.setdefault("EARTHSCOPE_S3_ACCESS_POINT",
                          "earthscope-mseed-v2-4fdodyzpsz8u8uyi3pa9qsw9oid1suse2a-s3alias")
    from sb_catalog.src import s3_helper as mod
    return mod


def _run_live_cases(args, mod, sent):
    print(f"{BOLD}LIVE - against api.earthscope.org with your own account{OFF}")
    print(f"  hard cap: {args.max_requests} requests. Every one is printed.\n")

    results = []
    cases = []
    if args.allowed_network:
        cases.append(("a network you CAN read", args.allowed_network,
                      args.allowed_year, "expect 1 request, then a credential"))
    if args.denied_network:
        cases.append(("a network you CANNOT read (403)",
                      args.denied_network, args.denied_year,
                      "expect 1 request, then silence"))
    if args.missing_network:
        cases.append(("a network-year that does not exist (404)",
                      args.missing_network, args.missing_year,
                      "expect 1 request, then silence"))

    for title, net, year, expectation in cases:
        mod.reset_earthscope_state()
        before = len(sent)
        print(f"  {BOLD}{title}{OFF}: {net} {year or ''}   {DIM}{expectation}{OFF}")
        h = mod.CompositeS3ObjectHelper()
        outcome = "credential issued"
        # 200 calls stands in for a shard. The deployed version would have sent
        # 200 requests here; this one sends at most 2.
        for i in range(200):
            try:
                h.get_filesystem(net, year)
            except Exception as exc:
                outcome = f"{type(exc).__name__}"
        n = len(sent) - before
        c = Check(f"{net} {year or ''}: {title}", expectation)
        c.sent, c.expected = n, 2
        c.ok = n <= 2
        c.note = f"{outcome}; 200 calls -> {n} request(s)"
        results.append(c)
        print(f"    {GREEN if c.ok else RED}{n} request(s) for 200 calls{OFF}"
              f"  ({outcome})\n")

    return results, sent


# --------------------------------------------------------------------------

def load_module(path, name):
    """Load a specific s3_helper.py, including an old one from git."""
    if path is None:
        os.environ.setdefault("EARTHSCOPE_S3_ACCESS_POINT", "selftest-access-point")
        from sb_catalog.src import s3_helper
        return s3_helper
    # An old copy has package-relative imports, so it has to live in the
    # package to load at all.
    dest = os.path.join(REPO, "sb_catalog", "src", f"_{name}.py")
    shutil.copyfile(path, dest)
    try:
        os.environ.setdefault("EARTHSCOPE_S3_ACCESS_POINT", "selftest-access-point")
        import sb_catalog.src  # noqa: F401
        spec = importlib.util.spec_from_file_location(
            f"sb_catalog.src._{name}", dest)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"sb_catalog.src._{name}"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        try:
            os.remove(dest)
        except OSError:
            pass


def report(title, results) -> bool:
    print(f"\n{BOLD}{title}{OFF}")
    print("-" * 78)
    allok = True
    for c in results:
        mark = f"{GREEN}PASS{OFF}" if c.ok else f"{RED}FAIL{OFF}"
        allok &= c.ok
        print(f"  {mark}  {c.name}")
        print(f"        requests sent: {BOLD}{c.sent}{OFF}   "
              f"expected: {c.expected}"
              + (f"   [{c.note}]" if c.note else ""))
        for line in _wrap(c.why, 68):
            print(f"        {DIM}{line}{OFF}")
    return allok


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def do_login() -> int:
    """Device-code sign-in with the caller's own EarthScope account.

    Uses the SDK directly, so there is no extra package to install and no
    password is typed into this script: it prints a URL and a code, you approve
    it in a browser, and the SDK writes the tokens to
    ~/.earthscope/default/tokens.json. Everything else here then just works.
    """
    from earthscope_sdk import EarthScopeClient
    with EarthScopeClient() as client:
        client.ctx.device_code_flow.do_flow()
        who = client.user.get_profile()
        print(f"\nsigned in as {who.first_name} {who.last_name} "
              f"<{who.primary_email}>  ({who.institution})")
        print(f"user_id: {who.user_id}")
    print("\nTokens are in ~/.earthscope/default/tokens.json. "
          "Now run the checks:")
    print("  python scripts/earthscope_selftest.py --offline")
    print("  python scripts/earthscope_selftest.py --live "
          "--denied-network <a network you may NOT read>")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--login", action="store_true",
                    help="sign in with your own EarthScope account (device "
                         "code, no password typed here) and exit")
    ap.add_argument("--offline", action="store_true",
                    help="send nothing; needs no EarthScope account")
    ap.add_argument("--live", action="store_true",
                    help="send a small, capped number of real requests")
    ap.add_argument("--baseline", metavar="PATH",
                    help="also run the checks against an older s3_helper.py, "
                         "for a side-by-side count")
    ap.add_argument("--max-requests", type=int, default=12,
                    help="hard ceiling in --live mode (default 12)")
    ap.add_argument("--allowed-network"), ap.add_argument("--allowed-year", type=int)
    ap.add_argument("--denied-network"), ap.add_argument("--denied-year", type=int)
    ap.add_argument("--missing-network"), ap.add_argument("--missing-year", type=int)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show the client's own log lines as well")
    args = ap.parse_args(argv)

    if args.login:
        return do_login()

    if not (args.offline or args.live):
        args.offline = True

    # The client logs a line every time it records a verdict. That is the right
    # behaviour in a campaign and pure noise here, where the counts are the
    # evidence.
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.CRITICAL,
        format="%(levelname)s | %(message)s")
    logging.getLogger("picker").setLevel(
        logging.DEBUG if args.verbose else logging.CRITICAL)

    print(f"{BOLD}EarthScope credential-retrieval self-test{OFF}")
    print(f"one shard = {SHARD_DAYS} days x {SHARD_STATIONS} stations = "
          f"{SHARD_CALLS:,} calls to get_filesystem() on one network\n")

    ok = True
    if args.offline:
        if args.baseline:
            base = load_module(args.baseline, "baseline")
            base_results = run_offline(base, "baseline")
            report("BASELINE - the version deployed during the incident",
                   base_results)
            print(f"\n{DIM}  (failures above are expected: that is the bug){OFF}")
            for k in [k for k in sys.modules if "s3_helper" in k]:
                del sys.modules[k]
        mod = load_module(None, "current")
        ok &= report("CURRENT - this working tree", run_offline(mod, "current"))

    if args.live:
        results, sent = run_live(args)
        ok &= report("LIVE - real requests to api.earthscope.org", results)
        print(f"\n  total requests sent this run: {BOLD}{len(sent)}{OFF} "
              f"(cap {args.max_requests})")
        yearless = [u for u in sent
                    if "network=FDSN%3A" in u and "year=" not in u
                    and u.split("network=FDSN%3A")[1][:1] in "0123456789XYZ"]
        print(f"  yearless temporary-network requests: "
              f"{BOLD}{len(yearless)}{OFF}  (must be 0)")
        ok &= not yearless

    print()
    print(f"{GREEN}{BOLD}ALL CHECKS PASSED{OFF}" if ok
          else f"{RED}{BOLD}CHECKS FAILED{OFF}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
