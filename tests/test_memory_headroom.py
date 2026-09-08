"""A floor cannot warn you about a ceiling.

126 jobs were OOM-killed on 2026-09-03 at a 4.8-6.2% rate, and nothing in the
logs saw it coming. The reason is that RSS was sampled immediately after
reclaim_memory(), which is by construction the low point: the last reclaim
before one obs worker died reported 775 MB against a 16,384 MB limit, and all
four worker processes together summed to 3.8 GB - under a quarter of the
container. Individual reclaims in that same run handed back 1.9 GB, so the
peaks the OOM killer acted on were multiples of the number being written down.

These pin the three readings that make headroom visible: the peak, the limit,
and the ability to reset the peak so it describes one shard rather than a
process's whole life.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.worker import (memory_limit_mb, peak_rss_mb,  # noqa: E402
                                   reset_peak_rss, rss_mb)

LINUX = sys.platform.startswith("linux")


def test_peak_is_never_below_current():
    """VmHWM is a high-water mark, so it bounds RSS from above."""
    if not LINUX:
        assert peak_rss_mb() == 0.0      # documented fallback, not a crash
        return
    peak, now = peak_rss_mb(), rss_mb()
    assert peak > 0
    assert peak >= now - 1.0             # a page of slack for sampling skew


def test_peak_notices_an_allocation_the_floor_forgets():
    """The distinction the whole change exists for.

    Allocate, free, and the floor comes back down while the peak remembers.
    That memory is exactly what killed the 2026-09-03 workers.
    """
    if not LINUX:
        return
    reset_peak_rss()
    base = peak_rss_mb()
    big = bytearray(256 * 1024 * 1024)   # 256 MB, touched so it is resident
    for i in range(0, len(big), 4096):
        big[i] = 1
    peak_while_held = peak_rss_mb()
    del big
    floor_after = rss_mb()
    peak_after = peak_rss_mb()
    assert peak_while_held > base + 200
    # The peak still reports it after the memory is gone; the floor does not.
    assert peak_after >= peak_while_held - 1.0
    assert peak_after > floor_after


def test_the_limit_is_read_or_honestly_absent():
    """A peak without its limit is not a headroom."""
    v = memory_limit_mb()
    assert v >= 0.0
    if v:
        # Never the v1 unlimited sentinel dressed up as a real number.
        assert v < 1024 ** 3


def test_reset_makes_the_peak_per_shard():
    """Without a reset, VmHWM is peak-since-start and says nothing about now."""
    if not LINUX:
        return
    big = bytearray(128 * 1024 * 1024)
    for i in range(0, len(big), 4096):
        big[i] = 1
    del big
    high = peak_rss_mb()
    reset_peak_rss()
    after = peak_rss_mb()
    # Either the kernel supports the reset and the mark drops, or it does not
    # and the mark holds - both are documented, neither may raise.
    assert after <= high + 1.0


def test_max_hours_is_a_real_option_with_a_safe_default():
    """Bounding worker AGE is the mitigation; it must not change today's runs.

    The OOM is not a shard that is too big - it is a floor that rises across
    shards. 21% of jobs that ran past 8 hours were OOM-killed on 2026-09-03,
    against none that finished sooner, and both mitigations already shipped
    (reclaim between shards, MALLOC_ARENA_MAX in the image) were present in the
    builds that died. So the worker leaves before the ratchet reaches the
    ceiling. Default 0 keeps every existing invocation identical.
    """
    import argparse
    import inspect

    from sb_catalog.src import worker

    src = inspect.getsource(worker)
    assert '"--max-hours"' in src

    # Default is off, so nothing that does not ask for it changes behaviour.
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", default=0.0, type=float)
    assert ap.parse_args([]).max_hours == 0.0
    assert ap.parse_args(["--max-hours", "6"]).max_hours == 6.0

    # It must break the shard loop, not abort mid-shard: a killed worker loses
    # its shard, a worker that finishes and leaves hands it over.
    # The guard lives in the shard loop, which is defined ABOVE the argparse
    # block, so search the whole source rather than forward from the flag.
    guard = src[src.index("if args.max_hours:"):]
    assert "break" in guard.split("\n\n")[0]
    assert "Exiting cleanly" in guard[:900]


def test_max_hours_is_never_passed_unless_asked():
    """A flag an older image does not know kills the whole fleet at once.

    The worker parses this command inside the container. argparse exits 2 on an
    unrecognised flag, so adding --max-hours to the governor's command before
    the deployed image accepts it would take down every worker the next top-up
    submitted - a fleet-wide outage from a one-word change. It must be absent
    from the command line entirely at the default, not passed as 0.
    """
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    import spot_governor

    def cmd_for(**kw):
        a = argparse.Namespace(campaign="s3://b/c", weight="w", procs=4,
                               checkpoint_every=40, lease_hours=1.0,
                               max_hours=0.0, threads=2, queue="q",
                               job_definition="jd", name_prefix="p")
        for k, v in kw.items():
            setattr(a, k, v)
        seen = {}

        class FakeBatch:
            def submit_job(self, **kwargs):
                seen.update(kwargs)
                return {"jobId": "x"}
        spot_governor.submit(FakeBatch(), a, 1)
        return seen["containerOverrides"]["command"]

    assert "--max-hours" not in cmd_for()
    assert "--max-hours" not in cmd_for(max_hours=0)
    on = cmd_for(max_hours=6)
    assert "--max-hours" in on and on[on.index("--max-hours") + 1] == "6"

    # checkpoint-every is always passed, and follows the argument rather than
    # the hardcoded 40 it used to be.
    c = cmd_for(checkpoint_every=20)
    assert c[c.index("--checkpoint-every") + 1] == "20"
