"""Memory must not build across shards.

The 2026-09-02 dry run did not die on a big shard - it died late. `esr4` OOM'd
after 55 minutes and 477 station-days at 16 GB, `west4` after 212. A per-shard
peak kills the first shard that is too big; dying late means the floor is
rising, which is a reclamation problem rather than a sizing one.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.worker import reclaim_memory, rss_mb


def test_reclaim_is_safe_to_call_anywhere():
    # Must never raise: it runs on the failure path, where an exception would
    # replace the real error with one from the cleanup.
    for _ in range(3):
        reclaim_memory()


def test_rss_never_raises():
    # Off Linux there is no /proc; the worker still has to run and log.
    v = rss_mb()
    assert isinstance(v, float) and v >= 0.0


def test_reclaim_runs_on_both_shard_paths():
    """Both the completion and the failure path must reclaim.

    Reclaiming only on success leaves the worse case unhandled: a shard that
    raised mid-read is holding a decoded stream, and its successor inherits
    that floor.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "sb_catalog/src/worker.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "loop")

    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "reclaim_memory"]
    assert len(calls) >= 2, (
        f"expected reclaim_memory() on the success AND failure paths, "
        f"found {len(calls)}"
    )

    # One of them must be inside an exception handler.
    in_handler = any(
        any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "reclaim_memory"
            for c in ast.walk(h))
        for node in ast.walk(fn) if isinstance(node, ast.Try)
        for h in node.handlers
    )
    assert in_handler, "the failure path must reclaim too"


def test_arena_cap_is_in_the_image():
    """MALLOC_ARENA_MAX must be ENV, not a runtime export.

    glibc reads it once at process start, so setting it from Python or a shell
    after launch does nothing at all - a silent no-op that looks like a fix.
    """
    import pathlib
    df = (pathlib.Path(__file__).parent.parent
          / "sb_catalog/Dockerfile").read_text()
    assert "ENV MALLOC_ARENA_MAX=" in df
