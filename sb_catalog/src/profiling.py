"""
Per-stage timing for one shard, so cost can be attributed rather than guessed.

The v3 cost model was built on a single wall-clock number per station-day and an
assumption that inference dominated it. That assumption was never tested, and the
pipeline has at least six stages that scale with different things: S3 LIST scales
with objects in a day prefix, GET with bytes, inference with samples, and the two
amplitude passes with *pick count* - which varies fourfold between an ordinary
day and a mainshock day.

Each stage records seconds, how many times it ran, and the quantity it scales
with (bytes, picks, windows), because a stage that is slow per call and a stage
that is called far too often need different fixes.

Disabled by default and costs a boolean check per call site, so instrumentation
can live in the real code path instead of a parallel harness that drifts from it.

Usage:
    from .profiling import profile, stage

    profile.enable()
    with stage("s3.get", bytes=n):
        ...
    print(profile.report())
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional


class Profile:
    """Thread-safe stage accumulator. One per process."""

    def __init__(self) -> None:
        self.enabled = False
        self._lock = threading.Lock()
        self._seconds: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)
        self._units: dict[str, float] = defaultdict(float)
        self._unit_name: dict[str, str] = {}
        self._t0: Optional[float] = None

    def enable(self) -> None:
        self.enabled = True
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        with self._lock:
            self._seconds.clear()
            self._calls.clear()
            self._units.clear()
            self._unit_name.clear()
        self._t0 = time.perf_counter()

    def record(self, name: str, seconds: float, unit: float, unit_name: str) -> None:
        with self._lock:
            self._seconds[name] += seconds
            self._calls[name] += 1
            self._units[name] += unit
            if unit_name:
                self._unit_name[name] = unit_name

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0 if self._t0 is not None else 0.0

    def report(self) -> str:
        """Stage table, slowest first, with the share of wall-clock accounted for.

        The unaccounted line is the point of the whole exercise: if the stages do
        not add up to wall-clock, something expensive is not instrumented yet.
        """
        with self._lock:
            rows = sorted(self._seconds.items(), key=lambda kv: -kv[1])
            total = self.elapsed()
            acct = sum(self._seconds.values())
            out = [
                "",
                f"{'stage':26s} {'seconds':>9s} {'%wall':>6s} {'calls':>8s} "
                f"{'s/call':>9s} {'per unit':>18s}",
                "-" * 84,
            ]
            for name, sec in rows:
                calls = self._calls[name]
                units = self._units[name]
                uname = self._unit_name.get(name, "")
                per_unit = ""
                if uname and units:
                    if uname == "bytes":
                        per_unit = f"{units / sec / 1e6:.1f} MB/s" if sec else ""
                    else:
                        per_unit = f"{sec / units * 1000:.3f} ms/{uname}"
                out.append(
                    f"{name:26s} {sec:9.2f} {100 * sec / total if total else 0:6.1f} "
                    f"{calls:8d} {sec / calls if calls else 0:9.3f} {per_unit:>18s}"
                )
            out += [
                "-" * 84,
                f"{'accounted':26s} {acct:9.2f} {100 * acct / total if total else 0:6.1f}",
                f"{'wall clock':26s} {total:9.2f} {100.0:6.1f}",
                f"{'UNACCOUNTED':26s} {total - acct:9.2f} "
                f"{100 * (total - acct) / total if total else 0:6.1f}",
                "",
            ]
            # Totals the cost model actually needs, spelled out.
            for name, uname in self._unit_name.items():
                if uname == "bytes":
                    out.append(f"  {name}: {self._units[name] / 1e6:,.1f} MB transferred")
                elif uname in ("pick", "window"):
                    out.append(f"  {name}: {int(self._units[name]):,} {uname}s")
            return "\n".join(out)


profile = Profile()


@contextmanager
def stage(name: str, unit: float = 0.0, unit_name: str = ""):
    """Time a stage. A no-op beyond a boolean check when profiling is off."""
    if not profile.enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        profile.record(name, time.perf_counter() - t0, unit, unit_name)
