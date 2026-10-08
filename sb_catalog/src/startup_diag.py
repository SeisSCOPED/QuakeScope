"""Startup attribution for one process (pick_workload's "startup-diag" lines; round 17, H19).

Log-only: nothing here changes what a process imports, the order it does so, its threads or its
scheduling. It reads /proc and getrusage, and times the first import of a fixed set of modules
from a meta-path finder that only looks the module up with the finders behind it and wraps that
module's own loader; every other import sees one set lookup. Standard library only, and loaded
by path (as tail_share and parent_fetch), so the parent's sys.path stays as it is.

A usage snapshot is (wall epoch, process CPU s, major faults, minor faults, voluntary and
involuntary context switches), all for the whole process.
"""

from __future__ import annotations

import os
import resource
import sys
import threading
import time

#: Modules whose first import is timed (inclusive of whatever they import first), by full name.
#: The worker's import set: numpy, obspy core and FDSN, s3fs (with aiobotocore/botocore) and
#: aiohttp, pandas, pyarrow, torch, SeisBench's model package, the model and amplitude modules,
#: and the scipy/obspy.signal/matplotlib modules the amplitude passes may pull in. Packages whose
#: own __init__ costs nothing measurable (obspy.io.mseed/stationxml, botocore, aiobotocore,
#: seisbench) are left out: their cost is in the modules above that import them.
WATCH = (
    "numpy", "obspy", "obspy.clients.fdsn", "aiohttp", "s3fs", "pandas", "pyarrow", "torch",
    "seisbench.models", "scipy", "scipy.signal", "obspy.signal", "matplotlib",
    "sb_catalog.src.s3_helper", "sb_catalog.src.picker", "sb_catalog.src.classifier",
    "sb_catalog.src.lean_phasenet", "sb_catalog.src.amplitude_extractor",
    "sb_catalog.src.slim_signal", "sb_catalog.src.slim_evalresp",
)
#: Modules whose cached bytecode (importlib.util.cache_from_source(__file__)) is probed at the end.
BYTECODE_PROBE = ("torch", "numpy", "obspy", "pandas", "scipy")


def usage() -> tuple:
    """(wall, cpu, majflt, minflt, nvcsw, nivcsw) of this process, now."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return (time.time(), ru.ru_utime + ru.ru_stime, ru.ru_majflt, ru.ru_minflt, ru.ru_nvcsw,
            ru.ru_nivcsw)


def loadavg() -> str:
    """The 1/5/15-minute load averages, as /proc/loadavg has them."""
    try:
        with open("/proc/loadavg") as f:
            return "/".join(f.read().split()[:3])
    except OSError:
        return "-"


def process_start() -> float:
    """This process's start as a wall epoch: /proc/self/stat starttime (clock ticks after boot)
    against CLOCK_BOOTTIME read now, so btime's whole-second rounding does not enter; ~10 ms
    resolution (CLK_TCK). NaN when /proc is unreadable."""
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().rpartition(")")[2].split()
        ticks = int(fields[19])                       # field 22, counting pid and comm
        now_wall, now_boot = time.time(), time.clock_gettime(time.CLOCK_BOOTTIME)
        return now_wall - (now_boot - ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError, AttributeError):
        return float("nan")


def fmt_usage(snap: tuple, t0: float) -> str:
    """`+wall/cpu/majflt/minflt/nvcsw/nivcsw`, wall in seconds from `t0`."""
    return (f"{snap[0] - t0:+.3f}/{snap[1]:.3f}/{snap[2]}/{snap[3]}/{snap[4]}/{snap[5]}")


def bytecode_probe() -> str:
    """dont_write_bytecode, the interpreter and prefix, and per probed module whether its cached
    bytecode file exists (1/0; '-' not loaded, '?' no source file)."""
    import importlib.util

    parts = [f"dwb={int(sys.flags.dont_write_bytecode)}", f"exe={sys.executable}",
             f"prefix={sys.prefix}"]
    for name in BYTECODE_PROBE:
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None) if module is not None else None
        if module is None:
            state = "-"
        elif not path or not path.endswith(".py"):
            state = "?"
        else:
            try:
                state = str(int(os.path.exists(importlib.util.cache_from_source(path))))
            except (NotImplementedError, ValueError):
                state = "?"
        parts.append(f"{name}={state}")
    return " ".join(parts)


class ImportClock:
    """First-import wall, importing-thread CPU and faults per watched module, inclusive.

    Installed first on sys.meta_path. For a watched name it asks the finders behind it for the
    spec (the spec the import system would have found) and times that module's exec_module by
    shadowing it on the loader instance for the one call; it returns None for every other name,
    so the import system carries on exactly as without it.
    """

    def __init__(self, t0: float, watch: tuple = WATCH) -> None:
        self.t0 = t0
        self.watch = frozenset(watch)
        #: name -> (start offset from t0, wall s, thread CPU s, thread majflt, thread minflt,
        #: importing thread's name)
        self.records: dict[str, tuple] = {}
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def install(self) -> "ImportClock":
        sys.meta_path.insert(0, self)
        return self

    def uninstall(self) -> None:
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass

    def find_spec(self, name, path, target=None):
        if name not in self.watch:
            return None
        with self._lock:
            if name in self.records or name in self._pending:
                return None
            self._pending.add(name)
        spec = None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(name, path, target)
            if spec is not None:
                break
        loader = getattr(spec, "loader", None)
        exec_module = getattr(loader, "exec_module", None)
        if exec_module is None:
            return spec
        clock = self

        def timed(module):
            start = time.time()
            cpu = time.thread_time()
            ru = resource.getrusage(resource.RUSAGE_THREAD)
            try:
                exec_module(module)
            finally:
                end = time.time()
                ru2 = resource.getrusage(resource.RUSAGE_THREAD)
                clock.records[name] = (start - clock.t0, end - start, time.thread_time() - cpu,
                                       ru2.ru_majflt - ru.ru_majflt, ru2.ru_minflt - ru.ru_minflt,
                                       threading.current_thread().name)
                try:
                    del loader.exec_module
                except AttributeError:
                    pass

        try:
            loader.exec_module = timed
        except (AttributeError, TypeError):
            pass                                      # e.g. a slotted loader: left untimed
        return spec

    def line(self) -> str:
        """`name=±start/wall/cpu/majflt/minflt` per timed module, in import order, with
        `@thread` appended when a thread other than the main thread imported it; the
        sb_catalog.src. prefix is shortened to sb."""
        items = sorted(self.records.items(), key=lambda kv: kv[1][0])
        return " ".join(
            f"{name.replace('sb_catalog.src.', 'sb.')}={s:+.3f}/{w:.3f}/{c:.3f}/{mj}/{mn}"
            f"{'' if thread == 'MainThread' else '@' + thread}"
            for name, (s, w, c, mj, mn, thread) in items) or "-"
