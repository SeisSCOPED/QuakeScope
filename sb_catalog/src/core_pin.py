"""
Pin each worker's forward to its own core pair ("core pinning", QS_PIN).

The runner lays a node out as P worker processes with `base` intra-op
threads each. Unpinned, a worker's classify thread and its OpenMP pool float
over every inherited core among the peers' amplitude pools and event-loop
threads, so the scheduler migrates them and they lose their L2-resident
weights and activations. With pinning, worker i's classify thread (and the
OpenMP pool it spawns) runs on cores [base*i, base*i + base) of the sorted
inherited affinity set.

Modes:

- "off": nothing is pinned; the masks and migrations are still recorded.
- "classify": the classify thread pins itself before its first parallel
  region, so the libgomp threads it spawns inherit the pair; the amplitude
  pools and the event loop keep the inherited set.
- "all": every task of the worker is pinned before torch is imported, so
  every thread the worker starts inherits the pair.

The pair is never wider than the inherited set; with fewer than
base * workers inherited cores it is the inherited set itself. The pinned
window ends when a peer finishes or this worker first raises a tail claim
(tail_share): every task of the worker is then reset to the inherited set,
before the intra-op thread count widens, and the window's migrations and
per-station-day classify times are kept for the group log.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Sequence

#: QS_PIN in pick_workload.py.
PIN_MODES = ("off", "classify", "all")
#: The classify thread's name (/proc comm): the libgomp threads it spawns
#: inherit it, which is how its OpenMP pool is told from every other task.
CLASSIFY_COMM = "qs-classify"


def worker_cores(inherited: Sequence[int], index: int, workers: int, per: int) -> list[int]:
    """Worker `index`'s cores: `per` consecutive cores of the sorted inherited
    set, or the whole set when it has fewer than per * workers cores."""
    cores = sorted(inherited)
    if per < 1 or len(cores) < per * workers or not 0 <= index < workers:
        return cores
    return cores[per * index:per * index + per]


def cpu_list(cores) -> str:
    """Kernel cpu-list notation (0-3,6) for a set of cores."""
    cores = sorted(cores)
    if not cores:
        return "-"
    runs, start, prev = [], cores[0], cores[0]
    for c in cores[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        runs.append(f"{start}-{prev}" if prev > start else f"{start}")
        if c is not None:
            start = prev = c
    return ",".join(runs)


def tasks() -> list[int]:
    """This process's thread ids."""
    try:
        return sorted(int(t) for t in os.listdir("/proc/self/task"))
    except OSError:
        return []


def _read(tid: int, name: str) -> Optional[str]:
    try:
        with open(f"/proc/self/task/{tid}/{name}") as f:
            return f.read()
    except OSError:
        return None


def comm(tid: int) -> Optional[str]:
    """A task's name, or None once it has exited."""
    text = _read(tid, "comm")
    return None if text is None else text.strip()


def mask(tid: int) -> Optional[str]:
    """A task's Cpus_allowed_list from /proc, or None once it has exited."""
    for line in (_read(tid, "status") or "").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return None


def migrations(tid: int) -> Optional[int]:
    """A task's se.nr_migrations (/proc/.../sched), or None without it."""
    for line in (_read(tid, "sched") or "").splitlines():
        if line.startswith("se.nr_migrations"):
            return int(line.split(":", 1)[1])
    return None


def set_mask(tid: int, cores) -> bool:
    """sched_setaffinity on one task; False if it has exited."""
    try:
        os.sched_setaffinity(tid, cores)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class CorePin:
    """One worker's pinning: its pair, the pinned window and its log line."""

    def __init__(self, mode: str, index: int, workers: int, per: int,
                 t0: Optional[float] = None) -> None:
        if mode not in PIN_MODES:
            raise ValueError(f"pin mode {mode!r} is not one of {PIN_MODES}")
        self.mode, self.index = mode, index
        #: The inherited set: the widest mask any task of this worker may get.
        self.inherited = sorted(os.sched_getaffinity(0))
        self.pair = worker_cores(self.inherited, index, workers, per)
        self.t0 = time.time() if t0 is None else t0
        self.classify_tid: Optional[int] = None
        #: (tid, mask) of the classify thread and its OpenMP threads after the first classify.
        self.first_masks: list[tuple[int, str]] = []
        self.omp_tids: list[int] = []
        #: OpenMP threads found outside the pair after the first classify (then pinned per tid).
        self.fixed = 0
        self.process_pinned = 0
        #: {tid: nr_migrations} at the first classify's end, and the window's deltas.
        self._mig0: dict[int, Optional[int]] = {}
        self.mig_classify: Optional[int] = None
        self.mig_omp: Optional[int] = None
        self.window: Optional[tuple[float, float]] = None
        #: (time, cause, tasks reset, masks after) of the window's end.
        self.reset_event: Optional[tuple[float, str, int, str]] = None
        #: (end time, seconds) of every station-day's classify.
        self.sd_times: list[tuple[float, float]] = []
        self._lock = threading.Lock()

    @property
    def window_open(self) -> bool:
        return self.reset_event is None

    def pin_process(self) -> None:
        """"all": pin every task of this process to the pair (before torch is imported)."""
        if self.mode == "all":
            self.process_pinned = sum(set_mask(t, self.pair) for t in tasks())

    def on_classify_thread(self) -> None:
        """In the classify thread, before its first parallel region: name it and
        ("classify") pin it, so the OpenMP threads it spawns inherit both."""
        if self.classify_tid is not None:
            return
        self.classify_tid = threading.get_native_id()
        try:
            with open(f"/proc/self/task/{self.classify_tid}/comm", "w") as f:
                f.write(CLASSIFY_COMM)
        except OSError:
            pass
        if self.mode == "classify":
            os.sched_setaffinity(0, self.pair)

    def after_first_classify(self) -> None:
        """In the classify thread, after its first station-day: find its OpenMP
        threads, pin any outside the pair, record the masks and the migrations baseline."""
        if self.first_masks or self.classify_tid is None:
            return
        python_tids = {t.native_id for t in threading.enumerate()}
        self.omp_tids = [t for t in tasks() if t != self.classify_tid and t not in python_tids
                         and comm(t) == CLASSIFY_COMM]
        if self.mode != "off" and self.window_open:
            want = cpu_list(self.pair)
            for t in self.omp_tids:
                if mask(t) != want and set_mask(t, self.pair):
                    self.fixed += 1
        self.first_masks = [(t, mask(t) or "?") for t in [self.classify_tid] + self.omp_tids]
        self._mig0 = {t: migrations(t) for t in [self.classify_tid] + self.omp_tids}
        self.window = (time.time(), time.time())

    def record_sd(self, seconds: float) -> None:
        """One station-day's classify time (classify thread)."""
        self.sd_times.append((time.time(), seconds))

    def _window_migrations(self) -> None:
        def delta(t):
            now, then = migrations(t), self._mig0.get(t)
            return None if now is None or then is None else now - then

        if self.classify_tid is None or not self._mig0:
            return
        self.mig_classify = delta(self.classify_tid)
        omp = [delta(t) for t in self.omp_tids]
        self.mig_omp = None if any(d is None for d in omp) else sum(omp)

    def end_window(self, cause: str) -> None:
        """A peer finished or this worker claims cores: close the window and (pinned
        modes) reset every task to the inherited set, before the forward widens."""
        with self._lock:
            if not self.window_open:
                return
            now = time.time()
            self._window_migrations()
            if self.window is not None:
                self.window = (self.window[0], now)
            reset = 0
            if self.mode != "off":
                reset = sum(set_mask(t, self.inherited) for t in tasks())
            masks = sorted({mask(t) or "?" for t in tasks()})
            self.reset_event = (now, cause, reset, "/".join(masks))

    def finish(self) -> None:
        """At the worker's end: close a window no peer or claim closed (no reset)."""
        if self.window_open and self.window is not None:
            self.window = (self.window[0], time.time())
            self._window_migrations()

    def loaded_sds(self) -> list[float]:
        """Classify seconds of the station-days that ended inside the pinned window,
        after the first (which carries the graph's first-call setup)."""
        end = self.reset_event[0] if self.reset_event is not None else float("inf")
        return [s for t, s in self.sd_times[1:] if t <= end]

    def summary(self) -> str:
        """One line for the group log."""
        loaded = self.loaded_sds()
        first = " ".join(f"{t}:{m}" for t, m in self.first_masks) or "-"
        win = (f"{self.window[0] - self.t0:+.3f}..{self.window[1] - self.t0:+.3f}"
               if self.window is not None else "-")
        return (
            f"pin={self.mode} inherited={cpu_list(self.inherited)} pair={cpu_list(self.pair)} "
            f"process_pinned={self.process_pinned} classify_tid={self.classify_tid} "
            f"omp_threads={len(self.omp_tids)} fixed={self.fixed}; masks after first classify {first}; "
            f"window {win} migrations classify={self.mig_classify} omp={self.mig_omp}; "
            f"loaded classify {len(loaded)} SDs "
            f"{(sum(loaded) / len(loaded)) if loaded else float('nan'):.3f} s/SD; "
            f"all SDs {' '.join(f'{s:.3f}' for _, s in self.sd_times) or '-'}"
        )

    def reset_line(self) -> str:
        """The window's end for the group log: 'pin reset' (pinned) or 'pin window end' (off)."""
        if self.reset_event is None:
            return "pin window end - (no peer finished and no claim before this worker ended)"
        t, cause, reset, masks = self.reset_event
        what = "pin reset" if self.mode != "off" else "pin window end"
        return (f"{what} at {t:.3f} ({t - self.t0:+.3f}) cause={cause} tasks_reset={reset} "
                f"masks after {masks}")


def schedstat(tid: int) -> Optional[tuple[int, int]]:
    """A task's (on-CPU ns, runqueue-wait ns) from /proc schedstat, or None once it has exited."""
    text = _read(tid, "schedstat")
    if not text:
        return None
    run, wait = text.split()[:2]
    return int(run), int(wait)


#: Native libraries whose thread-count getter openblas_threads() asks, by symbol.
OPENBLAS_GETTERS = ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
                    "openblas_get_num_threads64_", "openblas_get_num_threads")


def openblas_threads() -> Optional[int]:
    """The thread count in effect in the OpenBLAS this process loaded (numpy's), or None
    when none is loaded or it exports no known getter."""
    try:
        with open("/proc/self/maps") as f:
            paths = {line.split()[-1] for line in f if "openblas" in line.rsplit("/", 1)[-1]}
    except OSError:
        return None
    import ctypes

    for path in sorted(paths):
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for name in OPENBLAS_GETTERS:
            getter = getattr(lib, name, None)
            if getter is not None:
                return int(getter())
    return None


class ThreadCpu:
    """One worker's CPU per thread, summed by role, for the QS_BLAS group line.

    Live tasks are read from /proc schedstat at the end; Python threads that
    exit before (pool, executor and fetch threads) record their own schedstat
    as they leave, through a wrapper around threading.Thread.run installed at
    worker entry. The classify thread's exit also records its OpenMP pool
    (tasks named CLASSIFY_COMM), which libgomp ends with it. The OpenBLAS
    pool's threads are the unnamed native (non-Python) tasks present once
    numpy is loaded and before any forward (mark_blas); named native tasks
    (jemalloc_bg_thd) are other_native. "residual" is the process's CPU no
    thread accounts for.
    """

    ROLES = ("classify", "omp", "blas", "amp_pool", "asyncio", "loader", "import_prefetch", "main",
             "other_native", "other")
    #: Python thread name prefixes per role (first match wins; import_prefetch's readers are qs-prefetch-<k>).
    NAME_ROLES = (("amp-vel", "amp_pool"), ("asyncio_", "asyncio"), ("qs-early", "loader"),
                  ("qs-inventory", "loader"), ("qs-prefetch-local", "loader"), ("qs-prefetch-get", "loader"),
                  ("qs-firstsd", "loader"))

    def __init__(self) -> None:
        #: (tid, Python thread name or None, schedstat, role or None) of tasks recorded as they ended.
        self.exited: list[tuple[int, Optional[str], Optional[tuple[int, int]], Optional[str]]] = []
        self.blas_tids: list[int] = []
        self.blas_threads: Optional[int] = None

    def install(self) -> "ThreadCpu":
        """Wrap threading.Thread.run so every Python thread records its schedstat on exit."""
        exited, orig = self.exited, threading.Thread.run

        def run(thread: threading.Thread) -> None:
            try:
                orig(thread)
            finally:
                me = threading.get_native_id()
                if comm(me) == CLASSIFY_COMM:
                    exited.extend((t, None, schedstat(t), "omp") for t in tasks()
                                  if t != me and comm(t) == CLASSIFY_COMM)
                exited.append((thread.native_id, thread.name, schedstat(me), None))

        threading.Thread.run = run
        return self

    def mark_blas(self) -> None:
        """Once numpy is loaded, before any forward: the unnamed native tasks are OpenBLAS's pool."""
        python = {t.native_id for t in threading.enumerate()}
        main = comm(threading.main_thread().native_id)
        self.blas_tids = [t for t in tasks() if t not in python and comm(t) == main]
        self.blas_threads = openblas_threads()

    def _role(self, tid: int, name: Optional[str], classify_tid: Optional[int], main_tid: int) -> str:
        if tid == classify_tid:
            return "classify"
        if tid in self.blas_tids:
            return "blas"
        if tid == main_tid:
            return "main"
        if name is None:
            return "omp" if comm(tid) == CLASSIFY_COMM else "other_native"
        if name.startswith("qs-prefetch-") and name[len("qs-prefetch-"):].isdigit():
            return "import_prefetch"
        if name == "qs-prefetch":
            return "loader"
        for prefix, role in self.NAME_ROLES:
            if name.startswith(prefix):
                return role
        return "other"

    def summary(self, classify_tid: Optional[int]) -> str:
        """CPU and runqueue wait seconds and thread counts per role, the process total and residual."""
        main_tid = threading.main_thread().native_id
        names = {t.native_id: t.name for t in threading.enumerate()}
        rows = [(tid, names.get(tid), schedstat(tid), None) for tid in tasks()]
        seen = {tid for tid, _, _, _ in rows}
        for row in reversed(self.exited):        # a task recorded twice: its last record
            if row[0] not in seen:
                seen.add(row[0])
                rows.append(row)
        total = time.process_time()
        cpu = dict.fromkeys(self.ROLES, 0.0)
        wait = dict.fromkeys(self.ROLES, 0.0)
        count = dict.fromkeys(self.ROLES, 0)
        for tid, name, stat, role in rows:
            if stat is None:
                continue
            role = role or self._role(tid, name, classify_tid, main_tid)
            cpu[role] += stat[0] / 1e9
            wait[role] += stat[1] / 1e9
            count[role] += 1
        residual = total - sum(cpu.values())
        return (
            f"openblas_threads={self.blas_threads} blas_tids={','.join(map(str, self.blas_tids)) or '-'}; "
            f"cpu_s total={total:.3f} " + " ".join(f"{r}={cpu[r]:.3f}" for r in self.ROLES)
            + f" residual={residual:.3f}; wait_s " + " ".join(f"{r}={wait[r]:.3f}" for r in self.ROLES)
            + "; threads " + " ".join(f"{r}={count[r]}" for r in self.ROLES)
            + f" exited={len(self.exited)}"
        )
