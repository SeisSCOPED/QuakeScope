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

Log-only, in every mode: at the worker's end, the CPU seconds of every task
(/proc/self/task/*/stat), split into the classify thread, its OpenMP pool,
the BLAS threads (the unnamed native tasks present right after the imports), the
amplitude pool, the event loop, other tasks and those that already exited.
"""

from __future__ import annotations

import os
import resource
import threading
import time
from typing import Optional, Sequence

#: QS_PIN in pick_workload.py.
PIN_MODES = ("off", "classify", "all")
#: The classify thread's name (/proc comm): the libgomp threads it spawns
#: inherit it, which is how its OpenMP pool is told from every other task.
CLASSIFY_COMM = "qs-classify"
#: The per-thread CPU split's classes, in log order.
CPU_CLASSES = ("classify", "omp", "blas", "amp", "loop", "other", "exited")
CLK_TCK = os.sysconf("SC_CLK_TCK")


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


def cpu_seconds(tid: int) -> Optional[float]:
    """A task's user + system CPU seconds (/proc/.../stat), or None once it has exited."""
    text = _read(tid, "stat")
    if text is None:
        return None
    fields = text.rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / CLK_TCK


def blas_tasks() -> list[int]:
    """This process's native tasks (not Python threads) that kept the main thread's name:
    OpenBLAS's threads after numpy loads; named ones (pyarrow's jemalloc_bg_thd) are not."""
    python = {t.native_id for t in threading.enumerate()}
    name = comm(threading.main_thread().native_id)
    return [t for t in tasks() if t not in python and comm(t) == name]


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
        #: Native tasks right after the imports (the BLAS threads), set by the runner.
        self.blas_tids: set[int] = set()
        #: {class: (CPU seconds, tasks)} at the worker's end (sample_cpu).
        self.cpu_split: Optional[dict[str, tuple[float, int]]] = None

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

    def sample_cpu(self) -> None:
        """Before the worker's pools shut down: every task's CPU seconds, by class."""
        names = {t.native_id: t.name for t in threading.enumerate()}
        main = threading.main_thread().native_id
        split = {k: [0.0, 0] for k in CPU_CLASSES}
        live = 0.0
        for t in tasks():
            cpu = cpu_seconds(t)
            if cpu is None:
                continue
            if t == self.classify_tid:
                k = "classify"
            elif t in names:
                k = "loop" if t == main else "amp" if names[t].startswith("amp") else "other"
            elif comm(t) == CLASSIFY_COMM:
                k = "omp"
            else:
                k = "blas" if t in self.blas_tids else "other"
            split[k][0] += cpu
            split[k][1] += 1
            live += cpu
        ru = resource.getrusage(resource.RUSAGE_SELF)
        split["exited"][0] = max(0.0, ru.ru_utime + ru.ru_stime - live)
        self.cpu_split = {k: (round(c, 2), n) for k, (c, n) in split.items()}

    def cpu_line(self, forward_s: float, pool_forward_s: float) -> str:
        """The per-thread CPU split and the OpenMP spin-waste estimate: pool CPU
        minus the forward wall its threads could have spent working."""
        if self.cpu_split is None:
            return "thread cpu - (not sampled)"
        split = " ".join(f"{k}={c:.2f}/{n}" for k, (c, n) in self.cpu_split.items())
        omp = self.cpu_split["omp"][0]
        return (f"thread cpu s/tasks {split}; forward wall {forward_s:.2f} s pool-forward "
                f"{pool_forward_s:.2f} s; spin waste omp {omp - pool_forward_s:.2f} s "
                f"blas {self.cpu_split['blas'][0]:.2f} s")

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
