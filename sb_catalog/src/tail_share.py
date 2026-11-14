"""
Hand a finished worker's cores to the workers still running ("tail widening").

The runner lays a node out as P worker processes with `base` intra-op threads
each, and splits the stations statically. Groups finish 0.3-1.3 s apart, and
which one lags changes from run to run, so after the first group ends its
`base` cores idle while its peers keep running on `base` threads. This module
lets the peers claim those cores.

State, shared by the runner's processes (one lock, one int array):

- `claim[i]`: extra intra-op threads worker i runs with now (beyond `base`).
  A worker raises its claim *before* it widens, and lowers it only *after* it
  has narrowed, both under the lock, so the recorded total never falls below
  what the processes actually run and never exceeds `vcpus`.
- `finished[i]`: worker i is done (or failed); its cores are free.
- `remaining[i]`: worker i's classify work left, in thousandths of a
  station-day (the priority the allocation uses).
- `target[i]` and an epoch (the finished count when the targets were
  computed): the allocation is recomputed only when a worker finishes, so the
  thread counts change a handful of times per run and never oscillate.

`allocate` is the pure policy: free = vcpus - base * live goes, one core at a
time, to the live worker with the most remaining work per thread it would
run (water-filling), at most `cap` extra each; a worker with no classify work
left gets nothing. A worker moves its claim towards its target only as far as
the cores its peers have actually released allow.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

#: QS_TAIL in pick_workload.py: "widen" lets live workers claim finished
#: workers' cores; "static" keeps every worker at its base thread count.
TAIL_MODES = ("widen", "static")
#: Extra intra-op threads one worker may claim.
MAX_EXTRA = 4
#: Shared array layout: per worker (claim, finished, remaining, target), then
#: one slot for the epoch.
FIELDS = 4
CLAIM, FINISHED, REMAINING, TARGET = range(FIELDS)


def allocate(vcpus: int, base: int, finished: Sequence[int], remaining: Sequence[int],
             cap: int = MAX_EXTRA) -> list[int]:
    """Extra threads per worker: water-fill the free cores by remaining work per thread.

    free = vcpus - base * live. Each free core goes to the live worker with
    the largest remaining / (base + extra so far), ties to the lower index,
    skipping workers at `cap` or with no remaining work. Finished workers get
    0. The result sums to at most max(0, free).
    """
    n = len(finished)
    live = [i for i in range(n) if not finished[i]]
    free = vcpus - base * len(live)
    extra = [0] * n
    for _ in range(max(0, free)):
        candidates = [i for i in live if extra[i] < cap and remaining[i] > 0]
        if not candidates:
            break
        best = max(candidates, key=lambda i: (remaining[i] / (base + extra[i]), -i))
        extra[best] += 1
    return extra


def array_size(workers: int) -> int:
    """Ints the shared array needs for `workers` workers."""
    return FIELDS * workers + 1


class TailShare:
    """One worker's view of the shared claim table.

    `lock` is a multiprocessing lock and `table` an unsynchronized shared int
    array of `array_size(workers)` (both created by the runner's parent and
    passed to every worker); every access holds `lock`.
    """

    def __init__(self, lock, table, index: int, workers: int, vcpus: int, base: int,
                 total_units: int, cap: int = MAX_EXTRA, t0: Optional[float] = None) -> None:
        self.lock, self.table = lock, table
        self.index, self.workers, self.vcpus, self.base, self.cap = (
            index, workers, vcpus, base, cap)
        #: Station-day-channels this worker expects to classify (the priority's scale).
        self.total_units = max(1, total_units)
        self.classified = 0
        #: Event times in the summary are relative to this (the worker's start).
        self.t0 = time.time() if t0 is None else t0
        #: (time, from threads, to threads, live workers, total threads in use) per change.
        self.events: list[tuple[float, int, int, int, int]] = []
        #: Largest recorded total of threads in use seen by this worker.
        self.max_total = 0
        self.last_batch_threads: Optional[int] = None
        self.updates = 0
        #: Peers finished as of the last update (set before `apply` runs).
        self.peers_finished = 0
        self._finished = False

    def _slot(self, i: int, field: int) -> int:
        return FIELDS * i + field

    def _get(self, i: int, field: int) -> int:
        return self.table[self._slot(i, field)]

    def _set(self, i: int, field: int, value: int) -> None:
        self.table[self._slot(i, field)] = value

    def _total_locked(self) -> tuple[int, int]:
        """(live workers, threads in use) as recorded; caller holds the lock."""
        live = sum(1 for i in range(self.workers) if not self._get(i, FINISHED))
        claimed = sum(self._get(i, CLAIM) for i in range(self.workers))
        return live, self.base * live + claimed

    def remaining(self, batch_fraction: float = 0.0) -> int:
        """Classify work left, in thousandths of a station-day, given the
        fraction of the current station-day's batches already run."""
        left = self.total_units - self.classified - batch_fraction
        return max(0, int(round(1000 * left)))

    def update(self, remaining: int, apply: Callable[[int], None]) -> int:
        """Publish `remaining`, move this worker's claim towards its target and
        call `apply(threads)` with the thread count to run (inside the lock, so
        a lower claim is recorded only after the narrowing). Returns it."""
        i = self.index
        with self.lock:
            self.updates += 1
            self._set(i, REMAINING, remaining)
            finished = [self._get(j, FINISHED) for j in range(self.workers)]
            epoch = sum(finished)
            if self.table[FIELDS * self.workers] != epoch:
                # A worker finished since the targets were computed: recompute.
                targets = allocate(self.vcpus, self.base, finished,
                                   [self._get(j, REMAINING) for j in range(self.workers)],
                                   self.cap)
                for j in range(self.workers):
                    self._set(j, TARGET, targets[j])
                self.table[FIELDS * self.workers] = epoch
            self.peers_finished = epoch
            old = self._get(i, CLAIM)
            live = self.workers - epoch
            others = sum(self._get(j, CLAIM) for j in range(self.workers) if j != i)
            room = self.vcpus - self.base * live - others
            new = max(0, min(self._get(i, TARGET), room))
            if new > old:
                self._set(i, CLAIM, new)       # record first, then widen
                apply(self.base + new)
            elif new < old:
                apply(self.base + new)         # narrow first, then release
                self._set(i, CLAIM, new)
            else:
                apply(self.base + new)
            live, total = self._total_locked()
            self.max_total = max(self.max_total, total)
            if new != old:
                self.events.append((time.time(), self.base + old, self.base + new, live, total))
            return self.base + new

    def threads(self) -> int:
        """The thread count this worker's claim allows now."""
        with self.lock:
            return self.base + self._get(self.index, CLAIM)

    def finish(self) -> None:
        """Mark this worker finished and release its claim (idempotent)."""
        if self._finished:
            return
        self._finished = True
        i = self.index
        with self.lock:
            old = self._get(i, CLAIM)
            self._set(i, CLAIM, 0)
            self._set(i, REMAINING, 0)
            self._set(i, FINISHED, 1)
            live, total = self._total_locked()
            self.max_total = max(self.max_total, total)
        self.events.append((time.time(), self.base + old, 0, live, total))

    def summary(self) -> str:
        """One line for the group log: every claim/release event, relative to start."""
        widened = sum(1 for _, a, b, _, _ in self.events if b > a)
        events = " ".join(f"{t - self.t0:+.3f}:{a}->{b}(live{lv},total{tot})"
                          for t, a, b, lv, tot in self.events) or "-"
        return (f"{widened} claims, {len(self.events)} events, {self.updates} updates, "
                f"max total {self.max_total}/{self.vcpus}, threads at last batch "
                f"{self.last_batch_threads}; events {events}")
