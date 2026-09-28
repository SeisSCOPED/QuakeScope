"""Parent-process S3 prefetch for the picking runner (QS_PREFETCH=on).

Each worker process used to LIST every (net, day) and GET its own objects in
the process that runs the model, where that I/O shares the GIL and the
process's two cores with classify. With the prefetch, the runner's parent -
which otherwise only waits for its workers - LISTs each (net, day) once and
GETs every object the workers will read, from a bounded thread pool, in the
order the workers read them (round-robin across groups, so every group's next
station-day lands first). Each object is published as a local file under the
run's TMPDIR: written, then renamed, so a file that exists is complete. The
worker parses the local copy with the same obspy call and deletes it
(`S3DataSource._read_prefetched`).

Nothing outlives the run: the directory is the run's own and is removed when
the workers exit. What the parent cannot deliver - an archive that needs
credentials, a failed LIST or GET, an object over the size limit - is marked
failed (or simply never published), and the worker then reads it directly,
exactly as it does without the prefetch.

Memory stays bounded: each group may have at most PREFETCH_WINDOW published
objects it has not read yet; the parent waits on the group's semaphore before
fetching more.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

#: Concurrent LIST/GET requests in the parent.
PREFETCH_THREADS = int(os.environ.get("QS_PREFETCH_THREADS", "12"))
#: Published-but-unread objects allowed per worker group (3 per SCEDC station-day).
PREFETCH_WINDOW = int(os.environ.get("QS_PREFETCH_WINDOW", "12"))
#: Longest a worker waits for the parent before reading an object itself.
PREFETCH_WAIT = float(os.environ.get("QS_PREFETCH_WAIT", "60"))
POLL_SECONDS = 0.02
FAILED = ".failed"

logger = logging.getLogger("prefetch")


def object_path(root: str, uri: str) -> str:
    return os.path.join(root, "obj", uri.replace("/", "%"))


def listing_path(root: str, net: str, day) -> str:
    return os.path.join(root, "list", f"{net}.{day:%Y.%j}.json")


def done_path(root: str) -> str:
    return os.path.join(root, "DONE")


def _publish(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _mark_failed(path: str) -> None:
    open(path + FAILED, "wb").close()


class ParentPrefetcher:
    """LIST once, GET ahead, for the worker groups of one run."""

    def __init__(self, root: str, planners: list, slots: list | None = None,
                 alive=None) -> None:
        """`planners`: one `S3DataSource.planner` per worker group, in group
        order; `slots`: one semaphore per group (released by its worker per
        object read); `alive(g)`: whether group g's worker still runs."""
        self.root = root
        self.planners = planners
        self.slots = slots
        self.alive = alive or (lambda _g: True)
        self.helper = planners[0].s3helper
        self._lock = threading.Lock()
        self.stats = {"lists": 0, "list_failed": 0, "gets": 0, "get_failed": 0,
                      "bytes": 0, "gets_per_group": [0] * len(planners),
                      "slot_wait_s": 0.0, "slot_timeouts": 0,
                      "last_get_s": 0.0, "elapsed_s": 0.0}
        os.makedirs(os.path.join(root, "obj"), exist_ok=True)
        os.makedirs(os.path.join(root, "list"), exist_ok=True)

    def _prefetchable(self, net: str) -> bool:
        """Anonymous archives only: a restricted EarthScope credential and its
        refusal handling stay with the worker that reads."""
        from .s3_helper import EarthScopeS3ObjectHelper

        dc = self.helper.get_data_center(net)
        if dc == "earthscope":
            return EarthScopeS3ObjectHelper.is_open_data(net)
        return dc in ("scedc", "ncedc", "geonet")

    def _list(self, net: str, day) -> list | None:
        path = listing_path(self.root, net, day)
        if not self._prefetchable(net):
            _mark_failed(path)
            return None
        try:
            prefix = self.helper.get_prefix(net, day.strftime("%Y"), day.strftime("%j"))
            fs = self.helper.get_filesystem(net, day.year)
            listing = self.helper.list_day(net, fs, prefix)
            _publish(path, json.dumps(listing).encode())
        except Exception as exc:
            # The worker LISTs for itself and handles it as it always has.
            logger.warning(f"prefetch LIST {net} {day:%Y.%j}: {type(exc).__name__}: {exc}")
            _mark_failed(path)
            with self._lock:
                self.stats["list_failed"] += 1
            return None
        with self._lock:
            self.stats["lists"] += 1
        return listing

    def _get(self, g: int, uri: str, net: str, year: int) -> None:
        path = object_path(self.root, uri)
        try:
            fs = self.helper.get_filesystem(net, year)
            # Served from the LIST's directory cache, as in the worker.
            size = fs.info(uri)["size"]
            limit = self.planners[g].limit_mb
            if limit is not None and size / 1024**2 > limit:
                raise ValueError(f"{size / 1024**2:.1f} MB is over the {limit} MB limit")
            raw = fs.read_bytes(uri)
            _publish(path, raw)
        except Exception as exc:
            logger.warning(f"prefetch GET {uri}: {type(exc).__name__}: {exc}")
            _mark_failed(path)
            with self._lock:
                self.stats["get_failed"] += 1
            return
        with self._lock:
            self.stats["gets"] += 1
            self.stats["bytes"] += len(raw)
            self.stats["gets_per_group"][g] += 1
            self.stats["last_get_s"] = time.perf_counter() - self._t0

    def _take_slot(self, g: int) -> None:
        if self.slots is None:
            return
        t0 = time.perf_counter()
        while not self.slots[g].acquire(timeout=1.0):
            if not self.alive(g) or time.perf_counter() - t0 > PREFETCH_WAIT:
                with self._lock:
                    self.stats["slot_timeouts"] += 1
                break
        self.stats["slot_wait_s"] += time.perf_counter() - t0

    def run(self) -> dict:
        """Fetch everything the groups will read; always ends by publishing DONE."""
        import numpy as np

        self._t0 = time.perf_counter()
        first = self.planners[0]
        # The same day sequence as S3DataSource.load_waveforms.
        days = [d.astype(datetime.datetime) for d in
                np.arange(first.start, first.end, datetime.timedelta(days=1))]
        nets = sorted({n for p in self.planners for n in p.networks})
        try:
            with ThreadPoolExecutor(PREFETCH_THREADS, thread_name_prefix="prefetch") as pool:
                lists = {(net, day): pool.submit(self._list, net, day)
                         for day in days for net in nets}
                gets = []
                for day in days:
                    avail = {net: lists[(net, day)].result() for net in nets}
                    # Consumption order: each group reads its stations in order
                    # within a day; interleave the groups so all advance together.
                    for i in range(max(len(p.stations) for p in self.planners)):
                        for g, planner in enumerate(self.planners):
                            if i >= len(planner.stations):
                                continue
                            station = planner.stations[i]
                            net = station.split(".")[0]
                            if avail[net] is None:
                                continue
                            for uri, _ in planner.plan_station_day(station, day, avail):
                                self._take_slot(g)
                                gets.append(pool.submit(self._get, g, uri, net, day.year))
                for f in gets:
                    f.result()
        finally:
            _publish(done_path(self.root), b"")
            self.stats["elapsed_s"] = time.perf_counter() - self._t0
        return self.stats
