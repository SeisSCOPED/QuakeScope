"""Seed QuakeScope picking runner.

Picks every station-day of a workload with QuakeScope's production v3 code
path (``S3DataSource`` -> ``S3MongoSBBridge`` -> ``ParquetPickWriter``), laid
out the way a production Batch task runs it: ``procs`` independent worker
processes, each with ``threads`` BLAS/OpenMP threads (fleet.json: 4 x 2 on the
8 vCPU / 16 GB Fargate shape; QS_LAYOUT=8x1 selects one station per 1-thread worker).

The candidate contract (arguments, output, rules) is in the task's
OBJECTIVE.md. The evaluator keeps a pristine copy of this file to build the
reference picks.

Heavy imports (pandas, torch via QuakeScope) happen inside the spawned workers,
after the thread-count variables are set: a spawned child re-imports this module,
so a module-level numpy import would size BLAS before ``pick_group`` runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

QUAKESCOPE_ROOT = Path(__file__).resolve().parent / "quakescope"

#: Worker layout, named for the 8 vCPU shape: BLAS/OpenMP threads per worker process.
#: The process count is vcpus // threads. "4x2" is the fleet.json production layout;
#: "8x1" measured no faster (1-thread forward throughput drops under 8-core contention).
LAYOUT_THREADS = {"4x2": 2, "8x1": 1}
LAYOUT = os.environ.get("QS_LAYOUT", "4x2")
PICK_QUEUE_SIZE = 5
#: PhaseNet forward-pass precision (sb_catalog.src.picker.INFER_PRECISIONS):
#: bf16 autocast on AMX by default; QS_INFER_PRECISION=fp32 restores the seed path.
INFER_PRECISION = os.environ.get("QS_INFER_PRECISION", "bf16")
#: Amplitude passes (sb_catalog.src.amplitude_extractor.AMP_IMPLS): "fast" reuses
#: the per-station-day response spectra; QS_AMP_IMPL=obspy restores the seed path.
AMP_IMPL = os.environ.get("QS_AMP_IMPL", "fast")
#: PhaseNet forward implementation (sb_catalog.src.picker.CLASSIFY_IMPLS): "fast" runs the
#: BN-folded channels-last graph at 64 windows per batch; QS_CLASSIFY_IMPL=seisbench restores
#: SeisBench's own forward at 256.
CLASSIFY_IMPL = os.environ.get("QS_CLASSIFY_IMPL", "fast")
#: Waveform I/O (sb_catalog.src.prefetch): "on" has this parent process LIST each
#: (net, day) once and GET every object ahead of the workers into the run's TMPDIR,
#: so the workers only parse local copies; QS_PREFETCH=off restores the in-worker S3 path.
PREFETCH = os.environ.get("QS_PREFETCH", "on")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

logger = logging.getLogger("pick_workload")


class StationTable:
    """The four database methods the v3 picking path needs (cf. worker.S3StateAdapter)."""

    def __init__(self, stations: list[dict]) -> None:
        """Build an in-memory station table from the workload's station records."""
        import pandas as pd

        self._stations = pd.DataFrame(stations)
        self.db_uri = "local"
        self.database = None

    def get_station_metadata(self, stations: list[str], _key: dict | None = None) -> pd.DataFrame:
        """Return the metadata rows for the given station ids."""
        frame = self._stations
        return frame[frame["id"].isin(stations)].reset_index(drop=True)

    def get_stations(self, _extent: object = None, _network: object = None) -> pd.DataFrame:
        """Return every station in the workload."""
        return self._stations

    def get_picks_record(
        self, _station: object, _day: object, _channel: object, _key: dict | None = None
    ) -> None:
        """Report no prior checkpointed pick; the seed runner never resumes."""
        return

    def write_run_data(self, **_kwargs: object) -> str:
        """Return a fresh run id; the seed runner does not persist run metadata."""
        return str(uuid.uuid4())

    def insert_many_ignore_duplicates(self, collection: str, records: list[dict]) -> None:
        """Discard picks/classifier rows; the seed runner's output is the Parquet files."""


def stage_weights(cache_root: Path) -> None:
    """Expose the committed PhaseNet weights as a SeisBench cache (as the Dockerfile does)."""
    source = QUAKESCOPE_ROOT / "sb_catalog" / "models" / "v3" / "phasenet"
    target = cache_root / "models" / "v3" / "phasenet"
    target.mkdir(parents=True, exist_ok=True)
    for weight_file in source.glob("*.v*"):
        link = target / weight_file.name
        if not link.exists():
            link.symlink_to(weight_file)


@dataclass(frozen=True)
class PickJobConfig:
    """Picking parameters shared by every worker group in one run."""

    workload: dict
    output_dir: str
    threads: int
    data_queue_size: int
    prefetch_dir: str | None = None


def resolve_layout(vcpus: int, stations: list[dict], layout: str = LAYOUT) -> tuple[int, list[list[dict]], int]:
    """Return (threads per proc, station groups, data queue size) for a layout on ``vcpus``."""
    if layout not in LAYOUT_THREADS:
        raise ValueError(f"unknown QS_LAYOUT {layout!r}; expected one of {sorted(LAYOUT_THREADS)}")
    threads = min(LAYOUT_THREADS[layout], vcpus)
    procs = max(1, vcpus // threads)
    groups = [stations[i::procs] for i in range(procs) if stations[i::procs]]
    # worker._resolve_queue_size: a node-wide budget of 8 decoded station-days.
    data_queue_size = max(1, min(5, 8 // len(groups)))
    return threads, groups, data_queue_size


def pick_group(index: int, stations: list[dict], config: PickJobConfig,
               prefetch_slots: object = None) -> None:
    """Run one production worker loop over a subset of the workload's stations."""
    started = time.time()
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(config.threads)
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    sys.path.insert(0, str(QUAKESCOPE_ROOT))
    logging.basicConfig(
        level=logging.WARNING, format=f"[group {index}] %(levelname)s %(name)s: %(message)s"
    )

    from sb_catalog.src.picker import S3MongoSBBridge
    from sb_catalog.src.profiling import profile
    from sb_catalog.src.s3_helper import S3DataSource
    from sb_catalog.src.utils import parse_year_day

    if os.environ.get("QS_PROFILE") == "1":
        profile.enable()

    workload = config.workload
    db = StationTable(stations)
    source = S3DataSource(
        stations=",".join(s["id"] for s in stations),
        start=parse_year_day(workload["start"]),
        end=parse_year_day(workload["end"]),
        components=workload["components"],
        db=db,
        prefetch_dir=config.prefetch_dir,
        prefetch_slots=prefetch_slots,
    )
    bridge = S3MongoSBBridge(
        s3=source,
        db=db,
        model=workload["model"],
        weight=workload["weight"],
        p_threshold=workload["p_threshold"],
        s_threshold=workload["s_threshold"],
        data_queue_size=config.data_queue_size,
        pick_queue_size=PICK_QUEUE_SIZE,
        classifier=False,
        parquet_uri=config.output_dir,
        job_id=f"group-{index}",
        infer_precision=INFER_PRECISION,
        amp_impl=AMP_IMPL,
        classify_impl=CLASSIFY_IMPL,
    )
    bridge.run_picking()
    for fault in bridge.signal_faults:
        logger.warning("signal fault: %s", fault)
    # Per-group start/end wall timestamps make the tail spread across groups visible.
    finished = time.time()
    logger.setLevel(logging.INFO)
    graph = bridge.model.fast_graph
    logger.info(
        "group %d: classify impl %s, %s", index, CLASSIFY_IMPL,
        f"{graph.station_days} station-days / {graph.batches} batches / {graph.windows} windows "
        "through the fast graph"
        if graph is not None else "SeisBench forward",
    )
    logger.info(
        "group %d: prefetch %s, %s", index, "on" if config.prefetch_dir else "off",
        ", ".join(f"{k} {v}" for k, v in source.io_counts.items()),
    )
    logger.info(
        "group %d: %d stations, %d threads, start %.3f end %.3f (%.2f s)",
        index, len(stations), config.threads, started, finished, finished - started,
    )
    if profile.enabled:
        logger.info("group %d stage profile:%s", index, profile.report())


def run_prefetch(root: str, workload: dict, groups: list[list[dict]], slots: list,
                 workers: list) -> None:
    """Parent side of QS_PREFETCH=on: fetch every group's objects while the workers run."""
    from sb_catalog.src.prefetch import ParentPrefetcher
    from sb_catalog.src.s3_helper import S3DataSource
    from sb_catalog.src.utils import parse_year_day

    planners = [
        S3DataSource.planner(
            start=parse_year_day(workload["start"]),
            end=parse_year_day(workload["end"]),
            stations=",".join(s["id"] for s in group),
            components=workload["components"],
            db=StationTable(group),
        )
        for group in groups
    ]
    stats = ParentPrefetcher(root, planners, slots, alive=lambda g: workers[g].is_alive()).run()
    logger.info(
        "prefetch: parent LISTs %d (failed %d), GETs %d (failed %d, %.1f MB) per group %s, "
        "last GET at %.2f s, done at %.2f s, slot wait %.2f s (timeouts %d)",
        stats["lists"], stats["list_failed"], stats["gets"], stats["get_failed"],
        stats["bytes"] / 1e6, stats["gets_per_group"], stats["last_get_s"],
        stats["elapsed_s"], stats["slot_wait_s"], stats["slot_timeouts"],
    )


def main() -> int:
    """Pick every station-day of a workload with the production worker layout."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vcpus", required=True, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if PREFETCH not in ("on", "off"):
        raise ValueError(f"unknown QS_PREFETCH {PREFETCH!r}; expected 'on' or 'off'")
    workload = json.loads(args.workload.read_text())
    stations = workload["stations"]
    threads, groups, data_queue_size = resolve_layout(args.vcpus, stations)

    cache_root = Path(tempfile.mkdtemp(prefix="seisbench-"))
    stage_weights(cache_root)
    os.environ["SEISBENCH_CACHE_ROOT"] = str(cache_root)
    # SeisBench writes its cache config.json on first import; do it once here, in the
    # parent, so the spawned workers do not race on writing it.
    import seisbench

    logger.info("SeisBench %s, weights staged in %s", seisbench.__version__, cache_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "picking %d stations with %d procs x %d threads (layout %s, data queue %d, prefetch %s)",
        len(stations), len(groups), threads, LAYOUT, data_queue_size, PREFETCH,
    )

    context = mp.get_context("spawn")
    prefetch_dir = slots = None
    if PREFETCH == "on":
        sys.path.insert(0, str(QUAKESCOPE_ROOT))
        from sb_catalog.src.prefetch import PREFETCH_WINDOW

        # Under the run's own TMPDIR, removed below: never shared across runs.
        prefetch_dir = tempfile.mkdtemp(prefix="qs-prefetch-")
        slots = [context.Semaphore(PREFETCH_WINDOW) for _ in groups]
    config = PickJobConfig(
        workload=workload,
        output_dir=str(args.output_dir),
        threads=threads,
        data_queue_size=data_queue_size,
        prefetch_dir=prefetch_dir,
    )
    workers = [
        context.Process(target=pick_group,
                        args=(i, group, config, slots[i] if slots else None))
        for i, group in enumerate(groups)
    ]
    try:
        for worker in workers:
            worker.start()
        if prefetch_dir is not None:
            try:
                run_prefetch(prefetch_dir, workload, groups, slots, workers)
            except Exception:
                # The workers read whatever is missing directly.
                logger.exception("prefetch failed; the workers read the rest from S3")
        for worker in workers:
            worker.join()
    finally:
        if prefetch_dir is not None:
            shutil.rmtree(prefetch_dir, ignore_errors=True)
    failed = [i for i, worker in enumerate(workers) if worker.exitcode != 0]
    if failed:
        logger.error("worker groups %s failed", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
