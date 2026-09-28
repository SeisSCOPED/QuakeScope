"""Seed QuakeScope picking runner.

Picks every station-day of a workload with QuakeScope's production v3 code
path (``S3DataSource`` -> ``S3MongoSBBridge`` -> ``ParquetPickWriter``), laid
out the way a production Batch task runs it: ``procs`` independent worker
processes, each with ``threads`` BLAS/OpenMP threads (fleet.json: 4 x 2 on the
8 vCPU / 16 GB Fargate shape).

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
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

QUAKESCOPE_ROOT = Path(__file__).resolve().parent / "quakescope"

#: Production layout on 8 vCPU (fleet.json): 4 worker loops x 2 threads.
THREADS_PER_PROC = 2
PICK_QUEUE_SIZE = 5
#: PhaseNet forward-pass precision (sb_catalog.src.picker.INFER_PRECISIONS):
#: bf16 autocast on AMX by default; QS_INFER_PRECISION=fp32 restores the seed path.
INFER_PRECISION = os.environ.get("QS_INFER_PRECISION", "bf16")
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


def pick_group(index: int, stations: list[dict], config: PickJobConfig) -> None:
    """Run one production worker loop over a subset of the workload's stations."""
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(config.threads)
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    sys.path.insert(0, str(QUAKESCOPE_ROOT))
    logging.basicConfig(
        level=logging.WARNING, format=f"[group {index}] %(levelname)s %(name)s: %(message)s"
    )

    from sb_catalog.src.picker import S3MongoSBBridge
    from sb_catalog.src.s3_helper import S3DataSource
    from sb_catalog.src.utils import parse_year_day

    workload = config.workload
    db = StationTable(stations)
    source = S3DataSource(
        stations=",".join(s["id"] for s in stations),
        start=parse_year_day(workload["start"]),
        end=parse_year_day(workload["end"]),
        components=workload["components"],
        db=db,
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
    )
    bridge.run_picking()
    for fault in bridge.signal_faults:
        logger.warning("signal fault: %s", fault)


def main() -> int:
    """Pick every station-day of a workload with the production worker layout."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vcpus", required=True, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    workload = json.loads(args.workload.read_text())
    threads = min(THREADS_PER_PROC, args.vcpus)
    procs = max(1, args.vcpus // threads)
    stations = workload["stations"]
    groups = [stations[i::procs] for i in range(procs) if stations[i::procs]]
    # worker._resolve_queue_size: a node-wide budget of 8 decoded station-days.
    data_queue_size = max(1, min(5, 8 // len(groups)))

    cache_root = Path(tempfile.mkdtemp(prefix="seisbench-"))
    stage_weights(cache_root)
    os.environ["SEISBENCH_CACHE_ROOT"] = str(cache_root)
    # SeisBench writes its cache config.json on first import; do it once here, in the
    # parent, so the spawned workers do not race on writing it.
    import seisbench

    logger.info("SeisBench %s, weights staged in %s", seisbench.__version__, cache_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "picking %d stations with %d procs x %d threads", len(stations), len(groups), threads
    )

    config = PickJobConfig(
        workload=workload,
        output_dir=str(args.output_dir),
        threads=threads,
        data_queue_size=data_queue_size,
    )
    context = mp.get_context("spawn")
    workers = [
        context.Process(target=pick_group, args=(i, group, config))
        for i, group in enumerate(groups)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    failed = [i for i, worker in enumerate(workers) if worker.exitcode != 0]
    if failed:
        logger.error("worker groups %s failed", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
