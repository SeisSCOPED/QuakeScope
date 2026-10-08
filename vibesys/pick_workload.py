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
import datetime
import json
import logging
import multiprocessing as mp
import os
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
#: Amplitude passes (sb_catalog.src.amplitude_extractor.AMP_IMPLS): "fused" is the
#: "fast" arithmetic with the velocity prep fused onto a per-process filter pool and
#: the Wood-Anderson and velocity passes overlapped; QS_AMP_IMPL=fast is the serial
#: round-4 path; QS_AMP_IMPL=obspy restores the seed path.
AMP_IMPLS = ("fast", "obspy", "fused")
AMP_IMPL = os.environ.get("QS_AMP_IMPL", "fused")
#: PhaseNet forward implementation (sb_catalog.src.picker.CLASSIFY_IMPLS): "fast" runs the
#: BN-folded channels-last graph at 64 windows per batch; QS_CLASSIFY_IMPL=seisbench restores
#: SeisBench's own forward at 256.
CLASSIFY_IMPL = os.environ.get("QS_CLASSIFY_IMPL", "fast")
#: Windowing/stacking/picking around the fast graph (sb_catalog.src.picker.ANNOTATE_IMPLS):
#: "fast" cuts the windows from one float32 array per block and stacks them as an overlap-add;
#: QS_ANNOTATE_IMPL=seisbench restores SeisBench's annotate (the round-7 path).
ANNOTATE_IMPLS = ("seisbench", "fast")
ANNOTATE_IMPL = os.environ.get("QS_ANNOTATE_IMPL", "fast")
#: Worker startup (sb_catalog.src.s3_helper.S3DataSource): "overlap" starts the FDSN
#: inventory fetch and the first station-day's LIST and reads on background threads before
#: the heavy imports and the model build, and reads a station-day's components
#: concurrently; QS_STARTUP=serial is the round-8 order (inventory, model, then the first LIST).
STARTUP_ARMS = ("serial", "overlap")
STARTUP = os.environ.get("QS_STARTUP", "overlap")
#: The fast graph's transposed convs (sb_catalog.src.fast_forward.DECONV_IMPLS): "transpose" is
#: oneDNN's deconvolution (the round-9 graph); QS_DECONV=polyphase runs each as a stride-1 conv
#: over its 4 output phases (1.14x faster forward in isolation, but only 1.6% wall on the 8-core
#: workload, so not the default).
DECONV_IMPLS = ("transpose", "polyphase")
DECONV = os.environ.get("QS_DECONV", "transpose")
#: glibc malloc in each worker: "default" is glibc's dynamic threshold; QS_MALLOC=pinned fixes the
#: mmap and trim thresholds (mallopt, before torch is imported), so the forward's 3-12 MB
#: activations are reused from the heap instead of being mmapped and munmapped on every batch.
MALLOC_ARMS = ("default", "pinned")
MALLOC = os.environ.get("QS_MALLOC", "default")
#: How a worker finds a station-day's objects (sb_catalog.src.s3_helper.LIST_MODES): "station"
#: LISTs only the station's key stem under the day prefix on SCEDC/NCEDC (one request, a dozen
#: keys); QS_LIST=day LISTs the whole day prefix per network (~4,000 keys on SCEDC), the round-9
#: path. GeoNet and EarthScope list whole days either way.
LIST_MODES = ("station", "day")
LIST = os.environ.get("QS_LIST", "station")
#: Parent work before the workers launch: "early" writes SeisBench's cache config.json itself
#: (the bytes `import seisbench` writes) and imports SeisBench only after the launch;
#: QS_PARENT=serial imports it before the launch, the round-9 order.
PARENT_ARMS = ("early", "serial")
PARENT = os.environ.get("QS_PARENT", "early")
#: seisbench/__init__.py's default config, which it writes as json.dump(..., indent=4,
#: sort_keys=True) when the cache root has no config.json.
SEISBENCH_DEFAULT_CONFIG = {"dimension_order": "NCW", "component_order": "ZNE"}
#: (name, mallopt parameter, value) applied by the "pinned" arm.
MALLOPT_PINNED = (("M_TRIM_THRESHOLD", -1, 1 << 30), ("M_MMAP_THRESHOLD", -3, 1 << 30))
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


def write_seisbench_config(cache_root: Path) -> None:
    """Write the config.json `import seisbench` writes into an empty cache root, so the
    workers' imports find it and only read it, without the parent importing SeisBench."""
    with open(cache_root / "config.json", "x") as fconfig:
        json.dump(SEISBENCH_DEFAULT_CONFIG, fconfig, indent=4, sort_keys=True)


@dataclass(frozen=True)
class PickJobConfig:
    """Picking parameters shared by every worker group in one run."""

    workload: dict
    output_dir: str
    threads: int
    data_queue_size: int
    launched: float = 0.0


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


def apply_malloc(arm: str = MALLOC) -> dict[str, int]:
    """Apply a QS_MALLOC arm to this process's glibc malloc; return each mallopt's rc (1 = set)."""
    if arm == "default":
        return {}
    import ctypes

    mallopt = getattr(ctypes.CDLL(None), "mallopt", None)
    return {name: mallopt(param, value) if mallopt is not None else -1
            for name, param, value in MALLOPT_PINNED}


def pick_group(index: int, stations: list[dict], config: PickJobConfig) -> None:
    """Run one production worker loop over a subset of the workload's stations."""
    started = time.time()
    mallopt_rc = apply_malloc()
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(config.threads)
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    sys.path.insert(0, str(QUAKESCOPE_ROOT))
    logging.basicConfig(
        level=logging.WARNING, format=f"[group {index}] %(levelname)s %(name)s: %(message)s"
    )

    workload = config.workload
    source = None
    if STARTUP == "overlap":
        # The network work first, with only what it needs imported: obspy, s3fs and
        # the readers it will call, so the background threads import nothing while
        # this thread imports torch and SeisBench.
        import obspy.clients.fdsn  # noqa: F401
        import obspy.io.mseed.core  # noqa: F401
        import obspy.io.stationxml.core  # noqa: F401

        from sb_catalog.src.s3_helper import S3DataSource

        source = S3DataSource(
            stations=",".join(s["id"] for s in stations),
            start=datetime.datetime.strptime(workload["start"], "%Y.%j").date(),
            end=datetime.datetime.strptime(workload["end"], "%Y.%j").date(),
            components=workload["components"],
            db=None,
            early_channels={s["id"]: s["channels"] for s in stations},
            list_mode=LIST,
        )

    from sb_catalog.src.picker import S3MongoSBBridge
    from sb_catalog.src.profiling import timeline
    from sb_catalog.src.s3_helper import S3DataSource
    from sb_catalog.src.utils import parse_year_day

    timeline.mark("imports_done")
    db = StationTable(stations)
    start, end = parse_year_day(workload["start"]), parse_year_day(workload["end"])
    if source is None:
        source = S3DataSource(
            stations=",".join(s["id"] for s in stations),
            start=start,
            end=end,
            components=workload["components"],
            db=db,
            list_mode=LIST,
        )
    else:
        if (source.start, source.end) != (start, end):
            raise RuntimeError(f"startup dates {source.start}..{source.end} != {start}..{end}")
        source.attach_db(db)
    timeline.mark("source_ready")
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
        annotate_impl=ANNOTATE_IMPL,
        deconv_impl=DECONV,
    )
    timeline.mark("model_ready")
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
        "group %d: deconv=%s malloc=%s mallopt %s", index,
        graph.deconv if graph is not None else "-", MALLOC,
        " ".join(f"{k}={v}" for k, v in mallopt_rc.items()) or "-",
    )
    annotator = bridge.model.fast_annotator
    logger.info(
        "group %d: annotate impl %s, %s", index, ANNOTATE_IMPL,
        annotator.summary() if annotator is not None else "SeisBench annotate",
    )
    extor = bridge.amp_extor
    logger.info(
        "group %d: amp impl %s, filter pool %s, %d traces filtered over %d station passes "
        "(%d masked fallbacks), WA/velocity overlapped on %d of %d station-days",
        index, AMP_IMPL, extor.workers if AMP_IMPL == "fused" else "-",
        extor.vel_counters["traces_filtered"], extor.vel_counters["stations"],
        extor.vel_counters["masked_fallbacks"], bridge.amp_overlap["overlapped"],
        bridge.amp_overlap["station_days"],
    )
    timeline.mark("end", finished)
    logger.info(
        "group %d: startup %s timeline launch=%+.3f %s", index, STARTUP,
        config.launched - started if config.launched else float("nan"),
        timeline.line(started, [
            ("early", "early_started", "first"), ("imports", "imports_done", "first"),
            ("inv_start", "inventory_start", "first"), ("inv_ready", "inventory_ready", "first"),
            ("source", "source_ready", "first"), ("model", "model_ready", "first"),
            ("list_start", "list_start", "first"), ("list_end", "list_end", "first"),
            ("read_start", "read_start", "first"), ("first_sd", "sd_loaded", "first"),
            ("first_data", "data_enqueued", "first"), ("first_classify", "classify_start", "first"),
            ("first_amp", "amp_start", "first"), ("inv_joined", "inventory_joined", "first"),
            ("last_pick", "pick_written", "last"), ("end", "end", "first"),
        ]),
    )
    logger.info(
        "group %d: list=%s parent=%s list counters %s", index, source.list_mode, PARENT,
        " ".join(f"{k}={v}" for k, v in source.list_counters.items()),
    )
    logger.info(
        "group %d: startup %s counters %s", index, STARTUP,
        " ".join(f"{k}={v}" for k, v in source.startup_counters.items()),
    )
    logger.info(
        "group %d: %d stations, %d threads, start %.3f end %.3f (%.2f s)",
        index, len(stations), config.threads, started, finished, finished - started,
    )


def main() -> int:
    """Pick every station-day of a workload with the production worker layout."""
    main_entry = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vcpus", required=True, type=int)
    args = parser.parse_args()
    if AMP_IMPL not in AMP_IMPLS:
        parser.error(f"unknown QS_AMP_IMPL {AMP_IMPL!r}; expected one of {list(AMP_IMPLS)}")
    if STARTUP not in STARTUP_ARMS:
        parser.error(f"unknown QS_STARTUP {STARTUP!r}; expected one of {list(STARTUP_ARMS)}")
    if ANNOTATE_IMPL not in ANNOTATE_IMPLS:
        parser.error(
            f"unknown QS_ANNOTATE_IMPL {ANNOTATE_IMPL!r}; expected one of {list(ANNOTATE_IMPLS)}")
    if DECONV not in DECONV_IMPLS:
        parser.error(f"unknown QS_DECONV {DECONV!r}; expected one of {list(DECONV_IMPLS)}")
    if MALLOC not in MALLOC_ARMS:
        parser.error(f"unknown QS_MALLOC {MALLOC!r}; expected one of {list(MALLOC_ARMS)}")
    if LIST not in LIST_MODES:
        parser.error(f"unknown QS_LIST {LIST!r}; expected one of {list(LIST_MODES)}")
    if PARENT not in PARENT_ARMS:
        parser.error(f"unknown QS_PARENT {PARENT!r}; expected one of {list(PARENT_ARMS)}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    workload = json.loads(args.workload.read_text())
    stations = workload["stations"]
    threads, groups, data_queue_size = resolve_layout(args.vcpus, stations)

    cache_root = Path(tempfile.mkdtemp(prefix="seisbench-"))
    stage_weights(cache_root)
    os.environ["SEISBENCH_CACHE_ROOT"] = str(cache_root)
    # SeisBench writes its cache config.json on first import; do it once here, in the
    # parent, so the spawned workers do not race on writing it. The early arm writes the
    # same bytes without the import, which then runs after the launch and only reads them.
    if PARENT == "serial":
        import seisbench

        logger.info("SeisBench %s, weights staged in %s", seisbench.__version__, cache_root)
    else:
        write_seisbench_config(cache_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "picking %d stations with %d procs x %d threads (layout %s, data queue %d)",
        len(stations), len(groups), threads, LAYOUT, data_queue_size,
    )

    config = PickJobConfig(
        workload=workload,
        output_dir=str(args.output_dir),
        threads=threads,
        data_queue_size=data_queue_size,
        launched=time.time(),
    )
    context = mp.get_context("spawn")
    workers = [
        context.Process(target=pick_group, args=(i, group, config))
        for i, group in enumerate(groups)
    ]
    for worker in workers:
        worker.start()
    logger.info(
        "parent %s: pre-launch %.3f s (main entry %.3f -> launch %.3f), %d workers started %.3f s "
        "after launch", PARENT, config.launched - main_entry, main_entry, config.launched,
        len(workers), time.time() - config.launched,
    )
    if PARENT == "early":
        import seisbench

        logger.info("SeisBench %s, weights staged in %s", seisbench.__version__, cache_root)
    for worker in workers:
        worker.join()
    failed = [i for i, worker in enumerate(workers) if worker.exitcode != 0]
    if failed:
        logger.error("worker groups %s failed", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
