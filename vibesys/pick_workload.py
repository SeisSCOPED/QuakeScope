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
import importlib.util
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


def _load_by_path(name: str):
    """A sb_catalog/src module loaded by path, so this process's sys.path stays as it is."""
    spec = importlib.util.spec_from_file_location(
        f"_{name}", QUAKESCOPE_ROOT / "sb_catalog" / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Startup attribution (sb_catalog.src.startup_diag): log-only, in every process and every arm.
startup_diag = _load_by_path("startup_diag")

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
#: What a worker imports before its first classify (sb_catalog.src.picker.IMPORT_MODES): "full"
#: imports SeisBench's model package before the model is built, the round-13 order. QS_IMPORTS=lean
#: loads PhaseNet as sb_catalog.src.lean_phasenet.LeanPhaseNet on the fast classify/annotate path,
#: never importing seisbench.models (whose obspy.signal.trigger import pulls in scipy.signal,
#: scipy.stats and matplotlib, ~0.7 s), and the fused amplitude passes import scipy/obspy.signal
#: on a thread started by the first classify; the parent reads SeisBench's version from its package
#: metadata instead of importing it. Not the default with the scipy amp signal (QS_STARTUP2=off): in the round-14 A/B the lean workers were
#: importing 0.90 s earlier, but the import thread slowed the first classify (first amplitude pass
#: 0.56 s later) and lean+fast exit ran 2.9% slower than full+normal (19.06 vs 18.52 s median).
#: QS_STARTUP2 v3 (the default) and v2 make lean the default with the slim amp signal, which imports nothing on a thread.
IMPORT_ARMS = ("lean", "full")
#: Startup v2 (round 16): "v2" makes three changes to the overlap startup. (a) The lean imports
#: (QS_IMPORTS defaults to lean; an explicit QS_IMPORTS still selects). (b) The fused amplitude passes take their signal functions from
#: sb_catalog.src.slim_signal / slim_evalresp (amplitude_extractor.AMP_SIGNALS "slim"), imported
#: before the first classify, so no worker executes scipy.signal, obspy.signal or matplotlib.
#: (c) The parent, idle after the launch and with a GIL of its own, LISTs and GETs every group's
#: first station-day into a directory under TMPDIR (sb_catalog.src.parent_fetch); the worker's
#: startup waits up to s3_helper.PARENT_FIRST_WAIT for it and parses the local bytes, or reads
#: them itself (counted). QS_STARTUP2=off is the round-15 startup (full imports, scipy amp signal).
#: v2 is not the default: in the round-16 same-session 4+4 A/B its first classify came 0.86 s earlier and
#: the median wall was 3.3% lower (17.25 vs 17.84 s), picks bitwise equal, but the round-16 official run
#: was 21.54 s against round 15's 19.97 s at equal CPU-s: its workers reached the early startup at
#: +2.0-2.4 s (0.47 s locally) and their imports at +5.0 s (1.39 s), with every parent fetch ready at
#: 1.90 s. The round-17 official startup-diag (off) put the workers' early startup at +0.63 s and their
#: imports at +4.0 s, ~1.3 s of it obspy.signal/scipy.signal/matplotlib under seisbench.models, wall-bound
#: (wall/CPU ~1.8, no major faults). v2's pre-early imports are off's, so the v2 loss sits in that window,
#: and the parent's concurrent fetch (s3fs/aiohttp imports, HTTPS threads, TMPDIR writes) is the suspect.
#: "v3", the default (round 18): (a) and (b) without (c): the lean imports and the slim amp signal, with
#: no parent fetch and no worker wait, so a worker LISTs and GETs its first station-day itself, the
#: round-15/17 worker-own LIST/prefetch path.
STARTUP2_ARMS = ("v3", "v2", "off")
STARTUP2 = os.environ.get("QS_STARTUP2", "v3")
IMPORTS = os.environ.get("QS_IMPORTS", "full" if STARTUP2 == "off" else "lean")
AMP_SIGNAL = "scipy" if STARTUP2 == "off" else "slim"
#: (label, timeline mark) whose first mark's CPU, faults and context switches a worker's
#: startup-diag line reports.
DIAG_MARKS = (
    ("early", "early_started"), ("imports", "imports_done"), ("source", "source_ready"),
    ("model", "model_ready"), ("parent_wait", "parent_wait"), ("parent_ready", "parent_ready"),
    ("first_sd", "sd_loaded"), ("first_classify", "classify_start"), ("first_amp", "amp_start"),
    ("end", "end"),
)
#: Modules a worker reports at its end; under v2 and v3 none of them may be loaded.
END_MODULES = ("seisbench.models", "scipy.signal", "obspy.signal", "matplotlib")
#: How a process ends once its work is durably done: "fast" flushes the log streams, shuts
#: logging down and calls os._exit, skipping interpreter teardown (torch/oneDNN, module GC) in
#: each worker after its Parquet writer is closed, and in the parent after every worker joined
#: with exit code 0; QS_EXIT=normal returns through the interpreter's own exit. A worker that
#: raises always takes the normal path, so its exit code stays nonzero.
EXIT_ARMS = ("fast", "normal")
EXIT = os.environ.get("QS_EXIT", "fast")
#: Cores of finished workers (sb_catalog.src.tail_share): "widen" lets the workers still running
#: claim them, under a lock-protected table the parent shares with every worker, and run their
#: forward (per batch, in the classify thread) and fused filter pool (per station-day) on
#: threads + claim, never more than --vcpus in total; QS_TAIL=static keeps every worker at
#: `threads` until it ends, the round-14 behaviour.
TAIL_ARMS = ("widen", "static")
TAIL = os.environ.get("QS_TAIL", "widen")
#: A worker's import chain (round 19): "threads" starts one daemon thread ("imp") at pick_group
#: entry, after the thread-count variables and sys.path are set, that imports torch, pandas and
#: pyarrow, then the worker modules the main thread imports after its early network startup
#: (import_par_modules), while the main thread runs its early imports and starts the S3/FDSN
#: reads unchanged; the main thread joins it before its own picker import. The round-18 official
#: log had the workers' I/O done by +1.6 s but their imports at +2.9 s, torch starting at +1.04 s
#: and half its 1.86 s wall waiting. The thread runs imports only, no torch op. Anything it raises
#: (an importlib _DeadlockError included), or a main-thread early import failing while it runs,
#: is logged and counted (import_fallbacks) and the imports then run serially on the main thread.
#: QS_IMPORT_PAR=off is the round-18 serial order.
IMPORT_PAR_ARMS = ("threads", "off")
IMPORT_PAR = os.environ.get("QS_IMPORT_PAR", "threads")
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
    vcpus: int = 0
    workers: int = 1


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


def fast_exit(code: int) -> None:
    """Flush the log streams, shut logging down and leave without interpreter teardown."""
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(code)


def import_par_modules(imports: str = IMPORTS, amp_signal: str = AMP_SIGNAL) -> tuple[str, ...]:
    """What the QS_IMPORT_PAR=threads import thread imports, in order: torch, pandas and pyarrow
    first, then the worker modules the main thread imports after its early startup, and those
    the bridge's construction imports (the fast graph and annotate; the slim amp modules)."""
    full = ("seisbench.models", "sb_catalog.src.classifier") if imports == "full" else ()
    slim = (("scipy.fft", "obspy.core.inventory", "sb_catalog.src.slim_evalresp",
             "sb_catalog.src.slim_signal") if amp_signal == "slim" else ())
    return ("torch", "torch.nn", "torch.nn.functional", "pandas", "pyarrow", "pyarrow.parquet",
            "seisbench", "seisbench.util", *full, "sb_catalog.src.picker",
            "sb_catalog.src.profiling", "sb_catalog.src.s3_helper", "sb_catalog.src.utils",
            "sb_catalog.src.lean_phasenet", "sb_catalog.src.amplitude_extractor",
            "sb_catalog.src.fast_forward", "sb_catalog.src.fast_annotate", *slim)


class ImportThread:
    """QS_IMPORT_PAR=threads: one daemon thread importing ``modules`` in order, recording its
    start and end (wall epoch), its thread CPU, and anything it raised."""

    def __init__(self, modules: tuple[str, ...]) -> None:
        import threading

        self.modules = modules
        self.start = self.end = self.cpu = None
        self.error: BaseException | None = None
        self.done = 0
        self.join_wait = 0.0
        self.thread = threading.Thread(target=self._run, name="imp", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        import importlib

        self.start = time.time()
        cpu = time.thread_time()
        try:
            for name in self.modules:
                importlib.import_module(name)
                self.done += 1
        except BaseException as exc:                  # e.g. importlib's _DeadlockError
            self.error = exc
        finally:
            self.cpu = time.thread_time() - cpu
            self.end = time.time()

    def join(self) -> None:
        """Wait for the thread (once); the wait adds to join_wait."""
        begin = time.time()
        self.thread.join()
        self.join_wait += time.time() - begin

    def summary(self, t0: float) -> str:
        """`start=+s end=+s cpu=s modules=done/total join_wait_ms=ms`, marks relative to ``t0``."""
        mark = (lambda t: f"{t - t0:+.3f}" if t is not None else "-")
        return (f"start={mark(self.start)} end={mark(self.end)} "
                f"cpu={self.cpu if self.cpu is not None else float('nan'):.3f} "
                f"modules={self.done}/{len(self.modules)} join_wait_ms={self.join_wait * 1000:.0f}")


def early_imports():
    """The overlap startup's network imports, before the S3/FDSN reads start: obspy, s3fs and
    the readers the background threads will call; returns S3DataSource."""
    import obspy.clients.fdsn  # noqa: F401
    import obspy.io.mseed.core  # noqa: F401
    import obspy.io.stationxml.core  # noqa: F401

    from sb_catalog.src.s3_helper import S3DataSource

    return S3DataSource


def pick_group(index: int, stations: list[dict], config: PickJobConfig,
               tail_state: tuple | None = None, first_sd: tuple | None = None) -> None:
    """Run one production worker loop over a subset of the workload's stations.

    ``tail_state`` is the parent's (lock, shared int array) for QS_TAIL=widen, or None.
    ``first_sd`` is the parent's (directory, event) for this group's first station-day
    (QS_STARTUP2=v2), or None.
    """
    started = time.time()
    entry_usage, entry_load = startup_diag.usage(), startup_diag.loadavg()
    import_clock = startup_diag.ImportClock(started).install()
    mallopt_rc = apply_malloc()
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(config.threads)
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    sys.path.insert(0, str(QUAKESCOPE_ROOT))
    logging.basicConfig(
        level=logging.WARNING, format=f"[group {index}] %(levelname)s %(name)s: %(message)s"
    )

    workload = config.workload
    importer = None
    import_fallbacks = []
    if IMPORT_PAR == "threads":
        # numpy on this thread first: every import on both threads needs it.
        import numpy  # noqa: F401

        importer = ImportThread(import_par_modules())
    source = None
    if STARTUP == "overlap":
        # The network work first, with only what it needs imported: obspy, s3fs and
        # the readers it will call, so the background threads import nothing while
        # this thread imports torch and SeisBench.
        try:
            S3DataSource = early_imports()
        except Exception as exc:
            if importer is None:
                raise
            # Raced with the import thread (e.g. a _DeadlockError): let it finish, then serially.
            importer.join()
            import_fallbacks.append(f"main:{type(exc).__name__}")
            logger.warning("group %d: early imports failed beside the import thread (%r); "
                           "importing serially", index, exc)
            S3DataSource = early_imports()

        source = S3DataSource(
            stations=",".join(s["id"] for s in stations),
            start=datetime.datetime.strptime(workload["start"], "%Y.%j").date(),
            end=datetime.datetime.strptime(workload["end"], "%Y.%j").date(),
            components=workload["components"],
            db=None,
            early_channels={s["id"]: s["channels"] for s in stations},
            list_mode=LIST,
            parent_first=first_sd,
        )

    if importer is not None:
        importer.join()
        if importer.error is not None:
            # Whatever it did not import, the serial imports below import.
            import_fallbacks.append(f"thread:{type(importer.error).__name__}")
            logger.warning(
                "group %d: import thread failed after %d of %d modules (%r); importing serially",
                index, importer.done, len(importer.modules), importer.error)
    if IMPORTS == "full":
        # The round-13 import set: picker.py imported SeisBench's model package,
        # QuakeXNet's module and pandas (through utils) at module level.
        import pandas  # noqa: F401
        import seisbench.models  # noqa: F401

        import sb_catalog.src.classifier  # noqa: F401
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
    tail = None
    if tail_state is not None:
        from sb_catalog.src.tail_share import TailShare

        tail = TailShare(*tail_state, index=index, workers=config.workers, vcpus=config.vcpus,
                         base=config.threads, total_units=len(stations) * max(1, (end - start).days),
                         t0=started)
    try:
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
            imports=IMPORTS,
            tail=tail,
            amp_signal=AMP_SIGNAL,
        )
        timeline.mark("model_ready")
        import_clock.uninstall()
        bridge.run_picking()
    finally:
        # Done or failed: this worker's cores go to the peers still running.
        if tail is not None:
            tail.finish()
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
            ("first_amp", "amp_start", "first"), ("amp_imports", "amp_imports_done", "first"),
            ("parent_wait", "parent_wait", "first"), ("parent_ready", "parent_ready", "first"),
            ("inv_joined", "inventory_joined", "first"),
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
    first = bridge.modules_at_first_classify or {}
    logger.info(
        "group %d: imports=%s exit=%s model=%s amp_imports=%s at first classify %s; at end "
        "seisbench.models=%d", index, IMPORTS, EXIT, type(bridge.model).__name__,
        (bridge.amp_extor.warm_up_mode or "-").replace(" ", "_"),
        " ".join(f"{name}={int(loaded)}" for name, loaded in first.items()) or "-",
        "seisbench.models" in sys.modules,
    )
    logger.info(
        "group %d: startup2=%s first_sd_source=%s parent %s; amp signal %s (asked %s) slim %s; "
        "modules at end %s", index, STARTUP2, source.first_sd_source,
        " ".join(f"{k}={v}" for k, v in source.parent_counters.items()),
        bridge.amp_extor.signal_in_use, bridge.amp_extor.signal,
        " ".join(f"{k}={v}" for k, v in bridge.amp_extor.slim_counters.items()),
        " ".join(f"{name}={int(name in sys.modules)}" for name in END_MODULES),
    )
    extor = bridge.amp_extor
    logger.info(
        "group %d: tail=%s %s; amp pool widths %s", index, TAIL,
        tail.summary() if tail is not None else
        f"0 claims, 0 events, 0 updates, max total -, threads at last batch {config.threads}; events -",
        " ".join(f"{t - started:+.3f}:{a}->{b}" for t, a, b in extor.width_events) or "-",
    )
    logger.info(
        "group %d: import_par=%s thread %s import_fallbacks=%d %s", index, IMPORT_PAR,
        importer.summary(started) if importer is not None else "-", len(import_fallbacks),
        ",".join(import_fallbacks) or "-",
    )
    logger.info(
        "startup-diag group %d: startup2=%s imports=%s import_par=%s start=%+.3f entry=%+.3f load=%s "
        "marks module=%s entry=%s %s; imports %s; bytecode %s", index, STARTUP2, IMPORTS, IMPORT_PAR,
        startup_diag.process_start() - config.launched if config.launched else float("nan"),
        started - config.launched if config.launched else float("nan"), entry_load,
        startup_diag.fmt_usage(MODULE_USAGE, started), startup_diag.fmt_usage(entry_usage, started),
        timeline.usage_line(started, DIAG_MARKS), import_clock.line(), startup_diag.bytecode_probe(),
    )
    logger.info(
        "group %d: %d stations, %d threads, start %.3f end %.3f (%.2f s)",
        index, len(stations), config.threads, started, finished, finished - started,
    )
    if EXIT == "fast":
        # run_picking returned: the Parquet writer is closed, the pick queue is
        # drained and the amplitude pool is shut down.
        logger.info("group %d: exit fast at %.3f (%.3f s after end)", index, time.time(),
                    time.time() - finished)
        fast_exit(0)


def main() -> int:
    """Pick every station-day of a workload with the production worker layout."""
    main_entry = time.time()
    diag = {"entry": startup_diag.usage()}
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
    if IMPORTS not in IMPORT_ARMS:
        parser.error(f"unknown QS_IMPORTS {IMPORTS!r}; expected one of {list(IMPORT_ARMS)}")
    if EXIT not in EXIT_ARMS:
        parser.error(f"unknown QS_EXIT {EXIT!r}; expected one of {list(EXIT_ARMS)}")
    if TAIL not in TAIL_ARMS:
        parser.error(f"unknown QS_TAIL {TAIL!r}; expected one of {list(TAIL_ARMS)}")
    if STARTUP2 not in STARTUP2_ARMS:
        parser.error(f"unknown QS_STARTUP2 {STARTUP2!r}; expected one of {list(STARTUP2_ARMS)}")
    if IMPORT_PAR not in IMPORT_PAR_ARMS:
        parser.error(f"unknown QS_IMPORT_PAR {IMPORT_PAR!r}; expected one of {list(IMPORT_PAR_ARMS)}")
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
        vcpus=args.vcpus,
        workers=len(groups),
    )
    context = mp.get_context("spawn")
    tail_state = None
    if TAIL == "widen" and len(groups) > 1:
        # The table's layout is tail_share's; load the module by path, so the parent's
        # sys.path (which the spawned workers inherit) stays as it is.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_tail_share", QUAKESCOPE_ROOT / "sb_catalog" / "src" / "tail_share.py")
        tail_share = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tail_share)
        tail_state = (context.Lock(), context.RawArray("i", tail_share.array_size(len(groups))))
    fetch = None
    first_sds = [None] * len(groups)
    # Only v2 fetches in the parent; v3 and off leave the first station-day to the worker.
    if STARTUP2 == "v2" and STARTUP == "overlap" and LIST == "station":
        # Loaded by path, as tail_share: the parent's sys.path stays as it is.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_parent_fetch", QUAKESCOPE_ROOT / "sb_catalog" / "src" / "parent_fetch.py")
        parent_fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parent_fetch)
        events = [context.Event() for _ in groups]
        fetch = parent_fetch.FirstSDFetch(tempfile.mkdtemp(prefix="qs-firstsd-"), groups,
                                          workload, events)
        first_sds = [(fetch.dirs[i], events[i]) if fetch.plans[i] is not None else None
                     for i in range(len(groups))]
    workers = [
        context.Process(target=pick_group, args=(i, group, config, tail_state, first_sds[i]))
        for i, group in enumerate(groups)
    ]
    try:
        diag["starts"] = []
        for worker in workers:
            begin = time.time()
            worker.start()
            diag["starts"].append(time.time() - begin)
        diag["started"] = time.time()
        if fetch is not None:
            fetch.start()
        return _run_parent(workers, config, main_entry, cache_root, fetch, diag)
    finally:
        remove_first_sd_dir(fetch)


def remove_first_sd_dir(fetch) -> None:
    """Nothing outlives the run: remove the parent's first-SD directory, with any object a
    worker did not consume (QS_STARTUP2=v2; a no-op otherwise, or once removed)."""
    if fetch is None or not fetch.root.exists():
        return
    import shutil

    fetch.join(timeout=0.5)
    shutil.rmtree(fetch.root, ignore_errors=True)
    logger.info("parent startup2: fetch directory %s removed=%s", fetch.root,
                not fetch.root.exists())


def _run_parent(workers, config: PickJobConfig, main_entry: float, cache_root: Path,
                fetch, diag: dict) -> int:
    """The parent after the launch: SeisBench's version, join every worker, exit."""
    logger.info(
        "parent %s: pre-launch %.3f s (main entry %.3f -> launch %.3f), %d workers started %.3f s "
        "after launch", PARENT, config.launched - main_entry, main_entry, config.launched,
        len(workers), time.time() - config.launched,
    )
    diag["sbver_start"] = time.time()
    if PARENT == "early":
        if IMPORTS == "lean":
            from importlib.metadata import version

            seisbench_version = version("seisbench")
        else:
            import seisbench

            seisbench_version = seisbench.__version__
        logger.info("SeisBench %s, weights staged in %s", seisbench_version, cache_root)
    diag["sbver_end"] = time.time()
    for worker in workers:
        worker.join()
    joined = time.time()
    joined_usage = startup_diag.usage()
    logger.info(
        "startup-diag parent: startup2=%s imports=%s parent=%s start=%+.3f load=%s marks module=%s "
        "entry=%s launch=%+.3f worker_starts=%s started=%+.3f sbver=%+.3f/%.3f joined=%s",
        STARTUP2, IMPORTS, PARENT, startup_diag.process_start() - main_entry, startup_diag.loadavg(),
        startup_diag.fmt_usage(MODULE_USAGE, main_entry), startup_diag.fmt_usage(diag["entry"], main_entry),
        config.launched - main_entry, ",".join(f"{d:.3f}" for d in diag["starts"]),
        diag["started"] - main_entry, diag["sbver_start"] - main_entry,
        diag["sbver_end"] - diag["sbver_start"], startup_diag.fmt_usage(joined_usage, main_entry),
    )
    if fetch is not None:
        for i in range(len(workers)):
            logger.info("parent startup2 group %d: first-SD fetch %s", i, fetch.summary(i))
    failed = [i for i, worker in enumerate(workers) if worker.exitcode != 0]
    if failed:
        logger.error("worker groups %s failed", failed)
        return 1
    logger.info("parent exit=%s imports=%s startup2=%s: workers joined %.3f, exit %.3f", EXIT,
                IMPORTS, STARTUP2, joined, time.time())
    if EXIT == "fast":
        remove_first_sd_dir(fetch)
        fast_exit(0)
    return 0


#: This module's top level done (in a spawned worker: the spawn bootstrap ran it as __mp_main__).
MODULE_USAGE = startup_diag.usage()

if __name__ == "__main__":
    sys.exit(main())
