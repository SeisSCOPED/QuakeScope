"""The fused velocity pass's zero-phase highpass as a vendored C kernel (QS_VELIIR).

"c" filters a station's traces with veliir.c's `sos_filtfilt_df2t`: obspy's
highpass(zerophase=True) (sosfilt, flip, sosfilt, flip) bit for bit, the forward and backward
recurrences in place with no flip copies, the channels of one station interleaved in one sample
loop. "scipy" is scipy's sosfilt kernel on the filter pool, the round-31 path.

The kernel is compiled once per run, by the parent, into a fresh directory under TMPDIR
(`start_build`, a background thread running the C compiler with a hard timeout); nothing is
committed or reused across runs. The parent passes the library's path to the workers in
QS_VELIIR_SO before it spawns them. A worker loads it at its first velocity pass (`kernel()`);
until the build has finished, or if it failed (no compiler, compile error, timeout, load error),
each pass uses the scipy path, and every such pass is counted with its reason.

Standard library only: the parent loads this module by path, and the build's subprocess/shutil/
tempfile imports are left to the parent (a worker imports ctypes, json and threading).
"""
import ctypes
import json
import os
import threading
import time

ARMS = ("c", "scipy")
#: The worker's arm: "c" (the default since round 32) or "scipy".
ARM = os.environ.get("QS_VELIIR", "c")
SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "veliir.c")
#: No FMA contraction and no reassociation: every multiply and add rounds as scipy's loop does.
CFLAGS = ("-O2", "-fPIC", "-shared", "-ffp-contract=off", "-fno-fast-math")
COMPILE_TIMEOUT = 5.0
#: Most channels per kernel call (veliir.c's MAXC) and most sections (MAXSEC).
MAX_CHANNELS = 4
MAX_SECTIONS = 8
ENV_SO = "QS_VELIIR_SO"


class Build:
    """The parent's compile: veliir.c -> <dir>/veliir.so, plus <dir>/status.json."""

    def __init__(self, directory: str) -> None:
        self.dir = directory
        self.so = os.path.join(directory, "veliir.so")
        self.status = os.path.join(directory, "status.json")
        self._thread = threading.Thread(target=self._run, name="qs-veliir-build", daemon=True)

    def start(self) -> "Build":
        self._thread.start()
        return self

    def _run(self) -> None:
        import shutil
        import subprocess

        started = time.perf_counter()
        status = {"ok": False, "reason": None, "compiler": None}
        try:
            cc = shutil.which("gcc") or shutil.which("cc")
            if cc is None:
                status["reason"] = "no_compiler"
                return
            status["compiler"] = cc
            tmp = self.so + ".tmp"
            proc = subprocess.run([cc, *CFLAGS, "-o", tmp, SOURCE], stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=COMPILE_TIMEOUT)
            if proc.returncode != 0:
                status["reason"] = f"compile_rc_{proc.returncode}"
                status["stderr"] = proc.stderr.decode(errors="replace")[-400:]
                return
            os.replace(tmp, self.so)  # a worker sees the whole library or none
            status["ok"] = True
        except subprocess.TimeoutExpired:
            status["reason"] = "compile_timeout"
        except Exception as exc:  # the scipy path still works: say why and carry on
            status["reason"] = f"build_error_{type(exc).__name__}"
        finally:
            status["compile_s"] = round(time.perf_counter() - started, 4)
            try:
                tmp = self.status + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(status, fh)
                os.replace(tmp, self.status)
            except OSError:
                pass

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def remove(self) -> dict | None:
        """Remove the build directory (after the workers have exited); return the build status."""
        import shutil

        self.join(COMPILE_TIMEOUT + 1.0)
        try:
            with open(self.status) as fh:
                status = json.load(fh)
        except (OSError, ValueError):
            status = None
        shutil.rmtree(self.dir, ignore_errors=True)
        return status


def start_build(tmpdir: str | None = None) -> Build:
    """Create a fresh build directory under TMPDIR, export its library path, start the compile."""
    import tempfile

    build = Build(tempfile.mkdtemp(prefix="qs-veliir-", dir=tmpdir))
    os.environ[ENV_SO] = build.so
    return build.start()


class Kernel:
    """The worker's handle on the parent-built library, with its counters."""

    def __init__(self, so_path: str | None) -> None:
        self.so_path = so_path
        self._lock = threading.Lock()
        self._fn = None
        self._lib = None
        self._failed: str | None = None if so_path else "no_library_path"
        self.isa: str | None = None
        self.compile_s: float | None = None
        #: calls = kernel calls, channels = traces they filtered, samples = samples filtered
        #: (per trace, both passes counted once), fallbacks = traces sent to the scipy path.
        self.counters = {"calls": 0, "channels": 0, "samples": 0, "fallbacks": 0}
        self.reasons: dict[str, int] = {}

    def _status(self) -> dict | None:
        try:
            with open(os.path.join(os.path.dirname(self.so_path), "status.json")) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _load(self):
        """The kernel function, or None (with the reason in self._failed, or not built yet)."""
        if self._fn is not None or self._failed is not None:
            return self._fn
        with self._lock:
            if self._fn is not None or self._failed is not None:
                return self._fn
            status = self._status()
            if status is None:
                return None                     # still compiling: scipy for this pass
            self.compile_s = status.get("compile_s")
            if not status.get("ok"):
                self._failed = status.get("reason") or "build_failed"
                return None
            try:
                lib = ctypes.CDLL(self.so_path)
                fn = lib.sos_filtfilt_df2t
                ptr = ctypes.POINTER(ctypes.c_double)
                fn.argtypes = [ctypes.POINTER(ptr), ctypes.c_int64, ctypes.c_int, ptr,
                               ctypes.c_int, ctypes.POINTER(ptr)]
                fn.restype = ctypes.c_int
                lib.veliir_isa.restype = ctypes.c_int
                self.isa = "avx" if lib.veliir_isa() == 1 else "scalar"
            except (OSError, AttributeError) as exc:
                self._failed = f"load_{type(exc).__name__}"
                return None
            self._lib, self._fn = lib, fn
            return fn

    def fallback(self, reason: str, traces: int = 1) -> None:
        with self._lock:
            self.counters["fallbacks"] += traces
            self.reasons[reason] = self.reasons.get(reason, 0) + traces

    def ready(self) -> bool:
        """True if the library is loaded (loading it now if the build has finished)."""
        return self._load() is not None

    def why_not(self) -> str:
        return self._failed or "not_ready"

    def filtfilt(self, arrays: list, sos) -> None:
        """Zero-phase filter 1..MAX_CHANNELS C-contiguous float64 arrays of one length, in place.
        `sos` is a C-contiguous (nsec, 6) float64 array with a0 == 1. Call only when ready()."""
        ptr = ctypes.POINTER(ctypes.c_double)
        ptrs = (ptr * len(arrays))(*[a.ctypes.data_as(ptr) for a in arrays])
        n = arrays[0].shape[0]
        rc = self._fn(ptrs, n, len(arrays), sos.ctypes.data_as(ptr), sos.shape[0], ptrs)
        if rc != 0:
            raise ValueError(f"sos_filtfilt_df2t rejected C={len(arrays)} nsec={sos.shape[0]}")
        with self._lock:
            self.counters["calls"] += 1
            self.counters["channels"] += len(arrays)
            self.counters["samples"] += n * len(arrays)

    def summary(self) -> str:
        """The per-group log fields after `QS_VELIIR=<arm>`."""
        kernel = "c" if self._fn is not None and self.counters["calls"] else "scipy"
        compile_s = "-" if self.compile_s is None else f"{self.compile_s:.3f}"
        reasons = ",".join(f"{k}:{v}" for k, v in sorted(self.reasons.items())) or "-"
        return (f"kernel={kernel} isa={self.isa or '-'} compile_s={compile_s} "
                f"calls={self.counters['calls']} channels={self.counters['channels']} "
                f"samples={self.counters['samples']} fallbacks={self.counters['fallbacks']} "
                f"fallback_reasons={reasons}")


_kernel: Kernel | None = None
_kernel_lock = threading.Lock()


def kernel() -> Kernel:
    """This process's Kernel (the library at QS_VELIIR_SO), created once."""
    global _kernel
    if _kernel is None:
        with _kernel_lock:
            if _kernel is None:
                _kernel = Kernel(os.environ.get(ENV_SO))
    return _kernel
