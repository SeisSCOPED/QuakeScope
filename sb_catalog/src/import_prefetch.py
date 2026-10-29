"""Parallel bytecode prefetch for a worker's serial import chain (pick_workload QS_IMPORT_PREFETCH; round 29, H31).

On an NFS-rooted host every module a worker imports costs its main thread one synchronous round trip:
close-to-open consistency revalidates the file's attributes on each open() of its cached bytecode. Under
"mem", `ImportPrefetch.start()` reads the files a worker's imports will read (MANIFEST: paths relative
to the interpreter's stdlib and site-packages roots, resolved at runtime; code-path metadata, no bytes)
on a few daemon threads, so the opens and reads block those threads with the GIL released, and
`install()` puts a finder just ahead of the stock PathFinder on sys.meta_path. It asks PathFinder for
the spec - the spec the import system would have found - and, for a stock SourceFileLoader or
SourcelessFileLoader, shadows that loader instance's get_data with one that returns the prefetched
bytes of a path once, and reads any other path, or one not yet arrived after a bounded wait, with
the stock get_data. Bytecode validation against the source's mtime/size, the spec, origin, __file__
and the loader's class stay stock; extension modules and namespace packages are not touched.
`uninstall()` (at model_ready) removes the finder and every shadow, stops the readers and frees what
was not consumed. Every byte served was read from its file during this process; nothing is written.

`root_fstypes()` names the filesystem each root sits on (/proc/self/mountinfo, longest mount-point
prefix), logged in every arm: only a network filesystem has a per-open round trip to hide.

Standard library only, and only modules a worker has loaded before it imports this one; the reader
threads import nothing (the M21 lesson). Loaded by path, as startup_diag and tail_share. The H30
testbed and stage-0 recorder are progress-artifacts/evidence/h30/.
"""

from __future__ import annotations

import os
import resource
import sys
import threading
import time

ARMS = ("off", "mem")
#: The worker manifest (recorded by progress-artifacts/evidence/h30/record_manifest.py).
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_prefetch_manifest.txt")
#: Reader threads; the manifest is dealt to them round-robin in import order.
THREADS = 8
#: How long get_data waits for a listed file that has not arrived before it reads it itself.
WAIT_S = 0.05
#: A read larger than this loops.
CHUNK = 1 << 20
#: Root kinds a manifest line may name, as sysconfig names the paths.
ROOT_KINDS = ("stdlib", "purelib", "platlib")
_MISSING = object()


def default_roots() -> dict[str, str]:
    """{kind: directory} of this interpreter (sysconfig, which site has loaded)."""
    import sysconfig

    paths = sysconfig.get_paths()
    return {kind: paths[kind] for kind in ROOT_KINDS}


def load_manifest(path: str = MANIFEST) -> list[tuple[str, str]]:
    """[(root kind, relative path)] from a manifest: one `<kind> <relative path>` per line, `#` comments."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            kind, _, rel = line.partition(" ")
            entries.append((kind, rel))
    return entries


def _unescape(field: str) -> str:
    # mountinfo octal-escapes space, tab, newline and backslash.
    out, i = [], 0
    while i < len(field):
        code = field[i + 1:i + 4]
        if field[i] == "\\" and len(code) == 3 and code.isdigit():
            out.append(chr(int(code, 8)))
            i += 4
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def fstype_of(path: str, mountinfo: str = "/proc/self/mountinfo") -> str:
    """`<fstype>:<mount point>` of the mount holding a path (its realpath), or `?` if unknown."""
    try:
        target = os.path.realpath(path)
        with open(mountinfo) as f:
            lines = f.read().splitlines()
    except OSError:
        return "?"
    best, best_len = "?", -1
    for line in lines:
        pre, sep, post = line.partition(" - ")
        fields, tail = pre.split(), post.split()
        if not sep or len(fields) < 5 or not tail:
            continue
        mount = _unescape(fields[4])
        prefix = mount.rstrip("/") + "/"
        # The last of equal-length matches is the mount on top.
        if (target == mount or target.startswith(prefix)) and len(mount) >= best_len:
            best, best_len = f"{tail[0]}:{mount}", len(mount)
    return best


def root_fstypes(roots: dict[str, str] | None = None) -> str:
    """`stdlib=<fstype>:<mount> purelib=...` for this interpreter's roots."""
    roots = default_roots() if roots is None else roots
    return " ".join(f"{kind}={fstype_of(roots[kind])}" for kind in ("stdlib", "purelib") if kind in roots)


class ImportPrefetch:
    """The reader threads, the shadowing finder and their counters for one process."""

    def __init__(self, entries: list[tuple[str, str]], roots: dict[str, str] | None = None,
                 threads: int = THREADS, wait_s: float = WAIT_S, t0: float | None = None,
                 record: bool = False) -> None:
        self.t0 = time.time() if t0 is None else t0
        self.roots = dict(default_roots() if roots is None else roots)
        # Longest root first, so a path is attributed to its innermost root (a venv inside a prefix).
        self._prefixes = sorted(((r.rstrip(os.sep) + os.sep, kind) for kind, r in self.roots.items() if r),
                                key=lambda p: -len(p[0]))
        self.paths = []
        seen = set()
        unknown = 0
        for kind, rel in entries:
            root = self.roots.get(kind)
            if root is None:
                unknown += 1
                continue
            path = os.path.join(root, rel)
            if path not in seen:
                seen.add(path)
                self.paths.append(path)
        self.listed = frozenset(self.paths)
        self.threads = max(1, min(threads, len(self.paths))) if self.paths else 0
        self.wait_s = wait_s
        #: path -> bytes (a reader stores it; get_data pops it) or _MISSING (the read failed).
        self.data: dict[str, object] = {}
        self.stopped = False
        self.record = record
        #: Every path the shadowed get_data was asked for, in order (record=True only).
        self.reads: list[str] = []
        self._reader_idents: set[int] = set()
        self._main_ident = threading.get_ident()
        self._shadowed: list[object] = []
        self.shadowed = 0
        self._lock = threading.Lock()
        self._installed = False
        self._loader_types: tuple = ()
        self._path_finder = None
        self.read_bytes = 0
        self.read_files = 0
        self.read_failed = 0
        self.readers_done = 0
        self.done_at = None
        self.started_at = None
        #: stock_nvcsw: the creating (main) thread's voluntary switches inside stock get_data reads
        #: of bytecode (the targeted waits this does not remove); stock_reads: how many.
        self.counters = {"unknown_kind": unknown, "thread_finds": 0, "late_waits": 0,
                         "late_wait_s": 0.0, "stock_reads": 0, "stock_nvcsw": 0}
        #: (outcome, root kind) -> count; outcome hit / late_hit (arrived during the bounded wait) /
        #: late (listed, not arrived in time) / absent (listed, read failed) / unlisted (under a root,
        #: not listed); kind "other" outside the roots.
        self.served: dict[tuple[str, str], int] = {}
        self.fallbacks: dict[str, int] = {}
        self.installed_at = None
        self.uninstalled_at = None
        self.freed = 0

    # -- reader threads -------------------------------------------------------------------------

    def start(self) -> "ImportPrefetch":
        """Start the reader threads (daemon), each over every `threads`-th manifest path."""
        self.started_at = time.time()
        for k in range(self.threads):
            threading.Thread(target=self._read, args=(self.paths[k::self.threads],),
                             name=f"qs-prefetch-{k}", daemon=True).start()
        return self

    def _read(self, paths: list[str]) -> None:
        self._reader_idents.add(threading.get_ident())
        data = self.data
        nbytes = files = failed = 0
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        for path in paths:
            if self.stopped:
                break
            try:
                fd = os.open(path, flags)
            except OSError:
                data[path] = _MISSING
                failed += 1
                continue
            try:
                blob = os.read(fd, CHUNK)
                if len(blob) == CHUNK:
                    parts = [blob]
                    while True:
                        more = os.read(fd, CHUNK)
                        if not more:
                            break
                        parts.append(more)
                    blob = b"".join(parts)
            except OSError:
                data[path] = _MISSING
                failed += 1
                continue
            finally:
                os.close(fd)
            if self.stopped:
                break
            data[path] = blob
            nbytes += len(blob)
            files += 1
        with self._lock:
            self.read_bytes += nbytes
            self.read_files += files
            self.read_failed += failed
            self.readers_done += 1
            if self.readers_done == self.threads:
                self.done_at = time.time()
        # A later thread may reuse this ident; only a live reader's finder calls count.
        self._reader_idents.discard(threading.get_ident())

    # -- the finder -----------------------------------------------------------------------------

    def install(self) -> "ImportPrefetch":
        """Insert the finder just ahead of the stock PathFinder on sys.meta_path."""
        import importlib.machinery as machinery

        self._path_finder = machinery.PathFinder
        self._loader_types = (machinery.SourceFileLoader, machinery.SourcelessFileLoader)
        meta = sys.meta_path
        index = next((i for i, f in enumerate(meta) if f is machinery.PathFinder), len(meta))
        meta.insert(index, self)
        self._installed = True
        self.installed_at = time.time()
        return self

    def uninstall(self) -> None:
        """Remove the finder and every get_data shadow, stop the readers, free unconsumed bytes."""
        self.stopped = True
        if self._installed:
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            self._installed = False
        for loader in self._shadowed:
            try:
                del loader.get_data
            except AttributeError:
                pass
        self._shadowed = []
        self.freed = sum(len(v) for v in list(self.data.values()) if v is not _MISSING)
        self.data.clear()
        self.uninstalled_at = time.time()

    def find_spec(self, name, path=None, target=None):
        if threading.get_ident() in self._reader_idents:
            self.counters["thread_finds"] += 1
        spec = self._path_finder.find_spec(name, path, target)
        loader = getattr(spec, "loader", None)
        if type(loader) in self._loader_types and "get_data" not in loader.__dict__:
            stock = loader.get_data
            loader.get_data = lambda data_path, _stock=stock: self._get_data(data_path, _stock)
            self._shadowed.append(loader)
            self.shadowed += 1
        return spec

    def _kind(self, path: str) -> str:
        for prefix, kind in self._prefixes:
            if path.startswith(prefix):
                return kind
        return "other"

    def _count(self, outcome: str, path: str) -> None:
        key = (outcome, self._kind(path))
        self.served[key] = self.served.get(key, 0) + 1

    def _get_data(self, path, stock):
        if self.record:
            self.reads.append(path)
        if path in self.listed:
            data = self.data
            blob = data.pop(path, None)
            outcome = "hit"
            if blob is None and not self.stopped:
                waited = time.perf_counter()
                deadline = waited + self.wait_s
                while blob is None and time.perf_counter() < deadline:
                    time.sleep(0.0005)
                    blob = data.pop(path, None)
                self.counters["late_wait_s"] += time.perf_counter() - waited
                self.counters["late_waits"] += 1
                outcome = "late_hit"
            if blob is None:
                self._count("late", path)
            elif blob is _MISSING:
                self._count("absent", path)
            else:
                self._count(outcome, path)
                return blob
        else:
            self._count("unlisted", path)
        main = path.endswith(".pyc") and threading.get_ident() == self._main_ident
        if main:
            nvcsw = resource.getrusage(resource.RUSAGE_THREAD).ru_nvcsw
        try:
            return stock(path)
        except Exception as exc:
            key = type(exc).__name__
            self.fallbacks[key] = self.fallbacks.get(key, 0) + 1
            raise
        finally:
            if main:
                self.counters["stock_reads"] += 1
                self.counters["stock_nvcsw"] += resource.getrusage(resource.RUSAGE_THREAD).ru_nvcsw - nvcsw

    # -- report ---------------------------------------------------------------------------------

    def coverage(self, kinds: tuple = ("stdlib", "purelib", "platlib")) -> tuple[int, int]:
        """(hits, get_data calls) under the given root kinds; a late hit is a hit."""
        hits = sum(n for (o, k), n in self.served.items() if o in ("hit", "late_hit") and k in kinds)
        total = sum(n for (o, k), n in self.served.items() if k in kinds)
        return hits, total

    def summary(self) -> str:
        def rel(t):
            return f"{t - self.t0:+.3f}" if t is not None else "-"

        def by(outcome):
            parts = [f"{k}:{n}" for (o, k), n in sorted(self.served.items()) if o == outcome]
            return ",".join(parts) or "0"

        hits, total = self.coverage()
        return (
            f"manifest={len(self.paths)} threads={self.threads} wait_ms={self.wait_s * 1e3:.0f} "
            f"started={rel(self.started_at)} done={rel(self.done_at)} files={self.read_files} "
            f"bytes={self.read_bytes} read_failed={self.read_failed} hits={by('hit')} "
            f"late_hits={by('late_hit')} late={by('late')} absent={by('absent')} "
            f"unlisted={by('unlisted')} coverage={hits}/{total}"
            f"={hits / total if total else float('nan'):.3f} late_waits={self.counters['late_waits']}"
            f"/{self.counters['late_wait_s']:.3f}s stock_reads={self.counters['stock_reads']} "
            f"stock_nvcsw={self.counters['stock_nvcsw']} fallbacks="
            f"{','.join(f'{k}:{n}' for k, n in self.fallbacks.items()) or '-'} "
            f"thread_finds={self.counters['thread_finds']} unknown_kind={self.counters['unknown_kind']} "
            f"installed={rel(self.installed_at)} uninstalled={rel(self.uninstalled_at)} "
            f"shadowed={self.shadowed} freed={self.freed} "
            f"on_meta_path={int(self in sys.meta_path)}"
        )
