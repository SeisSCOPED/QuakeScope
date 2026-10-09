#!/usr/bin/env python
"""
Export the 2025 DocumentDB pick catalogue to the 2026 Parquet layout.

The 2025 campaign wrote its picks into a DocumentDB (MongoDB-compatible)
database through ``SeisBenchDatabase`` (``sb_catalog/src/mongo_db.py``). The
2026 campaigns write Parquet on S3 through ``ParquetPickWriter``
(``sb_catalog/src/parquet_writer.py``). This script reads the first and writes
the second, so the two catalogues can be read with the same code. The
specification it implements is ``docs/2025_documentdb_to_parquet.md``.

DocumentDB accepts connections only from inside its VPC, so this has to run on
a machine there. It only ever reads from the database.

Layout written under ``--out``::

    picks/network=CI/year=2019/month=07/<job>[-NNN].parquet   PICK_SCHEMA
    manifests/<job>.json        one per network: files written + picks_record
    runs/<rid>.json             one per sb_runs document
    stations.parquet            the stations collection
    _export/<prefix>/...        checkpoints; not part of the catalogue

``<job>`` is ``<prefix>-<NET>``, e.g. ``docdb2025-CI``. The unit of work is
the network: its stations are streamed one at a time over the ``pick_idx``
index, rows are staged on local disk by (year, month), and each month
partition is then sorted, de-duplicated and written. A crash costs at most the
network in progress, and less if ``--staging-dir`` survives the crash: a
network whose streaming finished is not streamed again, and a partition whose
files were written is not written again.

Usage::

    export QS_MONGO_URI='mongodb://user:pass@host:27017/?tls=true&tlsCAFile=global-bundle.pem&retryWrites=false'
    python scripts/export_documentdb_to_parquet.py --database earthscope --dry-run
    python scripts/export_documentdb_to_parquet.py --database earthscope \\
        --out s3://<bucket>/quakescope2025 --staging-dir /data/staging
    python scripts/export_documentdb_to_parquet.py --database earthscope \\
        --out s3://<bucket>/quakescope2025 --networks CI,NC

``--dry-run`` only counts. It writes nothing and prints, per network, the
stations, picks, ``picks_record`` documents and the expected Parquet size, plus
the indexes and the field types of a sample of documents in every collection.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # One schema for both catalogues. Importing it rather than restating it is
    # what stops the two drifting apart.
    from sb_catalog.src.parquet_writer import PICK_SCHEMA
except ImportError:  # pragma: no cover - only without s3fs on the export host
    # A verbatim copy, for a host that has pyarrow but not the rest of the
    # repo's dependencies. tests/test_export_documentdb.py fails if it drifts.
    PICK_SCHEMA = pa.schema(
        [
            ("tid", pa.string()),
            ("cha", pa.string()),
            ("pha", pa.string()),
            ("start", pa.timestamp("ms")),
            ("peak", pa.timestamp("ms")),
            ("end", pa.timestamp("ms")),
            ("conf", pa.float32()),
            ("amp", pa.float32()),
            ("amp_vel", pa.float32()),
            ("rid", pa.string()),
        ]
    )

logger = logging.getLogger("export")

# The unique index the 2025 writer created on `picks` (mongo_db.py:55). It is
# also the de-duplication key here, and the sort order inside every file.
PICK_KEY = ("tid", "cha", "pha", "peak")
PICK_PROJECTION = {
    "_id": 0, "tid": 1, "cha": 1, "pha": 1, "start": 1, "peak": 1, "end": 1,
    "conf": 1, "amp": 1, "amp_vel": 1, "rid": 1,
}
RECORD_PROJECTION = {
    "_id": 0, "tid": 1, "cha": 1, "yr": 1, "doy": 1, "npks": 1, "nclfs": 1,
    "rid": 1,
}

# Bytes per pick in Parquet, zstd + dictionary. 35 is the encoding test in
# docs/rerun_2026/archive/12_output_storage.md; 28.6 is what the published
# western catalogue measures (42.0 GB / 1.47 B picks, docs/data_access.md).
BYTES_PER_PICK = (28.6, 35.0)


# ------------------------------------------------------------ transformations
#
# Pure functions, no database. These are the whole of the 2025 -> 2026 mapping
# and are what the unit test exercises.

def network_of(station_id: Optional[str]) -> str:
    """Network code from a NET.STA.LOC id. Same rule as parquet_writer."""
    if not station_id or "." not in station_id:
        return "unknown"
    return station_id.split(".")[0]


def to_naive_utc(value: Any) -> Optional[datetime.datetime]:
    """A BSON date as the naive UTC datetime Parquet's timestamp('ms') expects.

    pymongo returns naive UTC by default and aware UTC with ``tz_aware=True``;
    both arrive here, and both must land on the same instant.
    """
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value
    raise TypeError(f"expected a datetime, got {type(value).__name__}: {value!r}")


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _id_str(value: Any) -> str:
    """An ObjectId (2025) or a string (anything later) as a string."""
    return "" if value is None else str(value)


def pick_doc_to_row(doc: dict) -> dict:
    """One `picks` document as one PICK_SCHEMA row.

    The fields are the ones picker.py wrote in 2025; nothing is renamed. What
    changes is representation: BSON date -> naive UTC, double -> float32 (on
    write), ObjectId -> 24-character hex string. ``amp_vel`` did not exist in
    2025 and is null unless a document carries it.
    """
    return {
        "tid": doc.get("tid"),
        "cha": doc.get("cha"),
        "pha": doc.get("pha"),
        "start": to_naive_utc(doc.get("start")),
        "peak": to_naive_utc(doc.get("peak")),
        "end": to_naive_utc(doc.get("end")),
        "conf": _float_or_none(doc.get("conf")),
        "amp": _float_or_none(doc.get("amp")),
        "amp_vel": _float_or_none(doc.get("amp_vel")),
        "rid": _id_str(doc.get("rid")),
    }


def partition_of(row: dict) -> tuple[str, int, int]:
    """(network, year, month) of a row, from its tid and its peak time.

    The 2026 writer partitions by the day it processed, which a 2025 document
    does not record. The peak is the only time it has; the two differ only for
    a pick whose peak falls past midnight of its processing day at a month
    boundary.
    """
    peak = row["peak"]
    return network_of(row["tid"]), peak.year, peak.month


def run_doc_to_record(doc: dict) -> dict:
    """One `sb_runs` document as a 2026 ``runs/<run_id>.json`` record.

    The 2026 worker stringifies every value (worker.py:249) and stamps
    ``created`` as ISO 8601 UTC (s3_state.py:299); this does the same, so a
    reader cannot tell which catalogue a record came from except by
    ``source``. Any field beyond the seven picker.py wrote is carried as is.
    """
    created = doc.get("timestamp")
    if isinstance(created, datetime.datetime):
        if created.tzinfo is None:
            created = created.replace(tzinfo=datetime.timezone.utc)
        created = created.astimezone(datetime.timezone.utc).isoformat()
    record = {"run_id": _id_str(doc.get("_id")), "created": created}
    for k, v in doc.items():
        if k in ("_id", "timestamp"):
            continue
        record[k] = None if v is None else str(v)
    record["source"] = "documentdb-2025"
    return record


def record_doc_to_manifest(doc: dict) -> dict:
    """One `picks_record` document as one manifest ``records`` entry."""
    def _int(v):
        return None if v is None else int(v)
    return {
        "tid": doc.get("tid"),
        "cha": doc.get("cha") or "",
        "yr": _int(doc.get("yr")),
        "doy": _int(doc.get("doy")),
        "npks": _int(doc.get("npks")),
        "nclfs": _int(doc.get("nclfs")),
        "rid": _id_str(doc.get("rid")),
    }


def sort_and_dedupe(table: pa.Table) -> tuple[pa.Table, int]:
    """Sort by the unique key and drop exact key duplicates; keep the first.

    The database's unique index (mongo_db.py:55) should mean there is nothing
    to drop, so a non-zero count is a finding, not routine. Sorting on the key
    and then ``rid`` makes "first" deterministic, which is what lets a re-run
    write byte-identical files.
    """
    n = table.num_rows
    if n == 0:
        return table, 0
    table = table.sort_by([(c, "ascending") for c in PICK_KEY] + [("rid", "ascending")])
    same = np.ones(n - 1, dtype=bool)
    for c in PICK_KEY:
        col = table.column(c).to_numpy(zero_copy_only=False)
        same &= col[1:] == col[:-1]
    keep = np.ones(n, dtype=bool)
    keep[1:] = ~same
    dropped = int(n - keep.sum())
    if dropped:
        table = table.filter(pa.array(keep))
    return table, dropped


def file_slices(n_rows: int, rows_per_file: int) -> list[tuple[int, int]]:
    """(offset, length) of each output file in a partition of n_rows."""
    if n_rows <= 0:
        return []
    n_files = math.ceil(n_rows / rows_per_file)
    # Even split, so the last file is not a runt.
    base, extra = divmod(n_rows, n_files)
    out, offset = [], 0
    for i in range(n_files):
        length = base + (1 if i < extra else 0)
        out.append((offset, length))
        offset += length
    return out


def file_name(job_id: str, seq: int) -> str:
    """Same naming as ParquetPickWriter._suffix: <job>.parquet, <job>-001.parquet."""
    return f"{job_id}-{seq:03d}.parquet" if seq else f"{job_id}.parquet"


def partition_dir(root: str, key: tuple[str, int, int]) -> str:
    network, year, month = key
    return f"{root}/picks/network={network}/year={year:04d}/month={month:02d}"


# ------------------------------------------------------------------- output

class Output:
    """The catalogue root, local or s3://, through fsspec."""

    def __init__(self, root: str, storage_options: Optional[dict] = None) -> None:
        self.root = root.rstrip("/")
        self.fs = fsspec.filesystem(
            fsspec.utils.get_protocol(self.root), **(storage_options or {})
        )

    def path(self, *parts: str) -> str:
        return "/".join([self.root, *parts])

    def _ensure_parent(self, path: str) -> None:
        # S3 has no directories; a local root does.
        try:
            self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        except Exception:
            pass

    def write_json(self, path: str, obj: Any) -> None:
        self._ensure_parent(path)
        with self.fs.open(path, "wb") as fh:
            fh.write(json.dumps(obj, default=str).encode())

    def read_json(self, path: str) -> Optional[dict]:
        if not self.fs.exists(path):
            return None
        with self.fs.open(path, "rb") as fh:
            return json.loads(fh.read())

    def write_table(self, path: str, table: pa.Table) -> None:
        self._ensure_parent(path)
        # Match the 2026 writer exactly (parquet_writer.py:340). The
        # compactor learned that defaults re-encode to a larger catalogue.
        with self.fs.open(path, "wb") as fh:
            pq.write_table(table, fh, compression="zstd", use_dictionary=True)

    def remove_job_files(self, directory: str, job_id: str) -> None:
        """Delete this job's files in one partition before rewriting it, so a
        re-run that writes fewer files cannot leave a stale one behind."""
        try:
            names = self.fs.glob(f"{directory}/{job_id}*.parquet")
        except FileNotFoundError:
            return
        for name in names:
            base = name.rsplit("/", 1)[-1]
            if base == f"{job_id}.parquet" or (
                base.startswith(f"{job_id}-") and base[len(job_id) + 1:-8].isdigit()
            ):
                self.fs.rm(name)


# ------------------------------------------------------------------ staging

class NetworkStager:
    """Rows of one network, staged on local disk by (year, month).

    A network is streamed station by station, so its months arrive
    interleaved; holding them all in memory does not bound for the large
    networks. Rows are buffered per month and spilled as small Parquet parts;
    each month is read back whole, once, when it is finalised.
    """

    def __init__(self, directory: str, spill_rows: int, max_buffered_rows: int) -> None:
        self.dir = directory
        self.spill_rows = spill_rows
        self.max_buffered_rows = max_buffered_rows
        self._buf: dict[tuple, list] = defaultdict(list)
        self._buffered = 0
        self._parts: dict[tuple, int] = defaultdict(int)
        os.makedirs(self.dir, exist_ok=True)

    def add(self, row: dict) -> None:
        key = partition_of(row)
        self._buf[key].append(row)
        self._buffered += 1
        if len(self._buf[key]) >= self.spill_rows:
            self._spill(key)
        elif self._buffered >= self.max_buffered_rows:
            for k in list(self._buf):
                self._spill(k)

    def _part_dir(self, key: tuple) -> str:
        network, year, month = key
        return os.path.join(self.dir, f"year={year:04d}", f"month={month:02d}")

    def _spill(self, key: tuple) -> None:
        rows = self._buf.pop(key, None)
        if not rows:
            return
        d = self._part_dir(key)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"part-{self._parts[key]:06d}.parquet")
        # Uncompressed: it is read back once, minutes later, on the same disk.
        pq.write_table(pa.Table.from_pylist(rows, schema=PICK_SCHEMA), path,
                       compression="none")
        self._parts[key] += 1
        self._buffered -= len(rows)

    def close(self) -> None:
        for k in list(self._buf):
            self._spill(k)

    def partitions(self, network: str) -> list[tuple[str, int, int]]:
        """Every (network, year, month) with staged parts, from the disk -
        so it also works after a restart that found the staging intact."""
        out = []
        for ydir in sorted(os.listdir(self.dir)):
            if not ydir.startswith("year="):
                continue
            for mdir in sorted(os.listdir(os.path.join(self.dir, ydir))):
                if mdir.startswith("month="):
                    out.append((network, int(ydir[5:]), int(mdir[6:])))
        return out

    def read(self, key: tuple) -> pa.Table:
        d = self._part_dir(key)
        parts = sorted(p for p in os.listdir(d) if p.endswith(".parquet"))
        if not parts:
            return PICK_SCHEMA.empty_table()
        return pa.concat_tables(
            [pq.read_table(os.path.join(d, p), schema=PICK_SCHEMA) for p in parts]
        )


# ------------------------------------------------------------------ database

def station_ids(db: Any, sources: Iterable[str]) -> dict[str, list[str]]:
    """Station ids grouped by network.

    ``stations`` is what the 2025 campaign was planned from; ``picks_record``
    is what it actually wrote. Their union is the safest list, and the
    global count check after the export catches any tid neither holds.
    """
    ids: set[str] = set()
    for source in sources:
        field = "id" if source == "stations" else "tid"
        ids |= {i for i in db[source].distinct(field) if i}
    by_net: dict[str, list[str]] = defaultdict(list)
    for tid in sorted(ids):
        by_net[network_of(tid)].append(tid)
    return dict(by_net)


def _find(coll: Any, filt: dict, projection: dict, batch_size: int,
          hint: Optional[str]) -> Any:
    cursor = coll.find(filt, projection, batch_size=batch_size)
    if hint:
        cursor = cursor.hint(hint)
    return cursor


def describe_collections(db: Any, sample: int = 200) -> dict:
    """Indexes, approximate counts and field types per collection.

    This is the part of the 2025 schema the code cannot establish - what was
    actually stored, as opposed to what the code meant to store. Read from a
    random sample; ``$sample`` is supported by DocumentDB.
    """
    out = {}
    for name in sorted(db.list_collection_names()):
        coll = db[name]
        types: dict[str, Counter] = defaultdict(Counter)
        try:
            docs = list(coll.aggregate([{"$sample": {"size": sample}}]))
        except Exception:
            docs = list(coll.find({}).limit(sample))
        for doc in docs:
            for k, v in doc.items():
                types[k][type(v).__name__] += 1
        out[name] = {
            "estimated_count": coll.estimated_document_count(),
            "indexes": {k: v.get("key") for k, v in coll.index_information().items()},
            "field_types": {k: dict(c) for k, c in sorted(types.items())},
            "sampled": len(docs),
        }
    return out


# ------------------------------------------------------------------- export

class Exporter:
    def __init__(self, db: Any, out: Output, prefix: str, staging_root: str,
                 rows_per_file: int = 1_000_000, batch_size: int = 10_000,
                 spill_rows: int = 200_000, max_buffered_rows: int = 2_000_000,
                 verify_counts: bool = True, spot_checks: int = 3,
                 database_name: str = "") -> None:
        self.db = db
        self.out = out
        self.prefix = prefix
        self.staging_root = staging_root
        self.rows_per_file = rows_per_file
        self.batch_size = batch_size
        self.spill_rows = spill_rows
        self.max_buffered_rows = max_buffered_rows
        self.verify_counts = verify_counts
        self.spot_checks = spot_checks
        self.database_name = database_name
        self.rng = random.Random(0)
        indexes = db["picks"].index_information()
        # Only hint an index that exists: a hint to a missing one is an error.
        self.pick_hint = "pick_idx" if "pick_idx" in indexes else None
        rindexes = db["picks_record"].index_information()
        self.record_hint = "picks_record_idx" if "picks_record_idx" in rindexes else None

    # -- paths
    def job_id(self, network: str) -> str:
        return f"{self.prefix}-{network}"

    def _ck_path(self, network: str, kind: str) -> str:
        return self.out.path("_export", self.prefix, f"network={network}.{kind}.json")

    def _staging(self, network: str) -> str:
        return os.path.join(self.staging_root, f"network={network}")

    # -- the unit of work
    def export_network(self, network: str, tids: list[str]) -> dict:
        done = self.out.read_json(self._ck_path(network, "done"))
        if done:
            logger.info(f"{network}: already exported, skipping")
            return done

        staging = self._staging(network)
        staged_marker = os.path.join(staging, "_STAGED.json")
        if os.path.exists(staged_marker):
            # Streaming finished before the crash; the rows are on disk.
            with open(staged_marker) as fh:
                stream_stats = json.load(fh)
            logger.info(f"{network}: reusing staged rows from {staging}")
            stager = NetworkStager(staging, self.spill_rows, self.max_buffered_rows)
        else:
            shutil.rmtree(staging, ignore_errors=True)
            stager = NetworkStager(staging, self.spill_rows, self.max_buffered_rows)
            stream_stats = self._stream(network, tids, stager)
            with open(staged_marker, "w") as fh:
                json.dump(stream_stats, fh)

        progress = self.out.read_json(self._ck_path(network, "progress")) or {"partitions": {}}
        job_id = self.job_id(network)
        for key in stager.partitions(network):
            label = f"{key[1]:04d}-{key[2]:02d}"
            if label in progress["partitions"]:
                continue
            progress["partitions"][label] = self._finalise_partition(key, stager, job_id)
            # Per-partition checkpoint: a crash in the next month does not
            # rewrite this one.
            self.out.write_json(self._ck_path(network, "progress"), progress)

        parts = progress["partitions"]
        files = [f for p in parts.values() for f in p["files"]]
        rows_written = sum(p["rows"] for p in parts.values())
        dropped = sum(p["duplicates_dropped"] for p in parts.values())
        records = self._records(network, tids)
        manifest = self._manifest(job_id, files, records, rows_written)
        self.out.write_json(self.out.path("manifests", f"{job_id}.json"), manifest)

        summary = {
            "network": network,
            "job_id": job_id,
            "stations": len(tids),
            "rows_streamed": stream_stats["rows_streamed"],
            "rows_written": rows_written,
            "duplicates_dropped": dropped,
            "count_mismatches": stream_stats["count_mismatches"],
            "rows_by_year": _rows_by_year(parts),
            "picks_record_docs": len(records),
            "picks_record_npks_by_year": _npks_by_year(records),
            "spot_check_failures": [f for p in parts.values() for f in p["spot_check_failures"]],
            "files": len(files),
            "finished_at": _utcnow(),
        }
        if rows_written + dropped != stream_stats["rows_streamed"]:
            # Rows went in and did not come out. Do not mark done.
            raise RuntimeError(f"{network}: streamed {stream_stats['rows_streamed']} rows "
                               f"but wrote {rows_written} + dropped {dropped}")
        self.out.write_json(self._ck_path(network, "done"), summary)
        shutil.rmtree(staging, ignore_errors=True)
        logger.info(f"{network}: {rows_written:,} picks in {len(files)} files, "
                    f"{dropped} duplicates dropped, "
                    f"{len(stream_stats['count_mismatches'])} count mismatches")
        return summary

    def _stream(self, network: str, tids: list[str], stager: NetworkStager) -> dict:
        coll = self.db["picks"]
        total, mismatches = 0, []
        for i, tid in enumerate(tids):
            n = 0
            for doc in _find(coll, {"tid": tid}, PICK_PROJECTION, self.batch_size, self.pick_hint):
                row = pick_doc_to_row(doc)
                if row["peak"] is None:
                    raise ValueError(f"{tid}: a pick without a peak time: {doc!r}")
                stager.add(row)
                n += 1
            if self.verify_counts:
                # A cursor that ends early looks exactly like a station with
                # fewer picks. The index answers this cheaply.
                expected = coll.count_documents({"tid": tid})
                if expected != n:
                    mismatches.append({"tid": tid, "streamed": n, "count": expected})
                    logger.error(f"{tid}: streamed {n} but the collection counts {expected}")
            total += n
            if (i + 1) % 100 == 0:
                logger.info(f"{network}: {i + 1}/{len(tids)} stations, {total:,} picks")
        stager.close()
        if mismatches:
            # Better to stop here than to write and mark done a network that
            # is known to be incomplete.
            raise RuntimeError(f"{network}: {len(mismatches)} station(s) streamed a "
                               f"different count than the index reports: {mismatches[:3]}")
        return {"rows_streamed": total, "count_mismatches": mismatches}

    def _finalise_partition(self, key: tuple, stager: NetworkStager, job_id: str) -> dict:
        table = stager.read(key)
        table, dropped = sort_and_dedupe(table)
        directory = partition_dir(self.out.root, key)
        self.out.remove_job_files(directory, job_id)
        files = []
        for seq, (offset, length) in enumerate(file_slices(table.num_rows, self.rows_per_file)):
            path = f"{directory}/{file_name(job_id, seq)}"
            self.out.write_table(path, table.slice(offset, length))
            files.append({"kind": "picks", "path": path, "rows": length,
                          "network": key[0], "year": key[1], "month": key[2]})
        failures = self._spot_check(table)
        logger.info(f"{key}: {table.num_rows:,} rows, {len(files)} file(s)")
        return {"rows": table.num_rows, "duplicates_dropped": dropped,
                "files": files, "spot_check_failures": failures}

    def _spot_check(self, table: pa.Table) -> list[dict]:
        """Look a few written rows up in the database by their unique key and
        compare every field. Catches a transformation error that counts
        cannot - a shifted time, a truncated value, a swapped column."""
        if not self.spot_checks or table.num_rows == 0:
            return []
        failures = []
        for i in self.rng.sample(range(table.num_rows), min(self.spot_checks, table.num_rows)):
            row = table.slice(i, 1).to_pylist()[0]
            doc = self.db["picks"].find_one({c: row[c] for c in PICK_KEY}, PICK_PROJECTION)
            if doc is None:
                failures.append({"key": {c: str(row[c]) for c in PICK_KEY}, "error": "not found"})
                continue
            expected = pick_doc_to_row(doc)
            for c in ("start", "end", "rid", "amp_vel"):
                if expected[c] != row[c]:
                    failures.append({"key": str(row["peak"]), "field": c,
                                     "db": str(expected[c]), "parquet": str(row[c])})
            for c in ("conf", "amp"):
                a, b = expected[c], row[c]
                if a is None or b is None or (math.isnan(a) and math.isnan(b)):
                    if (a is None) != (b is None):
                        failures.append({"key": str(row["peak"]), "field": c, "db": a, "parquet": b})
                    continue
                if not math.isclose(np.float32(a), b, rel_tol=1e-6, abs_tol=0.0):
                    failures.append({"key": str(row["peak"]), "field": c, "db": a, "parquet": b})
        return failures

    def _records(self, network: str, tids: list[str]) -> list[dict]:
        coll = self.db["picks_record"]
        records = []
        for tid in tids:
            for doc in _find(coll, {"tid": tid}, RECORD_PROJECTION, self.batch_size, self.record_hint):
                records.append(record_doc_to_manifest(doc))
        records.sort(key=lambda r: (r["tid"], r["yr"] or 0, r["doy"] or 0, r["cha"]))
        return records

    def _manifest(self, job_id: str, files: list[dict], records: list[dict],
                  n_picks: int) -> dict:
        """Same keys as ParquetPickWriter.close() (parquet_writer.py:467).

        ``run_id`` is null: one export job carries the picks of many 2025
        runs, and every record says which in its own ``rid``. ``outcomes`` is
        absent because 2025 never recorded them; ``export`` says where the
        manifest came from.
        """
        return {
            "job_id": job_id,
            "run_id": None,
            "n_picks": n_picks,
            "n_classifies": 0,
            "station_days": len(records),
            "written_at": _utcnow(),
            "files": files,
            "records": records,
            "export": {
                "source": "documentdb",
                "database": self.database_name,
                "collections": ["picks", "picks_record"],
                "exporter": "scripts/export_documentdb_to_parquet.py",
                "partitioned_by": "peak time (UTC)",
            },
        }

    # -- the small collections
    def export_runs(self) -> int:
        """runs/<rid>.json for every sb_runs document not already written."""
        existing = set()
        runs_dir = self.out.path("runs")
        if self.out.fs.exists(runs_dir):
            existing = {p.rsplit("/", 1)[-1] for p in self.out.fs.ls(runs_dir, detail=False)}
        n = 0
        for doc in self.db["sb_runs"].find({}):
            record = run_doc_to_record(doc)
            name = f"{record['run_id']}.json"
            if name in existing:
                continue
            self.out.write_json(self.out.path("runs", name), record)
            n += 1
        logger.info(f"Wrote {n} run records ({len(existing)} already present)")
        return n

    def export_stations(self) -> int:
        """stations.parquet from the stations collection, through the same
        normalisation the 2026 catalogues get (s3_state.write_stations)."""
        import pandas as pd

        df = pd.DataFrame(list(self.db["stations"].find({}, {"_id": 0})))
        try:
            from sb_catalog.src.s3_state import prepare_station_dates
            from sb_catalog.src.utils import normalize_station_codes

            df = prepare_station_dates(normalize_station_codes(df))
        except ImportError:
            # Standalone run (the exporter fetched on its own, as EarthScope or
            # a bare container would run it). The 2025 collection stores some
            # numeric station codes as integers ("001" as 1) next to strings,
            # which Arrow refuses as one column - the first full export died
            # on exactly that. Same rule as utils.normalize_station_codes:
            # identifiers are text, and the codes are rebuilt from `id`.
            logger.warning("sb_catalog not importable: codes rebuilt from id; "
                           "start_date/end_date left as stored")
            df = normalize_codes_standalone(df)
        path = self.out.path("stations.parquet")
        self.out._ensure_parent(path)
        with self.out.fs.open(path, "wb") as fh:
            df.to_parquet(fh, index=False)
        logger.info(f"Wrote {len(df)} stations to {path}")
        return len(df)


def normalize_codes_standalone(df):
    """Text identifiers, codes from `id` (NET.STA.LOC); a stand-in for
    sb_catalog.src.utils.normalize_station_codes when that is not importable."""
    df = df.copy()
    for c in ("id", "network_code", "station_code", "location_code", "channels"):
        if c in df.columns:
            df[c] = df[c].where(df[c].notna(), "").astype(str)
    if "id" in df.columns:
        parts = df["id"].str.split(".", n=2, expand=True)
        if parts.shape[1] == 3:
            df["network_code"], df["station_code"], df["location_code"] = parts[0], parts[1], parts[2]
    for c in ("start_date", "end_date"):
        if c in df.columns:
            df[c] = df[c].where(df[c].notna(), "").astype(str)
    return df


def _rows_by_year(parts: dict) -> dict:
    out: dict[str, int] = defaultdict(int)
    for label, p in parts.items():
        out[label[:4]] += p["rows"]
    return dict(sorted(out.items()))


def _npks_by_year(records: list[dict]) -> dict:
    out: dict[str, int] = defaultdict(int)
    for r in records:
        out[str(r["yr"])] += r["npks"] or 0
    return dict(sorted(out.items()))


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ----------------------------------------------------------------- dry run

def dry_run(db: Any, by_net: dict[str, list[str]], sample: int) -> dict:
    """Counts only. Reads indexes and counts; writes nothing anywhere."""
    report = {"collections": describe_collections(db, sample), "networks": {}}
    total = 0
    for network, tids in sorted(by_net.items()):
        picks = sum(db["picks"].count_documents({"tid": t}) for t in tids)
        recs = sum(db["picks_record"].count_documents({"tid": t}) for t in tids)
        total += picks
        report["networks"][network] = {
            "stations": len(tids), "picks": picks, "picks_record_docs": recs,
            "parquet_bytes_est": [int(picks * b) for b in BYTES_PER_PICK],
        }
        logger.info(f"{network:>4}: {len(tids):>6} stations {picks:>14,} picks "
                    f"{recs:>11,} records ~{picks * BYTES_PER_PICK[1] / 1e9:,.1f} GB")
    report["picks_total_over_listed_stations"] = total
    report["picks_estimated_document_count"] = report["collections"].get(
        "picks", {}).get("estimated_count")
    report["parquet_bytes_est"] = [int(total * b) for b in BYTES_PER_PICK]
    return report


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mongo-uri", default=os.environ.get("QS_MONGO_URI"),
                    help="Connection string. Defaults to $QS_MONGO_URI, so the "
                         "password does not land in shell history.")
    ap.add_argument("--database", required=True, help="2025 database name.")
    ap.add_argument("--out", help="Catalogue root, s3://... or a local path.")
    ap.add_argument("--prefix", default="docdb2025",
                    help="Job-id prefix; files are <prefix>-<NET>[-NNN].parquet.")
    ap.add_argument("--networks", default="", help="Comma-separated; default all.")
    ap.add_argument("--station-source", default="stations,picks_record",
                    help="Collections the station list is drawn from.")
    ap.add_argument("--staging-dir", default=None,
                    help="Local disk for staging. Keep it across a restart to "
                         "resume a network without re-streaming it.")
    ap.add_argument("--rows-per-file", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--spot-checks", type=int, default=3,
                    help="Rows per partition looked up again in the database.")
    ap.add_argument("--no-verify-counts", action="store_true")
    ap.add_argument("--skip-runs", action="store_true")
    ap.add_argument("--skip-stations", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Count only; write nothing.")
    ap.add_argument("--report", default=None, help="Local JSON path for the dry-run report.")
    ap.add_argument("--sample", type=int, default=200,
                    help="Documents sampled per collection to report field types.")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    if not a.mongo_uri:
        ap.error("no --mongo-uri and no $QS_MONGO_URI")
    if not a.dry_run and not a.out:
        ap.error("--out is required unless --dry-run")

    import pymongo

    # Read-only. retryWrites=False because DocumentDB rejects it; secondary
    # reads keep a long export off the primary if the cluster has a replica.
    client = pymongo.MongoClient(a.mongo_uri, retryWrites=False,
                                 readPreference="secondaryPreferred")
    db = client[a.database]

    by_net = station_ids(db, [s.strip() for s in a.station_source.split(",") if s.strip()])
    if a.networks:
        wanted = {n.strip() for n in a.networks.split(",") if n.strip()}
        by_net = {n: t for n, t in by_net.items() if n in wanted}
    logger.info(f"{len(by_net)} networks, {sum(map(len, by_net.values())):,} stations")

    if a.dry_run:
        report = dry_run(db, by_net, a.sample)
        text = json.dumps(report, indent=1, default=str)
        if a.report:
            with open(a.report, "w") as fh:
                fh.write(text)
        print(text)
        return

    out = Output(a.out)
    staging = a.staging_dir or tempfile.mkdtemp(prefix="docdb-export-")
    ex = Exporter(db, out, a.prefix, staging, rows_per_file=a.rows_per_file,
                  batch_size=a.batch_size, verify_counts=not a.no_verify_counts,
                  spot_checks=a.spot_checks, database_name=a.database)
    if not a.skip_runs:
        ex.export_runs()
    if not a.skip_stations:
        ex.export_stations()

    summaries = [ex.export_network(n, t) for n, t in sorted(by_net.items())]

    # The one check a per-station loop cannot make: picks whose tid is in
    # neither the stations collection nor picks_record were never queried.
    written = sum(s["rows_written"] + s["duplicates_dropped"] for s in summaries)
    if not a.networks:
        total = db["picks"].count_documents({})
        status = "OK" if total == written else "MISMATCH"
        logger.info(f"{status}: picks collection holds {total:,}; export read {written:,}")
        out.write_json(out.path("_export", a.prefix, "summary.json"), {
            "picks_collection_count": total, "rows_read": written,
            "networks": len(summaries), "finished_at": _utcnow(), "status": status,
        })


if __name__ == "__main__":
    main()
