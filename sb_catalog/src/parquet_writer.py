"""
Parquet output for picking campaigns.

At the 2026 launch's scale — roughly 52 million station-days and order 10^10
picks — a provisioned document store is the wrong container for the picks
themselves. Parquet holds the same rows in about a sixth of the space, needs
nothing running between campaigns, and removes the unique-index maintenance
that otherwise degrades insert throughput as a campaign grows. The reasoning
and the measurements are in ``docs/rerun_2026/12_output_storage.md``.

What this module does *not* do is replace the database. Station metadata and
the resume records stay where they are: they are small, they need point
lookups, and they are exactly what a database is good at. Only the bulk output
moves.

Layout::

    <root>/picks/network=CI/year=2019/month=07/<job>.parquet
    <root>/classifies/network=CI/year=2019/month=07/<job>.parquet
    <root>/manifests/<job>.json

One file per job per (network, year, month) rather than one per station-day.
A Batch job covers 40 stations x 20 days, so a file lands near 50 MB — close to
ideal for Parquet — and a campaign produces tens of thousands of objects rather
than tens of millions. Per-station-day files would be correct and unusable.

Concurrency needs no coordination. Every job writes its own keys, files are
immutable, and a retried job overwrites itself byte for byte, so retries are
idempotent without the ignore-duplicates machinery the database needs.
"""

import datetime
import json
import logging
import os
import uuid
from collections import defaultdict
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

logger = logging.getLogger("picker")

# Explicit schema rather than letting Arrow infer one. Inference across
# thousands of independently written files is how partitions end up with
# incompatible types that only surface when something tries to read them all.
PICK_SCHEMA = pa.schema(
    [
        ("tid", pa.string()),          # NET.STA.LOC
        ("cha", pa.string()),          # band code, e.g. HH
        ("pha", pa.string()),          # P or S
        ("start", pa.timestamp("ms")),
        ("peak", pa.timestamp("ms")),
        ("end", pa.timestamp("ms")),
        ("conf", pa.float32()),
        ("amp", pa.float32()),
        ("amp_raw", pa.float32()),
        ("rid", pa.string()),          # run id, ties picks to their weights
    ]
)

CLASSIFY_SCHEMA = pa.schema(
    [
        ("tid", pa.string()),
        ("cha", pa.string()),
        ("start", pa.timestamp("ms")),
        ("label", pa.string()),
        ("eq", pa.float32()),
        ("px", pa.float32()),
        ("su", pa.float32()),
        ("rid", pa.string()),
    ]
)


def _network_of(station_id: str) -> str:
    """Network code from a NET.STA.LOC id."""
    return station_id.split(".")[0] if "." in station_id else "unknown"


class ParquetPickWriter:
    """Accumulates a job's output in memory and writes it as Parquet on close.

    A job's picks are held until :meth:`close` because the file boundary is the
    job, not the station-day. At 40 stations x 20 days and roughly 1,700 picks
    per station-day that is on the order of a million rows — tens of megabytes
    as Python objects, which is comfortably inside a Fargate task and vastly
    cheaper than writing a file per station-day.

    ``flush_threshold`` bounds that: once a partition exceeds it, the partition
    is written and cleared. Long or unusually productive jobs therefore emit a
    few files per partition instead of one, which costs nothing.
    """

    def __init__(
        self,
        root: str,
        run_id: str,
        job_id: Optional[str] = None,
        compression: str = "zstd",
        flush_threshold: int = 4_000_000,
        storage_options: Optional[dict] = None,
    ) -> None:
        self.root = root.rstrip("/")
        self.run_id = str(run_id)
        self.job_id = job_id or self._infer_job_id()
        self.compression = compression
        self.flush_threshold = flush_threshold
        self.fs = s3fs.S3FileSystem(**(storage_options or {}))

        self._picks: dict[tuple, list] = defaultdict(list)
        self._classifies: dict[tuple, list] = defaultdict(list)
        self._records: list[dict] = []
        self._part_seq: dict[tuple, int] = defaultdict(int)
        self.n_picks = 0
        self.n_classifies = 0

    @staticmethod
    def _infer_job_id() -> str:
        """Prefer the Batch job id so a file can be traced back to its task."""
        for var in ("AWS_BATCH_JOB_ID", "AWS_BATCH_JOB_ATTEMPT_ID", "HOSTNAME"):
            value = os.environ.get(var)
            if value:
                return value.replace("/", "-").replace(":", "-")
        return uuid.uuid4().hex[:16]

    # ------------------------------------------------------------------ write

    def add(
        self,
        picks: Any,
        amplitudes: list[float],
        raw_amplitudes: list[float],
        classifies: list[dict],
        station: str,
        day: datetime.datetime,
        channel: str,
    ) -> None:
        """Buffer one station-day-channel. Mirrors the database writer's call."""
        key = (_network_of(station), day.year, day.month)

        for pick, amp, raw_amp in zip(picks, amplitudes, raw_amplitudes):
            self._picks[key].append(
                {
                    "tid": station,
                    "cha": channel,
                    "pha": pick.phase,
                    "start": pick.start_time.datetime,
                    "peak": pick.peak_time.datetime,
                    "end": pick.end_time.datetime,
                    "conf": float(pick.peak_value),
                    "amp": float(amp),
                    "amp_raw": float(raw_amp),
                    "rid": self.run_id,
                }
            )
        self.n_picks += len(picks)

        for c in classifies:
            self._classifies[key].append(
                {
                    "tid": station,
                    "cha": channel,
                    "start": c["start"].datetime,
                    "label": c.get("label", ""),
                    "eq": float(c["eq"]),
                    "px": float(c["px"]),
                    "su": float(c["su"]),
                    "rid": self.run_id,
                }
            )
        self.n_classifies += len(classifies)

        # One record per station-day-channel, mirroring picks_record. This is
        # what a resume needs, and it is three orders of magnitude smaller than
        # the picks themselves.
        self._records.append(
            {
                "tid": station,
                "cha": channel,
                "yr": day.year,
                "doy": int(day.strftime("%j")),
                "npks": len(picks),
                "nclfs": len(classifies),
                "rid": self.run_id,
            }
        )

        if len(self._picks[key]) >= self.flush_threshold:
            self._write_partition("picks", key, PICK_SCHEMA, self._picks)

    def _ensure_parent(self, path: str) -> None:
        """Create the parent directory where the filesystem has such a concept.

        S3 has no directories and s3fs treats this as a no-op, but the same
        writer is useful against a local path for testing and dry runs.
        """
        parent = path.rsplit("/", 1)[0]
        try:
            self.fs.makedirs(parent, exist_ok=True)
        except Exception:
            pass

    def _partition_path(self, kind: str, key: tuple, suffix: str) -> str:
        network, year, month = key
        return (
            f"{self.root}/{kind}/network={network}/year={year:04d}/"
            f"month={month:02d}/{self.job_id}{suffix}"
        )

    def _write_partition(self, kind: str, key: tuple, schema, buffers) -> None:
        rows = buffers.get(key)
        if not rows:
            return
        seq = self._part_seq[(kind, key)]
        self._part_seq[(kind, key)] += 1
        suffix = f"-{seq:03d}.parquet" if seq else ".parquet"
        path = self._partition_path(kind, key, suffix)

        table = pa.Table.from_pylist(rows, schema=schema)
        self._ensure_parent(path)
        with self.fs.open(path, "wb") as fh:
            pq.write_table(
                table, fh, compression=self.compression, use_dictionary=True
            )
        logger.info(f"Wrote {len(rows):>8} {kind} rows -> {path}")
        buffers[key] = []

    def close(self) -> dict:
        """Write everything still buffered, plus a manifest for the job."""
        for key in list(self._picks):
            self._write_partition("picks", key, PICK_SCHEMA, self._picks)
        for key in list(self._classifies):
            self._write_partition(
                "classifies", key, CLASSIFY_SCHEMA, self._classifies
            )

        summary = {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "n_picks": self.n_picks,
            "n_classifies": self.n_classifies,
            "station_days": len(self._records),
            "written_at": datetime.datetime.utcnow().isoformat() + "Z",
            "records": self._records,
        }
        # One manifest per job, flat rather than partitioned: it describes the
        # whole job, which may span several partitions, and it is what makes a
        # job's coverage auditable afterwards - which station-days it claimed
        # and how much each produced.
        path = f"{self.root}/manifests/{self.job_id}.json"
        self._ensure_parent(path)
        with self.fs.open(path, "wb") as fh:
            fh.write(json.dumps(summary).encode())
        logger.info(
            f"Job {self.job_id}: {self.n_picks} picks, "
            f"{self.n_classifies} classifications, "
            f"{len(self._records)} station-day-channels"
        )
        return summary


class DualWriter:
    """Writes to the database and to Parquet at once.

    For the campaign that validates the change: both paths see identical calls,
    so the row counts can be compared directly. Once Parquet is trusted this
    should be dropped rather than left running, since it pays for both.
    """

    def __init__(self, db_writer, parquet_writer: ParquetPickWriter) -> None:
        self.db_writer = db_writer
        self.parquet_writer = parquet_writer

    def add(self, *args, **kwargs) -> None:
        self.db_writer(*args, **kwargs)
        self.parquet_writer.add(*args, **kwargs)

    def close(self) -> dict:
        return self.parquet_writer.close()
