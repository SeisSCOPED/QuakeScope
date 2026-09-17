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
immutable, and a job that restarts from scratch overwrites itself byte for
byte, so retries are idempotent without the ignore-duplicates machinery the
database needs.

A job that *resumes* from a checkpoint is the one case where that is not
enough. Its writer must continue the earlier attempt's file sequence rather
than restart it - otherwise its first flush lands on ``<job>.parquet`` and
replaces the first attempt's first checkpoint - and its manifest must describe
the whole job, not the last attempt. Both were wrong until 2026-09-16 and cost
about 1.8% of the western campaign's processed station-days, re-picked on
2026-09-17 (``docs/rerun_2026/28_resumed_shard_overwrite.md``). ``prior_done``
is what tells the writer it is resuming.
"""

import datetime
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

import fsspec
import s3fs

from .profiling import stage

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
        ("amp_vel", pa.float32()),
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
    is written and dropped from memory.

    The default was 4,000,000 rows, which no shard ever reached: a busy
    Ridgecrest-rate shard produces about 3.4M picks in total and they are
    split across (network, year, month) partitions, so nothing flushed until
    the shard closed. That made the mid-shard checkpoint unreachable - it
    only runs after a flush - so a preempted Spot worker lost the whole
    shard, up to twelve hours, and nothing appeared in S3 until a shard
    finished. 250,000 rows is roughly 50 MB resident and flushes several
    times per shard.
    is written and cleared. Long or unusually productive jobs therefore emit a
    few files per partition instead of one, which costs nothing.
    """

    def __init__(
        self,
        root: str,
        run_id: str,
        job_id: Optional[str] = None,
        compression: str = "zstd",
        flush_threshold: int = 250_000,
        storage_options: Optional[dict] = None,
        prior_done: Optional[Any] = None,
    ) -> None:
        self.root = root.rstrip("/")
        self.run_id = str(run_id)
        self.job_id = job_id or self._infer_job_id()
        self.compression = compression
        self.flush_threshold = flush_threshold
        # Pick the filesystem from the URI scheme rather than assuming S3, so a
        # local root works for tests and dry runs - which _ensure_parent already
        # claimed to support, but could not, because this was hardcoded.
        if self.root.startswith("s3://"):
            self.fs = s3fs.S3FileSystem(**(storage_options or {}))
        else:
            self.fs = fsspec.filesystem(
                fsspec.utils.get_protocol(self.root), **(storage_options or {})
            )

        self._picks: dict[tuple, list] = defaultdict(list)
        self._classifies: dict[tuple, list] = defaultdict(list)
        self._records: list[dict] = []
        # Next file sequence per (kind, partition). Filled lazily by
        # _next_seq, which starts AFTER whatever an earlier attempt of this
        # job already wrote there - never at zero on faith.
        self._part_seq: dict[tuple, int] = {}
        self.n_picks = 0
        self.n_classifies = 0
        # Object keys written by this job, so readers never have to LIST.
        self._written: list[dict] = []
        # Station-day-channels an earlier attempt of this job wrote and
        # checkpointed, as (tid, yr, doy, cha). Their files are already in
        # the bucket; close() finds them and puts them in the manifest.
        self._prior_done: set[tuple] = {tuple(e) for e in (prior_done or ())}

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
                    "amp_vel": float(raw_amp),
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

    @staticmethod
    def _suffix(seq: int) -> str:
        return f"-{seq:03d}.parquet" if seq else ".parquet"

    def _existing_files(self, kind: str, key: tuple) -> list[tuple[int, str]]:
        """Objects this job already holds in a partition, as (seq, path).

        One LIST with the job id as prefix. Anything not shaped like this
        job's own ``<job>.parquet`` / ``<job>-NNN.parquet`` is ignored.
        """
        stem = self._partition_path(kind, key, "")
        # s3fs strips the scheme itself (checked against the real bucket), but
        # strip it here too so the listing cannot silently come back empty on
        # a filesystem that does not.
        stem = self.fs._strip_protocol(stem) if hasattr(self.fs, "_strip_protocol") else stem
        try:
            names = self.fs.glob(stem + "*.parquet")
        except FileNotFoundError:
            return []
        out = []
        for name in names:
            base = name.rsplit("/", 1)[-1]
            if base == f"{self.job_id}.parquet":
                out.append((0, self._partition_path(kind, key, self._suffix(0))))
            else:
                m = re.fullmatch(re.escape(self.job_id) + r"-(\d{3})\.parquet", base)
                if m:
                    seq = int(m.group(1))
                    out.append((seq, self._partition_path(kind, key, self._suffix(seq))))
        return sorted(out)

    def _next_seq(self, kind: str, key: tuple) -> int:
        """The sequence number for the next file in a partition.

        The first time a partition is touched, the sequence starts after the
        highest suffix an earlier attempt of this job left there. A fresh job
        finds nothing and starts at 0, as before. A resumed job continues the
        series instead of replacing it - the bug this closes overwrote the
        first attempt's first checkpoint on every resumed shard.
        """
        if (kind, key) not in self._part_seq:
            existing = self._existing_files(kind, key)
            self._part_seq[(kind, key)] = existing[-1][0] + 1 if existing else 0
            if existing:
                logger.info(
                    f"Job {self.job_id} resumes {kind} {key} after "
                    f"{len(existing)} earlier file(s); continuing at "
                    f"{self._suffix(self._part_seq[(kind, key)])}"
                )
            elif kind == "picks" and key in self._prior_partitions():
                # Progress says the earlier attempt flushed into this partition,
                # yet the listing found nothing. Whatever the cause - a listing
                # that lies, a bucket that lost the objects - starting at 0 is
                # the one thing that must not happen, so start far above any
                # sequence a first attempt could have reached and say so.
                self._part_seq[(kind, key)] = 900
                logger.error(
                    f"Job {self.job_id} resumes {kind} {key} but found none of the "
                    f"earlier attempt's files there; continuing at "
                    f"{self._suffix(900)} so nothing can be overwritten"
                )
        seq = self._part_seq[(kind, key)]
        self._part_seq[(kind, key)] = seq + 1
        return seq

    def _write_partition(self, kind: str, key: tuple, schema, buffers) -> None:
        rows = buffers.get(key)
        if not rows:
            return
        seq = self._next_seq(kind, key)
        suffix = self._suffix(seq)
        path = self._partition_path(kind, key, suffix)

        # Encode and upload are timed apart: one is CPU on the worker, the other
        # is network to S3, and they scale with different things. The comparison
        # against DocumentDB's per-station-day insert_many needs both.
        with stage("parquet.encode", unit=len(rows), unit_name="row"):
            table = pa.Table.from_pylist(rows, schema=schema)
        self._ensure_parent(path)
        with stage("parquet.put", unit=len(rows), unit_name="row"):
            with self.fs.open(path, "wb") as fh:
                pq.write_table(
                    table, fh, compression=self.compression, use_dictionary=True
                )
        logger.info(f"Wrote {len(rows):>8} {kind} rows -> {path}")
        # Record what was written, with the partition key and row count. This is
        # what lets a reader locate a job's output with GETs alone: without it
        # the only way to find these objects is to LIST the partition, because
        # the file name carries a content hash and a flush sequence number that
        # a reader cannot reconstruct.
        network, year, month = key
        self._written.append({
            "kind": kind, "path": path, "rows": len(rows),
            "network": network, "year": year, "month": month,
        })
        buffers[key] = []

    def checkpoint(self) -> list[dict]:
        """Flush every buffered partition and report what is now durable.

        Exists so a preempted shard does not start over. Picks are buffered
        until :meth:`close`, so without this a Spot interruption twelve hours
        into a shard loses twelve hours of work. Flushing periodically bounds
        that loss to the checkpoint interval.

        Returns the station-day-channel records covered by everything written so
        far. The caller records them **after** this returns, never before: the
        ordering is what stops a resume skipping station-days whose picks were
        never actually written.
        """
        for key in list(self._picks):
            self._write_partition("picks", key, PICK_SCHEMA, self._picks)
        for key in list(self._classifies):
            self._write_partition(
                "classifies", key, CLASSIFY_SCHEMA, self._classifies
            )
        return list(self._records)

    @property
    def pending_records(self) -> int:
        """Station-day-channels handled so far, flushed or not."""
        return len(self._records)

    def _prior_partitions(self) -> set[tuple]:
        """(network, year, month) partitions the earlier attempt's records fall in."""
        partitions = set()
        for tid, yr, doy, _cha in self._prior_done:
            day = datetime.date(int(yr), 1, 1) + datetime.timedelta(days=int(doy) - 1)
            partitions.add((_network_of(tid), day.year, day.month))
        return partitions

    def _prior_output(self) -> tuple[list[dict], list[dict]]:
        """What an earlier attempt of this job left behind: its files and the
        records they cover, rebuilt from the bucket rather than trusted.

        The partitions to look in follow from ``prior_done``; the files in
        each are this job's, minus the ones this attempt wrote. Pick counts and
        run ids per station-day come from reading those files, so a resumed
        shard's manifest says the same thing about the first attempt's
        station-days as it would have said had the shard never been preempted.
        A checkpointed station-day with no rows in any file reports
        ``npks: 0``, which is what it had.
        """
        if not self._prior_done:
            return [], []
        partitions = self._prior_partitions()
        mine = {f["path"] for f in self._written}
        files, counts, rids = [], {}, {}
        for key in sorted(partitions):
            for kind in ("picks", "classifies"):
                for _seq, path in self._existing_files(kind, key):
                    if path in mine:
                        continue
                    with self.fs.open(path, "rb") as fh:
                        if kind == "picks":
                            table = pq.read_table(fh, columns=["tid", "cha", "peak", "rid"])
                            rows = table.num_rows
                            df = table.to_pandas()
                            if len(df):
                                df["yr"] = df["peak"].dt.year
                                df["doy"] = df["peak"].dt.dayofyear
                                for (tid, cha, yr, doy, rid), n in (
                                    df.groupby(["tid", "cha", "yr", "doy", "rid"]).size().items()
                                ):
                                    entry = (tid, int(yr), int(doy), cha)
                                    counts[entry] = counts.get(entry, 0) + int(n)
                                    rids.setdefault(entry, rid)
                        else:
                            rows = pq.read_metadata(fh).num_rows
                    network, year, month = key
                    files.append({"kind": kind, "path": path, "rows": rows,
                                  "network": network, "year": year, "month": month,
                                  "attempt": "prior"})
        records = [
            {"tid": tid, "cha": cha, "yr": int(yr), "doy": int(doy),
             "npks": counts.get((tid, int(yr), int(doy), cha), 0), "nclfs": 0,
             "rid": rids.get((tid, int(yr), int(doy), cha), "")}
            for tid, yr, doy, cha in sorted(self._prior_done)
        ]
        return files, records

    def close(self) -> dict:
        """Write everything still buffered, plus a manifest for the job.

        The manifest covers the whole job. If this attempt resumed an earlier
        one, the earlier attempt's files and station-days are found in the
        bucket and listed first; a reader following ``files`` then sees every
        object the job produced, and ``records`` every station-day it covered.
        """
        for key in list(self._picks):
            self._write_partition("picks", key, PICK_SCHEMA, self._picks)
        for key in list(self._classifies):
            self._write_partition(
                "classifies", key, CLASSIFY_SCHEMA, self._classifies
            )

        prior_files, prior_records = self._prior_output()
        prior_picks = sum(f["rows"] for f in prior_files if f["kind"] == "picks")
        prior_classifies = sum(f["rows"] for f in prior_files if f["kind"] == "classifies")
        summary = {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "n_picks": self.n_picks + prior_picks,
            "n_classifies": self.n_classifies + prior_classifies,
            "station_days": len(prior_records) + len(self._records),
            "written_at": datetime.datetime.utcnow().isoformat() + "Z",
            "files": prior_files + self._written,
            "records": prior_records + self._records,
        }
        if prior_records:
            summary["resumed"] = {
                "prior_station_days": len(prior_records),
                "prior_files": len(prior_files),
                "prior_picks": prior_picks,
                "prior_classifies": prior_classifies,
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
