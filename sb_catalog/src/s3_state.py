"""
S3-backed campaign state: the replacement for DocumentDB in v3.

Everything the pipeline needs to keep between jobs now lives under one S3
prefix, so a campaign has no provisioned database, no VPC to launch inside, and
no endpoint to keep alive:

    s3://<bucket>/<campaign>/
        stations.parquet          station metadata          (was: stations)
        shards.jsonl              the work queue            (new)
        claims/<shard_id>.json    who is working on what    (new)
        complete/<shard_id>.json  what finished             (was: picks_record)
        progress/<shard_id>.json  mid-shard checkpoints     (new)
        manifests/<job_id>.json   what each job wrote       (ParquetPickWriter)
        runs/<run_id>.json        provenance                (was: sb_runs)
        picks/network=/year=/month=/*.parquet

**Claiming is the part that makes Spot safe.** SkyPilot recovers a preempted
managed job by relaunching it, so a worker can die mid-shard at any moment and
another can start on the same queue seconds later. Claims are taken with an S3
conditional write (`IfNoneMatch: "*"`), which fails if the key already exists -
an atomic compare-and-set, so two workers can never take the same shard. A claim
carries a timestamp and is reclaimable after `lease_hours` with no manifest,
which is what returns work abandoned by a preemption to the queue.

A shard is therefore in exactly one of three states, all readable from S3 alone:
finished (manifest exists), in flight (fresh claim, no manifest), or available
(no claim, or a stale one).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
from typing import Any, Iterator, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger("s3_state")

DEFAULT_LEASE_HOURS = 6.0


def _split_uri(uri: str) -> tuple[str, str]:
    """s3://bucket/a/b -> ("bucket", "a/b")"""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    body = uri[len("s3://"):].strip("/")
    bucket, _, key = body.partition("/")
    if not bucket:
        raise ValueError(f"No bucket in {uri!r}")
    return bucket, key


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ShardTaken(Exception):
    """Another worker holds this shard."""


class S3CampaignState:
    """Campaign state on S3. One instance per campaign prefix."""

    def __init__(
        self,
        root: str,
        client: Any = None,
        lease_hours: float = DEFAULT_LEASE_HOURS,
    ) -> None:
        self.root = root.rstrip("/")
        self.bucket, self.prefix = _split_uri(self.root)
        self.lease_hours = lease_hours
        self.s3 = client if client is not None else boto3.client("s3")
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"

    # ---------------------------------------------------------------- paths
    def _key(self, *parts: str) -> str:
        return "/".join([p for p in (self.prefix, *parts) if p])

    def uri(self, *parts: str) -> str:
        return f"s3://{self.bucket}/{self._key(*parts)}"

    # ------------------------------------------------------------ primitives
    def _put_json(self, key: str, obj: dict, if_absent: bool = False) -> bool:
        """Write JSON. With if_absent, returns False if the key already exists.

        The conditional write is the atomic compare-and-set the claim protocol
        rests on; S3 answers PreconditionFailed when another worker got there
        first. Older S3-compatible endpoints may reject the header outright
        (NotImplemented), which we surface rather than silently racing.
        """
        kwargs = dict(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(obj, indent=2).encode(),
            ContentType="application/json",
        )
        if if_absent:
            kwargs["IfNoneMatch"] = "*"
        try:
            self.s3.put_object(**kwargs)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if if_absent and code in ("PreconditionFailed", "ConditionalRequestConflict"):
                return False
            if if_absent and code == "NotImplemented":
                raise RuntimeError(
                    "This S3 endpoint does not support conditional writes "
                    "(IfNoneMatch), so shard claims cannot be made atomic. "
                    "Use AWS S3, or run with a single worker."
                ) from exc
            raise

    def _get_json(self, key: str) -> Optional[dict]:
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def _exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return False
            raise

    def _list(self, sub: str) -> Iterator[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(sub)):
            for item in page.get("Contents", []):
                yield item["Key"]

    # -------------------------------------------------------------- stations
    def write_stations(self, stations: pd.DataFrame) -> str:
        """Persist station metadata. Replaces the `stations` collection."""
        uri = self.uri("stations.parquet")
        stations.to_parquet(uri, index=False)
        logger.info(f"Wrote {len(stations)} stations to {uri}")
        return uri

    def get_stations(
        self,
        extent: tuple[float, float, float, float] | None = None,
        network: str | None = None,
    ) -> pd.DataFrame:
        """Station metadata, filtered like SeisBenchDatabase.get_stations did."""
        df = pd.read_parquet(self.uri("stations.parquet"))
        if network:
            wanted = {n.strip() for n in network.split(",") if n.strip()}
            df = df[df["network_code"].isin(wanted)]
        if extent:
            minlat, maxlat, minlon, maxlon = extent
            df = df[
                df["latitude"].between(minlat, maxlat)
                & df["longitude"].between(minlon, maxlon)
            ]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ runs
    def write_run(self, run_id: str, **meta: Any) -> str:
        """Provenance for one campaign run. Replaces `sb_runs`."""
        self._put_json(self._key("runs", f"{run_id}.json"),
                       {"run_id": run_id, "created": _utcnow(), **meta})
        return self.uri("runs", f"{run_id}.json")

    # ---------------------------------------------------------------- shards
    def write_shards(self, shards: list[dict]) -> str:
        """Write the immutable work queue. Refuses to clobber an existing one,
        because shard ids are how completed work is recognised on resume."""
        key = self._key("shards.jsonl")
        if self._exists(key):
            raise FileExistsError(
                f"{self.uri('shards.jsonl')} already exists. The queue is immutable "
                f"once a campaign starts - completed work is keyed on shard id. "
                f"Use a new campaign prefix, or delete it deliberately."
            )
        body = "\n".join(json.dumps(s) for s in shards).encode()
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                           ContentType="application/x-ndjson")
        logger.info(f"Wrote {len(shards)} shards to {self.uri('shards.jsonl')}")
        return self.uri("shards.jsonl")

    def read_shards(self) -> list[dict]:
        obj = self.s3.get_object(Bucket=self.bucket, Key=self._key("shards.jsonl"))
        return [json.loads(l) for l in obj["Body"].read().decode().splitlines() if l.strip()]

    def is_complete(self, shard_id: str) -> bool:
        return self._exists(self._key("complete", f"{shard_id}.json"))

    def completed_ids(self) -> set[str]:
        """One LIST instead of one HEAD per shard - matters at 65k shards.

        Reads `complete/`, deliberately NOT `manifests/`: ParquetPickWriter also
        writes a per-job manifest, and sharing the prefix made its object look
        like a finished shard, so resume skipped work that never ran.
        """
        n = len(self._key("complete")) + 1
        return {k[n:-len(".json")] for k in self._list("complete/") if k.endswith(".json")}

    def claim(self, shard_id: str) -> bool:
        """Atomically take a shard. False if someone else holds a live claim."""
        key = self._key("claims", f"{shard_id}.json")
        record = {"shard_id": shard_id, "worker": self.worker_id, "claimed": _utcnow()}
        if self._put_json(key, record, if_absent=True):
            return True

        # A claim exists. It is only ours to take if it went stale without ever
        # producing a manifest - which is what a Spot preemption leaves behind.
        existing = self._get_json(key)
        if existing is None:
            return False
        if self.is_complete(shard_id):
            return False
        try:
            claimed = datetime.datetime.fromisoformat(existing["claimed"])
        except (KeyError, ValueError):
            return False
        age_h = (datetime.datetime.now(datetime.timezone.utc) - claimed).total_seconds() / 3600
        if age_h < self.lease_hours:
            return False
        logger.warning(
            f"Reclaiming {shard_id}: claim by {existing.get('worker')} is "
            f"{age_h:.1f}h old (lease {self.lease_hours}h) with no manifest"
        )
        record["reclaimed_from"] = existing.get("worker")
        record["reclaimed_after_hours"] = round(age_h, 2)
        self._put_json(key, record)          # unconditional: we won the stale race
        return True

    def read_progress(self, shard_id: str) -> set[tuple]:
        """Station-day-channels of this shard already written and recorded.

        Returned as `(tid, yr, doy, cha)` tuples. A resumed shard skips these
        instead of starting over, which is what keeps a preemption from costing
        the whole shard - about twelve hours at the default grouping.
        """
        record = self._get_json(self._key("progress", f"{shard_id}.json"))
        if not record:
            return set()
        return {tuple(e) for e in record.get("done", [])}

    def write_progress(self, shard_id: str, records: list[dict]) -> None:
        """Record durable progress mid-shard.

        Only ever called *after* the Parquet flush that covers these records has
        returned. Written the other way round, a resume would skip station-days
        whose picks were never stored - the same ordering trap as `complete`.
        """
        # Identity only. The full records carry npks/nclfs/rid, which the final
        # manifest needs but a resume does not, and this object is rewritten at
        # every checkpoint - so carrying them would triple an O(n^2) write for
        # no benefit.
        done = [[r["tid"], r["yr"], r["doy"], r["cha"]] for r in records]
        self._put_json(
            self._key("progress", f"{shard_id}.json"),
            {"shard_id": shard_id, "worker": self.worker_id,
             "updated": _utcnow(), "n": len(done), "done": done},
        )

    def complete(self, shard_id: str, manifest: dict) -> str:
        """Mark a shard done. Written only after its Parquet is durable."""
        key = self._key("complete", f"{shard_id}.json")
        self._put_json(key, {"shard_id": shard_id, "worker": self.worker_id,
                             "completed": _utcnow(), **manifest})
        return self.uri("complete", f"{shard_id}.json")

    def heartbeat(self, shard_id: str) -> None:
        """Refresh a claim's timestamp so a running shard is never reclaimed.

        Without this the lease has to exceed the slowest possible shard, which
        makes abandoned work sit in the queue for that long after a hard kill.
        With it the lease can stay short: a live worker keeps saying so, and only
        a genuinely dead one goes stale.
        """
        self._put_json(self._key("claims", f"{shard_id}.json"),
                       {"shard_id": shard_id, "worker": self.worker_id,
                        "claimed": _utcnow(), "heartbeat": True})

    def release(self, shard_id: str) -> None:
        """Drop a claim after a failure so the shard returns to the queue at once
        rather than waiting out the lease."""
        try:
            self.s3.delete_object(Bucket=self.bucket,
                                  Key=self._key("claims", f"{shard_id}.json"))
        except ClientError as exc:      # not fatal: the lease still expires
            logger.warning(f"Could not release {shard_id}: {exc}")

    # ---------------------------------------------------------------- status
    def progress(self) -> dict:
        shards = self.read_shards()
        done = self.completed_ids()
        n = len(self._key("claims")) + 1
        claimed = {k[n:-len(".json")] for k in self._list("claims/") if k.endswith(".json")}
        return {
            "total": len(shards),
            "complete": len(done),
            "in_flight": len(claimed - done),
            "remaining": len(shards) - len(done),
        }
