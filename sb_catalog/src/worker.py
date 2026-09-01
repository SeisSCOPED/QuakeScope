"""
Pull shards from the S3 queue and pick them. The v3 execution unit.

One worker loop claims a shard, runs the existing picking bridge over it, flushes
Parquet, and writes the shard's manifest - in that order, so a manifest never
exists for work whose picks are not durable. Then it takes the next shard. N
loops run per node (`--procs`), and N nodes run per campaign, all against the
same queue; the claim protocol in `s3_state` is what keeps them off each other.

**Written for Spot.** Preemption arrives as SIGTERM with about two minutes'
notice, so the loop installs a handler that releases the in-flight claim and
exits, returning that shard to the queue immediately instead of making the next
worker wait out the lease. Work already flushed is kept, because completion is
per shard and recorded in S3. If the process is killed outright with no warning,
the lease expiry covers it instead.

There is no database. Station metadata, resume state and provenance all come
from the campaign prefix - see `s3_state`.

Usage (normally invoked by the Batch job definition, not by hand):
    python -m sb_catalog.src.worker \\
        --campaign s3://quakescope-picks-2026/scedc \\
        --weight jma_wc --procs 8
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import uuid
from typing import Optional

import pandas as pd

from .profiling import profile
from .s3_state import S3CampaignState

logger = logging.getLogger("worker")

# Seconds over which array tasks stagger their first S3 calls. 180 spreads
# 1,500 tasks to ~8 starts/s, well inside what a cold prefix absorbs.
STARTUP_SPREAD_SECONDS = int(os.environ.get("STARTUP_SPREAD_SECONDS", "180"))


class S3StateAdapter:
    """The slice of the SeisBenchDatabase interface the picking path uses,
    backed by campaign state on S3 instead of DocumentDB.

    Only four methods are needed; association still requires a real database and
    is not part of a v3 picking campaign.
    """

    def __init__(self, state: S3CampaignState, stations: pd.DataFrame,
                 done: Optional[set] = None):
        self.state = state
        self._stations = stations
        self.done = done or set()
        self.picks_record: list[dict] = []
        self.db_uri = state.root          # provenance strings expect these
        self.database = None

    def get_station_metadata(self, stations: list[str], key: dict = {}) -> pd.DataFrame:
        return self._stations[self._stations["id"].isin(stations)].reset_index(drop=True)

    def get_stations(self, extent=None, network=None) -> pd.DataFrame:
        return self.state.get_stations(extent=extent, network=network)

    def get_picks_record(self, station, day, channel, key: dict = {}):
        """Whether this station-day-channel is already done, from the shard's
        checkpointed progress.

        Answered from a set loaded once when the shard was claimed, so this
        costs nothing per call - the alternative would be ~800 S3 HEADs per
        shard. An empty set (a shard starting fresh) makes every answer None and
        the whole shard runs, which is the common case.

        This is what makes a preemption cheap: without it a resumed shard redoes
        every station-day from the beginning, about twelve hours at the default
        grouping.
        """
        entry = (station, day.year, int(day.strftime("%j")), channel)
        return {"_id": entry} if entry in self.done else None

    def write_run_data(self, **kwargs) -> str:
        run_id = str(uuid.uuid4())
        self.state.write_run(run_id, **{k: str(v) for k, v in kwargs.items()})
        return run_id

    def insert_many_ignore_duplicates(self, collection: str, records: list[dict]) -> None:
        if collection == "picks_record":
            self.picks_record.extend(records)
        else:                              # picks/classifies go to Parquet in v3
            logger.debug(f"Ignoring {len(records)} rows for '{collection}' (Parquet mode)")


def _log_instance_lifecycle() -> None:
    """Say whether this node is Spot or on-demand.

    A campaign budgeted for Spot should not discover it ran on-demand in Cost
    Explorer a week later. On Fargate this is normally answered by the compute
    environment (FARGATE_SPOT), so this is a backstop rather than the main
    signal. IMDSv2, short timeout, silent if unavailable.
    """
    try:
        import urllib.request
        def _req(url, headers, method="GET"):
            r = urllib.request.Request(url, headers=headers, method=method)
            return urllib.request.urlopen(r, timeout=2).read().decode()
        tok = _req("http://169.254.169.254/latest/api/token",
                   {"X-aws-ec2-metadata-token-ttl-seconds": "60"}, "PUT")
        h = {"X-aws-ec2-metadata-token": tok}
        itype = _req("http://169.254.169.254/latest/meta-data/instance-type", h)
        try:
            _req("http://169.254.169.254/latest/meta-data/spot/instance-action", h)
            life = "spot"
        except Exception as exc:                     # 404 = not interrupted yet
            life = "spot" if "404" in str(exc) else "on-demand"
        logger.info(f"Running on {itype} ({life})")
        if life != "spot":
            logger.warning(
                "This node is ON-DEMAND, not Spot - roughly 3x the hourly price."
            )
    except Exception:
        pass


class Preempted(Exception):
    """SIGTERM - Spot reclaim, or an operator stopping the job."""


def _run_shard(shard: dict, args, state: S3CampaignState, stations: pd.DataFrame) -> dict:
    """Pick one shard. Imports are local so a worker that never claims anything
    does not pay for loading torch."""
    from .picker import S3MongoSBBridge
    from .s3_helper import S3DataSource
    from .utils import parse_year_day

    done = state.read_progress(shard["shard_id"])
    if done:
        logger.info(
            f"Resuming {shard['shard_id']}: {len(done)} station-day-channels "
            f"already written, skipping them"
        )
    db = S3StateAdapter(state, stations, done=done)

    def _checkpoint(records: list[dict]) -> None:
        # Called only after the Parquet flush covering these records returned.
        state.write_progress(shard["shard_id"], records)
    # S3DataSource wants dates, not the "%Y.%j" strings the queue carries, and
    # treats `end` as exclusive - the planner writes it that way.
    s3 = S3DataSource(
        stations=",".join(shard["stations"]),
        start=parse_year_day(shard["start"]),
        end=parse_year_day(shard["end"]),
        components=args.components,
        db=db,
    )
    bridge = S3MongoSBBridge(
        s3=s3,
        db=db,
        model=args.model,
        weight=args.weight,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
        data_queue_size=args.data_queue_size,
        pick_queue_size=args.pick_queue_size,
        extent=None,
        classifier=False,          # the classifier is out for 2026
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=_checkpoint,
        flush_threshold=args.flush_threshold,
        parquet_uri=args.parquet_uri or state.uri(),
        # One Parquet object per shard. Anything less specific collides inside a
        # (network, year, month) partition when a node runs many shards.
        job_id=shard["shard_id"],
    )
    t0 = time.time()
    bridge.run_picking()
    return {
        "stations": len(shard["stations"]),
        "start": shard["start"],
        "end": shard["end"],
        "station_days": shard.get("n_station_days"),
        "picks_record": len(db.picks_record),
        "weight": args.weight,
        "model": args.model,
        "seconds": round(time.time() - t0, 1),
    }


def loop(args, proc_index: int = 0) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=f"%(asctime)s | worker{proc_index} | %(levelname)s | %(message)s",
    )
    state = S3CampaignState(args.campaign, lease_hours=args.lease_hours)
    state.worker_id = f"{os.uname().nodename}:{os.getpid()}"

    holding = {"shard": None}

    def on_term(signum, frame):
        raise Preempted()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    _log_instance_lifecycle()
    if args.profile:
        profile.enable()
        if args.procs > 1:
            logger.warning(
                "--profile with --procs %d: worker loops contend for the same "
                "cores, so per-stage attribution will be distorted. Use --procs 1.",
                args.procs,
            )

    # Spread the cold start. Every task's first act is to GET one shards.jsonl
    # (32k lines, a single hot key), LIST complete/, and PUT a claim - so 1,500
    # tasks launching together arrive as a burst on three prefixes that S3 has
    # not yet partitioned for. That burst, not the sustained rate, is what
    # returned SlowDown and killed 1,000 of 1,500 tasks: measured steady-state
    # is 0.4 writes/s against a 3,500/s limit.
    #
    # Keyed to the array index rather than random, so the spread is even rather
    # than merely uncorrelated, and reproducible when reading logs.
    idx = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if idx is not None and STARTUP_SPREAD_SECONDS > 0:
        try:
            wait = (int(idx) % max(int(STARTUP_SPREAD_SECONDS), 1))
        except ValueError:
            wait = 0
        if wait:
            logger.info(f"Staggering start by {wait}s (array index {idx}) so the "
                        f"fleet does not arrive on S3 all at once")
            time.sleep(wait)

    shards = state.read_shards()
    stations = state.get_stations()
    done = state.completed_ids()
    logger.info(
        f"Queue {args.campaign}: {len(shards)} shards, {len(done)} already complete"
    )

    # Offsetting the start point keeps N processes from contending on the same
    # head of the queue; the claim protocol would make it correct anyway, but
    # this makes it cheap.
    order = list(range(len(shards)))
    if len(order):
        off = (proc_index * 7919) % len(order)
        order = order[off:] + order[:off]

    n_done = n_failed = 0
    try:
        for idx in order:
            shard = shards[idx]
            sid = shard["shard_id"]
            if sid in done or state.is_complete(sid):
                continue
            if not state.claim(sid):
                continue
            holding["shard"] = sid
            logger.info(f"Claimed {sid} ({shard['start']}..{shard['end']}, "
                        f"{len(shard['stations'])} stations)")
            stop_beat = threading.Event()

            def _beat():
                # A 2025-sized shard runs ~23 h; the lease is hours. Refresh
                # while alive so only a dead worker's claim ever goes stale.
                while not stop_beat.wait(args.lease_hours * 3600 / 4):
                    try:
                        state.heartbeat(sid)
                    except Exception as exc:
                        logger.warning(f"Heartbeat failed for {sid}: {exc}")

            beat = threading.Thread(target=_beat, daemon=True)
            beat.start()
            try:
                manifest = _run_shard(shard, args, state, stations)
            except Preempted:
                stop_beat.set()
                raise
            except Exception as exc:
                stop_beat.set()
                n_failed += 1
                logger.exception(f"Shard {sid} failed: {exc}")
                state.release(sid)         # requeue at once rather than after the lease
                holding["shard"] = None
                if args.max_failures and n_failed >= args.max_failures:
                    logger.error(f"Stopping after {n_failed} failures")
                    break
                continue
            stop_beat.set()
            state.complete(sid, manifest)
            holding["shard"] = None
            n_done += 1
            logger.info(f"Completed {sid} in {manifest['seconds']}s "
                        f"({manifest['picks_record']} station-day-channels)")
            if args.profile:
                # Printed per shard, not per campaign: the stage mix depends on
                # pick count, which varies fourfold between an ordinary day and
                # a sequence, so a single aggregate would hide the variation.
                logger.info(f"stage profile for {sid}:{profile.report()}")
                profile.reset()
            if args.max_shards and n_done >= args.max_shards:
                logger.info(f"Reached --max-shards {args.max_shards}")
                break
    except Preempted:
        if holding["shard"]:
            logger.warning(
                f"Preempted while holding {holding['shard']} - releasing it back "
                f"to the queue"
            )
            state.release(holding["shard"])
        logger.info(f"Exiting on signal after {n_done} shards")
        sys.exit(0)

    logger.info(f"Queue drained for this worker: {n_done} shards done, {n_failed} failed")
    if n_done == 0 and n_failed > 0:
        # Exit non-zero so the jobs controller surfaces a wholly broken campaign
        # instead of reporting SUCCEEDED, as it did when every shard failed on a
        # missing environment variable.
        logger.error("No shard completed - failing the job")
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True, help="s3://bucket/campaign")
    ap.add_argument("--parquet_uri", default="",
                    help="Parquet output prefix (default: the campaign prefix)")
    ap.add_argument("--model", default="PhaseNet")
    ap.add_argument("--weight", default="jma_wc")
    ap.add_argument("--components", default="ZNE12")
    ap.add_argument("--p_threshold", default=0.2, type=float)
    ap.add_argument("--s_threshold", default=0.2, type=float)
    ap.add_argument("--data_queue_size", default=5, type=int)
    ap.add_argument("--pick_queue_size", default=5, type=int)
    ap.add_argument("--procs", default=1, type=int,
                    help="Worker loops per node. Match to vCPUs, allowing for the "
                         "picker's own threads.")
    ap.add_argument("--lease-hours", default=6.0, type=float,
                    help="A claim older than this with no manifest is reclaimable. "
                         "Set above the longest expected shard runtime.")
    ap.add_argument("--max-shards", default=0, type=int, help="Stop after N (0 = drain)")
    ap.add_argument("--max-failures", default=0, type=int, help="Stop after N failures")
    ap.add_argument("--flush-threshold", default=250_000, type=int,
                    help="Rows buffered per (network, year, month) partition "
                         "before it is written. Lower means more, smaller "
                         "objects but more frequent checkpoints - a preempted "
                         "worker only loses work since the last flush.")
    ap.add_argument("--checkpoint-every", default=40, type=int,
                    help="Flush Parquet and record progress every N "
                         "station-day-channels. Bounds what a Spot preemption "
                         "costs: at the default 40 stations x 20 days a shard is "
                         "~12 h, so without this a preemption discards all of "
                         "it. 0 disables checkpointing.")
    ap.add_argument("--profile", action="store_true",
                    help="Report a per-stage timing breakdown after each shard: "
                         "S3 list/head/get, mSEED parse, inference, both amplitude "
                         "passes, and Parquet encode/put. Use --procs 1, since "
                         "contention between worker loops distorts the attribution.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    if args.procs <= 1:
        loop(args, 0)
        return
    procs = [mp.Process(target=loop, args=(args, i)) for i in range(args.procs)]
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
    # Any worker loop failing outright fails the node, for the same reason.
    if any(p.exitcode not in (0, None) for p in procs):
        sys.exit(1)


if __name__ == "__main__":
    main()
