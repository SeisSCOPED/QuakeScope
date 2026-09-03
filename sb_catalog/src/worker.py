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
import ctypes
import gc
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

# Exit code used when a worker stops because it was preempted. Non-zero so that
# Batch evaluates `evaluateOnExit` at all - see the long comment at the
# `except Preempted` handler. Distinct from 1 so that "preempted" is separable
# from "this job is broken" when reading attempt histories.
PREEMPTED_EXIT_CODE = int(os.environ.get("PREEMPTED_EXIT_CODE", "75"))

# Ceiling on decoded station-day streams held across the whole node, used to
# size `--data_queue_size` when it is left unset. See _resolve_queue_size.
NODE_STREAM_BUDGET = int(os.environ.get("NODE_STREAM_BUDGET", "8"))


def rss_mb() -> float:
    """Resident set size, the number the container OOM killer actually uses.

    Read from /proc rather than psutil, which is not in the image and is not
    worth adding for two lines. Returns 0.0 off Linux.
    """
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**2
    except (OSError, IndexError, ValueError):
        return 0.0


def reclaim_memory() -> None:
    """Give freed memory back to the OS between shards.

    A shard decodes tens of station-days, each a numpy array of tens of MB, and
    frees them as it goes. Python returning those to its allocator is not the
    same as the process returning them to the kernel: glibc keeps freed blocks
    in per-arena free lists, and RSS - the only number the container OOM killer
    looks at - never falls. With `--procs 4` there are four such processes, and
    glibc sizes its arena pool from the core count, so the fragmentation is
    worst exactly where memory is tightest.

    That matches how the 2026-09-02 dry run died: not on a big shard, but late.
    `esr4` OOM'd after 55 minutes and 477 station-days, `west4` after 212 - a
    per-shard peak would have killed the first one.

    `malloc_trim` walks those free lists and releases what it can. Called
    between shards, where nothing large is live, so it has the most to give
    back and costs the least. See also MALLOC_ARENA_MAX in the job definitions,
    which limits how far the fragmentation can spread in the first place.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass          # not glibc - macOS or musl. gc.collect() still ran.


def _resolve_queue_size(requested: int, procs: int) -> int:
    """How many decoded station-days each loop may hold queued.

    The queue holds **decoded** obspy Streams, not bytes: one EarthScope
    station-day peaks around 0.4-0.5 GB. The default was a flat 5 per loop, so
    the node's exposure was 5 x procs and nobody was counting - at `--procs 4`
    that is ~20 station-days in flight, 10-15 GB, and it is what made the
    EarthScope I/O profile die with

        OutOfMemoryError: container killed due to memory usage

    on 8 vCPU / 16 GB while the same image at `--procs 1` was untroubled
    (OPTIMISE item 0d). SCEDC survived it only because it stores one object per
    channel, so its streams are a fraction of the size.

    Sizing from a NODE budget instead of per loop keeps that product bounded
    however `--procs` is set. At the default budget of 8:

        procs 1 -> 5   (capped, so single-process behaviour is unchanged)
        procs 2 -> 4
        procs 4 -> 2
        procs 8 -> 1

    An explicit `--data_queue_size` always wins; this only fills in the default.
    """
    if requested > 0:
        return requested
    return max(1, min(5, NODE_STREAM_BUDGET // max(procs, 1)))


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


class Preempted(BaseException):
    """SIGTERM - Spot reclaim, or an operator stopping the job.

    **Deliberately a BaseException, not an Exception**, for the same reason
    `KeyboardInterrupt` and `SystemExit` are: it is control flow, not a failure,
    and it must reach the handler that releases the claim rather than being
    treated as something to retry.

    As an Exception it was swallowed. Observed 2026-09-02 on a dry-run arm: the
    signal landed inside an FDSN metadata request, whose retry loop is a broad
    `except Exception ... sleep(5)`, so all four loops logged

        FDSN request failed (1/8): Preempted. Sleeping 5 s.

    and carried on working after being told to stop. Docker SIGKILLed the
    container ~120 s later - exit 137, four claims stranded for the full lease.
    Exactly the failure the SIGTERM forwarding fix was meant to end, reintroduced
    by a handler three modules away.

    There are 19 broad `except Exception` handlers in this package and one bare
    `except:`. Auditing them one by one is a losing game; making this
    uncatchable-by-accident fixes all of them at once, including any added
    later, and including the ones inside obspy, boto3 and seisbench.
    """


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
        data_queue_size=_resolve_queue_size(args.data_queue_size, args.procs),
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

    q = _resolve_queue_size(args.data_queue_size, args.procs)
    logger.info(
        f"Holding at most {q} decoded station-day(s) per loop "
        f"({q * args.procs} across {args.procs} loop(s), budget "
        f"{NODE_STREAM_BUDGET})"
        + ("" if args.data_queue_size > 0 else " - sized from --procs")
    )

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
                # A shard that died mid-read is holding the most, not the least.
                reclaim_memory()
                if args.max_failures and n_failed >= args.max_failures:
                    logger.error(f"Stopping after {n_failed} failures")
                    break
                continue
            stop_beat.set()
            state.complete(sid, manifest)
            holding["shard"] = None
            n_done += 1
            # Between shards is the one moment nothing large is live.
            before = rss_mb()
            reclaim_memory()
            after = rss_mb()
            logger.info(f"Completed {sid} in {manifest['seconds']}s "
                        f"({manifest['picks_record']} station-day-channels) "
                        f"| RSS {after:.0f} MB (freed {before - after:.0f})")
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
        # Exit NON-ZERO on a preemption, deliberately.
        #
        # Batch only consults `evaluateOnExit` for a FAILED attempt. An attempt
        # that exits 0 is a success, so the rule
        #
        #   {onStatusReason: "Your Spot Task was interrupted.", action: retry}
        #
        # never fires - even though Batch sets statusReason to exactly that
        # string. Observed on a real reclaim 2026-09-01: two workers released
        # their claims cleanly, exited 0, were recorded SUCCEEDED having done no
        # work, and were never replaced. The shard survives (it returns to the
        # queue and resumes from its checkpoint); the WORKER does not, so over a
        # multi-day campaign at 1,500 Spot workers the fleet decays with no
        # failures visible anywhere.
        #
        # The exit code is the only lever - no wording of the rule can rescue an
        # exit-0 attempt. The cost is cosmetic: ordinary preemptions now appear
        # as failed attempts in the console, and the catch-all
        # {onReason: "*", action: exit} still stops a genuinely broken job from
        # burning all 10 attempts.
        #
        # PREEMPTED_EXIT_CODE is distinct from 1 so "preempted" is separable
        # from "this job is broken" when reading attempt histories.
        logger.info(
            f"Exiting on signal after {n_done} shards "
            f"(exit {PREEMPTED_EXIT_CODE} so Batch retries this worker)"
        )
        sys.exit(PREEMPTED_EXIT_CODE)

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
    # 0 means "size it from --procs" - see _resolve_queue_size. An explicit
    # value is always honoured.
    ap.add_argument("--data_queue_size", default=0, type=int)
    ap.add_argument("--pick_queue_size", default=5, type=int)
    ap.add_argument("--procs", default=1, type=int,
                    help="Worker loops per node. Match to vCPUs, allowing for the "
                         "picker's own threads.")
    # 1 hour, not 6. The lease only matters for a DEAD worker - a live one
    # refreshes its claim every lease/4 - so it is really "how long a stranded
    # shard stays out of circulation". On a pool reclaiming 37-53% of attempts
    # (measured 2026-09-01/02), six hours is long enough to strand the whole
    # queue: a dry-run arm deadlocked with all 8 shards held by dead attempts
    # and its retries finding nothing to claim. Shards run ~2 minutes, so an
    # hour is still ~30x headroom.
    ap.add_argument("--lease-hours", default=1.0, type=float,
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

    # Forward SIGTERM to the worker loops.
    #
    # Docker delivers SIGTERM to PID 1 only, and PID 1 here is this parent, which
    # does nothing with it by default. The handler that releases a claim lives in
    # loop(), which now runs in the children - so without this the parent died on
    # the default action, the children never saw the signal, and ~120 s later the
    # whole task was SIGKILLed with every claim still held. Those shards then sat
    # unavailable for the full lease while the retried job found nothing to claim
    # and exited 0, reporting SUCCEEDED having done no work. Measured 2026-09-01:
    # two arms, exit 137 with no handler output, 8 stranded claims each.
    #
    # Only --procs 1 was ever safe, because there loop() runs in this process.
    stopping = threading.Event()

    def _forward(signum, frame):
        if stopping.is_set():          # second signal: stop waiting, let it die
            return
        stopping.set()
        logger.warning(
            f"Signal {signum} - forwarding to {len(procs)} worker loops so they "
            f"release their claims"
        )
        for p in procs:
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    # join() is not interruptible by a signal on all platforms, so poll instead:
    # a plain join() can swallow the handler until the child happens to exit.
    while any(p.is_alive() for p in procs):
        time.sleep(0.5)
        if stopping.is_set():
            break

    if stopping.is_set():
        # Spot allows about two minutes between SIGTERM and SIGKILL. Give the
        # loops most of it to finish the station-day they are on, flush Parquet
        # and release the claim - that ordering is what keeps a manifest from
        # ever describing picks that were not written.
        deadline = time.time() + float(os.environ.get("SHUTDOWN_GRACE_SECONDS", "90"))
        for p in procs:
            p.join(timeout=max(0.0, deadline - time.time()))
        for p in procs:
            if p.is_alive():
                logger.warning(
                    f"Worker {p.pid} still running after the grace period; killing it. "
                    f"Its claim will be reclaimed when the lease expires."
                )
                p.kill()
    for p in procs:
        p.join()

    # Report preemption explicitly, before looking at child exit codes.
    #
    # Otherwise the node's exit code is decided by a race. A child that released
    # its claim inside the grace period exits PREEMPTED_EXIT_CODE; one that was
    # still inside a long `model.classify` call when the grace ran out is
    # kill()ed and exits -9; and if every child happened to finish cleanly the
    # parent fell through to exit 0 and was never retried. All three happened on
    # 2026-09-01 - three reclaims exited 0 and were not retried, one exited 1 and
    # was. Whether the fleet self-heals must not depend on that timing.
    if stopping.is_set():
        logger.info(
            f"Node preempted - exiting {PREEMPTED_EXIT_CODE} so Batch retries it"
        )
        sys.exit(PREEMPTED_EXIT_CODE)

    # Any worker loop failing outright fails the node, for the same reason.
    if any(p.exitcode not in (0, None) for p in procs):
        sys.exit(1)


if __name__ == "__main__":
    main()
