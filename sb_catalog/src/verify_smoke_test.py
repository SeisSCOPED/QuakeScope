"""
Verify a tier-2 smoke-test run: does what landed in DocumentDB match what the
same weights produce locally?

Tier 1 is the notebooks in ``tutorials/``, which check the models on a laptop.
Tier 2 runs the real container on EC2 or Fargate against a small, bounded job
and asks whether the deployed path agrees with the local one. Passing means the
image, the weights, S3 access, and the database writes all work together; it is
the last gate before launching a full campaign.

Usage (from inside the container, or anywhere with database access):

    python -m src.verify_smoke_test \
        --db_uri "$DB_URI" --database quakescope_smoke \
        --stations CI.CLC.,CI.TOW2.,CI.SRT. \
        --start 2019.187 --end 2019.188

Exit status is 0 when every check passes and 1 otherwise, so it can gate a
pipeline step.
"""

import argparse
import datetime
import logging
import sys

from typing import Optional

from .utils import SeisBenchDatabase, parse_year_day

logger = logging.getLogger("verify")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)

# Picks produced locally by quakescope2026 at the picker defaults
# (p_threshold = s_threshold = 0.2) on SCEDC day 2019.187, whole day, HH
# channels. Regenerate with tutorials/ if the weights or thresholds change.
REFERENCE = {
    "CI.CLC.": {"P": 4697, "S": 4097},
    "CI.TOW2.": {"P": 3577, "S": 2810},
    "CI.SRT.": {"P": 3860, "S": 2708},
}

# Counts are deterministic for fixed weights and data, but BLAS and torch
# versions can move a handful of marginal picks across the threshold, so the
# check is a band rather than an equality.
COUNT_TOLERANCE = 0.10


class Check:
    """Collects pass/fail results so every check runs before reporting."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def record(self, ok: bool, label: str, detail: str = "") -> bool:
        line = f"{label}{' - ' + detail if detail else ''}"
        (self.passes if ok else self.failures).append(line)
        logger.info(f"[{'PASS' if ok else 'FAIL'}] {line}")
        return ok

    def report(self) -> int:
        logger.info("")
        logger.info(f"{len(self.passes)} passed, {len(self.failures)} failed")
        if self.failures:
            logger.error("failures:")
            for f in self.failures:
                logger.error(f"  - {f}")
            return 1
        logger.info("tier-2 smoke test PASSED")
        return 0


def verify(
    db: SeisBenchDatabase,
    stations: list[str],
    start: datetime.date,
    end: datetime.date,
    expect_classifier: bool,
    run_id: Optional[str] = None,
) -> int:
    check = Check()
    # Counts are meaningless if the database holds more than one run, which is
    # the normal state of a smoke-test database that has been re-used.
    scope = {"rid": run_id} if run_id else {}
    days = (end - start).days
    picks = db.database["picks"]
    classifies = db.database["classifies"]
    records = db.database["picks_record"]

    # 1. The job ran at all: one picks_record per station and day.
    n_records = records.count_documents({"tid": {"$in": stations}})
    check.record(
        n_records >= len(stations) * days,
        "picks_record written for every station-day",
        f"found {n_records}, expected at least {len(stations) * days}",
    )

    # 2. Pick counts land near what the same weights produce locally.
    for station in stations:
        got = {
            phase: picks.count_documents({"tid": station, "pha": phase, **scope})
            for phase in ("P", "S")
        }
        expected = REFERENCE.get(station)
        if expected is None:
            check.record(
                got["P"] + got["S"] > 0,
                f"{station} produced picks",
                f"P={got['P']}, S={got['S']} (no local reference to compare)",
            )
            continue
        for phase in ("P", "S"):
            lo = expected[phase] * (1 - COUNT_TOLERANCE)
            hi = expected[phase] * (1 + COUNT_TOLERANCE)
            check.record(
                lo <= got[phase] <= hi,
                f"{station} {phase} count within {COUNT_TOLERANCE:.0%} of local",
                f"got {got[phase]}, local reference {expected[phase]}",
            )

    # 3. Pick documents are well formed. A job can write the right number of
    #    garbage rows, so check the fields rather than only the counts.
    bad_conf = picks.count_documents(
        {"tid": {"$in": stations}, "$or": [{"conf": {"$lt": 0}}, {"conf": {"$gt": 1}}]}
    )
    check.record(bad_conf == 0, "pick confidences inside [0, 1]", f"{bad_conf} outside")

    bad_phase = picks.count_documents(
        {"tid": {"$in": stations}, "pha": {"$nin": ["P", "S"]}}
    )
    check.record(bad_phase == 0, "pick phases are P or S", f"{bad_phase} other")

    window_start = datetime.datetime.combine(start, datetime.time.min)
    window_end = datetime.datetime.combine(end, datetime.time.min)
    outside = picks.count_documents(
        {
            "tid": {"$in": stations},
            "$or": [{"peak": {"$lt": window_start}}, {"peak": {"$gte": window_end}}],
        }
    )
    check.record(
        outside == 0,
        "pick times inside the requested window",
        f"{outside} outside {start} to {end}",
    )

    # 4. Amplitudes are attached, since the 2026 run depends on them.
    missing_amp = picks.count_documents(
        {"tid": {"$in": stations}, "$or": [{"amp": None}, {"amp_raw": None}]}
    )
    check.record(
        missing_amp == 0, "amplitudes attached to picks", f"{missing_amp} missing"
    )

    # 5. Classifier output, when the job was asked for it. This is the path that
    #    used to crash on a schema mismatch, so an empty collection is a failure
    #    rather than an absence of events.
    if expect_classifier:
        n_class = classifies.count_documents({"tid": {"$in": stations}})
        check.record(n_class > 0, "classifier wrote records", f"{n_class} rows")
        if n_class:
            bad = classifies.count_documents(
                {
                    "tid": {"$in": stations},
                    "$or": [
                        {f: {"$lt": 0}} for f in ("eq", "px", "su")
                    ]
                    + [{f: {"$gt": 1}} for f in ("eq", "px", "su")],
                }
            )
            check.record(
                bad == 0, "class probabilities inside [0, 1]", f"{bad} outside"
            )
            sample = classifies.find_one({"tid": {"$in": stations}})
            missing = [f for f in ("eq", "px", "su", "start") if f not in sample]
            check.record(
                not missing,
                "classifier records carry every expected field",
                f"missing {missing}" if missing else "",
            )

    return check.report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db_uri", required=True, help="DocumentDB / MongoDB URI.")
    parser.add_argument("--database", required=True, help="Database name.")
    parser.add_argument(
        "--stations",
        required=True,
        help="Comma separated, format NET.STA.LOC (as written to the database).",
    )
    parser.add_argument("--start", required=True, type=parse_year_day, help="YYYY.DDD")
    parser.add_argument(
        "--end", required=True, type=parse_year_day, help="YYYY.DDD (exclusive)"
    )
    parser.add_argument(
        "--classifier",
        action="store_true",
        help="Also check classifier output. Off by default: the 2026 campaign "
        "runs without --classifier, so expecting those rows would fail a "
        "healthy run.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Restrict counts to one run id. Without it, a database reused "
        "across runs accumulates picks and the count checks compare against "
        "the wrong total.",
    )
    args = parser.parse_args()

    db = SeisBenchDatabase(args.db_uri, args.database)
    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    logger.info(
        f"verifying {len(stations)} stations, {args.start} to {args.end}, "
        f"database '{args.database}'"
    )
    sys.exit(
        verify(db, stations, args.start, args.end, args.classifier, args.run_id)
    )


if __name__ == "__main__":
    main()
