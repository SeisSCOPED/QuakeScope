"""An embargoed shard leaves the queue instead of spinning in it.

EarthScope embargoes recent years of a temporary FDSN code and opens them
later, and codes are reused between experiments, so a 403 is a statement about
today rather than a defect. The client used to treat it as a shard failure,
which called `state.release()` and put the shard straight back on the queue for
the next worker to claim, fail, and release again.

On 2026-09-05 a 57-worker fleet spent most of an hour doing exactly that: 2,010
failures on 7D and 2F, whose recent years are embargoed, alongside 496 on
network C that were our own throttling. Nothing was wrong with the data or the
credentials; the queue simply had no way to say "not yet".
"""

import datetime
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botocore.exceptions import ClientError

from sb_catalog.src.s3_state import S3CampaignState


class FakeS3:
    """Enough S3 for the state store, with real conditional-write semantics."""

    def __init__(self):
        self.obj, self.lock = {}, threading.Lock()

    def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None,
                   IfMatch=None):
        with self.lock:
            if IfNoneMatch is not None and Key in self.obj:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            self.obj[Key] = Body if isinstance(Body, bytes) else Body.encode()
            return {"ETag": '"x"'}

    def get_object(self, Bucket, Key):
        import io
        with self.lock:
            if Key not in self.obj:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": io.BytesIO(self.obj[Key]), "ETag": '"x"'}

    def head_object(self, Bucket, Key):
        with self.lock:
            if Key not in self.obj:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            return {"ETag": '"x"'}

    def delete_object(self, Bucket, Key):
        with self.lock:
            self.obj.pop(Key, None)
            return {}

    def get_paginator(self, _):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix, **kw):
                with outer.lock:
                    keys = [k for k in outer.obj if k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in keys]}
        return P()


def _state():
    st = S3CampaignState("s3://bucket/camp")
    st.s3 = FakeS3()
    return st


def test_a_blocked_shard_is_not_handed_out_again():
    st = _state()
    st.block("s1", "7D 2024 is embargoed", {"network": "7D", "year": 2024})
    assert st.blocked_ids() == {"s1"}
    # Blocked is not complete: the work still exists, it is just unavailable.
    assert "s1" not in st.completed_ids()


def test_a_blocked_shard_cannot_be_reclaimed():
    """The churn found by watching the 2026-09-05 obs run.

    `block()` deletes the claim, so the shard is immediately claimable again,
    and the worker's blocked set is read once at startup. Every process whose
    set predated the block re-claimed, re-ran the archive check and re-blocked:
    335 shards blocked 34,442 times - 103 each - while 11 runnable shards
    waited behind them. Checking S3 in `claim` is what makes the block stick,
    because the in-memory set cannot know about a block that happened after it
    was read.
    """
    st = _state()
    assert st.claim("s1") is True
    st.block("s1", "7D 2024 embargoed", {"network": "7D", "year": 2024})
    assert st.claim("s1") is False, "a blocked shard was handed out again"
    # And it comes back when the embargo lifts.
    st.unblock(["s1"])
    assert st.claim("s1") is True


def test_blocking_drops_the_claim():
    """Otherwise the shard reads as held by a worker that has gone away."""
    st = _state()
    assert st.claim("s1") is True
    st.block("s1", "embargoed", {"network": "2F", "year": 2023})
    # A claim left behind would make the shard wait out the whole lease before
    # anything could look at it again.
    assert "s1" not in {k.split("/")[-1][:-5] for k in st.s3.obj
                        if "/claims/" in k}


def test_blocked_shards_are_not_counted_as_remaining():
    """A campaign whose rest is embargoed should read as done, not stalled."""
    st = _state()
    st.write_shards([{"shard_id": f"s{i}", "stations": ["7D.AAA."],
                      "start": "2024.001", "end": "2024.021",
                      "n_station_days": 1} for i in range(4)])
    st.complete("s0", {"n_picks": 1})
    st.block("s1", "embargoed", {"network": "7D", "year": 2024})
    st.block("s2", "embargoed", {"network": "7D", "year": 2025})
    p = st.progress()
    assert p["total"] == 4 and p["complete"] == 1
    assert p["blocked"] == 2
    assert p["remaining"] == 1, (
        "blocked shards counted as remaining make a campaign look unfinished "
        "forever, which is what hid the spin")


def test_unblock_returns_them_when_the_embargo_lifts():
    """Blocked must be reversible: today's 403 is next year's data."""
    st = _state()
    st.block("s1", "embargoed", {"network": "7D", "year": 2022})
    st.block("s2", "embargoed", {"network": "7D", "year": 2023})
    assert st.unblock(["s1"]) == 1
    assert st.blocked_ids() == {"s2"}
    assert st.unblock() == 1
    assert st.blocked_ids() == set()


def test_metadata_faults_are_flagged_apart_from_embargo():
    """"335 blocked" and "335 blocked, 122 needing a human" differ.

    An embargo resolves itself when EarthScope opens the year; a shard naming a
    station the station table does not have, or an inventory request FDSN calls
    malformed, never will. Both leave the queue the same way, but only one of
    them is fine to leave alone, so the record carries which.
    """
    st = _state()
    st.block("s1", "7D 2024 embargoed", {"network": "7D", "year": 2024})
    st.block("s2", "LH.HDSE. absent from the station table",
             {"stations": ["LH.HDSE."]}, kind="metadata")
    st.block("s3", "FDSN rejected the inventory request",
             {"networks": ["LH"]}, kind="metadata")

    summary = st.blocked_summary()
    assert summary["embargo"]["count"] == 1
    assert summary["metadata"]["count"] == 2
    # And it says enough to act on without opening S3 by hand.
    ex = summary["metadata"]["examples"][0]
    assert ex["shard_id"] in ("s2", "s3") and ex["reason"]

    # Default stays embargo, so existing callers keep their meaning.
    assert json.loads(st.s3.obj["camp/blocked/s1.json"])["kind"] == "embargo"


def test_a_signal_fault_is_recorded_without_losing_the_shard():
    """The third way a shard got stuck, found on the 2026-09-07 western run.

    obspy raises from the signal path on data that is merely odd - a 6 Hz
    channel beside a 100 Hz one, a response corner above Nyquist, a zero in a
    gain. Those escaped the per-station-day loop and failed the whole shard,
    which was then released and re-claimed and failed again: six workers spent
    a night on 36 shards and completed none, while roughly 800 good
    station-days per shard went unpicked because of one trace.

    So the station-day is skipped and written down, and the shard still
    finishes. Recorded rather than blocked, because the shard DID complete -
    `review/` is a list for a person, not a queue state.
    """
    st = _state()
    st.note_review("s1", [
        {"station": "CI.ABC.", "channel": "HH", "day": "2024.268",
         "error": "ValueError: Sampling rate differs: 6.0 vs 100.0"},
        {"station": "CI.DEF.", "channel": "BH", "day": "2024.269",
         "error": "ValueError: Selected corner frequency is above Nyquist."},
    ])
    assert st.review_ids() == {"s1"}
    rec = json.loads(st.s3.obj["camp/review/s1.json"])
    assert rec["kind"] == "signal" and rec["count"] == 2
    assert "Nyquist" in rec["items"][1]["error"]

    # A shard with nothing to report writes nothing at all.
    st.note_review("s2", [])
    assert st.review_ids() == {"s1"}

    # And review is NOT blocked: the shard completed, so it must not be
    # excluded from the queue the way an embargoed one is.
    assert st.blocked_ids() == set()
    assert st.claim("s1") is True


def test_the_block_record_says_what_has_to_open():
    """A bare 'blocked' is useless six months later."""
    st = _state()
    st.block("s1", "7D 2024 is not readable today", {"network": "7D", "year": 2024})
    rec = json.loads(st.s3.obj["camp/blocked/s1.json"])
    assert rec["scope"] == {"network": "7D", "year": 2024}
    assert rec["shard_id"] == "s1" and rec["reason"]
    # Dated, so a survey can tell a stale block from a fresh one.
    datetime.datetime.fromisoformat(rec["blocked"].replace("Z", "+00:00"))
