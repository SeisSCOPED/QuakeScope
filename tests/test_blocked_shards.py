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


def test_the_block_record_says_what_has_to_open():
    """A bare 'blocked' is useless six months later."""
    st = _state()
    st.block("s1", "7D 2024 is not readable today", {"network": "7D", "year": 2024})
    rec = json.loads(st.s3.obj["camp/blocked/s1.json"])
    assert rec["scope"] == {"network": "7D", "year": 2024}
    assert rec["shard_id"] == "s1" and rec["reason"]
    # Dated, so a survey can tell a stale block from a fresh one.
    datetime.datetime.fromisoformat(rec["blocked"].replace("Z", "+00:00"))
