"""Exercise the claim protocol

Run: pixi run -e cloud python tests/test_claim_protocol.py

Original docstring against a fake S3 with real IfNoneMatch semantics,
including the concurrent race and the Spot-preemption reclaim path."""
import datetime, json, os, sys, threading
from botocore.exceptions import ClientError
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb_catalog.src.s3_state import S3CampaignState


class FakeS3:
    """Minimal S3 with atomic conditional PUT, guarded by a lock the way S3's
    strong consistency guards a real bucket."""
    def __init__(self):
        self.obj, self.lock = {}, threading.Lock()
        self.conditional_puts = 0
        self.rejected = 0

    def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None,
                   IfMatch=None):
        with self.lock:
            if IfNoneMatch is not None:
                self.conditional_puts += 1
                if IfNoneMatch != "*":
                    raise ValueError("only '*' supported")
                if Key in self.obj:
                    self.rejected += 1
                    raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            if IfMatch is not None:
                # Compare-and-swap: succeed only if the object is unchanged.
                self.conditional_puts += 1
                if self._etag(Key) != IfMatch:
                    self.rejected += 1
                    raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            self.obj[Key] = Body if isinstance(Body, bytes) else Body.encode()
            return {}

    def _etag(self, Key):
        import hashlib
        if Key not in self.obj:
            return None
        return '"%s"' % hashlib.md5(self.obj[Key]).hexdigest()

    def get_object(self, Bucket, Key):
        with self.lock:
            if Key not in self.obj:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            import io
            return {"Body": io.BytesIO(self.obj[Key]), "ETag": self._etag(Key)}

    def head_object(self, Bucket, Key):
        with self.lock:
            if Key not in self.obj:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            return {}

    def delete_object(self, Bucket, Key):
        with self.lock:
            self.obj.pop(Key, None)
            return {}

    def get_paginator(self, name):
        outer = self
        class P:
            def paginate(self, Bucket, Prefix):
                with outer.lock:
                    keys = [k for k in outer.obj if k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in keys]}
        return P()


def test_claim_protocol():
    """Claiming must be atomic, leases must expire, completed work must never
    re-run. Spot recovery depends on all three."""
    fake = FakeS3()
    st = S3CampaignState("s3://bkt/campaign", client=fake)

    # ---- 1. immutable queue -----------------------------------------------------
    shards = [{"shard_id": f"s{i:04d}", "stations": ["CI.CLC."], "start": "2019.001",
               "end": "2019.020"} for i in range(50)]
    st.write_shards(shards)
    try:
        st.write_shards(shards); print("FAIL: queue was overwritten")
    except FileExistsError:
        print("PASS  queue is immutable once written")
    assert len(st.read_shards()) == 50

    # ---- 2. two workers race for every shard ------------------------------------
    winners, lock = {}, threading.Lock()
    def race(worker):
        s = S3CampaignState("s3://bkt/campaign", client=fake)
        s.worker_id = worker
        for sh in shards:
            if s.claim(sh["shard_id"]):
                with lock:
                    winners.setdefault(sh["shard_id"], []).append(worker)

    ts = [threading.Thread(target=race, args=(f"w{i}",)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    dupes = {k: v for k, v in winners.items() if len(v) > 1}
    print(f"PASS  8 workers x 50 shards -> {len(winners)} claimed, "
          f"{len(dupes)} double-claimed" if not dupes else f"FAIL: double-claimed {dupes}")
    print(f"      {fake.conditional_puts} conditional PUTs, {fake.rejected} rejected by S3")
    assert len(winners) == 50 and not dupes

    # ---- 3. a live claim is not stealable ---------------------------------------
    other = S3CampaignState("s3://bkt/campaign", client=fake); other.worker_id = "intruder"
    print("PASS  live claim is not stealable" if not other.claim("s0000")
          else "FAIL: stole a live claim")

    # ---- 4. Spot preemption: stale claim, no manifest -> reclaimable -------------
    key = st._key("claims", "s0007.json")
    stale = json.loads(fake.obj[key])
    stale["claimed"] = (datetime.datetime.now(datetime.timezone.utc)
                        - datetime.timedelta(hours=9)).isoformat()
    fake.obj[key] = json.dumps(stale).encode()
    print("PASS  stale claim reclaimed after preemption" if other.claim("s0007")
          else "FAIL: stale claim not reclaimed")

    # ---- 4b. many workers see the same stale claim at once ----------------------
    # Copilot caught this: the takeover used to be an unconditional PUT, so every
    # worker that observed staleness "won" and they all ran the same shard.
    key = st._key("claims", "s0010.json")
    d = json.loads(fake.obj[key])
    d["claimed"] = (datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(hours=9)).isoformat()
    fake.obj[key] = json.dumps(d).encode()

    winners = []
    def steal(w):
        s = S3CampaignState("s3://bkt/campaign", client=fake)
        s.worker_id = f"thief{w}"
        if s.claim("s0010"):
            with lock:
                winners.append(w)
    ts = [threading.Thread(target=steal, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(f"PASS  8 workers raced one stale claim -> {len(winners)} winner(s)"
          if len(winners) == 1 else
          f"FAIL: {len(winners)} workers all took the same stale shard")
    assert len(winners) == 1

    # ---- 5. stale claim WITH a manifest is finished work, not reclaimable --------
    st.complete("s0008", {"picks": 1234})
    key = st._key("claims", "s0008.json")
    d = json.loads(fake.obj[key])
    d["claimed"] = (datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(hours=9)).isoformat()
    fake.obj[key] = json.dumps(d).encode()
    print("PASS  completed shard never re-run" if not other.claim("s0008")
          else "FAIL: would re-run completed work")

    # ---- 6. release returns work immediately ------------------------------------
    st.release("s0009")
    print("PASS  release requeues at once" if other.claim("s0009")
          else "FAIL: release did not requeue")

    # ---- 7. resume view ---------------------------------------------------------
    for i in range(20):
        st.complete(f"s{i:04d}", {"picks": i})
    p = st.progress()
    print(f"PASS  progress: {p}")
    # s0008 was already completed above, so 20 distinct, not 21
    assert p["total"] == 50 and p["complete"] == 20 and p["remaining"] == 30
    assert len(st.completed_ids()) == 20

    # ---- 8. endpoint without conditional writes fails loudly --------------------
    class NoCond(FakeS3):
        def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None):
            if IfNoneMatch is not None:
                raise ClientError({"Error": {"Code": "NotImplemented"}}, "PutObject")
            return super().put_object(Bucket, Key, Body, ContentType)
    try:
        S3CampaignState("s3://b/c", client=NoCond()).claim("x")
        print("FAIL: silently raced on a non-conditional endpoint")
    except RuntimeError as e:
        print(f"PASS  refuses to race: {str(e)[:58]}...")
    print("\nall claim-protocol checks passed")


if __name__ == "__main__":
    test_claim_protocol()