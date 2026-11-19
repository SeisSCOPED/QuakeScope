"""Convert the PNW EQTransformer pick tars (Ni et al. 2023) to the QuakeScope Parquet layout.

Input: ml_picks_pnw_<YYYY>.tar, one member per station-band-day, named
<DOY>/<NET>.<STA>.<BAND>.<DOY>, each a pickle of a list of seisbench Pick objects.

Output, under <out>/ (mirrors s3://quakescope-picks-2026/<prefix>/):
  picks/network=<NET>/year=<YYYY>/month=<MM>/pnweqt-<srcyear>-<part>.parquet   (partitioned by peak time)
  manifests/pnweqt-<srcyear>.json                                       (files + per station-day records)

The pickles are decoded with a restricted unpickler: only the four classes a
seisbench Pick list needs are allowed, mapped to inert stand-ins, so neither
seisbench nor obspy is imported and nothing else in the pickle can execute.
"""

import argparse
import datetime as dt
import io
import json
import pickle
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RID = "pnw-eqt-ni2023"
SCHEMA = pa.schema([
    ("tid", pa.string()), ("cha", pa.string()), ("pha", pa.string()),
    ("start", pa.timestamp("ms")), ("peak", pa.timestamp("ms")), ("end", pa.timestamp("ms")),
    ("conf", pa.float32()), ("amp", pa.float32()), ("amp_vel", pa.float32()), ("rid", pa.string()),
])
NAME = re.compile(r"^(\d{3})/([A-Z0-9]{1,2})\.([A-Z0-9]{1,5})\.([A-Z]{2})\.(\d{3})$")


class _Obj:
    """Inert stand-in: keeps the pickled state dict, nothing else."""
    def __setstate__(self, state):
        if isinstance(state, tuple):          # (dict, slotstate)
            state = state[0] or {}
        self.__dict__.update(state)


class _Pick(_Obj):
    pass


class _UTC(_Obj):
    pass


def _scalar(dtype, data=None):
    return np.frombuffer(data, dtype=dtype)[0]


def _dtype(*args):
    return np.dtype(args[0])


ALLOWED = {
    ("seisbench.util.annotations", "Pick"): _Pick,
    ("obspy.core.utcdatetime", "UTCDateTime"): _UTC,
    ("numpy.core.multiarray", "scalar"): _scalar,
    ("numpy._core.multiarray", "scalar"): _scalar,
    ("numpy", "dtype"): _dtype,
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refused class {module}.{name}") from None


def _ns(u):
    return int(u.__dict__["_UTCDateTime__ns"])


FLUSH_ROWS = 1_500_000


def _write(cols, out, srcyear, parts, files):
    for (net, yr, mo), c in sorted(cols.items()):
        n = len(c["tid"])
        tbl = pa.table({
            "tid": c["tid"], "cha": c["cha"], "pha": c["pha"],
            "start": pa.array(c["start"], pa.int64()).cast(pa.timestamp("ms")),
            "peak": pa.array(c["peak"], pa.int64()).cast(pa.timestamp("ms")),
            "end": pa.array(c["end"], pa.int64()).cast(pa.timestamp("ms")),
            "conf": pa.array(c["conf"], pa.float32()),
            "amp": pa.nulls(n, pa.float32()), "amp_vel": pa.nulls(n, pa.float32()),
            "rid": [RID] * n,
        }, schema=SCHEMA).sort_by([("tid", "ascending"), ("peak", "ascending")])
        k = parts[(net, yr, mo)]
        parts[(net, yr, mo)] += 1
        rel = f"picks/network={net}/year={yr}/month={mo:02d}/pnweqt-{srcyear}-{k:02d}.parquet"
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, p, compression="zstd")
        files.append({"kind": "picks", "path": rel, "rows": n, "network": net, "year": yr, "month": mo})
    cols.clear()


def convert(tar_path, out, srcyear):
    cols = defaultdict(lambda: defaultdict(list))   # (net, yr, mo) -> column lists
    parts, files = defaultdict(int), []
    records, bad, refused = [], [], 0
    n_members = n_buf = 0
    src = sys.stdin.buffer if str(tar_path) == "-" else open(tar_path, "rb")
    with tarfile.open(fileobj=src, mode="r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            n_members += 1
            g = NAME.match(m.name)
            if not g:
                bad.append(("name", m.name)); continue
            doy, net, sta, band, doy2 = g.groups()
            raw = tf.extractfile(m).read()
            try:
                picks = SafeUnpickler(io.BytesIO(raw)).load()
            except pickle.UnpicklingError as e:
                refused += 1; bad.append(("unpickle", m.name, str(e))); continue
            tid0 = None
            for p in picks:
                d = p.__dict__
                tid = d["trace_id"]
                tid0 = tid0 or tid
                if not tid.startswith(f"{net}.{sta}."):
                    bad.append(("tid", m.name, tid)); continue
                pk = _ns(d["peak_time"])
                t = dt.datetime.utcfromtimestamp(pk / 1e9)
                c = cols[(net, t.year, t.month)]
                c["tid"].append(tid); c["cha"].append(band); c["pha"].append(d["phase"])
                c["start"].append(_ns(d["start_time"]) // 1_000_000)
                c["peak"].append(pk // 1_000_000)
                c["end"].append(_ns(d["end_time"]) // 1_000_000)
                c["conf"].append(float(d["peak_value"]))
            records.append({"tid": tid0 or f"{net}.{sta}.", "cha": band, "yr": srcyear,
                            "doy": int(doy), "npks": len(picks), "rid": RID})
            n_buf += len(picks)
            if n_buf >= FLUSH_ROWS:
                _write(cols, out, srcyear, parts, files)
                n_buf = 0
    _write(cols, out, srcyear, parts, files)
    n_picks = sum(f["rows"] for f in files)

    manifest = {
        "job_id": f"pnweqt-{srcyear}", "run_id": RID, "n_picks": n_picks,
        "station_days": len(records), "members": n_members,
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": files, "records": records, "problems": bad[:200], "n_problems": len(bad),
        "export": {"source": f"ml_picks_pnw_{srcyear}.tar", "exporter": "convert_pnw_eqt_picks.py",
                   "partitioned_by": "peak time (UTC)"},
    }
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / f"pnweqt-{srcyear}.json").write_text(json.dumps(manifest))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", type=Path, help="tar file, or - for stdin")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    srcyear = a.year
    m = convert(a.tar, a.out, srcyear)
    print(json.dumps({k: m[k] for k in ("job_id", "members", "station_days", "n_picks", "n_problems")}))
    for p in m["problems"][:10]:
        print("  problem:", p, file=sys.stderr)


if __name__ == "__main__":
    main()
