"""Waveforms for the western-US sequence track, into the bundle's layout.

The us_sequences notebook fetches at run time and keeps nothing, so the track
has picks and no waveforms. This stores the same windows and stations the
notebook scored, so the evaluation set can ship with its data.
"""
import io, sys
from pathlib import Path
import obspy, pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from s3fs import S3FileSystem

OUT = Path("docs/benchmark/results/us_sequences/waveforms"); OUT.mkdir(parents=True, exist_ok=True)
S3_BUCKET = {"CI": "scedc", "BK": "ncedc", "NC": "ncedc", "NP": "ncedc"}
ROUTE = {"CI": "SCEDC", "BK": "NCEDC", "NC": "NCEDC", "NP": "NCEDC",
         "UW": "EARTHSCOPE", "NN": "EARTHSCOPE", "LB": "EARTHSCOPE", "UO": "EARTHSCOPE"}
PREF = ["HH", "BH", "EH"]
WINDOW_START = 600
# origin (UTC), window length in minutes, and the offset from the origin at
# which the window opens. The default 600 s skips the mainshock coda so the
# window scores the aftershock sequence. Monroe is an M4.6 whose aftershocks
# are sparse and spread over days: skipping its mainshock leaves one arrival,
# so its window opens before the origin and the mainshock is the sequence.
SEQ = {
    "Ridgecrest":     ("2019-07-06T03:19:53", 30, WINDOW_START),
    "San Simeon":     ("2003-12-22T19:15:56", 120, WINDOW_START),
    "Monte Cristo":   ("2020-05-15T11:03:27", 120, WINDOW_START),
    "Mendocino 2024": ("2024-12-05T18:44:22", 120, WINDOW_START),
    "Monroe WA":      ("2019-07-12T09:51:38", 120, -60),
}
_fs = S3FileSystem(anon=True); _lst = {}; _cl = {}

def listing(bucket, net, year, doy):
    pre = (f"scedc-pds/continuous_waveforms/{year}/{year}_{doy:03d}/" if bucket == "scedc"
           else f"ncedc-pds/continuous_waveforms/{net}/{year}/{year}.{doy:03d}/")
    if pre not in _lst:
        try: _lst[pre] = {k.split("/")[-1] for k in _fs.ls(pre)}
        except Exception: _lst[pre] = set()
    return pre, _lst[pre]

def from_bucket(net, sta, cha, t0, t1):
    b = S3_BUCKET.get(net)
    if b is None: return None
    pre, names = listing(b, net, t0.year, int(t0.strftime("%j")))
    if not names: return None
    st = obspy.Stream()
    for comp in "ZNE":
        if b == "scedc":
            head, tail = f"{net}{sta.ljust(5,'_')}{cha}{comp}", f"{t0.year}{int(t0.strftime('%j')):03d}.ms"
        else:
            head, tail = f"{sta}.{net}.{cha}{comp}.", f".D.{t0.year}.{int(t0.strftime('%j')):03d}"
        hits = [n for n in names if n.startswith(head) and n.endswith(tail)]
        if not hits: return None
        with _fs.open(pre + sorted(hits)[0]) as fh:
            st += obspy.read(io.BytesIO(fh.read()))
    st.merge(fill_value=0); st.trim(t0, t1)
    return st if len(st) >= 3 else None

def from_fdsn(net, sta, cha, t0, t1):
    prov = ROUTE.get(net, "EARTHSCOPE")
    if prov not in _cl: _cl[prov] = Client(prov, timeout=180)
    try:
        st = _cl[prov].get_waveforms(net, sta, "*", f"{cha}?", t0, t1)
        st.merge(fill_value=0); st.trim(t0, t1)
        return st if len(st) >= 3 else None
    except Exception:
        return None

ref = pd.read_csv("docs/benchmark/results/us_sequences/reference_picks.csv")
mon = pd.read_csv("/private/tmp/claude-501/-Users-marinedenolle-GitHub-QuakeScope/"
                  "8f282710-68a1-4cdd-b77e-e07856ef31af/scratchpad/monroe_picks.csv")
# Candidates in order of how many arrivals they carry. A three-component
# picker cannot use a vertical-only short-period station, and the PNSN regional
# network is full of them, so take the first six that actually return 3C rather
# than the first six by arrival count.
want = {}
for seq in SEQ:
    s = ref[ref.sequence == seq] if seq != "Monroe WA" else mon
    want[seq] = list(s.station.value_counts().head(40).index)
N_WANT = 6

rows = []
for seq, (t, mins, _ws) in SEQ.items():
    t0 = UTCDateTime(t) + _ws; t1 = t0 + mins * 60
    print(f"{seq}  {t0} .. {t1}", flush=True)
    kept = 0
    for station in want[seq]:
        if kept >= N_WANT:
            break
        net, sta = station.split(".")[0], station.split(".")[1]
        got = None
        for cha in PREF:
            got = from_bucket(net, sta, cha, t0, t1) or from_fdsn(net, sta, cha, t0, t1)
            if got is not None: break
        if got is None:
            print(f"   {station:12s} skipped: no three-component data in the window", flush=True)
            continue
        kept += 1
        key = seq.lower().replace(" ", "")
        p = OUT / f"{key}_{net}{sta}.mseed"
        got.write(str(p), format="MSEED")
        rows.append(dict(sequence=seq, station=f"{net}.{sta}", file=p.name,
                         start=str(t0), end=str(t1),
                         sampling_rate=float(got[0].stats.sampling_rate),
                         channels=",".join(sorted({tr.stats.channel for tr in got}))))
        print(f"   {station:12s} {len(got)} traces", flush=True)
pd.DataFrame(rows).to_csv("docs/benchmark/results/us_sequences/stations.csv", index=False)
print(f"\n{len(rows)} station-windows stored")
