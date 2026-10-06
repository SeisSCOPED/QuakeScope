#!/usr/bin/env python
"""Add an `epochs` column to a station table, from FDSN channel epochs.

`start_date`/`end_date` are the hull of a station-location's epochs, and the
planner used to plan the hull: XA.AZ01 (a 1993 deployment and a 2017 one) was
planned for the 23 years between them. With `epochs` present the planner
(`shard_planner.parse_epochs`) plans each window separately.

An epoch here is the union of the channel epochs of the station-location's
pickable vertical channels (bands in constants.channel_priority for the weight),
merged where they touch. Metadata only: the archive may still hold nothing for
part of an epoch, which the reader records as `no_data`.

Read-only against FDSN; writes a local file. The station service is chosen by
the network's data center: SCEDC for CI, NCEDC for its networks, EarthScope for
the rest. One request per network.

    python scripts/add_station_epochs.py --table s3://.../western/stations.parquet \
        --out stations_with_epochs.parquet [--networks UU,TA]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sb_catalog.src.constants import NETWORK_MAPPING, channel_priority  # noqa: E402
from sb_catalog.src.utils import normalize_station_codes  # noqa: E402

SERVICE = {
    "scedc": "https://service.scedc.caltech.edu/fdsnws/station/1/query",
    "ncedc": "https://service.ncedc.org/fdsnws/station/1/query",
    "earthscope": "https://service.earthscope.org/fdsnws/station/1/query",
}
OPEN_END = datetime.date(2599, 12, 31)


def fetch(net: str, stations: list[str], retries: int = 4) -> str:
    dc = NETWORK_MAPPING.get(net, "earthscope")
    url = SERVICE.get(dc, SERVICE["earthscope"])
    # Ask for the station codes in the table only, in chunks: a whole-network
    # request for TA or UU returns tens of thousands of channel epochs.
    out = []
    for i in range(0, len(stations), 150):
        q = (f"{url}?net={net}&sta={','.join(stations[i:i + 150])}"
             f"&cha=??Z&level=channel&format=text&nodata=404")
        for k in range(retries):
            try:
                out.append(urllib.request.urlopen(q, timeout=180).read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code in (204, 404):
                    break
                time.sleep(2 ** k)
            except Exception:
                time.sleep(2 ** k)
    return "\n".join(out)


def epochs_from_text(text: str, bands: set[str]) -> dict[str, list]:
    raw: dict[str, list] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("|")
        if len(f) < 17 or f[3][:2] not in bands:
            continue
        sid = f"{f[0]}.{f[1]}.{f[2]}"
        a = datetime.date.fromisoformat(f[15][:10])
        b = datetime.date.fromisoformat(f[16][:10]) if f[16].strip() else OPEN_END
        raw.setdefault(sid, []).append((a, b))
    merged = {}
    for sid, w in raw.items():
        w.sort()
        cur = [list(w[0])]
        for a, b in w[1:]:
            if a <= cur[-1][1] + datetime.timedelta(days=1):
                cur[-1][1] = max(cur[-1][1], b)
            else:
                cur.append([a, b])
        merged[sid] = cur
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--networks", help="comma-separated subset")
    ap.add_argument("--weight", default="original")
    a = ap.parse_args()

    st = normalize_station_codes(pd.read_parquet(a.table))
    if a.networks:
        st = st[st.network_code.isin(a.networks.split(","))].copy()
    bands = set(channel_priority(a.weight))
    by_net = st.groupby("network_code").station_code.apply(lambda s: sorted(set(s))).to_dict()
    found: dict[str, list] = {}
    with cf.ThreadPoolExecutor(6) as ex:
        for net, txt in zip(by_net, ex.map(lambda n: fetch(n, by_net[n]), by_net)):
            found.update(epochs_from_text(txt, bands))
    st["epochs"] = [";".join(f"{x}/{y}" for x, y in found[i]) if i in found else None
                    for i in st["id"]]
    have = st["epochs"].notna()
    print(f"{len(st):,} station-locations; epochs found for {have.sum():,}; "
          f"{(~have).sum():,} keep their start/end hull")
    multi = st[have & st["epochs"].str.contains(";", na=False)]
    print(f"{len(multi):,} have more than one epoch (gaps the hull used to plan)")
    if a.out.endswith(".csv"):
        st.to_csv(a.out, index=False)
    else:
        st.to_parquet(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
