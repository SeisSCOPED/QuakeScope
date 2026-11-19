"""Build stations.parquet for pnw-eqt-2023 from PNWStationXML.tar (one StationXML per station).

One row per NET.STA.LOC, the same columns as western/stations.parquet. start/end are
the earliest channel start and latest channel end at that location; `channels` lists
the two-letter band codes. Reads the tar as a stream (tar path or - for stdin).
"""
import sys
import tarfile
from collections import defaultdict

import pandas as pd
from lxml import etree

PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)


def local(tag):
    return tag.rsplit("}", 1)[-1]


def child(el, name):
    for c in el:
        if local(c.tag) == name:
            return c
    return None


def text(el, name):
    c = child(el, name)
    return c.text.strip() if c is not None and c.text else None


def main(src, out):
    rows = defaultdict(lambda: {"bands": set(), "start": None, "end": None})
    n_xml = bad = 0
    fh = sys.stdin.buffer if src == "-" else open(src, "rb")
    with tarfile.open(fileobj=fh, mode="r|*") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".xml") or "/._" in m.name:
                continue
            n_xml += 1
            try:
                root = etree.fromstring(tf.extractfile(m).read(), PARSER)
            except etree.XMLSyntaxError:
                bad += 1
                continue
            for net in (e for e in root if local(e.tag) == "Network"):
                ncode = net.get("code")
                for sta in (e for e in net if local(e.tag) == "Station"):
                    scode = sta.get("code")
                    for ch in (e for e in sta if local(e.tag) == "Channel"):
                        loc = (ch.get("locationCode") or "").strip()
                        r = rows[(ncode, scode, loc)]
                        r["bands"].add(ch.get("code")[:2])
                        r["lat"] = float(text(ch, "Latitude"))
                        r["lon"] = float(text(ch, "Longitude"))
                        r["elev"] = float(text(ch, "Elevation"))
                        s, e = ch.get("startDate"), ch.get("endDate") or "3000-01-01"
                        r["start"] = min(r["start"] or s, s)
                        r["end"] = max(r["end"] or e, e)
    out_rows = []
    for (n, s, l), r in sorted(rows.items()):
        sd, ed = pd.Timestamp(r["start"][:10]), pd.Timestamp(r["end"][:10])
        out_rows.append({
            "id": f"{n}.{s}.{l}", "network_code": n, "station_code": s, "location_code": l,
            "channels": ",".join(sorted(r["bands"])), "latitude": r["lat"], "longitude": r["lon"],
            "elevation": r["elev"], "start_date": sd.strftime("%Y-%m-%d"), "end_date": ed.strftime("%Y-%m-%d"),
            "state": None, "start_yearday": float(f"{sd.year}.{sd.dayofyear:03d}"),
            "end_yearday": float(f"{ed.year}.{ed.dayofyear:03d}"),
        })
    df = pd.DataFrame(out_rows)
    df.to_parquet(out, index=False)
    print(f"{n_xml} xml files, {bad} unparseable, {len(df)} station-locations, "
          f"{df.network_code.nunique()} networks -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
