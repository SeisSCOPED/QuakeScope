#!/usr/bin/env python
"""Re-derive every number the audit report quotes, from the exported events.

`export_incident_logs.py` keeps the raw evidence; this reads it back and
recomputes the figures, so each one can be checked rather than trusted. It
parses the actual HTTP request lines rather than grepping for substrings - the
difference matters, because a warning that merely mentions a year is not a
request that carried one, and counting it as one silently changes the headline.

    python scripts/analyse_incident_logs.py ./incident_export

Windows are named so a claim can cite one:

    token-storm    15:34-16:40, the 66 minutes the report calls the storm
    yearless       15:00-16:40, the 100 minutes the year-less table covers
    full-export    everything exported, 14:50-16:45
"""

from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import json
import os
import re
import sys

# Only real outbound requests. httpx logs them in one shape:
#   HTTP Request: GET https://api.earthscope.org/... "HTTP/1.1 403 Forbidden"
REQ = re.compile(
    r'HTTP Request:\s+(?P<method>GET|POST)\s+(?P<url>\S+)\s+"HTTP/[\d.]+\s+'
    r'(?P<status>\d{3})')
NET = re.compile(r"network=FDSN(?:%3A|:)(?P<net>[A-Z0-9]{1,2})")
YEAR = re.compile(r"[?&]year=(?P<year>\d{4})")
TEMPORARY = set("0123456789XYZ")

WINDOWS = {
    "token-storm": ("15:34", "16:40"),
    "yearless":    ("15:00", "16:40"),
    "full-export": ("00:00", "23:59"),
}


def in_window(utc: str, lo: str, hi: str) -> bool:
    hhmm = utc[11:16]
    return lo <= hhmm <= hi


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir")
    ap.add_argument("--window", default="incident",
                    help="which exported window file to read (default incident)")
    a = ap.parse_args(argv)

    path = os.path.join(a.export_dir, "raw", f"{a.window}.jsonl.gz")
    if not os.path.exists(path):
        print(f"no such export: {path}", file=sys.stderr)
        return 1

    tok = {k: 0 for k in WINDOWS}
    tok_min = {k: collections.Counter() for k in WINDOWS}
    cred = {k: 0 for k in WINDOWS}
    yearless = {k: collections.Counter() for k in WINDOWS}
    status = {k: collections.Counter() for k in WINDOWS}
    scopes = collections.Counter()
    n = 0

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            m = REQ.search(e["message"])
            if not m:
                continue
            n += 1
            utc, url, st = e["utc"], m.group("url"), m.group("status")
            minute = utc[11:16]
            for wname, (lo, hi) in WINDOWS.items():
                if not in_window(utc, lo, hi):
                    continue
                if "login.earthscope.org/oauth/token" in url:
                    tok[wname] += 1
                    tok_min[wname][minute] += 1
                    status[wname][f"token {st}"] += 1
                elif "credentials/aws/" in url:
                    cred[wname] += 1
                    status[wname][f"cred {st}"] += 1
                    net = NET.search(url)
                    has_year = YEAR.search(url) is not None
                    if net and not has_year and net.group("net")[0] in TEMPORARY:
                        yearless[wname][net.group("net")] += 1
                    if net:
                        scopes[(net.group("net"),
                                YEAR.search(url).group("year") if has_year else None,
                                st)] += 1

    print(f"parsed {n:,} outbound HTTP request lines from {os.path.basename(path)}\n")
    for wname, (lo, hi) in WINDOWS.items():
        pk = tok_min[wname].most_common(1)
        mins = len(tok_min[wname])
        print(f"[{wname}]  {lo}-{hi} UTC")
        print(f"  token-endpoint POSTs      {tok[wname]:>9,}"
              + (f"   over {mins} min, mean {tok[wname]/mins:,.0f}/min" if mins else ""))
        if pk:
            print(f"  peak minute               {pk[0][1]:>9,}   at {pk[0][0]} UTC")
        print(f"  credential requests       {cred[wname]:>9,}")
        yl = yearless[wname]
        print(f"  YEAR-LESS temporary       {sum(yl.values()):>9,}"
              f"   across {len(yl)} codes")
        if yl:
            top = ", ".join(f"{k} {v:,}" for k, v in yl.most_common(8))
            print(f"     {top}")
        print(f"  statuses                  {dict(status[wname].most_common(8))}")
        print()

    print("distinct credential scopes asked (top 15):")
    for (net, yr, st), k in scopes.most_common(15):
        print(f"  {net:3} {str(yr or 'NO YEAR'):8} -> {st}   {k:,}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
