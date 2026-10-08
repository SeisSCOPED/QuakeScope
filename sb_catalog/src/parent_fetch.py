"""The parent's first station-day fetch (pick_workload QS_STARTUP2=v2).

While the workers import torch and build their model, the parent - otherwise idle, with a GIL
of its own - LISTs and GETs each worker group's first station-day into a directory under the
run's TMPDIR, the same objects the worker's overlap startup would read itself
(s3_helper.S3DataSource._prefetch_first): the group's first station, its selected channel, the
first day, station-scoped LIST (QS_LIST=station) under the day prefix, every component of the
workload whose key the LIST returned. Each object is written to `<name>.part` and renamed; then
`manifest.json` the same way, then the group's event is set. The worker waits on that event for
a bounded time and parses the local bytes with its own obspy.read and downsample path; a late,
failed or partial fetch is the worker's own LIST and GET, counted (S3DataSource.parent_first).

Standard library only, so the parent stays light: an anonymous HTTPS client for the public SCEDC
bucket (the only archive this covers; any other group gets no parent fetch). The key rules are
s3_helper's SCEDCS3ObjectHelper ones, restated here; the H18 recipe checks them against it.
Nothing outlives the run: the caller removes the directory once the workers have joined.
"""

from __future__ import annotations

import datetime
import http.client
import importlib.util
import json
import os
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

#: SCEDC's public bucket and its regional endpoint (anonymous, virtual-hosted style).
SCEDC_BUCKET = "scedc-pds"
SCEDC_HOST = "scedc-pds.s3.us-west-2.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
#: Socket timeout per request; the worker stops waiting long before this.
TIMEOUT = 10.0
CHUNK = 1 << 20

_constants = None


def constants():
    """sb_catalog/src/constants.py (NETWORK_MAPPING, select_channel), loaded by path."""
    global _constants
    if _constants is None:
        spec = importlib.util.spec_from_file_location(
            "_qs_constants", Path(__file__).with_name("constants.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _constants = module
    return _constants


def scedc_prefix(year: str, day: str) -> str:
    """SCEDCS3ObjectHelper.get_prefix."""
    return f"{SCEDC_BUCKET}/continuous_waveforms/{year}/{year}_{day}/"


def scedc_stem(net: str, sta: str) -> str:
    """SCEDCS3ObjectHelper.station_stem."""
    return f"{net}{sta.ljust(5, '_')}"


def scedc_basename(net, sta, loc, cha, year, day, comp) -> str:
    """SCEDCS3ObjectHelper.get_basename."""
    return f"{net}{sta.ljust(5, '_')}{cha}{comp}{loc.ljust(3, '_')}{year}{day}.ms"


def first_station_day(stations: list[dict], start: str, components: str) -> dict | None:
    """What a worker's overlap startup reads first for a group: its first station on the first
    day, as {key: [net, prefix, stem], uris: [...]}; None when the parent does not cover it."""
    if not stations:
        return None
    station = stations[0]
    net, sta, loc = station["id"].split(".")
    c = constants()
    if c.NETWORK_MAPPING.get(net) != "scedc":
        return None
    channel = c.select_channel(station["channels"].split(","))
    if channel is None:
        return None
    day = datetime.datetime.strptime(start, "%Y.%j")
    year, doy = day.strftime("%Y"), day.strftime("%j")
    prefix = scedc_prefix(year, doy)
    return {
        "station": station["id"],
        "key": [net, prefix, scedc_stem(net, sta)],
        "uris": [prefix + scedc_basename(net, sta, loc, channel, year, doy, comp)
                 for comp in components],
    }


def _request(path: str) -> http.client.HTTPResponse:
    conn = http.client.HTTPSConnection(SCEDC_HOST, timeout=TIMEOUT)
    conn.request("GET", path)
    response = conn.getresponse()
    if response.status != 200:
        body = response.read(512)
        conn.close()
        raise OSError(f"GET {path}: HTTP {response.status} {body[:200]!r}")
    return response


def list_station_day(prefix: str, stem: str) -> tuple[list[str], int]:
    """S3Helper.list_station_day over HTTPS: ListObjectsV2 with the stem as the Prefix and "/" as
    the delimiter, every page; sorted `bucket/key` names, and the request count."""
    bucket, _, key = prefix.partition("/")
    params = {"list-type": "2", "prefix": f"{key}{stem}", "delimiter": "/"}
    names, requests = [], 0
    while True:
        response = _request("/?" + urllib.parse.urlencode(params))
        root = ET.fromstring(response.read())
        requests += 1
        names += [f"{bucket}/{item.findtext(S3_NS + 'Key')}"
                  for item in root.iter(S3_NS + "Contents")]
        if root.findtext(S3_NS + "IsTruncated") != "true":
            return sorted(names), requests
        params["continuation-token"] = root.findtext(S3_NS + "NextContinuationToken")


def get_object(uri: str, target: Path, marks: list | None = None) -> int:
    """GET one `bucket/key` into `target` (via `target.part`); returns its size. `marks`, when
    given, gets the wall times of the request's start, its response headers and its end."""
    _, _, key = uri.partition("/")
    if marks is not None:
        marks.append(time.time())
    response = _request("/" + urllib.parse.quote(key))
    if marks is not None:
        marks.append(time.time())
    expected = response.getheader("Content-Length")
    part = target.with_name(target.name + ".part")
    size = 0
    with open(part, "wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            size += len(chunk)
    if expected is not None and int(expected) != size:
        os.unlink(part)
        raise OSError(f"GET {uri}: {size} of {expected} bytes")
    os.replace(part, target)
    if marks is not None:
        marks.append(time.time())
    return size


class FirstSDFetch:
    """One fetch thread per worker group (plus one per object); see the module docstring."""

    def __init__(self, root: str, groups: list[list[dict]], workload: dict, events: list):
        self.root = Path(root)
        self.events = events
        self.plans = []
        for group in groups:
            try:
                plan = first_station_day(group, workload["start"], workload["components"])
            except Exception:            # e.g. a bad date: the worker raises it, as before
                plan = None
            self.plans.append(plan)
        self.dirs = [str(self.root / f"group-{i}") for i in range(len(groups))]
        #: Per group: {"source": parent|none|failed, "list_s", "ready_s", "objects", "bytes", "error"}.
        self.report: list[dict] = [{"source": "none"} for _ in groups]
        #: Per group, per wanted object in order: [start, headers, end] wall times (startup-diag).
        self.object_marks: list[list[list[float]]] = [[] for _ in groups]
        self.t0 = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self.t0 = time.time()
        for index, plan in enumerate(self.plans):
            if plan is None:
                self.events[index].set()            # nothing to wait for
                continue
            thread = threading.Thread(target=self._fetch_group, args=(index, plan),
                                      name=f"qs-firstsd-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def join(self, timeout: float) -> None:
        deadline = time.time() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.time()))

    def _fetch_group(self, index: int, plan: dict) -> None:
        report = self.report[index]
        folder = Path(self.dirs[index])
        manifest = {"station": plan["station"], "key": plan["key"], "listing": None,
                    "objects": {}, "errors": []}
        try:
            folder.mkdir(parents=True, exist_ok=True)
            net, prefix, stem = plan["key"]
            listing, requests = list_station_day(prefix, stem)
            report["list_s"] = round(time.time() - self.t0, 3)
            manifest["listing"] = listing
            manifest["list_requests"] = requests
            wanted = [uri for uri in plan["uris"] if uri in listing]
            results: dict = {}
            marks = self.object_marks[index]

            def fetch(uri, times):
                name = uri.rsplit("/", 1)[1]
                try:
                    results[uri] = (name, get_object(uri, folder / name, times), None)
                except Exception as exc:             # the worker reads this one itself
                    results[uri] = (None, 0, f"{type(exc).__name__}: {exc}")

            marks += [[] for _ in wanted]
            threads = [threading.Thread(target=fetch, args=(uri, times), daemon=True)
                       for uri, times in zip(wanted, marks)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for uri in wanted:
                name, size, error = results[uri]
                if error is None:
                    manifest["objects"][uri] = name
                    report["bytes"] = report.get("bytes", 0) + size
                else:
                    manifest["errors"].append(f"{uri}: {error}")
            report["objects"] = len(manifest["objects"])
            report["wanted"] = len(wanted)
            report["source"] = "parent" if manifest["objects"] else "failed"
        except Exception as exc:
            manifest["errors"].append(f"{type(exc).__name__}: {exc}")
            report["source"] = "failed"
        finally:
            try:
                part = folder / "manifest.json.part"
                part.write_text(json.dumps(manifest))
                os.replace(part, folder / "manifest.json")
            except OSError as exc:
                manifest["errors"].append(f"manifest: {exc}")
                report["source"] = "failed"
            report["errors"] = len(manifest["errors"])
            report["ready_s"] = round(time.time() - self.t0, 3)
            self.events[index].set()

    def summary(self, index: int) -> str:
        """The group's report, then `objs=start/headers/end,...` per object, seconds from start()."""
        r = self.report[index]
        objs = ",".join("/".join(f"{t - self.t0:.3f}" for t in times) or "-"
                        for times in self.object_marks[index]) if self.t0 is not None else ""
        return " ".join(f"{k}={v}" for k, v in r.items()) + f" objs={objs or '-'}"
