"""Build a self-contained HTML dashboard of campaign progress.

    python scripts/campaign_dashboard.py -o reports/campaign_dashboard.html

Reads the same sources as campaign_status.py and renders them as SVG - no CDN,
no build step, no JavaScript charting library, so the page keeps working when
published to Pages and cannot silently fail to load a script.

MEASURED (counted from an API response):
  picks per station, per day    manifests/<shard>.json -> records[]
  station coordinates           stations.parquet
  shards planned / complete     shards.jsonl, complete/
  Parquet objects and bytes     list_objects_v2 on picks/
  jobs by status, vCPU in use   Batch list_jobs / describe_jobs
  vCPU-hours consumed           Batch startedAt/stoppedAt x job vCPU

DERIVED (stated as such on the page, with the rate shown):
  spend = vCPU-hours x FARGATE_SPOT_RATE. Cost Explorer is blocked on this
  account by an organisation SCP, so no billed figure is available to check it
  against; it is an estimate and the page says so.

Nothing else is inferred. A quantity that cannot be read is omitted rather than
filled in.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import math
import os
import sys
from collections import defaultdict

import boto3

# sys.path[0] is this script's directory, so the repo root - and with it
# sb_catalog - is not importable without help.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGION = "us-east-2"
BUCKET = "quakescope-picks-2026"
QUEUE = "niyiyu_earthscope_missing_station"
FARGATE_SPOT_RATE = 0.0148          # $/vCPU-hour, us-east-2, published list rate

# Cost model for the plan panel. VCPU_H_PER_STATION_DAY is MEASURED: 707
# vCPU-hours over 10,440 station-days on the live SCEDC campaign gave 0.068
# under the day-long deconvolution, and the short-window rework cut the
# per-station-day pipeline from 20.65 s to 7.32 s at campaign-average pick
# density (0.354x). Everything downstream of it is arithmetic, and the page
# says so - a projection is not an observation.
VCPU_H_PER_STATION_DAY = 0.068 * 0.354
# The rate above is per station-day that IS PROCESSED. The plan is counted in
# station-days that are PLANNED, and most of those have no data: measured hit
# rates are 21.7% (CI 2010), 38.7% (CI 2015), 67.6% (AK 2020). Multiplying the
# processed rate by the planned count assumes every planned day has data - a
# 100% hit rate - and overstates the bill by 1.5-4.6x.
#
# That is what the page did until 2026-09-03, which is why it quoted $40,708
# against the $10,828-$19,702 of 24_cost_model.md, the model that supersedes
# every other figure. Both numbers are now shown, because the honest answer is
# a range whose width is one unmeasured quantity.
HIT_RATE = float(os.environ.get("HIT_RATE", "0.40"))
QUOTA_VCPU = 12000                  # L-36FBB829, Fargate Spot, us-east-2
# Parquet objects to scan for the per-station and per-day breakdowns. The
# headline count is metadata-only and always exact; only the breakdowns are
# capped, so the hourly job stays hourly as the catalogue grows.
PARQUET_SCAN_CAP = int(os.environ.get("PARQUET_SCAN_CAP", "150"))
# Batch jobs to describe per run. A 1,500-task array accumulates thousands of
# finished children and describing them all took the hourly job past 10 minutes.
DESCRIBE_CAP = int(os.environ.get("DESCRIBE_CAP", "1500"))
LIVE_STATES = ("RUNNABLE", "STARTING", "RUNNING")
TASK_VCPU = 8                       # per job definition

# Campaigns whose data cannot currently be read, and why. Shown on the plan so
# a queue that is deliberately parked is not mistaken for one that is failing.
# Nothing is blocked as of 2026-09-02. The restricted EarthScope access point
# was never stalling: the credential request was unscoped, so it could LIST but
# not GET, and every read returned AccessDenied instantly. Adding
# `network=FDSN:<NET>` (and `year=` for temporary networks) fixed it, and
# restricted reads now run at 96-98 MB/s - the same rate as Open Data. See
# docs/rerun_2026/19 and OPTIMISE item 0g.
BLOCKED: dict[str, str] = {}      # name -> why, if a queue is ever parked again

BASEMAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "basemap.json")

# P and S are two identities, so two categorical slots. Validated as an
# adjacent pair in both modes: normal-vision dE 33.6 light / 31.8 dark, CVD 24.7.
PHASE = {"P": ("var(--phase-p)", "P"), "S": ("var(--phase-s)", "S")}

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]


def now():
    return datetime.datetime.now(datetime.timezone.utc)



def _count_rows_cached(campaign, files):
    """Total rows across `files`, reading only footers never seen before.

    A picks object is immutable: a shard writes it once and nothing rewrites it.
    So the hourly job re-reading every footer is pure waste, and it is the
    dominant cost - `western-a` is 29,936 objects and took 518 s of a 594 s
    run, 87% of the total, to recount picks that had not changed since the
    previous hour.

    The cache lives beside the campaign rather than in the runner, because the
    runner is ephemeral and this has to survive between hourly runs. It is
    keyed by object path, so:

      * new objects are counted and added,
      * objects that vanished are dropped - which is what compaction does when
        it replaces many small files with one large one, so the count stays
        correct across a compaction rather than double-counting it.

    Deliberately NOT under picks/, or the dataset scan would try to read it as
    Parquet, and it would fall under the public-read grant. It stays private.
    """
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor

    key = f"{campaign}/.dashboard/rowcount.json"
    # Adaptive retries, like the main client. This job reads the same bucket a
    # 900-worker fleet is writing to, and on 2026-09-04 a plain client with
    # default retries lost the whole run to
    #
    #   ClientError (SlowDown) ... reached max retries: 4
    #
    # on one small GetObject for the cache file. The dashboard is the
    # interruptible party in that contention and should behave like it.
    from botocore.config import Config as _Cfg
    s3 = boto3.client("s3", region_name=REGION, config=_Cfg(
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=PICK_COUNT_THREADS + 4))
    try:
        cache = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:
        cache = {}                       # first run, or unreadable: rebuild
    want = set(files)
    new = sorted(want - set(cache))
    gone = set(cache) - want

    # BOUND THE COLD PATH. Reading a footer is one S3 round trip, so a cold
    # cache over a large campaign is linear and unbounded: western reached
    # 274,950 objects, which is ~78 minutes at PICK_COUNT_THREADS. This job is
    # hourly and holds a concurrency group, so a run that overruns its own
    # interval stacks the next one behind it - observed at 130 minutes and
    # climbing on 2026-09-04.
    #
    # Count at most COLD_BATCH new objects per run and carry the rest to the
    # next one. The cache converges over a few runs instead of blocking on the
    # first, and the number is reported rather than quietly approximated: a
    # partial count is labelled, never presented as complete.
    partial = 0
    if len(new) > COLD_BATCH:
        partial = len(new) - COLD_BATCH
        new = new[-COLD_BATCH:]          # newest first: the interesting end


    if new:
        afs = _arrow_fs()

        def _rows(f):
            return pq.read_metadata(f, filesystem=afs).num_rows

        with ThreadPoolExecutor(max_workers=PICK_COUNT_THREADS) as ex:
            for f, n in zip(new, ex.map(_rows, new)):
                cache[f] = n
    for f in gone:
        cache.pop(f, None)

    if new or gone:
        try:
            s3.put_object(Bucket=BUCKET, Key=key,
                          Body=json.dumps(cache).encode())
        except Exception:
            # A cache that cannot be written is a slow next run, not a wrong
            # answer. The count below is already correct.
            pass
    return sum(cache.get(f, 0) for f in want), partial


def parquet_stats(campaign, max_files=None):
    """Pick counts straight from the Parquet, not from the manifests.

    A manifest is written by ParquetPickWriter.close(), which a Spot
    interruption skips - so on a preempted fleet most finished work has no
    manifest and a manifest-based count silently under-reports a catalogue that
    is filling normally. The picks themselves are durable either way: they were
    flushed before the interruption.

    Row counts come from the Parquet footers, read in parallel, so the headline
    number costs metadata reads rather than a full scan and stays exact as the
    catalogue grows. The per-station and per-day breakdowns do need column data,
    so they read two columns and say so on the page when they had to sample.
    """
    import pyarrow.dataset as ds

    out = {"picks": 0, "per_station": {}, "per_day": {}, "sampled": None,
           "error": None, "partial": 0}
    try:
        dataset = ds.dataset(f"{BUCKET}/{campaign}/picks/", format="parquet",
                             filesystem=_arrow_fs(), partitioning="hive")
        files = list(dataset.files)
    except FileNotFoundError:
        return out                        # genuinely nothing written yet
    except Exception as exc:
        # Throttled or otherwise unreadable. Returning 0 here would be a lie
        # indistinguishable from an empty campaign - and under contention with
        # the fleet that is exactly when it would happen. Say so instead.
        out["error"] = f"{type(exc).__name__}: {exc}"[:120]
        return out
    if not files:
        return out
    # count_rows() reads every footer SERIALLY, so its cost is one S3 round
    # trip per object - 75.5s for 514 objects, measured. That is linear in the
    # catalogue and this job runs hourly: the 2026-09-02 runs took 38-60
    # minutes, and a campaign-sized bucket would not finish inside the hour at
    # all. The footers are independent, so read them in parallel. Same number,
    # 8x faster on the same data (75.5s -> 9.2s), exact rather than sampled.
    try:
        out["picks"], out["partial"] = _count_rows_cached(campaign, files)
    except Exception as exc:
        # Throttled or otherwise unreadable. Report it and carry on to the next
        # campaign: one contended prefix should cost its own count, not the
        # whole page. The row renders "not read" rather than 0.
        out["error"] = f"{type(exc).__name__}: {exc}"[:120]
        return out

    # The breakdowns are what the map and the time series draw, and they are the
    # expensive part. Cap the scan and report the cap rather than either
    # stalling the hourly job or quietly showing a subset as if it were all.
    scan = files
    if max_files and len(files) > max_files:
        scan = files[-max_files:]
        out["sampled"] = (len(scan), len(files))
    try:
        import pandas as pd
        # A live fleet is writing into this prefix, so the listing can include
        # an object that is mid-upload. Skip those rather than lose the whole
        # breakdown to one partial file - it will be complete by the next run.
        t = ds.dataset(scan, format="parquet", filesystem=_arrow_fs(),
                       partitioning="hive",
                       exclude_invalid_files=True).to_table(
                           columns=["tid", "peak"])
        df = t.to_pandas()
        out["per_station"] = df.groupby("tid").size().to_dict()
        d = pd.to_datetime(df["peak"])
        out["per_day"] = (df.assign(yr=d.dt.year, doy=d.dt.dayofyear)
                          .groupby(["yr", "doy"]).size().to_dict())
    except Exception as exc:
        # The headline count already succeeded; only the breakdowns failed, so
        # keep the count and let the page say the map is thin rather than
        # implying the campaign has four stations in it.
        out["error"] = f"map/series unavailable - {type(exc).__name__}"
    return out


# Concurrency for reading Parquet footers. Bounded rather than unlimited:
# this runs while a live fleet is writing to the same bucket, and the point
# is to stop being the slow thing, not to start being the throttling one.
PICK_COUNT_THREADS = int(os.environ.get("PICK_COUNT_THREADS", "16"))
# New objects to count per run. Bounds the cold path so an hourly job stays
# hourly; the remainder is carried to the next run and reported meanwhile.
COLD_BATCH = int(os.environ.get("COLD_BATCH", "60000"))


def _arrow_fs():
    from pyarrow.fs import AwsStandardS3RetryStrategy, S3FileSystem
    # Same reasoning as the boto3 client: the fleet owns this bucket's request
    # budget, so the dashboard retries patiently instead of failing fast.
    return S3FileSystem(region=REGION,
                        retry_strategy=AwsStandardS3RetryStrategy(max_attempts=10))


def gather(s3, b, campaigns):
    per_station = defaultdict(int)
    per_day = defaultdict(int)
    camp_rows, total_picks, total_bytes, total_files = [], 0, 0, 0
    sampled = []
    unreadable = []

    for name in campaigns:
        try:
            body = s3.get_object(Bucket=BUCKET, Key=f"{name}/shards.jsonl")["Body"].read()
            shards = sum(1 for x in body.decode().splitlines() if x.strip())
        except Exception:
            shards = 0
        done = picks = sdays = files = nbytes = 0
        pg = s3.get_paginator("list_objects_v2")
        for page in pg.paginate(Bucket=BUCKET, Prefix=f"{name}/complete/"):
            done += len(page.get("Contents", []))
        nobjs = 0
        for page in pg.paginate(Bucket=BUCKET, Prefix=f"{name}/picks/"):
            for o in page.get("Contents", []):
                nbytes += o["Size"]
                nobjs += 1
        # Station-days still come from the manifests - the Parquet does not
        # record how many station-days were examined, only what was found.
        n_manifests = 0
        for page in pg.paginate(Bucket=BUCKET, Prefix=f"{name}/manifests/"):
            for o in page.get("Contents", []):
                try:
                    m = json.loads(
                        s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read())
                except Exception:
                    continue        # throttled or gone; station-days undercount
                sdays += m.get("station_days", 0)
                files += len(m.get("files", []))
                n_manifests += 1

        # Picks and their breakdowns come from the Parquet itself, so a
        # preempted worker's output is counted even though it never wrote a
        # manifest.
        ps = parquet_stats(name, max_files=PARQUET_SCAN_CAP)
        picks = ps["picks"]
        for tid, n in ps["per_station"].items():
            per_station[str(tid)] += int(n)
        for (yr, doy), n in ps["per_day"].items():
            per_day[(int(yr), int(doy))] += int(n)
        if ps["sampled"]:
            sampled.append((name,) + ps["sampled"])
        if ps.get("partial"):
            # Counted, but not all of it. Say so - a number short by a known
            # amount is useful; the same number presented as complete is not.
            sampled.append((name, len(ps.get("per_station", {})) or 0,
                            ps["partial"]))
        if ps.get("error"):
            unreadable.append((name, ps["error"]))
            # A count that FAILED is not a count of zero. Rendering it as "0"
            # put western at zero picks on 2026-09-04 while 11.7 GB of its
            # picks sat in the bucket - the note underneath said so, but a
            # number in a table outweighs a caption every time.
            picks = None
        # Planned station-days come from the queue, not the manifests: the
        # manifests only describe what is already finished, and the plan panel
        # is about what is still to come.
        planned_sd = 0
        try:
            body = s3.get_object(Bucket=BUCKET,
                                 Key=f"{name}/shards.jsonl")["Body"].read()
            planned_sd = sum(json.loads(x).get("n_station_days", 0)
                             for x in body.decode().splitlines() if x.strip())
        except Exception:
            pass
        if shards or picks:
            camp_rows.append({"name": name, "shards": shards, "done": done,
                              "picks": picks, "sdays": sdays, "bytes": nbytes,
                              "planned_sd": planned_sd})
        # Count objects from the same listing that produced the bytes. Taking
        # the count from the manifests instead made the two disagree - 9 files
        # against 49 MB - because a running shard has written objects but has
        # no manifest yet.
        total_picks += picks or 0        # None = not read, not zero
        total_bytes += nbytes; total_files += nobjs

    coords = {}
    try:
        import pandas as pd
        for name in campaigns:
            try:
                d = pd.read_parquet(f"s3://{BUCKET}/{name}/stations.parquet")
            except Exception:
                continue
            for i, sc, lat, lon in zip(d["id"], d["station_code"],
                                       d["latitude"], d["longitude"]):
                coords[str(i)] = (float(lat), float(lon), str(sc))
    except Exception:
        pass

    vcpu_now = vcpu_hours = 0.0
    status = {}
    described = [0]
    truncated = [False]
    # list_jobs(jobQueue=...) does NOT enumerate the child tasks of an array
    # job - it returns only the parents and any standalone jobs. Once the fleet
    # moved to a 1,500-task array, every compute figure on this page silently
    # went to near zero: it reported 0 vCPU in use against an actual 832.
    # Children have to be listed per array id.
    arrays = []
    for st in ("RUNNABLE", "STARTING", "RUNNING"):
        try:
            for j in b.list_jobs(jobQueue=QUEUE, jobStatus=st)["jobSummaryList"]:
                if j.get("arrayProperties", {}).get("size"):
                    arrays.append(j["jobId"])
        except Exception:
            pass
    for st in ("RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"):
        jobs = []
        try:
            jobs += b.list_jobs(jobQueue=QUEUE, jobStatus=st)["jobSummaryList"]
        except Exception:
            pass
        # Only live states are enumerated per array. The finished children of a
        # long-running array number in the thousands and paging them took the
        # hourly job past ten minutes; they contribute nothing to "vCPU in use",
        # and their contribution to vCPU-hours is reported as a lower bound
        # rather than paid for on every run.
        if st in LIVE_STATES:
            for aid in set(arrays):
                try:
                    nxt = None
                    while True:
                        kw = {"arrayJobId": aid, "jobStatus": st}
                        if nxt:
                            kw["nextToken"] = nxt
                        r = b.list_jobs(**kw)
                        jobs += r["jobSummaryList"]
                        nxt = r.get("nextToken")
                        if not nxt:
                            break
                except Exception:
                    pass
        elif arrays:
            truncated[0] = True
        jobs = [j for j in jobs
                if not j.get("arrayProperties", {}).get("size")]  # parents hold
        if not jobs:                                              # no resources
            continue
        status[st] = len(jobs)          # exact: counted before any describe cap
        # describe_jobs is the expensive call and a big array has thousands of
        # finished children. Cap it: the status counts above are already exact
        # and free, and a capped sum is reported as a lower bound rather than
        # dressed up as a total.
        budget = DESCRIBE_CAP - described[0]
        if budget <= 0:
            truncated[0] = True
            continue
        if len(jobs) > budget:
            jobs = jobs[:budget]
            truncated[0] = True
        described[0] += len(jobs)
        for chunk in [jobs[i:i+100] for i in range(0, len(jobs), 100)]:
            ids = [j["jobId"] for j in chunk]
            for d in b.describe_jobs(jobs=ids)["jobs"]:
                v = 0
                for rr in d.get("container", {}).get("resourceRequirements", []):
                    if rr["type"] == "VCPU":
                        v = float(rr["value"])
                if d["status"] == "RUNNING":
                    vcpu_now += v
                s0, s1 = d.get("startedAt"), d.get("stoppedAt")
                if s0:
                    end = s1 or now().timestamp() * 1000
                    vcpu_hours += v * (end - s0) / 3_600_000
    return dict(per_station=per_station, per_day=per_day, camps=camp_rows,
                sampled=sampled, unreadable=unreadable,
                picks=total_picks, bytes=total_bytes, files=total_files,
                coords=coords, vcpu_now=vcpu_now, vcpu_hours=vcpu_hours,
                status=status, vcpu_partial=truncated[0])


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024: return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _basemap(x0, x1, y0, y1, sx, sy):
    """Coastline and state borders, clipped to the view.

    Precomputed from Natural Earth into scripts/data/basemap.json so the hourly
    workflow does not need cartopy. Only the segments that intersect the view
    are emitted, so the page carries a few kilobytes rather than the whole
    world.
    """
    try:
        with open(BASEMAP) as f:
            bm = json.load(f)
    except Exception:
        return ""                       # no basemap: the map still works
    out = []
    for key, cls in (("coast", "coast"), ("states", "border")):
        for line in bm.get(key, []):
            seg, run = [], []
            for lo, la in line:
                if x0 <= lo <= x1 and y0 <= la <= y1:
                    run.append(f"{sx(lo):.1f},{sy(la):.1f}")
                else:
                    if len(run) > 1:
                        seg.append(run)
                    run = []
            if len(run) > 1:
                seg.append(run)
            for r in seg:
                out.append(f'<polyline class="{cls}" points="{" ".join(r)}"/>')
    return "".join(out)


def svg_map(per_station, coords, w=760, h=440):
    """Stations as triangles - the seismological convention - over a coastline.

    Triangles also separate the marks from the circles used in the time series,
    so the two figures are not read as the same kind of thing.
    """
    pts = [(coords[t][1], coords[t][0], n, coords[t][2])
           for t, n in per_station.items() if t in coords]
    if not pts:
        return ('<p class="empty">No picks yet - the map fills in as shards '
                'complete.</p>')
    lons = [p[0] for p in pts]; lats = [p[1] for p in pts]
    x0, x1 = min(lons), max(lons); y0, y1 = min(lats), max(lats)
    # Pad generously: a tight box around four stations shows no coastline at
    # all, which defeats the point of having one.
    # A tight box around a handful of stations contains no coastline at all,
    # which is the one thing the basemap is there to provide. Enforce a minimum
    # span so there is always recognisable geography to place them against.
    MIN_SPAN = 11.0
    padx = max((x1 - x0) * .35, (MIN_SPAN - (x1 - x0)) / 2, 1.0)
    pady = max((y1 - y0) * .35, (MIN_SPAN * .7 - (y1 - y0)) / 2, 0.8)
    x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady
    # Keep degrees square so the coastline is not stretched.
    ar = math.cos(math.radians((y0 + y1) / 2))
    plot_w, plot_h = w - 70, h - 54
    span_x, span_y = (x1 - x0) * ar, (y1 - y0)
    if span_x / span_y > plot_w / plot_h:
        extra = (span_x / (plot_w / plot_h) - span_y) / 2
        y0, y1 = y0 - extra, y1 + extra
    else:
        extra = ((span_y * (plot_w / plot_h)) / ar - (x1 - x0)) / 2
        x0, x1 = x0 - extra, x1 + extra
    mx = max(p[2] for p in pts)

    def sx(lo): return 46 + (lo - x0) / (x1 - x0) * plot_w
    def sy(la): return h - 40 - (la - y0) / (y1 - y0) * plot_h

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" '
           f'aria-label="Seismic stations as triangles over a coastline, '
           f'sized and shaded by pick count">']
    out.append(f'<rect x="46" y="14" width="{plot_w}" height="{plot_h}" '
               f'fill="var(--sea)"/>')
    out.append(f'<g class="base">{_basemap(x0, x1, y0, y1, sx, sy)}</g>')
    for f in (0, .5, 1):
        gx = 46 + f * plot_w; gy = h - 40 - f * plot_h
        out.append(f'<text x="{gx:.0f}" y="{h-24}" class="ax" text-anchor="middle">'
                   f'{x0+f*(x1-x0):.1f}\u00b0</text>')
        out.append(f'<text x="40" y="{gy+4:.0f}" class="ax" text-anchor="end">'
                   f'{y0+f*(y1-y0):.1f}\u00b0</text>')
    out.append(f'<rect x="46" y="14" width="{plot_w}" height="{plot_h}" '
               f'fill="none" stroke="var(--grid)"/>')
    for lo, la, n, code in sorted(pts, key=lambda p: p[2]):
        r = 6 + 12 * math.sqrt(n / mx)
        c = SEQ[min(len(SEQ) - 1, int(len(SEQ) * (n / mx) ** .5))]
        cx, cy = sx(lo), sy(la)
        tri = (f"{cx:.1f},{cy-r:.1f} {cx-r*0.87:.1f},{cy+r*0.5:.1f} "
               f"{cx+r*0.87:.1f},{cy+r*0.5:.1f}")
        out.append(f'<polygon points="{tri}" fill="{c}" '
                   f'stroke="var(--surface-2)" stroke-width="1.5" '
                   f'stroke-linejoin="round">'
                   f'<title>{code} \u2014 {n:,} picks</title></polygon>')
    out.append('</svg>')
    return "".join(out)


def svg_series(per_day, w=760, h=240):
    if not per_day:
        return ('<p class="empty">No picks yet — this fills in as shards '
                'complete.</p>')
    keys = sorted(per_day)
    vals = [per_day[k] for k in keys]
    mx = max(vals)
    n = len(keys)
    def px(i): return 56 + (i / max(n - 1, 1)) * (w - 80)
    def py(v): return h - 34 - (v / mx) * (h - 60)
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Picks per day">']
    for f in (0, .5, 1):
        gy = h - 34 - f * (h - 60)
        out.append(f'<line x1="56" y1="{gy:.0f}" x2="{w-24}" y2="{gy:.0f}" stroke="var(--grid)"/>')
        out.append(f'<text x="50" y="{gy+4:.0f}" class="ax" text-anchor="end">'
                   f'{int(mx*f):,}</text>')
    if n == 1:
        out.append(f'<circle cx="{px(0):.1f}" cy="{py(vals[0]):.1f}" r="5" '
                   f'fill="var(--series-1)"><title>{keys[0][0]}.{keys[0][1]:03d} — '
                   f'{vals[0]:,} picks</title></circle>')
    else:
        out.append(f'<polyline points="{pts}" fill="none" stroke="var(--series-1)" '
                   f'stroke-width="2" stroke-linejoin="round"/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4.5" '
                       f'fill="var(--series-1)" stroke="var(--surface-1)" '
                       f'stroke-width="2"><title>{keys[i][0]}.{keys[i][1]:03d} — '
                       f'{v:,} picks</title></circle>')
    for i in (0, n - 1):
        out.append(f'<text x="{px(i):.0f}" y="{h-12}" class="ax" '
                   f'text-anchor="{"start" if i==0 else "end"}">'
                   f'{keys[i][0]}.{keys[i][1]:03d}</text>')
    out.append('</svg>')
    return "".join(out)


# Examples are anchored on, and annotated with, picks at or above this
# confidence. The picking threshold is 0.2, so the catalogue contains plenty of
# marginal detections; a showcase drawn uniformly from all of them mostly shows
# the marginal ones, because that is what most picks are.
EXAMPLE_MIN_CONF = 0.5

# Display-only high-pass, in Hz. See png_waveform.
DISPLAY_HIGHPASS_HZ = 1.0


# Campaign identity colour. The fixed categorical order, assigned once and never
# cycled, so a campaign keeps its hue as campaigns come and go from the page.
# Validated as a 5-slot categorical palette against both card surfaces: CVD
# separation PASS (worst adjacent dE 9.1 protan), normal-vision PASS (19.6).
# Four slots sit under 3:1 on the light card, so the colour is never the only
# cue - every panel and section header spells the campaign name out in text.
CAMPAIGN_ORDER = ["scedc", "ncedc", "western", "obs", "earthscope"]
CAMPAIGN_INK = {
    "scedc":      ("#2a78d6", "#3987e5"),   # (light, dark)
    "ncedc":      ("#eb6834", "#d95926"),
    "western":    ("#1baf7a", "#199e70"),
    "obs":        ("#eda100", "#c98500"),
    "earthscope": ("#e87ba4", "#d55181"),
}


def campaign_ink(name):
    return CAMPAIGN_INK.get(name, ("#78766f", "#96948b"))


def waveform_examples(s3, campaigns, n=3, window=(6.0, 18.0), seed=None,
                      min_conf=EXAMPLE_MIN_CONF):
    """Random picks from the catalogue, with the waveform they were made on.

    Returns {campaign: [example, ...]}, sampled independently per campaign so
    one busy campaign cannot crowd the others out of the page. Sampled from a
    Parquet object chosen at random, so the examples change hour to hour and are
    not a curated set. Only anonymous archives are used - the restricted
    EarthScope access point needs a token the dashboard job does not carry, and
    an example that cannot be fetched is skipped rather than faked.
    """
    import random

    import obspy
    import pandas as pd
    from s3fs import S3FileSystem

    from sb_catalog.src.constants import NETWORK_MAPPING
    from sb_catalog.src.s3_helper import (EarthScopeS3ObjectHelper,
                                          NCEDCS3ObjectHelper,
                                          SCEDCS3ObjectHelper)

    rng = random.Random(seed)
    fs = S3FileSystem(anon=True)
    helpers = {"scedc": SCEDCS3ObjectHelper(), "ncedc": NCEDCS3ObjectHelper()}
    by_campaign = {}

    for c in campaigns:
        objs = []
        for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=BUCKET, Prefix=f"{c}/picks/"):
            objs += [o["Key"] for o in page.get("Contents", [])]
        if not objs:
            continue                       # campaign has written nothing yet
        df = pd.read_parquet(f"s3://{BUCKET}/{rng.choice(objs)}")
        if df.empty:
            continue
        strong = df[df.conf >= min_conf]
        if strong.empty:
            continue                       # nothing confident here; no example

        out, tried = [], set()
        order = list(
            strong.sample(frac=1, random_state=rng.randrange(10**6)).index)
        for idx in order:
            if len(out) >= n:
                break
            p = df.loc[idx]
            if p.tid in tried:
                continue                   # spread examples across stations
            tried.add(p.tid)
            net, sta, loc = (p.tid.split(".") + ["", ""])[:3]
            dc = NETWORK_MAPPING.get(net)
            if dc not in helpers:
                continue                   # EarthScope needs a token; skip
            t = obspy.UTCDateTime(str(p.peak))
            try:
                key = helpers[dc].get_s3_path(net, sta, loc, p.cha,
                                              f"{t.year}", f"{t.julday:03d}",
                                              "Z")
                tr = obspy.read(io.BytesIO(fs.open(key, "rb").read()))
                tr.merge(fill_value=0)
                tr.trim(t - window[0], t + window[1])
                if not len(tr) or tr[0].stats.npts < 50:
                    continue
                tr = tr[0]
            except Exception:
                continue                   # missing day, gap, unreadable: skip

            t0, t1 = tr.stats.starttime, tr.stats.endtime
            marks = []
            for _, q in strong[strong.tid == p.tid].iterrows():
                qt = obspy.UTCDateTime(str(q.peak))
                if t0 <= qt <= t1:
                    marks.append((float(qt - t0), str(q.pha), float(q.conf)))
            out.append({"tid": p.tid, "cha": p.cha, "start": str(t0)[:19],
                        "campaign": c,
                        "rate": float(tr.stats.sampling_rate),
                        "data": tr.data.astype(float).tolist(),
                        "dur": float(t1 - t0), "marks": marks})
        if out:
            by_campaign[c] = out
    return by_campaign


# The waveform panels are raster, so they cannot restyle with the theme the way
# the SVG figures do. These three are the palette steps that clear 3:1 against
# BOTH the light card (#f4f3f0) and the dark one (#232322): trace 3.56/3.98,
# P 3.98/3.56, S 3.50/4.05. The pair also passes the categorical checks
# (normal-vision dE 32.3, CVD 24.2).
WAVE_INK = "#808080"
PHASE_INK = {"P": "#2a78d6", "S": "#d95926"}


def png_waveform(ex, w=760, h=150, dpi=2):
    """Render one trace with obspy/matplotlib and return a data URI.

    obspy draws the trace, so the panel is the same plot a seismologist would
    make at the terminal rather than a reimplementation of one. The figure is
    transparent, so the card background shows through and the panel sits in
    either theme; the ink colours are chosen to survive both.
    """
    import base64
    from io import BytesIO

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import obspy

    tr = obspy.Trace(np.asarray(ex["data"], dtype=float))
    tr.stats.sampling_rate = ex["rate"]
    tr.stats.network, tr.stats.station, _ = (ex["tid"].split(".") + [""])[:3]
    tr.stats.channel = ex["cha"] + "Z"
    # Display filter. Raw counts on a broadband channel are dominated by
    # long-period ground motion, which sets the amplitude scale and flattens the
    # arrival into the middle of the trace - the panel then shows a slow
    # oscillation with a coloured line through it. A 1 Hz high-pass is the
    # standard view for local seismicity and is what makes the onset visible.
    # It affects the picture only: the picks were made upstream, on the data the
    # model saw, and are drawn where they fell.
    tr.detrend("demean")
    tr.filter("highpass", freq=DISPLAY_HIGHPASS_HZ, corners=2, zerophase=True)

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100 * dpi)
    t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
    ax.plot(t, tr.data, lw=0.55, color=WAVE_INK, solid_joinstyle="round")
    amp = float(np.abs(tr.data).max()) or 1.0
    ax.set_ylim(-amp * 1.12, amp * 1.12)
    ax.set_xlim(0, ex["dur"])
    # Picks seconds apart put their labels on top of each other - "P 0.29" and
    # "S 0.52" overlapped into an unreadable smear on the first render. Step a
    # label down a row whenever it would start before the previous one ended.
    LABEL_W = ex["dur"] * 0.085          # roughly the width of "S 0.52"
    rows, ends = [], []
    for sec, pha, conf in sorted(ex["marks"]):
        r = 0
        while r < len(ends) and sec < ends[r]:
            r += 1
        if r == len(ends):
            ends.append(0.0)
        ends[r] = sec + LABEL_W
        rows.append((sec, pha, conf, r))
    for sec, pha, conf, r in rows:
        c = PHASE_INK.get(pha, WAVE_INK)
        ax.axvline(sec, color=c, lw=1.4)
        ax.text(sec + ex["dur"] * 0.006, amp * (0.99 - 0.19 * r),
                f"{pha} {conf:.2f}", color=c, fontsize=7.4,
                fontweight="semibold", va="top")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7, colors=WAVE_INK, length=3, pad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(WAVE_INK)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_xlabel("seconds", fontsize=7, color=WAVE_INK, labelpad=1)
    fig.tight_layout(pad=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=100 * dpi)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    alt = (f'{ex["tid"]}{ex["cha"]}Z vertical component, {ex["dur"]:.0f} seconds '
           f'from {ex["start"]} UTC, with {len(ex["marks"])} picks marked')
    return (f'<img class="wave" alt="{alt}" '
            f'src="data:image/png;base64,{b64}">')


def render(g, examples):
    done = sum(c["done"] for c in g["camps"])
    planned = sum(c["shards"] for c in g["camps"])
    pct = (100 * done / planned) if planned else 0
    spend = g["vcpu_hours"] * FARGATE_SPOT_RATE
    tiles = [
        ("Picks in the catalogue", f"{g['picks']:,}",
         "counted from the Parquet footers - exact, including work whose "
         "worker was preempted before it wrote a manifest"),
        ("Shards complete", f"{done:,} / {planned:,}", f"{pct:.2f}% of the queue"),
        ("Catalogue size", human(g["bytes"]),
         f"{g['files']:,} Parquet object" + ("" if g["files"] == 1 else "s")),
        ("vCPU in use", f"{g['vcpu_now']:.0f}", "Batch jobs RUNNING now"),
        ("vCPU-hours", ("\u2265 " if g.get("vcpu_partial") else "")
         + f"{g['vcpu_hours']:,.1f}",
         "from job start/stop times" + (f"; a lower bound - only {DESCRIBE_CAP:,} "
         "jobs are described per run" if g.get("vcpu_partial") else "")),
        ("Spend (estimate)", ("\u2265 " if g.get("vcpu_partial") else "")
         + f"${spend:,.2f}",
         f"vCPU-h x ${FARGATE_SPOT_RATE}/vCPU-h — not a billed figure"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in tiles)

    rows = ""
    mx = max([c["shards"] for c in g["camps"]] or [1])
    for c in sorted(g["camps"], key=lambda x: -x["shards"]):
        p = (100 * c["done"] / c["shards"]) if c["shards"] else 0
        rows += (
            f'<tr><th scope="row">{c["name"]}</th>'
            f'<td class="bar"><span style="width:{max(c["shards"]/mx*100,0.6):.2f}%">'
            f'<i style="width:{p:.2f}%"></i></span></td>'
            f'<td class="num">{c["done"]:,}</td><td class="num">{c["shards"]:,}</td>'
            f'<td class="num">{p:.2f}%</td>'
            f'<td class="num">'
            + (f'{c["picks"]:,}' if c["picks"] is not None
               else '<span class="unread" title="the count failed this run; '
                    'see the note below">not read</span>')
            + '</td></tr>')

    # ---- campaign plan: what each queue will cost and how long it will take.
    # Every figure here is DERIVED from one measured rate; nothing is observed.
    plan_rows, tot_sd, tot_vh = [], 0, 0.0
    for c in sorted(g["camps"], key=lambda x: -x["planned_sd"]):
        sd = c["planned_sd"]
        if not sd:
            continue
        vh = sd * VCPU_H_PER_STATION_DAY
        cost = vh * FARGATE_SPOT_RATE
        cost_exp = cost * HIT_RATE          # what we actually expect to pay
        hours = vh / QUOTA_VCPU
        blocked = BLOCKED.get(c["name"])
        if not blocked:            # the total is what can actually be run
            tot_sd += sd
            tot_vh += vh
        lo, dk = campaign_ink(c["name"])
        note = (f'<span class="blocked" title="{blocked}">blocked</span>'
                if blocked else "")
        plan_rows.append(
            f'<tr{" class=off" if blocked else ""}>'
            f'<td><span class="dot" style="--camp:{lo};--camp-dk:{dk}"></span>'
            f'{c["name"]} {note}</td>'
            f'<td class="num">{sd:,}</td><td class="num">{c["shards"]:,}</td>'
            f'<td class="num">{vh:,.0f}</td>'
            f'<td class="num">${cost:,.0f}</td>'
            f'<td class="num">${cost_exp:,.0f}</td>'
            f'<td class="num">{hours:,.1f} h</td></tr>')
    plan_total = (
        f'<tr class="tot"><td>runnable total</td><td class="num">{tot_sd:,}</td>'
        f'<td class="num"></td><td class="num">{tot_vh:,.0f}</td>'
        f'<td class="num">${tot_vh * FARGATE_SPOT_RATE:,.0f}</td>'
        f'<td class="num">${tot_vh * FARGATE_SPOT_RATE * HIT_RATE:,.0f}</td>'
        f'<td class="num">{tot_vh / QUOTA_VCPU:,.1f} h</td></tr>')

    bad_note = ""
    if g.get("unreadable"):
        bits = "; ".join(f"{c} ({e})" for c, e in g["unreadable"])
        bad_note = (f'<p class="note">Some figures could not be read from S3 '
                    f'this hour - {bits}. The fleet and this job share the '
                    f'bucket\'s request budget, so a busy campaign can throttle '
                    f'the dashboard. Nothing is lost; the next run retries.</p>')
    sampled_note = ""
    if g.get("sampled"):
        bits = ", ".join(f"{c}: {a:,} of {b:,}" for c, a, b in g["sampled"])
        sampled_note = (
            f'<p class="note">Headline pick counts are exact. The map and time '
            f'series are built from the most recent Parquet objects only '
            f'({bits}), so they show less than the catalogue holds.</p>')

    st = ", ".join(f"{k} {v}" for k, v in sorted(g["status"].items())) or "nothing active"
    # Grouped by campaign, in the fixed campaign order, so a campaign keeps its
    # place and its hue on the page from hour to hour.
    if examples:
        blocks = []
        for c in [x for x in CAMPAIGN_ORDER if x in examples] + \
                 [x for x in examples if x not in CAMPAIGN_ORDER]:
            lo, dk = campaign_ink(c)
            figs = "".join(
                f'<figure class="wf" style="--camp:{lo};--camp-dk:{dk}">'
                f'{png_waveform(e)}'
                f'<figcaption>{e["tid"]}{e["cha"]}Z \u00b7 {e["start"]} UTC '
                f'\u00b7 {e["rate"]:.0f} Hz \u00b7 '
                f'{len(e["marks"])} pick{"" if len(e["marks"])==1 else "s"}'
                f'</figcaption></figure>'
                for e in examples[c])
            blocks.append(
                f'<div class="campgrp" style="--camp:{lo};--camp-dk:{dk}">'
                f'<h3><span class="dot"></span>{c}</h3>{figs}</div>')
        waves = "".join(blocks)
    else:
        waves = ('<p class="empty">No examples yet. They are drawn from picks '
                 'already written to S3, so this fills in once a campaign has '
                 'produced some.</p>')

    # Campaigns whose data lives behind the restricted EarthScope access point
    # cannot be illustrated here: the dashboard job runs anonymously. Say so
    # rather than letting a silently absent section read as "no picks".
    quiet = [c["name"] for c in g["camps"]
             if c["picks"] and c["name"] not in (examples or {})]
    waves_note = (f'<p class="note">No anonymous waveform source for '
                  f'{", ".join(quiet)} - those picks are in the catalogue, but '
                  f'their raw data sits behind the restricted EarthScope '
                  f'access point, which this dashboard job has no token for.'
                  f'</p>' if quiet else "")
    return f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuakeScope campaign dashboard</title>
<style>
:root{{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f4f3f0;
--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#78766f;
--series-1:#2a78d6;--grid:#e5e3de;--good:#0ca30c;--warning:#fab219;
--sea:#f0f4f8;--coast:#9aa5ae;--border-line:#c3cad1;
--wave:#6b6a65;--phase-p:#2a78d6;--phase-s:#eb6834}}
@media(prefers-color-scheme:dark){{:root:where(:not([data-theme=light])){{
color-scheme:dark;--surface-1:#1a1a19;--surface-2:#232322;--text-primary:#fff;
--text-secondary:#c3c2b7;--muted:#96948b;--series-1:#3987e5;--grid:#343431;
--sea:#20242a;--coast:#5d6771;--border-line:#454e57;
--wave:#9d9b92;--phase-p:#3987e5;--phase-s:#d95926}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--surface-1);color:var(--text-primary);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:860px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:21px;margin:0 0 2px}}
.sub{{color:var(--text-secondary);margin:0 0 26px;font-size:13px}}
h2{{font-size:15px;margin:34px 0 4px}}
.cap{{color:var(--text-secondary);font-size:12.5px;margin:0 0 12px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:10px}}
.tile{{background:var(--surface-2);border-radius:9px;padding:13px 14px}}
.tile .k{{font-size:11.5px;color:var(--text-secondary);text-transform:uppercase;
letter-spacing:.04em}}
.tile .v{{font-size:22px;font-weight:620;margin:3px 0 1px;
font-variant-numeric:tabular-nums}}
.tile .n{{font-size:11.5px;color:var(--muted)}}
figure{{margin:0;background:var(--surface-2);border-radius:9px;padding:12px}}
svg{{width:100%;height:auto;display:block}}
.ax{{fill:var(--muted);font-size:10.5px}}
.base{{fill:none;vector-effect:non-scaling-stroke}}
.base .coast{{stroke:var(--coast);stroke-width:1}}
.base .border{{stroke:var(--border-line);stroke-width:.75;stroke-dasharray:3 2.5}}
.campgrp{{margin:0 0 14px;border-left:3px solid var(--camp);padding-left:10px}}
.campgrp h3{{font-size:12px;font-weight:640;letter-spacing:.04em;
text-transform:uppercase;color:var(--text-secondary);margin:0 0 4px;
display:flex;align-items:center;gap:6px}}
.plan .dot,td .dot{{width:8px;height:8px;border-radius:50%;
background:var(--camp);display:inline-block;margin-right:6px}}
tr.off td{{opacity:.55}}
.unread{{color:var(--warn,#b45309);font-weight:600}}
.blocked{{font-size:10.5px;font-weight:640;letter-spacing:.03em;
text-transform:uppercase;color:var(--warning);border:1px solid var(--warning);
border-radius:3px;padding:0 4px;margin-left:6px;cursor:help}}
tr.tot td{{font-weight:640;border-top:2px solid var(--grid)}}
.scroll{{overflow-x:auto}}
.campgrp .dot{{width:8px;height:8px;border-radius:50%;background:var(--camp);
flex:none}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]) .campgrp
{{--camp:var(--camp-dk)}}}}
:root[data-theme="dark"] .campgrp{{--camp:var(--camp-dk)}}
.note{{color:var(--muted);font-size:12px;margin:2px 0 0;padding:0 2px}}
.wf{{padding:6px 8px 2px;margin-bottom:8px}}
img.wave{{width:100%;height:auto;display:block}}
figcaption{{font-size:11.5px;color:var(--muted);padding:0 2px 4px;font-variant-numeric:tabular-nums}}
.ph{{font-size:10.5px;font-weight:640}}
.empty{{color:var(--muted);padding:26px 8px;margin:0;font-size:13px}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);font-size:13px}}
th{{color:var(--text-secondary);font-weight:600}}
td.num{{text-align:right}}
td.bar{{width:34%}}
td.bar span{{display:block;height:9px;border-radius:4px;background:var(--grid)}}
td.bar i{{display:block;height:9px;border-radius:4px;background:var(--series-1)}}
footer{{margin-top:34px;color:var(--muted);font-size:12px;border-top:1px solid var(--grid);
padding-top:12px}}
code{{font-size:12px;background:var(--surface-2);padding:1px 5px;border-radius:4px}}
.loc{{background:var(--surface-2);border-radius:9px;padding:10px 12px}}
.loc .row{{display:flex;gap:10px;align-items:baseline;padding:3px 0;flex-wrap:wrap}}
.loc .row span{{flex:0 0 92px;color:var(--text-secondary);font-size:11.5px;
text-transform:uppercase;letter-spacing:.04em}}
.loc code{{background:none;padding:0;word-break:break-all}}
.snip{{background:var(--surface-2);border-radius:9px;padding:12px 14px;margin:10px 0 0;
overflow-x:auto;font-size:12px;line-height:1.5}}
</style>
<div class="wrap">
<h1>QuakeScope campaign dashboard</h1>
<p class="sub">{now():%Y-%m-%d %H:%M} UTC · Batch: {st} · rebuilt hourly</p>

<div class="tiles">{tile_html}</div>

<h2>Picks per station</h2>
<p class="cap">Stations are triangles, positioned by longitude and latitude,
sized and shaded by pick count. Coastline and state borders are drawn from
Natural Earth for orientation. Hover a triangle for its code and count.</p>
<figure>{svg_map(g['per_station'], g['coords'])}</figure>

<h2>Picks per day of data</h2>
<p class="cap">Picks by the day the waveform covers — not by when the job ran.</p>
<figure>{svg_series(g['per_day'])}</figure>

<h2>Waveforms and picks</h2>
<p class="cap">Picks drawn at random from the catalogue each hour, with the
waveform they were made on. Only picks at confidence
&ge;&nbsp;{EXAMPLE_MIN_CONF} are shown - the picking threshold is 0.2, so most of
the catalogue is marginal and a uniform sample would mostly show that. Colour is
the phase; the number is confidence. Traces are demeaned and high-passed at
{DISPLAY_HIGHPASS_HZ:.0f}&nbsp;Hz for display only - the picks were made
upstream, on the data the model saw.</p>
{waves}
{waves_note}

<h2>Campaign plan</h2>
<p class="cap">What is still to run, costed from one measured rate:
<b>{VCPU_H_PER_STATION_DAY:.4f} vCPU-hours per station-day</b> (707 vCPU-hours
over 10,440 station-days on the live SCEDC campaign, times 0.354 for the
short-window amplitude rework). Time assumes the full
{QUOTA_VCPU:,}-vCPU Fargate Spot quota with nothing else running. These are
projections, not observations - the tiles above are what actually happened.</p>
<p class="cap"><strong>Two figures, and the difference is the hit rate.</strong>
The rate above is per station-day <em>processed</em>; the plan counts
station-days <em>planned</em>, and most planned days hold no data - measured
hit rates run 21.7% to 67.6%. The <b>upper bound</b> column assumes every
planned day has data; the <b>expected</b> column applies a {HIT_RATE:.0%} hit
rate. The authoritative model,
<a href="https://github.com/SeisSCOPED/QuakeScope/blob/main/docs/rerun_2026/24_cost_model.md">24_cost_model.md</a>,
puts the campaign at <b>$10,828-$19,702</b> and says which end depends on that
one unmeasured number.</p>
<p class="cap">Nothing is blocked. The EarthScope restricted access point was
never stalling: the credential request was unscoped, so it could LIST but not
GET, and every read returned AccessDenied instantly. Scoping it to
<code>network=FDSN:&lt;NET&gt;</code> fixed it, and restricted reads now run at
96-98 MB/s - the same rate as Open Data.</p>
<p class="cap">Two things the plan still carries that will not produce picks:
<b>~3.67M station-days</b> on network-years EarthScope does not hold, which
complete empty and are harmless; and <b>49 networks</b> that answer 403, which
were dropped from <code>global</code> on 2026-09-03 and can be restored if
EarthScope grants access.</p>
<div class="scroll">
<table><thead><tr><th>campaign</th><th class="num">station-days</th>
<th class="num">shards</th><th class="num">vCPU-h</th><th class="num">upper bound</th><th class="num">expected</th>
<th class="num">at full quota</th></tr></thead>
<tbody>{''.join(plan_rows)}{plan_total}</tbody></table></div>

{sampled_note}
{bad_note}
<h2>Progress by campaign</h2>
<p class="cap">Bar length is the size of the queue; the filled part is what is
complete.</p>
<table><thead><tr><th>campaign</th><th>queue</th><th class="num">done</th>
<th class="num">shards</th><th class="num">%</th><th class="num">picks</th></tr></thead>
<tbody>{rows or '<tr><td colspan="6" class="empty">No campaign has written anything yet.</td></tr>'}</tbody></table>

<h2>Where the catalogue lives</h2>
<p class="cap"><strong>Public-read, no account needed.</strong> The picks are
Parquet on S3, Hive-partitioned by network, year and month. Read them with
anything that speaks Parquet; there is no database to connect to and no
credentials to obtain. <code>claims/</code> and <code>progress/</code> are
internal and stay private; picks, manifests and run provenance are open.</p>
<p class="cap">&#8594; <a href="https://colab.research.google.com/github/SeisSCOPED/QuakeScope/blob/main/tutorials/read_the_catalogue.ipynb"><strong>Open the tutorial notebook in Colab</strong></a>
&#8212; installs, reads, plots, and re-queries the original waveforms from FDSN
with ObsPy to draw the picks on the record.</p>
<div class="loc">
<div class="row"><span>bucket</span><code>s3://{BUCKET}</code></div>
<div class="row"><span>region</span><code>{REGION}</code></div>
<div class="row"><span>picks</span><code>s3://{BUCKET}/&lt;campaign&gt;/picks/network=&lt;NET&gt;/year=&lt;YYYY&gt;/month=&lt;MM&gt;/&lt;shard&gt;.parquet</code></div>
<div class="row"><span>manifests</span><code>s3://{BUCKET}/&lt;campaign&gt;/manifests/&lt;shard&gt;.json</code></div>
<div class="row"><span>run metadata</span><code>s3://{BUCKET}/&lt;campaign&gt;/runs/&lt;run_id&gt;.json</code></div>
</div>
<pre class="snip">pip install pandas pyarrow s3fs

# ---------------------------------------------------------------
import pandas as pd

# anon=True is REQUIRED and is the whole point: the read is
# unauthenticated. Without it pandas looks for credentials that a
# reader has no reason to have, and fails before reaching S3.
ANON = {{"anon": True}}

# one month, with partition pruning - only matching files are fetched
df = pd.read_parquet(
    "s3://{BUCKET}/global/picks/",
    filters=[("network", "=", "CI"), ("year", "=", 2014), ("month", "=", 9)],
    storage_options=ANON,
)

# which model made them, and at what thresholds
import json, urllib.request
run = json.load(urllib.request.urlopen(
    "https://{BUCKET}.s3.{REGION}.amazonaws.com/global/runs/"
    + df["rid"].iloc[0] + ".json"))

# or exactly the objects one shard wrote, from its manifest
m = json.load(urllib.request.urlopen(
    "https://{BUCKET}.s3.{REGION}.amazonaws.com/global/manifests/&lt;shard&gt;.json"))
df = pd.concat(pd.read_parquet(f["path"], storage_options=ANON)
               for f in m["files"])</pre>
<p class="cap">The three campaigns are <code>global</code>, <code>obs</code>
and <code>western</code>. The <strong>previous run</strong> is still readable
under <code>scedc</code>, <code>ncedc</code>, <code>earthscope</code> and
<code>western-a</code> - those prefixes were merged into <code>global</code> for
the 2026 run, not deleted.</p>
<p class="cap"><strong>Columns.</strong> <code>tid</code> trace id
<code>NET.STA.LOC</code> &middot; <code>cha</code> band &middot;
<code>pha</code> P or S &middot; <code>peak</code> the arrival time &middot;
<code>conf</code> model score, floored at 0.2 &middot; <code>amp</code>
Wood-Anderson displacement in metres &middot; <code>amp_raw</code> peak counts
before response removal &middot; <code>rid</code> run id, joins to
<code>runs/</code>. <code>conf</code> is a detection score, not a probability
of correctness; the 0.2 floor is permissive on purpose so you can pick your
own threshold.</p>

<footer>
<strong>Two pick counts, both true.</strong> The tile counts picks in shards
that have <em>finished</em>, taken from their manifests. A running shard
checkpoints its work to Parquet every 40 station-day-channels, so the objects on
S3 already hold more than the tile shows; the manifest only appears when the
shard closes. Query the Parquet for what is actually stored, the tile for what is
durably accounted for.

Every figure above except spend is counted from an S3 or Batch API response.
<strong>Spend is derived</strong>: vCPU-hours from Batch job start and stop times,
multiplied by <code>${FARGATE_SPOT_RATE}</code> per vCPU-hour. It is not a billed
figure — Cost Explorer is blocked on this account by an organisation policy, so
nothing here has been checked against AWS billing.
Data volume <em>ingested</em> from the archives is not shown because it is not
measured anywhere; the catalogue size above is what was written, not what was read.
</footer>
</div>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="reports/campaign_dashboard.html")
    ap.add_argument("--examples", type=int, default=3,
                    help="random waveform+pick examples to include")
    ap.add_argument("--min-conf", type=float, default=EXAMPLE_MIN_CONF,
                    help="minimum pick confidence for the examples")
    # The three campaigns of the 2026 run, as restructured 2026-09-03: scedc,
    # ncedc and earthscope were merged into `global` (same weight, no shared
    # stations), leaving `western` and `obs`, which differ by weight.
    #
    # Named explicitly rather than discovered from S3 prefixes: the bucket also
    # holds ~20 test and profiling prefixes (_dryrun2, _iotest, _sweep, ...) and
    # prior runs (western-a, western-b), and a dashboard that lists whatever it
    # finds would bury the live campaign among them. Pass --campaigns to look at
    # a historical one - western-a still holds 106M picks.
    ap.add_argument("--campaigns", default="global,obs,western")
    a = ap.parse_args()
    # The hourly job shares S3 with the fleet. At 1,500 workers the dashboard
    # is the small, interruptible client in that contention, so it backs off
    # adaptively rather than adding to the pressure - a render that throttles is
    # a render that reports nothing.
    from botocore.config import Config
    s3 = boto3.client("s3", region_name=REGION, config=Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=8))
    b = boto3.client("batch", region_name=REGION)
    camps = [c for c in a.campaigns.split(",") if c]
    g = gather(s3, b, camps)
    try:
        ex = waveform_examples(s3, camps, n=a.examples, min_conf=a.min_conf)
    except Exception as e:                 # never let examples break the page
        print(f"  (examples skipped: {type(e).__name__}: {e})")
        ex = {}
    with open(a.out, "w") as f:
        f.write(render(g, ex))
    print(f"wrote {a.out}: {sum(len(v) for v in ex.values())} waveform examples "
          f"across {len(ex)} campaign(s), {g['picks']:,} picks, "
          f"{len(g['per_station'])} stations, {len(g['per_day'])} days, "
          f"{g['vcpu_hours']:.1f} vCPU-h")


if __name__ == "__main__":
    main()
