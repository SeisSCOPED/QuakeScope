"""Diagnose EarthScope S3 access from wherever this runs.

Written because the restricted access point behaves differently from inside
us-east-2 than from a laptop, and neither behaviour was what we assumed:

  - from a laptop, GetObject returns 403. That is correct and documented:
    EarthScope permit ListObjectsV2 from anywhere but GetObject only from
    us-east-2. A laptop test therefore proves nothing about entitlement.
  - from Fargate in us-east-2, where GetObject is permitted, the same read
    neither succeeds nor errors. It stalls until the caller gives up.

This walks the request apart - DNS, TCP, a signed HEAD, a 1 KB ranged GET, a
full GET - so the failure can be attributed to a layer rather than guessed at,
and compares the restricted access point against an Open Data object taken over
the identical code path as a control.

    python -m src.picker diag-earthscope [NET|NET:YEAR|NET:YEAR:DOY ...]
"""

from __future__ import annotations

import datetime
import socket
import sys
import time

DEFAULT_NETS = ["CC", "UW"]          # one restricted, one open-data control
DAY = ("2019", "187")


def _parse(arg):
    """`NET`, `NET:YEAR`, or `NET:YEAR:DOY`.

    Temporary network codes are reused, so a network alone does not identify a
    deployment - and asking for a network-year that never existed is refused at
    the credential exchange, which looks exactly like a missing entitlement.
    Testing one therefore means naming a year the network actually operated.
    """
    parts = arg.split(":")
    net = parts[0]
    year = parts[1] if len(parts) > 1 else DAY[0]
    doy = parts[2] if len(parts) > 2 else DAY[1]
    return net, year, doy


def _t(fn, *a, **k):
    t0 = time.time()
    try:
        # Bind the result before reading the clock: a tuple literal evaluates
        # left to right, so `return time.time() - t0, fn(...)` times nothing.
        out = fn(*a, **k)
        return time.time() - t0, out, None
    except Exception as exc:            # noqa: BLE001 - reporting, not handling
        return time.time() - t0, None, f"{type(exc).__name__}: {exc}"[:200]


def main(argv=None) -> None:
    import boto3
    from botocore.config import Config

    from .s3_helper import (EARTHSCOPE_OPEN_DATA_BUCKET,
                            EARTHSCOPE_RESTRICTED_ACCESS_POINT,
                            EARTHSCOPE_ROLE, CompositeS3ObjectHelper,
                            EarthScopeS3ObjectHelper)

    targets = [_parse(a) for a in (argv or DEFAULT_NETS)]
    print(f"=== EarthScope S3 diagnostic  {datetime.datetime.utcnow()}Z ===")
    # Fargate does not serve the EC2 IMDS address; it exposes
    # ECS_CONTAINER_METADATA_URI_V4. Probing the wrong one printed "laptop?"
    # from a task that was genuinely in us-east-2 - which would have undermined
    # the one claim this whole diagnostic rests on.
    import json as _json
    import os as _os
    import urllib.request as _u
    where = "unknown"
    uri = _os.environ.get("ECS_CONTAINER_METADATA_URI_V4") or \
        _os.environ.get("ECS_CONTAINER_METADATA_URI")
    try:
        if uri:
            meta = _json.loads(_u.urlopen(f"{uri}/task", timeout=3).read())
            where = (f"ECS/Fargate  AZ={meta.get('AvailabilityZone')}  "
                     f"cluster={str(meta.get('Cluster','')).split('/')[-1]}")
        else:
            where = "EC2 AZ=" + _u.urlopen(
                "http://169.254.169.254/latest/meta-data/placement/"
                "availability-zone", timeout=2).read().decode()
    except Exception:
        where = "not an AWS metadata host (laptop?)"
    print(f"running in: {where}")

    helper = CompositeS3ObjectHelper()
    # Short timeouts: the point is to observe a stall, not to sit through one.
    cfg = Config(connect_timeout=10, read_timeout=30,
                 retries={"max_attempts": 1, "mode": "standard"})

    for net, yr, doy in targets:
        open_data = EarthScopeS3ObjectHelper.is_open_data(net)
        bucket = (EARTHSCOPE_OPEN_DATA_BUCKET if open_data
                  else EARTHSCOPE_RESTRICTED_ACCESS_POINT)
        tier = "OPEN DATA (anonymous)" if open_data else "RESTRICTED (role)"
        print(f"\n--- {net} {yr}.{doy}: {tier}\n    bucket/alias: {bucket}")

        host = f"{bucket}.s3.us-east-2.amazonaws.com"
        dt, ips, err = _t(socket.gethostbyname_ex, host)
        print(f"    DNS   {host}\n          {dt*1000:7.0f} ms  "
              f"{(ips[2] if ips else err)}")
        if ips:
            dt, _, err = _t(lambda: socket.create_connection((ips[2][0], 443), 10))
            print(f"    TCP   443 -> {ips[2][0]}   {dt*1000:7.0f} ms  "
                  f"{'ok' if not err else err}")

        # LIST, via the same helper the worker uses.
        prefix = helper.get_prefix(net, yr, doy)
        year = int(yr)
        dt, ls, err = _t(lambda: helper.get_filesystem(net, year).ls(prefix))
        print(f"    LIST  {dt:7.1f} s  "
              f"{f'{len(ls)} objects' if ls is not None else err}")
        if not ls:
            continue
        key = ls[0].split("/", 1)[1]

        # Signed requests through plain boto3, so s3fs is not in the picture.
        if open_data:
            import botocore
            s3 = boto3.client("s3", region_name="us-east-2", config=cfg.merge(
                Config(signature_version=botocore.UNSIGNED)))
        else:
            c = helper.get_es_credential(net, year)
            s3 = boto3.client(
                "s3", region_name="us-east-2", config=cfg,
                aws_access_key_id=c.aws_access_key_id,
                aws_secret_access_key=helper._secret(c.aws_secret_access_key),
                aws_session_token=helper._secret(c.aws_session_token))

        dt, r, err = _t(s3.head_object, Bucket=bucket, Key=key)
        size = f"{r['ContentLength'] / 1e6:.1f} MB" if r else err
        print(f"    HEAD  {dt:7.1f} s  {size}")

        dt, r, err = _t(lambda: s3.get_object(
            Bucket=bucket, Key=key, Range="bytes=0-1023")["Body"].read())
        print(f"    GET   1 KB ranged   {dt:7.1f} s  "
              f"{f'{len(r)} bytes' if r else err}")

        dt, r, err = _t(lambda: s3.get_object(
            Bucket=bucket, Key=key)["Body"].read())
        print(f"    GET   full object   {dt:7.1f} s  "
              f"{f'{len(r)/1e6:.1f} MB at {len(r)/1e6/max(dt,1e-9):.1f} MB/s' if r else err}")

    print(f"\nrole: {EARTHSCOPE_ROLE}")
    print("=== end ===")


if __name__ == "__main__":
    main(sys.argv[1:])
