"""Check EarthScope access, both tiers.

    pixi run -e cloud python scripts/check_earthscope_access.py

Open Data needs no credentials, so tier 1 should pass for anyone. Tier 2 needs
the 's3-miniseed-v2' role on the access point. The retired 's3-miniseed' answers
"You are not allowed to assume role", which reads like a permissions problem
rather than a renamed role - it is checked here so that is unambiguous.

Exit 0 = both tiers reachable, 2 = Open Data only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3fs import S3FileSystem

from sb_catalog.src.s3_helper import (EARTHSCOPE_OPEN_DATA_BUCKET,
                                      EARTHSCOPE_RESTRICTED_ACCESS_POINT,
                                      EARTHSCOPE_ROLE)

rc = 0

print("[1] open data   ", end="", flush=True)
try:
    n = S3FileSystem(anon=True).ls(
        f"{EARTHSCOPE_OPEN_DATA_BUCKET}/miniseed/", detail=False)
    print(f": OK, {len(n)} networks, no credentials "
          f"({', '.join(sorted(x.split('/')[-1] for x in n))})")
except Exception as e:
    print(f": FAILED {type(e).__name__}: {e}")
    rc = 1

print("[2] identity    ", end="", flush=True)
from earthscope_sdk import EarthScopeClient  # noqa: E402
with EarthScopeClient() as c:
    try:
        print(f": {c.user.get_profile().primary_email}")
    except Exception as e:
        print(f": NOT LOGGED IN ({type(e).__name__}) - run `es login`")
        sys.exit(2)

    print(f"[3] role        ", end="", flush=True)
    try:
        cr = c.user.get_aws_credentials(role=EARTHSCOPE_ROLE)
        print(f": OK '{EARTHSCOPE_ROLE}', expires {cr.expiration}")
    except Exception as e:
        print(f": DENIED '{EARTHSCOPE_ROLE}' -> {type(e).__name__}: {str(e)[:90]}")
        print("    Open Data still works; restricted networks do not.")
        sys.exit(2)

print("[4] access point", end="", flush=True)
try:
    fs = S3FileSystem(key=cr.aws_access_key_id, secret=cr.aws_secret_access_key,
                      token=cr.aws_session_token,
                      client_kwargs={"region_name": "us-east-2"})
    got = fs.ls(f"{EARTHSCOPE_RESTRICTED_ACCESS_POINT}/miniseed/ZI/2019/187/",
                detail=False)
    print(f": OK, {EARTHSCOPE_RESTRICTED_ACCESS_POINT[:34]}... "
          f"({len(got)} objects in ZI/2019/187)")
except Exception as e:
    print(f": FAILED {type(e).__name__}: {str(e)[:90]}")
    rc = 2

sys.exit(rc)
