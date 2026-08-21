"""Three-stage check of direct EarthScope S3 access.

Run:  pixi run -e cloud python scripts/check_earthscope_access.py

Exit 0 = reachable, 2 = the s3-miniseed role is not granted to this
account (EarthScope has to fix it; re-running `es login` will not),
3 = role fine but EARTHSCOPE_S3_ACCESS_POINT is unset.

The stages are separate because they fail for unrelated reasons and the
fixes go to different people.
"""
import os, sys
from earthscope_sdk import EarthScopeClient
ap = os.environ.get("EARTHSCOPE_S3_ACCESS_POINT", "")
with EarthScopeClient() as c:
    u = c.user.get_profile()
    print(f"[1] logged in as : {u.primary_email}")
    try:
        cr = c.user.get_aws_credentials(role="s3-miniseed")
    except Exception as e:
        print(f"[2] s3-miniseed  : DENIED -> {type(e).__name__}: {str(e)[:120]}")
        sys.exit(2)
    print(f"[2] s3-miniseed  : OK, expires {cr.expiration}")
    if not ap:
        print("[3] bucket       : SKIPPED, EARTHSCOPE_S3_ACCESS_POINT is unset")
        sys.exit(3)
    from s3fs import S3FileSystem
    fs = S3FileSystem(key=cr.aws_access_key_id, secret=cr.aws_secret_access_key,
                      token=cr.aws_session_token)
    got = fs.ls(f"{ap}/miniseed/", detail=False)[:5]
    print(f"[3] bucket       : OK, {ap} -> {got}")
