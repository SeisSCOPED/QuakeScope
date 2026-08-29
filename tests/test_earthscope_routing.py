"""EarthScope is two archives behind one data-centre name.

Open Data needs no credentials and covers eight networks; everything else needs
a role on an access point. Preferring Open Data means a campaign over those
networks cannot fail on an expired token or a role that was never granted.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.s3_helper import (EARTHSCOPE_OPEN_DATA_BUCKET,
                                      EARTHSCOPE_OPEN_DATA_NETWORKS,
                                      EARTHSCOPE_RESTRICTED_ACCESS_POINT,
                                      EARTHSCOPE_ROLE,
                                      EarthScopeS3ObjectHelper)


def test_earthscope_routing():
    E = EarthScopeS3ObjectHelper

    # The Open Data Program network list, per the SDK tutorial.
    assert set(EARTHSCOPE_OPEN_DATA_NETWORKS) == {
        "AK", "II", "IU", "N4", "PB", "TA", "UU", "UW"}

    # Open Data wins for the networks it serves; everything else is restricted.
    for net in ("UW", "TA", "UU", "AK"):
        assert E.is_open_data(net)
        assert E.bucket_for(net) == EARTHSCOPE_OPEN_DATA_BUCKET
    for net in ("XD", "ZI", "ZG", "1D", "7D"):
        assert not E.is_open_data(net)
        assert E.bucket_for(net) == EARTHSCOPE_RESTRICTED_ACCESS_POINT

    # The v1 role is retired and denies accounts in good standing.
    assert EARTHSCOPE_ROLE == "s3-miniseed-v2"

    # Same key layout in both buckets, so only the bucket differs.
    h = E()
    assert h.get_prefix("UW", 2019, "187") == \
        f"{EARTHSCOPE_OPEN_DATA_BUCKET}/miniseed/UW/2019/187/"
    assert h.get_prefix("XD", 2019, "187") == \
        f"{EARTHSCOPE_RESTRICTED_ACCESS_POINT}/miniseed/XD/2019/187/"

    # The restricted access point appends a version; Open Data does not.
    # Requiring the suffix matched nothing on Open Data - silently, because a
    # station with no matching object is indistinguishable from a station that
    # was not recording.
    rgx = h.get_basename("UW", "RATT", "", "HH", 2019, "187", "Z")
    assert re.match(rgx, "RATT.UW.2019.187")      # Open Data
    assert re.match(rgx, "RATT.UW.2019.187#2")    # restricted
    assert not re.match(rgx, "RATTX.UW.2019.187") # not a prefix match
    assert not re.match(rgx, "RATT.UW.2019.1870")

    print("PASS  Open Data network list matches the SDK tutorial")
    print("PASS  Open Data preferred; other networks routed to the access point")
    print("PASS  role is s3-miniseed-v2, not the retired s3-miniseed")
    print("PASS  identical key layout, bucket is the only difference")
    print("PASS  basename regex accepts versioned and unversioned names")
    print("\nall EarthScope routing checks passed")


if __name__ == "__main__":
    test_earthscope_routing()
