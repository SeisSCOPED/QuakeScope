"""GeoNet is shaped differently from the other three archives.

Two differences, both of which fail silently if unhandled:

  * an extra STA.NET directory between the day prefix and the objects, so
    `fs.ls` on the day returns directories and the day looks empty;
  * the object path is therefore NOT prefix + basename, which is what the base
    `get_s3_path` assumes.

Paths are asserted against a real object key rather than a restatement of the
format string, so a wrong-but-self-consistent implementation cannot pass.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.constants import NETWORK_MAPPING
from sb_catalog.src.s3_helper import (GEONET_BUCKET, GEONET_REGION,
                                      CompositeS3ObjectHelper,
                                      GeoNetS3ObjectHelper)

# Verified present in geonet-open-data on 2026-09-03.
REAL_KEY = ("geonet-open-data/waveforms/miniseed/2019/2019.001/ABAZ.NZ/"
            "2019.001.ABAZ.10-EHE.NZ.D")


def test_nz_routes_to_geonet():
    # NZ used to route to earthscope, which serves it correctly but mirrors
    # only 29 stations against GeoNet's 423.
    assert NETWORK_MAPPING["NZ"] == "geonet"


def test_path_matches_a_real_object():
    h = GeoNetS3ObjectHelper()
    assert h.get_s3_path("NZ", "ABAZ", "10", "EH", "2019", "001", "E") == REAL_KEY


def test_composite_delegates_the_composed_path():
    # The base get_s3_path is prefix + basename, which is wrong here. If the
    # composite inherits it instead of delegating, the station directory is
    # dropped and every read 404s.
    h = CompositeS3ObjectHelper()
    assert h.get_s3_path("NZ", "ABAZ", "10", "EH", "2019", "001", "E") == REAL_KEY
    assert h.get_data_center("NZ") == "geonet"


def test_path_is_not_prefix_plus_basename():
    h = GeoNetS3ObjectHelper()
    prefix = h.get_prefix("NZ", "2019", "001")
    base = h.get_basename("NZ", "ABAZ", "10", "EH", "2019", "001", "E")
    assert prefix + base != REAL_KEY, (
        "if these ever concatenate correctly the station directory has been "
        "lost from the layout - re-check the bucket"
    )
    assert REAL_KEY.startswith(prefix) and REAL_KEY.endswith(base)


def test_list_day_recurses():
    # fs.ls stops at the station directories. A day that lists no objects is
    # indistinguishable from a day with no data, so this must recurse.
    class _FS:
        def __init__(self): self.called = None
        def ls(self, p): self.called = "ls"; return []
        def find(self, p): self.called = "find"; return ["a", "b"]

    fs = _FS()
    assert GeoNetS3ObjectHelper().list_day(fs, "x/") == ["a", "b"]
    assert fs.called == "find"


def test_geonet_is_its_own_region():
    # The bucket is in ap-southeast-2 while everything else we read is
    # us-east-2. Without the pin s3fs signs for the wrong region.
    assert GEONET_REGION == "ap-southeast-2"
    assert GEONET_BUCKET == "geonet-open-data"
    fs = CompositeS3ObjectHelper().get_filesystem("NZ")
    assert fs.client_kwargs.get("region_name") == GEONET_REGION


def test_geonet_reads_like_scedc_not_earthscope():
    """One object per channel-component, so the read path is the SCEDC branch.

    EarthScope's branch assumes one multi-channel object per station-day and
    filters records out of it; applied to GeoNet it would look for a version
    suffix that does not exist.
    """
    import pathlib
    src = (pathlib.Path(__file__).parent.parent
           / "sb_catalog/src/s3_helper.py").read_text()
    assert 'if dc in ["scedc", "ncedc", "geonet"]:' in src


def test_bn_ranks_with_broadband_not_with_accelerometers():
    """BN is a BROADBAND accelerometer and sits at the BH tier.

    M. Denolle, 2026-09-03: closer in character to the Paroscientific
    instruments this group works with than to strong-motion HN. It is also
    GeoNet's most common band - 878 of 2,036 objects on a sample day - so
    excluding it would drop most of the New Zealand network.
    """
    from sb_catalog.src.constants import CHANNEL_PRIORITY, select_channel

    i = CHANNEL_PRIORITY.index
    assert i("BH") < i("BN") < i("HN"), "BN belongs between BH and HN"

    # A BN-only station is picked, not skipped.
    assert select_channel(["BN"]) == "BN"
    # Broadband velocity still wins where a station offers both.
    assert select_channel(["HH", "BN"]) == "HH"
    assert select_channel(["EH", "BN"]) == "EH"
    # But BN beats the strong-motion accelerometers.
    assert select_channel(["BN", "HN"]) == "BN"


def test_the_accelerometer_caveat_is_written_down():
    """No picker here is trained on accelerometer data.

    jma_wc, original and obs are all trained on velocity seismometers, so BN
    and HN picks are out of distribution. Ranking BN highly is a claim about
    instrument quality, not about training coverage, and the distinction has to
    survive in the source rather than living only in a conversation.
    """
    import pathlib
    src = (pathlib.Path(__file__).parent.parent
           / "sb_catalog/src/constants.py").read_text()
    lo = src.index('"BN"')
    window = src[max(0, lo - 1400):lo]
    assert "trained" in window and "accelerometer" in window, (
        "the out-of-distribution caveat must sit with the BN entry"
    )
