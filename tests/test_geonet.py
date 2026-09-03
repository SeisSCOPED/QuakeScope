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
