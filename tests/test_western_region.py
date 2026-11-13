"""The western station selection must take every station obs refuses.

docs/rerun_2026/35: a station outside every admin-1 polygon was called
offshore, obs rejected it as land (above -10 m), and it landed in neither
catalogue. These use two boxes in place of the Natural Earth polygons.
"""
from shapely.geometry import box
from shapely.prepared import prep

from scripts.plan_western_fill import OFFSHORE_DEPTH, where

POLYS = [(name, prep(g), g) for name, g in
         [("Washington", box(-124.0, 46.0, -117.0, 49.0)),
          ("Oregon", box(-124.0, 42.0, -117.0, 46.0))]]


def test_inside_a_polygon():
    assert where(POLYS, 47.5, -122.3, 50.0) == "Washington"


def test_shoreline_station_above_sea_level_goes_to_nearest_region():
    # just west of the Washington box, 3 m above sea level: an UW.ALKI case
    assert where(POLYS, 47.58, -124.05, 3.0) == "Washington"


def test_shallow_station_above_the_obs_floor_stays_on_land():
    # BK.MCCM is at -7.7 m: obs sends it to review, so western must keep it
    assert where(POLYS, 44.0, -124.2, -7.7) == "Oregon"


def test_deep_station_is_offshore():
    assert where(POLYS, 44.0, -125.5, OFFSHORE_DEPTH) == "offshore"
    assert where(POLYS, 44.0, -125.5, -2500.0) == "offshore"


def test_unknown_elevation_keeps_the_old_rule():
    assert where(POLYS, 44.0, -125.5, None) == "offshore"
    assert where(POLYS, 44.0, -125.5, float("nan")) == "offshore"
