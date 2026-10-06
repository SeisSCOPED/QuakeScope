"""Station codes are text, and `id` is their authority.

Until 2026-10-05 the published station tables carried location codes like
"0.0", "1.0" and "40.0" (2,065 western rows, 6,597 global), station codes
like "1" for "001" and an empty network for "NA". The CSVs behind them were
read with pandas' defaults, which parse "00" as 0.0 and "NA" as NaN, and
`write_stations` then cast the floats to str. The `id` column was always
right, so picking was unaffected, but any join from the table to picks on the
code columns silently lost those stations.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.utils import normalize_station_codes, read_station_csv

CSV = """id,network_code,station_code,location_code,channels,latitude,longitude
SB.DBR1.00,SB,DBR1,00,HN,34.1,-117.1
BP.JCSB.40,BP,JCSB,40,DP,35.9,-120.4
2Q.001.,2Q,001,,HH,10.0,10.0
NA.SABA.,NA,SABA,,BH,17.6,-63.2
"""


def test_default_read_csv_is_what_did_the_damage():
    # Documents the failure: if pandas ever stops doing this, the test says so.
    import io
    d = pd.read_csv(io.StringIO(CSV))
    assert d["location_code"].astype(str).tolist()[0] == "0.0"
    assert pd.isna(d["network_code"].iloc[3])


def test_read_station_csv_keeps_codes_as_written(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text(CSV)
    d = read_station_csv(p)
    assert d["location_code"].tolist() == ["00", "40", "", ""]
    assert d["station_code"].tolist() == ["DBR1", "JCSB", "001", "SABA"]
    assert d["network_code"].tolist()[3] == "NA"
    assert d["latitude"].dtype.kind == "f"


def test_codes_are_rebuilt_from_id():
    damaged = pd.DataFrame({
        "id": ["SB.DBR1.00", "BP.JCSB.40", "2Q.001.", "NA.SABA."],
        "network_code": ["SB", "BP", "2Q", float("nan")],
        "station_code": ["DBR1", "JCSB", "1", "SABA"],
        "location_code": ["0.0", "40.0", "", ""],
    })
    d = normalize_station_codes(damaged)
    assert d["location_code"].tolist() == ["00", "40", "", ""]
    assert d["station_code"].tolist() == ["DBR1", "JCSB", "001", "SABA"]
    assert d["network_code"].tolist() == ["SB", "BP", "2Q", "NA"]


def test_without_id_a_float_code_is_refused():
    d = pd.DataFrame({"network_code": ["SB"], "station_code": ["DBR1"],
                      "location_code": ["0.0"]})
    with pytest.raises(ValueError, match="went through a float"):
        normalize_station_codes(d)


def test_without_id_the_id_is_built():
    d = normalize_station_codes(pd.DataFrame(
        {"network_code": ["SB"], "station_code": ["DBR1"], "location_code": ["00"]}))
    assert d["id"].tolist() == ["SB.DBR1.00"]


def test_malformed_id_is_refused():
    with pytest.raises(ValueError, match="NET.STA.LOC"):
        normalize_station_codes(pd.DataFrame({"id": ["SB.DBR1"]}))


# Scripts that put a station table straight to S3 must rebuild its codes first.
# A static check, because the failure is silent at run time: the table writes
# fine and only a reader joining on the code columns ever notices.
# One-off scripts that already ran and only wrote subsets of an existing table
# are listed with the reason; a new script does not get onto this list lightly.
_ALREADY_RAN = {
    "scripts/split_obs_land.py": "2026-10-03 split; wrote row subsets of obs/stations.parquet",
    "scripts/repair_western_fill_duplicates.py": "2026-09-25 epoch merge on _queues/western-fill",
}


def _writes_station_table(src: str) -> bool:
    import re
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"\bput_object\(|\bput\(|\.to_parquet\(|write_table\(", line):
            window = " ".join(lines[i:i + 3])
            if "stations.parquet" in window and "copy_object" not in window:
                return True
    return False


def test_every_station_table_writer_normalizes_codes():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for f in sorted([*root.glob("scripts/*.py"), *root.glob("sb_catalog/src/*.py")]):
        rel = f.relative_to(root).as_posix()
        src = f.read_text()
        if rel in _ALREADY_RAN or not _writes_station_table(src):
            continue
        if "normalize_station_codes" not in src and "write_stations(" not in src:
            offenders.append(rel)
    assert not offenders, (
        f"{offenders} write stations.parquet without normalize_station_codes; "
        "route the table through it, or through S3CampaignState.write_stations")
