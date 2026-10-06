"""Pure helpers shared across the pipeline.

Deliberately dependency-light: the v3 worker imports `parse_year_day` from here
on every shard, and this module must not pull in `pymongo`. The DocumentDB
client lives in `mongo_db` and is re-exported lazily below, so existing
`from .utils import SeisBenchDatabase` keeps working without making every
picking job depend on a database driver it never opens.
"""

import datetime
import math
import re
from typing import Any, Optional

import pandas as pd


def __getattr__(name: str) -> Any:
    # PEP 562. Only pays for pymongo if someone actually asks for the client.
    if name == "SeisBenchDatabase":
        from .mongo_db import SeisBenchDatabase
        return SeisBenchDatabase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def filter_station_by_start_end_date(
    stations: pd.DataFrame, start: datetime.date, end: datetime.date
) -> pd.DataFrame:
    match = []
    for i, sta in stations.iterrows():
        sta_start = station_date(sta["start_date"])
        sta_end = station_date(sta["end_date"])
        if sta_start is None or sta_end is None:
            match.append(i)                 # unknown window: keep it, plan it
        elif (sta_start <= end) and (sta_end >= start):
            match.append(i)
    return stations.iloc[match]


# Station identifiers are SEED codes: text, never numbers, and never containing
# a dot. A CSV read with pandas' defaults turns them into numbers or NaN:
# location "00" -> 0.0 -> "0.0", "01" -> "1.0", station "001" -> 1, network
# "NA" -> NaN -> "". Until 2026-10-05 every station table carried that damage in
# its code columns (2,065 western rows, 6,597 global; the `id` column was always
# right, which is why picking was unaffected). Read with `read_station_csv`, and
# let `normalize_station_codes` rebuild the codes from `id` before writing.
STATION_CODE_COLUMNS = ("network_code", "station_code", "location_code")
STATION_TEXT_COLUMNS = ("id",) + STATION_CODE_COLUMNS + ("channels",)
_NUMBER_LIKE_RE = re.compile(r"^\d+\.\d*$")


def read_station_csv(path, **kwargs) -> pd.DataFrame:
    """A station CSV with every identifier kept as the text it was written as."""
    return pd.read_csv(path, dtype={c: str for c in STATION_TEXT_COLUMNS},
                       keep_default_na=False, na_values={"latitude": [""],
                       "longitude": [""], "elevation": [""]}, **kwargs)


def normalize_station_codes(stations: pd.DataFrame) -> pd.DataFrame:
    """Make `id` and the three code columns agree, with `id` as the authority.

    With an `id` column, the codes are rebuilt by splitting it on its two dots;
    SEED codes cannot contain a dot, so the split is exact. Without one, the
    id is built from the codes, and a code that reads like a number that went
    through a float (`"0.0"`, `"1.0"`) is refused, because its leading zeros
    are gone and cannot be recovered from that column alone.
    """
    stations = stations.copy()
    for c in STATION_TEXT_COLUMNS:
        if c in stations.columns:
            stations[c] = stations[c].fillna("").astype(str)
    if "id" in stations.columns:
        parts = stations["id"].str.split(".", expand=True)
        if parts.shape[1] != 3 or parts.isna().any(axis=None):
            bad = stations["id"][stations["id"].str.count(r"\.") != 2]
            raise ValueError(f"{len(bad)} station id(s) are not NET.STA.LOC: "
                             f"{list(bad[:5])}")
        for i, c in enumerate(STATION_CODE_COLUMNS):
            stations[c] = parts[i]
        return stations
    missing = [c for c in STATION_CODE_COLUMNS if c not in stations.columns]
    if missing:
        raise ValueError(f"station table has no `id` and no {missing}")
    for c in STATION_CODE_COLUMNS:
        bad = stations[c][stations[c].str.match(_NUMBER_LIKE_RE)]
        if len(bad):
            raise ValueError(f"{len(bad)} {c} value(s) went through a float "
                             f"({sorted(bad.unique())[:5]}); re-read the source "
                             f"with read_station_csv")
    stations["id"] = (stations["network_code"] + "." + stations["station_code"]
                      + "." + stations["location_code"])
    return stations


def parse_year_day(x: str) -> datetime.date:
    """A `%Y.%j` STRING, as the shard queue writes it: always three digits."""
    return datetime.datetime.strptime(x, "%Y.%j").date()


# Day-of-year encodings that a station table may carry. Since 2026-09-29
# `stations.parquet` stores real dates; tables written before that carry a
# float `YYYY.DDD`, and both must decode to the same day.
_YEAR_DAY_RE = re.compile(r"^\s*(\d{4})\.(\d{1,3})\s*$")


def station_date(value) -> Optional[datetime.date]:
    """One station date, from whatever the table holds, or None if unusable.

    Accepts a date, a datetime/Timestamp, a `YYYY.DDD` string, and the legacy
    float. **The float must be decoded numerically, never through `str()`.**
    `str(2010.21)` is `"2010.21"`, and `strptime(..., "%Y.%j")` reads that as
    day 21 rather than day 210 - the encoding is three zero-padded digits and
    the float drops the trailing zero. That misparse planned 837 western and
    331 obs station-locations to stop early, 121,692 station-days that were
    never picked; see docs/rerun_2026/30_station_dates.md. Every day-of-year
    divisible by ten was wrong, and always in the direction of doing less.
    """
    if value is None:
        return None
    if value != value:                              # NaN, and pandas NaT
        return None
    if isinstance(value, datetime.datetime):        # NaT is one of these too
        return value.date()
    if isinstance(value, datetime.date):
        return value
    # numpy/pandas datetime64 and NaT
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if to_pydatetime is not None:
        try:
            return to_pydatetime().date()
        except Exception:
            return None
    if isinstance(value, str):
        m = _YEAR_DAY_RE.match(value)  # noqa: E501
        if not m:
            try:
                return datetime.date.fromisoformat(value[:10])
            except ValueError:
                return None
        year, doy = int(m.group(1)), int(m.group(2).ljust(3, "0"))
    else:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        year = int(math.floor(v))
        # Three zero-padded digits after the point, so the fraction is the
        # day-of-year in thousandths: 2010.21 is day 210, not day 21.
        doy = int(round((v - year) * 1000))
    if doy < 1 or doy > 366:
        return None
    try:
        return datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
    except ValueError:                              # year out of range
        return None
