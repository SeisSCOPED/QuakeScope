"""Pure helpers shared across the pipeline.

Deliberately dependency-light: the v3 worker imports `parse_year_day` from here
on every shard, and this module must not pull in `pymongo`. The DocumentDB
client lives in `mongo_db` and is re-exported lazily below, so existing
`from .utils import SeisBenchDatabase` keeps working without making every
picking job depend on a database driver it never opens.
"""

import datetime
from typing import Any

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
        sta_start = parse_year_day(str(sta["start_date"]))
        sta_end = parse_year_day(str(sta["end_date"]))
        if (sta_start <= end) and (sta_end >= start):
            match.append(i)
    return stations.iloc[match]


def parse_year_day(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, "%Y.%j").date()
