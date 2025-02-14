import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd
import pymongo
from bson import ObjectId
from pymongo.errors import BulkWriteError, DuplicateKeyError
from pymongo.results import InsertManyResult

logger = logging.getLogger("sb_picker")


class SeisBenchDatabase(pymongo.MongoClient):
    """
    A MongoDB Client designed to handle all necessary tables for creating a simple earthquake catalog.
    It provides useful helper functions and a structure.
    """

    def __init__(self, db_uri: str, database: str, **kwargs: Any) -> None:
        super().__init__(db_uri, **kwargs)

        self.db_uri = db_uri
        self.database = super().__getitem__(database)

        self.colls = {"picks", "stations", "sb_runs", "events", "assignments"}
        self._setup()

    def _setup(self) -> None:
        """
        Setup indices for the main tables for faster access.
        Tables are generally created lazily.
        """
        pick_coll = self.database["picks"]
        if "pick_idx" not in pick_coll.index_information():
            pick_coll.create_index(
                ["trace_id", "channel", "phase", "time"], unique=True, name="pick_idx"
            )

        station_coll = self.database["stations"]
        if "station_idx" not in station_coll.index_information():
            station_coll.create_index(["id"], unique=True, name="station_idx")

        picks_record_coll = self.database["picks_record"]
        if "picks_record_idx" not in picks_record_coll.index_information():
            picks_record_coll.create_index(
                ["trace_id", "channel", "year", "doy"],
                unique=True,
                name="picks_record_idx",
            )

    def get_picks_record(
        self, station: str, day: datetime.date, channel: str, key: dict = {}
    ) -> dict:
        filt = {
            "trace_id": station,
            "year": day.year,
            "doy": int(day.strftime("%-j")),
            "channel": channel,
        }
        return self.database["picks_record"].find_one(filt, key)

    def get_stations(
        self, extent: tuple[float, float, float, float] = None, network: str = None
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with all stations by a certain filter
        """
        filt = {}
        if extent:
            minlat, maxlat, minlon, maxlon = extent
            filt["latitude"] = {"$gt": minlat, "$lt": maxlat}
            filt["longitude"] = {"$gt": minlon, "$lt": maxlon}

        if network:
            nets = network.split(",")
            filt["network_code"] = {"$in": nets}

        cursor = self.database["stations"].find(filt)

        return pd.DataFrame(list(cursor))

    def get_picks(
        self, station_ids: list[str], t0: datetime.datetime, t1: datetime.datetime
    ) -> pd.DataFrame:
        """
        Loads picks for a list of stations during a given time range from the database.
        The database has already been configured with indices that speed up this query.
        """
        cursor = self.database["picks"].find(
            {
                "time": {"$gt": t0, "$lt": t1},
                "trace_id": {"$in": station_ids},
            }
        )

        return pd.DataFrame(cursor)

    def write_stations(self, stations: pd.DataFrame) -> None:
        self.insert_many_ignore_duplicates("stations", stations.to_dict("records"))

    def write_run_data(self, **kwargs: Any) -> ObjectId:
        kwargs["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
        return self.database["sb_runs"].insert_one(kwargs).inserted_id

    def write_events(
        self, events: pd.DataFrame, assignments: pd.DataFrame, picks: pd.DataFrame
    ) -> None:
        """
        Writes events and the associated picks into the MongoDB. Ensures that the pick and event ids are consistent
        with the ones used in the database.
        """
        # Put events and get mongodb ids, replace event and pick ids with their mongodb counterparts,
        # write assignments to database
        event_result = self.insert_many_ignore_duplicates(
            "events", events.to_dict("records")
        )

        event_key = pd.DataFrame(
            {
                "event_id": event_result.inserted_ids,
                "event_idx": events["idx"].values,
            }
        )
        pick_key = pd.DataFrame(
            {
                "pick_id": picks["_id"],
                "pick_idx": np.arange(len(picks)),
            }
        )

        merged = pd.merge(event_key, assignments, on="event_idx")
        merged = pd.merge(merged, pick_key, on="pick_idx")

        merged = merged[["event_id", "pick_id"]]

        self.database["assignments"].insert_many(merged.to_dict("records"))

    def insert_many_ignore_duplicates(
        self, key: str, entries: list[dict[str, Any]]
    ) -> InsertManyResult:
        """
        Inserts many keys into a table while ignoring any duplicates.
        All other errors in inserting the data are passed to the user.
        """
        try:
            return self.database[key].insert_many(
                entries,
                ordered=False,  # Not ordered to make sure every query is sent
            )
        except DuplicateKeyError:
            logger.warning(
                f"Some duplicate entries have been skipped while writing to collection {key}"
            )
        except BulkWriteError as e:
            # See https://www.mongodb.com/docs/manual/reference/error-codes/ for full error code
            if all(x["code"] == 11000 for x in e.details["writeErrors"]):
                logger.warning(
                    f"Some duplicate entries have been skipped in collection {key}"
                )
            else:
                raise e


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
