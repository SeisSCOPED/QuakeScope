import argparse
import datetime
import logging
import os

import boto3
import numpy as np
import pandas as pd
import pytz
from botocore.config import Config

from .parameters import *
from .utils import SeisBenchDatabase, filter_station_by_start_end_date

logger = logging.getLogger("submit_helper")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class SubmitHelper:
    """
    A helper class to submit picking and association jobs.

    Make sure the job queue and job names are set in the parameters file

    :param start: Start date
    :param end: End date
    :param extent: Study area (minlat, maxlat, minlon, maxlon)
    :param region: AWS region to run jobs
    :param station_group_size: Number of stations to process in a single picking job
    :param day_group_size: Number of days to process in a single picking/association job
    :param station_ids: Explicit list of station ids (NET.STA.LOC) to restrict the submission to
    """

    def __init__(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        extent: tuple[float, float, float, float],
        network: str,
        db: SeisBenchDatabase,
        region: str,
        environ: dict = {},
        station_group_size: int = 40,
        day_group_size: int = 20,
        model: str = "PhaseNet",
        weight: str = "instance",
        station_ids: list = None,
    ):
        self.start = start
        self.end = end
        self.extent = extent
        self.network = network
        self.station_ids = station_ids
        self.db = db
        self.environ = environ
        self.region = region
        self.station_group_size = station_group_size
        self.day_group_size = day_group_size
        self.model = model
        self.weight = weight
        self.client = boto3.client("batch", config=Config(region_name=region))
        self.shared_parameters = {
            "db_uri": self.db.db_uri,
            "database": self.db.database.name,
            "model": self.model,
            "weight": self.weight,
        }

        self._environ_kv = [{"name": k, "value": v} for k, v in self.environ.items()]

    def submit_jobs(self, command: str) -> None:
        if command == "pick":
            self.submit_pick_jobs()
        elif command == "associate":
            self.submit_association_jobs()
        else:
            raise ValueError(f"Unknown command '{command}'")

    def submit_pick_jobs(self) -> None:
        # for submission logging
        submissions = []
        os.makedirs("../submissions/", exist_ok=True)
        ts = (
            datetime.datetime.today()
            .astimezone(tz=pytz.timezone("US/Pacific"))
            .strftime("%Y-%m-%dT%H-%M")
        )

        # get stations based on extent and/or network code
        # ... and operation time
        stations = self.db.get_stations(extent=self.extent, network=self.network)
        stations = filter_station_by_start_end_date(stations, self.start, self.end)

        # restrict to an explicit station list if one was provided
        if self.station_ids:
            requested = set(self.station_ids)
            stations = stations[stations["id"].isin(requested)]
            missing = requested - set(stations["id"])
            if missing:
                logger.warning(
                    f"{len(missing)} stations from the list not found in the database "
                    f"(or outside extent/network/date filters): {','.join(sorted(missing))}"
                )

        days = np.arange(self.start, self.end, datetime.timedelta(days=1))
        logger.info(
            f"Starting picking jobs for {len(stations)} stations and {len(days)} days"
        )

        njobs = 0
        i = 0
        while i < len(stations) - 1:
            sub_stations = ",".join(
                stations["id"].iloc[i : i + self.station_group_size]
            )

            pick_jobs = []
            j = 0
            while j < len(days) - 1:
                job_name = f"picking_{ts}_{i}_{j}"
                day0 = days[j].astype(datetime.datetime).strftime("%Y.%j")
                day1 = (
                    days[min(j + self.day_group_size, len(days) - 1)]
                    .astype(datetime.datetime)
                    .strftime("%Y.%j")
                )
                parameters = {
                    "start": day0,
                    "end": day1,
                    "job_name": job_name,
                    "stations": sub_stations,
                }
                submissions += [parameters]

                pick_jobs.append(
                    self.client.submit_job(
                        jobName=job_name,
                        jobQueue=JOB_QUEUE,
                        jobDefinition=JOB_DEFINITION_PICKING,
                        parameters={
                            **parameters,
                            **self.shared_parameters,
                        },
                        containerOverrides={"environment": self._environ_kv},
                    )
                )

                j += self.day_group_size
            i += self.station_group_size
            njobs += len(pick_jobs)

        pd.DataFrame().from_records(submissions).to_csv(
            f"../submissions/picking_{ts}.csv", index=False
        )
        logger.info(f"{njobs} jobs submitted in total.")
        logger.info(f"See ../submissions/picking_{ts}.csv for logging.")

    def submit_association_jobs(self) -> None:
        stations = self.db.get_stations(self.extent)
        days = np.arange(self.start, self.end, datetime.timedelta(days=1))
        extent = ",".join([str(x) for x in self.extent])

        logger.debug(
            f"Starting association jobs for {len(stations)} stations and {len(days)} days"
        )

        i = 0
        while i < len(days) - 1:
            day0 = days[i].astype(datetime.datetime).strftime("%Y.%j")
            day1 = (
                days[min(i + self.day_group_size, len(days) - 1)]
                .astype(datetime.datetime)
                .strftime("%Y.%j")
            )

            association_jobs = []
            parameters = {"start": day0, "end": day1, "extent": extent}
            logger.debug(f"Submitting association job with: {parameters}")
            association_jobs.append(
                self.client.submit_job(
                    jobName=f"association_{i}",
                    jobQueue=JOB_QUEUE,
                    jobDefinition=JOB_DEFINITION_ASSOCIATION,
                    parameters={**parameters, **self.shared_parameters},
                )
            )
            i += self.day_group_size


def parse_year_day(x: str) -> datetime.date:
    return datetime.datetime.strptime(x, "%Y.%j").date()


def read_station_file(path: str) -> list[str]:
    """
    Reads a station list: either a CSV with an 'id' column or a plain text
    file with one NET.STA.LOC id per line. Lines starting with # are ignored.
    """
    with open(path) as f:
        first = f.readline()
    if "id" in [c.strip() for c in first.split(",")]:
        ids = pd.read_csv(path)["id"].astype(str).str.strip()
        return sorted(set(ids[ids != ""]))
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line.split(",")[0].strip())
    return sorted(set(ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        type=str,
        help="Subroutine to execute. Should be either pick or associate.",
    )
    parser.add_argument(
        "start",
        type=parse_year_day,
        help="Format: YYYY.DDD (included)",
    )
    parser.add_argument(
        "end",
        type=parse_year_day,
        help="Format: YYYY.DDD (not included)",
    )
    parser.add_argument(
        "--extent",
        type=str,
        help="Comma separated: minlat, maxlat, minlon, maxlon",
    )
    parser.add_argument(
        "--network",
        type=str,
        help="Network to pick. Comma separated if multiple submitted.",
    )
    parser.add_argument(
        "--database", type=str, default="tutorial", help="MongoDB database name."
    )
    parser.add_argument(
        "--region", type=str, default="us-east-2", help="Working region on AWS."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PhaseNet",
        help="SeisBench model type for picking jobs.",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="instance",
        help="Model weights for picking jobs, e.g. a custom weight baked into the image.",
    )
    parser.add_argument(
        "--station_file",
        type=str,
        help="Path to a station list restricting the submission: a text file with "
        "one NET.STA.LOC id per line (# comments allowed), or a CSV with an 'id' column.",
    )
    args = parser.parse_args()

    assert (
        args.extent or args.network or args.station_file
    ), "Either extent, network, or station_file needs to be set"

    if args.station_file:
        station_ids = read_station_file(args.station_file)
        logger.info(f"Read {len(station_ids)} station ids from {args.station_file}")
    else:
        station_ids = None

    if args.extent:
        extent = tuple([float(x) for x in args.extent.split(",")])
        assert len(extent) == 4, "Extent needs to be exactly 4 coordinates"
    else:
        extent = None

    if ES_OAUTH2__REFRESH_TOKEN:
        environ = {"ES_OAUTH2__REFRESH_TOKEN": ES_OAUTH2__REFRESH_TOKEN}
        logger.info(f"EarthScope refresh token applied: {ES_OAUTH2__REFRESH_TOKEN}")
    else:
        environ = {}
        logger.info(f"EarthScope refresh token empty.")

    environ["EARTHSCOPE_S3_ACCESS_POINT"] = EARTHSCOPE_S3_ACCESS_POINT

    db = SeisBenchDatabase(DOCDB_ENDPOINT_URI, args.database)
    helper = SubmitHelper(
        start=args.start,
        end=args.end,
        extent=extent,
        network=args.network,
        db=db,
        region=args.region,
        environ=environ,
        model=args.model,
        weight=args.weight,
        station_ids=station_ids,
    )
    helper.submit_jobs(args.command)


if __name__ == "__main__":
    main()
