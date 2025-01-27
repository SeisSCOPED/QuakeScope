import argparse
import asyncio
import datetime
import functools
import io
import logging
import re
import time
from typing import AsyncIterator, Optional

import numpy as np
import obspy
import pyocto
import seisbench
import seisbench.models as sbm
import seisbench.util as sbu
from botocore.exceptions import ClientError
from bson import ObjectId

from .s3_helper import CompositeS3ObjectHelper
from .utils import SeisBenchDatabase, parse_year_day

logger = logging.getLogger("sb_picker")


def main() -> None:
    """
    This main function serves as the entry point to all functionality available in the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        type=str,
        help="Subroutine to execute. See below for available functions.",
    )
    parser.add_argument(
        "--db_uri", type=str, required=True, help="URI of the MongoDB cluster."
    )
    parser.add_argument(
        "--database", type=str, required=True, help="MongoDB database name."
    )
    parser.add_argument(
        "--stations",
        type=str,
        required=False,
        help="Stations (comma separated) in format NET.STA.LOC.CHA without component.",
    )
    parser.add_argument(
        "--start",
        type=parse_year_day,
        required=False,
        help="Format: YYYY.DDD (included).",
    )
    parser.add_argument(
        "--end",
        type=parse_year_day,
        required=False,
        help="Format: YYYY.DDD (not included).",
    )
    parser.add_argument(
        "--extent",
        type=str,
        required=False,
        help="Comma separated: minlat, maxlat, minlon, maxlon",
    )
    parser.add_argument(
        "--components", type=str, default="ZNE12", help="Components to scan."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PhaseNet",
        help="Model type. Must be available in SeisBench.",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="instance",
        help="Model weights to load through SeisBench from_pretrained.",
    )
    parser.add_argument(
        "--p_threshold", default=0.2, type=float, help="Picking threshold for P waves."
    )
    parser.add_argument(
        "--s_threshold", default=0.2, type=float, help="Picking threshold for S waves."
    )
    parser.add_argument(
        "--data_queue_size",
        default=5,
        type=int,
        help="Buffer size for data preloading.",
    )
    parser.add_argument(
        "--pick_queue_size",
        default=5,
        type=int,
        help="Buffer size for picking results.",
    )
    parser.add_argument(
        "--delay", default=30, type=int, help="Add random delay when starting the job."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enables additional debug output."
    )
    args = parser.parse_args()

    if args.debug:  # Setup debug logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    # Set up data base for results and data source
    db = SeisBenchDatabase(args.db_uri, args.database)
    s3 = S3DataSource(
        stations=args.stations,
        start=args.start,
        end=args.end,
        components=args.components,
        db=db,
    )
    if args.extent is None:
        extent = None
    else:
        extent = tuple([float(x) for x in args.extent.split(",")])
        assert len(extent) == 4, "Extent needs to be exactly 4 coordinates"

    # Set up main class handling the commands
    picker = S3MongoSBBridge(
        s3=s3,
        db=db,
        model=args.model,
        weight=args.weight,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
        data_queue_size=args.data_queue_size,
        pick_queue_size=args.pick_queue_size,
        extent=extent,
    )

    if args.command == "pick":
        delay = np.random.randint(args.delay)
        logger.debug(f"Delaying this job for {delay} sec.")
        time.sleep(delay)
        picker.run_picking()
    elif args.command == "associate":
        picker.run_association(args.start, args.end)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


class S3DataSource:
    """
    This class provides functionality to load waveform data from an S3 bucket.
    """

    def __init__(
        self,
        start: Optional[datetime.date] = None,
        end: Optional[datetime.date] = None,
        stations: Optional[str] = None,
        components: str = "ZNE12",
        db: SeisBenchDatabase = None,
    ):
        self.start = start
        self.end = end
        self.components = components
        if stations is None:
            self.stations = []
            self.networks = []
        else:
            self.stations = stations.split(",")
            self.networks = list(set([s.split(".")[0] for s in self.stations]))
        self.db = db
        self.s3helper = CompositeS3ObjectHelper()
        logger.debug(f"Initializing s3 access to {', '.join(self.s3helper.fs.keys())}")

    async def load_waveforms(self) -> AsyncIterator[list]:
        """
        Load the waveforms. This function is async to allow loading data in parallel with processing.
        The function releases the GIL when reading from the S3 bucket.
        The iterator returns data by station and within each station day by day.
        Data from all channels of a station is returned simultaneously.
        This matches the typical access pattern required for single-station phase pickers.
        """
        days = np.arange(self.start, self.end, datetime.timedelta(days=1))

        for day in days:
            day = day.astype(datetime.datetime)

            # get a list of exist URIs
            # ls can be slow, but it merges many small open request
            # and effectively reduced the total number of requests
            avail_uri = {}
            for net in self.networks:
                avail_uri[net] = []
                # use the corresponding fs for the network
                fs = self.s3helper.get_filesystem(net)
                prefix = self.s3helper.get_prefix(
                    net, day.strftime("%Y"), day.strftime("%j")
                )
                try:
                    avail_uri[net] += fs.ls(prefix)
                except FileNotFoundError:
                    logger.debug(f"Path does not exist {prefix}")
                    pass
                except PermissionError as e:
                    logger.debug(e.args[0])
                    raise e

            for station in self.stations:
                meta = self.db.database["stations"].find_one({"id": station})
                all_channels = meta["channels"].split(",")
                check = {
                    cha: self.db.get_picks_record(station, day, cha)
                    for cha in all_channels
                }
                # if all channel got results
                if all(check.values()):
                    logger.debug(
                        f"Skipping {station.ljust(11)} {day.strftime('%Y.%j')} < picks found at all {meta['channels']} channel"
                    )
                    continue

                net, sta, loc = station.split(".")
                dc = self.s3helper.get_data_center(net)
                logger.debug(
                    f"Loading  {station.ljust(11)} {day.strftime('%Y.%j')} @ {dc}"
                )
                stream = obspy.Stream()

                if dc in ["scedc", "ncedc"]:
                    for channel in all_channels:
                        if check[channel]:
                            logger.debug(
                                f"Skipping {station.ljust(11)} {day.strftime('%Y.%j')} < picks found at {channel} channel"
                            )
                            continue
                        for uri in self._generate_waveform_uris(
                            net, sta, loc, channel, day
                        ):
                            if uri in avail_uri[net]:
                                stream += await asyncio.to_thread(
                                    self._read_waveform_from_s3, uri, net
                                )
                elif dc == "earthscope":
                    # use the first one: they should be all same
                    r = self._generate_waveform_uris(net, sta, loc, "NA", day)[0]
                    # earthscope object name has version number
                    uri = list(filter(lambda v: re.match(r, v), avail_uri[net]))
                    if len(uri) > 0:
                        s = await asyncio.to_thread(
                            self._read_waveform_from_s3, uri[0], net
                        )
                        for channel in all_channels:
                            if check[channel]:
                                logger.debug(
                                    f"Skipping {station.ljust(11)} {day.strftime('%Y.%j')} < picks found at {channel} channel"
                                )
                                continue
                            stream += s.select(channel=f"{channel}?")
                else:
                    raise NotImplemented(f"Data center not supported: {dc}")

                if len(stream) > 0:
                    # yield stream with all candidate channels for one station, day long stream, with metadata
                    yield [stream, station, day]
                else:
                    logger.debug(
                        f"Empty \t{station.ljust(11)} {day.strftime('%Y.%j')} @ {dc}"
                    )

    def _read_waveform_from_s3(self, uri, net) -> obspy.Stream:
        """
        Failure tolerant method for reading data from S3.

        OSError#5: accessing non-authorized earthscope data. Return empty stream.
        PermissionError: EarthScope temporary credential expired. Refresh the credential and retry.
        ClientError: S3 overloaded, the job will sleep for 5 seconds and retry until return.
        FileNotFoundError: file not exist.
        ValueError: certain types of corrupt files.
        TypeError: certain types of empty mSEED files, i.e. in NCEDC

        """
        while True:
            fs = self.s3helper.get_filesystem(net)
            try:
                buff = io.BytesIO(fs.read_bytes(uri))
                return obspy.read(buff)
            except OSError as e:
                if e.errno == 5:
                    logger.warning(f"Not authorized to access this resource: {uri}")
                    return obspy.Stream()
            except PermissionError as e:
                logger.debug(e.args[0])
                self.s3helper.update_es_filesystem()
                logger.warning("Credential refreshed.")
            except ClientError:
                logger.warning(f"S3 might be busy. Sleep for 5 seconds and retry.")
                time.sleep(5)
            except FileNotFoundError:
                return obspy.Stream()
            except ValueError:
                return obspy.Stream()
            except TypeError:
                return obspy.Stream()

    def _generate_waveform_uris(
        self, net: str, sta: str, loc: str, cha: str, date: datetime.date
    ) -> list[str]:
        """
        Generates a list of S3 uris for the requested data
        """
        uris = []
        year = date.strftime("%Y")
        day = date.strftime("%j")
        for c in self.components:
            # go through all possible components...
            uris.append(self.s3helper.get_s3_path(net, sta, loc, cha, year, day, c))

        return uris


class S3MongoSBBridge:
    """
    This bridge connects an S3DataSource, a MongoDB database (represented by the SeisBenchDatabase) and
    the processing for picking and association (implemented directly in the class).
    Additional functionality is provided for submitting jobs to AWS Batch, however, these functions are also
    available separately in submit.py.
    """

    def __init__(
        self,
        s3: S3DataSource,
        db: SeisBenchDatabase,
        model: Optional[str] = None,
        weight: Optional[str] = None,
        p_threshold: Optional[float] = None,
        s_threshold: Optional[float] = None,
        data_queue_size: Optional[int] = None,
        pick_queue_size: Optional[int] = None,
        extent: Optional[tuple[float, float, float, float]] = None,
    ):
        self.extent = extent
        if model is not None:
            self.model = self.create_model(model, weight, p_threshold, s_threshold)
        else:
            self.model = None
        self.model_name = model
        self.weight = weight
        self.p_threshold = p_threshold
        self.s_threshold = s_threshold

        self.s3 = s3
        self.db = db

        self.data_queue_size = data_queue_size
        self.pick_queue_size = pick_queue_size

        self.station_group_size = 8
        self.day_group_size = 2

        self._run_id = None

    @property
    def run_id(self) -> ObjectId:
        """
        A unique run_id that is saved in the database along with the configuration for reproducibility.
        """
        if self._run_id is None:
            self._run_id = self.db.write_run_data(
                model=self.model_name,
                weight=self.weight,
                p_threshold=self.p_threshold,
                s_threshold=self.s_threshold,
                components_loaded=self.s3.components,
                seisbench_version=seisbench.__version__,
                weight_version=self.model.weights_version,
            )
        return self._run_id

    @staticmethod
    def create_model(
        model: str, weight: str, p_threshold: float, s_threshold: float
    ) -> sbm.WaveformModel:
        """
        Loads a SeisBench model
        """
        model = sbm.__getattribute__(model).from_pretrained(weight)
        model.default_args["P_threshold"] = p_threshold
        model.default_args["S_threshold"] = s_threshold
        return model

    def run_association(self, t0: datetime.datetime, t1: datetime.datetime):
        """
        Runs the phase association for the provided time range and the extent defined in self.extent.
        """
        t0 = self._date_to_datetime(t0)
        t1 = self._date_to_datetime(t1)
        stations = self.db.get_stations(self.extent)
        logger.debug(
            f"Associating {len(stations)} stations: " + ",".join(stations["id"].values)
        )

        picks = self.db.get_picks(list(stations["id"].values), t0, t1)
        picks.rename(columns={"trace_id": "station"}, inplace=True)
        picks["time"] = picks["time"].apply(lambda x: x.timestamp())
        logger.debug(f"Associating {len(picks)} picks")

        if len(picks) == 0:
            logger.warning("Found no picks, exiting")
            return

        minlat, maxlat, minlon, maxlon = self.extent
        # TODO: PyOcto configuration
        velocity_model = pyocto.VelocityModel0D(
            p_velocity=6.0,
            s_velocity=6.0 / 1.75,
            tolerance=1.5,
            association_cutoff_distance=150,
        )
        associator = pyocto.OctoAssociator.from_area(
            (minlat, maxlat),
            (minlon, maxlon),
            (0, 50),
            velocity_model,
            time_before=150,
        )
        stations = associator.transform_stations(stations)

        events, assignments = associator.associate(picks, stations)
        logger.debug(
            f"Found {len(events)} events with {len(assignments)} total picks (of {len(picks)} input picks)"
        )

        utc_from_timestamp = functools.partial(
            datetime.datetime.fromtimestamp, tz=datetime.timezone.utc
        )
        if len(events) > 0:
            events = associator.transform_events(events)
            events["time"] = events["time"].apply(utc_from_timestamp)

        self.db.write_events(events, assignments, picks)

    def run_picking(self) -> None:
        """
        Perform the picking
        """
        asyncio.run(self._run_picking_async())

    async def _run_picking_async(self) -> None:
        """
        An async implementation of the data loading, picking, and output routine.
        All three tasks are started in parallel with buffer queues in between.
        This means that the next input data is loaded while the current one is picked.
        Similarly, the outputs are written to MongoDB while the next data is already being processed.
        To guarantee this, all underlying functions have been designed to release the GIL.
        """
        data = asyncio.Queue(self.data_queue_size)
        picks = asyncio.Queue(self.pick_queue_size)

        task_load = self._load_data(data)
        task_pick = self._pick_data(data, picks)
        task_db = self._write_picks_to_db(picks)

        await asyncio.gather(task_load, task_pick, task_db)

    async def _load_data(
        self,
        data: asyncio.Queue[list | None],
    ) -> None:
        """
        An async function getting data from the S3 sources and putting it into a queue.
        """
        async for stream, station, day in self.s3.load_waveforms():
            if len(stream) > 0:
                for channel in list(set([t.stats.channel[:2] for t in stream])):
                    stream_c = stream.select(channel=f"{channel}?")
                    if self.db.get_picks_record(station, day, channel) is not None:
                        logger.debug(
                            f"Skipping {station.ljust(11)} {day.strftime('%Y.%j')} < picks found"
                        )
                        continue

                    # put stream with one channel type
                    await data.put([stream_c, station, day, channel])
            else:
                # put empty stream
                await data.put([stream, station, day, None])

        # put None marking the end of the data queue
        await data.put(None)

    async def _pick_data(
        self,
        data: asyncio.Queue[list | None],
        picks: asyncio.Queue[list | None],
    ) -> None:
        """
        An async function taking data from a queue, picking it and returning the results to an output queue.
        """
        while True:
            _st_sta_day_cha = await data.get()
            if _st_sta_day_cha is None:
                await picks.put(None)
                break

            stream, station, day, channel = _st_sta_day_cha
            if len(stream) == 0:
                continue

            logger.debug(f"Picking  {station.ljust(11)} {day.strftime('%Y.%j')}")

            stream_annotations = await asyncio.to_thread(self.model.classify, stream)
            await picks.put([stream_annotations.picks, station, day, channel])

    async def _write_picks_to_db(self, picks: asyncio.Queue[list | None]) -> None:
        """
        An async function reading picks from a queue and putting them into the MongoDB.
        """
        while True:
            _pk_sta_day_cha = await picks.get()
            if _pk_sta_day_cha is None:
                break

            stream_picks, station, day, channel = _pk_sta_day_cha
            if len(stream_picks) == 0:
                continue

            id = f"{station}.{channel}"
            logger.debug(
                f"Putting  {id.ljust(11)} {day.strftime('%Y.%j')}"
                f" > {(str(len(stream_picks))).ljust(3)} picks"
            )
            await asyncio.to_thread(
                self._write_single_picklist_to_db, stream_picks, channel
            )

    def _write_single_picklist_to_db(self, picks: sbu.PickList, channel: str) -> None:
        """
        Converts picks into records that can be submitted to MongoDB and writes them.
        Populates the `picks` and `picks_record` collection.
        """
        self.db.insert_many_ignore_duplicates(
            "picks",
            [
                {
                    "trace_id": pick.trace_id,
                    "channel": channel,
                    "time": pick.peak_time.datetime,
                    "confidence": float(pick.peak_value),
                    "phase": pick.phase,
                    "run_id": self.run_id,
                }
                for pick in picks
            ],
        )

        self.db.insert_many_ignore_duplicates(
            "picks_record",
            [
                {
                    "trace_id": picks[0].trace_id,
                    "channel": channel,
                    "year": picks[0].peak_time.datetime.year,
                    "doy": int(picks[0].peak_time.datetime.strftime("%-j")),
                    "picks_number": len(picks),
                    "run_id": self.run_id,
                }
            ],
        )

    @staticmethod
    def _date_to_datetime(t: datetime.date | datetime.datetime) -> datetime.datetime:
        """
        Helper function to homogenize time formats
        """
        if isinstance(t, datetime.date):
            return datetime.datetime.combine(t, datetime.datetime.min.time())
        return t


if __name__ == "__main__":
    main()
