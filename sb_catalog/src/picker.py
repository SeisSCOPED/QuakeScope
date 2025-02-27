import argparse
import asyncio
import datetime
import functools
import logging
import time
from typing import Optional

import numpy as np
import obspy
import pyocto
import seisbench
import seisbench.models as sbm
import seisbench.util as sbu
from bson import ObjectId

from .amplitude_extractor import AmplitudeExtractor
from .classifier import QuakeXNet
from .s3_helper import S3DataSource
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
        "--classifier",
        action="store_true",
        help="Append the classifier to the picking job.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable additional debug output."
    )
    args = parser.parse_args()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.debug:  # Setup debug logging
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Delay the job to scatter requests
    delay = np.random.randint(args.delay)
    logger.info(f"Delaying this job for {delay} sec.")
    time.sleep(delay)

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
    bridge = S3MongoSBBridge(
        s3=s3,
        db=db,
        model=args.model,
        weight=args.weight,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
        data_queue_size=args.data_queue_size,
        pick_queue_size=args.pick_queue_size,
        extent=extent,
        classifier=args.classifier,
    )

    if args.command == "pick":
        bridge.run_picking()
    elif args.command == "associate":
        bridge.run_association(args.start, args.end)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


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
        classifier: Optional[bool] = None,
    ):
        self.extent = extent

        # model preparation
        if model is not None:
            self.model = self.create_model(model, weight, p_threshold, s_threshold)
        else:
            self.model = None

        if classifier is not None:
            self.classifier = QuakeXNet.from_pretrained("base")
        else:
            self.classifier = None
        self.amp_extor = AmplitudeExtractor()
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
        picks.rename(columns={"tid": "station"}, inplace=True)
        picks["time"] = picks["peak"].apply(lambda x: x.timestamp())
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
        data_queue = asyncio.Queue(self.data_queue_size)
        picks_queue = asyncio.Queue(self.pick_queue_size)

        task_load = self._load_data(data_queue)
        task_pick = self._pick_data(data_queue, picks_queue)
        task_db = self._write_picks_to_db(picks_queue)

        await asyncio.gather(task_load, task_pick, task_db)

    async def _load_data(
        self,
        data_queue: asyncio.Queue[list | None],
    ) -> None:
        """
        An async function getting data from the S3 sources and putting it into a queue.
        """
        async for stream, station, day in self.s3.load_waveforms():
            if len(stream) > 0:
                for channel in list(set([t.stats.channel[:2] for t in stream])):
                    stream_c = stream.select(channel=f"{channel}?")

                    # put stream with one channel type
                    id = f"{station}.{channel}"
                    if (
                        len(stream_c) > 150
                    ):  # maximum number of data gap (3*50 per component)
                        logger.debug(
                            f"Skip {id.ljust(11)} {day.strftime('%Y.%j')} < too many gaps"
                        )
                        stream_c = obspy.Stream()
                    else:
                        logger.debug(f"Send {id.ljust(11)} {day.strftime('%Y.%j')}")

                    await data_queue.put([stream_c, station, day, channel])
            else:
                # put empty stream
                await data_queue.put([stream, station, day, None])

        # put None marking the end of the data queue
        await data_queue.put(None)

    async def _pick_data(
        self,
        data_queue: asyncio.Queue[list | None],
        picks_queue: asyncio.Queue[list | None],
    ) -> None:
        """
        An async function taking data from a queue, picking it and returning the results to an output queue.
        """
        while True:
            _st_sta_day_cha = await data_queue.get()
            if _st_sta_day_cha is None:
                await picks_queue.put(None)
                break

            stream, station, day, channel = _st_sta_day_cha
            id = f"{station}.{channel}"
            logger.debug(f"Pick {id.ljust(11)} {day.strftime('%Y.%j')}")
            if len(stream) == 0:
                logger.debug(
                    f"Skip {station.ljust(11)} {day.strftime('%Y.%j')} < stream is empty due to exception"
                )
                await picks_queue.put([sbu.PickList(), [], [], station, day, channel])
            else:
                # do picking
                stream_annotations = await asyncio.to_thread(
                    self.model.classify, stream
                )
                # extract amplitudes
                stream_amplitudes = await asyncio.to_thread(
                    self.amp_extor.extract_amplitudes,
                    stream,
                    stream_annotations.picks,
                    self.s3.inventory,
                )

                # classifier
                if self.classifier and (channel in ["BH", "HH"]):
                    stream_classifier = await asyncio.to_thread(
                        self.classifier.classify, stream
                    )
                else:
                    stream_classifier = []
                await picks_queue.put(
                    [
                        stream_annotations.picks,
                        stream_amplitudes,
                        stream_classifier,
                        station,
                        day,
                        channel,
                    ]
                )

    async def _write_picks_to_db(self, picks_queue: asyncio.Queue[list | None]) -> None:
        """
        An async function reading picks from a queue and putting them into the MongoDB.
        """
        while True:
            _pk_amp_clf_sta_day_cha = await picks_queue.get()
            if _pk_amp_clf_sta_day_cha is None:
                break

            (
                picks,
                amplitudes,
                classifies,
                station,
                day,
                channel,
            ) = _pk_amp_clf_sta_day_cha

            id = f"{station}.{channel}"
            logger.info(
                f"Put  {id.ljust(11)} {day.strftime('%Y.%j')}"
                f" > {(str(len(picks))).ljust(3)} phase picks"
            )
            if self.classifier and (channel in ["BH", "HH"]):
                logger.info(
                    f"Put  {id.ljust(11)} {day.strftime('%Y.%j')}"
                    f" > {(str(len(classifies))).ljust(3)} classifier picks"
                )
            await asyncio.to_thread(
                self._write_single_picklist_to_db,
                picks,
                amplitudes,
                classifies,
                station,
                day,
                channel,
            )

    def _write_single_picklist_to_db(
        self,
        picks: sbu.PickList,
        amplitudes: list[float],
        classifies: list[tuple],
        station: str,
        day: datetime.datetime,
        channel: str,
    ) -> None:
        """
        Converts picks into records that can be submitted to MongoDB and writes them.
        Populates the `picks`, `classifies`, and `picks_record` collection.
        """
        if len(picks) > 0:
            self.db.insert_many_ignore_duplicates(
                "picks",
                [
                    {
                        "tid": station,
                        "cha": channel,
                        "start": pick.start_time.datetime,
                        "peak": pick.peak_time.datetime,
                        "end": pick.end_time.datetime,
                        "conf": float(pick.peak_value),
                        "amp": float(amp),
                        "pha": pick.phase,
                        "rid": self.run_id,
                    }
                    for pick, amp in zip(picks, amplitudes)
                ],
            )

        if len(classifies) > 0:
            self.db.insert_many_ignore_duplicates(
                "classifies",
                [
                    {
                        "tid": station,
                        "cha": channel,
                        "rid": self.run_id,
                        "start": c["start"].datetime,
                        "eq": float(c["eq"]),
                        "px": float(c["px"]),
                        "su": float(c["su"]),
                    }
                    for c in classifies
                ],
            )

        self.db.insert_many_ignore_duplicates(
            "picks_record",
            [
                {
                    "tid": station,
                    "cha": channel,
                    "yr": day.year,
                    "doy": int(day.strftime("%-j")),
                    "npks": len(picks),
                    "nclfs": len(classifies),
                    "rid": self.run_id,
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
    logger.info(f"Start job at {datetime.datetime.now()}")
    main()
    logger.info(f"Finish job at {datetime.datetime.now()}")
