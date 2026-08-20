import argparse
import asyncio
import datetime
import functools
import logging
import sys
import time
from typing import Any, Optional

import numpy as np
import obspy
import pyocto
import seisbench
import seisbench.models as sbm
import seisbench.util as sbu
from bson import ObjectId

from .amplitude_extractor import AmplitudeExtractor
from .classifier import QuakeXNet
from .parquet_writer import ParquetPickWriter
from .profiling import stage
from .s3_helper import S3DataSource
from .utils import SeisBenchDatabase, parse_year_day

logger = logging.getLogger("picker")


def main() -> None:
    """
    This main function serves as the entry point to all functionality available in the script.
    """
    # `work` delegates to the v3 queue worker. The container's ENTRYPOINT is
    # fixed at `python -m src.picker`, and AWS Batch's containerProperties has no
    # entryPoint field, so on Fargate this subcommand is the only way to reach
    # the worker without publishing a second image. Dispatched before argparse
    # because the worker takes a completely different set of options - notably
    # no --db_uri, since v3 has no database.
    if len(sys.argv) > 1 and sys.argv[1] == "work":
        from .worker import main as worker_main

        return worker_main(sys.argv[2:])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        type=str,
        help="Subroutine to execute. See below for available functions. "
        "Use `work` to run the v3 S3-queue worker instead of a single job.",
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
        "--parquet_uri",
        type=str,
        default=None,
        help="S3 prefix for Parquet output, e.g. s3://bucket/quakescope2026. "
        "When set, picks and classifications are written there instead of into "
        "the picks/classifies collections; station metadata and picks_record "
        "still use the database.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable additional debug output."
    )
    args = parser.parse_args()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
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
        parquet_uri=args.parquet_uri,
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
        classifier: Optional[bool] = False,
        parquet_uri: Optional[str] = None,
        job_id: Optional[str] = None,
        checkpoint_every: int = 0,
        on_checkpoint: Optional[Any] = None,
    ):
        self.extent = extent
        # When set, bulk picks go to Parquet on S3 instead of into the
        # database. Station metadata and picks_record stay in the database:
        # they are small and want point lookups. See
        # docs/rerun_2026/12_output_storage.md.
        # Batch always supplies the parameter, so an empty string is how "no
        # Parquet output" arrives from a job definition.
        # `or None` matters: Batch always supplies the parameter, so "no Parquet
        # output" arrives as an empty string, and the writer would otherwise be
        # constructed with an empty root.
        self.parquet_uri = parquet_uri or None
        # Names the Parquet object. Without it ParquetPickWriter falls back to
        # HOSTNAME, and every shard a node runs writes the SAME key inside a
        # (network, year, month) partition - silently overwriting the last.
        self.job_id = job_id
        # Flush and record progress every N station-day-channels, so a Spot
        # preemption costs at most that much rather than the whole shard.
        # 0 disables it, which is the right default for the database path where
        # every station-day is committed as it completes anyway.
        self.checkpoint_every = int(checkpoint_every or 0)
        self.on_checkpoint = on_checkpoint
        self._next_checkpoint = self.checkpoint_every
        self._parquet = None

        # model preparation
        if model is not None:
            self.model = self.create_model(model, weight, p_threshold, s_threshold)
        else:
            self.model = None

        if classifier:
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

    @property
    def parquet(self) -> Optional[ParquetPickWriter]:
        """Parquet writer for this job, created once the run_id exists."""
        if self.parquet_uri is None:
            return None
        if self._parquet is None:
            self._parquet = ParquetPickWriter(
                root=self.parquet_uri, run_id=str(self.run_id), job_id=self.job_id
            )
        return self._parquet

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

        if self._parquet is not None:
            # Written at the end because the file boundary is the job, not the
            # station-day; see parquet_writer for why that matters.
            await asyncio.to_thread(self._finalise_parquet)

    def _finalise_parquet(self) -> None:
        """Flush the job's Parquet, then record the station-days it covered.

        The order is the point. picks_record is what lets a re-submission skip
        completed work, so it must never be written for a station-day whose
        picks are still only in memory. Flushing first means an interrupted job
        leaves no records and its whole retry re-does the work, which is exactly
        what a job-sized file boundary implies.
        """
        summary = self._parquet.close()
        records = summary.get("records", [])
        if records:
            self.db.insert_many_ignore_duplicates("picks_record", records)
        logger.info(
            f"Committed {len(records)} picks_record entries after the Parquet flush"
        )

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
                            f"Skip {id.ljust(14)} {day.strftime('%Y.%j')} < too many gaps"
                        )
                        stream_c = obspy.Stream()
                    else:
                        logger.debug(f"Send {id.ljust(14)} {day.strftime('%Y.%j')}")

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
            logger.debug(f"Pick {id.ljust(14)} {day.strftime('%Y.%j')}")
            if len(stream) == 0:
                logger.info(
                    f"Skip {station.ljust(14)} {day.strftime('%Y.%j')} < stream is empty due to exception"
                )
                await picks_queue.put(
                    [sbu.PickList(), [], [], [], station, day, channel]
                )
            else:
                # do picking
                def _classify():
                    with stage("model.classify"):
                        return self.model.classify(stream)

                stream_annotations = await asyncio.to_thread(_classify)
                n_picks = len(stream_annotations.picks)

                # extract amplitudes. Timed against PICK COUNT, not station-days:
                # this pass runs once per pick and pick counts vary ~4x between an
                # ordinary day and a mainshock day, so per-station-day timing hides
                # how it actually scales.
                def _amps():
                    with stage("amp.wood_anderson", unit=n_picks, unit_name="pick"):
                        return self.amp_extor.extract_amplitudes(
                            stream, stream_annotations.picks, self.s3.inventory
                        )

                stream_amplitudes = await asyncio.to_thread(_amps)

                # extract raw amplitudes around each pick
                def _raw_amps():
                    with stage("amp.raw", unit=n_picks, unit_name="pick"):
                        return self.amp_extor.extract_raw_amplitudes(
                            stream, stream_annotations.picks
                        )

                stream_raw_amplitudes = await asyncio.to_thread(_raw_amps)

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
                        stream_raw_amplitudes,
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
                raw_amplitudes,
                classifies,
                station,
                day,
                channel,
            ) = _pk_amp_clf_sta_day_cha

            id = f"{station}.{channel}"
            logger.info(
                f"Put  {id.ljust(14)} {day.strftime('%Y.%j')}"
                f" > {(str(len(picks))).ljust(3)} phase picks"
            )
            if self.classifier and (channel in ["BH", "HH"]):
                logger.info(
                    f"Put  {id.ljust(14)} {day.strftime('%Y.%j')}"
                    f" > {(str(len(classifies))).ljust(3)} classifier picks"
                )
            await asyncio.to_thread(
                self._write_single_picklist_to_db,
                picks,
                amplitudes,
                raw_amplitudes,
                classifies,
                station,
                day,
                channel,
            )

    def _write_single_picklist_to_db(
        self,
        picks: sbu.PickList,
        amplitudes: list[float],
        raw_amplitudes: list[float],
        classifies: list[tuple],
        station: str,
        day: datetime.datetime,
        channel: str,
    ) -> None:
        """
        Converts picks into records that can be submitted to MongoDB and writes them.
        Populates the `picks`, `classifies`, and `picks_record` collection.
        """
        writer = self.parquet
        if writer is not None:
            # Bulk output goes to Parquet. picks_record below still goes to the
            # database, because the resume logic needs point lookups and the
            # record table is ~2.6 GB against ~3 TB of picks.
            writer.add(
                picks, amplitudes, raw_amplitudes, classifies, station, day, channel
            )

        if writer is None and len(picks) > 0:
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
                        "amp_raw": float(raw_amp),
                        "pha": pick.phase,
                        "rid": self.run_id,
                    }
                    for pick, amp, raw_amp in zip(picks, amplitudes, raw_amplitudes)
                ],
            )

        if writer is None and len(classifies) > 0:
            self.db.insert_many_ignore_duplicates(
                "classifies",
                [
                    {
                        "tid": station,
                        "cha": channel,
                        "start": c["start"].datetime,
                        "eq": float(c["eq"]),
                        "px": float(c["px"]),
                        "su": float(c["su"]),
                        "rid": self.run_id,
                    }
                    for c in classifies
                ],
            )

        if writer is not None:
            # In Parquet mode the picks are buffered, so committing picks_record
            # here would let a Spot interruption strand station-days that are
            # marked done but were never written. Records are committed only
            # after a flush succeeds - either at a checkpoint below, or in
            # _finalise_parquet at the end of the job.
            #
            # Checkpointing exists so a preemption does not discard the whole
            # shard: at 40 stations x 20 days a shard runs about twelve hours,
            # and without this every one of those hours is lost. Flush first,
            # then report, never the reverse.
            if (
                self.checkpoint_every
                and self.on_checkpoint is not None
                and writer.pending_records >= self._next_checkpoint
            ):
                records = writer.checkpoint()
                self.on_checkpoint(records)
                self._next_checkpoint = (
                    writer.pending_records + self.checkpoint_every
                )
                logger.info(
                    f"Checkpointed {len(records)} station-day-channels; a "
                    f"preemption now costs at most {self.checkpoint_every}"
                )
            return

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
