import asyncio
import datetime
import io
import logging
import os
import re
import time
from abc import abstractmethod
from typing import AsyncIterator, Optional

import numpy as np
import obspy
from botocore.exceptions import ClientError
from earthscope_sdk import EarthScopeClient
from obspy.clients.fdsn.header import FDSNNoDataException
from s3fs import S3FileSystem

from .constants import NETWORK_MAPPING
from .utils import SeisBenchDatabase

EARTHSCOPE_S3_ACCESS_POINT = os.environ["EARTHSCOPE_S3_ACCESS_POINT"]

logger = logging.getLogger("picker")


class S3ObjectHelper:
    def get_data_center(self, net):
        return NETWORK_MAPPING[net]

    def get_s3_path(self, net, sta, loc, cha, year, day, comp) -> str:
        prefix = self.get_prefix(net, year, day)
        basename = self.get_basename(net, sta, loc, cha, year, day, comp)
        return f"{prefix}{basename}"

    @abstractmethod
    def get_prefix(self) -> str:
        pass

    @abstractmethod
    def get_basename(self) -> str:
        pass

    @abstractmethod
    def get_filesystem(self):
        pass


class SCEDCS3ObjectHelper(S3ObjectHelper):
    def get_prefix(self, net, year, day) -> str:
        return f"scedc-pds/continuous_waveforms/{year}/{year}_{day}/"

    def get_basename(self, net, sta, loc, cha, year, day, comp) -> str:
        return f"{net}{sta.ljust(5, '_')}{cha}{comp}{loc.ljust(3, '_')}{year}{day}.ms"


class NCEDCS3ObjectHelper(S3ObjectHelper):
    def get_prefix(self, net, year, day) -> str:
        return f"ncedc-pds/continuous_waveforms/{net}/{year}/{year}.{day}/"

    def get_basename(self, net, sta, loc, cha, year, day, comp) -> str:
        return f"{sta}.{net}.{cha}{comp}.{loc}.D.{year}.{day}"


class EarthScopeS3ObjectHelper(S3ObjectHelper):
    def get_prefix(self, net, year, day) -> str:
        return f"{EARTHSCOPE_S3_ACCESS_POINT}/miniseed/{net}/{year}/{day}/"

    def get_basename(self, net, sta, loc, cha, year, day, comp) -> str:
        return f"{sta}.{net}.{year}.{day}#."  # as regexp


class CompositeS3ObjectHelper(S3ObjectHelper):
    def __init__(self):
        self.helpers = {
            "scedc": SCEDCS3ObjectHelper(),
            "ncedc": NCEDCS3ObjectHelper(),
            "earthscope": EarthScopeS3ObjectHelper(),
        }

        self.s3 = {
            "scedc": "scedc-pds",
            "ncedc": "ncedc-pds",
            "earthscope": EARTHSCOPE_S3_ACCESS_POINT,
        }

        self.ttl_threshold = datetime.timedelta(minutes=5)
        self.credential = self.get_es_credential()
        self.fs = {
            "scedc": S3FileSystem(anon=True),
            "ncedc": S3FileSystem(anon=True),
        }
        self.set_es_filesystem()

    def get_prefix(self, net, year, day) -> str:
        return self.helpers[self.get_data_center(net)].get_prefix(net, year, day)

    def get_basename(self, net, sta, loc, cha, year, day, c) -> str:
        return self.helpers[self.get_data_center(net)].get_basename(
            net, sta, loc, cha, year, day, c
        )

    def get_filesystem(self, net):
        dc = self.get_data_center(net)
        if dc == "earthscope":
            self.update_es_filesystem()
        return self.fs[dc]

    def get_es_credential(self):
        """
        Set 5 minutes buffer time to update credential
        """
        while True:
            try:
                with EarthScopeClient() as client:
                    return client.user.get_aws_credentials(
                        ttl_threshold=self.ttl_threshold
                    )
            except:
                logger.warning(
                    f"EarthScope credential client might be busy. Sleep for 5 seconds and retry."
                )
                time.sleep(5)

    def set_es_filesystem(self):
        self.fs["earthscope"] = S3FileSystem(
            key=self.credential.aws_access_key_id,
            secret=self.credential.aws_secret_access_key,
            token=self.credential.aws_session_token,
        )

    def update_es_filesystem(self):
        if (
            self.credential.expiration - datetime.datetime.now(tz=datetime.timezone.utc)
        ) < self.ttl_threshold:
            # credential should be updated
            self.credential = self.get_es_credential()
            self.set_es_filesystem()
            logger.warning(f"EarthScope credential renewed.")
        else:
            pass

        return


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
        logger.info(f"Done preparing s3 access to {', '.join(self.s3helper.fs.keys())}")

        self.meta = self.db.get_station_metadata(
            self.stations, {"_id": 0, "id": 1, "channels": 1}
        ).set_index("id")
        logger.info(f"Done preparing metadata for the assigned stations")

        self.inventory = self._get_inventory()
        logger.info(f"Done preparing inventory for the assigned stations")

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
                all_channels = self.meta.loc[station, "channels"].split(",")
                check = {
                    cha: self.db.get_picks_record(
                        station, day, cha, {"_id": 1}
                    )  # return _id would be sufficient
                    for cha in all_channels
                }
                # if all channel got results
                if all(check.values()):
                    logger.info(
                        f"Skip {station.ljust(14)} {day.strftime('%Y.%j')} < picks found at all {self.meta.loc[station, 'channels']} channel"
                    )
                    continue

                net, sta, loc = station.split(".")
                dc = self.s3helper.get_data_center(net)
                logger.info(f"Load {station.ljust(14)} {day.strftime('%Y.%j')} @ {dc}")
                stream = obspy.Stream()

                if dc in ["scedc", "ncedc"]:
                    for channel in all_channels:
                        if check[channel]:
                            logger.debug(
                                f"Skip {station.ljust(14)} {day.strftime('%Y.%j')} < picks found at {channel} channel"
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
                                    f"Skip {station.ljust(14)} {day.strftime('%Y.%j')} < picks found at {channel} channel"
                                )
                                continue
                            stream += s.select(channel=f"{channel}?", location=loc)

                else:
                    raise NotImplemented(f"Data center not supported: {dc}")

                # the sampling rate should be at least 40 Hz (for amplitude extractor)
                for tr in stream:
                    if tr.stats.sampling_rate < 40.0:
                        stream.remove(tr)

                if len(stream) > 0:
                    # yield stream with all candidate channels for one station, day long stream, with metadata
                    yield [stream, station, day]
                else:
                    logger.info(
                        f"Skip {station.ljust(14)} {day.strftime('%Y.%j')} @ {dc}"
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
                bytes_mb = fs.info(uri)["size"] / 1024**2
                if bytes_mb > 100:  # skip stream bigger than 100 MB
                    logger.warning(
                        f"SEED too big ({bytes_mb} MB) and may cause OOM: {uri}"
                    )
                    return obspy.Stream()
                else:
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
            except (FileNotFoundError, ValueError, TypeError):
                return obspy.Stream()
            except:
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

    def _get_inventory(self):
        sta_code = ",".join([i.split(".")[1] for i in self.stations])
        net_code = ",".join(self.networks)
        cha_code = ",".join(
            set([f"{j}?" for i in self.meta.channels for j in i.split(",")])
        )

        while True:
            try:
                # Use IRIS web service for inventory request
                client = obspy.clients.fdsn.Client("IRIS")

                inv = client.get_stations(
                    network=net_code,
                    station=sta_code,
                    channel=cha_code,
                    level="response",
                    starttime=obspy.UTCDateTime(self.start),
                    endtime=obspy.UTCDateTime(self.end),
                )
                return inv
            except FDSNNoDataException:
                logger.warning(
                    f"No metadata at EarthScope FDSN service. Return empty inventory."
                )
                return obspy.Inventory()
            except:
                logger.warning(
                    f"EarthScope FDSN web service might be busy. Sleep for 5 seconds and retry."
                )
                time.sleep(5)
