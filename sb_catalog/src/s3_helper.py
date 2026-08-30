from __future__ import annotations

import asyncio
import datetime
import io
import logging
import os
import re
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Optional

import numpy as np
import obspy
from botocore.exceptions import ClientError
from obspy.clients.fdsn.header import FDSNNoDataException
from s3fs import S3FileSystem

from .constants import NETWORK_MAPPING, select_channel
from .profiling import stage
if TYPE_CHECKING:                     # pymongo is a 2025 DocumentDB
    from .utils import SeisBenchDatabase   # dependency; v3 writes Parquet
    # and never touches Mongo, so importing it eagerly made the path
    # helpers unusable anywhere pymongo is not installed.

# Default to empty rather than KeyError at import: parameters.py has always
# defaulted this to "" for campaigns that never touch EarthScope, and the
# module is imported by every picking job regardless of archive.
EARTHSCOPE_S3_ACCESS_POINT = os.environ.get("EARTHSCOPE_S3_ACCESS_POINT", "")

# EarthScope's sponsored open-data bucket serves a subset of networks with no
# credentials at all (docs.earthscope.org/sponsored-open-data). Pointing the
# access point at it and still exchanging credentials fails for anyone without
# the s3-miniseed role - which is everyone it is meant to help - so recognise
# it and read anonymously instead. Same region as the compute, so those reads
# are not cross-region either.
EARTHSCOPE_OPEN_DATA_BUCKET = "earthscope-geophysical-data"

# Networks served by the Open Data Program. Anonymous, global, no role.
# docs.earthscope.org/sdk/s3-direct-access-tutorial
EARTHSCOPE_OPEN_DATA_NETWORKS = frozenset(
    {"AK", "II", "IU", "N4", "PB", "TA", "UU", "UW"}
)

# Everything else lives behind a credentialed S3 access point. The alias is
# published in the tutorial above, so it is a default rather than a secret to
# be recovered from a previous campaign's notes; override it if EarthScope
# issues a different one.
EARTHSCOPE_RESTRICTED_ACCESS_POINT = os.environ.get(
    "EARTHSCOPE_S3_ACCESS_POINT",
    "earthscope-mseed-v2-4fdodyzpsz8u8uyi3pa9qsw9oid1suse2a-s3alias",
)

# The v1 role is retired: it answers "You are not allowed to assume role
# 's3-miniseed'" even for accounts in good standing, which reads like a
# permissions problem rather than a renamed role.
EARTHSCOPE_ROLE = os.environ.get("EARTHSCOPE_ROLE", "s3-miniseed-v2")
ES_CREDENTIAL_ATTEMPTS = int(os.environ.get("ES_CREDENTIAL_ATTEMPTS", "5"))

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
    """Two buckets, identical layout, different access.

    Open Data is preferred wherever it serves the network: it needs no
    credentials, so it cannot fail on an expired token or a role that was never
    granted, and it is not scoped to a region.
    """

    @staticmethod
    def is_open_data(net) -> bool:
        return net in EARTHSCOPE_OPEN_DATA_NETWORKS

    @classmethod
    def bucket_for(cls, net) -> str:
        return (EARTHSCOPE_OPEN_DATA_BUCKET if cls.is_open_data(net)
                else EARTHSCOPE_RESTRICTED_ACCESS_POINT)

    def get_prefix(self, net, year, day) -> str:
        return f"{self.bucket_for(net)}/miniseed/{net}/{year}/{day}/"

    def get_basename(self, net, sta, loc, cha, year, day, comp) -> str:
        # A regexp, matched against the listing. The restricted access point
        # appends a version ("ADO.CI.2019.187#2"); Open Data does not
        # ("ALCT.UW.2019.187"). Requiring the suffix silently matched nothing
        # on Open Data, so both forms have to be accepted.
        return rf"{re.escape(f'{sta}.{net}.{year}.{day}')}(#.*)?$"


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
        self.fs = {
            "scedc": S3FileSystem(anon=True),
            "ncedc": S3FileSystem(anon=True),
        }
        # EarthScope needs credentials; SCEDC and NCEDC are anonymous. Skip the
        # credential exchange entirely when the campaign has not been configured
        # for EarthScope, so a public-bucket run neither stalls nor fails on it.
        # Open Data needs nothing, so it is always available.
        self.fs["earthscope_open"] = S3FileSystem(anon=True)
        self.credential = None

        # The restricted access point is credentialed lazily: a campaign that
        # only touches Open Data networks should not fail, or even pause, on a
        # role it never uses.
        self.earthscope_restricted_ready = False
        logger.info(
            f"EarthScope: open data anonymous for "
            f"{','.join(sorted(EARTHSCOPE_OPEN_DATA_NETWORKS))}; "
            f"all other networks via {EARTHSCOPE_RESTRICTED_ACCESS_POINT} "
            f"(role {EARTHSCOPE_ROLE}, acquired on first use)"
        )

    def get_prefix(self, net, year, day) -> str:
        return self.helpers[self.get_data_center(net)].get_prefix(net, year, day)

    def get_basename(self, net, sta, loc, cha, year, day, c) -> str:
        return self.helpers[self.get_data_center(net)].get_basename(
            net, sta, loc, cha, year, day, c
        )

    def get_filesystem(self, net):
        dc = self.get_data_center(net)
        if dc == "earthscope":
            if EarthScopeS3ObjectHelper.is_open_data(net):
                return self.fs["earthscope_open"]
            if not self.earthscope_restricted_ready:
                self.credential = self.get_es_credential()
                self.set_es_filesystem()
            self.update_es_filesystem()
            return self.fs["earthscope_restricted"]
        return self.fs[dc]

    def get_es_credential(self):
        """
        Set 5 minutes buffer time to update credential
        """
        # Imported here, not at module scope: building an object key needs no
        # credentials machinery, and an eager import made every consumer of the
        # path helpers - including the dashboard job in CI - depend on the SDK.
        from earthscope_sdk import EarthScopeClient

        last = None
        for attempt in range(1, ES_CREDENTIAL_ATTEMPTS + 1):
            try:
                with EarthScopeClient() as client:
                    return client.user.get_aws_credentials(
                        role=EARTHSCOPE_ROLE, ttl_threshold=self.ttl_threshold
                    )
            except Exception as exc:
                last = exc
                logger.warning(
                    f"EarthScope credential client might be busy "
                    f"({attempt}/{ES_CREDENTIAL_ATTEMPTS}): {type(exc).__name__}. "
                    f"Sleeping 5 seconds."
                )
                time.sleep(5)
        # Fail rather than retry forever. An unbounded loop here never completes
        # and never errors, so a Spot worker holds its shard until the lease
        # expires and the next worker inherits the same hang.
        raise RuntimeError(
            f"Could not obtain EarthScope credentials after "
            f"{ES_CREDENTIAL_ATTEMPTS} attempts. Set ES_OAUTH2__REFRESH_TOKEN and "
            f"EARTHSCOPE_S3_ACCESS_POINT, or restrict the campaign to the public "
            f"SCEDC/NCEDC buckets."
        ) from last

    def set_es_filesystem(self):
        # Access-point requests are only valid in us-east-2; without the pin
        # s3fs may sign for another region and 400.
        self.fs["earthscope_restricted"] = S3FileSystem(
            key=self.credential.aws_access_key_id,
            secret=self.credential.aws_secret_access_key,
            token=self.credential.aws_session_token,
            client_kwargs={"region_name": "us-east-2"},
        )
        self.earthscope_restricted_ready = True

    def update_es_filesystem(self):
        # No-op when EarthScope was never configured: self.credential is None and
        # dereferencing it would AttributeError on the first refresh check.
        if self.credential is None:
            return                            # anonymous or not yet acquired
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
        limit_mb: Optional[int] = 200,
    ):
        self.start = start
        self.end = end
        self.components = components
        self.limit_mb = limit_mb
        if stations is None:
            self.stations = []
            self.networks = []
        else:
            self.stations = stations.split(",")
            self.networks = list(set([s.split(".")[0] for s in self.stations]))
        self.db = db
        self.s3helper = CompositeS3ObjectHelper()
        logger.info(f"Done preparing s3 access to {', '.join(self.s3helper.fs.keys())}")

        # Fail before reading anything if this shard needs an archive the job
        # cannot reach. Discovering it station by station means a campaign that
        # is 87% EarthScope fails 87% of its work one station at a time.
        self._check_archives_reachable()

        self.meta = self.db.get_station_metadata(
            self.stations, {"_id": 0, "id": 1, "channels": 1}
        ).set_index("id")
        logger.info(f"Done preparing metadata for the assigned stations")

        self.inventory = self._get_inventory()
        logger.info(f"Done preparing inventory for the assigned stations")

    def _check_archives_reachable(self) -> None:
        """Fail before reading anything if this shard needs an archive the job
        cannot reach.

        Only the restricted access point can fail this way; Open Data needs no
        credentials. Discovering it station by station means a campaign that is
        mostly restricted fails most of its work one station at a time, and the
        error it raised - KeyError, or a role denial buried in a retry loop -
        read like a defect in the reader.
        """
        restricted = sorted(
            n for n in self.networks
            if self.s3helper.get_data_center(n) == "earthscope"
            and not EarthScopeS3ObjectHelper.is_open_data(n)
        )
        if not restricted:
            return
        try:
            self.s3helper.get_filesystem(restricted[0])
        except Exception as exc:
            n_sta = sum(1 for s in self.stations
                        if s.split(".")[0] in set(restricted))
            raise RuntimeError(
                f"{n_sta} of {len(self.stations)} stations in this shard are on "
                f"EarthScope networks that are not in the Open Data Program "
                f"({len(restricted)}: {','.join(restricted[:6])}"
                f"{'...' if len(restricted) > 6 else ''}), so they need the "
                f"'{EARTHSCOPE_ROLE}' role on the {EARTHSCOPE_RESTRICTED_ACCESS_POINT} "
                f"access point, and it could not be obtained: "
                f"{type(exc).__name__}: {exc}. Run "
                f"scripts/check_earthscope_access.py, or plan this campaign over "
                f"Open Data networks "
                f"({','.join(sorted(EARTHSCOPE_OPEN_DATA_NETWORKS))}) plus "
                f"SCEDC/NCEDC only."
            ) from exc

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
                    # One LIST per day per network. A SCEDC day prefix holds
                    # ~4,000 objects, so this is not free and is paid again for
                    # every day in the shard.
                    with stage("s3.list"):
                        listing = fs.ls(prefix)
                    avail_uri[net] += listing
                except FileNotFoundError:
                    logger.debug(f"Path does not exist {prefix}")
                    pass
                except PermissionError as e:
                    logger.debug(e.args[0])
                    raise e

            for station in self.stations:
                # One channel code per station-location, chosen by the fixed
                # order in constants.CHANNEL_PRIORITY. Picking every band a
                # station carries duplicates the same ground motion at different
                # sampling rates: 2.83x the inference on SCEDC's permanent
                # stations, and it includes bands like LH at 1 Hz that cannot
                # produce a usable arrival at all. Location codes stay separate,
                # as in the 2025 study - they are genuinely different sensors.
                offered = self.meta.loc[station, "channels"].split(",")
                channel = select_channel(offered)
                if channel is None:
                    logger.info(
                        f"Skip {station.ljust(14)} {day.strftime('%Y.%j')} "
                        f"< no pickable channel among {','.join(offered)}"
                    )
                    continue
                all_channels = [channel]
                if len(offered) > 1:
                    logger.debug(
                        f"{station}: picking {channel} of {','.join(offered)}"
                    )
                check = {
                    cha: self.db.get_picks_record(
                        station, day, cha, {"_id": 1}
                    )  # return _id would be sufficient
                    for cha in all_channels
                }
                # if all channel got results
                if all(check.values()):
                    logger.info(
                        f"Skip {station.ljust(14)} {day.strftime('%Y.%j')} < picks found at {channel} channel"
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
                # Separately timed: this is a HEAD round trip before every GET,
                # which is pure latency and doubles the request count.
                with stage("s3.head"):
                    size = fs.info(uri)["size"]
                bytes_mb = size / 1024**2
                if self.limit_mb is not None and bytes_mb > self.limit_mb:
                    logger.warning(f"mSEED is too big (%.3f MB): %s" % (bytes_mb, uri))
                    return obspy.Stream()
                else:
                    with stage("s3.get", unit=size, unit_name="bytes"):
                        raw = fs.read_bytes(uri)
                    buff = io.BytesIO(raw)
                    with stage("mseed.parse", unit=size, unit_name="bytes"):
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
