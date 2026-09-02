from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import re
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Optional

import numpy as np
import obspy
from botocore.exceptions import ClientError
from obspy.clients.fdsn.header import (FDSNException,
                                       FDSNNoDataException)
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
# FDSN reserves codes beginning with a digit or X/Y/Z for temporary deployments,
# and reassigns them, so a credential for one is additionally scoped by year.
ES_TEMPORARY_NETWORK_PREFIXES = frozenset("0123456789XYZ")


class EarthScopeNetworkYearNotFound(FileNotFoundError):
    """EarthScope has no such network-year: HTTP 404 from the token exchange.

    Distinct from a wrong scope and from a missing entitlement. Temporary codes
    are reused, so the campaign plan legitimately contains network-years the
    archive never held - our station metadata says ZI has 2019 stations, and
    EarthScope answers `network FDSN:ZI year 2019 not found`.

    A FileNotFoundError because that is exactly what it is, and because the
    listing loop already treats one as "nothing recorded that day" and moves
    on. Anything stronger would abandon a shard over data that does not exist.
    """


FDSN_ATTEMPTS = int(os.environ.get("FDSN_ATTEMPTS", "8"))
# Denied GETs to tolerate before concluding the role lacks the entitlement.
ES_DENIED_ATTEMPTS = int(os.environ.get("ES_DENIED_ATTEMPTS", "2"))
# Retries to tolerate on a transient S3 error before giving the station-day up.
S3_BUSY_ATTEMPTS = int(os.environ.get("S3_BUSY_ATTEMPTS", "6"))
# Hard ceiling on one station-day. A worker that blocks past this loses one
# station-day; a worker with no ceiling loses the whole shard and its lease.
STATION_DAY_TIMEOUT = float(os.environ.get("STATION_DAY_TIMEOUT", "900"))

# Socket-level timeouts. Without these a stalled connection never raises: a
# profiling shard held a single GET open for 447 minutes and logged nothing
# until Spot reclaimed the task. botocore's defaults are generous and s3fs
# layers its own retries on top, so set them explicitly and keep them tight -
# a day-long mSEED object is tens of MB, not gigabytes.
S3_CONFIG = {
    "connect_timeout": 15,
    "read_timeout": 120,
    "retries": {"max_attempts": 4, "mode": "standard"},
}

logger = logging.getLogger("picker")

# earthscope_sdk logs credentials at DEBUG:
#
#     logger.debug(f"Refreshed tokens: {self._tokens}")     auth_flow.py
#     logger.debug(f"Refresh token revoked: {refresh_token}")
#
# `worker.py` configures logging with `logging.basicConfig(level=DEBUG)` when
# `--debug` is passed, and that sets the ROOT level, so those lines would be
# emitted - writing the refresh token AND the access token to CloudWatch, where
# anyone with logs:GetLogEvents on /aws/batch/job can read them.
#
# Pin the SDK's logger here rather than at each entry point: it is the library
# that leaks, so the floor belongs next to the library, and then no future
# caller can reintroduce it by configuring logging differently. DEBUG output
# from our own modules is unaffected.
logging.getLogger("earthscope_sdk").setLevel(logging.INFO)

# Every 2026 weight - jma_wc, original, obs - declares sampling_rate 100.
TARGET_SAMPLING_RATE = 100.0


def downsample_to_target(stream, target: float = TARGET_SAMPLING_RATE):
    """Bring anything recorded above `target` down to it, in place.

    Done at read time rather than left to SeisBench, for three reasons:

    * **Memory.** The decoded stream sits in `data_queue` until a picker loop
      takes it, so a 250 Hz DP trace occupies 2.5x what the model will ever use.
      That queue is what put `--procs 4` over 16 GB (OPTIMISE item 0d).
    * **Amplitude cost.** `annotate` resamples its own copy, but
      `amplitude_extractor` runs on the stream as read - so without this the
      Wood-Anderson and velocity stages process 2-2.5x more samples than the
      picks were made on.
    * **Comparability.** Amplitudes across the catalogue are then measured on
      uniformly 100 Hz data instead of whatever each instrument happened to run
      at.

    **Uses SeisBench's own resampler**, not a reimplementation, so the picks are
    unchanged: `annotate` sees traces already at its `sampling_rate` and skips
    resampling them. Matching its `zerophase=True` default matters - the
    docstring for `zerophase_resample` warns that a different filter in
    application than in training causes out-of-distribution issues.

    Only downsamples. A 40 Hz BH trace is left alone: upsampling it here would
    inflate the queue for no benefit, and `annotate` will do it anyway on its
    own copy, where the memory is short-lived.
    """
    from seisbench.models import WaveformModel

    fast = obspy.Stream([tr for tr in stream
                         if tr.stats.sampling_rate > target])
    if not len(fast):
        return stream
    WaveformModel.resample(fast, target, zerophase=True)
    return stream


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
            "scedc": S3FileSystem(anon=True, config_kwargs=S3_CONFIG),
            "ncedc": S3FileSystem(anon=True, config_kwargs=S3_CONFIG),
        }
        # EarthScope needs credentials; SCEDC and NCEDC are anonymous. Skip the
        # credential exchange entirely when the campaign has not been configured
        # for EarthScope, so a public-bucket run neither stalls nor fails on it.
        # Open Data needs nothing, so it is always available.
        self.fs["earthscope_open"] = S3FileSystem(anon=True, config_kwargs=S3_CONFIG)

        # Restricted EarthScope credentials are scoped per network (and per year
        # for temporary networks), so there is no single "the" credential: both
        # the credential and the filesystem built from it are cached by scope.
        # A shard is planned within one network, so in practice this holds one
        # entry - two when the shard straddles a new year.
        self.credentials = {}                 # scope key -> AwsTemporaryCredentials
        self.es_fs = {}                       # scope key -> S3FileSystem
        # Which scoping each network wants. Seeded from a guess, corrected by
        # the first denial - the token exchange returns 200 for either scoping,
        # so nothing short of a GET can tell them apart.
        self.es_scope_mode = {}               # net -> mode, below
        self.es_scope_tried = {}              # net -> {modes already denied}
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

    def get_filesystem(self, net, year=None):
        dc = self.get_data_center(net)
        if dc == "earthscope":
            if EarthScopeS3ObjectHelper.is_open_data(net):
                return self.fs["earthscope_open"]
            return self.get_es_filesystem(net, year)
        return self.fs[dc]

    @staticmethod
    def default_scope_mode(net) -> str:
        """The scoping to TRY FIRST for a network. A guess, not a rule.

        FDSN reserves codes beginning with a digit or X/Y/Z for temporary
        deployments and reassigns them, so those are expected to need a year as
        well as a network. That expectation comes from EarthScope's docs, not
        from a measurement, and `escalate_scope_mode` exists because it may be
        wrong in either direction - for a whole network, or for one year of one
        network. Guessing right merely avoids a wasted denial.
        """
        return ("network+year" if net[:1] in ES_TEMPORARY_NETWORK_PREFIXES
                else "network")

    def es_scope(self, net, year=None) -> dict:
        """Query parameters that scope a credential to what this shard reads.

        Unscoped credentials for `s3-miniseed-v2` carry `s3:ListBucket` but not
        `s3:GetObject`: every LIST succeeds and every GET returns AccessDenied.
        That asymmetry is why this looked for two weeks like a missing
        entitlement - listing works, so the role is obviously assumed, and the
        denial arrives only at the read.
        """
        mode = self.es_scope_mode.setdefault(net, self.default_scope_mode(net))
        scope = {"network": f"FDSN:{net}"}
        if mode == "network+year" and year is not None:
            scope["year"] = int(year)
        return scope

    def escalate_scope_mode(self, net) -> bool:
        """Switch a network to the other scoping and discard what was cached
        under the old one. Returns False once both have been tried.

        The token exchange itself cannot tell us which scoping a network wants:
        it returns 200 and a valid-looking credential either way, and only the
        GET distinguishes them. So the scoping is *learned from a denial* - and
        learned once per network per worker, not per object, since the flip is
        remembered in `es_scope_mode`.
        """
        tried = self.es_scope_tried.setdefault(net, set())
        current = self.es_scope_mode.setdefault(
            net, self.default_scope_mode(net))
        tried.add(current)
        other = "network" if current == "network+year" else "network+year"
        if other in tried:
            return False
        self.es_scope_mode[net] = other
        # Drop every credential and filesystem held for this network, whichever
        # year they were scoped to: they were all built under the old mode.
        stale = [k for k in self.credentials if f'"FDSN:{net}"' in k]
        for k in stale:
            self.credentials.pop(k, None)
            self.es_fs.pop(k, None)
        logger.warning(
            f"EarthScope: {net} denied under '{current}' scoping; retrying as "
            f"'{other}'. If this is consistent for {net}, the default in "
            f"`default_scope_mode` is wrong for it."
        )
        return True

    def get_es_filesystem(self, net, year=None):
        """Filesystem for one credential scope, built once and reused.

        Keyed by scope rather than fetched per object: a station-day shard is
        thousands of GETs behind one credential, and the token exchange is an
        OAuth round trip. Renewal is driven by expiry, not by call count.
        """
        scope = self.es_scope(net, year)
        key = json.dumps(scope, sort_keys=True)
        cred = self.credentials.get(key)
        if cred is None or (
            cred.expiration - datetime.datetime.now(tz=datetime.timezone.utc)
        ) < self.ttl_threshold:
            if cred is not None:
                logger.warning(f"EarthScope credential renewed for {scope}.")
            try:
                self.credentials[key] = self.get_es_credential(net, year)
            except RuntimeError as exc:
                # A wrong scope can fail at either end: refused here at the
                # exchange, or accepted here and denied at the GET. Escalate on
                # both, or the fallback only covers half the ways to be wrong.
                # Terminates because escalate_scope_mode returns False once
                # both scopings have been tried.
                #
                # Log the refusal BEFORE escalating. Escalation replaces one
                # error with another, and without this the surviving message is
                # the second attempt's - which explains why the fallback failed
                # and says nothing about why the fallback was needed.
                logger.warning(f"Scope {scope} refused: {exc}")
                if not self.escalate_scope_mode(net):
                    raise
                return self.get_es_filesystem(net, year)
            self.set_es_filesystem(key)
        return self.es_fs[key]

    def get_es_credential(self, net, year=None):
        """
        Set 5 minutes buffer time to update credential
        """
        # Imported here, not at module scope: building an object key needs no
        # credentials machinery, and an eager import made every consumer of the
        # path helpers - including the dashboard job in CI - depend on the SDK.
        from earthscope_sdk import EarthScopeClient

        scope = self.es_scope(net, year)
        last = None
        for attempt in range(1, ES_CREDENTIAL_ATTEMPTS + 1):
            try:
                with EarthScopeClient() as client:
                    return client.user.get_aws_credentials(
                        role=EARTHSCOPE_ROLE,
                        ttl_threshold=self.ttl_threshold,
                        **scope,
                    )
            except Exception as exc:
                last = exc
                # Report the HTTP status and body, not just the class.
                # "HTTPStatusError" alone is indistinguishable between "not
                # entitled", "no such network-year" and "malformed parameter",
                # and the retry loop then hides it five times over.
                resp = getattr(exc, "response", None)
                detail = f"{type(exc).__name__}"
                if resp is not None:
                    body = (resp.text or "").strip().replace("\n", " ")[:300]
                    detail = f"HTTP {resp.status_code} {body}"
                    # 4xx is a verdict, not congestion. Retrying a rejected
                    # scope just burns the budget - 25 s per network here -
                    # and buries the one line that says why.
                    code = resp.status_code
                    # 404 means this network-year is not in the archive at
                    # all. Not a scope error and not an entitlement error, so
                    # it must NOT trigger escalation: flipping the scope would
                    # ask a malformed question, get a 400, and then mark the
                    # network as having exhausted both scopings - poisoning
                    # every OTHER year of that network for the rest of the
                    # worker's life. ZI 2019 does not exist; ZI 2011 reads at
                    # 92 MB/s, and must keep doing so.
                    if code == 404:
                        raise EarthScopeNetworkYearNotFound(
                            f"EarthScope has no {scope}: {detail}"
                        ) from exc
                    if 400 <= code < 500 and code != 429:
                        raise RuntimeError(
                            f"EarthScope refused credentials for scope "
                            f"{scope} on role {EARTHSCOPE_ROLE}: {detail}. "
                            f"This is a "
                            f"verdict on the request, not congestion - check "
                            f"the scope before retrying."
                        ) from exc
                logger.warning(
                    f"EarthScope credential request failed for {scope} "
                    f"({attempt}/{ES_CREDENTIAL_ATTEMPTS}): {detail}. "
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

    @staticmethod
    def _secret(value) -> str:
        """SDK >= 1.4.1 returns the secret and session token as pydantic
        `SecretStr`. Passing one to s3fs signs the request with the literal
        string `**********`, which fails as a signature mismatch rather than as
        a type error - so unwrap explicitly."""
        return value.get_secret_value() if hasattr(value, "get_secret_value") else value

    def set_es_filesystem(self, key):
        credential = self.credentials[key]
        # Access-point requests are only valid in us-east-2; without the pin
        # s3fs may sign for another region and 400.
        self.es_fs[key] = S3FileSystem(
            key=credential.aws_access_key_id,
            secret=self._secret(credential.aws_secret_access_key),
            token=self._secret(credential.aws_session_token),
            client_kwargs={"region_name": "us-east-2"},
            config_kwargs=S3_CONFIG,
        )

    def update_es_filesystem(self, net, year=None, escalate=False):
        """Force a new credential after a denial, discarding the cached one.

        `escalate` switches to the other scoping first. A denial has two
        plausible causes and they need opposite responses: an expired token
        wants the same request again, a mis-scoped one wants a different
        request. The caller tries them in that order, cheapest first.
        """
        if self.get_data_center(net) != "earthscope" or \
                EarthScopeS3ObjectHelper.is_open_data(net):
            return                            # anonymous; nothing to renew
        if escalate and self.escalate_scope_mode(net):
            return self.get_es_filesystem(net, year)
        key = json.dumps(self.es_scope(net, year), sort_keys=True)
        self.credentials.pop(key, None)
        self.es_fs.pop(key, None)
        return self.get_es_filesystem(net, year)


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
            # The shard's own first year: a temporary network's credential is
            # year-scoped, so probing without one proves nothing about the days
            # this shard will actually read.
            self.s3helper.get_filesystem(restricted[0], self.start.year)
        except EarthScopeNetworkYearNotFound as exc:
            # The probe network has no data for this year. That is a fact about
            # the archive, not about our credentials - the shard's other
            # networks may well be readable - so it must not fail the shard.
            logger.info(
                f"EarthScope preflight: {exc}. Credentials are working; this "
                f"network-year simply is not in the archive. Continuing."
            )
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
                prefix = self.s3helper.get_prefix(
                    net, day.strftime("%Y"), day.strftime("%j")
                )
                try:
                    # INSIDE the try: acquiring the filesystem can itself raise
                    # FileNotFoundError, because a restricted EarthScope
                    # credential is refused with 404 when the archive holds no
                    # such network-year. Left outside, that escaped the handler
                    # and failed the whole shard - 16 of 48 in the 2026-09-02
                    # dry run, every one of them 5A/2018, which simply does not
                    # exist. A network-year with no data is a day with no data.
                    fs = self.s3helper.get_filesystem(net, day.year)
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
                                stream += await self._read_with_timeout(
                                    uri, net, station, day
                                )
                elif dc == "earthscope":
                    # use the first one: they should be all same
                    r = self._generate_waveform_uris(net, sta, loc, "NA", day)[0]
                    # earthscope object name has version number
                    uri = list(filter(lambda v: re.match(r, v), avail_uri[net]))
                    if len(uri) > 0:
                        # Ask libmseed for the bands we will keep, so the rest
                        # is never decoded. One EarthScope object holds every
                        # channel for the station-day - a UW sample had 214
                        # traces across 38 codes - and decoding all of them is
                        # what made --procs 4 exceed 16 GB (OPTIMISE item 0d).
                        wanted = [c for c in all_channels if not check[c]]
                        if not wanted:
                            continue
                        # obspy takes ONE pattern - a "a|b" form raises rather
                        # than matching both - so filter only in the single-band
                        # case, which is what CHANNEL_PRIORITY always yields.
                        # More than one band falls back to a full read, which is
                        # correct, just not lean.
                        sel = (f"{net}.{sta}.{loc}.{wanted[0]}?"
                               if len(wanted) == 1 else None)
                        s = await self._read_with_timeout(
                            uri[0], net, station, day, sourcename=sel
                        )
                        for channel in wanted:
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

    async def _read_with_timeout(self, uri, net, station, day,
                                 sourcename=None) -> obspy.Stream:
        """One object read, bounded by STATION_DAY_TIMEOUT.

        `wait_for` cannot kill the worker thread - it only stops waiting on it -
        so this is the second line of defence, not the first. The socket
        timeouts in S3_CONFIG are what actually bound the thread; this bounds
        the *shard*, so one pathological object costs a station-day rather than
        the whole shard and its lease. Both exist because the failure we hit
        (447 minutes on one GET, no log line, killed by Spot) got past having
        neither.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._read_waveform_from_s3, uri, net,
                                  sourcename, day.year),
                timeout=STATION_DAY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout after {STATION_DAY_TIMEOUT:.0f}s reading {uri} for "
                f"{station} {day.strftime('%Y.%j')}; abandoning this "
                f"station-day so the shard can continue."
            )
            return obspy.Stream()

    def _read_waveform_from_s3(self, uri, net, sourcename=None,
                               year=None) -> obspy.Stream:
        """
        Failure tolerant method for reading data from S3.

        `sourcename` is a libmseed record selector, e.g. "AK.PS09..HH?". Records
        that do not match are skipped **before** being decoded, which matters on
        EarthScope: it stores one multi-channel object per station-day, and the
        picker uses one band of it. Measured on AK.PS09 2020.309 (140 MB, 18
        traces): peak allocation falls from 540 MB to 348 MB and the parse is
        slightly faster, for a byte-identical selection.

        The saving is smaller than the trace count suggests, because the traces
        we keep are most of the samples - the 15 discarded on that station are
        1 Hz state-of-health channels. It is larger on a station carrying
        HH+BH+HN, where the discards are whole broadband sets.

        OSError#5: accessing non-authorized earthscope data. Return empty stream.
        PermissionError: EarthScope temporary credential expired. Refresh the credential and retry.
        ClientError: S3 overloaded, the job will sleep for 5 seconds and retry until return.
        FileNotFoundError: file not exist.
        ValueError: certain types of corrupt files.
        TypeError: certain types of empty mSEED files, i.e. in NCEDC

        """
        # A refresh fixes an *expired* credential. It cannot fix a credential
        # that was never entitled to the object, and retrying one forever is
        # indistinguishable from a hang: a profiling shard sat on a single
        # denied GET for 447 minutes, logging nothing, until Spot reclaimed it.
        # Bound the attempts and say what is actually wrong.
        denied = busy = 0
        while True:
            fs = self.s3helper.get_filesystem(net, year)
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
                        if sourcename:
                            st = obspy.read(buff, format="MSEED",
                                            sourcename=sourcename)
                        else:
                            st = obspy.read(buff)
                    with stage("resample"):
                        return downsample_to_target(st)
            except OSError as e:
                if e.errno == 5:
                    logger.warning(f"Not authorized to access this resource: {uri}")
                    return obspy.Stream()
            except PermissionError as e:
                denied += 1
                if denied > ES_DENIED_ATTEMPTS:
                    logger.error(
                        f"Access denied {denied} times for {uri} - refreshing "
                        f"the credential did not help, so this is not an "
                        f"expiry. Credential scope was "
                        f"{self.s3helper.es_scope(net, year)} on role "
                        f"{EARTHSCOPE_ROLE}. An unscoped credential lists but "
                        f"never reads, so first check that scope is reaching "
                        f"the token exchange; if it is, the account genuinely "
                        f"lacks {net}. Restrict the campaign to Open Data "
                        f"({','.join(sorted(EARTHSCOPE_OPEN_DATA_NETWORKS))}) "
                        f"plus SCEDC/NCEDC to proceed without it. Skipping."
                    )
                    return obspy.Stream()
                logger.debug(e.args[0])
                # First denial: assume an expired token and re-request the same
                # scope. Second: assume the scope itself is wrong and flip it.
                # Cheapest explanation first, and both are tried before the
                # station-day is abandoned.
                self.s3helper.update_es_filesystem(
                    net, year, escalate=denied > 1)
                logger.warning(
                    f"Credential refreshed after access denied "
                    f"({denied}/{ES_DENIED_ATTEMPTS}) for {uri}; scope now "
                    f"{self.s3helper.es_scope(net, year)}"
                )
            except ClientError:
                busy += 1
                if busy > S3_BUSY_ATTEMPTS:
                    logger.error(
                        f"S3 still failing after {busy} attempts for {uri}; "
                        f"giving up on this object rather than holding the "
                        f"shard open."
                    )
                    return obspy.Stream()
                logger.warning(
                    f"S3 might be busy ({busy}/{S3_BUSY_ATTEMPTS}). "
                    f"Sleep 5 seconds and retry."
                )
                time.sleep(5)
            except (FileNotFoundError, ValueError, TypeError):
                return obspy.Stream()
            except Exception:
                # `except Exception`, never a bare `except:`. A bare clause also
                # catches BaseException, which is what `worker.Preempted` is -
                # so a preemption landing on this read would have been turned
                # into an empty stream and the shard would have carried on
                # working after being told to stop.
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
        # Take the band code and add the component wildcard. The metadata may
        # hold either form: western_states.csv stores bands ("HH"), the 2025
        # per-network lists store full codes ("HHZ"). Appending "?" to a full
        # code yields "HHZ?", which FDSN rejects with a 400 - and the retry loop
        # below used to treat that permanent error as a busy server and spin on
        # it forever, so a whole campaign burned vCPU without picking anything.
        cha_code = ",".join(sorted(
            {f"{str(j).strip()[:2]}?" for i in self.meta.channels
             for j in str(i).split(",") if str(j).strip()}
        ))

        for attempt in range(1, FDSN_ATTEMPTS + 1):
            try:
                client = obspy.clients.fdsn.Client("EARTHSCOPE")
                return client.get_stations(
                    network=net_code,
                    station=sta_code,
                    channel=cha_code,
                    level="response",
                    starttime=obspy.UTCDateTime(self.start),
                    endtime=obspy.UTCDateTime(self.end),
                )
            except FDSNNoDataException:
                logger.warning(
                    "No metadata at the EarthScope FDSN service. "
                    "Returning an empty inventory."
                )
                return obspy.Inventory()
            except FDSNException as exc:
                # A 4xx will not become a 2xx by waiting. Retrying it is how a
                # bad request turns into an unbounded loop that looks like a
                # busy server.
                if "400" in str(exc) or "Bad request" in str(exc):
                    raise RuntimeError(
                        f"FDSN rejected the inventory request for this shard: "
                        f"{str(exc)[:200]}. channel={cha_code[:80]}"
                    ) from exc
                logger.warning(
                    f"FDSN error ({attempt}/{FDSN_ATTEMPTS}): "
                    f"{type(exc).__name__}. Sleeping 5 s."
                )
                time.sleep(5)
            except Exception as exc:
                logger.warning(
                    f"FDSN request failed ({attempt}/{FDSN_ATTEMPTS}): "
                    f"{type(exc).__name__}. Sleeping 5 s."
                )
                time.sleep(5)
        raise RuntimeError(
            f"Could not fetch station inventory after {FDSN_ATTEMPTS} attempts. "
            f"Without instrument response the amplitudes cannot be computed, so "
            f"the shard is failed rather than written without them."
        )
