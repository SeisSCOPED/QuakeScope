from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import random
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

# GeoNet (New Zealand) on the AWS Open Data Program: public, anonymous, and
# in ap-southeast-2 rather than us-east-2 like everything else we read.
GEONET_BUCKET = os.environ.get("GEONET_BUCKET", "geonet-open-data")
GEONET_REGION = os.environ.get("GEONET_REGION", "ap-southeast-2")

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
# Ceiling on how often ONE scope may be re-exchanged inside
# ES_REFRESH_WINDOW seconds. A credential is good for an hour, so anything
# above this is our retry logic churning, not a real expiry - and churn on a
# shared endpoint is indistinguishable from an attack. Exceeding it raises
# locally, costing EarthScope nothing.
ES_REFRESH_BUDGET = int(os.environ.get("ES_REFRESH_BUDGET", "3"))
ES_REFRESH_WINDOW = float(os.environ.get("ES_REFRESH_WINDOW", "300"))
# FDSN reserves codes beginning with a digit or X/Y/Z for temporary deployments,
# and reassigns them, so a credential for one is additionally scoped by year.
ES_TEMPORARY_NETWORK_PREFIXES = frozenset("0123456789XYZ")


class EarthScopeNoAccess(RuntimeError):
    """HTTP 403: the account may not read this network at all.

    Different from `EarthScopeNetworkYearNotFound` (404, the archive has no such
    network-year) and from a wrong scope (400). This one is a real access
    gap and should be loud - it means shards on that network cannot run, and
    somebody has to ask EarthScope.

    Found by the 2026-09-03 sweep, which no dry run had reached: network `LH`
    answers 403 for every year.
    """


class EarthScopeNetworkYearNotFound(FileNotFoundError):
    """EarthScope has no such network-year: HTTP 404 from the token exchange.

    Distinct from a wrong scope and from a missing grant of access. Temporary codes
    are reused, so the campaign plan legitimately contains network-years the
    archive never held - our station metadata says ZI has 2019 stations, and
    EarthScope answers `network FDSN:ZI year 2019 not found`.

    A FileNotFoundError because that is exactly what it is, and because the
    listing loop already treats one as "nothing recorded that day" and moves
    on. Anything stronger would abandon a shard over data that does not exist.
    """


class EarthScopeScopeRefused(RuntimeError):
    """HTTP 400: the request itself is malformed for this network.

    The case that matters is a temporary network asked for without a year.
    FDSN reuses temporary codes across experiments, so EarthScope authorises
    them at the year level and a year-less request for one can never succeed.
    Retrying it is pure load, which is how this fleet came to be reported as a
    denial of service on 2026-09-04.
    """


class EarthScopeScopeIncomplete(EarthScopeScopeRefused):
    """A 400 we recognised without sending it.

    Raised when a network is on `network+year` scoping and the caller has no
    year to give. EarthScope would answer 400; we answer it ourselves.
    """


class EarthScopeRequestRefused(RuntimeError):
    """Any other 4xx from the credentials endpoint.

    Terminal for the same reason 400/403/404 are: a 4xx is a verdict on the
    request. Without its own type it was raised as a bare RuntimeError, which
    `ES_TERMINAL` does not cover - so an uncommon status (405, 409, 410, 422)
    would have gone un-remembered and been re-asked on every one of the 11,315
    calls a shard makes. Exactly the bug this module exists to fix, surviving
    in the codes nobody had seen yet.
    """


class EarthScopeAuthFailed(RuntimeError):
    """HTTP 401: the token itself was rejected.

    Not scoped to a network - every request this worker makes will fail the
    same way - so it is remembered process-wide and the shard fails fast
    rather than re-presenting a bad token once per network-day.
    """


class EarthScopeExchangeThrottled(RuntimeError):
    """Refused locally by our own rate limit, never sent.

    Deliberately NOT one of the terminal verdicts: the window reopens, so a
    genuine renewal an hour later still goes through.
    """


# Verdicts on the request, not congestion. None of these is ever retried, and
# each is remembered for the life of the process so the same question is never
# asked twice. EarthScope's 2026-09-04 report - "400/403/404s should not be
# retried" - is enforced here and nowhere else.
ES_TERMINAL = (
    EarthScopeNoAccess,           # 403 - no access to this network/year
    EarthScopeNetworkYearNotFound,   # 404 - no such FDSN code
    EarthScopeScopeRefused,          # 400 - malformed, incl. missing year
    EarthScopeAuthFailed,            # 401 - bad token
    EarthScopeRequestRefused,        # any other 4xx
)


# Process-wide, not per helper. A worker claims shard after shard in one
# process and builds a fresh CompositeS3ObjectHelper for each, so per-instance
# state made every shard re-learn the same verdicts from EarthScope and
# re-bootstrap OAuth. These are facts about the account and the archive, not
# about a shard, so they outlive one.
_ES_STATE = {
    "refused": {},        # scope key -> terminal exception, never re-asked
    "auth_failed": None,  # 401: not scoped to a network, so it stops everything
    "exchanges": {},      # scope key -> [monotonic stamps], for the rate limit
    "denials": {},        # scope key -> consecutive AccessDenied count
    "client": None,       # the one earthscope_sdk client this process uses
    # Which scoping each network turned out to want. Lives here rather than on
    # the helper because `refused` does: the scope key is BUILT from the mode,
    # so a per-helper mode files a verdict under one key and looks it up under
    # another on the next shard. See CompositeS3ObjectHelper.__init__.
    "scope_mode": {},     # net -> "network" | "network+year"
    "scope_tried": {},    # net -> {modes already refused}
}


def reset_earthscope_state() -> None:
    """Forget every cached verdict and drop the SDK client.

    For tests and for a long-lived process whose access has genuinely changed;
    a campaign worker should never need it.

    Clears each mapping IN PLACE rather than rebinding it. Helpers hold the
    dicts directly (`self.es_refused`, `self.es_scope_mode`), so replacing them
    here would leave any helper built before the reset writing into an orphaned
    copy - the same lifetime trap that made a per-helper scope mode disagree
    with a process-wide verdict store.
    """
    client = _ES_STATE["client"]
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    for key in ("refused", "exchanges", "denials", "scope_mode",
                "scope_tried"):
        _ES_STATE[key].clear()
    _ES_STATE["auth_failed"] = None
    _ES_STATE["client"] = None


FDSN_ATTEMPTS = int(os.environ.get("FDSN_ATTEMPTS", "8"))
# Denied GETs to tolerate before concluding the role lacks the access.
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

    # Zero-length traces reach here from truncated or empty records and blow up
    # later inside the model as "cannot reshape array of size 0 into shape (0)",
    # which names nothing useful. Drop them at the boundary instead.
    empty = [tr for tr in stream if tr.stats.npts == 0]
    for tr in empty:
        stream.remove(tr)

    fast = obspy.Stream([tr for tr in stream
                         if tr.stats.sampling_rate > target])
    if not len(fast):
        return stream
    WaveformModel.resample(fast, target, zerophase=True)

    # Resampling returns float64. Traces ALREADY at or below the target are not
    # touched and stay int32, so a station whose sampling rate changes during
    # the day ends up with both in one stream - and obspy refuses to merge
    # across dtypes:
    #
    #   TypeError: Data type differs: int32 vs float64
    #
    # raised deep inside SeisBench's annotate, where it reads as a model
    # problem. It failed 693 western shards on 2026-09-04, all of them in the
    # tail of the campaign because those are the stations that had a rate
    # change to hit.
    #
    # Promote the whole stream rather than demoting the resampled traces:
    # float64 is what the model will use anyway, and rounding samples back to
    # int to satisfy a merge would be losing data to tidy a type.
    dtypes = {tr.data.dtype for tr in stream}
    if len(dtypes) > 1:
        for tr in stream:
            if tr.data.dtype != np.float64:
                tr.data = tr.data.astype(np.float64)
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

    def list_day(self, fs, prefix) -> list:
        """Objects under one day prefix.

        A hook, because the archives are not shaped alike. SCEDC, NCEDC and
        EarthScope put every object for a day directly under the day prefix, so
        `ls` - one level - sees them all. GeoNet adds a station directory
        underneath, where `ls` would return directories and no objects at all,
        and the day would silently look empty.
        """
        return fs.ls(prefix)


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


class GeoNetS3ObjectHelper(S3ObjectHelper):
    """GeoNet (New Zealand), AWS Open Data, anonymous.

        waveforms/miniseed/2019/2019.001/ABAZ.NZ/2019.001.ABAZ.10-EHE.NZ.D
                           YEAR YEAR.DOY STA.NET  YEAR.DOY.STA.LOC-CHACOMP.NET.D

    One object per channel-component per station-day, like SCEDC - so a station
    fetches only the band it needs, unlike EarthScope's single multi-channel
    object. The extra STA.NET directory is the one structural difference, and
    it is why `list_day` exists.

    Location codes are always two digits here (10, 20, 21, 11), never blank,
    and accelerometer components are 1/2/Z rather than E/N/Z - both already
    covered by the default `--components ZNE12`.

    **The bucket is in ap-southeast-2 and the fleet is in us-east-2.** Reads are
    cross-region, which is a throughput question rather than a cost one - the
    Open Data Program sponsors egress - but it has to be measured from a task
    before this campaign is sized. See docs/rerun_2026.
    """

    def get_prefix(self, net, year, day) -> str:
        return f"{GEONET_BUCKET}/waveforms/miniseed/{year}/{year}.{day}/"

    def get_basename(self, net, sta, loc, cha, year, day, comp) -> str:
        return f"{year}.{day}.{sta}.{loc}-{cha}{comp}.{net}.D"

    def get_s3_path(self, net, sta, loc, cha, year, day, comp) -> str:
        # NOT prefix + basename: the station directory sits between them.
        prefix = self.get_prefix(net, year, day)
        base = self.get_basename(net, sta, loc, cha, year, day, comp)
        return f"{prefix}{sta}.{net}/{base}"

    def list_day(self, fs, prefix) -> list:
        # Recursive: the objects are one level below the day prefix.
        return fs.find(prefix)


class CompositeS3ObjectHelper(S3ObjectHelper):
    def __init__(self):
        self.helpers = {
            "scedc": SCEDCS3ObjectHelper(),
            "ncedc": NCEDCS3ObjectHelper(),
            "earthscope": EarthScopeS3ObjectHelper(),
            "geonet": GeoNetS3ObjectHelper(),
        }

        self.s3 = {
            "scedc": "scedc-pds",
            "ncedc": "ncedc-pds",
            "earthscope": EARTHSCOPE_S3_ACCESS_POINT,
            "geonet": GEONET_BUCKET,
        }

        self.ttl_threshold = datetime.timedelta(minutes=5)
        self.fs = {
            "scedc": S3FileSystem(anon=True, config_kwargs=S3_CONFIG),
            "ncedc": S3FileSystem(anon=True, config_kwargs=S3_CONFIG),
            # Pinned to its own region: without client_kwargs s3fs signs for
            # us-east-2 and the request 301s to a redirect it does not follow.
            "geonet": S3FileSystem(anon=True, config_kwargs=S3_CONFIG,
                                   client_kwargs={"region_name": GEONET_REGION}),
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
        #
        # PROCESS-WIDE, and it has to be, because `es_refused` below is.
        # These two are read together: `es_scope_key` builds its key from the
        # mode, so a mode that resets per shard files a verdict under one key
        # and looks it up under another on the next shard. EarthScope's
        # developers found what that costs and reported it on 2026-09-04.
        # Shard 1 asks NP year-less, gets a 400, files it under the year-less
        # key, escalates, and succeeds with the year. Shard 2 builds a fresh
        # helper, defaults back to year-less, rebuilds that same key, finds
        # shard 1's 400 and raises it - without ever asking for the year. The
        # refusal outlived the helper; the escalation that made it survivable
        # did not, and the network went unreadable for the rest of the
        # process. Whatever lifetime the verdicts have, the mode must match.
        self.es_scope_mode = _ES_STATE["scope_mode"]   # net -> mode, below
        self.es_scope_tried = _ES_STATE["scope_tried"]  # net -> modes denied
        # Answers EarthScope has already given. 400/403/404/401 are verdicts on
        # the request, so the same request is never sent twice: the entry is
        # the exception itself, re-raised without touching the network.
        #
        # This is the fix for the 2026-09-04 report. `load_waveforms` asks for a
        # filesystem once per day per network and `_read_waveform_from_s3` once
        # per object, and every one of those calls used to re-run the exchange
        # after a refusal - a year-long shard on a network we cannot read
        # sent ~366 credential requests, multiplied by every worker in the
        # fleet, forever.
        self.es_refused = _ES_STATE["refused"]   # scope key -> terminal exc
        # The SDK caches issued credentials in memory on the service object
        # (`_aws_creds_by_key`) and, since 1.8.0, nowhere else - so building a
        # fresh EarthScopeClient per call threw that cache away every time and
        # guaranteed a round trip, plus an OAuth token refresh whenever the
        # access token had not been persisted. One client per process keeps the
        # SDK's own cache, its access token and its connection pool alive; see
        # `_ES_STATE`.
        logger.info(
            f"EarthScope: open data anonymous for "
            f"{','.join(sorted(EARTHSCOPE_OPEN_DATA_NETWORKS))}; "
            f"all other networks via {EARTHSCOPE_RESTRICTED_ACCESS_POINT} "
            f"(role {EARTHSCOPE_ROLE}, acquired on first use)"
        )

    @property
    def es_auth_failed(self):
        return _ES_STATE["auth_failed"]

    @es_auth_failed.setter
    def es_auth_failed(self, exc):
        _ES_STATE["auth_failed"] = exc

    @property
    def es_exchanges(self):
        return _ES_STATE["exchanges"]

    @property
    def es_denials(self):
        return _ES_STATE["denials"]

    def es_client(self):
        """The one EarthScope client this process uses.

        Built lazily - constructing it bootstraps OAuth, which a campaign on
        the public buckets must never pay for - and then kept, so the SDK can
        serve a repeat request for the same scope from its own memory cache
        instead of asking EarthScope again.
        """
        if _ES_STATE["client"] is None:
            # Imported here, not at module scope: building an object key needs
            # no credentials machinery, and an eager import made every consumer
            # of the path helpers - including the dashboard job in CI - depend
            # on the SDK.
            from earthscope_sdk import EarthScopeClient
            _ES_STATE["client"] = EarthScopeClient()
        return _ES_STATE["client"]

    def get_prefix(self, net, year, day) -> str:
        return self.helpers[self.get_data_center(net)].get_prefix(net, year, day)

    def get_basename(self, net, sta, loc, cha, year, day, c) -> str:
        return self.helpers[self.get_data_center(net)].get_basename(
            net, sta, loc, cha, year, day, c
        )

    def get_s3_path(self, net, sta, loc, cha, year, day, c) -> str:
        # Delegated, not inherited: GeoNet composes prefix + STA.NET/ + basename
        # rather than prefix + basename, so the base implementation is wrong
        # for it.
        return self.helpers[self.get_data_center(net)].get_s3_path(
            net, sta, loc, cha, year, day, c
        )

    def list_day(self, net, fs, prefix) -> list:
        return self.helpers[self.get_data_center(net)].list_day(fs, prefix)

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
        access - listing works, so the role is obviously assumed, and the
        denial arrives only at the read.
        """
        mode = self.es_scope_mode.setdefault(net, self.default_scope_mode(net))
        scope = {"network": f"FDSN:{net}"}
        if mode == "network+year" and year is not None:
            scope["year"] = int(year)
        return scope

    def es_scope_key(self, net, year=None) -> str:
        return json.dumps(self.es_scope(net, year), sort_keys=True)

    @staticmethod
    def _verdict_record(exc):
        """What we keep: the class and its message, never the instance.

        A caught exception holds `__traceback__`, `__context__` and
        `__cause__`, and through the traceback every frame it passed - which
        here includes the helper, the shard and whatever those reference. This
        store is process-wide and lives as long as the worker, so keeping
        instances would pin that graph for the life of the process.

        Re-raising one instance is worse than the memory. `raise verdict`
        happens inside an `except` block, and Python assigns `__context__` on
        every raise, so the single cached object would accumulate a chain
        thousands deep - and eventually a cycle, when a later verdict is raised
        while an earlier one is being handled.
        """
        return (type(exc), exc.args)

    @staticmethod
    def _verdict_raise(record):
        """A clean instance, built fresh for each raise."""
        cls, args = record
        exc = cls(*args)
        exc.__suppress_context__ = True
        return exc

    def es_verdict(self, net, year=None):
        """The terminal answer already held for this scope, or None.

        Checked before every exchange. A 401 is not scoped to a network - the
        token is bad for everything - so it short-circuits all of them.
        """
        if self.es_auth_failed is not None:
            return self._verdict_raise(self.es_auth_failed)
        record = self.es_refused.get(self.es_scope_key(net, year))
        return self._verdict_raise(record) if record is not None else None

    def es_record_verdict(self, net, year, exc) -> None:
        """Remember a refusal so the same request is never sent again."""
        if isinstance(exc, EarthScopeAuthFailed):
            self.es_auth_failed = self._verdict_record(exc)
            return
        key = self.es_scope_key(net, year)
        if key not in self.es_refused:
            logger.warning(
                f"EarthScope refused {key} ({type(exc).__name__}). Not "
                f"retrying: this is a verdict on the request. Every later "
                f"read of that scope is answered from memory."
            )
        self.es_refused[key] = self._verdict_record(exc)

    def escalate_scope_mode(self, net) -> bool:
        """Add the year to a network's scope after a 400. One direction only.

        `default_scope_mode` is a guess and it can be wrong: a code we read as
        permanent may in fact be authorised per year. Discovering that costs
        one extra request, and the corrected mode is remembered per network, so
        it is paid once per PROCESS rather than per object - and, since
        2026-09-04, per process rather than per shard. The mode lives in
        `_ES_STATE` alongside the verdicts it shares a key space with; a
        per-helper mode made the next shard re-ask the year-less question,
        find the 400 this call had just filed, and give up on the network.

        The reverse direction is gone. Dropping the year from a temporary
        network's request produces exactly what EarthScope reported on
        2026-09-04 - "requests for a temporary network without a year - this
        will never succeed" - because temporary FDSN codes are reused across
        experiments and are therefore authorised at the year level. It was
        reachable from a 403, which is the one status that guarantees the
        retry is pointless, and every object of a denied station-day walked
        into it. So: escalation never removes scope, only adds it, and it is
        driven only by a 400.
        """
        current = self.es_scope_mode.setdefault(
            net, self.default_scope_mode(net))
        tried = self.es_scope_tried.setdefault(net, set())
        tried.add(current)
        if current != "network":
            return False                      # nothing safe left to ask
        if "network+year" in tried:
            return False
        self.es_scope_mode[net] = "network+year"
        # Drop every credential and filesystem held for this network: they were
        # all built under the old mode.
        stale = [k for k in self.credentials if f'"FDSN:{net}"' in k]
        for k in stale:
            self.credentials.pop(k, None)
            self.es_fs.pop(k, None)
        logger.warning(
            f"EarthScope: {net} refused a year-less scope with 400; retrying "
            f"once as 'network+year'. If this is consistent for {net}, the "
            f"default in `default_scope_mode` is wrong for it."
        )
        return True

    def get_es_filesystem(self, net, year=None):
        """Filesystem for one credential scope, built once and reused.

        Keyed by scope rather than fetched per object: a station-day shard is
        thousands of GETs behind one credential, and the token exchange is an
        OAuth round trip. Renewal is driven by expiry, not by call count.
        """
        verdict = self.es_verdict(net, year)
        if verdict is not None:
            # Already answered. `load_waveforms` asks once per day per network
            # and `_read_waveform_from_s3` once per object, so re-asking here
            # is what turned one refusal into thousands of requests.
            raise verdict

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
            except ES_TERMINAL as exc:
                # Recorded here rather than in `get_es_credential` so a
                # monkeypatched exchange is covered too, and so this holds
                # whichever layer produced the verdict.
                self.es_record_verdict(net, year, exc)
                # A 400 is the ONE status where a different request may
                # legitimately succeed, and only by adding the year we omitted.
                # `escalate_scope_mode` returns False the second time, so this
                # recursion is at most one level deep.
                if isinstance(exc, EarthScopeScopeRefused) and \
                        not isinstance(exc, EarthScopeScopeIncomplete) and \
                        self.escalate_scope_mode(net):
                    logger.warning(f"Scope {scope} refused with 400: {exc}")
                    return self.get_es_filesystem(net, year)
                raise
            self.set_es_filesystem(key)
        return self.es_fs[key]

    def _es_throttle(self, key) -> None:
        """Refuse locally when one scope is being re-exchanged too often.

        Nothing here reaches EarthScope. A credential lives an hour, so more
        than ES_REFRESH_BUDGET exchanges of the same scope inside
        ES_REFRESH_WINDOW seconds is our own retry logic churning - the
        signature of the traffic EarthScope reported - and the cheapest place
        to stop it is before the socket.
        """
        now = time.monotonic()
        stamps = [t for t in self.es_exchanges.get(key, [])
                  if now - t < ES_REFRESH_WINDOW]
        if len(stamps) >= ES_REFRESH_BUDGET:
            self.es_exchanges[key] = stamps
            raise EarthScopeExchangeThrottled(
                f"Refusing to exchange {key} again: {len(stamps)} exchanges "
                f"in the last {ES_REFRESH_WINDOW:.0f}s, budget is "
                f"{ES_REFRESH_BUDGET}. A credential is valid for an hour, so "
                f"this is a retry loop, not an expiry. Raised locally - no "
                f"request was sent."
            )
        stamps.append(now)
        self.es_exchanges[key] = stamps

    def get_es_credential(self, net, year=None):
        """Exchange one scope for temporary AWS credentials.

        Every 4xx except 429 is terminal AND remembered - each one has a type
        in `ES_TERMINAL`, so none can slip past `es_record_verdict` and be
        re-asked. Only 429 and 5xx are retried, with
        exponential backoff and jitter so a fleet-wide blip does not resolve
        into every worker knocking in lockstep.

        The verdict is recorded HERE as well as in `get_es_filesystem`, so that
        the callers that use the exchange directly - `netyear_sweep`,
        `diag_earthscope` - get the same memory. Recording twice is harmless;
        not recording at one of the two entry points is not.
        """
        try:
            return self._exchange_es_credential(net, year)
        except ES_TERMINAL as exc:
            self.es_record_verdict(net, year, exc)
            raise

    def _exchange_es_credential(self, net, year=None):
        verdict = self.es_verdict(net, year)
        if verdict is not None:
            raise verdict

        scope = self.es_scope(net, year)
        key = json.dumps(scope, sort_keys=True)

        # A network on year scoping with no year to give: EarthScope would
        # answer 400, and this is the request they saw. Answer it here.
        if self.es_scope_mode.get(net) == "network+year" and "year" not in scope:
            exc = EarthScopeScopeIncomplete(
                f"Refusing to ask for {scope} without a year: {net} is a "
                f"temporary FDSN code, those are reused across experiments, "
                f"and EarthScope authorises them per year. A year-less request "
                f"can only ever return 400, so it is not sent. The caller must "
                f"pass the year of the data it is about to read."
            )
            self.es_record_verdict(net, year, exc)
            raise exc

        self._es_throttle(key)

        last = None
        for attempt in range(1, ES_CREDENTIAL_ATTEMPTS + 1):
            try:
                return self.es_client().user.get_aws_credentials(
                    role=EARTHSCOPE_ROLE,
                    ttl_threshold=self.ttl_threshold,
                    **scope,
                )
            except Exception as exc:
                last = exc
                # The SDK raises its OWN types for 401 and 403 instead of an
                # HTTPStatusError, so neither carries `.response` and the 4xx
                # fast-fail below never saw them. Both are verdicts, and the
                # sweep measured the cost of not knowing that: network `LH`
                # burned 25 s per year - five attempts, five-second sleeps -
                # re-asking a question already answered 403.
                from earthscope_sdk.auth.error import (AuthFlowError,
                                                       UnauthenticatedError,
                                                       UnauthorizedError)
                if isinstance(exc, UnauthorizedError):
                    raise EarthScopeNoAccess(
                        f"EarthScope returned 403 for {scope} on role "
                        f"{EARTHSCOPE_ROLE}: the account does not have access to "
                        f"network {net}"
                        f"{' for ' + str(year) if 'year' in scope else ''}. "
                        f"Not retryable - ask EarthScope for access, or drop "
                        f"it from the campaign."
                    ) from exc
                if isinstance(exc, UnauthenticatedError):
                    raise EarthScopeAuthFailed(
                        f"EarthScope returned 401 for {scope}: the credential "
                        f"itself was rejected. Retrying cannot fix a bad "
                        f"token - check ES_OAUTH2__REFRESH_TOKEN in Secrets "
                        f"Manager (quakescope/earthscope-refresh-token)."
                    ) from exc
                if isinstance(exc, AuthFlowError):
                    # Every remaining AuthFlowError is about OUR credentials,
                    # not about this scope: InvalidRefreshTokenError,
                    # NoRefreshTokenError, the device-code errors. None is
                    # congestion, and none is fixed by asking again - but they
                    # carry no `.response`, so the status-code branch below
                    # never saw them and they fell into the retry loop. Each
                    # attempt re-ran the refresh grant against
                    # login.earthscope.org, so a revoked token became five
                    # token-endpoint hits per scope, per shard, per worker.
                    raise EarthScopeAuthFailed(
                        f"EarthScope authentication failed before the request "
                        f"for {scope} was made ({type(exc).__name__}: "
                        f"{str(exc)[:200]}). This is our token, not this "
                        f"scope - retrying re-runs the refresh grant and "
                        f"cannot succeed. Check ES_OAUTH2__REFRESH_TOKEN in "
                        f"Secrets Manager (quakescope/earthscope-refresh-token)."
                    ) from exc
                # Report the HTTP status and body, not just the class.
                # "HTTPStatusError" alone is indistinguishable between "not
                # allowed", "no such network-year" and "malformed parameter",
                # and the retry loop then hides it five times over.
                resp = getattr(exc, "response", None)
                detail = f"{type(exc).__name__}"
                if resp is not None:
                    body = (resp.text or "").strip().replace("\n", " ")[:300]
                    detail = f"HTTP {resp.status_code} {body}"
                    code = resp.status_code
                    # 404 means this network-year is not in the archive at
                    # all. Not a scope error and not an access error, so
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
                    if code == 400:
                        raise EarthScopeScopeRefused(
                            f"EarthScope rejected the scope {scope} as a bad "
                            f"request: {detail}. Temporary FDSN codes need a "
                            f"year; this is a verdict on the request, not "
                            f"congestion, and is not retried."
                        ) from exc
                    if 400 <= code < 500 and code != 429:
                        raise EarthScopeRequestRefused(
                            f"EarthScope refused credentials for scope "
                            f"{scope} on role {EARTHSCOPE_ROLE}: {detail}. "
                            f"This is a verdict on the request, not "
                            f"congestion - check the scope before retrying."
                        ) from exc
                if attempt == ES_CREDENTIAL_ATTEMPTS:
                    break
                # Exponential backoff with full jitter. The old fixed 5 s put
                # every worker that saw the same blip back on the endpoint at
                # the same instant, which is how a transient 5xx turned into a
                # synchronised stampede.
                delay = random.uniform(
                    0, min(30.0, 2.0 * (2 ** (attempt - 1))))
                retry_after = None
                if resp is not None:
                    with_header = resp.headers.get("retry-after")
                    if with_header:
                        try:
                            retry_after = float(with_header)
                        except ValueError:
                            retry_after = None
                if retry_after is not None:
                    delay = max(delay, min(retry_after, 60.0))
                logger.warning(
                    f"EarthScope credential request failed for {scope} "
                    f"({attempt}/{ES_CREDENTIAL_ATTEMPTS}): {detail}. "
                    f"Sleeping {delay:.1f}s."
                )
                time.sleep(delay)
        # Fail rather than retry forever. An unbounded loop here never completes
        # and never errors, so a Spot worker holds its shard until the lease
        # expires and the next worker inherits the same hang.
        raise RuntimeError(
            f"Could not obtain EarthScope credentials for {scope} after "
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

        `escalate` adds the year to the scope first, and only for a network
        that was not year-scoped already - see `escalate_scope_mode`. A denial
        has two plausible causes and they need different responses: an expired
        token wants the same request again, a scope that was missing the year
        wants a more specific one. The caller tries them in that order.
        """
        if self.get_data_center(net) != "earthscope" or \
                EarthScopeS3ObjectHelper.is_open_data(net):
            return                            # anonymous; nothing to renew
        verdict = self.es_verdict(net, year)
        if verdict is not None:
            raise verdict                     # already answered; do not re-ask
        if escalate and self.escalate_scope_mode(net):
            return self.get_es_filesystem(net, year)
        key = json.dumps(self.es_scope(net, year), sort_keys=True)
        self.credentials.pop(key, None)
        self.es_fs.pop(key, None)
        return self.get_es_filesystem(net, year)

    def _is_restricted(self, net) -> bool:
        return (self.get_data_center(net) == "earthscope"
                and not EarthScopeS3ObjectHelper.is_open_data(net))

    def note_access_denied(self, net, year=None) -> int:
        """Count an AccessDenied on this scope and return the running total.

        Per scope, not per object. The old counter lived in
        `_read_waveform_from_s3` and reset on every call, so a station-day we
        genuinely cannot read re-ran the whole refresh dance for each of its
        objects - thousands of token exchanges to re-learn one fact.
        Reset by `clear_access_denied` as soon as a read on that scope
        succeeds, so a real expiry still gets its refresh.

        Only meaningful for the credentialed tier: a denial on an anonymous
        bucket has no credential to renew, so it keeps the per-object bound and
        nothing accumulates.
        """
        if not self._is_restricted(net):
            return 0
        key = self.es_scope_key(net, year)
        self.es_denials[key] = self.es_denials.get(key, 0) + 1
        return self.es_denials[key]

    def clear_access_denied(self, net, year=None) -> None:
        """A read succeeded, so the scope is good; restore its refresh budget."""
        if self._is_restricted(net):
            self.es_denials.pop(self.es_scope_key(net, year), None)


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
        self._logged = set()                  # see `_log_once`
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
                        listing = self.s3helper.list_day(net, fs, prefix)
                    avail_uri[net] += listing
                except EarthScopeNoAccess as exc:
                    # 403 is a fact about access, not a transient fault.
                    # Failing the shard means ten retries, permanent failure,
                    # and a requeue to fail again - so a network we are simply
                    # not allowed would look exactly like a broken fleet. Treat
                    # it as no data for this network and let the shard finish.
                    #
                    # Loud, not silent: unlike a 404 this is worth acting on,
                    # and the sweep exists to find them before launch rather
                    # than during it.
                    # Once per network, not once per day: the verdict is
                    # cached, so the only thing repeating would be the log.
                    self._log_once(f"denied:{net}",
                                   f"{net}: {exc} Skipping this network for "
                                   f"the rest of the shard.")
                    continue
                except EarthScopeNetworkYearNotFound as exc:
                    # The archive has no such network-year. Quiet, and answered
                    # from `es_refused` on every later day of the shard rather
                    # than re-asked - 5A/2018 alone would otherwise have sent
                    # one credential request per day it does not exist for.
                    logger.debug(f"{net} {day:%Y.%j}: {exc}")
                except (EarthScopeScopeRefused,
                        EarthScopeRequestRefused) as exc:
                    # 400. Either the scope is wrong in a way escalation could
                    # not fix, or we declined to send a request that cannot
                    # succeed. Loud, because it means the plan and the archive
                    # disagree about this network - but not fatal, and not
                    # retried.
                    self._log_once(f"scope:{net}:{day.year}",
                                   f"{net} {day.year}: {exc}")
                except EarthScopeExchangeThrottled as exc:
                    # Our own rate limit, not EarthScope's answer. Nothing was
                    # sent; skip the day and let the window reopen.
                    logger.warning(f"{net} {day:%Y.%j}: {exc}")
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

                if dc in ["scedc", "ncedc", "geonet"]:
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

    def _log_once(self, key: str, message: str) -> None:
        """Say it at ERROR the first time and at DEBUG after.

        The listing loop runs once per day per network, so a cached verdict
        would otherwise print an identical error line for every day of the
        shard - hundreds of them, all describing one fact.
        """
        if key in self._logged:
            logger.debug(message)
            return
        self._logged.add(key)
        logger.error(message)

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with full jitter, capped at 30 s.

        A fixed sleep put every worker that saw the same blip back on the
        endpoint at the same instant: a transient fault became a synchronised
        stampede, which from the far side is indistinguishable from an attack.
        """
        return random.uniform(0, min(30.0, 2.0 * (2 ** (attempt - 1))))

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
        # that was never allowed to read the object, and retrying one forever is
        # indistinguishable from a hang: a profiling shard sat on a single
        # denied GET for 447 minutes, logging nothing, until Spot reclaimed it.
        # Bound the attempts and say what is actually wrong.
        denied = busy = 0
        while True:
            try:
                # Inside the loop and inside a handler: after a denial this can
                # raise the verdict EarthScope already gave, and that must end
                # the object rather than escape into the picking loop.
                fs = self.s3helper.get_filesystem(net, year)
            except EarthScopeAuthFailed:
                raise                         # bad token: fail the shard, loudly
            except (EarthScopeNoAccess, EarthScopeScopeRefused,
                    EarthScopeRequestRefused,
                    EarthScopeExchangeThrottled) as exc:
                logger.error(f"No credential for {uri}: {exc}")
                return obspy.Stream()
            except FileNotFoundError:
                return obspy.Stream()
            except RuntimeError as exc:
                logger.error(f"No credential for {uri}: {exc}")
                return obspy.Stream()
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
                    # The scope reads, so any earlier denial on it was a real
                    # expiry. Give the refresh budget back.
                    self.s3helper.clear_access_denied(net, year)
                    with stage("resample"):
                        return downsample_to_target(st)
            # ORDER MATTERS. PermissionError and FileNotFoundError both
            # subclass OSError, so the `except OSError` that used to sit first
            # shadowed both of them completely. Neither handler could run: an
            # AccessDenied (s3fs maps it to PermissionError, with no errno) fell
            # out of the OSError branch without returning, the `while True`
            # went round again, and the object was re-HEADed as fast as the
            # network allowed until STATION_DAY_TIMEOUT killed it 900 s later.
            # That is a tight request loop against EarthScope's access point
            # per denied object, and it meant the whole ES_DENIED_ATTEMPTS
            # budget below - the code that is supposed to bound refreshes - had
            # never once executed.
            except PermissionError as e:
                denied += 1
                # Per scope, not per object: `denied` resets on every call, so
                # on its own it re-ran the whole refresh dance for each of the
                # thousands of objects in a station-day we cannot read.
                # `scope_denied` remembers across them, so the fleet asks
                # EarthScope once per scope instead of once per object.
                scope_denied = self.s3helper.note_access_denied(net, year)
                if scope_denied > ES_DENIED_ATTEMPTS:
                    # Said once per scope, not once per object.
                    if scope_denied == ES_DENIED_ATTEMPTS + 1:
                        logger.error(
                            f"Access denied for {uri} after "
                            f"{ES_DENIED_ATTEMPTS} refreshes - refreshing the "
                            f"credential did not help, so this is not an "
                            f"expiry. Credential scope was "
                            f"{self.s3helper.es_scope(net, year)} on role "
                            f"{EARTHSCOPE_ROLE}. An unscoped credential lists "
                            f"but never reads, so first check that scope is "
                            f"reaching the token exchange; if it is, the "
                            f"account genuinely has no access to {net}. Restrict the "
                            f"campaign to Open Data "
                            f"({','.join(sorted(EARTHSCOPE_OPEN_DATA_NETWORKS))}) "
                            f"plus SCEDC/NCEDC to proceed without it. Every "
                            f"later object on this scope is skipped without "
                            f"asking EarthScope again."
                        )
                    return obspy.Stream()
                if denied > ES_DENIED_ATTEMPTS:
                    return obspy.Stream()
                logger.debug(e.args[0])
                # First denial: assume an expired token and re-request the same
                # scope. Second: assume the scope was missing its year and add
                # one. Cheapest explanation first, and neither ever drops scope
                # - see `escalate_scope_mode`.
                try:
                    self.s3helper.update_es_filesystem(
                        net, year, escalate=denied > 1)
                except EarthScopeAuthFailed:
                    raise                     # bad token: fail the shard
                except ES_TERMINAL as exc:
                    # EarthScope has now answered definitively. Stop.
                    logger.error(f"Cannot renew the credential for {uri}: {exc}")
                    return obspy.Stream()
                except EarthScopeExchangeThrottled as exc:
                    logger.warning(f"{exc}")
                    return obspy.Stream()
                except RuntimeError as exc:
                    logger.error(f"Cannot renew the credential for {uri}: {exc}")
                    return obspy.Stream()
                logger.warning(
                    f"Credential refreshed after access denied "
                    f"({denied}/{ES_DENIED_ATTEMPTS}) for {uri}; scope now "
                    f"{self.s3helper.es_scope(net, year)}"
                )
            except FileNotFoundError:
                return obspy.Stream()
            except OSError as e:
                if e.errno == 5:
                    logger.warning(f"Not authorized to access this resource: {uri}")
                    return obspy.Stream()
                # Any other OSError used to fall through without returning and
                # spin the loop. Treat it as the transient it usually is, under
                # the same budget as a busy S3.
                busy += 1
                if busy > S3_BUSY_ATTEMPTS:
                    logger.error(
                        f"{type(e).__name__} on {uri} after {busy} attempts; "
                        f"giving up on this object rather than holding the "
                        f"shard open."
                    )
                    return obspy.Stream()
                time.sleep(self._backoff(busy))
            except ClientError:
                busy += 1
                if busy > S3_BUSY_ATTEMPTS:
                    logger.error(
                        f"S3 still failing after {busy} attempts for {uri}; "
                        f"giving up on this object rather than holding the "
                        f"shard open."
                    )
                    return obspy.Stream()
                delay = self._backoff(busy)
                logger.warning(
                    f"S3 might be busy ({busy}/{S3_BUSY_ATTEMPTS}). "
                    f"Sleep {delay:.1f}s and retry."
                )
                time.sleep(delay)
            except (ValueError, TypeError):
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
