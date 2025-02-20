import os
import time
from abc import abstractmethod
from datetime import datetime, timedelta, timezone

from earthscope_sdk import EarthScopeClient
from s3fs import S3FileSystem

from .constants import NETWORK_MAPPING

EARTHSCOPE_S3_ACCESS_POINT = os.environ["EARTHSCOPE_S3_ACCESS_POINT"]


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

        self.ttl_threshold = timedelta(minutes=5)
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
                time.sleep(5)

    def set_es_filesystem(self):
        self.fs["earthscope"] = S3FileSystem(
            key=self.credential.aws_access_key_id,
            secret=self.credential.aws_secret_access_key,
            token=self.credential.aws_session_token,
        )

    def update_es_filesystem(self):
        if (
            self.credential.expiration - datetime.now(tz=timezone.utc)
        ) < self.ttl_threshold:
            # credential should be updated
            self.credential = self.get_es_credential()
            self.set_es_filesystem()
        else:
            pass

        return
