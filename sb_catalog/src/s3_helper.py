import os
from abc import abstractmethod
from datetime import timedelta

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

        self.es_client = EarthScopeClient()
        self.credential = self.es_client.user.get_aws_credentials()

        self.fs = {
            "scedc": S3FileSystem(anon=True),
            "ncedc": S3FileSystem(anon=True),
        }
        self.fs["earthscope"] = S3FileSystem(
            key=self.credential.aws_access_key_id,
            secret=self.credential.aws_secret_access_key,
            token=self.credential.aws_session_token,
        )

    def get_prefix(self, net, year, day) -> str:
        return self.helpers[self.get_data_center(net)].get_prefix(net, year, day)

    def get_basename(self, net, sta, loc, cha, year, day, c) -> str:
        return self.helpers[self.get_data_center(net)].get_basename(
            net, sta, loc, cha, year, day, c
        )

    def get_filesystem(self, net):
        return self.fs[self.get_data_center(net)]

    def get_es_credential(self):
        """
        Set 5 minutes buffer time to update credential
        """
        return self.es_client.user.get_aws_credentials(
            ttl_threshold=timedelta(minutes=5, seconds=0)
        )

    def update_es_filesystem(self):
        _credential = self.get_es_credential()
        if _credential.aws_access_key_id != self.credential.aws_access_key_id:
            # credential was updated
            self.credential = _credential
            self.set_es_filesystem()
        else:
            pass

        return
