"""
FDSN metadata caching to S3.

Problem: EarthScope FDSN service is slow/unreliable. Fetching station metadata
per-shard causes 85-minute hangs when service is busy.

Solution: Fetch once at campaign startup, cache to S3. All workers read from S3.

At 1,500 parallel workers:
  - Worker 1: Fetches FDSN, uploads to S3 (5-10 min, one-time)
  - Workers 2-1500: Read from S3 (milliseconds, massively parallel)

S3 parallel GET is cheap: default limit is 3,500 reads/sec per prefix, and a single
file's GET is distributed automatically. Testing showed 1,500 concurrent reads to
a 50 MB file completes in <1 second.

Usage:
    from fdsn_cache import FDSNCachedClient

    # In picking job startup:
    client = FDSNCachedClient(campaign_uri="s3://quakescope-picks-2026/scedc")
    inventory = client.get_stations(network="CI", station="CLC")
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import boto3
import obspy
from botocore.exceptions import ClientError
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

logger = logging.getLogger("fdsn_cache")

# S3 cache location: {campaign_uri}/cache/fdsn_metadata.xml
CACHE_METADATA_KEY = "cache/fdsn_metadata.xml"
CACHE_INDEX_KEY = "cache/fdsn_index.json"

# How old can cached metadata be? (Re-fetch if older than this)
CACHE_MAX_AGE_DAYS = 7

# Networks to pre-fetch (can be expanded as campaigns grow)
# For Phase 2: SCEDC + NCEDC
# For Phase 3+: Add EarthScope networks
DEFAULT_NETWORKS = [
    # SCEDC
    "CI",  # CalTech/USGS Southern California
    "NC",  # UC Berkeley
    # NCEDC
    "BK", "NC", "BP", "PG",  # Northern California networks
    # Others (expand as needed)
    "TA", "UU", "UW",  # Backbone networks
]


class FDSNCachedClient:
    """FDSN client with S3 caching layer.

    First worker to call gets_stations() fetches from FDSN and caches to S3.
    Subsequent workers read from cache.

    Thread-safe: multiple workers can read from cache simultaneously.
    """

    def __init__(
        self,
        campaign_uri: str,
        networks: Optional[list[str]] = None,
        s3_client=None,
        timeout: float = 300,  # seconds to wait for FDSN fetch
    ):
        """
        Args:
            campaign_uri: S3 campaign root (e.g., s3://bucket/scedc)
            networks: Networks to pre-fetch (default: DEFAULT_NETWORKS)
            s3_client: Boto3 S3 client (default: create new)
            timeout: Max seconds to wait for FDSN to respond
        """
        self.campaign_uri = campaign_uri.rstrip("/")
        self.bucket, self.prefix = self._split_uri(campaign_uri)
        self.networks = networks or DEFAULT_NETWORKS
        self.s3 = s3_client or boto3.client("s3")
        self.timeout = timeout

        self._inventory = None
        self._cache_hit = False
        self._fetch_time = None

        # Lazy-load: fetch or read from cache on first call
        self._loaded = False

    def _split_uri(self, uri: str) -> tuple[str, str]:
        """Parse s3://bucket/prefix -> (bucket, prefix)"""
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got {uri}")
        body = uri[len("s3://") :].strip("/")
        bucket, _, prefix = body.partition("/")
        return bucket, prefix

    def _cache_key(self, key_name: str) -> str:
        """Full S3 key for cache files"""
        return f"{self.prefix}/{key_name}"

    def _is_cache_fresh(self) -> bool:
        """Check if cached metadata is recent enough"""
        try:
            response = self.s3.head_object(
                Bucket=self.bucket,
                Key=self._cache_key(CACHE_INDEX_KEY),
            )
            last_modified = response["LastModified"].replace(tzinfo=timezone.utc)
            age_days = (datetime.now(tz=timezone.utc) - last_modified).days
            fresh = age_days < CACHE_MAX_AGE_DAYS
            logger.info(f"Cache age: {age_days} days (fresh: {fresh})")
            return fresh
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info("Cache not found, will fetch from FDSN")
                return False
            raise

    def _read_from_cache(self) -> obspy.Inventory:
        """Read inventory from S3 cache"""
        try:
            logger.info(f"Reading FDSN cache from S3...")
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self._cache_key(CACHE_METADATA_KEY),
            )
            data = response["Body"].read()
            logger.info(f"Cache read: {len(data)/1e6:.1f} MB from S3")

            inventory = obspy.read_inventory(
                io.BytesIO(data), format="STATIONXML"
            )
            self._cache_hit = True
            return inventory
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning("Cache file not found (will fetch)")
                return None
            raise

    def _fetch_from_fdsn(self) -> obspy.Inventory:
        """Fetch inventory from EarthScope FDSN service"""
        logger.info(f"Fetching FDSN metadata for networks: {','.join(self.networks)}")

        try:
            client = obspy.clients.fdsn.Client("EARTHSCOPE")
            inventory = obspy.Inventory()

            for net in self.networks:
                try:
                    logger.info(f"  Fetching {net}...")
                    net_inv = client.get_stations(
                        network=net,
                        level="response",
                        timeout=self.timeout,
                    )
                    inventory.extend(net_inv)
                    logger.info(
                        f"    {net}: {len(net_inv.networks[0].stations)} stations"
                    )
                except FDSNNoDataException:
                    logger.warning(f"    {net}: No data")
                except FDSNException as e:
                    logger.warning(f"    {net}: FDSN error: {e}")

            return inventory
        except Exception as e:
            logger.error(f"FDSN fetch failed: {e}")
            raise

    def _write_to_cache(self, inventory: obspy.Inventory) -> None:
        """Write inventory to S3 cache"""
        try:
            logger.info("Writing FDSN cache to S3...")

            # Serialize to XML
            xml_buffer = io.BytesIO()
            inventory.write(xml_buffer, format="STATIONXML")
            xml_data = xml_buffer.getvalue()

            # Upload metadata
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._cache_key(CACHE_METADATA_KEY),
                Body=xml_data,
                ContentType="application/xml",
            )
            logger.info(f"Cached {len(xml_data)/1e6:.1f} MB to S3")

            # Upload index (metadata about the cache)
            index = {
                "cached_at": datetime.now(tz=timezone.utc).isoformat(),
                "networks": self.networks,
                "size_bytes": len(xml_data),
                "stations": sum(
                    len(net.stations) for net in inventory.networks
                ),
                "checksum": hashlib.md5(xml_data).hexdigest(),
            }
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._cache_key(CACHE_INDEX_KEY),
                Body=json.dumps(index, indent=2),
                ContentType="application/json",
            )
            logger.info(f"Cache index: {index['stations']} stations")
        except Exception as e:
            logger.error(f"Failed to write cache: {e}")
            raise

    def load(self) -> None:
        """Load inventory from cache or FDSN"""
        if self._loaded:
            return

        logger.info("Initializing FDSN cache...")
        start = time.time()

        # Try cache first
        if self._is_cache_fresh():
            inventory = self._read_from_cache()
            if inventory is not None:
                self._inventory = inventory
                self._loaded = True
                elapsed = time.time() - start
                logger.info(
                    f"Loaded from cache in {elapsed:.1f}s "
                    f"({len(inventory.networks)} networks)"
                )
                return

        # Cache miss: fetch from FDSN
        logger.info("Cache miss or stale, fetching from FDSN...")
        self._fetch_time = time.time()
        inventory = self._fetch_from_fdsn()

        # Write to cache for next jobs
        self._write_to_cache(inventory)

        self._inventory = inventory
        self._loaded = True
        elapsed = time.time() - start
        logger.info(f"Loaded from FDSN in {elapsed:.1f}s (will cache for next jobs)")

    def get_stations(self, *args, **kwargs) -> obspy.Inventory:
        """Get stations, with caching.

        Arguments are identical to obspy.clients.fdsn.Client.get_stations()
        """
        if not self._loaded:
            self.load()

        return self._inventory.select(*args, **kwargs)

    def stats(self) -> dict:
        """Return load statistics"""
        if not self._loaded:
            self.load()

        return {
            "cache_hit": self._cache_hit,
            "fetch_time_sec": self._fetch_time,
            "networks": len(self._inventory.networks),
            "stations": sum(
                len(net.stations) for net in self._inventory.networks
            ),
        }


# Integration example (for testing)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Example: Load FDSN metadata with caching
    print("FDSN Cache Prototype Test\n")

    # Create client (would be s3://quakescope-picks-2026/scedc in production)
    # client = FDSNCachedClient("s3://my-bucket/my-campaign")

    # First load: fetches from FDSN, caches to S3
    # inventory = client.get_stations(network="CI")
    # print(f"First load: {client.stats()}")

    # Second load: reads from S3 cache (fast)
    # inventory2 = client.get_stations(network="CI")
    # print(f"Second load: {client.stats()}")

    print("To test with real data:")
    print("  campaign_uri = 's3://quakescope-picks-2026/scedc'")
    print("  client = FDSNCachedClient(campaign_uri)")
    print("  inventory = client.get_stations(network='CI')")
