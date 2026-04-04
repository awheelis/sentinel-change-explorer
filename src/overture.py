"""src/overture.py — Overture Maps data fetching with disk cache.

Fetches building footprints, road segments, and place points from the
Overture Maps dataset (hosted on AWS S3) for a given bounding box, and
caches results to disk to avoid redundant network requests.

Cache files are stored as parquet (not GeoJSON) for faster I/O and lossless
GeoDataFrame roundtrip.
"""
from __future__ import annotations

import hashlib
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Tuple

import geopandas as gpd

logger = logging.getLogger(__name__)

# Project root cache directory: <repo_root>/cache/overture/
_CACHE_DIR = Path(__file__).parent.parent / "cache" / "overture"

# Overture type names as accepted by the overturemaps package
_CONTEXT_LAYERS = ("building", "segment", "place")

BBox = Tuple[float, float, float, float]  # (west, south, east, north)

_TRANSIENT_ERRORS = (TimeoutError, ConnectionError, OSError)
_MAX_RETRIES = 3
_RETRY_DELAYS = (2, 4, 8)
_LAYER_TIMEOUT = 15  # seconds per layer fetch
_TIMEOUT_AND_TRANSIENT = (FuturesTimeoutError,) + _TRANSIENT_ERRORS


def _import_overture_core():
    """Import overturemaps.core, raising ImportError if missing."""
    from overturemaps import core
    return core


def _cache_path(layer: str, bbox: BBox) -> Path:
    """Return the parquet cache file path for a (layer, bbox) pair.

    Args:
        layer: Overture type name (e.g. ``"building"``).
        bbox: Bounding box as ``(west, south, east, north)``.

    Returns:
        Path to the cache file (may not exist yet).
    """
    key = f"{layer}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
    # Full MD5 hex (32 chars) used for collision-free filenames;
    # the plan suggested 12-char prefix but full hash is strictly safer.
    digest = hashlib.md5(key.encode()).hexdigest()
    return _CACHE_DIR / f"{layer}_{digest}.parquet"


def fetch_overture_layer(
    layer: str,
    bbox: BBox,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """Fetch a single Overture Maps layer for the given bounding box.

    Results are cached to ``<repo_root>/cache/overture/`` as GeoParquet
    files keyed by a hash of ``(layer, bbox)``.  Subsequent calls with the
    same arguments return the cached result without touching the network.

    Args:
        layer: Overture type name, e.g. ``"building"``, ``"segment"``,
            or ``"place"``.
        bbox: Bounding box as ``(west, south, east, north)`` in WGS-84
            decimal degrees.
        use_cache: When ``True`` (default), read from / write to the on-disk
            cache.  Set to ``False`` to always fetch from S3.

    Returns:
        A :class:`geopandas.GeoDataFrame` with the requested features.
        Returns an empty GeoDataFrame if the fetch fails for any reason.
    """
    cache_file = _cache_path(layer, bbox)

    # --- cache read ---
    if use_cache and cache_file.exists():
        logger.debug("Cache hit: %s", cache_file)
        try:
            return gpd.read_parquet(cache_file)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read cache file %s: %s — refetching.", cache_file, exc)

    # --- fetch from Overture ---
    try:
        core = _import_overture_core()
    except ImportError:
        logger.warning("overturemaps package not installed — skipping layer '%s'", layer)
        return gpd.GeoDataFrame()

    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            logger.debug(
                "Fetching Overture layer '%s' for bbox %s (attempt %d)",
                layer, bbox, attempt + 1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(core.geodataframe, layer, bbox=bbox)
                    gdf: gpd.GeoDataFrame = future.result(timeout=_LAYER_TIMEOUT)
            last_exc = None
            break
        except _TIMEOUT_AND_TRANSIENT as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning(
                    "Overture fetch attempt %d failed (%s), retrying in %ds…",
                    attempt + 1, exc, delay,
                )
                time.sleep(delay)
        except (ValueError, TypeError, ArithmeticError) as exc:
            logger.warning("Failed to fetch Overture layer '%s': %s", layer, exc)
            return gpd.GeoDataFrame()

    if last_exc is not None:
        logger.warning(
            "All %d Overture fetch attempts failed for layer '%s': %s",
            _MAX_RETRIES, layer, last_exc,
        )
        return gpd.GeoDataFrame()

    # --- cache write ---
    if use_cache and len(gdf) > 0:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            gdf.to_parquet(cache_file)
            logger.debug("Cached %d features to %s", len(gdf), cache_file)
        except OSError as exc:
            logger.warning("Failed to write cache file %s: %s", cache_file, exc)

    return gdf


def get_overture_context(
    bbox: BBox,
    use_cache: bool = True,
) -> dict[str, gpd.GeoDataFrame]:
    """Fetch building, road-segment, and place layers for a bounding box.

    Calls :func:`fetch_overture_layer` for each of the three context layers
    concurrently using a thread pool.  Individual layer failures are logged
    as warnings and represented by empty GeoDataFrames so the caller always
    receives a complete dict.
    """
    with ThreadPoolExecutor(max_workers=len(_CONTEXT_LAYERS)) as pool:
        futures = {
            layer: pool.submit(fetch_overture_layer, layer, bbox=bbox, use_cache=use_cache)
            for layer in _CONTEXT_LAYERS
        }
        return {layer: fut.result() for layer, fut in futures.items()}
