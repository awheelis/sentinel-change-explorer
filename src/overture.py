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

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        logger.debug(
            "Fetching Overture layer '%s' for bbox %s",
            layer, bbox,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            future = pool.submit(core.geodataframe, layer, bbox=bbox)
            gdf: gpd.GeoDataFrame = future.result(timeout=_LAYER_TIMEOUT)
    except _TIMEOUT_AND_TRANSIENT as exc:
        logger.warning(
            "Overture fetch failed for layer '%s': %s",
            layer, exc,
        )
        gdf = gpd.GeoDataFrame()
    except (ValueError, TypeError, ArithmeticError) as exc:
        logger.warning("Failed to fetch Overture layer '%s': %s", layer, exc)
        pool.shutdown(wait=False, cancel_futures=True)
        return gpd.GeoDataFrame()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # --- cache write (including empty results to avoid repeated slow S3 scans) ---
    if use_cache:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if len(gdf) > 0:
                gdf.to_parquet(cache_file)
            else:
                # Write an empty parquet so future calls get a cache hit
                # instead of re-scanning S3 for minutes.
                gpd.GeoDataFrame({"geometry": []}).to_parquet(cache_file)
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
