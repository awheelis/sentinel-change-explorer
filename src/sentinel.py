"""Sentinel-2 STAC search and band loading from Element84 Earth Search v1.

Earth Search v1 asset keys for sentinel-2-l2a (confirmed):
  blue   → B02 (10m)
  green  → B03 (10m)
  red    → B04 (10m)
  nir    → B08 (10m)
  swir16 → B11 (20m)
"""
from __future__ import annotations

import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject
from pystac_client import Client

_BAND_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "bands"

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
SENTINEL_COLLECTION = "sentinel-2-l2a"

# Rasterio environment for unsigned S3 access to public COGs
_RASTERIO_ENV = {
    "AWS_NO_SIGN_REQUEST": os.environ.get("AWS_NO_SIGN_REQUEST", "YES"),
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": os.environ.get("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES"),
    "GDAL_HTTP_MULTIPLEX": os.environ.get("GDAL_HTTP_MULTIPLEX", "YES"),
    "GDAL_HTTP_VERSION": os.environ.get("GDAL_HTTP_VERSION", "2"),
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": os.environ.get("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff"),
    "GDAL_DISABLE_READDIR_ON_OPEN": os.environ.get("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"),
}


def search_scenes(
    bbox: tuple[float, float, float, float],
    date_range: str,
    max_cloud_cover: float = 20.0,
    max_items: int = 10,
) -> list[dict[str, Any]]:
    """Search for Sentinel-2 L2A scenes matching criteria.

    Args:
        bbox: Bounding box as (west, south, east, north) in WGS84.
        date_range: ISO 8601 interval string, e.g. "2023-06-01/2023-06-30".
        max_cloud_cover: Maximum cloud cover percentage (0-100).
        max_items: Maximum number of results to return.

    Returns:
        List of scene dicts sorted by cloud cover ascending. Each dict has:
        - id: Scene identifier string
        - cloud_cover: Cloud cover percentage (float)
        - datetime: Scene acquisition datetime string
        - assets: Dict mapping asset key → asset href string
        - bbox: Scene bounding box [west, south, east, north]
    """
    client = Client.open(EARTH_SEARCH_URL)
    search = client.search(
        collections=[SENTINEL_COLLECTION],
        bbox=list(bbox),
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        sortby=["+properties.eo:cloud_cover"],
        max_items=max_items,
    )
    results = []
    for item in search.items():
        results.append({
            "id": item.id,
            "cloud_cover": item.properties.get("eo:cloud_cover", 0.0),
            "datetime": str(item.datetime),
            "assets": {k: v.href for k, v in item.assets.items()},
            "bbox": list(item.bbox),
        })
    return results


def load_bands(
    scene: dict[str, Any],
    bbox: tuple[float, float, float, float],
    band_keys: list[str],
    target_res: int = 10,
) -> dict[str, np.ndarray]:
    """Load specified bands from a STAC scene as windowed numpy arrays.

    Reads only the pixels within bbox using COG windowed reads (no full tile
    download). All bands are resampled to match target_res (meters).

    Args:
        scene: Scene dict from search_scenes().
        bbox: Bounding box as (west, south, east, north) in WGS84.
        band_keys: List of asset keys to load, e.g. ["red", "green", "blue"].
        target_res: Target pixel resolution in meters. Bands at coarser
            resolution (e.g. swir16 at 20m) are upsampled to this.

    Returns:
        Dict mapping band key → 2D uint16 numpy array in EPSG:4326 (WGS84).
        Arrays are reprojected from the scene's native UTM CRS so they
        align correctly with WGS84 map overlays.

    Raises:
        KeyError: If a requested band_key is not in scene["assets"].
        RuntimeError: If the band cannot be read from S3.
    """
    assets = scene["assets"]
    for key in band_keys:
        if key not in assets:
            raise KeyError(
                f"Band '{key}' not found in scene assets. "
                f"Available: {list(assets.keys())}"
            )

    # ── Disk cache: check for cached reprojected bands ───────────────────
    cache_key_str = f"v2|{scene['id']}|{bbox}|{sorted(band_keys)}|{target_res}"
    cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_path = _BAND_CACHE_DIR / f"{cache_hash}.npz"

    if cache_path.exists():
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in band_keys}

    west, south, east, north = bbox
    arrays: dict[str, np.ndarray] = {}

    # Determine output shape and CRS from the reference (first) band
    with rasterio.Env(**_RASTERIO_ENV):
        with rasterio.open(assets[band_keys[0]]) as ref_ds:
            dst_crs = ref_ds.crs
            native_bounds = transform_bounds("EPSG:4326", dst_crs, west, south, east, north)
            win = from_bounds(*native_bounds, ref_ds.transform)
            native_res_m = ref_ds.res[0]  # meters per pixel (approximate)
            if target_res <= 0:
                raise ValueError(f"target_res must be positive, got {target_res}")
            scale = native_res_m / target_res
            out_h = max(1, int(round(win.height * scale)))
            out_w = max(1, int(round(win.width * scale)))

    # Load all bands in parallel for faster S3 reads
    def _read_band(key: str) -> np.ndarray:
        with rasterio.Env(**_RASTERIO_ENV):
            with rasterio.open(assets[key]) as ds:
                native_bounds_k = transform_bounds(
                    "EPSG:4326", dst_crs, west, south, east, north,
                )
                window = from_bounds(*native_bounds_k, ds.transform)
                return ds.read(
                    1,
                    window=window,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.bilinear,
                )

    with ThreadPoolExecutor(max_workers=len(band_keys)) as executor:
        futures = {executor.submit(_read_band, key): key for key in band_keys}
        for future in as_completed(futures):
            key = futures[future]
            arrays[key] = future.result()

    # ── Reproject all bands from native UTM to EPSG:4326 ─────────────────
    # Use a canonical WGS84 grid derived from the bbox so that every scene
    # for the same bbox/target_res produces an identical, axis-aligned output.
    # calculate_default_transform preserves the UTM grid rotation, creating a
    # rotated parallelogram with black triangular corners — avoid that.
    dst_crs_4326 = "EPSG:4326"
    src_transform = rasterio.transform.from_bounds(
        *native_bounds, out_w, out_h,
    )
    center_lat_rad = math.radians((south + north) / 2.0)
    m_per_deg_lon = 111_320.0 * math.cos(center_lat_rad)
    m_per_deg_lat = 110_540.0
    dst_width = max(1, int(round((east - west) * m_per_deg_lon / target_res)))
    dst_height = max(1, int(round((north - south) * m_per_deg_lat / target_res)))
    dst_transform = rasterio.transform.from_bounds(
        west, south, east, north, dst_width, dst_height,
    )

    reprojected: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        dst_arr = np.zeros((dst_height, dst_width), dtype=arr.dtype)
        reproject(
            source=arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=dst_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs_4326,
            resampling=Resampling.bilinear,
        )
        reprojected[key] = dst_arr

    # ── Save reprojected bands to disk cache ─────────────────────────────
    _BAND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **reprojected)

    return reprojected
