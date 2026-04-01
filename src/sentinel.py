"""Sentinel-2 STAC search and band loading from Element84 Earth Search v1.

Earth Search v1 asset keys for sentinel-2-l2a (confirmed):
  blue   → B02 (10m)
  green  → B03 (10m)
  red    → B04 (10m)
  nir    → B08 (10m)
  swir16 → B11 (20m)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pystac_client import Client

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
SENTINEL_COLLECTION = "sentinel-2-l2a"

# Rasterio environment for unsigned S3 access to public COGs
_RASTERIO_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
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
        Dict mapping band key → 2D uint16 numpy array.

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

    west, south, east, north = bbox
    arrays: dict[str, np.ndarray] = {}

    with rasterio.Env(**_RASTERIO_ENV):
        # Determine output shape and CRS from the reference (first) band
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
        # NOTE: output shape is derived from band_keys[0]'s native resolution.
        # Callers should pass a 10m band (red/green/blue/nir) as band_keys[0]
        # to get the target_res scaling correct.

        for key in band_keys:
            href = assets[key]
            with rasterio.open(href) as ds:
                native_bounds_k = transform_bounds("EPSG:4326", dst_crs, west, south, east, north)
                window = from_bounds(*native_bounds_k, ds.transform)
                data = ds.read(
                    1,
                    window=window,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.bilinear,
                )
            arrays[key] = data

    return arrays
