"""GeoTIFF export for change detection rasters."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def create_geotiff(
    delta: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    index_type: str = "",
    before_date: str = "",
    after_date: str = "",
    before_scene_id: str = "",
    after_scene_id: str = "",
) -> bytes:
    """Create an in-memory GeoTIFF from a 2D delta array.

    Args:
        delta: 2D float32 array of index change values.
        bbox: (west, south, east, north) bounding box in EPSG:4326.
        index_type: Spectral index name (e.g. "ndvi").
        before_date: ISO date string for the before scene.
        after_date: ISO date string for the after scene.
        before_scene_id: Sentinel scene ID for the before image.
        after_scene_id: Sentinel scene ID for the after image.

    Returns:
        bytes of a valid GeoTIFF file.
    """
    height, width = delta.shape
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)

    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
        ) as ds:
            ds.write(delta.astype(np.float32), 1)

            # Add non-empty metadata as tags
            tags = {
                k: v
                for k, v in {
                    "index_type": index_type,
                    "before_date": before_date,
                    "after_date": after_date,
                    "before_scene_id": before_scene_id,
                    "after_scene_id": after_scene_id,
                }.items()
                if v
            }
            if tags:
                ds.update_tags(**tags)

        return memfile.read()
