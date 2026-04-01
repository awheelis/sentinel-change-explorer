"""Spectral index computation for Sentinel-2 bands.

All functions accept uint16 raw Sentinel-2 values or float32 arrays.
All output arrays are float32 in the range [-1, 1].
"""
from __future__ import annotations

import numpy as np


def _safe_normalized_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute (a - b) / (a + b), returning 0 where denominator is zero.

    Args:
        a: Numerator-contributing band array.
        b: Denominator-contributing band array.

    Returns:
        float32 array of normalized difference values clipped to [-1, 1].
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom == 0, 0.0, (a - b) / denom)
    return np.clip(result.astype(np.float32), -1.0, 1.0)


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    For Sentinel-2: nir=B08, red=B04.
    Positive values → vegetation; negative → water/urban.

    Args:
        nir: Near-infrared band array (B08).
        red: Red band array (B04).

    Returns:
        float32 NDVI array in [-1, 1].
    """
    return _safe_normalized_diff(nir, red)


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Built-up Index.

    NDBI = (SWIR - NIR) / (SWIR + NIR)

    For Sentinel-2: swir=B11 (20m, must be resampled to 10m before calling),
    nir=B08.
    Positive values → built-up/urban areas.

    Args:
        swir: Short-wave infrared band array (B11, resampled to match nir).
        nir: Near-infrared band array (B08).

    Returns:
        float32 NDBI array in [-1, 1].
    """
    return _safe_normalized_diff(swir, nir)


def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute Modified Normalized Difference Water Index.

    MNDWI = (Green - SWIR) / (Green + SWIR)

    For Sentinel-2: green=B03, swir=B11 (20m, must be resampled to 10m).
    Positive values → water bodies.

    Args:
        green: Green band array (B03).
        swir: Short-wave infrared band array (B11, resampled to match green).

    Returns:
        float32 MNDWI array in [-1, 1].
    """
    return _safe_normalized_diff(green, swir)


def compute_change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Compute pixel-wise change between two index arrays (after minus before).

    Args:
        before: Index array for the earlier date.
        after: Index array for the later date.

    Returns:
        float32 difference array. Positive = gain, negative = loss.
    """
    return (after.astype(np.float32) - before.astype(np.float32))
