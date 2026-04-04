"""SCL (Scene Classification Layer) cloud and shadow masking for Sentinel-2.

Masks pixels classified as cloud shadow, cloud medium/high probability,
or thin cirrus before index computation to eliminate false positives.

SCL class reference (ESA):
  0  No data          4  Vegetation       8  Cloud medium prob
  1  Saturated        5  Bare soil        9  Cloud high prob
  2  Dark area        6  Water           10  Thin cirrus
  3  Cloud shadow     7  Unclassified    11  Snow/ice
"""
from __future__ import annotations

import numpy as np

SCL_MASK_VALUES = {3, 8, 9, 10}


def build_scl_mask(scl: np.ndarray) -> np.ndarray:
    """Build a boolean mask from an SCL band array.

    Args:
        scl: 2D array of SCL class values (uint8, values 0-11).

    Returns:
        Boolean 2D array: True where pixel should be masked.
    """
    mask = np.zeros(scl.shape, dtype=bool)
    for val in SCL_MASK_VALUES:
        mask |= scl == val
    return mask


def apply_mask(
    bands: dict[str, np.ndarray], mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Set masked pixels to NaN in all bands.

    Converts bands to float32 (required for NaN) and sets pixels where
    mask is True to NaN. Does not mutate the input arrays.

    Args:
        bands: Dict mapping band key to 2D array.
        mask: Boolean 2D array (True = mask out).

    Returns:
        New dict with float32 arrays; masked pixels are NaN.
    """
    result = {}
    for key, arr in bands.items():
        out = arr.astype(np.float32)
        out[mask] = np.nan
        result[key] = out
    return result


def union_masks(*masks: np.ndarray) -> np.ndarray:
    """OR multiple boolean masks together.

    A pixel masked in any input is masked in the output.

    Args:
        masks: One or more boolean 2D arrays of the same shape.

    Returns:
        Boolean 2D array: union of all input masks.
    """
    result = masks[0].copy()
    for m in masks[1:]:
        result |= m
    return result


def mask_percentage(mask: np.ndarray) -> float:
    """Return the percentage of True pixels in a boolean mask.

    Args:
        mask: Boolean 2D array.

    Returns:
        Percentage (0-100) of masked pixels.
    """
    return float(np.mean(mask) * 100)
