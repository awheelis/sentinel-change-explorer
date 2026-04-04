"""PIF-based relative radiometric normalization for Sentinel-2 band pairs."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.indices import compute_ndvi, compute_ndbi, compute_mndwi


def normalize_pif(
    before_bands: dict[str, np.ndarray],
    after_bands: dict[str, np.ndarray],
    pif_threshold: float = 0.02,
    min_pif_fraction: float = 0.01,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply PIF-based relative radiometric normalization to after-scene bands.

    Identifies pseudo-invariant features (PIFs) — pixels where NDVI, NDBI, and
    MNDWI all changed by less than pif_threshold between dates. Fits a per-band
    linear regression on PIF pixels and applies the correction to the full
    after-scene.

    Args:
        before_bands: Dict mapping band key -> 2D uint16 array for the earlier date.
        after_bands: Dict mapping band key -> 2D uint16 array for the later date.
        pif_threshold: Maximum absolute index change for a pixel to be considered a PIF.
        min_pif_fraction: Minimum fraction of pixels that must be PIFs to proceed.

    Returns:
        Tuple of (corrected_after_bands, info_dict).
        corrected_after_bands: Dict with same keys as after_bands, values are uint16 arrays.
        info_dict keys: pif_count, pif_fraction, skipped, bands (per-band gain/offset).

    Raises:
        ValueError: If before and after band shapes don't match.
    """
    # Validate shapes
    for key in before_bands:
        if key not in after_bands:
            raise ValueError(f"Band '{key}' missing from after_bands")
        if before_bands[key].shape != after_bands[key].shape:
            raise ValueError(
                f"Band shape mismatch for '{key}': "
                f"{before_bands[key].shape} vs {after_bands[key].shape}"
            )

    # Need nir, red, swir16, green for all three indices
    required = {"nir", "red", "swir16", "green"}
    available = set(before_bands.keys()) & set(after_bands.keys())
    if not required.issubset(available):
        return dict(after_bands), {
            "pif_count": 0, "pif_fraction": 0.0, "skipped": True, "bands": {},
        }

    ndvi_before = compute_ndvi(before_bands["nir"], before_bands["red"])
    ndvi_after = compute_ndvi(after_bands["nir"], after_bands["red"])
    ndbi_before = compute_ndbi(before_bands["swir16"], before_bands["nir"])
    ndbi_after = compute_ndbi(after_bands["swir16"], after_bands["nir"])
    mndwi_before = compute_mndwi(before_bands["green"], before_bands["swir16"])
    mndwi_after = compute_mndwi(after_bands["green"], after_bands["swir16"])

    pif_mask = (
        (np.abs(ndvi_after - ndvi_before) < pif_threshold)
        & (np.abs(ndbi_after - ndbi_before) < pif_threshold)
        & (np.abs(mndwi_after - mndwi_before) < pif_threshold)
    )

    total_pixels = pif_mask.size
    pif_count = int(np.sum(pif_mask))
    pif_fraction = pif_count / total_pixels if total_pixels > 0 else 0.0

    if pif_fraction < min_pif_fraction:
        return dict(after_bands), {
            "pif_count": pif_count, "pif_fraction": pif_fraction,
            "skipped": True, "bands": {},
        }

    corrected: dict[str, np.ndarray] = {}
    band_info: dict[str, dict[str, float]] = {}

    for key in after_bands:
        before_pif = before_bands[key][pif_mask].astype(np.float64)
        after_pif = after_bands[key][pif_mask].astype(np.float64)
        gain, offset = np.polyfit(after_pif, before_pif, deg=1)
        corrected_float = after_bands[key].astype(np.float64) * gain + offset
        corrected[key] = np.clip(corrected_float, 0, 65535).astype(np.uint16)
        band_info[key] = {"gain": float(gain), "offset": float(offset)}

    return corrected, {
        "pif_count": pif_count, "pif_fraction": pif_fraction,
        "skipped": False, "bands": band_info,
    }
