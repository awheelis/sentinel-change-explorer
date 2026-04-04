"""Spectral index computation for Sentinel-2 bands.

All functions accept uint16 raw Sentinel-2 values or float32 arrays.
All output arrays are float32 in the range [-1, 1].
"""
from __future__ import annotations

import numpy as np


_CHUNK_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB


def _safe_normalized_diff(
    a: np.ndarray, b: np.ndarray, chunk_rows: int | None = None,
) -> np.ndarray:
    """Compute (a - b) / (a + b), returning 0 where denominator is zero.

    For large arrays, processes in row-chunks to bound peak memory.

    Args:
        a: Numerator-contributing band array.
        b: Denominator-contributing band array.
        chunk_rows: If set, process in chunks of this many rows. If None,
            auto-selects based on array size (chunked if > 50 MB per band).

    Returns:
        float32 array of normalized difference values clipped to [-1, 1].
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Band shape mismatch: {a.shape} vs {b.shape}. "
            f"Ensure both bands are resampled to the same resolution."
        )

    if chunk_rows is None and a.nbytes > _CHUNK_THRESHOLD_BYTES:
        chunk_rows = 512

    if chunk_rows is None:
        # Single-pass: original fast path
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        denom = a + b
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(denom == 0, 0.0, (a - b) / denom)
        return np.clip(result.astype(np.float32), -1.0, 1.0)

    # Chunked path: process row slices to bound memory
    h = a.shape[0]
    result = np.empty(a.shape, dtype=np.float32)
    for start in range(0, h, chunk_rows):
        end = min(start + chunk_rows, h)
        a_chunk = a[start:end].astype(np.float32)
        b_chunk = b[start:end].astype(np.float32)
        denom = a_chunk + b_chunk
        with np.errstate(invalid="ignore", divide="ignore"):
            chunk_result = np.where(denom == 0, 0.0, (a_chunk - b_chunk) / denom)
        result[start:end] = np.clip(chunk_result, -1.0, 1.0)
    return result


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


def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Compute Enhanced Vegetation Index.

    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

    Handles atmospheric effects and canopy saturation better than NDVI,
    especially at high leaf-area index (LAI > 3).

    Args:
        nir: Near-infrared band array (B08).
        red: Red band array (B04).
        blue: Blue band array (B02).

    Returns:
        float32 EVI array clipped to [-1, 1].
    """
    if not (nir.shape == red.shape == blue.shape):
        raise ValueError(
            f"Band shape mismatch: nir={nir.shape}, red={red.shape}, blue={blue.shape}. "
            f"Ensure all bands are resampled to the same resolution."
        )
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    blue = blue.astype(np.float32)
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom == 0, 0.0, 2.5 * (nir - red) / denom)
    return np.clip(result.astype(np.float32), -1.0, 1.0)


def compute_change(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Compute pixel-wise change between two index arrays (after minus before).

    Args:
        before: Index array for the earlier date.
        after: Index array for the later date.

    Returns:
        float32 difference array. Positive = gain, negative = loss.
    """
    return (after.astype(np.float32) - before.astype(np.float32))


def compute_adaptive_threshold(
    delta: np.ndarray, fallback: float = 0.10, n_bins: int = 256,
) -> float:
    """Compute an adaptive change threshold using Otsu's method.

    Applies Otsu's algorithm to the absolute delta values to find the
    optimal boundary between "unchanged" and "changed" pixel populations.

    Args:
        delta: 2D float32 change array (after - before).
        fallback: Value returned when data has no variance (e.g. all zeros).
        n_bins: Number of histogram bins for Otsu computation.

    Returns:
        Optimal threshold as a positive float.
    """
    abs_delta = np.abs(delta).ravel().astype(np.float64)
    abs_delta = abs_delta[np.isfinite(abs_delta)]
    if len(abs_delta) == 0 or abs_delta.max() == abs_delta.min():
        return fallback

    counts, bin_edges = np.histogram(abs_delta, bins=n_bins, range=(0, abs_delta.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return fallback

    weight_bg = 0.0
    sum_bg = 0.0
    sum_total = np.dot(counts.astype(np.float64), bin_centers)
    best_thresh = fallback
    best_variance = 0.0

    for i in range(len(counts)):
        weight_bg += counts[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += counts[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_thresh = float(bin_centers[i])

    return best_thresh


# ── Multi-index change classification ────────────────────────────────────

# Category labels returned by classify_change()
UNCHANGED = 0
URBAN_CONVERSION = 1
VEGETATION_LOSS = 2
FLOODING = 3
VEGETATION_GAIN = 4


def classify_change(
    ndvi_delta: np.ndarray,
    ndbi_delta: np.ndarray,
    mndwi_delta: np.ndarray,
    threshold: float = 0.10,
) -> np.ndarray:
    """Classify each pixel's change type using multi-index decision rules.

    Rules are applied in priority order — earlier rules win when multiple
    conditions match:

    1. NDVI down AND NDBI up  → Urban Conversion
    2. MNDWI up AND NDVI down → Flooding / Water Gain
    3. NDVI down (alone)      → Vegetation Loss / Deforestation
    4. NDVI up                → Vegetation Gain / Recovery
    5. Everything else        → Unchanged

    Args:
        ndvi_delta: 2D float32 array of NDVI change (after - before).
        ndbi_delta: 2D float32 array of NDBI change (after - before).
        mndwi_delta: 2D float32 array of MNDWI change (after - before).
        threshold: Minimum absolute change to count as significant.

    Returns:
        uint8 array with category codes (0-4).
    """
    result = np.full(ndvi_delta.shape, UNCHANGED, dtype=np.uint8)

    ndvi_down = ndvi_delta < -threshold
    ndvi_up = ndvi_delta > threshold
    ndbi_up = ndbi_delta > threshold
    mndwi_up = mndwi_delta > threshold

    # Apply in reverse priority so higher-priority rules overwrite
    result[ndvi_up] = VEGETATION_GAIN
    result[ndvi_down] = VEGETATION_LOSS
    result[ndvi_down & mndwi_up] = FLOODING
    result[ndvi_down & ndbi_up] = URBAN_CONVERSION

    # NaN pixels → unchanged
    nan_mask = ~np.isfinite(ndvi_delta) | ~np.isfinite(ndbi_delta) | ~np.isfinite(mndwi_delta)
    result[nan_mask] = UNCHANGED

    return result
