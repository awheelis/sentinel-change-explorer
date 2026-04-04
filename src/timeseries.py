"""Time-series analysis for Sentinel-2 spectral indices.

Fetches multiple scenes across a date span, computes per-scene mean index
values, and provides rolling statistics with anomaly detection.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.indices import compute_evi, compute_mndwi, compute_ndbi, compute_ndvi
from src.masking import apply_mask, build_scl_mask
from src.sentinel import load_bands, search_scenes


_CLOUDY_THRESHOLD = 30.0  # valid_pixel_pct below this → "mostly clouded"

_TS_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "timeseries"

INDEX_BANDS: dict[str, list[str]] = {
    "ndvi": ["nir", "red"],
    "ndbi": ["swir16", "nir"],
    "mndwi": ["green", "swir16"],
    "evi": ["nir", "red", "blue"],
}

_INDEX_COMPUTE = {
    "ndvi": lambda b: compute_ndvi(b["nir"], b["red"]),
    "ndbi": lambda b: compute_ndbi(b["swir16"], b["nir"]),
    "mndwi": lambda b: compute_mndwi(b["green"], b["swir16"]),
    "evi": lambda b: compute_evi(b["nir"], b["red"], b["blue"]),
}


def fetch_time_series(
    bbox: tuple[float, float, float, float],
    date_span: str,
    index_name: str,
    max_cloud_cover: float = 20.0,
    apply_scl_mask: bool = True,
) -> list[dict]:
    """Fetch all scenes in a date span and compute per-scene mean index values.

    Args:
        bbox: Bounding box as (west, south, east, north) in WGS84.
        date_span: ISO 8601 interval, e.g. "2023-06-01/2024-06-30".
        index_name: One of "ndvi", "ndbi", "mndwi", "evi".
        max_cloud_cover: Maximum cloud cover percentage filter.
        apply_scl_mask: If True, mask clouds/shadows via SCL before computing index.

    Returns:
        List of scene dicts sorted by datetime, each with keys:
        datetime, scene_id, mean_index, cloud_cover, valid_pixel_pct.
        Returns empty list if no scenes found.
    """
    # Check disk cache
    cache_key_str = f"{bbox}|{date_span}|{index_name}|{max_cloud_cover}|{apply_scl_mask}"
    cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_path = _TS_CACHE_DIR / f"{cache_hash}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    scenes = search_scenes(bbox=bbox, date_range=date_span, max_cloud_cover=max_cloud_cover, max_items=20)
    if not scenes:
        return []

    band_keys = list(INDEX_BANDS[index_name])
    if apply_scl_mask:
        band_keys.append("scl")

    compute_fn = _INDEX_COMPUTE[index_name]

    def _process_scene(scene):
        bands = load_bands(scene=scene, bbox=bbox, band_keys=band_keys, target_res=60)

        if apply_scl_mask and "scl" in bands:
            scl_mask = build_scl_mask(bands["scl"])
            bands = apply_mask(bands, scl_mask)

        index_arr = compute_fn(bands)
        valid_mask = np.isfinite(index_arr)
        valid_count = int(valid_mask.sum())
        total_count = index_arr.size
        valid_pct = (valid_count / total_count * 100) if total_count > 0 else 0.0
        mean_val = float(np.nanmean(index_arr)) if valid_count > 0 else float("nan")

        return {
            "datetime": scene["datetime"],
            "scene_id": scene["id"],
            "mean_index": mean_val,
            "cloud_cover": scene["cloud_cover"],
            "valid_pixel_pct": valid_pct,
        }

    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_process_scene, s): s for s in scenes}
        for future in as_completed(futures):
            results.append(future.result())

    # Sort chronologically (search_scenes returns by cloud cover)
    results.sort(key=lambda r: r["datetime"])

    # Cache to disk
    _TS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f)

    return results


def compute_anomalies(
    series: list[dict],
    rolling_window: int = 3,
    sigma_threshold: float = 2.0,
    stable_slope_limit: float = 0.005,
) -> dict | None:
    """Compute rolling statistics, trend, and anomaly flags for a time-series.

    Args:
        series: List of scene dicts from fetch_time_series(), each with keys:
            datetime, scene_id, mean_index, cloud_cover, valid_pixel_pct.
        rolling_window: Number of scenes for rolling mean/std.
        sigma_threshold: Number of standard deviations for anomaly detection.
        stable_slope_limit: Absolute slope below this is "stable" (per month).

    Returns:
        Dict with rolling_mean, rolling_std, is_anomaly, is_cloudy,
        trend_slope, trend_direction, volatility, anomaly_count,
        max_jump, max_jump_date. Returns None if all scenes are cloudy.
    """
    n = len(series)
    is_cloudy = [s["valid_pixel_pct"] < _CLOUDY_THRESHOLD for s in series]

    # Usable scenes: not cloudy
    usable_indices = [i for i in range(n) if not is_cloudy[i]]
    if len(usable_indices) == 0:
        return None

    usable_values = [series[i]["mean_index"] for i in usable_indices]

    # Trend via linear regression on usable scenes (x = month offset)
    from datetime import datetime

    usable_datetimes = []
    for i in usable_indices:
        dt_str = series[i]["datetime"]
        usable_datetimes.append(datetime.fromisoformat(dt_str.replace("Z", "+00:00")))

    t0 = usable_datetimes[0]
    x_months = [(dt - t0).total_seconds() / (30.44 * 86400) for dt in usable_datetimes]
    x_arr = np.array(x_months, dtype=np.float64)
    y_arr = np.array(usable_values, dtype=np.float64)

    if len(x_arr) >= 2:
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
    else:
        slope = 0.0

    if abs(slope) < stable_slope_limit:
        trend_direction = "stable"
    elif slope > 0:
        trend_direction = "increasing"
    else:
        trend_direction = "decreasing"

    volatility = float(np.std(y_arr))

    # Max absolute jump (scene-to-scene) among usable scenes.
    # Uses absolute value so large drops are captured as well as large spikes.
    max_jump = 0.0
    max_jump_date = series[usable_indices[0]]["datetime"]
    for j in range(1, len(usable_values)):
        jump = abs(usable_values[j] - usable_values[j - 1])
        if jump > max_jump:
            max_jump = jump
            max_jump_date = series[usable_indices[j]]["datetime"]

    # Rolling mean and std (over all scenes, but using only usable neighbors)
    rolling_mean = [None] * n
    rolling_std = [None] * n
    is_anomaly = [False] * n

    for i in range(n):
        if is_cloudy[i]:
            rolling_mean[i] = None
            rolling_std[i] = None
            continue

        # Gather usable values in the rolling window centered on this scene
        half = rolling_window // 2
        window_vals = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if not is_cloudy[j]:
                window_vals.append(series[j]["mean_index"])

        if len(window_vals) > 0:
            rolling_mean[i] = float(np.mean(window_vals))
            rolling_std[i] = float(np.std(window_vals)) if len(window_vals) > 1 else 0.0
        else:
            rolling_mean[i] = series[i]["mean_index"]
            rolling_std[i] = 0.0

    # Anomaly detection (only if >= 4 usable scenes).
    # Uses rolling mean/std already computed above: is_anomaly[i] = |value - rolling_mean[i]| > sigma * rolling_std[i]
    if len(usable_indices) >= 4:
        for i in usable_indices:
            r_mean = rolling_mean[i]
            r_std = rolling_std[i]
            if r_mean is not None and r_std is not None and r_std > 0:
                deviation = abs(series[i]["mean_index"] - r_mean)
                if deviation > sigma_threshold * r_std:
                    is_anomaly[i] = True

    anomaly_count = sum(is_anomaly)

    return {
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "is_anomaly": is_anomaly,
        "is_cloudy": is_cloudy,
        "trend_slope": float(slope),
        "trend_direction": trend_direction,
        "volatility": volatility,
        "anomaly_count": anomaly_count,
        "max_jump": float(max_jump),
        "max_jump_date": max_jump_date,
    }


def format_summary_caption(anomalies: dict) -> str:
    """Format a one-line summary caption for the time-series chart.

    Args:
        anomalies: Dict from compute_anomalies().

    Returns:
        Summary string like "Trend: ↑ increasing (+0.032/month) · Volatility: 0.045 · 2 anomalies detected"
    """
    arrows = {"increasing": "\u2191", "decreasing": "\u2193", "stable": "\u2192"}
    direction = anomalies["trend_direction"]
    arrow = arrows[direction]
    slope = anomalies["trend_slope"]
    vol = anomalies["volatility"]
    anom_count = anomalies["anomaly_count"]
    cloudy_count = sum(anomalies["is_cloudy"])

    parts = [
        f"Trend: {arrow} {direction} ({slope:+.3f}/month)",
        f"Volatility: {vol:.3f}",
    ]

    if anom_count == 1:
        parts.append("1 anomaly detected")
    elif anom_count > 1:
        parts.append(f"{anom_count} anomalies detected")
    else:
        parts.append("0 anomalies")

    if cloudy_count > 0:
        label = "scene" if cloudy_count == 1 else "scenes"
        parts.append(f"{cloudy_count} {label} excluded (cloud cover)")

    return " \u00b7 ".join(parts)
