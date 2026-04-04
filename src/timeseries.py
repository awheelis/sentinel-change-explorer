"""Time-series analysis for Sentinel-2 spectral indices.

Fetches multiple scenes across a date span, computes per-scene mean index
values, and provides rolling statistics with anomaly detection.
"""
from __future__ import annotations

import numpy as np


_CLOUDY_THRESHOLD = 30.0  # valid_pixel_pct below this → "mostly clouded"


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
