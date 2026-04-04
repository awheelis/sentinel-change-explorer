"""Tests for time-series data pipeline and anomaly detection."""
from __future__ import annotations

import numpy as np
import pytest

from src.timeseries import compute_anomalies


def _make_series(values, valid_pcts=None):
    """Helper: build a list of scene dicts from a list of mean_index values."""
    if valid_pcts is None:
        valid_pcts = [100.0] * len(values)
    series = []
    for i, (v, vp) in enumerate(zip(values, valid_pcts)):
        series.append({
            "datetime": f"2023-{(i + 1):02d}-15T00:00:00Z",
            "scene_id": f"scene_{i}",
            "mean_index": v,
            "cloud_cover": 5.0,
            "valid_pixel_pct": vp,
        })
    return series


class TestComputeAnomalies:
    def test_detects_anomaly_in_synthetic_series(self):
        """A single large drop in an otherwise stable series should be flagged."""
        values = [0.6, 0.62, 0.59, 0.61, 0.15, 0.58, 0.60, 0.61]
        result = compute_anomalies(_make_series(values))
        assert result["anomaly_count"] >= 1
        assert result["is_anomaly"][4] is True

    def test_stable_series_has_no_anomalies(self):
        """A flat series should produce zero anomalies."""
        values = [0.5, 0.51, 0.49, 0.50, 0.52, 0.48, 0.51, 0.50]
        result = compute_anomalies(_make_series(values))
        assert result["anomaly_count"] == 0

    def test_fewer_than_4_scenes_skips_anomaly_detection(self):
        """With < 4 usable scenes, all is_anomaly should be False."""
        values = [0.5, 0.1, 0.5]
        result = compute_anomalies(_make_series(values))
        assert all(a is False for a in result["is_anomaly"])
        assert result["anomaly_count"] == 0

    def test_trend_direction_increasing(self):
        values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
        result = compute_anomalies(_make_series(values))
        assert result["trend_direction"] == "increasing"
        assert result["trend_slope"] > 0

    def test_trend_direction_decreasing(self):
        values = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
        result = compute_anomalies(_make_series(values))
        assert result["trend_direction"] == "decreasing"
        assert result["trend_slope"] < 0

    def test_trend_direction_stable(self):
        values = [0.5, 0.501, 0.499, 0.5, 0.501, 0.5]
        result = compute_anomalies(_make_series(values))
        assert result["trend_direction"] == "stable"

    def test_cloudy_scenes_excluded_from_stats(self):
        """Scenes with valid_pixel_pct < 30 should not affect trend or anomalies."""
        values = [0.5, 0.51, 0.49, 0.50, 0.52, 0.51]
        valid_pcts = [100, 100, 20, 100, 100, 100]  # scene 2 is cloudy
        result = compute_anomalies(_make_series(values, valid_pcts))
        assert result["is_cloudy"][2] is True
        # Trend should still be computed from the 5 usable scenes
        assert result["trend_direction"] == "stable"

    def test_all_cloudy_returns_none(self):
        """If all scenes are cloudy, return None."""
        values = [0.5, 0.51, 0.49]
        valid_pcts = [10, 15, 20]
        result = compute_anomalies(_make_series(values, valid_pcts))
        assert result is None

    def test_max_jump(self):
        values = [0.5, 0.52, 0.80, 0.51, 0.50]
        result = compute_anomalies(_make_series(values))
        assert abs(result["max_jump"] - 0.28) < 0.01
        assert result["max_jump_date"] == "2023-03-15T00:00:00Z"

    def test_output_structure(self):
        values = [0.5, 0.51, 0.49, 0.50, 0.52]
        result = compute_anomalies(_make_series(values))
        assert "rolling_mean" in result
        assert "rolling_std" in result
        assert "is_anomaly" in result
        assert "is_cloudy" in result
        assert "trend_slope" in result
        assert "trend_direction" in result
        assert "volatility" in result
        assert "anomaly_count" in result
        assert "max_jump" in result
        assert "max_jump_date" in result
        assert len(result["rolling_mean"]) == 5
        assert len(result["is_anomaly"]) == 5
