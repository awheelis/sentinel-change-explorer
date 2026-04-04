# Time-Series Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an integrated temporal trend chart with anomaly detection below the folium map, fetching all available scenes across the before→after date span at reduced resolution.

**Architecture:** New `src/timeseries.py` module handles scene fetching and anomaly statistics. New `time_series_chart()` in `src/visualization.py` renders the matplotlib chart. `app.py` wires the time-series section between the map and statistics panels. All scene fetches use existing `search_scenes()` + `load_bands()` at 60m resolution with `ThreadPoolExecutor` for parallelism.

**Tech Stack:** numpy, matplotlib, pystac-client, rasterio (all existing dependencies)

---

### Task 1: `compute_anomalies()` — rolling statistics and anomaly detection

**Files:**
- Create: `src/timeseries.py`
- Create: `tests/unit/test_timeseries.py`

This task builds the pure-computation function that takes a list of scene measurement dicts and returns rolling stats, trend, and anomaly flags. No I/O — just math on lists of floats.

- [ ] **Step 1: Write failing tests for `compute_anomalies()`**

```python
# tests/unit/test_timeseries.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_timeseries.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.timeseries'`

- [ ] **Step 3: Implement `compute_anomalies()` in `src/timeseries.py`**

```python
# src/timeseries.py
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

    # Max jump (scene-to-scene) among usable scenes
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

    # Anomaly detection (only if >= 4 usable scenes)
    if len(usable_indices) >= 4:
        for i in usable_indices:
            if rolling_mean[i] is not None and rolling_std[i] is not None and rolling_std[i] > 0:
                deviation = abs(series[i]["mean_index"] - rolling_mean[i])
                if deviation > sigma_threshold * rolling_std[i]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_timeseries.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/timeseries.py tests/unit/test_timeseries.py
git commit -m "feat: add compute_anomalies() for time-series rolling stats and anomaly detection"
```

---

### Task 2: `fetch_time_series()` — data pipeline with caching

**Files:**
- Modify: `src/timeseries.py`
- Create: `tests/unit/test_fetch_timeseries.py`

This task adds the I/O function that queries STAC, loads bands at 60m, computes the index, and returns the scene measurement list. Uses mocked tests to avoid network calls.

- [ ] **Step 1: Write failing tests for `fetch_time_series()`**

```python
# tests/unit/test_fetch_timeseries.py
"""Tests for fetch_time_series() data pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import numpy as np
import pytest

from src.timeseries import fetch_time_series, INDEX_BANDS


class TestIndexBands:
    def test_ndvi_bands(self):
        assert INDEX_BANDS["ndvi"] == ["nir", "red"]

    def test_ndbi_bands(self):
        assert INDEX_BANDS["ndbi"] == ["swir16", "nir"]

    def test_mndwi_bands(self):
        assert INDEX_BANDS["mndwi"] == ["green", "swir16"]

    def test_evi_bands(self):
        assert INDEX_BANDS["evi"] == ["nir", "red", "blue"]


def _mock_scene(scene_id, datetime_str, cloud_cover=5.0):
    """Create a fake scene dict matching search_scenes() output."""
    return {
        "id": scene_id,
        "cloud_cover": cloud_cover,
        "datetime": datetime_str,
        "assets": {
            "red": f"https://fake/{scene_id}/red.tif",
            "green": f"https://fake/{scene_id}/green.tif",
            "blue": f"https://fake/{scene_id}/blue.tif",
            "nir": f"https://fake/{scene_id}/nir.tif",
            "swir16": f"https://fake/{scene_id}/swir16.tif",
            "scl": f"https://fake/{scene_id}/scl.tif",
        },
        "bbox": [-115.2, 36.1, -115.15, 36.15],
        "sun_elevation": 60.0,
    }


def _mock_bands(band_keys, value=3000):
    """Create fake band arrays matching load_bands() output."""
    shape = (50, 50)
    return {k: np.full(shape, value, dtype=np.uint16) for k in band_keys}


class TestFetchTimeSeries:
    @patch("src.timeseries.load_bands")
    @patch("src.timeseries.search_scenes")
    def test_returns_sorted_by_datetime(self, mock_search, mock_load):
        """Results should be sorted chronologically, not by cloud cover."""
        mock_search.return_value = [
            _mock_scene("s2", "2023-07-15T00:00:00Z", cloud_cover=3.0),
            _mock_scene("s1", "2023-06-15T00:00:00Z", cloud_cover=8.0),
            _mock_scene("s3", "2023-08-15T00:00:00Z", cloud_cover=5.0),
        ]
        mock_load.side_effect = lambda **kw: _mock_bands(kw["band_keys"])

        result = fetch_time_series(
            bbox=(-115.2, 36.1, -115.15, 36.15),
            date_span="2023-06-01/2023-09-01",
            index_name="ndvi",
            max_cloud_cover=20.0,
            apply_scl_mask=False,
        )

        dates = [r["datetime"] for r in result]
        assert dates == sorted(dates)
        assert len(result) == 3

    @patch("src.timeseries.load_bands")
    @patch("src.timeseries.search_scenes")
    def test_correct_bands_loaded_for_ndbi(self, mock_search, mock_load):
        """NDBI should load swir16 and nir only (+ scl if masking enabled)."""
        mock_search.return_value = [_mock_scene("s1", "2023-06-15T00:00:00Z")]
        mock_load.side_effect = lambda **kw: _mock_bands(kw["band_keys"])

        fetch_time_series(
            bbox=(-115.2, 36.1, -115.15, 36.15),
            date_span="2023-06-01/2023-07-01",
            index_name="ndbi",
            max_cloud_cover=20.0,
            apply_scl_mask=False,
        )

        call_kwargs = mock_load.call_args[1]
        assert set(call_kwargs["band_keys"]) == {"swir16", "nir"}

    @patch("src.timeseries.load_bands")
    @patch("src.timeseries.search_scenes")
    def test_scl_band_added_when_masking_enabled(self, mock_search, mock_load):
        """When apply_scl_mask=True, scl should be in the band list."""
        mock_search.return_value = [_mock_scene("s1", "2023-06-15T00:00:00Z")]

        def _load_with_scl(**kw):
            bands = _mock_bands(kw["band_keys"])
            if "scl" in kw["band_keys"]:
                # SCL value 4 = vegetation (not masked)
                bands["scl"] = np.full((50, 50), 4, dtype=np.uint16)
            return bands

        mock_load.side_effect = _load_with_scl

        fetch_time_series(
            bbox=(-115.2, 36.1, -115.15, 36.15),
            date_span="2023-06-01/2023-07-01",
            index_name="ndvi",
            max_cloud_cover=20.0,
            apply_scl_mask=True,
        )

        call_kwargs = mock_load.call_args[1]
        assert "scl" in call_kwargs["band_keys"]

    @patch("src.timeseries.load_bands")
    @patch("src.timeseries.search_scenes")
    def test_target_res_is_60(self, mock_search, mock_load):
        """Time-series should fetch at 60m resolution for speed."""
        mock_search.return_value = [_mock_scene("s1", "2023-06-15T00:00:00Z")]
        mock_load.side_effect = lambda **kw: _mock_bands(kw["band_keys"])

        fetch_time_series(
            bbox=(-115.2, 36.1, -115.15, 36.15),
            date_span="2023-06-01/2023-07-01",
            index_name="ndvi",
            max_cloud_cover=20.0,
            apply_scl_mask=False,
        )

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["target_res"] == 60

    @patch("src.timeseries.load_bands")
    @patch("src.timeseries.search_scenes")
    def test_mean_index_is_computed(self, mock_search, mock_load):
        """Each scene dict should have a finite mean_index value."""
        mock_search.return_value = [_mock_scene("s1", "2023-06-15T00:00:00Z")]
        mock_load.side_effect = lambda **kw: _mock_bands(kw["band_keys"])

        result = fetch_time_series(
            bbox=(-115.2, 36.1, -115.15, 36.15),
            date_span="2023-06-01/2023-07-01",
            index_name="ndvi",
            max_cloud_cover=20.0,
            apply_scl_mask=False,
        )

        assert len(result) == 1
        assert np.isfinite(result[0]["mean_index"])
        assert 0 <= result[0]["valid_pixel_pct"] <= 100

    @patch("src.timeseries.search_scenes")
    def test_no_scenes_returns_empty_list(self, mock_search):
        """If STAC returns no scenes, return an empty list."""
        mock_search.return_value = []

        result = fetch_time_series(
            bbox=(-115.2, 36.1, -115.15, 36.15),
            date_span="2023-06-01/2023-07-01",
            index_name="ndvi",
            max_cloud_cover=20.0,
            apply_scl_mask=False,
        )

        assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_fetch_timeseries.py -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_time_series' from 'src.timeseries'`

- [ ] **Step 3: Implement `fetch_time_series()` and `INDEX_BANDS` in `src/timeseries.py`**

Add the following to the top of `src/timeseries.py`, after the existing imports:

```python
# Add these imports to the top of src/timeseries.py
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.indices import compute_evi, compute_mndwi, compute_ndbi, compute_ndvi
from src.masking import apply_mask, build_scl_mask
from src.sentinel import load_bands, search_scenes

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_fetch_timeseries.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `python -m pytest tests/unit/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/timeseries.py tests/unit/test_fetch_timeseries.py
git commit -m "feat: add fetch_time_series() data pipeline with caching and parallel scene loading"
```

---

### Task 3: `time_series_chart()` — matplotlib visualization

**Files:**
- Modify: `src/visualization.py`
- Modify: `tests/unit/test_visualization.py`

- [ ] **Step 1: Write failing tests for `time_series_chart()`**

Add to `tests/unit/test_visualization.py`:

```python
# Add these imports at the top of the file:
from src.visualization import time_series_chart


def _make_ts_series(n=8):
    """Create a synthetic time-series for chart testing."""
    values = [0.5, 0.52, 0.48, 0.51, 0.15, 0.49, 0.53, 0.50][:n]
    series = []
    for i, v in enumerate(values):
        series.append({
            "datetime": f"2023-{(i + 1):02d}-15T00:00:00Z",
            "scene_id": f"scene_{i}",
            "mean_index": v,
            "cloud_cover": 5.0,
            "valid_pixel_pct": 100.0 if i != 3 else 20.0,
        })
    return series


def _make_ts_anomalies(n=8):
    """Create matching anomaly results for chart testing."""
    return {
        "rolling_mean": [0.5, 0.51, 0.50, None, 0.50, 0.49, 0.51, 0.50][:n],
        "rolling_std": [0.01, 0.02, 0.02, None, 0.02, 0.02, 0.02, 0.01][:n],
        "is_anomaly": [False, False, False, False, True, False, False, False][:n],
        "is_cloudy": [False, False, False, True, False, False, False, False][:n],
        "trend_slope": -0.01,
        "trend_direction": "stable",
        "volatility": 0.045,
        "anomaly_count": 1,
        "max_jump": 0.36,
        "max_jump_date": "2023-05-15T00:00:00Z",
    }


class TestTimeSeriesChart:
    def test_returns_figure(self):
        fig = time_series_chart(
            _make_ts_series(), _make_ts_anomalies(),
            index_name="ndvi",
            before_date="2023-01-01",
            after_date="2023-08-30",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_axes_with_data(self):
        fig = time_series_chart(
            _make_ts_series(), _make_ts_anomalies(),
            index_name="ndvi",
            before_date="2023-01-01",
            after_date="2023-08-30",
        )
        ax = fig.axes[0]
        # Should have at least the main data line
        assert len(ax.get_lines()) >= 1

    def test_title_contains_index_name(self):
        fig = time_series_chart(
            _make_ts_series(), _make_ts_anomalies(),
            index_name="mndwi",
            before_date="2023-01-01",
            after_date="2023-08-30",
        )
        ax = fig.axes[0]
        assert "MNDWI" in ax.get_title()

    def test_works_with_all_index_names(self):
        """Chart should render for every supported index."""
        for idx in ("ndvi", "ndbi", "mndwi", "evi"):
            fig = time_series_chart(
                _make_ts_series(5), _make_ts_anomalies(5),
                index_name=idx,
                before_date="2023-01-01",
                after_date="2023-05-30",
            )
            assert isinstance(fig, matplotlib.figure.Figure)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_visualization.py::TestTimeSeriesChart -v`
Expected: FAIL — `ImportError: cannot import name 'time_series_chart'`

- [ ] **Step 3: Implement `time_series_chart()` in `src/visualization.py`**

Add at the end of `src/visualization.py`:

```python
# ── Time-series temporal trend chart ──────────────────────────────────

_INDEX_COLORS: dict[str, str] = {
    "ndvi": "#4CAF50",
    "ndbi": "#FF9800",
    "mndwi": "#2196F3",
    "evi": "#2E7D32",
}


def time_series_chart(
    series: list[dict],
    anomalies: dict,
    index_name: str,
    before_date: str,
    after_date: str,
) -> matplotlib.figure.Figure:
    """Create a time-series trend chart with rolling stats and anomaly markers.

    Args:
        series: List of scene dicts with datetime, mean_index, valid_pixel_pct.
        anomalies: Dict from compute_anomalies() with rolling_mean, rolling_std,
            is_anomaly, is_cloudy.
        index_name: Index key ("ndvi", "ndbi", "mndwi", "evi").
        before_date: Before reference date string (YYYY-MM-DD) for vertical marker.
        after_date: After reference date string (YYYY-MM-DD) for vertical marker.

    Returns:
        matplotlib Figure with the trend chart.
    """
    from datetime import datetime as dt

    color = _INDEX_COLORS.get(index_name, "#4CAF50")
    display_name = index_name.upper()

    dates = []
    for s in series:
        dates.append(dt.fromisoformat(s["datetime"].replace("Z", "+00:00")))
    values = [s["mean_index"] for s in series]

    is_cloudy = anomalies["is_cloudy"]
    is_anomaly = anomalies["is_anomaly"]
    rolling_mean = anomalies["rolling_mean"]
    rolling_std = anomalies["rolling_std"]

    # Separate usable vs cloudy points
    usable_dates = [d for d, c in zip(dates, is_cloudy) if not c]
    usable_vals = [v for v, c in zip(values, is_cloudy) if not c]
    cloudy_dates = [d for d, c in zip(dates, is_cloudy) if c]
    cloudy_vals = [v for v, c in zip(values, is_cloudy) if c]

    # Anomaly points
    anom_dates = [d for d, a, c in zip(dates, is_anomaly, is_cloudy) if a and not c]
    anom_vals = [v for v, a, c in zip(values, is_anomaly, is_cloudy) if a and not c]

    # Rolling mean/std for usable scenes
    rm_dates = [d for d, rm, c in zip(dates, rolling_mean, is_cloudy) if rm is not None and not c]
    rm_vals = [rm for rm, c in zip(rolling_mean, is_cloudy) if rm is not None and not c]
    rs_vals = [rs for rs, c in zip(rolling_std, is_cloudy) if rs is not None and not c]

    fig, ax = plt.subplots(figsize=(12, 4))

    # ±2σ confidence band
    if rm_vals and rs_vals:
        upper = [m + 2 * s for m, s in zip(rm_vals, rs_vals)]
        lower = [m - 2 * s for m, s in zip(rm_vals, rs_vals)]
        ax.fill_between(rm_dates, lower, upper, alpha=0.2, color=color, label="\u00b12\u03c3 band")

    # Rolling mean line
    if rm_vals:
        ax.plot(rm_dates, rm_vals, color=color, alpha=0.5, linestyle="--", linewidth=1.5, label="Rolling mean")

    # Main data line (usable scenes)
    if usable_dates:
        ax.plot(usable_dates, usable_vals, color=color, linewidth=2.5, marker="o", markersize=5, label=f"Mean {display_name}", zorder=3)

    # Cloudy scenes (gray hollow dots, not connected)
    if cloudy_dates:
        ax.scatter(cloudy_dates, cloudy_vals, facecolors="none", edgecolors="#888888", linewidths=2, s=60, zorder=4, linestyle="--", label="Cloudy scene")

    # Anomaly markers (red, larger)
    if anom_dates:
        ax.scatter(anom_dates, anom_vals, color="#E53935", s=120, zorder=5, label="Anomaly")
        # Annotate with sigma deviation
        for ad, av in zip(anom_dates, anom_vals):
            idx = dates.index(ad)
            rm = rolling_mean[idx]
            rs = rolling_std[idx]
            if rm is not None and rs is not None and rs > 0:
                sigma_dev = (av - rm) / rs
                ax.annotate(
                    f"{sigma_dev:+.1f}\u03c3",
                    (ad, av),
                    textcoords="offset points",
                    xytext=(10, -10),
                    fontsize=9,
                    color="#E53935",
                )

    # Before/after reference lines
    before_dt = dt.fromisoformat(before_date) if "T" not in before_date else dt.fromisoformat(before_date)
    after_dt = dt.fromisoformat(after_date) if "T" not in after_date else dt.fromisoformat(after_date)
    ax.axvline(before_dt, color="#888888", linestyle="--", linewidth=1.5)
    ax.axvline(after_dt, color="#888888", linestyle="--", linewidth=1.5)
    y_top = ax.get_ylim()[1]
    ax.text(before_dt, y_top, f" Before", fontsize=9, color="#aaaaaa", va="top")
    ax.text(after_dt, y_top, f" After", fontsize=9, color="#aaaaaa", va="top")

    # Labels and title
    n_scenes = len(series)
    date_strs = [d.strftime("%b %Y") for d in dates]
    start_label = date_strs[0] if date_strs else ""
    end_label = date_strs[-1] if date_strs else ""
    ax.set_title(f"{display_name} Temporal Trend \u2014 {start_label} to {end_label} ({n_scenes} scenes)")
    ax.set_ylabel(f"Mean {display_name}")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_visualization.py::TestTimeSeriesChart -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full visualization test suite to verify no regressions**

Run: `python -m pytest tests/unit/test_visualization.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/visualization.py tests/unit/test_visualization.py
git commit -m "feat: add time_series_chart() for temporal trend visualization with anomaly markers"
```

---

### Task 4: `format_summary_caption()` — summary text builder

**Files:**
- Modify: `src/timeseries.py`
- Modify: `tests/unit/test_timeseries.py`

- [ ] **Step 1: Write failing tests for `format_summary_caption()`**

Add to `tests/unit/test_timeseries.py`:

```python
from src.timeseries import compute_anomalies, format_summary_caption


class TestFormatSummaryCaption:
    def test_increasing_trend(self):
        anomalies = {
            "trend_direction": "increasing",
            "trend_slope": 0.032,
            "volatility": 0.045,
            "anomaly_count": 2,
            "is_cloudy": [False, False, True, False, False],
        }
        caption = format_summary_caption(anomalies)
        assert "\u2191" in caption  # ↑
        assert "increasing" in caption
        assert "+0.032" in caption
        assert "0.045" in caption
        assert "2 anomalies" in caption
        assert "1 scene excluded" in caption

    def test_decreasing_trend(self):
        anomalies = {
            "trend_direction": "decreasing",
            "trend_slope": -0.018,
            "volatility": 0.03,
            "anomaly_count": 0,
            "is_cloudy": [False, False, False],
        }
        caption = format_summary_caption(anomalies)
        assert "\u2193" in caption  # ↓
        assert "decreasing" in caption

    def test_stable_trend(self):
        anomalies = {
            "trend_direction": "stable",
            "trend_slope": 0.001,
            "volatility": 0.01,
            "anomaly_count": 0,
            "is_cloudy": [False, False],
        }
        caption = format_summary_caption(anomalies)
        assert "\u2192" in caption  # →

    def test_no_cloudy_scenes(self):
        anomalies = {
            "trend_direction": "stable",
            "trend_slope": 0.001,
            "volatility": 0.01,
            "anomaly_count": 0,
            "is_cloudy": [False, False, False],
        }
        caption = format_summary_caption(anomalies)
        assert "excluded" not in caption
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_timeseries.py::TestFormatSummaryCaption -v`
Expected: FAIL — `ImportError: cannot import name 'format_summary_caption'`

- [ ] **Step 3: Implement `format_summary_caption()` in `src/timeseries.py`**

Add at the end of `src/timeseries.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_timeseries.py::TestFormatSummaryCaption -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/timeseries.py tests/unit/test_timeseries.py
git commit -m "feat: add format_summary_caption() for time-series summary text"
```

---

### Task 5: Wire time-series into `app.py`

**Files:**
- Modify: `app.py`

This task integrates the time-series chart into the main app, placing it between the Overture Maps panel (or folium map if Overture is off) and Panel D (Summary Statistics).

- [ ] **Step 1: Add imports to `app.py`**

Add to the import block at the top of `app.py` (after the existing `src.` imports around line 37-39):

```python
from src.timeseries import compute_anomalies, fetch_time_series, format_summary_caption
from src.visualization import (
    build_folium_map,
    change_histogram,
    classification_to_rgba,
    downscale_array,
    google_maps_url,
    index_to_rgba,
    label_image,
    time_series_chart,
    true_color_image,
)
```

(This replaces the existing `from src.visualization import (...)` block to add `time_series_chart`.)

- [ ] **Step 2: Add time-series section after the Overture panel**

Insert the following block in `app.py` between the Panel C (Overture Maps Context) section ending around line 664 and the Panel D (Summary Statistics) section starting at line 667. The exact insertion point is after `st_folium(panel_c_map, width="100%", height=500)` and before `st.subheader("Panel D — Summary Statistics")`:

```python
    # ── Temporal Trend ──────────────────────────────────────────────────────
    ts_cache_key = f"{bbox}|{before_range}|{after_range}|{max_cloud}|ts|{index_choice}|{st.session_state.get('apply_scl_mask', True)}"
    ts_cached = st.session_state["_results"].get(ts_cache_key)

    if ts_cached is not None:
        ts_series = ts_cached["series"]
        ts_anomalies = ts_cached["anomalies"]
    else:
        ts_series = None
        ts_anomalies = None

    if ts_series is None:
        with st.status("Loading temporal trend\u2026", expanded=False) as ts_status:
            try:
                date_span = f"{before_start}/{after_end}"
                ts_series = fetch_time_series(
                    bbox=bbox,
                    date_span=date_span,
                    index_name=index_choice,
                    max_cloud_cover=max_cloud,
                    apply_scl_mask=st.session_state.get("apply_scl_mask", True),
                )
                if ts_series:
                    ts_anomalies = compute_anomalies(ts_series)
                    st.session_state["_results"][ts_cache_key] = {
                        "series": ts_series,
                        "anomalies": ts_anomalies,
                    }
                    ts_status.update(label=f"Temporal trend loaded ({len(ts_series)} scenes)", state="complete")
                else:
                    ts_status.update(label="No scenes found for temporal trend", state="complete")
            except Exception as exc:
                ts_status.update(label=f"Temporal trend failed: {exc}", state="error")
                ts_series = None
                ts_anomalies = None

    if ts_series and ts_anomalies is not None:
        st.subheader("Temporal Trend")
        fig = time_series_chart(
            ts_series,
            ts_anomalies,
            index_name=index_choice,
            before_date=str(before_start),
            after_date=str(after_end),
        )
        st.pyplot(fig)
        plt.close(fig)
        st.caption(format_summary_caption(ts_anomalies))
    elif ts_series is not None and ts_anomalies is None and len(ts_series) > 0:
        st.info("All scenes in the time-series had insufficient cloud-free coverage for temporal analysis.")
```

- [ ] **Step 3: Add `plt` import if not already present**

Check that `matplotlib.pyplot as plt` is available in `app.py`. It's imported via `src.visualization` but not directly. Add to the import block:

```python
import matplotlib.pyplot as plt
```

- [ ] **Step 4: Run the app manually to verify integration**

Run: `streamlit run app.py`

1. Select any preset location
2. Click "Analyze Change"
3. Verify: the temporal trend chart appears below the Overture panel / map
4. Verify: the summary caption appears below the chart
5. Verify: before/after reference lines appear on the chart
6. Verify: switching index in the sidebar and re-running shows a different colored chart

- [ ] **Step 5: Run all unit tests to verify no regressions**

Run: `python -m pytest tests/unit/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: integrate time-series temporal trend chart into main app"
```

---

### Task 6: Final verification and cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v --ignore=tests/e2e_visual_test.py --ignore=tests/integration/ --ignore=tests/perf/`
Expected: All unit tests PASS

- [ ] **Step 2: Verify the time-series cache works**

1. Run the app, analyze a preset
2. Note the time-series load time in the status indicator
3. Switch to another preset, then switch back
4. Second load should be near-instant (cached)

- [ ] **Step 3: Verify edge cases in the app**

1. Select a very narrow date range (e.g., same month for before and after) → should show few or no time-series scenes, graceful handling
2. Select a date range with heavy cloud cover → chart should show gray dots for cloudy scenes

- [ ] **Step 4: Commit any fixes from verification**

```bash
git add -u
git commit -m "fix: address issues found during time-series verification"
```

(Skip this step if no fixes were needed.)
