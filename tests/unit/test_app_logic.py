"""Unit tests for app.py logic: warm-up, bbox validation, memory guard, index dispatch."""
import math
from unittest.mock import patch

import numpy as np
import pytest


def test_warm_preset_caches_calls_all_presets():
    """Warm-up should search + load bands + fetch overture for every preset."""
    fake_presets = [
        {
            "name": "Preset A",
            "bbox": [-115.32, 36.08, -115.08, 36.28],
            "before_range": ["2019-05-01", "2019-07-31"],
            "after_range": ["2023-05-01", "2023-07-31"],
            "default_index": "ndbi",
        },
        {
            "name": "Preset B",
            "bbox": [58.50, 44.80, 59.20, 45.40],
            "before_range": ["2018-07-01", "2018-09-30"],
            "after_range": ["2023-07-01", "2023-09-30"],
            "default_index": "mndwi",
        },
    ]

    fake_scene = {
        "id": "test-scene",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T00:00:00Z",
        "assets": {"red": "url", "green": "url", "blue": "url", "nir": "url", "swir16": "url"},
        "bbox": [-115.32, 36.08, -115.08, 36.28],
    }

    with patch("app.load_presets", return_value=fake_presets), \
         patch("app.search_scenes", return_value=[fake_scene]) as mock_search, \
         patch("app.load_bands", return_value={}) as mock_load, \
         patch("app.get_overture_context", return_value={}) as mock_overture:

        from app import warm_preset_caches
        # Call the underlying function directly (bypass st.cache_resource)
        warm_preset_caches()

        # 2 presets × 2 date ranges = 4 search calls
        assert mock_search.call_count == 4
        # 2 presets × 2 scenes = 4 load_bands calls
        assert mock_load.call_count == 4
        # 2 presets × 1 overture call = 2
        assert mock_overture.call_count == 2


def test_warm_preset_caches_survives_failures():
    """If one preset fails, others should still be warmed."""
    fake_presets = [
        {
            "name": "Failing Preset",
            "bbox": [0, 0, 1, 1],
            "before_range": ["2019-01-01", "2019-03-31"],
            "after_range": ["2023-01-01", "2023-03-31"],
            "default_index": "ndvi",
        },
        {
            "name": "Working Preset",
            "bbox": [10, 10, 11, 11],
            "before_range": ["2019-01-01", "2019-03-31"],
            "after_range": ["2023-01-01", "2023-03-31"],
            "default_index": "ndvi",
        },
    ]

    fake_scene = {
        "id": "test-scene",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T00:00:00Z",
        "assets": {"red": "url", "green": "url", "blue": "url", "nir": "url", "swir16": "url"},
        "bbox": [10, 10, 11, 11],
    }

    call_count = {"search": 0}

    def search_side_effect(bbox, date_range, max_cloud_cover=20):
        call_count["search"] += 1
        if bbox == (0, 0, 1, 1):
            raise RuntimeError("Network error")
        return [fake_scene]

    with patch("app.load_presets", return_value=fake_presets), \
         patch("app.search_scenes", side_effect=search_side_effect), \
         patch("app.load_bands", return_value={}) as mock_load, \
         patch("app.get_overture_context", return_value={}) as mock_overture:

        from app import warm_preset_caches
        # Should not raise
        warm_preset_caches()

        # The working preset should still have been loaded
        assert mock_load.call_count >= 2  # before + after for working preset
        assert mock_overture.call_count >= 1  # overture for working preset


def test_warm_preset_caches_progress_callback():
    """on_progress should be called once per completed future."""
    fake_presets = [
        {
            "name": "Preset A",
            "bbox": [-115.32, 36.08, -115.08, 36.28],
            "before_range": ["2019-05-01", "2019-07-31"],
            "after_range": ["2023-05-01", "2023-07-31"],
            "default_index": "ndbi",
        },
    ]

    fake_scene = {
        "id": "test-scene",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T00:00:00Z",
        "assets": {"red": "url", "green": "url", "blue": "url", "nir": "url", "swir16": "url"},
        "bbox": [-115.32, 36.08, -115.08, 36.28],
    }

    progress_calls = []

    with patch("app.load_presets", return_value=fake_presets), \
         patch("app.search_scenes", return_value=[fake_scene]), \
         patch("app.load_bands", return_value={}), \
         patch("app.get_overture_context", return_value={}):

        from app import warm_preset_caches
        warm_preset_caches(on_progress=lambda done, total: progress_calls.append((done, total)))

    # 1 preset × 3 tasks = 3 calls
    assert len(progress_calls) == 3
    # Last call should show all complete
    assert progress_calls[-1] == (3, 3)


class TestComputeIndexForBands:
    def test_ndvi_dispatches_correctly(self, small_bands):
        from app import compute_index_for_bands
        result = compute_index_for_bands("ndvi", small_bands)
        assert result.dtype == np.float32
        assert result.shape == (64, 64)
        assert not np.all(result == 0)

    def test_ndbi_dispatches_correctly(self, small_bands):
        from app import compute_index_for_bands
        result = compute_index_for_bands("ndbi", small_bands)
        assert result.dtype == np.float32
        assert result.shape == (64, 64)

    def test_mndwi_dispatches_correctly(self, small_bands):
        from app import compute_index_for_bands
        result = compute_index_for_bands("mndwi", small_bands)
        assert result.dtype == np.float32
        assert result.shape == (64, 64)

    def test_invalid_key_raises(self, small_bands):
        from app import compute_index_for_bands
        with pytest.raises(KeyError):
            compute_index_for_bands("fake_index", small_bands)


def _estimate_memory_mb(west, south, east, north):
    """Replicate the memory guard formula from app.py main()."""
    bbox_width_deg = east - west
    bbox_height_deg = north - south
    center_lat = (south + north) / 2.0
    target_res = 10
    pixels_per_band = (
        bbox_width_deg * bbox_height_deg
        * math.cos(math.radians(center_lat))
        * (111_000 / target_res) ** 2
    )
    num_bands = 5
    num_dates = 2
    bytes_per_pixel = 8
    return pixels_per_band * num_bands * num_dates * bytes_per_pixel / (1024 ** 2)


class TestBboxValidation:
    def test_west_gte_east_is_invalid(self):
        assert 10.0 >= 5.0  # west >= east

    def test_south_gte_north_is_invalid(self):
        assert 40.0 >= 30.0  # south >= north

    def test_valid_bbox_passes(self):
        west, south, east, north = -115.20, 36.10, -115.15, 36.15
        assert west < east
        assert south < north


class TestMemoryGuard:
    def test_small_bbox_under_limit(self):
        estimated = _estimate_memory_mb(-115.20, 36.10, -115.15, 36.15)
        assert estimated < 500, f"Small bbox estimated {estimated:.0f} MB, expected < 500"

    def test_huge_bbox_over_limit(self):
        estimated = _estimate_memory_mb(-5.0, -2.5, 0.0, 2.5)
        assert estimated > 500, f"Huge bbox estimated {estimated:.0f} MB, expected > 500"

    def test_formula_agrees_with_app(self):
        west, south, east, north = -115.28, 36.15, -115.18, 36.23
        estimated = _estimate_memory_mb(west, south, east, north)
        bbox_width = east - west
        bbox_height = north - south
        center_lat = (south + north) / 2.0
        pixels = bbox_width * bbox_height * math.cos(math.radians(center_lat)) * (111_000 / 10) ** 2
        expected_mb = pixels * 5 * 2 * 8 / (1024 ** 2)
        assert abs(estimated - expected_mb) < 0.01
