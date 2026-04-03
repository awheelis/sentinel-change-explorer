"""Unit tests for app.py logic: warm-up, bbox validation, memory guard, index dispatch."""
from unittest.mock import patch


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
