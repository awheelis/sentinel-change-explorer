"""Tests for fetch_time_series() data pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import numpy as np
import pytest

import src.timeseries as _ts_mod
from src.timeseries import fetch_time_series, INDEX_BANDS


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the on-disk cache to a per-test temp directory."""
    monkeypatch.setattr(_ts_mod, "_TS_CACHE_DIR", tmp_path / "ts_cache")


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
