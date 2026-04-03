"""Unit tests for src/sentinel.py with mocked network dependencies."""
from pathlib import Path
from unittest.mock import patch, MagicMock
import hashlib

import numpy as np
import pytest

from src.sentinel import search_scenes, load_bands


class TestSearchScenes:
    def test_calls_stac_client_with_correct_args(self):
        """Verify search_scenes passes bbox, datetime, and query to STAC client."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_client.search.return_value = mock_search

        with patch("src.sentinel.Client") as MockClient:
            MockClient.open.return_value = mock_client
            result = search_scenes(
                bbox=(-115.20, 36.10, -115.15, 36.15),
                date_range="2023-06-01/2023-06-30",
                max_cloud_cover=25.0,
                max_items=5,
            )

        mock_client.search.assert_called_once_with(
            collections=["sentinel-2-l2a"],
            bbox=[-115.20, 36.10, -115.15, 36.15],
            datetime="2023-06-01/2023-06-30",
            query={"eo:cloud_cover": {"lt": 25.0}},
            sortby=["+properties.eo:cloud_cover"],
            max_items=5,
        )
        assert result == []

    def test_returns_scenes_in_stac_order(self):
        """Results should preserve the STAC sort order."""
        mock_items = []
        for cloud, scene_id in [(5.0, "scene-a"), (10.0, "scene-b"), (15.0, "scene-c")]:
            item = MagicMock()
            item.id = scene_id
            item.properties = {"eo:cloud_cover": cloud}
            item.datetime = "2023-06-15T00:00:00Z"
            item.assets = {}
            item.bbox = [-115.20, 36.10, -115.15, 36.15]
            mock_items.append(item)

        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = mock_items
        mock_client.search.return_value = mock_search

        with patch("src.sentinel.Client") as MockClient:
            MockClient.open.return_value = mock_client
            results = search_scenes(
                bbox=(-115.20, 36.10, -115.15, 36.15),
                date_range="2023-06-01/2023-06-30",
            )

        assert len(results) == 3
        assert results[0]["id"] == "scene-a"
        assert results[1]["id"] == "scene-b"
        assert results[2]["id"] == "scene-c"

    def test_empty_results(self):
        """No matching scenes returns empty list."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_client.search.return_value = mock_search

        with patch("src.sentinel.Client") as MockClient:
            MockClient.open.return_value = mock_client
            result = search_scenes(
                bbox=(0, 0, 1, 1),
                date_range="2099-01-01/2099-01-02",
            )

        assert result == []


class TestLoadBands:
    def test_cache_hit_skips_rasterio(self, tmp_path, mock_scene):
        """When a cache file exists, rasterio should not be called."""
        fake_bands = {
            "red": np.ones((10, 10), dtype=np.uint16),
            "green": np.ones((10, 10), dtype=np.uint16) * 2,
        }
        bbox = (-115.20, 36.10, -115.15, 36.15)
        cache_key_str = f"v2|{mock_scene['id']}|{bbox}|{sorted(['red', 'green'])}|10"
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_path = tmp_path / f"{cache_hash}.npz"
        np.savez_compressed(cache_path, **fake_bands)

        with patch("src.sentinel._BAND_CACHE_DIR", tmp_path), \
             patch("src.sentinel.rasterio") as mock_rasterio:
            result = load_bands(
                scene=mock_scene,
                bbox=bbox,
                band_keys=["red", "green"],
                target_res=10,
            )

        mock_rasterio.open.assert_not_called()
        np.testing.assert_array_equal(result["red"], fake_bands["red"])
        np.testing.assert_array_equal(result["green"], fake_bands["green"])

    def test_missing_band_key_raises(self, mock_scene):
        """Requesting a band not in scene assets should raise KeyError."""
        with pytest.raises(KeyError, match="nonexistent_band"):
            load_bands(
                scene=mock_scene,
                bbox=(-115.20, 36.10, -115.15, 36.15),
                band_keys=["nonexistent_band"],
                target_res=10,
            )
