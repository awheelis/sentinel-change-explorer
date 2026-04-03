"""Unit tests for src/overture.py with mocked network dependencies."""
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import geopandas as gpd
import pytest
from shapely.geometry import box

from src.overture import fetch_overture_layer, get_overture_context, _cache_path


BBOX = (-115.20, 36.10, -115.15, 36.15)


class TestCachePath:
    def test_deterministic(self):
        """Same args should produce the same cache path."""
        p1 = _cache_path("building", BBOX)
        p2 = _cache_path("building", BBOX)
        assert p1 == p2

    def test_different_args_differ(self):
        """Different bbox or layer should produce different paths."""
        p1 = _cache_path("building", BBOX)
        p2 = _cache_path("segment", BBOX)
        p3 = _cache_path("building", (0.0, 0.0, 1.0, 1.0))
        assert p1 != p2
        assert p1 != p3


class TestFetchOvertureLayer:
    def _make_cache_file(self, tmp_path, layer, bbox, gdf):
        """Write a parquet cache file at the expected path."""
        key = f"{layer}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
        digest = hashlib.md5(key.encode()).hexdigest()
        cache_file = tmp_path / f"{layer}_{digest}.parquet"
        gdf.to_parquet(cache_file)
        return cache_file

    def test_cache_hit(self, tmp_path):
        """When cached parquet exists, should return from disk without fetching."""
        fake_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        self._make_cache_file(tmp_path, "building", BBOX, fake_gdf)

        with patch("src.overture._CACHE_DIR", tmp_path):
            result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_corrupt_cache_refetches(self, tmp_path):
        """If the cached parquet is corrupt, should fall through to network fetch."""
        key = f"building_{BBOX[0]:.4f}_{BBOX[1]:.4f}_{BBOX[2]:.4f}_{BBOX[3]:.4f}"
        digest = hashlib.md5(key.encode()).hexdigest()
        corrupt_file = tmp_path / f"building_{digest}.parquet"
        corrupt_file.write_bytes(b"not a valid parquet file")

        fake_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )

        with patch("src.overture._CACHE_DIR", tmp_path):
            mock_core_module = MagicMock()
            mock_core_module.geodataframe.return_value = fake_gdf
            mock_overturemaps = MagicMock()
            mock_overturemaps.core = mock_core_module
            with patch.dict("sys.modules", {"overturemaps": mock_overturemaps, "overturemaps.core": mock_core_module}):
                result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_network_failure_returns_empty(self, tmp_path):
        """If the network fetch fails, should return empty GeoDataFrame."""
        with patch("src.overture._CACHE_DIR", tmp_path):
            mock_core_module = MagicMock()
            mock_core_module.geodataframe.side_effect = RuntimeError("S3 timeout")
            mock_overturemaps = MagicMock()
            mock_overturemaps.core = mock_core_module
            with patch.dict("sys.modules", {"overturemaps": mock_overturemaps, "overturemaps.core": mock_core_module}):
                result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0


class TestGetOvertureContext:
    def test_returns_all_three_keys(self):
        """Should return dict with building, segment, place keys."""
        fake_gdf = gpd.GeoDataFrame()

        with patch("src.overture.fetch_overture_layer", return_value=fake_gdf) as mock_fetch:
            result = get_overture_context(bbox=BBOX)

        assert set(result.keys()) == {"building", "segment", "place"}
        assert mock_fetch.call_count == 3
