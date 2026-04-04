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

        mock_core = MagicMock()
        mock_core.geodataframe.return_value = fake_gdf

        with patch("src.overture._CACHE_DIR", tmp_path), \
             patch("src.overture._import_overture_core", return_value=mock_core):
                result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_network_failure_returns_empty(self, tmp_path):
        """If the network fetch fails, should return empty GeoDataFrame."""
        mock_core = MagicMock()
        mock_core.geodataframe.side_effect = ValueError("S3 timeout")

        with patch("src.overture._CACHE_DIR", tmp_path), \
             patch("src.overture._import_overture_core", return_value=mock_core):
                result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_empty_result_is_cached(self, tmp_path):
        """Empty results should be cached to avoid repeated slow S3 scans."""
        mock_core = MagicMock()
        mock_core.geodataframe.return_value = gpd.GeoDataFrame()

        with patch("src.overture._CACHE_DIR", tmp_path), \
             patch("src.overture._import_overture_core", return_value=mock_core):
            # First call — fetches from network
            result1 = fetch_overture_layer("building", bbox=BBOX)
            assert len(result1) == 0
            assert mock_core.geodataframe.call_count == 1

            # Second call — should hit cache, NOT call geodataframe again
            result2 = fetch_overture_layer("building", bbox=BBOX)
            assert len(result2) == 0
            assert mock_core.geodataframe.call_count == 1  # still 1


class TestGetOvertureContext:
    def test_returns_all_three_keys(self):
        """Should return dict with building, segment, place keys."""
        fake_gdf = gpd.GeoDataFrame()

        with patch("src.overture.fetch_overture_layer", return_value=fake_gdf) as mock_fetch:
            result = get_overture_context(bbox=BBOX)

        assert set(result.keys()) == {"building", "segment", "place"}
        assert mock_fetch.call_count == 3


def test_fetch_overture_layer_timeout_returns_empty():
    """A slow-but-alive connection should be killed by the timeout wrapper."""
    import time as _time
    mock_core = MagicMock()

    def _slow_fetch(*a, **kw):
        _time.sleep(30)
        return gpd.GeoDataFrame()

    mock_core.geodataframe.side_effect = _slow_fetch

    with patch("src.overture._import_overture_core", return_value=mock_core), \
         patch("src.overture._LAYER_TIMEOUT", 1):
        result = fetch_overture_layer("building", bbox=(-115.2, 36.1, -115.1, 36.2), use_cache=False)

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


def test_get_overture_context_fetches_in_parallel():
    """All three layers should be fetched concurrently."""
    import time as _time
    call_times = []

    def _track_fetch(layer, bbox, use_cache=True):
        call_times.append(_time.monotonic())
        _time.sleep(0.3)
        return gpd.GeoDataFrame()

    with patch("src.overture.fetch_overture_layer", side_effect=_track_fetch):
        result = get_overture_context(bbox=(-115.2, 36.1, -115.1, 36.2))

    assert set(result.keys()) == {"building", "segment", "place"}
    assert max(call_times) - min(call_times) < 0.2, "Layers should be fetched in parallel"


def test_fetch_overture_layer_fails_fast_on_transient_error():
    """Should fail after a single attempt on transient network errors (no retries)."""
    mock_core = MagicMock()
    mock_core.geodataframe.side_effect = ConnectionError("timeout")

    with patch("src.overture._import_overture_core", return_value=mock_core):
        result = fetch_overture_layer(
            "building",
            bbox=(-115.2, 36.1, -115.1, 36.2),
            use_cache=False,
        )
    assert mock_core.geodataframe.call_count == 1
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


def test_keyboard_interrupt_propagates():
    """KeyboardInterrupt should not be swallowed by exception handlers."""
    mock_core = MagicMock()
    mock_core.geodataframe.side_effect = KeyboardInterrupt

    with patch("src.overture._import_overture_core", return_value=mock_core):
        with pytest.raises(KeyboardInterrupt):
            fetch_overture_layer("building", bbox=(-115.2, 36.1, -115.1, 36.2), use_cache=False)
