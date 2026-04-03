"""Unit tests for GeoTIFF export in src/export.py."""
import io

import numpy as np
import pytest
import rasterio

from src.export import create_geotiff


@pytest.fixture
def sample_delta():
    """A small 10x20 float32 delta array."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 20)).astype(np.float32)


@pytest.fixture
def sample_bbox():
    """west, south, east, north bounding box."""
    return (-122.5, 37.5, -122.0, 38.0)


class TestCreateGeotiff:
    def test_returns_bytes(self, sample_delta, sample_bbox):
        result = create_geotiff(sample_delta, sample_bbox)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_geotiff_is_readable(self, sample_delta, sample_bbox):
        tiff_bytes = create_geotiff(sample_delta, sample_bbox)
        with rasterio.open(io.BytesIO(tiff_bytes)) as ds:
            assert ds.count == 1
            assert ds.height == 10
            assert ds.width == 20
            assert ds.crs.to_epsg() == 4326
            assert ds.dtypes[0] == "float32"

    def test_metadata_tags(self, sample_delta, sample_bbox):
        tiff_bytes = create_geotiff(
            sample_delta,
            sample_bbox,
            index_type="ndvi",
            before_date="2024-01-01",
            after_date="2024-06-01",
            before_scene_id="S2A_BEFORE",
            after_scene_id="S2A_AFTER",
        )
        with rasterio.open(io.BytesIO(tiff_bytes)) as ds:
            tags = ds.tags()
            assert tags["index_type"] == "ndvi"
            assert tags["before_date"] == "2024-01-01"
            assert tags["after_date"] == "2024-06-01"
            assert tags["before_scene_id"] == "S2A_BEFORE"
            assert tags["after_scene_id"] == "S2A_AFTER"

    def test_metadata_omits_empty_strings(self, sample_delta, sample_bbox):
        """Empty metadata strings should not appear as tags."""
        tiff_bytes = create_geotiff(
            sample_delta, sample_bbox, index_type="ndvi",
        )
        with rasterio.open(io.BytesIO(tiff_bytes)) as ds:
            tags = ds.tags()
            assert "index_type" in tags
            assert "before_date" not in tags

    def test_transform_matches_bbox(self, sample_delta, sample_bbox):
        tiff_bytes = create_geotiff(sample_delta, sample_bbox)
        with rasterio.open(io.BytesIO(tiff_bytes)) as ds:
            bounds = ds.bounds
            assert pytest.approx(bounds.left, abs=1e-6) == sample_bbox[0]
            assert pytest.approx(bounds.bottom, abs=1e-6) == sample_bbox[1]
            assert pytest.approx(bounds.right, abs=1e-6) == sample_bbox[2]
            assert pytest.approx(bounds.top, abs=1e-6) == sample_bbox[3]

    def test_pixel_values_roundtrip(self, sample_delta, sample_bbox):
        """Data written should match data read back."""
        tiff_bytes = create_geotiff(sample_delta, sample_bbox)
        with rasterio.open(io.BytesIO(tiff_bytes)) as ds:
            data = ds.read(1)
            np.testing.assert_array_almost_equal(data, sample_delta)
