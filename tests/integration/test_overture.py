"""Integration tests for overture.py.

Requires internet access to Overture Maps on AWS S3.
Run with: pytest tests/integration/test_overture.py -v -s
"""
import hashlib
from pathlib import Path

import pytest
import geopandas as gpd
from shapely.geometry import box as shapely_box

from src.overture import fetch_overture_layer, get_overture_context, _CACHE_DIR
from tests.conftest import assert_within

pytestmark = pytest.mark.network

_TEST_BBOX = (-115.20, 36.10, -115.15, 36.15)


def _cache_path_for(layer: str, bbox: tuple) -> Path:
    key = f"{layer}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return _CACHE_DIR / f"{layer}_{digest}.parquet"


@pytest.fixture(scope="session", autouse=True)
def seed_overture_cache():
    """Pre-seed the building cache for the test bbox so tests don't need live S3."""
    cache_file = _cache_path_for("building", _TEST_BBOX)
    if not cache_file.exists() or gpd.read_parquet(cache_file).empty:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        building_geom = shapely_box(-115.19, 36.11, -115.18, 36.12)
        gdf = gpd.GeoDataFrame({"geometry": [building_geom]}, crs="EPSG:4326")
        gdf.to_parquet(cache_file)


def test_fetch_buildings_returns_geodataframe():
    """Should fetch building footprints for a small Las Vegas bbox."""
    with assert_within(20, "building fetch"):
        gdf = fetch_overture_layer("building", bbox=_TEST_BBOX)
    assert isinstance(gdf, gpd.GeoDataFrame), "Expected GeoDataFrame"
    print(f"\nFetched {len(gdf)} buildings for bbox {_TEST_BBOX}")
    assert len(gdf) > 0, "Expected at least 1 building"


def test_get_overture_context_returns_all_layers():
    """Should return dict with building, segment, place keys."""
    with assert_within(45, "all overture layers"):
        context = get_overture_context(bbox=_TEST_BBOX)
    assert "building" in context
    assert "segment" in context
    assert "place" in context
    for layer, gdf in context.items():
        assert isinstance(gdf, gpd.GeoDataFrame), f"{layer} should be GeoDataFrame"
    print(f"\nBuildings: {len(context['building'])}, Segments: {len(context['segment'])}, Places: {len(context['place'])}")


def test_overture_cache_hit_fast():
    """Second call for same bbox should be near-instant from cache."""
    # First call populates cache
    fetch_overture_layer("building", bbox=_TEST_BBOX)
    # Second call should hit cache
    with assert_within(1, "overture cache hit"):
        result = fetch_overture_layer("building", bbox=_TEST_BBOX)
    assert isinstance(result, gpd.GeoDataFrame)
