# tests/test_overture_smoke.py
"""Integration smoke test for overture.py.

Requires internet access to Overture Maps on AWS S3.
Run with: pytest tests/test_overture_smoke.py -v -s
"""
import pytest
import geopandas as gpd
from src.overture import fetch_overture_layer, get_overture_context
from tests.conftest import assert_within


def test_fetch_buildings_returns_geodataframe():
    """Should fetch building footprints for a small Las Vegas bbox."""
    # Small bbox to keep download fast
    bbox = (-115.20, 36.10, -115.15, 36.15)
    with assert_within(15, "building fetch"):
        gdf = fetch_overture_layer("building", bbox=bbox)
    assert isinstance(gdf, gpd.GeoDataFrame), "Expected GeoDataFrame"
    print(f"\nFetched {len(gdf)} buildings for bbox {bbox}")
    # Even in Las Vegas there should be buildings in this box
    assert len(gdf) > 0, "Expected at least 1 building"


def test_get_overture_context_returns_all_layers():
    """Should return dict with building, segment, place keys."""
    bbox = (-115.20, 36.10, -115.15, 36.15)
    with assert_within(30, "all overture layers"):
        context = get_overture_context(bbox=bbox)
    assert "building" in context
    assert "segment" in context
    assert "place" in context
    for layer, gdf in context.items():
        assert isinstance(gdf, gpd.GeoDataFrame), f"{layer} should be GeoDataFrame"
    print(f"\nBuildings: {len(context['building'])}, Segments: {len(context['segment'])}, Places: {len(context['place'])}")


def test_overture_cache_hit_fast():
    """Second call for same bbox should be near-instant from cache."""
    bbox = (-115.20, 36.10, -115.15, 36.15)
    # First call populates cache
    fetch_overture_layer("building", bbox=bbox)
    # Second call should hit cache
    with assert_within(1, "overture cache hit"):
        result = fetch_overture_layer("building", bbox=bbox)
    assert isinstance(result, gpd.GeoDataFrame)
