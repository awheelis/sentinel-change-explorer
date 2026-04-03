"""Shared fixtures for all test tiers."""
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Point, box, LineString


PRESETS_FILE = Path(__file__).resolve().parent.parent / "config" / "presets.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "network: requires internet access to AWS S3 / STAC / Overture")
    config.addinivalue_line("markers", "perf: performance benchmark with timing assertions")


@contextmanager
def assert_within(max_seconds: float, label: str = ""):
    """Context manager that asserts the block completes within max_seconds."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed <= max_seconds, (
        f"{label or 'Block'} took {elapsed:.1f}s, expected < {max_seconds}s"
    )


@pytest.fixture
def small_bands():
    """Dict of 5 synthetic uint16 64x64 arrays with realistic Sentinel-2 ranges."""
    rng = np.random.RandomState(42)
    return {
        "red": rng.randint(100, 3000, (64, 64), dtype=np.uint16),
        "green": rng.randint(100, 3000, (64, 64), dtype=np.uint16),
        "blue": rng.randint(100, 3000, (64, 64), dtype=np.uint16),
        "nir": rng.randint(500, 5000, (64, 64), dtype=np.uint16),
        "swir16": rng.randint(200, 4000, (64, 64), dtype=np.uint16),
    }


@pytest.fixture(scope="session")
def large_bands():
    """Dict of 5 synthetic uint16 2000x2000 arrays. Session-scoped to avoid reallocation."""
    rng = np.random.RandomState(123)
    return {
        "red": rng.randint(100, 3000, (2000, 2000), dtype=np.uint16),
        "green": rng.randint(100, 3000, (2000, 2000), dtype=np.uint16),
        "blue": rng.randint(100, 3000, (2000, 2000), dtype=np.uint16),
        "nir": rng.randint(500, 5000, (2000, 2000), dtype=np.uint16),
        "swir16": rng.randint(200, 4000, (2000, 2000), dtype=np.uint16),
    }


@pytest.fixture
def sample_bbox():
    """Las Vegas bbox as (west, south, east, north)."""
    return (-115.20, 36.10, -115.15, 36.15)


@pytest.fixture
def mock_scene():
    """Fake scene dict matching search_scenes() output structure."""
    return {
        "id": "S2A_MSIL2A_20230615T000000_TEST",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T10:30:00Z",
        "assets": {
            "red": "https://example.com/B04.tif",
            "green": "https://example.com/B03.tif",
            "blue": "https://example.com/B02.tif",
            "nir": "https://example.com/B08.tif",
            "swir16": "https://example.com/B11.tif",
        },
        "bbox": [-115.20, 36.10, -115.15, 36.15],
    }


@pytest.fixture
def mock_overture_context():
    """Dict with small GeoDataFrames for building/segment/place layers."""
    buildings = gpd.GeoDataFrame(
        geometry=[box(-115.19, 36.11, -115.188, 36.112) for _ in range(5)],
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        geometry=[
            LineString([(-115.19, 36.11), (-115.18, 36.12)]),
            LineString([(-115.18, 36.12), (-115.17, 36.13)]),
            LineString([(-115.17, 36.13), (-115.16, 36.14)]),
        ],
        crs="EPSG:4326",
    )
    places = gpd.GeoDataFrame(
        {"names": [{"primary": "Place A"}, {"primary": "Place B"}],
         "geometry": [Point(-115.18, 36.12), Point(-115.17, 36.13)]},
        crs="EPSG:4326",
    )
    return {"building": buildings, "segment": segments, "place": places}


@pytest.fixture
def sample_presets():
    """Load and return the actual config/presets.json."""
    with open(PRESETS_FILE) as f:
        return json.load(f)
