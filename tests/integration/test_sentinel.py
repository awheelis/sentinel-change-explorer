# tests/test_sentinel_smoke.py
"""Integration smoke test for sentinel.py.

Requires internet access and unsigned S3 access (no AWS creds needed).
Run with: pytest tests/test_sentinel_smoke.py -v -s
"""
import shutil
from pathlib import Path

import numpy as np
from src.sentinel import search_scenes, load_bands
from tests.conftest import assert_within

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "bands"


def test_search_scenes_returns_results():
    """Should find at least one Sentinel-2 scene for Las Vegas in June 2023."""
    with assert_within(10, "STAC search"):
        scenes = search_scenes(
            bbox=(-115.32, 36.08, -115.08, 36.28),
            date_range="2023-06-01/2023-06-30",
            max_cloud_cover=50,
        )
    assert len(scenes) > 0, "Expected at least one scene"
    scene = scenes[0]
    assert "id" in scene
    assert "cloud_cover" in scene
    assert "assets" in scene
    assert "red" in scene["assets"] or any(
        k in scene["assets"] for k in ["red", "B04"]
    ), f"Expected red band in assets, got: {list(scene['assets'].keys())}"
    print(f"\nFound {len(scenes)} scene(s). Best: {scene['id']}, cloud: {scene['cloud_cover']:.1f}%")


def test_load_bands_returns_numpy_arrays():
    """Should load RGB bands as numpy arrays for the Las Vegas bbox."""
    bbox = (-115.32, 36.08, -115.08, 36.28)
    scenes = search_scenes(bbox=bbox, date_range="2023-06-01/2023-06-30", max_cloud_cover=50)
    assert scenes, "Need at least one scene to test band loading"

    with assert_within(30, "band loading"):
        bands = load_bands(
            scene=scenes[0],
            bbox=bbox,
            band_keys=["red", "green", "blue"],
            target_res=60,  # Use 60m for fast smoke test
        )
    assert set(bands.keys()) >= {"red", "green", "blue"}
    for k, arr in bands.items():
        assert isinstance(arr, np.ndarray), f"{k} should be ndarray"
        assert arr.ndim == 2, f"{k} should be 2D"
        assert arr.dtype == np.uint16, f"{k} expected uint16 dtype, got {arr.dtype}"
    print(f"\nBand shapes: { {k: v.shape for k, v in bands.items()} }")


def test_load_bands_returns_wgs84_aligned_arrays():
    """Bands should be reprojected to EPSG:4326 with consistent shapes."""
    bbox = (-115.32, 36.08, -115.08, 36.28)
    scenes = search_scenes(bbox=bbox, date_range="2023-06-01/2023-06-30", max_cloud_cover=50)
    assert scenes, "Need at least one scene"

    with assert_within(30, "WGS84 reprojection"):
        bands = load_bands(scene=scenes[0], bbox=bbox, band_keys=["red", "green", "blue"], target_res=60)

    shapes = [arr.shape for arr in bands.values()]
    # All bands must have the same shape after reprojection
    assert len(set(shapes)) == 1, f"Band shapes should match, got {shapes}"
    h, w = shapes[0]
    # WGS84 bbox is 0.24° wide × 0.20° tall. At 60m, width > height in pixels
    # (because longitude degrees are narrower than latitude degrees at lat 36).
    # The key assertion: reprojected shape should have a reasonable aspect ratio,
    # not be wildly stretched (which would happen if UTM pixels were naively
    # mapped to WGS84 bounds).
    aspect = w / h
    assert 0.5 < aspect < 2.0, f"Aspect ratio {aspect:.2f} looks distorted"
    print(f"\nReprojected band shape: {h}x{w}, aspect ratio: {aspect:.2f}")


def test_disk_cache_creates_and_reuses_file():
    """Second call to load_bands should reuse cached .npz file."""
    # Clear any existing cache for this test
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    bbox = (-115.32, 36.08, -115.08, 36.28)
    scenes = search_scenes(bbox=bbox, date_range="2023-06-01/2023-06-30", max_cloud_cover=50)
    assert scenes, "Need at least one scene"
    scene = scenes[0]

    # First call: cache miss, downloads from S3
    bands1 = load_bands(scene=scene, bbox=bbox, band_keys=["red", "green", "blue"], target_res=60)

    # Cache file should exist now
    npz_files = list(CACHE_DIR.glob("*.npz"))
    assert len(npz_files) == 1, f"Expected 1 cache file, found {len(npz_files)}"

    # Second call: cache hit, loads from disk
    with assert_within(1, "cache hit"):
        bands2 = load_bands(scene=scene, bbox=bbox, band_keys=["red", "green", "blue"], target_res=60)

    for key in bands1:
        np.testing.assert_array_equal(bands1[key], bands2[key])
    print(f"\nCache file: {npz_files[0].name}, size: {npz_files[0].stat().st_size / 1024:.0f} KB")
