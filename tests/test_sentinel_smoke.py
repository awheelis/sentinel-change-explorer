# tests/test_sentinel_smoke.py
"""Integration smoke test for sentinel.py.

Requires internet access and unsigned S3 access (no AWS creds needed).
Run with: pytest tests/test_sentinel_smoke.py -v -s
"""
import pytest
import numpy as np
from src.sentinel import search_scenes, load_bands


def test_search_scenes_returns_results():
    """Should find at least one Sentinel-2 scene for Las Vegas in June 2023."""
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
        assert arr.dtype in (np.uint16, np.float32, np.float64)
    print(f"\nBand shapes: { {k: v.shape for k, v in bands.items()} }")
