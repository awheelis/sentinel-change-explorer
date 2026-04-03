"""Tests for reprojection alignment between before/after scenes.

Verifies that load_bands() produces:
1. Axis-aligned WGS84 output (no rotated parallelogram with black corners)
2. Identical output shapes for different scenes covering the same bbox

Requires internet access (S3). Mark with @pytest.mark.network.
Run with: pytest tests/test_reprojection_alignment.py -v -s
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.sentinel import search_scenes, load_bands

PRESETS_FILE = Path(__file__).resolve().parent.parent / "config" / "presets.json"


def _load_presets():
    with open(PRESETS_FILE) as f:
        return json.load(f)


PRESETS = _load_presets()

LAS_VEGAS = next(p for p in PRESETS if "Las Vegas" in p["name"])


@pytest.mark.network
def test_no_nodata_border_las_vegas():
    """Reprojected bands must fill the output array with no black corners.

    The current calculate_default_transform approach produces a rotated
    parallelogram of valid data within the output array, leaving triangular
    zero-filled corners. A canonical WGS84 grid should fill edge rows/cols.
    """
    bbox = tuple(LAS_VEGAS["bbox"])
    after_range = f"{LAS_VEGAS['after_range'][0]}/{LAS_VEGAS['after_range'][1]}"

    scenes = search_scenes(bbox=bbox, date_range=after_range, max_cloud_cover=50)
    assert scenes

    bands = load_bands(scene=scenes[0], bbox=bbox, band_keys=["red"], target_res=60)
    red = bands["red"]

    h, w = red.shape
    print(f"\n  Scene: {scenes[0]['id']}")
    print(f"  Shape: {red.shape}")

    # Check that edge rows and columns have mostly valid (nonzero) data.
    # With the buggy calculate_default_transform, the first/last rows have
    # very few nonzero pixels (diamond/parallelogram pattern).
    # After the fix, edges should be >50% filled (no black triangular corners).
    top_fill = np.count_nonzero(red[0, :]) / w
    bot_fill = np.count_nonzero(red[-1, :]) / w
    left_fill = np.count_nonzero(red[:, 0]) / h
    right_fill = np.count_nonzero(red[:, -1]) / h

    print(f"  Edge fill: top={top_fill:.1%}, bot={bot_fill:.1%}, "
          f"left={left_fill:.1%}, right={right_fill:.1%}")

    min_fill = 0.50  # at least 50% of edge pixels should be valid
    assert top_fill > min_fill, f"Top row only {top_fill:.1%} filled — black corner artifact"
    assert bot_fill > min_fill, f"Bottom row only {bot_fill:.1%} filled — black corner artifact"
    assert left_fill > min_fill, f"Left col only {left_fill:.1%} filled — black corner artifact"
    assert right_fill > min_fill, f"Right col only {right_fill:.1%} filled — black corner artifact"


@pytest.mark.network
def test_before_after_shapes_match_las_vegas():
    """Before and after bands must have identical shapes for change detection."""
    bbox = tuple(LAS_VEGAS["bbox"])
    before_range = f"{LAS_VEGAS['before_range'][0]}/{LAS_VEGAS['before_range'][1]}"
    after_range = f"{LAS_VEGAS['after_range'][0]}/{LAS_VEGAS['after_range'][1]}"

    before_scenes = search_scenes(bbox=bbox, date_range=before_range, max_cloud_cover=50)
    after_scenes = search_scenes(bbox=bbox, date_range=after_range, max_cloud_cover=50)
    assert before_scenes and after_scenes

    band_keys = ["red", "green", "blue", "nir", "swir16"]
    before_bands = load_bands(scene=before_scenes[0], bbox=bbox, band_keys=band_keys, target_res=60)
    after_bands = load_bands(scene=after_scenes[0], bbox=bbox, band_keys=band_keys, target_res=60)

    before_shape = before_bands["red"].shape
    after_shape = after_bands["red"].shape

    print(f"\n  Before scene: {before_scenes[0]['id']}")
    print(f"  After scene:  {after_scenes[0]['id']}")
    print(f"  Before shape: {before_shape}")
    print(f"  After shape:  {after_shape}")

    assert before_shape == after_shape, (
        f"Before/after shapes must match for change detection. "
        f"Before={before_shape}, After={after_shape}"
    )


@pytest.mark.network
@pytest.mark.parametrize("preset", PRESETS, ids=[p["name"] for p in PRESETS])
def test_before_after_shapes_match_all_presets(preset):
    """Before and after bands must match for every preset."""
    bbox = tuple(preset["bbox"])
    before_range = f"{preset['before_range'][0]}/{preset['before_range'][1]}"
    after_range = f"{preset['after_range'][0]}/{preset['after_range'][1]}"

    before_scenes = search_scenes(bbox=bbox, date_range=before_range, max_cloud_cover=50)
    after_scenes = search_scenes(bbox=bbox, date_range=after_range, max_cloud_cover=50)
    assert before_scenes and after_scenes, f"No scenes for {preset['name']}"

    band_keys = ["red", "green", "blue"]
    before_bands = load_bands(scene=before_scenes[0], bbox=bbox, band_keys=band_keys, target_res=60)
    after_bands = load_bands(scene=after_scenes[0], bbox=bbox, band_keys=band_keys, target_res=60)

    before_shape = before_bands["red"].shape
    after_shape = after_bands["red"].shape

    print(f"\n  {preset['name']}: before={before_shape}, after={after_shape}")

    assert before_shape == after_shape, (
        f"{preset['name']}: shape mismatch. Before={before_shape}, After={after_shape}"
    )
