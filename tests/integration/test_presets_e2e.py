"""End-to-end tests for each preset in config/presets.json.

Runs the full pipeline: STAC search → band loading → index computation →
change detection → image rendering. Uses 60m resolution to keep tests fast.

Requires internet access (S3). Mark with @pytest.mark.network.
Run with: pytest tests/test_presets_e2e.py -v -s
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.sentinel import search_scenes, load_bands
from src.indices import compute_ndvi, compute_ndbi, compute_mndwi, compute_change
from src.visualization import true_color_image, index_to_rgba
from tests.conftest import assert_within

PRESETS_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "presets.json"
ALL_BAND_KEYS = ["red", "green", "blue", "nir", "swir16"]

INDEX_FN = {
    "ndvi": lambda b: compute_ndvi(b["nir"], b["red"]),
    "ndbi": lambda b: compute_ndbi(b["swir16"], b["nir"]),
    "mndwi": lambda b: compute_mndwi(b["green"], b["swir16"]),
}


def _load_presets():
    with open(PRESETS_FILE) as f:
        return json.load(f)


PRESETS = _load_presets()


@pytest.mark.network
@pytest.mark.parametrize(
    "preset",
    PRESETS,
    ids=[p["name"] for p in PRESETS],
)
def test_preset_full_pipeline(preset):
    """Full pipeline for a preset: search, load, compute, render."""
    bbox = tuple(preset["bbox"])
    before_range = f"{preset['before_range'][0]}/{preset['before_range'][1]}"
    after_range = f"{preset['after_range'][0]}/{preset['after_range'][1]}"
    index_key = preset.get("default_index", "ndvi")

    with assert_within(90, f"full pipeline for {preset['name']}"):
        # ── Search ───────────────────────────────────────────────────────────
        before_scenes = search_scenes(bbox=bbox, date_range=before_range, max_cloud_cover=50)
        assert len(before_scenes) > 0, f"No before scenes for {preset['name']}"
        print(f"\n  Before: {before_scenes[0]['id']} ({before_scenes[0]['cloud_cover']:.1f}% cloud)")

        after_scenes = search_scenes(bbox=bbox, date_range=after_range, max_cloud_cover=50)
        assert len(after_scenes) > 0, f"No after scenes for {preset['name']}"
        print(f"  After: {after_scenes[0]['id']} ({after_scenes[0]['cloud_cover']:.1f}% cloud)")

        # ── Load bands at 60m (fast) ─────────────────────────────────────────
        before_bands = load_bands(
            scene=before_scenes[0], bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=60,
        )
        after_bands = load_bands(
            scene=after_scenes[0], bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=60,
        )

        # All bands present with matching shapes
        for label, bands in [("before", before_bands), ("after", after_bands)]:
            assert set(bands.keys()) >= set(ALL_BAND_KEYS), f"{label} missing bands"
            shapes = [arr.shape for arr in bands.values()]
            assert len(set(shapes)) == 1, f"{label} band shapes mismatch: {shapes}"
            for key, arr in bands.items():
                assert arr.ndim == 2, f"{label}/{key} should be 2D"
                assert arr.dtype == np.uint16, f"{label}/{key} expected uint16, got {arr.dtype}"
        print(f"  Band shape: {before_bands['red'].shape}")

        # ── Compute index ────────────────────────────────────────────────────
        compute_fn = INDEX_FN[index_key]
        before_index = compute_fn(before_bands)
        after_index = compute_fn(after_bands)

        for label, idx in [("before_index", before_index), ("after_index", after_index)]:
            assert idx.dtype == np.float32, f"{label} dtype should be float32"
            assert idx.min() >= -1.0, f"{label} min {idx.min()} < -1"
            assert idx.max() <= 1.0, f"{label} max {idx.max()} > 1"
            assert not np.any(np.isnan(idx)), f"{label} contains NaN"
            assert not np.any(np.isinf(idx)), f"{label} contains Inf"

        # ── Compute change ───────────────────────────────────────────────────
        delta = compute_change(before=before_index, after=after_index)
        assert delta.shape == before_index.shape, "Delta shape mismatch"
        assert delta.dtype == np.float32, "Delta should be float32"
        assert not np.any(np.isnan(delta)), "Delta contains NaN"
        assert not np.any(np.isinf(delta)), "Delta contains Inf"
        print(f"  Delta range: [{delta.min():.3f}, {delta.max():.3f}]")

        # ── Render images ────────────────────────────────────────────────────
        before_img = true_color_image(
            before_bands["red"], before_bands["green"], before_bands["blue"],
        )
        assert isinstance(before_img, Image.Image)
        assert before_img.mode == "RGB"

        after_img = true_color_image(
            after_bands["red"], after_bands["green"], after_bands["blue"],
        )
        assert isinstance(after_img, Image.Image)
        assert after_img.mode == "RGB"

        heatmap_img = index_to_rgba(delta, threshold=0.05)
        assert isinstance(heatmap_img, Image.Image)
        assert heatmap_img.mode == "RGBA"

        print(f"  Images: before={before_img.size}, after={after_img.size}, heatmap={heatmap_img.size}")
        print(f"  PASSED: {preset['name']}")


@pytest.mark.network
@pytest.mark.parametrize("preset", PRESETS, ids=[p["name"] for p in PRESETS])
def test_preset_computation_phase_fast(preset):
    """Given cached bands, compute + render should be fast."""
    bbox = tuple(preset["bbox"])
    before_range = f"{preset['before_range'][0]}/{preset['before_range'][1]}"
    after_range = f"{preset['after_range'][0]}/{preset['after_range'][1]}"
    index_key = preset.get("default_index", "ndvi")

    before_scenes = search_scenes(bbox=bbox, date_range=before_range, max_cloud_cover=50)
    after_scenes = search_scenes(bbox=bbox, date_range=after_range, max_cloud_cover=50)
    assert before_scenes and after_scenes

    before_bands = load_bands(scene=before_scenes[0], bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=60)
    after_bands = load_bands(scene=after_scenes[0], bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=60)

    compute_fn = INDEX_FN[index_key]
    with assert_within(5, f"compute phase for {preset['name']}"):
        before_index = compute_fn(before_bands)
        after_index = compute_fn(after_bands)
        delta = compute_change(before=before_index, after=after_index)
        before_img = true_color_image(before_bands["red"], before_bands["green"], before_bands["blue"])
        after_img = true_color_image(after_bands["red"], after_bands["green"], after_bands["blue"])
        heatmap_img = index_to_rgba(delta, threshold=0.05)

    assert isinstance(before_img, Image.Image)
    assert isinstance(heatmap_img, Image.Image)
