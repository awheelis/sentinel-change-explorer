"""Unit tests for src.experimental.inference pure helpers.

These only cover the numpy / PIL visualization path and the feature-extraction
shape flow against a freshly-initialized encoder. They don't exercise HF Hub
downloads or the Streamlit panel itself — those are verified manually.

Guarded by importorskip for torch + streamlit so the suite still runs cleanly
in a minimal env without the experimental extras.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("streamlit")

from src.experimental.encoders import (  # noqa: E402
    FiveChannelResNet18,
    FiveChannelViTPatch8,
)
from src.experimental.inference import (  # noqa: E402
    _center_crop_or_pad,
    _stack_bands,
    extract_features,
    features_to_change_map,
    features_to_rgb,
)


# ── _stack_bands ─────────────────────────────────────────────────────────────


def _fake_bands_dict(h: int = 200, w: int = 200) -> dict:
    """Minimal reflectance band dict with the 5 expected keys (+ extra SCL)."""
    return {
        "red": np.full((h, w), 1500, dtype=np.uint16),
        "green": np.full((h, w), 1200, dtype=np.uint16),
        "blue": np.full((h, w), 900, dtype=np.uint16),
        "nir": np.full((h, w), 2200, dtype=np.uint16),
        "swir16": np.full((h, w), 3100, dtype=np.uint16),
        "scl": np.full((h, w), 4, dtype=np.uint8),  # should be ignored
    }


def test_stack_bands_shape_and_order():
    stacked = _stack_bands(_fake_bands_dict(120, 150))
    assert stacked.shape == (5, 120, 150)
    assert stacked.dtype == np.float32
    # Order must match _REFLECTANCE_BANDS: red, green, blue, nir, swir16
    assert stacked[0, 0, 0] == 1500  # red
    assert stacked[1, 0, 0] == 1200  # green
    assert stacked[4, 0, 0] == 3100  # swir16


def test_stack_bands_raises_on_missing_band():
    d = _fake_bands_dict()
    d.pop("nir")
    with pytest.raises(KeyError, match="nir"):
        _stack_bands(d)


def test_stack_bands_handles_ragged_shapes():
    """If two bands have slightly different shapes (reprojection rounding),
    _stack_bands should crop to the common minimum instead of crashing."""
    d = _fake_bands_dict(100, 100)
    d["nir"] = np.ones((98, 100), dtype=np.uint16)
    stacked = _stack_bands(d)
    assert stacked.shape == (5, 98, 100)


# ── _center_crop_or_pad ──────────────────────────────────────────────────────


def test_center_crop_from_larger_input():
    x = np.random.rand(5, 200, 200).astype(np.float32)
    out = _center_crop_or_pad(x, size=128)
    assert out.shape == (5, 128, 128)
    # Centered: out[0,0,0] should equal x[0, 36, 36] (start = (200-128)//2 = 36)
    np.testing.assert_array_equal(out[0, 0, 0], x[0, 36, 36])


def test_center_pad_from_smaller_input():
    """Smaller input should be reflect-padded to 128 on both axes."""
    x = np.ones((5, 50, 60), dtype=np.float32)
    out = _center_crop_or_pad(x, size=128)
    assert out.shape == (5, 128, 128)
    assert (out == 1.0).all()  # reflect-pad of ones is still all ones


def test_center_crop_passthrough_when_matching():
    x = np.arange(5 * 128 * 128, dtype=np.float32).reshape(5, 128, 128)
    out = _center_crop_or_pad(x, size=128)
    np.testing.assert_array_equal(out, x)


def test_center_crop_or_pad_asymmetric_input():
    """Rectangular input gets cropped on the long axis and padded on the short."""
    x = np.ones((5, 200, 80), dtype=np.float32)
    out = _center_crop_or_pad(x, size=128)
    assert out.shape == (5, 128, 128)


# ── features_to_rgb ──────────────────────────────────────────────────────────


def test_features_to_rgb_shape_and_dtype():
    feat = np.random.randn(512, 4, 4).astype(np.float32)
    rgb = features_to_rgb(feat, display_size=256)
    assert rgb.shape == (256, 256, 3)
    assert rgb.dtype == np.uint8


def test_features_to_rgb_not_all_zero_on_random_input():
    """With random features the projection should not collapse to uniform."""
    np.random.seed(0)
    feat = np.random.randn(512, 4, 4).astype(np.float32)
    rgb = features_to_rgb(feat)
    # Per-channel min-max is applied, so the output should cover a range
    assert rgb.std() > 1.0


def test_features_to_rgb_handles_degenerate_flat_features():
    """All-equal features give zero variance — min-max should not divide by zero."""
    feat = np.full((512, 4, 4), 3.14, dtype=np.float32)
    rgb = features_to_rgb(feat)
    assert rgb.shape == (256, 256, 3)
    assert not np.isnan(rgb).any()


# ── features_to_change_map ───────────────────────────────────────────────────


def test_change_map_shape_and_dtype():
    a = np.random.randn(512, 4, 4).astype(np.float32)
    b = np.random.randn(512, 4, 4).astype(np.float32)
    out = features_to_change_map(a, b, display_size=256)
    assert out.shape == (256, 256, 4)  # RGBA
    assert out.dtype == np.uint8


def test_change_map_identical_features_low_signal():
    """Identical features → zero distance → uniform low-end of colormap."""
    feat = np.random.randn(512, 4, 4).astype(np.float32)
    out = features_to_change_map(feat, feat.copy())
    # magma at 0 is near-black; we just check no crashes and no NaNs
    assert not np.isnan(out).any()


def test_change_map_asserts_matching_shapes():
    a = np.zeros((512, 4, 4), dtype=np.float32)
    b = np.zeros((512, 2, 2), dtype=np.float32)
    with pytest.raises(AssertionError):
        features_to_change_map(a, b)


# ── extract_features end-to-end ──────────────────────────────────────────────


def test_extract_features_end_to_end_shape():
    """Full pipeline: bands dict → encoder → [512, 4, 4]."""
    enc = FiveChannelResNet18(in_channels=5)
    enc.eval()
    model = {
        "encoder": enc,
        "mean": torch.zeros(1, 5, 1, 1),
        "std": torch.ones(1, 5, 1, 1),
        "source": "test",
    }
    feat = extract_features(_fake_bands_dict(160, 160), model)
    assert feat.shape == (512, 4, 4)
    assert feat.dtype == np.float32


def test_extract_features_smaller_aoi_gets_padded():
    """An AOI smaller than 128x128 should be padded up to the required size."""
    enc = FiveChannelResNet18(in_channels=5)
    enc.eval()
    model = {
        "encoder": enc,
        "mean": torch.zeros(1, 5, 1, 1),
        "std": torch.ones(1, 5, 1, 1),
        "source": "test",
    }
    feat = extract_features(_fake_bands_dict(60, 80), model)
    assert feat.shape == (512, 4, 4)


def test_extract_features_with_vit_tiny_encoder():
    """Gold path: ViT-Tiny/8 returns a 16x16 feature grid (256 positions).

    This is the architecture that actually ships in the app; if this breaks
    the PCA→RGB visualization degrades back to 4x4 watercolor blobs.
    """
    enc = FiveChannelViTPatch8(variant="tiny")
    enc.eval()
    model = {
        "encoder": enc,
        "mean": torch.zeros(1, 5, 1, 1),
        "std": torch.ones(1, 5, 1, 1),
        "source": "test",
        "kind": "vit_tiny_patch8",
    }
    feat = extract_features(_fake_bands_dict(160, 160), model)
    assert feat.shape == (192, 16, 16)


def test_features_to_rgb_works_on_vit_shape():
    """The PCA→RGB path must be shape-agnostic across encoder families."""
    feat = np.random.randn(192, 16, 16).astype(np.float32)
    rgb = features_to_rgb(feat, display_size=256)
    assert rgb.shape == (256, 256, 3)
    assert rgb.dtype == np.uint8
    # With 256 data points SVD has enough variance to produce a real range
    assert rgb.std() > 1.0


def test_features_to_change_map_works_on_vit_shape():
    a = np.random.randn(192, 16, 16).astype(np.float32)
    b = np.random.randn(192, 16, 16).astype(np.float32)
    out = features_to_change_map(a, b, display_size=256)
    assert out.shape == (256, 256, 4)
    assert out.dtype == np.uint8
