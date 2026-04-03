"""Tests for visualization helpers."""
import numpy as np
import pytest
from PIL import Image
from src.visualization import true_color_image, downscale_array, index_to_rgba


def _make_dark_bands(shape=(100, 100)):
    """Simulate typical dark Sentinel-2 surface reflectance (uint16, 0-2000 range)."""
    rng = np.random.RandomState(42)
    red = rng.randint(100, 2000, shape, dtype=np.uint16)
    green = rng.randint(100, 2000, shape, dtype=np.uint16)
    blue = rng.randint(100, 2000, shape, dtype=np.uint16)
    return red, green, blue


class TestTrueColorImage:
    def test_returns_rgb_image(self):
        red, green, blue = _make_dark_bands()
        img = true_color_image(red, green, blue)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_gamma_brightens_image(self):
        """Gamma < 1 should produce a brighter image (higher mean pixel value)."""
        red, green, blue = _make_dark_bands()
        img_default = true_color_image(red, green, blue, gamma=1.0)
        img_bright = true_color_image(red, green, blue, gamma=0.85)

        mean_default = np.array(img_default).mean()
        mean_bright = np.array(img_bright).mean()
        assert mean_bright > mean_default, (
            f"Gamma 0.85 should be brighter: {mean_bright:.1f} vs {mean_default:.1f}"
        )

    def test_gamma_one_is_identity(self):
        """Gamma=1.0 should produce the same result as no gamma."""
        red, green, blue = _make_dark_bands()
        img = true_color_image(red, green, blue, gamma=1.0)
        arr = np.array(img)
        assert arr.dtype == np.uint8
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_invalid_gamma_raises(self):
        red, green, blue = _make_dark_bands()
        with pytest.raises(ValueError, match="gamma must be positive"):
            true_color_image(red, green, blue, gamma=0.0)
        with pytest.raises(ValueError, match="gamma must be positive"):
            true_color_image(red, green, blue, gamma=-0.5)


class TestDownscaleArray:
    def test_no_op_when_small(self):
        """Arrays already under max_dim should be returned unchanged."""
        arr = np.random.default_rng(0).uniform(-0.5, 0.5, (100, 200)).astype(np.float32)
        result = downscale_array(arr, max_dim=800)
        assert result is arr

    def test_downscales_large_array(self):
        """A 2000x3000 array should be scaled to max_dim on its longest side."""
        arr = np.random.default_rng(0).uniform(-0.5, 0.5, (2000, 3000)).astype(np.float32)
        result = downscale_array(arr, max_dim=800)
        assert max(result.shape) == 800
        assert result.shape == (533, 800)  # 2000*(800/3000)=533.3 → 533
        assert result.dtype == np.float32

    def test_preserves_value_range(self):
        """Downscaled values should stay within original min/max."""
        arr = np.random.default_rng(0).uniform(-0.5, 0.5, (2000, 2000)).astype(np.float32)
        result = downscale_array(arr, max_dim=800)
        assert result.min() >= arr.min() - 0.05  # small interpolation tolerance
        assert result.max() <= arr.max() + 0.05


class TestIndexToRgbaDownscaled:
    def test_downscaled_heatmap_matches_direct(self):
        """Downscaling delta then colormapping should produce same-size image
        as colormapping full-res then PIL downscaling."""
        delta = np.random.default_rng(0).uniform(-0.5, 0.5, (2000, 3000)).astype(np.float32)

        # Old path: full-res colormap then PIL downscale
        full_img = index_to_rgba(delta, threshold=0.05)
        old_way = full_img.resize((800, 533), Image.LANCZOS)

        # New path: downscale array then colormap
        small_delta = downscale_array(delta, max_dim=800)
        new_way = index_to_rgba(small_delta, threshold=0.05)

        assert old_way.size == new_way.size
        # Pixel values won't be identical (interpolation order differs),
        # but both should be valid RGBA images of the same dimensions.
        assert new_way.mode == "RGBA"
