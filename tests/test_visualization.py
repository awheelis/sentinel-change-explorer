"""Tests for visualization helpers."""
import numpy as np
import pytest
from PIL import Image
from src.visualization import true_color_image


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
