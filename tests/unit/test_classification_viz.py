"""Tests for classification RGBA rendering."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.indices import UNCHANGED, URBAN_CONVERSION, VEGETATION_LOSS, FLOODING, VEGETATION_GAIN
from src.visualization import classification_to_rgba

# Expected colors (R, G, B) from the spec
EXPECTED_COLORS = {
    URBAN_CONVERSION: (255, 165, 0),    # Orange
    VEGETATION_LOSS: (220, 38, 38),      # Red
    FLOODING: (59, 130, 246),            # Blue
    VEGETATION_GAIN: (34, 197, 94),      # Green
}


class TestClassificationToRgba:
    """Tests for classification_to_rgba()."""

    def test_returns_rgba_image(self):
        cats = np.array([[UNCHANGED, URBAN_CONVERSION]], dtype=np.uint8)
        img = classification_to_rgba(cats)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"
        assert img.size == (2, 1)

    def test_unchanged_is_transparent(self):
        cats = np.full((3, 3), UNCHANGED, dtype=np.uint8)
        img = classification_to_rgba(cats)
        arr = np.array(img)
        assert np.all(arr[:, :, 3] == 0)

    def test_category_colors(self):
        """Each category should produce the correct RGB color."""
        for cat, (r, g, b) in EXPECTED_COLORS.items():
            cats = np.full((1, 1), cat, dtype=np.uint8)
            img = classification_to_rgba(cats, alpha=1.0)
            arr = np.array(img)
            assert arr[0, 0, 0] == r, f"Category {cat}: red mismatch"
            assert arr[0, 0, 1] == g, f"Category {cat}: green mismatch"
            assert arr[0, 0, 2] == b, f"Category {cat}: blue mismatch"
            assert arr[0, 0, 3] == 255, f"Category {cat}: should be opaque"

    def test_alpha_parameter(self):
        cats = np.full((1, 1), VEGETATION_GAIN, dtype=np.uint8)
        img = classification_to_rgba(cats, alpha=0.5)
        arr = np.array(img)
        assert arr[0, 0, 3] == 127  # int(0.5 * 255)

    def test_mixed_categories(self):
        cats = np.array([
            [UNCHANGED, URBAN_CONVERSION],
            [FLOODING, VEGETATION_GAIN],
        ], dtype=np.uint8)
        img = classification_to_rgba(cats)
        arr = np.array(img)
        # UNCHANGED pixel should be transparent
        assert arr[0, 0, 3] == 0
        # All other pixels should be opaque (default alpha=0.7 -> 178)
        assert arr[0, 1, 3] == 178
        assert arr[1, 0, 3] == 178
        assert arr[1, 1, 3] == 178
