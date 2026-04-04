"""Tests for multi-index change classification."""
from __future__ import annotations

import numpy as np
import pytest

from src.indices import classify_change

# Category constants (must match implementation)
UNCHANGED = 0
URBAN_CONVERSION = 1
VEGETATION_LOSS = 2
FLOODING = 3
VEGETATION_GAIN = 4


class TestClassifyChange:
    """Tests for classify_change()."""

    def _make_deltas(self, size=(10, 10), ndvi=0.0, ndbi=0.0, mndwi=0.0):
        """Helper: create uniform delta arrays."""
        return (
            np.full(size, ndvi, dtype=np.float32),
            np.full(size, ndbi, dtype=np.float32),
            np.full(size, mndwi, dtype=np.float32),
        )

    def test_urban_conversion(self):
        """NDVI down AND NDBI up -> Urban Conversion."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.15, mndwi=0.0)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == URBAN_CONVERSION)

    def test_vegetation_loss(self):
        """NDVI down AND MNDWI stable -> Vegetation Loss."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.0, mndwi=0.0)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == VEGETATION_LOSS)

    def test_flooding(self):
        """MNDWI up AND NDVI down -> Flooding."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.0, mndwi=0.2)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == FLOODING)

    def test_vegetation_gain(self):
        """NDVI up -> Vegetation Gain."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=0.2, ndbi=0.0, mndwi=0.0)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == VEGETATION_GAIN)

    def test_unchanged(self):
        """Small deltas -> Unchanged."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=0.01, ndbi=-0.01, mndwi=0.02)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == UNCHANGED)

    def test_mixed_pixels(self):
        """Different pixels get different categories."""
        ndvi_d = np.array([[-0.2, 0.2, -0.2, 0.01]], dtype=np.float32)
        ndbi_d = np.array([[0.15, 0.0, 0.0, 0.0]], dtype=np.float32)
        mndwi_d = np.array([[0.0, 0.0, 0.2, 0.0]], dtype=np.float32)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert result[0, 0] == URBAN_CONVERSION
        assert result[0, 1] == VEGETATION_GAIN
        assert result[0, 2] == FLOODING
        assert result[0, 3] == UNCHANGED

    def test_nan_handling(self):
        """NaN pixels should be classified as UNCHANGED."""
        ndvi_d = np.array([[np.nan, -0.2]], dtype=np.float32)
        ndbi_d = np.array([[0.0, 0.15]], dtype=np.float32)
        mndwi_d = np.array([[0.0, 0.0]], dtype=np.float32)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert result[0, 0] == UNCHANGED
        assert result[0, 1] == URBAN_CONVERSION

    def test_output_dtype_and_shape(self):
        """Output should be uint8 with same shape as input."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(size=(5, 7))
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert result.shape == (5, 7)
        assert result.dtype == np.uint8

    def test_priority_urban_over_vegetation_loss(self):
        """Urban conversion takes priority over vegetation loss."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.15, mndwi=0.0)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == URBAN_CONVERSION)

    def test_priority_flooding_over_vegetation_loss(self):
        """Flooding takes priority over vegetation loss."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.0, mndwi=0.2)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == FLOODING)
