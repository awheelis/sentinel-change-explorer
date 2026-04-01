"""Unit tests for spectral index computation in src/indices.py."""
import numpy as np
import pytest
from src.indices import compute_ndvi, compute_ndbi, compute_mndwi, compute_change


def _make_band(value: float, shape: tuple = (4, 4)) -> np.ndarray:
    """Create a constant-value float32 band array."""
    return np.full(shape, value, dtype=np.float32)


class TestNDVI:
    def test_pure_vegetation(self):
        """High NIR, low Red → NDVI close to 1."""
        nir = _make_band(0.9)
        red = _make_band(0.1)
        result = compute_ndvi(nir, red)
        np.testing.assert_allclose(result, 0.8, atol=0.01)

    def test_bare_soil(self):
        """Equal NIR and Red → NDVI = 0."""
        band = _make_band(0.5)
        result = compute_ndvi(band, band)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_no_division_by_zero(self):
        """Zero NIR + Zero Red should not raise, returns 0."""
        zeros = _make_band(0.0)
        result = compute_ndvi(zeros, zeros)
        assert not np.any(np.isnan(result)), "NaN found in NDVI output"
        assert not np.any(np.isinf(result)), "Inf found in NDVI output"

    def test_clipped_to_minus1_plus1(self):
        """NDVI values should be in [-1, 1]."""
        nir = np.random.rand(10, 10).astype(np.float32)
        red = np.random.rand(10, 10).astype(np.float32)
        result = compute_ndvi(nir, red)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_uint16_input(self):
        """Should accept uint16 arrays (raw Sentinel-2 values 0-10000)."""
        nir = np.full((4, 4), 9000, dtype=np.uint16)
        red = np.full((4, 4), 1000, dtype=np.uint16)
        result = compute_ndvi(nir, red)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, (9000 - 1000) / (9000 + 1000), atol=0.01)


class TestNDBI:
    def test_high_built_up(self):
        """High SWIR, low NIR → positive NDBI."""
        swir = _make_band(0.8)
        nir = _make_band(0.2)
        result = compute_ndbi(swir, nir)
        assert result.mean() > 0

    def test_no_division_by_zero(self):
        zeros = _make_band(0.0)
        result = compute_ndbi(zeros, zeros)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestMNDWI:
    def test_water_positive(self):
        """High Green, low SWIR → positive MNDWI (water present)."""
        green = _make_band(0.8)
        swir = _make_band(0.1)
        result = compute_mndwi(green, swir)
        assert result.mean() > 0

    def test_no_division_by_zero(self):
        zeros = _make_band(0.0)
        result = compute_mndwi(zeros, zeros)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestComputeChange:
    def test_positive_change(self):
        before = _make_band(0.3)
        after = _make_band(0.7)
        delta = compute_change(before=before, after=after)
        np.testing.assert_allclose(delta, 0.4, atol=0.01)

    def test_negative_change(self):
        before = _make_band(0.7)
        after = _make_band(0.2)
        delta = compute_change(before=before, after=after)
        np.testing.assert_allclose(delta, -0.5, atol=0.01)

    def test_shape_preserved(self):
        shape = (15, 20)
        before = np.random.rand(*shape).astype(np.float32)
        after = np.random.rand(*shape).astype(np.float32)
        delta = compute_change(before=before, after=after)
        assert delta.shape == shape
