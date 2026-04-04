"""Unit tests for spectral index computation in src/indices.py."""
import numpy as np
import pytest
from src.indices import compute_ndvi, compute_ndbi, compute_mndwi, compute_change, compute_evi
from src.indices import compute_adaptive_threshold


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


class TestChunkedComputation:
    def test_chunked_matches_single_pass(self):
        """Chunked path should produce identical results to single-pass."""
        np.random.seed(42)
        nir = np.random.randint(0, 10000, (100, 100), dtype=np.uint16)
        red = np.random.randint(0, 10000, (100, 100), dtype=np.uint16)

        # Force single-pass by using small arrays
        result_single = compute_ndvi(nir, red)

        # Force chunked by calling internal function with small chunk size
        from src.indices import _safe_normalized_diff
        result_chunked = _safe_normalized_diff(nir, red, chunk_rows=10)

        np.testing.assert_allclose(result_single, result_chunked, atol=1e-6)

    def test_chunked_no_nan_or_inf(self):
        """Chunked computation should handle zeros safely."""
        zeros = np.zeros((100, 100), dtype=np.uint16)
        from src.indices import _safe_normalized_diff
        result = _safe_normalized_diff(zeros, zeros, chunk_rows=10)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestEVI:
    def test_known_values(self):
        """EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
        nir = _make_band(0.5)
        red = _make_band(0.1)
        blue = _make_band(0.05)
        result = compute_evi(nir, red, blue)
        expected = 2.5 * (0.5 - 0.1) / (0.5 + 6 * 0.1 - 7.5 * 0.05 + 1)
        np.testing.assert_allclose(result, expected, atol=0.01)

    def test_zero_denominator(self):
        nir = _make_band(0.0)
        red = _make_band(0.0)
        blue_zero = _make_band(1.0 / 7.5)
        result = compute_evi(nir, red, blue_zero)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_clipped_to_minus1_plus1(self):
        nir = _make_band(10000.0)
        red = _make_band(1.0)
        blue = _make_band(1.0)
        result = compute_evi(nir, red, blue)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_uint16_input(self):
        nir = np.full((4, 4), 5000, dtype=np.uint16)
        red = np.full((4, 4), 1000, dtype=np.uint16)
        blue = np.full((4, 4), 500, dtype=np.uint16)
        result = compute_evi(nir, red, blue)
        assert result.dtype == np.float32

    def test_output_shape(self):
        shape = (10, 15)
        nir = np.random.rand(*shape).astype(np.float32)
        red = np.random.rand(*shape).astype(np.float32)
        blue = np.random.rand(*shape).astype(np.float32)
        result = compute_evi(nir, red, blue)
        assert result.shape == shape

    def test_shape_mismatch_raises(self):
        """Should raise ValueError when input shapes differ."""
        nir = np.ones((4, 4), dtype=np.float32)
        red = np.ones((4, 4), dtype=np.float32)
        blue = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_evi(nir, red, blue)


class TestAdaptiveThreshold:
    def test_bimodal_distribution(self):
        """Threshold should fall between two clear clusters."""
        unchanged = np.random.normal(0.0, 0.02, size=8000).astype(np.float32)
        changed = np.random.normal(0.3, 0.02, size=2000).astype(np.float32)
        delta = np.concatenate([unchanged, changed]).reshape(100, 100)
        threshold = compute_adaptive_threshold(delta)
        assert 0.05 < threshold < 0.25, f"Expected threshold between clusters, got {threshold}"

    def test_uniform_returns_fallback(self):
        """Uniform data with no clear separation should return fallback."""
        delta = np.zeros((50, 50), dtype=np.float32)
        threshold = compute_adaptive_threshold(delta)
        assert threshold == 0.10

    def test_returns_float(self):
        delta = np.random.randn(20, 20).astype(np.float32) * 0.2
        threshold = compute_adaptive_threshold(delta)
        assert isinstance(threshold, float)

    def test_positive_result(self):
        delta = np.random.randn(50, 50).astype(np.float32) * 0.3
        threshold = compute_adaptive_threshold(delta)
        assert threshold > 0


def test_safe_normalized_diff_rejects_shape_mismatch():
    """Should raise ValueError when input arrays have different shapes."""
    import pytest
    import numpy as np
    from src.indices import _safe_normalized_diff
    a = np.ones((64, 64), dtype=np.uint16)
    b = np.ones((64, 32), dtype=np.uint16)
    with pytest.raises(ValueError, match="shape mismatch"):
        _safe_normalized_diff(a, b)
