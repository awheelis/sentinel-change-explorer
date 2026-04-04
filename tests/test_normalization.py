"""Tests for PIF-based radiometric normalization."""
from __future__ import annotations

import numpy as np
import pytest

from src.normalization import normalize_pif


class TestNormalizePif:
    """Tests for normalize_pif()."""

    def _make_bands(self, shape=(100, 100), seed=42):
        """Create synthetic before bands with realistic Sentinel-2 ranges."""
        rng = np.random.RandomState(seed)
        return {
            "red": rng.randint(200, 3000, shape, dtype=np.uint16),
            "green": rng.randint(200, 3000, shape, dtype=np.uint16),
            "blue": rng.randint(200, 3000, shape, dtype=np.uint16),
            "nir": rng.randint(500, 5000, shape, dtype=np.uint16),
            "swir16": rng.randint(200, 4000, shape, dtype=np.uint16),
        }

    def test_recovers_shifted_bands(self):
        """Normalization should reduce error when after bands have a known gain/offset shift."""
        before = self._make_bands()
        after = {}
        for key, arr in before.items():
            shifted = np.clip(arr.astype(np.float64) * 1.15 + 80, 0, 65535)
            after[key] = shifted.astype(np.uint16)

        corrected, info = normalize_pif(before, after)

        assert not info["skipped"], f"Normalization was skipped: {info}"
        assert info["pif_count"] > 0

        for key in before:
            raw_error = np.mean(np.abs(after[key].astype(np.float64) - before[key].astype(np.float64)))
            corrected_error = np.mean(np.abs(corrected[key].astype(np.float64) - before[key].astype(np.float64)))
            assert corrected_error < raw_error * 0.5, (
                f"Band {key}: corrected_error={corrected_error:.1f} not < 50% of raw_error={raw_error:.1f}"
            )

    def test_identity_when_no_shift(self):
        """Identical before/after should produce near-identity correction."""
        before = self._make_bands()
        after = {k: v.copy() for k, v in before.items()}

        corrected, info = normalize_pif(before, after)

        assert not info["skipped"]
        for key in before:
            gain = info["bands"][key]["gain"]
            offset = info["bands"][key]["offset"]
            assert abs(gain - 1.0) < 0.05, f"Band {key}: gain={gain}, expected ~1.0"
            assert abs(offset) < 50, f"Band {key}: offset={offset}, expected ~0"

    def test_skips_when_insufficient_pifs(self):
        """When all pixels changed, normalization should be skipped."""
        before = self._make_bands()
        after = self._make_bands(seed=999)
        after["nir"] = np.full_like(after["nir"], 5000)
        after["red"] = np.full_like(after["red"], 200)

        corrected, info = normalize_pif(before, after)

        assert info["skipped"]
        for key in before:
            np.testing.assert_array_equal(corrected[key], after[key])

    def test_shape_mismatch_raises(self):
        """Mismatched band shapes should raise ValueError."""
        before = self._make_bands(shape=(100, 100))
        after = self._make_bands(shape=(50, 50))

        with pytest.raises(ValueError, match="shape mismatch"):
            normalize_pif(before, after)
