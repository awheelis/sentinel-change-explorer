"""Tests for SCL cloud/shadow masking module."""
import numpy as np
import pytest

from src.masking import SCL_MASK_VALUES, apply_mask, build_scl_mask, mask_percentage, union_masks


class TestBuildSclMask:
    """Tests for build_scl_mask function."""

    def test_masks_cloud_shadow(self):
        scl = np.array([[3, 0], [1, 4]], dtype=np.uint8)
        mask = build_scl_mask(scl)
        assert mask[0, 0] is np.True_  # shadow (3)
        assert mask[0, 1] is np.False_  # no data but not masked
        assert mask[1, 0] is np.False_
        assert mask[1, 1] is np.False_  # vegetation

    def test_masks_cloud_medium(self):
        scl = np.array([[8]], dtype=np.uint8)
        assert build_scl_mask(scl)[0, 0] is np.True_

    def test_masks_cloud_high(self):
        scl = np.array([[9]], dtype=np.uint8)
        assert build_scl_mask(scl)[0, 0] is np.True_

    def test_masks_thin_cirrus(self):
        scl = np.array([[10]], dtype=np.uint8)
        assert build_scl_mask(scl)[0, 0] is np.True_

    def test_all_mask_values_covered(self):
        """Ensure SCL_MASK_VALUES matches {3, 8, 9, 10}."""
        assert SCL_MASK_VALUES == {3, 8, 9, 10}

    def test_does_not_mask_clear_classes(self):
        """Classes 4 (vegetation), 5 (bare soil), 6 (water), 7 (unclassified low),
        11 (snow) should NOT be masked."""
        clear_classes = [0, 1, 2, 4, 5, 6, 7, 11]
        scl = np.array(clear_classes, dtype=np.uint8).reshape(1, -1)
        mask = build_scl_mask(scl)
        assert not mask.any()

    def test_output_is_boolean(self):
        scl = np.zeros((10, 10), dtype=np.uint8)
        mask = build_scl_mask(scl)
        assert mask.dtype == bool

    def test_preserves_shape(self):
        scl = np.zeros((50, 30), dtype=np.uint8)
        mask = build_scl_mask(scl)
        assert mask.shape == (50, 30)


class TestApplyMask:
    """Tests for apply_mask function."""

    def test_sets_masked_pixels_to_nan(self):
        bands = {
            "red": np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16),
            "nir": np.array([[5000, 6000], [7000, 8000]], dtype=np.uint16),
        }
        mask = np.array([[True, False], [False, True]])
        result = apply_mask(bands, mask)
        assert np.isnan(result["red"][0, 0])
        assert np.isnan(result["nir"][0, 0])
        assert np.isnan(result["red"][1, 1])
        assert np.isnan(result["nir"][1, 1])

    def test_preserves_unmasked_pixels(self):
        bands = {
            "red": np.array([[1000, 2000]], dtype=np.uint16),
        }
        mask = np.array([[True, False]])
        result = apply_mask(bands, mask)
        assert result["red"][0, 1] == 2000.0

    def test_converts_to_float32(self):
        bands = {"red": np.array([[100]], dtype=np.uint16)}
        mask = np.array([[False]])
        result = apply_mask(bands, mask)
        assert result["red"].dtype == np.float32

    def test_does_not_mutate_input(self):
        original = np.array([[1000, 2000]], dtype=np.uint16)
        bands = {"red": original.copy()}
        mask = np.array([[True, False]])
        apply_mask(bands, mask)
        np.testing.assert_array_equal(original, np.array([[1000, 2000]], dtype=np.uint16))

    def test_empty_mask_no_nans(self):
        bands = {"red": np.array([[100, 200]], dtype=np.uint16)}
        mask = np.array([[False, False]])
        result = apply_mask(bands, mask)
        assert not np.any(np.isnan(result["red"]))

    def test_full_mask_all_nans(self):
        bands = {"red": np.array([[100, 200]], dtype=np.uint16)}
        mask = np.array([[True, True]])
        result = apply_mask(bands, mask)
        assert np.all(np.isnan(result["red"]))


class TestUnionMasks:
    """Tests for union_masks function."""

    def test_ors_two_masks(self):
        a = np.array([[True, False], [False, False]])
        b = np.array([[False, True], [False, False]])
        result = union_masks(a, b)
        expected = np.array([[True, True], [False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_single_mask_returned_as_is(self):
        a = np.array([[True, False]])
        result = union_masks(a)
        np.testing.assert_array_equal(result, a)

    def test_three_masks(self):
        a = np.array([[True, False, False]])
        b = np.array([[False, True, False]])
        c = np.array([[False, False, True]])
        result = union_masks(a, b, c)
        assert result.all()

    def test_output_is_boolean(self):
        a = np.array([[True, False]])
        b = np.array([[False, True]])
        assert union_masks(a, b).dtype == bool


class TestMaskPercentage:
    """Tests for mask_percentage function."""

    def test_no_masked_pixels(self):
        mask = np.array([[False, False, False, False]])
        assert mask_percentage(mask) == 0.0

    def test_all_masked(self):
        mask = np.array([[True, True]])
        assert mask_percentage(mask) == 100.0

    def test_half_masked(self):
        mask = np.array([[True, False, True, False]])
        assert mask_percentage(mask) == pytest.approx(50.0)

    def test_quarter_masked(self):
        mask = np.array([[True, False], [False, False]])
        assert mask_percentage(mask) == pytest.approx(25.0)


class TestNanPropagation:
    """Verify NaN from masking propagates correctly through index computation."""

    def test_nan_propagates_through_normalized_diff(self):
        from src.indices import compute_ndvi
        nir = np.array([[5000.0, np.nan], [3000.0, 4000.0]], dtype=np.float32)
        red = np.array([[2000.0, 1000.0], [np.nan, 1500.0]], dtype=np.float32)
        result = compute_ndvi(nir, red)
        assert np.isnan(result[0, 1])  # NaN in nir
        assert np.isnan(result[1, 0])  # NaN in red
        assert not np.isnan(result[0, 0])  # both valid
        assert not np.isnan(result[1, 1])

    def test_nan_propagates_through_change(self):
        from src.indices import compute_change
        before = np.array([[0.5, np.nan]], dtype=np.float32)
        after = np.array([[0.3, 0.4]], dtype=np.float32)
        result = compute_change(before, after)
        assert np.isnan(result[0, 1])
        assert not np.isnan(result[0, 0])
