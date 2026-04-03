"""Performance benchmarks for computation and rendering.

All tests use synthetic data (no network). Thresholds are generous —
failure means a genuine regression, not normal machine variance.

Run with: pytest tests/perf/ -v
"""
import time

import numpy as np
import pytest
from PIL import Image

from src.indices import compute_ndvi, compute_ndbi, compute_mndwi, compute_change, _safe_normalized_diff
from src.visualization import true_color_image, index_to_rgba, downscale_array


pytestmark = pytest.mark.perf


class TestIndexBenchmarks:
    def test_ndvi_2000x2000_under_100ms(self, large_bands):
        start = time.perf_counter()
        result = compute_ndvi(large_bands["nir"], large_bands["red"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"NDVI took {elapsed:.3f}s, expected < 0.1s"
        assert result.shape == (2000, 2000)

    def test_ndbi_2000x2000_under_100ms(self, large_bands):
        start = time.perf_counter()
        result = compute_ndbi(large_bands["swir16"], large_bands["nir"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"NDBI took {elapsed:.3f}s, expected < 0.1s"
        assert result.shape == (2000, 2000)

    def test_mndwi_2000x2000_under_100ms(self, large_bands):
        start = time.perf_counter()
        result = compute_mndwi(large_bands["green"], large_bands["swir16"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"MNDWI took {elapsed:.3f}s, expected < 0.1s"
        assert result.shape == (2000, 2000)

    def test_compute_change_2000x2000_under_50ms(self, large_bands):
        before = compute_ndvi(large_bands["nir"], large_bands["red"])
        after = compute_ndvi(large_bands["nir"], large_bands["green"])
        start = time.perf_counter()
        delta = compute_change(before=before, after=after)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05, f"compute_change took {elapsed:.3f}s, expected < 0.05s"
        assert delta.shape == (2000, 2000)

    def test_chunked_ndvi_not_slower_than_2x(self, large_bands):
        nir = large_bands["nir"]
        red = large_bands["red"]

        start = time.perf_counter()
        _safe_normalized_diff(nir, red, chunk_rows=None)
        single_time = time.perf_counter() - start

        start = time.perf_counter()
        _safe_normalized_diff(nir, red, chunk_rows=512)
        chunked_time = time.perf_counter() - start

        assert chunked_time < single_time * 2, (
            f"Chunked ({chunked_time:.3f}s) > 2x single-pass ({single_time:.3f}s)"
        )


class TestRenderingBenchmarks:
    def test_true_color_2000x2000_under_200ms(self, large_bands):
        start = time.perf_counter()
        img = true_color_image(large_bands["red"], large_bands["green"], large_bands["blue"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.2, f"true_color_image took {elapsed:.3f}s, expected < 0.2s"
        assert isinstance(img, Image.Image)
        assert img.size == (2000, 2000)

    def test_index_to_rgba_2000x2000_under_200ms(self, large_bands):
        delta = compute_ndvi(large_bands["nir"], large_bands["red"])
        start = time.perf_counter()
        img = index_to_rgba(delta, threshold=0.05)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.2, f"index_to_rgba took {elapsed:.3f}s, expected < 0.2s"
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"

    def test_downscale_2000x2000_to_800_under_50ms(self, large_bands):
        arr = compute_ndvi(large_bands["nir"], large_bands["red"])
        start = time.perf_counter()
        result = downscale_array(arr, max_dim=800)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05, f"downscale_array took {elapsed:.3f}s, expected < 0.05s"
        assert max(result.shape) == 800


class TestPipelineBenchmark:
    def test_full_compute_pipeline_under_500ms(self, large_bands):
        """Full compute path: 3 indices, 3 deltas, 2 true-color, 1 heatmap."""
        start = time.perf_counter()

        before_ndvi = compute_ndvi(large_bands["nir"], large_bands["red"])
        after_ndvi = compute_ndvi(large_bands["nir"], large_bands["green"])
        delta_ndvi = compute_change(before=before_ndvi, after=after_ndvi)

        before_ndbi = compute_ndbi(large_bands["swir16"], large_bands["nir"])
        after_ndbi = compute_ndbi(large_bands["swir16"], large_bands["red"])
        delta_ndbi = compute_change(before=before_ndbi, after=after_ndbi)

        before_mndwi = compute_mndwi(large_bands["green"], large_bands["swir16"])
        after_mndwi = compute_mndwi(large_bands["green"], large_bands["nir"])
        delta_mndwi = compute_change(before=before_mndwi, after=after_mndwi)

        before_img = true_color_image(large_bands["red"], large_bands["green"], large_bands["blue"])
        after_img = true_color_image(large_bands["red"], large_bands["green"], large_bands["blue"])

        small_delta = downscale_array(delta_ndvi, max_dim=800)
        heatmap = index_to_rgba(small_delta, threshold=0.05)

        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Full pipeline took {elapsed:.3f}s, expected < 0.5s"

        assert isinstance(before_img, Image.Image)
        assert isinstance(after_img, Image.Image)
        assert isinstance(heatmap, Image.Image)
        assert heatmap.mode == "RGBA"
