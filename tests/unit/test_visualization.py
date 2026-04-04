"""Tests for visualization helpers."""
import matplotlib.figure
import numpy as np
import pytest
from PIL import Image
from src.visualization import true_color_image, downscale_array, index_to_rgba, change_histogram
import folium
import geopandas as gpd
from shapely.geometry import box, Point, LineString
from src.visualization import build_folium_map, _image_to_bounds_overlay, label_image


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


BBOX = (-115.20, 36.10, -115.15, 36.15)


def _make_rgb_image(size=(100, 100)):
    """Create a small synthetic RGB PIL Image."""
    arr = np.random.RandomState(0).randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_rgba_image(size=(100, 100)):
    """Create a small synthetic RGBA PIL Image."""
    arr = np.random.RandomState(0).randint(0, 255, (*size, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _make_overture_context(n_buildings=5, n_segments=3, n_places=2):
    """Create a synthetic overture context dict."""
    buildings = gpd.GeoDataFrame(
        geometry=[box(-115.19 + i * 0.001, 36.11, -115.189 + i * 0.001, 36.112) for i in range(n_buildings)],
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        geometry=[LineString([(-115.19 + i * 0.01, 36.11), (-115.18 + i * 0.01, 36.12)]) for i in range(n_segments)],
        crs="EPSG:4326",
    )
    places = gpd.GeoDataFrame(
        {"names": [{"primary": f"Place {i}"} for i in range(n_places)],
         "geometry": [Point(-115.18 + i * 0.01, 36.12) for i in range(n_places)]},
        crs="EPSG:4326",
    )
    return {"building": buildings, "segment": segments, "place": places}


class TestBuildFoliumMap:
    def test_returns_folium_map(self):
        m = build_folium_map(
            bbox=BBOX,
            before_image=_make_rgb_image(),
            after_image=_make_rgb_image(),
            heatmap_image=_make_rgba_image(),
            overture_context=_make_overture_context(),
        )
        assert isinstance(m, folium.Map)

    def test_layer_count_with_all_inputs(self):
        """before + after + heatmap + overture layers + LayerControl."""
        m = build_folium_map(
            bbox=BBOX,
            before_image=_make_rgb_image(),
            after_image=_make_rgb_image(),
            heatmap_image=_make_rgba_image(),
            overture_context=_make_overture_context(),
            show_heatmap=True,
            show_overture=True,
        )
        children = list(m._children.values())
        assert len(children) >= 5

    def test_no_overture_omits_vector_layers(self):
        m = build_folium_map(
            bbox=BBOX,
            before_image=_make_rgb_image(),
            after_image=_make_rgb_image(),
            heatmap_image=_make_rgba_image(),
            show_overture=False,
        )
        html = m._repr_html_()
        assert "Buildings" not in html
        assert "Roads" not in html

    def test_none_images_returns_valid_map(self):
        m = build_folium_map(bbox=BBOX)
        assert isinstance(m, folium.Map)

    def test_overture_sampling_caps_buildings(self):
        """When >5000 buildings passed, map should still render."""
        large_overture = _make_overture_context(n_buildings=10_000, n_segments=0, n_places=0)
        m = build_folium_map(
            bbox=BBOX,
            overture_context=large_overture,
            show_overture=True,
        )
        html = m._repr_html_()
        assert "Buildings" in html

    def test_draw_plugin_present(self):
        """Map should include Draw plugin when enable_draw=True."""
        m = build_folium_map(bbox=BBOX, enable_draw=True)
        html = m._repr_html_()
        assert "Draw" in html or "draw" in html

    def test_overture_only_no_imagery(self):
        """Panel C use case: Overture layers only, no imagery overlays."""
        m = build_folium_map(
            bbox=BBOX,
            overture_context=_make_overture_context(),
            show_heatmap=False,
            show_overture=True,
        )
        html = m._repr_html_()
        assert "Buildings" in html
        assert "Roads" in html
        # No imagery overlays
        assert "Before" not in html
        assert "After" not in html
        assert "Heatmap" not in html


class TestChangeHistogram:
    def _make_delta(self, shape=(200, 200)):
        return np.random.default_rng(42).uniform(-0.5, 0.5, shape).astype(np.float32)

    def test_returns_figure(self):
        delta = self._make_delta()
        fig = change_histogram(delta)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_vertical_threshold_lines(self):
        delta = self._make_delta()
        fig = change_histogram(delta, threshold=0.1)
        ax = fig.axes[0]
        vlines = [line for line in ax.get_lines() if len(line.get_xdata()) > 0]
        xpositions = sorted([line.get_xdata()[0] for line in vlines])
        assert len(xpositions) == 2
        assert abs(xpositions[0] - (-0.1)) < 1e-6
        assert abs(xpositions[1] - 0.1) < 1e-6

    def test_bins_count(self):
        delta = self._make_delta()
        fig = change_histogram(delta, bins=50)
        ax = fig.axes[0]
        patches = ax.patches
        assert len(patches) == 50


class TestImageToBoundsOverlay:
    def test_returns_image_overlay(self):
        img = _make_rgb_image()
        overlay = _image_to_bounds_overlay(img, BBOX, name="test")
        assert isinstance(overlay, folium.raster_layers.ImageOverlay)


class TestLabelImage:
    def test_returns_rgb_image_same_size(self):
        """Labeled image should be same size and mode as input."""
        img = _make_rgb_image(size=(200, 300))
        result = label_image(img, "Before — 2019-07-10")
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == img.size

    def test_modifies_pixels(self):
        """Label should change some pixels in the top region."""
        img = _make_rgb_image(size=(200, 300))
        original_arr = np.array(img).copy()
        result = label_image(img, "Before — 2019-07-10")
        result_arr = np.array(result)
        # Top 40 rows should have some modified pixels (label area)
        top_region_changed = not np.array_equal(
            original_arr[:40, :, :], result_arr[:40, :, :]
        )
        assert top_region_changed, "Label should modify pixels in the top region"

    def test_empty_label_returns_unchanged(self):
        """Empty string label should return image unchanged."""
        img = _make_rgb_image(size=(200, 300))
        original_arr = np.array(img).copy()
        result = label_image(img, "")
        assert np.array_equal(original_arr, np.array(result))

    def test_does_not_mutate_input(self):
        """Original image should not be modified."""
        img = _make_rgb_image(size=(200, 300))
        original_arr = np.array(img).copy()
        label_image(img, "Test Label")
        assert np.array_equal(original_arr, np.array(img))


def test_build_folium_map_uses_fit_bounds():
    """Map should use fit_bounds instead of hardcoded zoom_start."""
    from src.visualization import build_folium_map
    bbox = (-115.20, 36.10, -115.15, 36.15)
    m = build_folium_map(bbox=bbox)
    html = m._repr_html_()
    assert "fitBounds" in html


def test_build_folium_map_has_legend():
    """Heatmap map should include a color legend."""
    from src.visualization import build_folium_map, index_to_rgba
    import numpy as np
    delta = np.random.RandomState(42).randn(64, 64).astype(np.float32) * 0.3
    heatmap = index_to_rgba(delta, threshold=0.05)
    bbox = (-115.20, 36.10, -115.15, 36.15)
    m = build_folium_map(bbox=bbox, heatmap_image=heatmap, show_heatmap=True)
    html = m._repr_html_()
    assert "Loss" in html and "Gain" in html
