"""Visualization helpers for Sentinel-2 imagery and change detection.

Provides:
- true_color_image(): convert uint16 R/G/B arrays to a displayable PIL Image
- index_to_rgba(): convert a change delta array to a diverging RGBA image
- build_folium_map(): create a folium Map with image overlay and Overture layers
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import folium
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image


def true_color_image(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    percentile_clip: tuple[float, float] = (2.0, 98.0),
) -> Image.Image:
    """Convert raw Sentinel-2 uint16 R/G/B arrays to a displayable RGB PIL Image.

    Clips to the given percentile range and normalizes to 0-255. This handles
    the uint16 0-10000 range of Sentinel-2 L2A surface reflectance values.

    Args:
        red: Red band array (B04), uint16.
        green: Green band array (B03), uint16.
        blue: Blue band array (B02), uint16.
        percentile_clip: Lower and upper percentile for contrast stretch.

    Returns:
        RGB PIL Image sized to match the input arrays.
    """
    if not (red.shape == green.shape == blue.shape):
        raise ValueError(
            f"Band shapes must match: red={red.shape}, green={green.shape}, blue={blue.shape}"
        )

    if percentile_clip[0] >= percentile_clip[1]:
        raise ValueError(
            f"percentile_clip must be (low, high), got {percentile_clip}"
        )

    stack = np.stack([red, green, blue], axis=-1).astype(np.float32)

    lo = np.percentile(stack, percentile_clip[0])
    hi = np.percentile(stack, percentile_clip[1])
    if hi == lo:
        hi = lo + 1.0

    stretched = np.clip((stack - lo) / (hi - lo), 0.0, 1.0)
    rgb_uint8 = (stretched * 255).astype(np.uint8)
    return Image.fromarray(rgb_uint8, mode="RGB")


def index_to_rgba(
    delta: np.ndarray,
    threshold: float = 0.05,
    colormap: str = "RdBu",
    vmin: float = -0.5,
    vmax: float = 0.5,
    alpha: float = 0.7,
) -> Image.Image:
    """Convert a change delta array to a diverging RGBA heatmap image.

    Pixels within ±threshold of zero are rendered transparent (no change).
    Negative values (loss) render red; positive values (gain) render blue.

    Args:
        delta: 2D float32 array of index difference (after - before).
        threshold: Pixels with |delta| <= threshold are transparent.
        colormap: Matplotlib diverging colormap name. Default "RdBu".
        vmin: Value mapped to the low end of the colormap.
        vmax: Value mapped to the high end of the colormap.
        alpha: Opacity for changed pixels (0-1).

    Returns:
        RGBA PIL Image sized to match the input array.
    """
    if vmin >= vmax:
        raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax})")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    try:
        cmap = cm.get_cmap(colormap)
    except ValueError as exc:
        raise ValueError(f"Invalid colormap '{colormap}'") from exc
    rgba = cmap(norm(delta))  # shape (H, W, 4), values 0-1

    # Make near-zero pixels transparent
    mask = np.abs(delta) <= threshold
    rgba[..., 3] = np.where(mask, 0.0, alpha)

    rgba_uint8 = (rgba * 255).astype(np.uint8)
    return Image.fromarray(rgba_uint8, mode="RGBA")


def _image_to_bounds_overlay(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    name: str = "overlay",
    opacity: float = 0.8,
) -> folium.raster_layers.ImageOverlay:
    """Convert a PIL Image to a folium ImageOverlay for the given bbox.

    Args:
        image: PIL Image (RGB or RGBA) to overlay.
        bbox: (west, south, east, north) in WGS84.
        name: Layer name shown in the folium layer control.
        opacity: Overlay opacity (0-1), used for RGB images.

    Returns:
        folium.raster_layers.ImageOverlay ready to add to a map.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    img_url = f"data:image/png;base64,{img_b64}"

    west, south, east, north = bbox
    return folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[south, west], [north, east]],
        opacity=opacity,
        name=name,
        cross_origin=False,
    )


def build_folium_map(
    bbox: tuple[float, float, float, float],
    before_image: Optional[Image.Image] = None,
    after_image: Optional[Image.Image] = None,
    heatmap_image: Optional[Image.Image] = None,
    overture_context: Optional[dict[str, gpd.GeoDataFrame]] = None,
    show_heatmap: bool = True,
    show_overture: bool = True,
) -> folium.Map:
    """Build a folium Map with imagery overlays and Overture Maps context.

    Args:
        bbox: (west, south, east, north) in WGS84 — sets map center and view.
        before_image: PIL Image for "before" true-color overlay.
        after_image: PIL Image for "after" true-color overlay.
        heatmap_image: RGBA PIL Image for change heatmap overlay.
        overture_context: Dict from get_overture_context() with "building",
            "segment", "place" GeoDataFrame values.
        show_heatmap: If True, add heatmap overlay to map.
        show_overture: If True, add Overture vector layers to map.

    Returns:
        Configured folium.Map ready for st_folium().
    """
    west, south, east, north = bbox
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    if before_image is not None:
        _image_to_bounds_overlay(before_image, bbox, name="Before (True Color)", opacity=0.9).add_to(m)

    if after_image is not None:
        _image_to_bounds_overlay(after_image, bbox, name="After (True Color)", opacity=0.9).add_to(m)

    if heatmap_image is not None and show_heatmap:
        _image_to_bounds_overlay(heatmap_image, bbox, name="Change Heatmap", opacity=1.0).add_to(m)

    if overture_context is not None and show_overture:
        buildings = overture_context.get("building", gpd.GeoDataFrame())
        if not buildings.empty:
            folium.GeoJson(
                buildings.__geo_interface__,
                name="Buildings",
                style_function=lambda _: {
                    "fillColor": "#ff7800",
                    "color": "#cc4400",
                    "weight": 0.5,
                    "fillOpacity": 0.3,
                },
            ).add_to(m)

        segments = overture_context.get("segment", gpd.GeoDataFrame())
        if not segments.empty:
            folium.GeoJson(
                segments.__geo_interface__,
                name="Roads",
                style_function=lambda _: {
                    "color": "#4477ff",
                    "weight": 1.5,
                    "opacity": 0.6,
                },
            ).add_to(m)

        places = overture_context.get("place", gpd.GeoDataFrame())
        if not places.empty:
            for _, row in places.iterrows():
                if row.geometry and row.geometry.geom_type == "Point":
                    name_val = row.get("names", {})
                    if isinstance(name_val, dict):
                        label = name_val.get("primary", "Place")
                    else:
                        label = str(name_val) if name_val else "Place"
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=4,
                        color="#9900cc",
                        fill=True,
                        fill_opacity=0.7,
                        popup=label,
                        tooltip=label,
                    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
