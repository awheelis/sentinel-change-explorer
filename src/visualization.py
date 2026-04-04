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
from folium.plugins import Draw
import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def true_color_image(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    gamma: float = 0.85,
) -> Image.Image:
    """Convert raw Sentinel-2 uint16 R/G/B arrays to a displayable RGB PIL Image.

    Clips to the given percentile range and normalizes to 0-255. This handles
    the uint16 0-10000 range of Sentinel-2 L2A surface reflectance values.

    Args:
        red: Red band array (B04), uint16.
        green: Green band array (B03), uint16.
        blue: Blue band array (B02), uint16.
        percentile_clip: Lower and upper percentile for contrast stretch.
        gamma: Gamma correction exponent. Values < 1 brighten midtones,
            values > 1 darken them. Default 0.85 for Sentinel-2 imagery.

    Returns:
        RGB PIL Image sized to match the input arrays.
    """
    if not (red.shape == green.shape == blue.shape):
        raise ValueError(
            f"Band shapes must match: red={red.shape}, green={green.shape}, blue={blue.shape}"
        )

    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    if percentile_clip[0] >= percentile_clip[1]:
        raise ValueError(
            f"percentile_clip must be (low, high), got {percentile_clip}"
        )

    # Convert each band separately to avoid a large intermediate stack allocation.
    r = red.astype(np.float32)
    g = green.astype(np.float32)
    b = blue.astype(np.float32)

    # Estimate percentiles from a 1-in-16 sample of the red band for speed.
    # A ~250 k-element sample gives stable results for 2000×2000 imagery.
    sample = r.ravel()[::16]
    lo, hi = np.percentile(sample, [percentile_clip[0], percentile_clip[1]])
    if hi == lo:
        hi = lo + 1.0

    rng_val = hi - lo
    for band in (r, g, b):
        np.subtract(band, lo, out=band)
        np.divide(band, rng_val, out=band)
        np.clip(band, 0.0, 1.0, out=band)

    if gamma != 1.0:
        # Apply gamma via a 256-entry uint8 LUT — avoids three np.power calls
        # on 4 M-element float arrays (≈ 3× faster on large images).
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
        r8 = lut[(r * 255).astype(np.uint8)]
        g8 = lut[(g * 255).astype(np.uint8)]
        b8 = lut[(b * 255).astype(np.uint8)]
    else:
        r8 = (r * 255).astype(np.uint8)
        g8 = (g * 255).astype(np.uint8)
        b8 = (b * 255).astype(np.uint8)

    rgb_uint8 = np.stack([r8, g8, b8], axis=-1)
    return Image.fromarray(rgb_uint8, mode="RGB")


def label_image(
    image: Image.Image,
    label: str,
    font_size: int = 18,
) -> Image.Image:
    """Burn a bold text label onto the top-left corner of a PIL Image.

    Returns a new image; the original is not mutated.

    Args:
        image: RGB PIL Image to label.
        label: Text to render. Empty string returns image unchanged.
        font_size: Approximate font size in pixels.

    Returns:
        New RGB PIL Image with the label overlaid.
    """
    if not label:
        return image.copy()

    from PIL import ImageDraw, ImageFont

    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    padding = 6
    # Semi-transparent background rectangle
    bg_box = (0, 0, text_w + padding * 2, text_h + padding * 2)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(bg_box, fill=(0, 0, 0, 140))
    overlay_draw.text((padding, padding), label, fill=(255, 255, 255, 255), font=font)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img


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
        cmap = matplotlib.colormaps[colormap]
    except KeyError as exc:
        raise ValueError(f"Invalid colormap '{colormap}'") from exc

    # Use bytes=True to get a uint8 RGBA array directly, avoiding a large
    # intermediate float64 allocation (≈ 2× faster on 2000×2000 inputs).
    rgba_uint8 = cmap(norm(delta), bytes=True)  # shape (H, W, 4), dtype uint8

    # Make near-zero and NaN pixels transparent; apply requested alpha to the rest.
    nan_mask = np.isnan(delta)
    unchanged_mask = np.abs(delta) <= threshold
    transparent = nan_mask | unchanged_mask
    rgba_uint8[transparent, 3] = 0
    rgba_uint8[~transparent, 3] = int(alpha * 255)

    return Image.fromarray(rgba_uint8, mode="RGBA")


def change_histogram(
    delta: np.ndarray,
    threshold: float = 0.05,
    bins: int = 50,
    index_name: str = "Index",
) -> matplotlib.figure.Figure:
    """Create a histogram of change delta values with threshold markers.

    Bars are colored by their bin center relative to the threshold:
    red for loss (< -threshold), blue for gain (> threshold), gray otherwise.

    Args:
        delta: 2D float32 array of index difference (after - before).
        threshold: Boundary for classifying gain/loss.
        bins: Number of histogram bins.

    Returns:
        A matplotlib Figure containing the histogram.
    """
    flat = delta.ravel()
    flat = flat[np.isfinite(flat)]

    fig, ax = plt.subplots()
    if len(flat) == 0:
        ax.text(0.5, 0.5, "No valid pixels", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig
    weights = np.ones_like(flat) / len(flat) * 100
    counts, edges, patches = ax.hist(
        flat, bins=bins, weights=weights, edgecolor="black", linewidth=0.3,
    )

    # Color each bar based on its bin center
    for patch, left_edge, right_edge in zip(patches, edges[:-1], edges[1:]):
        center = (left_edge + right_edge) / 2
        if center < -threshold:
            patch.set_facecolor("#d73027")
        elif center > threshold:
            patch.set_facecolor("#4575b4")
        else:
            patch.set_facecolor("#999999")

    ax.axvline(-threshold, color="black", linestyle="--", linewidth=1)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)

    y_top = ax.get_ylim()[1]
    ax.text(-threshold, y_top * 0.95, " Loss\n threshold", fontsize=8, va="top")
    ax.text(threshold, y_top * 0.95, " Gain\n threshold", fontsize=8, va="top")

    ax.set_xlabel(f"{index_name} Change Magnitude")
    ax.set_ylabel("Proportion of Area (%)")
    ax.set_title("Change Distribution")

    fig.tight_layout()
    return fig


def downscale_array(
    arr: np.ndarray,
    max_dim: int = 800,
) -> np.ndarray:
    """Downscale a 2D float32 array so its longest side is at most max_dim.

    Uses PIL bilinear resize on the raw array, avoiding the need to create
    a full-resolution RGBA colormapped image first.

    Args:
        arr: 2D float32 array to downscale.
        max_dim: Maximum allowed dimension (width or height).

    Returns:
        Downscaled float32 array, or the original if already small enough.
    """
    h, w = arr.shape
    if max(h, w) <= max_dim:
        return arr
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = Image.fromarray(arr, mode="F")
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


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
    enable_draw: bool = False,
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
        enable_draw: If True, add Draw plugin for interactive AOI selection.

    Returns:
        Configured folium.Map ready for st_folium().
    """
    west, south, east, north = bbox
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        tiles="CartoDB positron",
    )
    m.fit_bounds([[south, west], [north, east]])

    # Trigger Leaflet resize when embedded in Streamlit iframes to prevent
    # blank map rendering on initial scroll-into-view.
    m.get_root().html.add_child(folium.Element(
        "<script>document.addEventListener('DOMContentLoaded', function() {"
        "  setTimeout(function() {"
        "    window.dispatchEvent(new Event('resize'));"
        "  }, 200);"
        "});</script>"
    ))

    if before_image is not None:
        _image_to_bounds_overlay(before_image, bbox, name="Before (True Color)", opacity=0.9).add_to(m)

    if after_image is not None:
        _image_to_bounds_overlay(after_image, bbox, name="After (True Color)", opacity=0.9).add_to(m)

    if heatmap_image is not None and show_heatmap:
        _image_to_bounds_overlay(heatmap_image, bbox, name="Change Heatmap", opacity=1.0).add_to(m)

        # Add color legend
        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px; right: 30px;
            z-index: 1000;
            background: white;
            padding: 10px 14px;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            font-size: 13px;
            line-height: 1.6;
        ">
            <b>Change Legend</b><br>
            <span style="display:inline-block;width:16px;height:12px;background:#d73027;border:1px solid #999;"></span> Loss<br>
            <span style="display:inline-block;width:16px;height:12px;background:#ccc;border:1px solid #999;"></span> No Change<br>
            <span style="display:inline-block;width:16px;height:12px;background:#4575b4;border:1px solid #999;"></span> Gain
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    if overture_context is not None and show_overture:
        # Cap feature counts to keep the folium HTML payload under
        # Streamlit's 200 MB message-size limit.  Dense urban areas
        # (e.g. Las Vegas) can have 300k+ buildings.
        _MAX_BUILDINGS = 5_000
        _MAX_SEGMENTS = 5_000
        _MAX_PLACES = 2_000

        buildings = overture_context.get("building", gpd.GeoDataFrame())
        if not buildings.empty:
            if len(buildings) > _MAX_BUILDINGS:
                buildings = buildings.sample(n=_MAX_BUILDINGS, random_state=42)
            # Strip to geometry-only to avoid JSON serialization errors
            # from complex nested Overture properties.
            folium.GeoJson(
                buildings[["geometry"]].__geo_interface__,
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
            if len(segments) > _MAX_SEGMENTS:
                segments = segments.sample(n=_MAX_SEGMENTS, random_state=42)
            folium.GeoJson(
                segments[["geometry"]].__geo_interface__,
                name="Roads",
                style_function=lambda _: {
                    "color": "#4477ff",
                    "weight": 1.5,
                    "opacity": 0.6,
                },
            ).add_to(m)

        places = overture_context.get("place", gpd.GeoDataFrame())
        if not places.empty:
            if len(places) > _MAX_PLACES:
                places = places.sample(n=_MAX_PLACES, random_state=42)
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

    if enable_draw:
        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "polygon": True,
                "rectangle": True,
            },
            edit_options={"edit": False},
        ).add_to(m)

    # Only add LayerControl when there are multiple togglable layers
    overlay_count = sum([
        before_image is not None,
        after_image is not None,
        heatmap_image is not None and show_heatmap,
        bool(overture_context is not None and show_overture),
    ])
    if overlay_count >= 2:
        folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── Multi-index change classification rendering ─────────────────────────

# (R, G, B) per category — keyed by the integer codes from src.indices
_CLASSIFICATION_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),          # UNCHANGED — not rendered (transparent)
    1: (255, 165, 0),      # URBAN_CONVERSION — Orange
    2: (220, 38, 38),      # VEGETATION_LOSS — Red
    3: (59, 130, 246),     # FLOODING — Blue
    4: (34, 197, 94),      # VEGETATION_GAIN — Green
}

_CLASSIFICATION_LABELS: dict[int, str] = {
    1: "Urban Conversion",
    2: "Vegetation Loss",
    3: "Flooding / Water Gain",
    4: "Vegetation Gain",
}


def classification_to_rgba(
    categories: np.ndarray,
    alpha: float = 0.7,
) -> Image.Image:
    """Convert a classification category array to an RGBA PIL Image.

    Args:
        categories: 2D uint8 array of category codes from classify_change().
        alpha: Opacity for classified pixels (0-1). Unchanged pixels are
            always transparent.

    Returns:
        RGBA PIL Image sized to match the input array.
    """
    h, w = categories.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for code, (r, g, b) in _CLASSIFICATION_COLORS.items():
        mask = categories == code
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b

    # Unchanged pixels stay transparent; all others get requested alpha
    changed = categories != 0
    rgba[changed, 3] = int(alpha * 255)

    return Image.fromarray(rgba, mode="RGBA")
