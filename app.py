"""Sentinel-2 Change Detection Explorer — Streamlit application.

Run with: streamlit run app.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

# Ensure src/ is on path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.export import create_geotiff
from src.sentinel import load_bands, search_scenes
from src.indices import compute_change, compute_evi, compute_mndwi, compute_ndbi, compute_ndvi
from src.overture import get_overture_context
from src.visualization import (
    build_folium_map,
    change_histogram,
    downscale_array,
    index_to_rgba,
    label_image,
    true_color_image,
)

PRESETS_FILE = Path(__file__).parent / "config" / "presets.json"

INDEX_FUNCTIONS = {
    "ndvi": ("NDVI", compute_ndvi, ["nir", "red"]),
    "ndbi": ("NDBI", compute_ndbi, ["swir16", "nir"]),
    "mndwi": ("MNDWI", compute_mndwi, ["green", "swir16"]),
    "evi": ("EVI", compute_evi, ["nir", "red", "blue"]),
}

# Bands needed for all indices + true color
ALL_BAND_KEYS = ["red", "green", "blue", "nir", "swir16"]


@st.cache_data
def load_presets() -> list[dict]:
    """Load preset locations from config/presets.json."""
    try:
        with open(PRESETS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        st.warning(f"Could not load presets: {exc}")
        return []


logger = logging.getLogger(__name__)


def warm_preset_caches(on_progress=None) -> None:
    """Pre-fetch search results, bands, and Overture data for all presets.

    Runs all presets concurrently via ThreadPoolExecutor. Individual preset
    failures are logged and swallowed so one bad preset doesn't block the rest.
    Populates the disk caches used by load_bands and get_overture_context.

    Args:
        on_progress: Optional callback called with (completed, total) after
            each task finishes. Useful for driving a progress bar.
    """
    presets = load_presets()
    if not presets:
        return

    def _warm_one_date(preset, date_range_key):
        """Search + load bands for one preset and one date range."""
        bbox = tuple(preset["bbox"])
        dr = preset[date_range_key]
        date_range = f"{dr[0]}/{dr[1]}"
        scenes = search_scenes(bbox=bbox, date_range=date_range, max_cloud_cover=20)
        if scenes:
            load_bands(scene=scenes[0], bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=10)

    def _warm_overture(preset):
        """Fetch Overture context for one preset."""
        bbox = tuple(preset["bbox"])
        get_overture_context(bbox=bbox)

    futures = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        for preset in presets:
            futures.append(executor.submit(_warm_one_date, preset, "before_range"))
            futures.append(executor.submit(_warm_one_date, preset, "after_range"))
            futures.append(executor.submit(_warm_overture, preset))

        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.warning("Preset warm-up task failed", exc_info=True)
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)


def compute_index_for_bands(
    index_key: str,
    bands: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute the specified spectral index from a bands dict.

    Args:
        index_key: One of "ndvi", "ndbi", "mndwi".
        bands: Dict mapping band key → 2D array.

    Returns:
        2D float32 index array.
    """
    _, fn, band_order = INDEX_FUNCTIONS[index_key]
    return fn(*[bands[k] for k in band_order])



def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Sentinel-2 Change Explorer",
        page_icon=":earth_americas:",
        layout="wide",
    )
    st.markdown(
        """<style>
        /* Sidebar section headers: stronger visual weight */
        section[data-testid="stSidebar"] h2 {
            font-size: 1.1rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 2px solid rgba(255,75,75,0.4);
            padding-bottom: 0.3rem;
            margin-top: 1.2rem;
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 0.85rem;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-top: 1rem;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Sentinel-2 Change Detection Explorer")
    st.caption(
        "Compare satellite imagery across two dates to detect vegetation loss, "
        "urbanization, and water change using Sentinel-2 L2A imagery."
    )

    # Warm-up removed — data is fetched lazily when user clicks Analyze Change.

    presets = load_presets()
    preset_names = ["Custom…"] + [p["name"] for p in presets]

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Location & Dates")

        # Initialize widget state on first load
        if "_last_preset" not in st.session_state:
            st.session_state["_last_preset"] = None
            # Default coordinates match Lahaina Wildfire preset (index=0)
            st.session_state["west"] = -156.695
            st.session_state["east"] = -156.660
            st.session_state["south"] = 20.860
            st.session_state["north"] = 20.895
            from datetime import date as _date
            st.session_state["before_start"] = _date.fromisoformat("2023-05-01")
            st.session_state["before_end"] = _date.fromisoformat("2023-07-31")
            st.session_state["after_start"] = _date.fromisoformat("2023-09-01")
            st.session_state["after_end"] = _date.fromisoformat("2023-11-30")
            st.session_state["index_choice"] = "ndvi"
            st.session_state["threshold"] = 0.12  # Lahaina default

        preset_choice = st.selectbox("Preset location", preset_names, index=0)
        if preset_choice != "Custom…":
            preset = next(p for p in presets if p["name"] == preset_choice)
            default_bbox = preset["bbox"]
            default_before_start = preset["before_range"][0]
            default_before_end = preset["before_range"][1]
            default_after_start = preset["after_range"][0]
            default_after_end = preset["after_range"][1]
            default_index = preset.get("default_index", "ndvi")
            default_threshold = preset.get("threshold", 0.10)
            if "what_to_expect" in preset:
                st.info(f"**What to expect:** {preset['what_to_expect']}")
            if "notes" in preset:
                st.caption(f"💡 {preset['notes']}")
        else:
            default_bbox = [-115.32, 36.08, -115.08, 36.28]
            default_before_start, default_before_end = "2019-05-01", "2019-07-31"
            default_after_start, default_after_end = "2023-05-01", "2023-07-31"
            default_index = "ndvi"
            default_threshold = 0.10

        # Reset widget values when preset changes
        if st.session_state.get("_last_preset") != preset_choice:
            st.session_state["_last_preset"] = preset_choice
            st.session_state["west"] = float(default_bbox[0])
            st.session_state["east"] = float(default_bbox[2])
            st.session_state["south"] = float(default_bbox[1])
            st.session_state["north"] = float(default_bbox[3])
            from datetime import date as _date
            st.session_state["before_start"] = _date.fromisoformat(default_before_start)
            st.session_state["before_end"] = _date.fromisoformat(default_before_end)
            st.session_state["after_start"] = _date.fromisoformat(default_after_start)
            st.session_state["after_end"] = _date.fromisoformat(default_after_end)
            st.session_state["index_choice"] = default_index
            st.session_state["threshold"] = default_threshold
            st.session_state.pop("_auto_run_done", None)

        with st.expander("Bounding Box (WGS84)", expanded=False):
            col_w, col_e = st.columns(2)
            col_s, col_n = st.columns(2)
            west = col_w.number_input("West", key="west", format="%.4f")
            east = col_e.number_input("East", key="east", format="%.4f")
            south = col_s.number_input("South", key="south", format="%.4f")
            north = col_n.number_input("North", key="north", format="%.4f")
        bbox = (west, south, east, north)

        st.subheader("Before Date Range")
        before_start = st.date_input("Start", key="before_start")
        before_end = st.date_input("End", key="before_end")

        st.subheader("After Date Range")
        after_start = st.date_input("Start", key="after_start")
        after_end = st.date_input("End", key="after_end")

        max_cloud = st.slider("Max cloud cover %", 0, 100, 20, step=5)

        st.subheader("Display")
        _INDEX_LABELS = {"ndvi": "NDVI — Vegetation", "ndbi": "NDBI — Built-up", "mndwi": "MNDWI — Water", "evi": "EVI — Vegetation (dense canopy)"}
        index_choice = st.radio(
            "Change index",
            options=list(INDEX_FUNCTIONS.keys()),
            format_func=lambda k: _INDEX_LABELS[k],
            key="index_choice",
        )
        auto_threshold = st.checkbox("Auto threshold (Otsu)", value=False, key="auto_threshold",
            help="Automatically compute the optimal change threshold from the data.")
        if not auto_threshold:
            st.slider(
                "Change threshold",
                min_value=0.01,
                max_value=0.30,
                step=0.01,
                key="threshold",
                help="Minimum change magnitude to classify as gain/loss.",
            )
        colormap = st.selectbox(
            "Colormap",
            ["RdBu", "RdYlBu", "PiYG"],
            key="colormap",
            help="Color scheme for the change heatmap.",
        )
        gamma = st.slider(
            "Gamma correction",
            min_value=0.5,
            max_value=1.5,
            step=0.05,
            value=0.85,
            key="gamma",
            help="Adjust brightness of true-color images. Lower = brighter.",
        )
        show_overture = st.checkbox("Show Overture Maps layers", value=True)

        run_button = st.button("Analyze Change", type="primary", width="stretch")

        st.divider()
        if st.button("Clear Disk Cache", type="secondary"):
            import shutil
            cache_dir = Path(__file__).parent / "cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            st.session_state["_results"] = {}
            st.toast("Cache cleared!")
            st.rerun()

    # ── Main Panel ────────────────────────────────────────────────────────────
    before_range = f"{before_start}/{before_end}"
    after_range = f"{after_start}/{after_end}"

    cache_key = f"{bbox}|{before_range}|{after_range}|{max_cloud}"

    # Initialize per-preset results store
    if "_results" not in st.session_state:
        st.session_state["_results"] = {}

    # Load cached results for current key if available
    cached = st.session_state["_results"].get(cache_key, {})
    for k in ("before_scene", "after_scene", "before_bands", "after_bands", "overture"):
        if k in cached:
            st.session_state[k] = cached[k]
        else:
            st.session_state.pop(k, None)

    has_data = "before_scene" in st.session_state and "after_scene" in st.session_state

    # Auto-run analysis when switching to an uncached preset (including first load)
    auto_run = False
    if not has_data and not st.session_state.get("_auto_run_done"):
        if preset_choice != "Custom…":
            auto_run = True
            st.session_state["_auto_run_done"] = True

    # Show instructions when no data is available and button not clicked
    if not has_data and not run_button and not auto_run:
        st.info(
            "Select a preset location or enter custom coordinates, choose date ranges, "
            "and click **Analyze Change** to begin."
        )
        return

    # Validate bbox inputs
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        st.error("Longitude must be between -180 and 180.")
        return
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        st.error("Latitude must be between -90 and 90.")
        return
    if west >= east:
        st.error(f"West ({west:.4f}) must be less than East ({east:.4f}).")
        return
    if south >= north:
        st.error(f"South ({south:.4f}) must be less than North ({north:.4f}).")
        return

    # ── Memory guard: estimate pixel count and abort if bbox is too large ──
    bbox_width_deg = east - west
    bbox_height_deg = north - south
    center_lat = (south + north) / 2.0
    target_res = 10  # metres
    pixels_per_band = (
        bbox_width_deg * bbox_height_deg
        * math.cos(math.radians(center_lat))
        * (111_000 / target_res) ** 2
    )
    num_bands = len(ALL_BAND_KEYS)  # 5
    num_dates = 2  # before + after
    # Peak memory estimate: raw uint16 bands (2 bytes) + float32 intermediates
    # during index computation (~4 bytes each, with index + delta alive = ~2x).
    # Conservative multiplier: 8 bytes/pixel accounts for peak working set.
    bytes_per_pixel = 8  # accounts for float32 index intermediates
    estimated_mb = pixels_per_band * num_bands * num_dates * bytes_per_pixel / (1024 ** 2)
    max_mb = 500
    if estimated_mb > max_mb:
        st.error(
            f"**Bounding box too large** — estimated memory usage is "
            f"{estimated_mb:,.0f} MB ({max_mb} MB limit). "
            f"Please shrink your bounding box and try again."
        )
        return

    # ── Fetch data concurrently ──────────────────────────────────────────────
    needs_before = "before_scene" not in st.session_state
    needs_after = "after_scene" not in st.session_state
    needs_overture = show_overture and "overture" not in st.session_state

    if needs_before or needs_after or needs_overture:
        with st.status("Analyzing change detection…", expanded=True) as status:

            def _fetch_date(date_range):
                """Search + load bands for one date range. Returns (scene, bands) or raises."""
                scenes = search_scenes(bbox=bbox, date_range=date_range, max_cloud_cover=max_cloud)
                if not scenes:
                    raise RuntimeError(f"No scenes found with cloud cover < {max_cloud:.0f}%")
                scene = scenes[0]
                bands = load_bands(scene=scene, bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=10)
                return scene, bands

            with ThreadPoolExecutor(max_workers=3) as executor:
                future_before = executor.submit(_fetch_date, before_range) if needs_before else None
                future_after = executor.submit(_fetch_date, after_range) if needs_after else None
                future_overture = executor.submit(get_overture_context, bbox=bbox) if needs_overture else None

                if future_before is not None:
                    st.write("Searching for before-period scenes…")
                    try:
                        scene, bands = future_before.result()
                        st.session_state["before_scene"] = scene
                        st.session_state["before_bands"] = bands
                        st.write(f"Before scene loaded: {scene['id']}")
                    except Exception as exc:
                        st.error(f"Failed to fetch before data: {exc}")
                        status.update(label="Analysis failed", state="error")
                        return

                if future_after is not None:
                    st.write("Searching for after-period scenes…")
                    try:
                        scene, bands = future_after.result()
                        st.session_state["after_scene"] = scene
                        st.session_state["after_bands"] = bands
                        st.write(f"After scene loaded: {scene['id']}")
                    except Exception as exc:
                        st.error(f"Failed to fetch after data: {exc}")
                        status.update(label="Analysis failed", state="error")
                        return

                if future_overture is not None:
                    st.write("Fetching Overture Maps context…")
                    try:
                        st.session_state["overture"] = future_overture.result(timeout=30)
                        st.write("Overture context loaded")
                    except TimeoutError:
                        logger.warning("Overture context fetch timed out")
                        st.write("Overture context timed out — proceeding without it")
                        st.session_state["overture"] = None
                    except Exception as exc:
                        logger.warning("Overture context fetch failed: %s", exc)
                        st.write("Overture context unavailable")
                        st.session_state["overture"] = None

            status.update(label="Analysis complete!", state="complete", expanded=False)

        # Cache results for this preset key
        st.session_state["_results"][cache_key] = {
            k: st.session_state[k]
            for k in ("before_scene", "after_scene", "before_bands", "after_bands", "overture")
            if k in st.session_state
        }

    before_scene = st.session_state["before_scene"]
    after_scene = st.session_state["after_scene"]
    before_bands = st.session_state["before_bands"]
    after_bands = st.session_state["after_bands"]
    overture = st.session_state.get("overture") if show_overture else None

    # ── Compute indices ───────────────────────────────────────────────────────
    before_index = compute_index_for_bands(index_choice, before_bands)
    after_index = compute_index_for_bands(index_choice, after_bands)
    delta = compute_change(before=before_index, after=after_index)
    if st.session_state.get("auto_threshold", False):
        from src.indices import compute_adaptive_threshold
        THRESHOLD = compute_adaptive_threshold(delta)
        st.session_state["threshold"] = THRESHOLD
    else:
        THRESHOLD = st.session_state.get("threshold", 0.10)

    # ── NDVI saturation warning ──────────────────────────────────────────
    if index_choice == "ndvi":
        p90 = float(np.percentile(before_index, 90))
        if p90 > 0.75:
            st.warning(
                f"NDVI appears saturated in this region (90th percentile: {p90:.2f}). "
                f"Dense vegetation compresses NDVI's dynamic range. "
                f"Consider switching to **EVI** for better sensitivity to canopy changes."
            )

    # ── Build images ──────────────────────────────────────────────────────────
    before_img = true_color_image(
        before_bands["red"], before_bands["green"], before_bands["blue"], gamma=gamma,
    )
    after_img = true_color_image(
        after_bands["red"], after_bands["green"], after_bands["blue"], gamma=gamma,
    )
    heatmap_img = index_to_rgba(downscale_array(delta, max_dim=800), threshold=THRESHOLD, colormap=colormap)

    # ── Panel A: True Color Comparison ────────────────────────────────────────
    st.subheader("Panel A — True Color Comparison")
    col_before, col_after = st.columns(2)
    before_labeled = label_image(before_img, f"Before — {before_scene['datetime'][:10]}")
    after_labeled = label_image(after_img, f"After — {after_scene['datetime'][:10]}")
    col_before.image(before_labeled, use_container_width=True)
    col_after.image(after_labeled, use_container_width=True)

    # Downscale overlay images to cap folium HTML payload size
    MAX_OVERLAY_DIM = 800
    def _downscale(img):
        w, h = img.size
        if max(w, h) <= MAX_OVERLAY_DIM:
            return img
        scale = MAX_OVERLAY_DIM / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ── Panel B: Change Detection Heatmap ────────────────────────────────────
    st.subheader(f"Panel B — {INDEX_FUNCTIONS[index_choice][0]} Change Heatmap")

    panel_b_map = build_folium_map(
        bbox=bbox,
        before_image=_downscale(before_img),
        after_image=_downscale(after_img),
        heatmap_image=heatmap_img,
        show_heatmap=True,
        show_overture=False,
        enable_draw=True,
    )
    map_data = st_folium(panel_b_map, width="100%", height=500, returned_objects=["last_active_drawing"])

    # ── Panel C: Overture Maps Context ───────────────────────────────────────
    if show_overture:
        st.subheader("Panel C — Overture Maps Context")
        if not overture or all(
            len(overture.get(layer, [])) == 0
            for layer in ("building", "segment", "place")
        ):
            st.info("No Overture Maps data available for this region. "
                     "This may be due to sparse coverage or a fetch timeout.")
        else:
            st.caption(
                f"{len(overture.get('building', []))} buildings, "
                f"{len(overture.get('segment', []))} road segments, "
                f"{len(overture.get('place', []))} places"
            )
            panel_c_map = build_folium_map(
                bbox=bbox,
                heatmap_image=heatmap_img,
                overture_context=overture,
                show_heatmap=True,
                show_overture=True,
            )
            st_folium(panel_c_map, width="100%", height=500)

    # ── Panel D: Summary Statistics ──────────────────────────────────────────
    st.subheader("Panel D — Summary Statistics")

    area_deg2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    center_lat_rad = math.radians((bbox[1] + bbox[3]) / 2)
    area_km2 = area_deg2 * 111.0 * 111.0 * math.cos(center_lat_rad)

    pct_gain = float(np.mean(delta > THRESHOLD) * 100)
    pct_loss = float(np.mean(delta < -THRESHOLD) * 100)
    pct_unchanged = 100.0 - pct_gain - pct_loss

    stat_cols = st.columns(4)
    stat_cols[0].metric("Area analyzed", f"{area_km2:.1f} km²")
    stat_cols[1].metric(f"{INDEX_FUNCTIONS[index_choice][0]} gain", f"{pct_gain:.1f}%")
    stat_cols[2].metric(f"{INDEX_FUNCTIONS[index_choice][0]} loss", f"{pct_loss:.1f}%")
    stat_cols[3].metric("Unchanged", f"{pct_unchanged:.1f}%")
    threshold_mode = "auto / Otsu" if st.session_state.get("auto_threshold", False) else "manual"
    st.caption(f"Threshold: {THRESHOLD:.3f} ({threshold_mode})")

    # All indices summary (spec requirement: show all three simultaneously)
    st.markdown("**All Indices Summary**")
    idx_cols = st.columns(len(INDEX_FUNCTIONS))
    for col, (idx_key, (idx_name, _, _)) in zip(idx_cols, INDEX_FUNCTIONS.items()):
        if idx_key == index_choice:
            col.metric(f"{idx_name} gain", f"{pct_gain:.1f}%")
            col.metric(f"{idx_name} loss", f"{pct_loss:.1f}%")
        else:
            idx_before = compute_index_for_bands(idx_key, before_bands)
            idx_after = compute_index_for_bands(idx_key, after_bands)
            idx_delta = compute_change(before=idx_before, after=idx_after)
            idx_gain = float(np.mean(idx_delta > THRESHOLD) * 100)
            idx_loss = float(np.mean(idx_delta < -THRESHOLD) * 100)
            col.metric(f"{idx_name} gain", f"{idx_gain:.1f}%")
            col.metric(f"{idx_name} loss", f"{idx_loss:.1f}%")

    tiff_bytes = create_geotiff(
        delta,
        bbox,
        index_type=index_choice,
        before_date=before_scene["datetime"][:10],
        after_date=after_scene["datetime"][:10],
        before_scene_id=before_scene["id"],
        after_scene_id=after_scene["id"],
    )
    st.download_button(
        label="Download Change Raster (.tif)",
        data=tiff_bytes,
        file_name=f"change_{index_choice}_{before_scene['datetime'][:10]}_{after_scene['datetime'][:10]}.tif",
        mime="image/tiff",
    )

    detail_cols = st.columns(2)
    detail_cols[0].write(f"**Before:** {before_scene['id']}  \n"
                         f"Date: {before_scene['datetime'][:10]}  \n"
                         f"Cloud: {before_scene['cloud_cover']:.1f}%")
    detail_cols[1].write(f"**After:** {after_scene['id']}  \n"
                         f"Date: {after_scene['datetime'][:10]}  \n"
                         f"Cloud: {after_scene['cloud_cover']:.1f}%")

    st.pyplot(change_histogram(delta, threshold=THRESHOLD, index_name=INDEX_FUNCTIONS[index_choice][0]))

    # Read drawn geometry and update bbox session state
    if map_data and map_data.get("last_active_drawing"):
        drawing = map_data["last_active_drawing"]
        geom = drawing.get("geometry")
        if geom and geom.get("type") in ("Polygon", "Rectangle"):
            coords = geom["coordinates"][0]  # outer ring
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            drawn_west = min(lons)
            drawn_east = max(lons)
            drawn_south = min(lats)
            drawn_north = max(lats)

            # Validate against memory guard
            drawn_width = drawn_east - drawn_west
            drawn_height = drawn_north - drawn_south
            drawn_center_lat = (drawn_south + drawn_north) / 2.0
            drawn_pixels = (
                drawn_width * drawn_height
                * math.cos(math.radians(drawn_center_lat))
                * (111_000 / 10) ** 2
            )
            drawn_mb = drawn_pixels * len(ALL_BAND_KEYS) * 2 * 8 / (1024 ** 2)

            if drawn_mb <= 500:
                st.session_state["west"] = round(drawn_west, 4)
                st.session_state["east"] = round(drawn_east, 4)
                st.session_state["south"] = round(drawn_south, 4)
                st.session_state["north"] = round(drawn_north, 4)
                st.info(
                    f"AOI updated from drawing: "
                    f"({drawn_west:.4f}, {drawn_south:.4f}, {drawn_east:.4f}, {drawn_north:.4f}). "
                    f"Click **Analyze Change** to re-run."
                )
            else:
                st.warning(
                    f"Drawn area too large (~{drawn_mb:,.0f} MB). "
                    f"Please draw a smaller region (limit: 500 MB)."
                )


if __name__ == "__main__":
    main()
