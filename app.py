"""Sentinel-2 Change Detection Explorer — Streamlit application.

Run with: streamlit run app.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_folium import st_folium

# Ensure src/ is on path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.sentinel import load_bands, search_scenes
from src.indices import compute_change, compute_mndwi, compute_ndbi, compute_ndvi
from src.overture import get_overture_context
from src.visualization import (
    build_folium_map,
    index_to_rgba,
    true_color_image,
)

PRESETS_FILE = Path(__file__).parent / "config" / "presets.json"

INDEX_FUNCTIONS = {
    "ndvi": ("NDVI — Vegetation", compute_ndvi, ["nir", "red"]),
    "ndbi": ("NDBI — Built-up", compute_ndbi, ["swir16", "nir"]),
    "mndwi": ("MNDWI — Water", compute_mndwi, ["green", "swir16"]),
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
    return fn(bands[band_order[0]], bands[band_order[1]])


def fetch_scene_data(
    bbox: tuple[float, float, float, float],
    date_range: str,
    max_cloud: float,
    label: str,
) -> tuple[dict | None, dict[str, np.ndarray] | None, str]:
    """Search for and load a Sentinel-2 scene.

    Returns:
        Tuple of (scene_meta, bands_dict, status_message).
        scene_meta and bands_dict are None if no scene found.
    """
    with st.spinner(f"Searching for {label} scene…"):
        scenes = search_scenes(bbox=bbox, date_range=date_range, max_cloud_cover=max_cloud)

    if not scenes:
        return None, None, f"No {label} scenes found with cloud cover < {max_cloud:.0f}%"

    scene = scenes[0]
    try:
        with st.spinner(f"Loading {label} bands from S3…"):
            bands = load_bands(scene=scene, bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=10)
    except Exception as exc:
        return None, None, f"Failed to load {label} bands: {exc}"

    return scene, bands, f"Loaded {label}: {scene['id']} ({scene['cloud_cover']:.1f}% cloud)"


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Sentinel-2 Change Explorer",
        page_icon="🛰️",
        layout="wide",
    )
    st.title("🛰️ Sentinel-2 Change Detection Explorer")
    st.caption(
        "Compare satellite imagery across two dates to detect vegetation loss, "
        "urbanization, and water change using Sentinel-2 L2A imagery."
    )

    presets = load_presets()
    preset_names = ["Custom…"] + [p["name"] for p in presets]

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Location & Dates")

        # Initialize widget state on first load
        if "_last_preset" not in st.session_state:
            st.session_state["_last_preset"] = None
            st.session_state["west"] = -115.32
            st.session_state["east"] = -115.08
            st.session_state["south"] = 36.08
            st.session_state["north"] = 36.28
            from datetime import date as _date
            st.session_state["before_start"] = _date.fromisoformat("2019-05-01")
            st.session_state["before_end"] = _date.fromisoformat("2019-07-31")
            st.session_state["after_start"] = _date.fromisoformat("2023-05-01")
            st.session_state["after_end"] = _date.fromisoformat("2023-07-31")

        preset_choice = st.selectbox("Preset location", preset_names)
        if preset_choice != "Custom…":
            preset = next(p for p in presets if p["name"] == preset_choice)
            default_bbox = preset["bbox"]
            default_before_start = preset["before_range"][0]
            default_before_end = preset["before_range"][1]
            default_after_start = preset["after_range"][0]
            default_after_end = preset["after_range"][1]
            default_index = preset.get("default_index", "ndvi")
            if "notes" in preset:
                st.info(preset["notes"])
        else:
            default_bbox = [-115.32, 36.08, -115.08, 36.28]
            default_before_start, default_before_end = "2019-05-01", "2019-07-31"
            default_after_start, default_after_end = "2023-05-01", "2023-07-31"
            default_index = "ndvi"

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

        st.subheader("Bounding Box (WGS84)")
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
        index_choice = st.radio(
            "Change index",
            options=list(INDEX_FUNCTIONS.keys()),
            format_func=lambda k: INDEX_FUNCTIONS[k][0],
            index=list(INDEX_FUNCTIONS.keys()).index(default_index),
        )
        show_overture = st.checkbox("Show Overture Maps layers", value=True)

        run_button = st.button("Analyze Change", type="primary", use_container_width=True)

    # ── Main Panel ────────────────────────────────────────────────────────────
    if not run_button:
        st.info(
            "Select a preset location or enter custom coordinates, choose date ranges, "
            "and click **Analyze Change** to begin."
        )
        return

    before_range = f"{before_start}/{before_end}"
    after_range = f"{after_start}/{after_end}"

    # Session-state caching: re-fetch only if inputs changed
    cache_key = f"{bbox}|{before_range}|{after_range}|{max_cloud}"
    cached = st.session_state.get("cache_key")
    if cached != cache_key:
        st.session_state["cache_key"] = cache_key
        st.session_state.pop("before_scene", None)
        st.session_state.pop("after_scene", None)
        st.session_state.pop("before_bands", None)
        st.session_state.pop("after_bands", None)
        st.session_state.pop("overture", None)

    if "before_scene" not in st.session_state:
        scene, bands, msg = fetch_scene_data(bbox, before_range, max_cloud, "before")
        if scene is None:
            st.error(msg)
            return
        st.session_state["before_scene"] = scene
        st.session_state["before_bands"] = bands
        st.success(msg)

    if "after_scene" not in st.session_state:
        scene, bands, msg = fetch_scene_data(bbox, after_range, max_cloud, "after")
        if scene is None:
            st.error(msg)
            return
        st.session_state["after_scene"] = scene
        st.session_state["after_bands"] = bands
        st.success(msg)

    before_scene = st.session_state["before_scene"]
    after_scene = st.session_state["after_scene"]
    before_bands = st.session_state["before_bands"]
    after_bands = st.session_state["after_bands"]

    if "overture" not in st.session_state and show_overture:
        with st.spinner("Fetching Overture Maps context…"):
            st.session_state["overture"] = get_overture_context(bbox=bbox)

    overture = st.session_state.get("overture") if show_overture else None

    # ── Compute indices ───────────────────────────────────────────────────────
    before_index = compute_index_for_bands(index_choice, before_bands)
    after_index = compute_index_for_bands(index_choice, after_bands)
    delta = compute_change(before=before_index, after=after_index)

    # ── Build images ──────────────────────────────────────────────────────────
    before_img = true_color_image(
        before_bands["red"], before_bands["green"], before_bands["blue"]
    )
    after_img = true_color_image(
        after_bands["red"], after_bands["green"], after_bands["blue"]
    )
    heatmap_img = index_to_rgba(delta, threshold=0.05)

    # ── Panel A: True Color Comparison ────────────────────────────────────────
    st.subheader("Panel A — True Color Comparison")
    col_before, col_after = st.columns(2)
    col_before.image(before_img, caption=f"Before — {before_scene['datetime'][:10]}", use_container_width=True)
    col_after.image(after_img, caption=f"After — {after_scene['datetime'][:10]}", use_container_width=True)

    # ── Panel B+C: Change Heatmap + Overture Context ──────────────────────────
    st.subheader(f"Panel B+C — {INDEX_FUNCTIONS[index_choice][0]} Change Heatmap")
    folium_map = build_folium_map(
        bbox=bbox,
        before_image=before_img,
        after_image=after_img,
        heatmap_image=heatmap_img,
        overture_context=overture,
        show_heatmap=True,
        show_overture=show_overture,
    )
    st_folium(folium_map, width="100%", height=500, returned_objects=[])

    # ── Panel D: Summary Statistics ───────────────────────────────────────────
    st.subheader("Panel D — Summary Statistics")

    area_deg2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    center_lat_rad = math.radians((bbox[1] + bbox[3]) / 2)
    area_km2 = area_deg2 * 111.0 * 111.0 * math.cos(center_lat_rad)

    THRESHOLD = 0.05
    pct_gain = float(np.mean(delta > THRESHOLD) * 100)
    pct_loss = float(np.mean(delta < -THRESHOLD) * 100)
    pct_unchanged = 100.0 - pct_gain - pct_loss

    stat_cols = st.columns(4)
    stat_cols[0].metric("Area analyzed", f"{area_km2:.1f} km²")
    stat_cols[1].metric(f"{INDEX_FUNCTIONS[index_choice][0]} gain", f"{pct_gain:.1f}%")
    stat_cols[2].metric(f"{INDEX_FUNCTIONS[index_choice][0]} loss", f"{pct_loss:.1f}%")
    stat_cols[3].metric("Unchanged", f"{pct_unchanged:.1f}%")

    detail_cols = st.columns(2)
    detail_cols[0].write(f"**Before:** {before_scene['id']}  \n"
                         f"Date: {before_scene['datetime'][:10]}  \n"
                         f"Cloud: {before_scene['cloud_cover']:.1f}%")
    detail_cols[1].write(f"**After:** {after_scene['id']}  \n"
                         f"Date: {after_scene['datetime'][:10]}  \n"
                         f"Cloud: {after_scene['cloud_cover']:.1f}%")

    if overture:
        st.caption(
            f"Overture context: {len(overture.get('building', []))} buildings, "
            f"{len(overture.get('segment', []))} road segments, "
            f"{len(overture.get('place', []))} places"
        )


if __name__ == "__main__":
    main()
