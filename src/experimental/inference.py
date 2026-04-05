"""Inference and visualization for the experimental LeJEPA feature.

This module will eventually load a trained ResNet-18 LeJEPA encoder from
HuggingFace, extract features for before/after scenes, and render PCA->RGB
feature images plus a learned change heatmap.

Right now (Phase 1) this file only contains the render stub so the app-side
plumbing can be wired and tested. Real model loading and feature extraction
land in Phase 6.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


def render_experimental_panel(
    before_bands: dict[str, Any],
    after_bands: dict[str, Any],
) -> None:
    """Render the experimental foundation-model panel in the Streamlit app.

    Phase 1: placeholder that confirms the module loads, the sidebar toggle
    gated it correctly, and the before/after bands flowed through from the
    app's session state. No model, no features, no PCA — that all lands in
    Phase 6.

    Args:
        before_bands: Dict mapping band key -> 2D uint16 numpy array for the
            before scene (output of ``src.sentinel.load_bands``).
        after_bands: Same shape/schema as before_bands, for the after scene.
    """
    st.markdown("### Experimental: Foundation Model (PoC)")
    st.info(
        "**Phase 1:** experimental module loaded successfully. "
        "The trained LeJEPA model and PCA->RGB feature visualization will land "
        "in Phase 6. For now this panel confirms the sidebar toggle, optional "
        "dependency gating, and the before/after band plumbing are all wired up."
    )

    # Tiny sanity readout so the user can see the bands actually arrived.
    try:
        before_shape = next(iter(before_bands.values())).shape
        after_shape = next(iter(after_bands.values())).shape
        cols = st.columns(2)
        cols[0].caption(
            f"Before scene bands: {sorted(before_bands.keys())} "
            f"@ shape {before_shape}"
        )
        cols[1].caption(
            f"After scene bands: {sorted(after_bands.keys())} "
            f"@ shape {after_shape}"
        )
    except (StopIteration, AttributeError):
        st.warning("Experimental panel received empty or malformed bands dict.")
