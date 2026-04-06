# Multi-Index Change Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a semantic change classification layer that combines NDVI, NDBI, and MNDWI deltas to explain *why* an area changed (urbanization, deforestation, flooding, recovery), displayed as a toggleable RGBA overlay on the folium map with legend.

**Architecture:** New `classify_change()` function in `src/indices.py` applies numpy boolean rules to three index deltas, producing a categorical integer array. New `classification_to_rgba()` in `src/visualization.py` converts categories to colored RGBA. The overlay is added to the Panel B folium map as a toggleable layer. A sidebar checkbox controls visibility.

**Tech Stack:** numpy, PIL, folium (existing dependencies only)

---

### Task 1: Add `classify_change()` with TDD

**Files:**
- Create: `tests/unit/test_classification.py`
- Modify: `src/indices.py:209` (append new function)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_classification.py`:

```python
"""Tests for multi-index change classification."""
from __future__ import annotations

import numpy as np
import pytest

from src.indices import classify_change

# Category constants (must match implementation)
UNCHANGED = 0
URBAN_CONVERSION = 1
VEGETATION_LOSS = 2
FLOODING = 3
VEGETATION_GAIN = 4


class TestClassifyChange:
    """Tests for classify_change()."""

    def _make_deltas(self, size=(10, 10), ndvi=0.0, ndbi=0.0, mndwi=0.0):
        """Helper: create uniform delta arrays."""
        return (
            np.full(size, ndvi, dtype=np.float32),
            np.full(size, ndbi, dtype=np.float32),
            np.full(size, mndwi, dtype=np.float32),
        )

    def test_urban_conversion(self):
        """NDVI down AND NDBI up -> Urban Conversion."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.15, mndwi=0.0)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == URBAN_CONVERSION)

    def test_vegetation_loss(self):
        """NDVI down AND MNDWI stable -> Vegetation Loss."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.0, mndwi=0.0)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == VEGETATION_LOSS)

    def test_flooding(self):
        """MNDWI up AND NDVI down -> Flooding."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.0, mndwi=0.2)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == FLOODING)

    def test_vegetation_gain(self):
        """NDVI up -> Vegetation Gain."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=0.2, ndbi=0.0, mndwi=0.0)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == VEGETATION_GAIN)

    def test_unchanged(self):
        """Small deltas -> Unchanged."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=0.01, ndbi=-0.01, mndwi=0.02)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == UNCHANGED)

    def test_mixed_pixels(self):
        """Different pixels get different categories."""
        ndvi_d = np.array([[-0.2, 0.2, -0.2, 0.01]], dtype=np.float32)
        ndbi_d = np.array([[0.15, 0.0, 0.0, 0.0]], dtype=np.float32)
        mndwi_d = np.array([[0.0, 0.0, 0.2, 0.0]], dtype=np.float32)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert result[0, 0] == URBAN_CONVERSION
        assert result[0, 1] == VEGETATION_GAIN
        assert result[0, 2] == FLOODING
        assert result[0, 3] == UNCHANGED

    def test_nan_handling(self):
        """NaN pixels should be classified as UNCHANGED."""
        ndvi_d = np.array([[np.nan, -0.2]], dtype=np.float32)
        ndbi_d = np.array([[0.0, 0.15]], dtype=np.float32)
        mndwi_d = np.array([[0.0, 0.0]], dtype=np.float32)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert result[0, 0] == UNCHANGED
        assert result[0, 1] == URBAN_CONVERSION

    def test_output_dtype_and_shape(self):
        """Output should be uint8 with same shape as input."""
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(size=(5, 7))
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert result.shape == (5, 7)
        assert result.dtype == np.uint8

    def test_priority_urban_over_vegetation_loss(self):
        """Urban conversion should take priority over vegetation loss.

        When NDVI is down and NDBI is up, it's urban even though NDVI-down
        alone would be vegetation loss.
        """
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.15, mndwi=0.0)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == URBAN_CONVERSION)

    def test_priority_flooding_over_vegetation_loss(self):
        """Flooding should take priority over vegetation loss.

        When NDVI is down and MNDWI is up, it's flooding even though NDVI-down
        alone would be vegetation loss.
        """
        ndvi_d, ndbi_d, mndwi_d = self._make_deltas(ndvi=-0.2, ndbi=0.0, mndwi=0.2)
        result = classify_change(ndvi_d, ndbi_d, mndwi_d, threshold=0.10)
        assert np.all(result == FLOODING)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_classification.py -v`
Expected: FAIL — `ImportError: cannot import name 'classify_change' from 'src.indices'`

- [ ] **Step 3: Implement `classify_change()` in `src/indices.py`**

Append to end of `src/indices.py` (after line 208):

```python

# ── Multi-index change classification ────────────────────────────────────

# Category labels returned by classify_change()
UNCHANGED = 0
URBAN_CONVERSION = 1
VEGETATION_LOSS = 2
FLOODING = 3
VEGETATION_GAIN = 4


def classify_change(
    ndvi_delta: np.ndarray,
    ndbi_delta: np.ndarray,
    mndwi_delta: np.ndarray,
    threshold: float = 0.10,
) -> np.ndarray:
    """Classify each pixel's change type using multi-index decision rules.

    Rules are applied in priority order — earlier rules win when multiple
    conditions match:

    1. NDVI down AND NDBI up  → Urban Conversion
    2. MNDWI up AND NDVI down → Flooding / Water Gain
    3. NDVI down (alone)      → Vegetation Loss / Deforestation
    4. NDVI up                → Vegetation Gain / Recovery
    5. Everything else        → Unchanged

    Args:
        ndvi_delta: 2D float32 array of NDVI change (after - before).
        ndbi_delta: 2D float32 array of NDBI change (after - before).
        mndwi_delta: 2D float32 array of MNDWI change (after - before).
        threshold: Minimum absolute change to count as significant.

    Returns:
        uint8 array with category codes (0-4).
    """
    result = np.full(ndvi_delta.shape, UNCHANGED, dtype=np.uint8)

    ndvi_down = ndvi_delta < -threshold
    ndvi_up = ndvi_delta > threshold
    ndbi_up = ndbi_delta > threshold
    mndwi_up = mndwi_delta > threshold

    # Apply in reverse priority so higher-priority rules overwrite
    result[ndvi_up] = VEGETATION_GAIN
    result[ndvi_down] = VEGETATION_LOSS
    result[ndvi_down & mndwi_up] = FLOODING
    result[ndvi_down & ndbi_up] = URBAN_CONVERSION

    # NaN pixels → unchanged
    nan_mask = ~np.isfinite(ndvi_delta) | ~np.isfinite(ndbi_delta) | ~np.isfinite(mndwi_delta)
    result[nan_mask] = UNCHANGED

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_classification.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_classification.py src/indices.py
git commit -m "feat: add multi-index change classification function with tests"
```

---

### Task 2: Add `classification_to_rgba()` with TDD

**Files:**
- Create: `tests/unit/test_classification_viz.py`
- Modify: `src/visualization.py:466` (append new function)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_classification_viz.py`:

```python
"""Tests for classification RGBA rendering."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.indices import UNCHANGED, URBAN_CONVERSION, VEGETATION_LOSS, FLOODING, VEGETATION_GAIN
from src.visualization import classification_to_rgba

# Expected colors (R, G, B) from the spec
EXPECTED_COLORS = {
    URBAN_CONVERSION: (255, 165, 0),    # Orange
    VEGETATION_LOSS: (220, 38, 38),      # Red
    FLOODING: (59, 130, 246),            # Blue
    VEGETATION_GAIN: (34, 197, 94),      # Green
}


class TestClassificationToRgba:
    """Tests for classification_to_rgba()."""

    def test_returns_rgba_image(self):
        cats = np.array([[UNCHANGED, URBAN_CONVERSION]], dtype=np.uint8)
        img = classification_to_rgba(cats)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"
        assert img.size == (2, 1)

    def test_unchanged_is_transparent(self):
        cats = np.full((3, 3), UNCHANGED, dtype=np.uint8)
        img = classification_to_rgba(cats)
        arr = np.array(img)
        assert np.all(arr[:, :, 3] == 0)

    def test_category_colors(self):
        """Each category should produce the correct RGB color."""
        for cat, (r, g, b) in EXPECTED_COLORS.items():
            cats = np.full((1, 1), cat, dtype=np.uint8)
            img = classification_to_rgba(cats, alpha=1.0)
            arr = np.array(img)
            assert arr[0, 0, 0] == r, f"Category {cat}: red mismatch"
            assert arr[0, 0, 1] == g, f"Category {cat}: green mismatch"
            assert arr[0, 0, 2] == b, f"Category {cat}: blue mismatch"
            assert arr[0, 0, 3] == 255, f"Category {cat}: should be opaque"

    def test_alpha_parameter(self):
        cats = np.full((1, 1), VEGETATION_GAIN, dtype=np.uint8)
        img = classification_to_rgba(cats, alpha=0.5)
        arr = np.array(img)
        assert arr[0, 0, 3] == 127  # int(0.5 * 255)

    def test_mixed_categories(self):
        cats = np.array([
            [UNCHANGED, URBAN_CONVERSION],
            [FLOODING, VEGETATION_GAIN],
        ], dtype=np.uint8)
        img = classification_to_rgba(cats)
        arr = np.array(img)
        # UNCHANGED pixel should be transparent
        assert arr[0, 0, 3] == 0
        # All other pixels should be opaque (default alpha=0.7 -> 178)
        assert arr[0, 1, 3] == 178
        assert arr[1, 0, 3] == 178
        assert arr[1, 1, 3] == 178
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_classification_viz.py -v`
Expected: FAIL — `ImportError: cannot import name 'classification_to_rgba' from 'src.visualization'`

- [ ] **Step 3: Implement `classification_to_rgba()` in `src/visualization.py`**

Append to end of `src/visualization.py` (after line 465):

```python


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_classification_viz.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_classification_viz.py src/visualization.py
git commit -m "feat: add classification-to-RGBA rendering function with tests"
```

---

### Task 3: Integrate classification into app.py

**Files:**
- Modify: `app.py:24` (add imports)
- Modify: `app.py:329-342` (add sidebar checkbox)
- Modify: `app.py:544-572` (compute classification + build image)
- Modify: `app.py:594-603` (add classification layer to Panel B map)

- [ ] **Step 1: Add imports to `app.py`**

At `app.py:24`, change:

```python
from src.indices import compute_change, compute_evi, compute_mndwi, compute_ndbi, compute_ndvi
```

to:

```python
from src.indices import (
    classify_change,
    compute_change,
    compute_evi,
    compute_mndwi,
    compute_ndbi,
    compute_ndvi,
    FLOODING,
    UNCHANGED,
    URBAN_CONVERSION,
    VEGETATION_GAIN,
    VEGETATION_LOSS,
)
```

At `app.py:28-35`, change:

```python
from src.visualization import (
    build_folium_map,
    change_histogram,
    downscale_array,
    index_to_rgba,
    label_image,
    true_color_image,
)
```

to:

```python
from src.visualization import (
    build_folium_map,
    change_histogram,
    classification_to_rgba,
    downscale_array,
    index_to_rgba,
    label_image,
    true_color_image,
)
```

- [ ] **Step 2: Add sidebar checkbox**

After the `show_overture` checkbox (line 342), add:

```python
        show_classification = st.checkbox(
            "Show change classification",
            value=True,
            key="show_classification",
            help="Multi-index classification overlay: urbanization, vegetation loss, flooding, recovery.",
        )
```

- [ ] **Step 3: Compute classification after index computation**

After the heatmap image is built (after line 572 `heatmap_img = ...`), add:

```python
    # ── Multi-index change classification ────────────────────────────────
    classification_img = None
    if st.session_state.get("show_classification", True):
        ndvi_b = compute_index_for_bands("ndvi", before_bands)
        ndvi_a = compute_index_for_bands("ndvi", after_bands)
        ndbi_b = compute_index_for_bands("ndbi", before_bands)
        ndbi_a = compute_index_for_bands("ndbi", after_bands)
        mndwi_b = compute_index_for_bands("mndwi", before_bands)
        mndwi_a = compute_index_for_bands("mndwi", after_bands)
        ndvi_delta = compute_change(before=ndvi_b, after=ndvi_a)
        ndbi_delta = compute_change(before=ndbi_b, after=ndbi_a)
        mndwi_delta = compute_change(before=mndwi_b, after=mndwi_a)
        categories = classify_change(ndvi_delta, ndbi_delta, mndwi_delta, threshold=THRESHOLD)
        classification_img = classification_to_rgba(downscale_array(categories.astype(np.float32), max_dim=800).astype(np.uint8))
```

- [ ] **Step 4: Pass classification image to `build_folium_map()`**

Modify the `build_folium_map` function signature in `src/visualization.py:306` to accept a new parameter. Change:

```python
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
```

to:

```python
def build_folium_map(
    bbox: tuple[float, float, float, float],
    before_image: Optional[Image.Image] = None,
    after_image: Optional[Image.Image] = None,
    heatmap_image: Optional[Image.Image] = None,
    classification_image: Optional[Image.Image] = None,
    overture_context: Optional[dict[str, gpd.GeoDataFrame]] = None,
    show_heatmap: bool = True,
    show_overture: bool = True,
    enable_draw: bool = False,
) -> folium.Map:
```

Update the docstring `Args:` section to add:
```
        classification_image: RGBA PIL Image for change classification overlay.
```

After the heatmap legend block (after line 380), add the classification overlay and legend:

```python
    if classification_image is not None:
        _image_to_bounds_overlay(
            classification_image, bbox, name="Change Classification", opacity=1.0,
        ).add_to(m)

        classification_legend_html = """
        <div style="
            position: fixed;
            bottom: 30px; left: 30px;
            z-index: 1000;
            background: white;
            padding: 10px 14px;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            font-size: 13px;
            line-height: 1.6;
        ">
            <b>Classification</b><br>
            <span style="display:inline-block;width:16px;height:12px;background:rgb(255,165,0);border:1px solid #999;"></span> Urban Conversion<br>
            <span style="display:inline-block;width:16px;height:12px;background:rgb(220,38,38);border:1px solid #999;"></span> Vegetation Loss<br>
            <span style="display:inline-block;width:16px;height:12px;background:rgb(59,130,246);border:1px solid #999;"></span> Flooding / Water Gain<br>
            <span style="display:inline-block;width:16px;height:12px;background:rgb(34,197,94);border:1px solid #999;"></span> Vegetation Gain
        </div>
        """
        m.get_root().html.add_child(folium.Element(classification_legend_html))
```

Update the overlay_count calculation to include classification:

```python
    overlay_count = sum([
        before_image is not None,
        after_image is not None,
        heatmap_image is not None and show_heatmap,
        classification_image is not None,
        bool(overture_context is not None and show_overture),
    ])
```

- [ ] **Step 5: Pass classification image in Panel B map call**

In `app.py`, change the Panel B `build_folium_map()` call (around line 594):

```python
    panel_b_map = build_folium_map(
        bbox=bbox,
        before_image=_downscale(before_img),
        after_image=_downscale(after_img),
        heatmap_image=heatmap_img,
        show_heatmap=True,
        show_overture=False,
        enable_draw=True,
    )
```

to:

```python
    panel_b_map = build_folium_map(
        bbox=bbox,
        before_image=_downscale(before_img),
        after_image=_downscale(after_img),
        heatmap_image=heatmap_img,
        classification_image=classification_img,
        show_heatmap=True,
        show_overture=False,
        enable_draw=True,
    )
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/ -v --ignore=tests/integration --ignore=tests/perf --ignore=tests/e2e_visual_test.py`
Expected: All unit tests PASS

- [ ] **Step 7: Commit**

```bash
git add app.py src/visualization.py
git commit -m "feat: integrate multi-index change classification into UI and map"
```

---

### Task 4: Add classification summary to Panel D

**Files:**
- Modify: `app.py:691-692` (add classification breakdown after All Indices Summary)

- [ ] **Step 1: Add classification pixel counts after All Indices Summary**

After the All Indices Summary block (after line 691 in the current code), add:

```python
    # Classification breakdown
    if classification_img is not None:
        st.markdown("**Change Classification Breakdown**")
        cls_cols = st.columns(5)
        cls_labels = {
            UNCHANGED: "Unchanged",
            URBAN_CONVERSION: "Urban Conversion",
            VEGETATION_LOSS: "Vegetation Loss",
            FLOODING: "Flooding",
            VEGETATION_GAIN: "Vegetation Gain",
        }
        cls_total = categories.size
        for col, (code, label) in zip(cls_cols, cls_labels.items()):
            count = int(np.sum(categories == code))
            pct = count / cls_total * 100 if cls_total > 0 else 0.0
            col.metric(label, f"{pct:.1f}%")
```

Note: `categories` is the numpy array computed in the classification block earlier. It needs to be in scope — ensure it's defined before this code runs. If `show_classification` is False, `classification_img` is None and this block is skipped.

- [ ] **Step 2: Run all unit tests**

Run: `python -m pytest tests/ -v --ignore=tests/integration --ignore=tests/perf --ignore=tests/e2e_visual_test.py`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add classification breakdown metrics to Panel D"
```
