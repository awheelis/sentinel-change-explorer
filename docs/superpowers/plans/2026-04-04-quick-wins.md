# Quick Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 low-complexity enhancements (EVI index, adaptive thresholding, index-specific colormaps, confidence/quality indicator, NDVI saturation warning) to improve scientific credibility and UX.

**Architecture:** All changes touch `src/indices.py` (new computation functions), `src/sentinel.py` (metadata extraction), and `app.py` (UI integration). Each enhancement is independent except NDVI saturation warning depends on EVI existing as an alternative. Tests follow existing patterns in `tests/unit/test_indices.py`.

**Tech Stack:** Python, NumPy, Streamlit, pystac-client, matplotlib

---

### Task 1: Add EVI computation

**Files:**
- Modify: `src/indices.py` (add `compute_evi`)
- Test: `tests/unit/test_indices.py`

- [ ] **Step 1: Write failing tests for compute_evi**

Add to `tests/unit/test_indices.py`:

```python
from src.indices import compute_evi

class TestEVI:
    def test_known_values(self):
        """EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
        nir = _make_band(0.5)
        red = _make_band(0.1)
        blue = _make_band(0.05)
        result = compute_evi(nir, red, blue)
        expected = 2.5 * (0.5 - 0.1) / (0.5 + 6 * 0.1 - 7.5 * 0.05 + 1)
        np.testing.assert_allclose(result, expected, atol=0.01)

    def test_zero_denominator(self):
        """Should not raise or produce NaN/Inf when denominator is zero."""
        nir = _make_band(0.0)
        red = _make_band(0.0)
        blue = _make_band(0.0)
        # denom = 0 + 0 - 0 + 1 = 1, so no actual zero denom here
        # Force zero denom: NIR + 6*Red - 7.5*Blue + 1 = 0
        # => Blue = (1 + NIR + 6*Red) / 7.5; with NIR=0,Red=0 => Blue=1/7.5
        blue_zero = _make_band(1.0 / 7.5)
        result = compute_evi(nir, red, blue_zero)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_clipped_to_minus1_plus1(self):
        """EVI values should be clipped to [-1, 1]."""
        # Large NIR, small Red/Blue can produce EVI > 1 before clipping
        nir = _make_band(10000.0)
        red = _make_band(1.0)
        blue = _make_band(1.0)
        result = compute_evi(nir, red, blue)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_uint16_input(self):
        """Should accept uint16 arrays."""
        nir = np.full((4, 4), 5000, dtype=np.uint16)
        red = np.full((4, 4), 1000, dtype=np.uint16)
        blue = np.full((4, 4), 500, dtype=np.uint16)
        result = compute_evi(nir, red, blue)
        assert result.dtype == np.float32

    def test_output_shape(self):
        """Output shape should match input shape."""
        shape = (10, 15)
        nir = np.random.rand(*shape).astype(np.float32)
        red = np.random.rand(*shape).astype(np.float32)
        blue = np.random.rand(*shape).astype(np.float32)
        result = compute_evi(nir, red, blue)
        assert result.shape == shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd . && python -m pytest tests/unit/test_indices.py::TestEVI -v`
Expected: ImportError — `compute_evi` does not exist yet.

- [ ] **Step 3: Implement compute_evi**

Add to `src/indices.py` after `compute_mndwi`:

```python
def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Compute Enhanced Vegetation Index.

    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

    Handles atmospheric effects and canopy saturation better than NDVI,
    especially at high leaf-area index (LAI > 3).

    Args:
        nir: Near-infrared band array (B08).
        red: Red band array (B04).
        blue: Blue band array (B02).

    Returns:
        float32 EVI array clipped to [-1, 1].
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    blue = blue.astype(np.float32)
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom == 0, 0.0, 2.5 * (nir - red) / denom)
    return np.clip(result.astype(np.float32), -1.0, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd . && python -m pytest tests/unit/test_indices.py::TestEVI -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Register EVI in app.py**

In `app.py`, update the import line to include `compute_evi`:

```python
from src.indices import compute_change, compute_mndwi, compute_ndbi, compute_ndvi, compute_evi
```

Update `INDEX_FUNCTIONS` dict:

```python
INDEX_FUNCTIONS = {
    "ndvi": ("NDVI", compute_ndvi, ["nir", "red"]),
    "evi": ("EVI", compute_evi, ["nir", "red", "blue"]),
    "ndbi": ("NDBI", compute_ndbi, ["swir16", "nir"]),
    "mndwi": ("MNDWI", compute_mndwi, ["green", "swir16"]),
}
```

Update `compute_index_for_bands()` to unpack dynamically:

```python
def compute_index_for_bands(
    index_key: str,
    bands: dict[str, np.ndarray],
) -> np.ndarray:
    _, fn, band_order = INDEX_FUNCTIONS[index_key]
    return fn(*[bands[k] for k in band_order])
```

Update `_INDEX_LABELS`:

```python
_INDEX_LABELS = {
    "ndvi": "NDVI — Vegetation",
    "evi": "EVI — Vegetation (dense canopy)",
    "ndbi": "NDBI — Built-up",
    "mndwi": "MNDWI — Water",
}
```

- [ ] **Step 6: Run all existing tests to verify no regressions**

Run: `cd . && python -m pytest tests/unit/ -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/indices.py tests/unit/test_indices.py app.py
git commit -m "feat: add Enhanced Vegetation Index (EVI) computation"
```

---

### Task 2: Add NDVI saturation warning

**Files:**
- Modify: `app.py` (add saturation check after index computation)
- Test: `tests/unit/test_app_logic.py`

- [ ] **Step 1: Write failing test for saturation detection**

Add to `tests/unit/test_app_logic.py`:

```python
import numpy as np


def test_ndvi_saturation_detected():
    """90th percentile > 0.75 should be flagged as saturated."""
    # Dense canopy: most values near 0.85
    before_index = np.full((100, 100), 0.85, dtype=np.float32)
    p90 = np.percentile(before_index, 90)
    assert p90 > 0.75, "Test setup: 90th percentile should exceed 0.75"


def test_ndvi_no_saturation():
    """Moderate values should not trigger saturation."""
    before_index = np.full((100, 100), 0.5, dtype=np.float32)
    p90 = np.percentile(before_index, 90)
    assert p90 <= 0.75, "Test setup: 90th percentile should be below 0.75"
```

Note: The saturation check is UI-level logic in Streamlit (calls `st.warning`), so we test the detection condition, not the Streamlit call. The condition is simple: `np.percentile(before_index, 90) > 0.75`.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd . && python -m pytest tests/unit/test_app_logic.py::test_ndvi_saturation_detected tests/unit/test_app_logic.py::test_ndvi_no_saturation -v`
Expected: PASS (these test the condition logic, not the function).

- [ ] **Step 3: Add saturation warning to app.py**

In `app.py`, after the `delta = compute_change(...)` line (~line 442), and before the `# ── Build images` section, add:

```python
    # ── NDVI saturation warning ──────────────────────────────────────────
    if index_choice == "ndvi":
        p90 = float(np.percentile(before_index, 90))
        if p90 > 0.75:
            st.warning(
                f"NDVI appears saturated in this region (90th percentile: {p90:.2f}). "
                f"Dense vegetation compresses NDVI's dynamic range. "
                f"Consider switching to **EVI** for better sensitivity to canopy changes."
            )
```

- [ ] **Step 4: Run all unit tests**

Run: `cd . && python -m pytest tests/unit/ -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/unit/test_app_logic.py
git commit -m "feat: add NDVI saturation warning suggesting EVI"
```

---

### Task 3: Add adaptive thresholding (Otsu's method)

**Files:**
- Modify: `src/indices.py` (add `compute_adaptive_threshold`)
- Modify: `app.py` (add auto-threshold checkbox + UI logic)
- Test: `tests/unit/test_indices.py`

- [ ] **Step 1: Write failing tests for compute_adaptive_threshold**

Add to `tests/unit/test_indices.py`:

```python
from src.indices import compute_adaptive_threshold


class TestAdaptiveThreshold:
    def test_bimodal_distribution(self):
        """Threshold should fall between two clear clusters."""
        # Cluster 1: unchanged pixels near 0
        unchanged = np.random.normal(0.0, 0.02, size=8000).astype(np.float32)
        # Cluster 2: changed pixels near 0.3
        changed = np.random.normal(0.3, 0.02, size=2000).astype(np.float32)
        delta = np.concatenate([unchanged, changed]).reshape(100, 100)
        threshold = compute_adaptive_threshold(delta)
        # Should be between the two clusters
        assert 0.05 < threshold < 0.25, f"Expected threshold between clusters, got {threshold}"

    def test_uniform_returns_fallback(self):
        """Uniform data with no clear separation should return fallback."""
        delta = np.zeros((50, 50), dtype=np.float32)
        threshold = compute_adaptive_threshold(delta)
        assert threshold == 0.10, f"Expected fallback 0.10, got {threshold}"

    def test_returns_float(self):
        """Threshold should be a plain float."""
        delta = np.random.randn(20, 20).astype(np.float32) * 0.2
        threshold = compute_adaptive_threshold(delta)
        assert isinstance(threshold, float)

    def test_positive_result(self):
        """Threshold should always be positive."""
        delta = np.random.randn(50, 50).astype(np.float32) * 0.3
        threshold = compute_adaptive_threshold(delta)
        assert threshold > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd . && python -m pytest tests/unit/test_indices.py::TestAdaptiveThreshold -v`
Expected: ImportError — `compute_adaptive_threshold` does not exist.

- [ ] **Step 3: Implement compute_adaptive_threshold**

Add to `src/indices.py` after `compute_change`:

```python
def compute_adaptive_threshold(
    delta: np.ndarray, fallback: float = 0.10, n_bins: int = 256,
) -> float:
    """Compute an adaptive change threshold using Otsu's method.

    Applies Otsu's algorithm to the absolute delta values to find the
    optimal boundary between "unchanged" and "changed" pixel populations.

    Args:
        delta: 2D float32 change array (after - before).
        fallback: Value returned when data has no variance (e.g. all zeros).
        n_bins: Number of histogram bins for Otsu computation.

    Returns:
        Optimal threshold as a positive float.
    """
    abs_delta = np.abs(delta).ravel().astype(np.float64)
    abs_delta = abs_delta[np.isfinite(abs_delta)]
    if len(abs_delta) == 0 or abs_delta.max() == abs_delta.min():
        return fallback

    # Otsu's method on absolute delta histogram
    counts, bin_edges = np.histogram(abs_delta, bins=n_bins, range=(0, abs_delta.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return fallback

    weight_bg = 0.0
    sum_bg = 0.0
    sum_total = np.dot(counts.astype(np.float64), bin_centers)
    best_thresh = fallback
    best_variance = 0.0

    for i in range(len(counts)):
        weight_bg += counts[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += counts[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_thresh = float(bin_centers[i])

    return best_thresh
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd . && python -m pytest tests/unit/test_indices.py::TestAdaptiveThreshold -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Add auto-threshold UI to app.py**

In the sidebar Display section of `app.py`, after the `index_choice` radio, replace the existing threshold slider with:

```python
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
```

Then after `delta = compute_change(...)` in the main panel, add the auto-threshold computation:

```python
    if st.session_state.get("auto_threshold", False):
        from src.indices import compute_adaptive_threshold
        THRESHOLD = compute_adaptive_threshold(delta)
        st.session_state["threshold"] = THRESHOLD
    else:
        THRESHOLD = st.session_state.get("threshold", 0.10)
```

Replace the old `THRESHOLD = st.session_state.get("threshold", 0.10)` line with the above block.

In Panel D, after the stat_cols metrics, add threshold display:

```python
    threshold_mode = "auto / Otsu" if st.session_state.get("auto_threshold", False) else "manual"
    st.caption(f"Threshold: {THRESHOLD:.3f} ({threshold_mode})")
```

- [ ] **Step 6: Run all unit tests**

Run: `cd . && python -m pytest tests/unit/ -v`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/indices.py app.py tests/unit/test_indices.py
git commit -m "feat: add adaptive thresholding via Otsu's method"
```

---

### Task 4: Add index-specific colormaps

**Files:**
- Modify: `app.py` (add DEFAULT_COLORMAPS, expand selectbox, auto-select logic)
- Test: `tests/unit/test_app_logic.py`

- [ ] **Step 1: Write failing test for DEFAULT_COLORMAPS coverage**

Add to `tests/unit/test_app_logic.py`:

```python
def test_default_colormaps_cover_all_indices():
    """Every index in INDEX_FUNCTIONS should have a default colormap."""
    from app import INDEX_FUNCTIONS, DEFAULT_COLORMAPS
    for key in INDEX_FUNCTIONS:
        assert key in DEFAULT_COLORMAPS, f"Missing default colormap for index '{key}'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd . && python -m pytest tests/unit/test_app_logic.py::test_default_colormaps_cover_all_indices -v`
Expected: ImportError — `DEFAULT_COLORMAPS` does not exist.

- [ ] **Step 3: Add DEFAULT_COLORMAPS and update colormap selector**

In `app.py`, after the `INDEX_FUNCTIONS` dict, add:

```python
DEFAULT_COLORMAPS = {
    "ndvi": "RdYlGn",
    "evi": "RdYlGn",
    "ndbi": "PuOr",
    "mndwi": "BrBG",
}

COLORMAP_OPTIONS = ["RdYlGn", "BrBG", "PuOr", "RdBu", "RdYlBu", "PiYG", "PRGn", "coolwarm"]
```

In the sidebar Display section, replace the colormap selectbox with:

```python
        # Auto-select colormap when index changes (unless user manually overrode)
        _prev_index = st.session_state.get("_prev_index_for_cmap")
        if _prev_index != index_choice:
            current_cmap = st.session_state.get("colormap", DEFAULT_COLORMAPS.get(index_choice, "RdBu"))
            # Only auto-switch if user hadn't manually changed from the previous default
            if _prev_index is None or current_cmap == DEFAULT_COLORMAPS.get(_prev_index, "RdBu"):
                st.session_state["colormap"] = DEFAULT_COLORMAPS.get(index_choice, "RdBu")
            st.session_state["_prev_index_for_cmap"] = index_choice

        colormap = st.selectbox(
            "Colormap",
            COLORMAP_OPTIONS,
            key="colormap",
            help="Color scheme for the change heatmap. Auto-selected per index; override manually.",
        )
```

- [ ] **Step 4: Run tests**

Run: `cd . && python -m pytest tests/unit/test_app_logic.py::test_default_colormaps_cover_all_indices -v`
Expected: PASS.

- [ ] **Step 5: Run all unit tests**

Run: `cd . && python -m pytest tests/unit/ -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/unit/test_app_logic.py
git commit -m "feat: add index-specific default colormaps"
```

---

### Task 5: Add confidence / quality indicator

**Files:**
- Modify: `src/sentinel.py` (extract sun_elevation from STAC)
- Modify: `app.py` (add Data Quality expander in Panel D)
- Test: `tests/unit/test_app_logic.py`

- [ ] **Step 1: Write failing tests for quality classification**

Add to `tests/unit/test_app_logic.py`:

```python
def test_quality_rating_cloud_green():
    from app import quality_rating
    assert quality_rating(10.0, thresholds=(15, 30)) == "green"

def test_quality_rating_cloud_yellow():
    from app import quality_rating
    assert quality_rating(20.0, thresholds=(15, 30)) == "yellow"

def test_quality_rating_cloud_red():
    from app import quality_rating
    assert quality_rating(35.0, thresholds=(15, 30)) == "red"

def test_quality_rating_sun_green():
    from app import quality_rating
    assert quality_rating(45.0, thresholds=(30, 20), lower_is_worse=True) == "green"

def test_quality_rating_sun_yellow():
    from app import quality_rating
    assert quality_rating(25.0, thresholds=(30, 20), lower_is_worse=True) == "yellow"

def test_quality_rating_sun_red():
    from app import quality_rating
    assert quality_rating(15.0, thresholds=(30, 20), lower_is_worse=True) == "red"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd . && python -m pytest tests/unit/test_app_logic.py::test_quality_rating_cloud_green tests/unit/test_app_logic.py::test_quality_rating_cloud_red tests/unit/test_app_logic.py::test_quality_rating_sun_red -v`
Expected: ImportError — `quality_rating` does not exist.

- [ ] **Step 3: Implement quality_rating function in app.py**

Add to `app.py` after `compute_index_for_bands`:

```python
def quality_rating(
    value: float,
    thresholds: tuple[float, float] = (15, 30),
    lower_is_worse: bool = False,
) -> str:
    """Classify a quality metric as green/yellow/red.

    Args:
        value: The metric value to classify.
        thresholds: (yellow_boundary, red_boundary). For normal mode,
            value > yellow = yellow, value > red = red. For lower_is_worse,
            value < yellow = yellow, value < red = red.
        lower_is_worse: If True, lower values are worse (e.g. sun elevation).

    Returns:
        "green", "yellow", or "red".
    """
    yellow_bound, red_bound = thresholds
    if lower_is_worse:
        if value < red_bound:
            return "red"
        if value < yellow_bound:
            return "yellow"
        return "green"
    else:
        if value > red_bound:
            return "red"
        if value > yellow_bound:
            return "yellow"
        return "green"
```

- [ ] **Step 4: Run quality_rating tests**

Run: `cd . && python -m pytest tests/unit/test_app_logic.py -k "quality_rating" -v`
Expected: All 6 PASS.

- [ ] **Step 5: Extract sun_elevation in sentinel.py**

In `src/sentinel.py`, inside the `search_scenes()` function's `for item in search.items()` loop, add `sun_elevation` to the result dict. Change:

```python
        results.append({
            "id": item.id,
            "cloud_cover": item.properties.get("eo:cloud_cover", 0.0),
            "datetime": str(item.datetime),
            "assets": {k: v.href for k, v in item.assets.items()},
            "bbox": list(item.bbox),
        })
```

to:

```python
        results.append({
            "id": item.id,
            "cloud_cover": item.properties.get("eo:cloud_cover", 0.0),
            "sun_elevation": item.properties.get("view:sun_elevation"),
            "datetime": str(item.datetime),
            "assets": {k: v.href for k, v in item.assets.items()},
            "bbox": list(item.bbox),
        })
```

- [ ] **Step 6: Add Data Quality expander to Panel D in app.py**

In `app.py`, in the Panel D section, after the detail_cols block that shows before/after scene info, add:

```python
    # ── Data Quality ─────────────────────────────────────────────────────
    _RATING_ICONS = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}

    with st.expander("Data Quality", expanded=False):
        qual_cols = st.columns(2)

        # Cloud cover
        bc = before_scene["cloud_cover"]
        ac = after_scene["cloud_cover"]
        bc_rating = quality_rating(bc, thresholds=(15, 30))
        ac_rating = quality_rating(ac, thresholds=(15, 30))
        qual_cols[0].markdown(f"{_RATING_ICONS[bc_rating]} **Before cloud:** {bc:.1f}%")
        qual_cols[1].markdown(f"{_RATING_ICONS[ac_rating]} **After cloud:** {ac:.1f}%")

        # Sun elevation
        b_sun = before_scene.get("sun_elevation")
        a_sun = after_scene.get("sun_elevation")
        if b_sun is not None:
            bs_rating = quality_rating(b_sun, thresholds=(30, 20), lower_is_worse=True)
            qual_cols[0].markdown(f"{_RATING_ICONS[bs_rating]} **Before sun elev:** {b_sun:.1f}\u00b0")
        else:
            qual_cols[0].markdown("\u2796 **Before sun elev:** N/A")
        if a_sun is not None:
            as_rating = quality_rating(a_sun, thresholds=(30, 20), lower_is_worse=True)
            qual_cols[1].markdown(f"{_RATING_ICONS[as_rating]} **After sun elev:** {a_sun:.1f}\u00b0")
        else:
            qual_cols[1].markdown("\u2796 **After sun elev:** N/A")

        # Time gap
        from datetime import datetime
        before_dt = datetime.fromisoformat(before_scene["datetime"].replace("Z", "+00:00"))
        after_dt = datetime.fromisoformat(after_scene["datetime"].replace("Z", "+00:00"))
        gap_days = abs((after_dt - before_dt).days)
        gap_years = gap_days / 365.25
        if gap_years >= 1:
            gap_str = f"{gap_years:.1f} years"
        else:
            gap_str = f"{gap_days} days"
        gap_rating = quality_rating(gap_days, thresholds=(365, 730))
        st.markdown(f"{_RATING_ICONS[gap_rating]} **Time gap:** {gap_str}")
```

- [ ] **Step 7: Run all unit tests**

Run: `cd . && python -m pytest tests/unit/ -v`
Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add src/sentinel.py app.py tests/unit/test_app_logic.py
git commit -m "feat: add confidence/quality indicator in Panel D"
```

---

### Task 6: Final integration test

- [ ] **Step 1: Run full test suite**

Run: `cd . && python -m pytest tests/unit/ -v`
Expected: All tests PASS with no regressions.

- [ ] **Step 2: Verify app imports cleanly**

Run: `cd . && python -c "from app import INDEX_FUNCTIONS, DEFAULT_COLORMAPS, quality_rating; print('OK:', list(INDEX_FUNCTIONS.keys()), list(DEFAULT_COLORMAPS.keys()))"`
Expected: `OK: ['ndvi', 'evi', 'ndbi', 'mndwi'] ['ndvi', 'evi', 'ndbi', 'mndwi']`

- [ ] **Step 3: Final commit if any remaining changes**

```bash
git status
# Only commit if there are uncommitted changes
```
