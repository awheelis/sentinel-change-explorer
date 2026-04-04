# Google Maps Button & Preset News Links — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "See on Google Maps" button that opens the analyzed bounding box in Google Maps, and a "Read News Article" link for each preset.

**Architecture:** A pure function `google_maps_url(bbox)` in `src/visualization.py` constructs the URL. Preset news URLs are stored as a `news_url` field in `config/presets.json`. Both buttons render in a row alongside the existing download button in `app.py` post-analysis area.

**Tech Stack:** Python, Streamlit (`st.link_button`), pytest

---

### Task 1: Write failing tests for `google_maps_url()`

**Files:**
- Modify: `tests/unit/test_visualization.py`

- [ ] **Step 1: Write the failing tests**

Add at the bottom of `tests/unit/test_visualization.py`:

```python
from src.visualization import google_maps_url


class TestGoogleMapsUrl:
    def test_url_starts_with_google_maps(self):
        url = google_maps_url((-115.20, 36.10, -115.15, 36.15))
        assert url.startswith("https://www.google.com/maps/@")

    def test_center_is_correct(self):
        url = google_maps_url((-115.20, 36.10, -115.15, 36.15))
        # Center: lat=36.125, lng=-115.175
        assert "36.125" in url
        assert "-115.175" in url

    def test_zoom_small_bbox(self):
        """A small bbox (~0.05 deg) should produce a high zoom (roughly 12-15)."""
        url = google_maps_url((-115.20, 36.10, -115.15, 36.15))
        # Extract zoom from URL: ...@lat,lng,{zoom}z
        zoom = float(url.split(",")[-1].rstrip("z"))
        assert 12 <= zoom <= 16

    def test_zoom_large_bbox(self):
        """A large bbox (~10 deg) should produce a low zoom (roughly 4-7)."""
        url = google_maps_url((-10.0, -5.0, 0.0, 5.0))
        zoom = float(url.split(",")[-1].rstrip("z"))
        assert 4 <= zoom <= 8

    def test_zoom_clamped_minimum(self):
        """Even for a huge bbox, zoom should not go below 2."""
        url = google_maps_url((-180.0, -80.0, 180.0, 80.0))
        zoom = float(url.split(",")[-1].rstrip("z"))
        assert zoom >= 2

    def test_zoom_clamped_maximum(self):
        """Even for a tiny bbox, zoom should not exceed 18."""
        url = google_maps_url((13.0, 52.0, 13.001, 52.001))
        zoom = float(url.split(",")[-1].rstrip("z"))
        assert zoom <= 18
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_visualization.py::TestGoogleMapsUrl -v`
Expected: FAIL with `ImportError` (function doesn't exist yet)

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/unit/test_visualization.py
git commit -m "test: add failing tests for google_maps_url()"
```

---

### Task 2: Implement `google_maps_url()`

**Files:**
- Modify: `src/visualization.py`

- [ ] **Step 1: Add the function to `src/visualization.py`**

Add after the imports section (around line 23, before `true_color_image`):

```python
import math


def google_maps_url(bbox: tuple[float, float, float, float]) -> str:
    """Build a Google Maps URL centered on the given bounding box.

    Parameters
    ----------
    bbox : (west, south, east, north) in WGS-84 degrees.

    Returns
    -------
    URL like ``https://www.google.com/maps/@{lat},{lng},{zoom}z``.
    """
    west, south, east, north = bbox
    center_lat = (south + north) / 2.0
    center_lng = (west + east) / 2.0
    bbox_width = east - west
    # zoom ≈ log2(360 / width), clamped to [2, 18]
    if bbox_width > 0:
        zoom = math.log2(360.0 / bbox_width)
    else:
        zoom = 18
    zoom = int(max(2, min(18, zoom)))
    return f"https://www.google.com/maps/@{center_lat},{center_lng},{zoom}z"
```

Note: `math` is already used transitively but add the import if not present at the top of the file.

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/unit/test_visualization.py::TestGoogleMapsUrl -v`
Expected: All 6 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/visualization.py
git commit -m "feat: add google_maps_url() to visualization module"
```

---

### Task 3: Write failing tests for preset news URLs

**Files:**
- Modify: `tests/unit/test_app_logic.py`

- [ ] **Step 1: Write the failing tests**

Add at the bottom of `tests/unit/test_app_logic.py`:

```python
import json
from pathlib import Path

PRESETS_FILE = Path(__file__).resolve().parents[2] / "config" / "presets.json"


class TestPresetNewsUrls:
    def _load_presets(self):
        with open(PRESETS_FILE) as f:
            return json.load(f)

    def test_every_preset_has_news_url(self):
        presets = self._load_presets()
        for preset in presets:
            assert "news_url" in preset, f"Preset '{preset['name']}' missing 'news_url'"

    def test_news_urls_are_https(self):
        presets = self._load_presets()
        for preset in presets:
            url = preset["news_url"]
            assert isinstance(url, str), f"Preset '{preset['name']}' news_url is not a string"
            assert url.startswith("https://"), (
                f"Preset '{preset['name']}' news_url must start with https://, got: {url}"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_app_logic.py::TestPresetNewsUrls -v`
Expected: FAIL with `KeyError: 'news_url'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/unit/test_app_logic.py
git commit -m "test: add failing tests for preset news_url field"
```

---

### Task 4: Add `news_url` to each preset in `presets.json`

**Files:**
- Modify: `config/presets.json`

- [ ] **Step 1: Add `news_url` to each preset**

Add a `"news_url"` field to each preset object:

| Preset | news_url |
|--------|----------|
| Lahaina Wildfire, Maui | `https://apnews.com/article/hawaii-wildfires-maui-lahaina-evacuation-fire-d333e5e4e3be4be1b8b362e102e0e38e` |
| Pakistan Mega-Flood, Sindh | `https://www.bbc.com/news/science-environment-62758811` |
| Gigafactory Berlin | `https://www.reuters.com/business/autos-transportation/teslas-german-gigafactory-wins-high-court-approval-2024-02-22/` |
| Black Summer Bushfires, Australia | `https://www.bbc.com/news/world-australia-50951043` |
| Egypt's New Capital | `https://www.bbc.com/news/world-middle-east-65086871` |

Place the `"news_url"` field after `"description"` in each preset object for consistency.

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/unit/test_app_logic.py::TestPresetNewsUrls -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Commit**

```bash
git add config/presets.json
git commit -m "feat: add news_url to all presets"
```

---

### Task 5: Wire up buttons in `app.py`

**Files:**
- Modify: `app.py` (~lines 620-625)

- [ ] **Step 1: Add import for `google_maps_url`**

In `app.py`, find the existing import from `src.visualization` and add `google_maps_url`:

```python
from src.visualization import (
    ...,
    google_maps_url,
)
```

If it's a single-line import, add `google_maps_url` to it.

- [ ] **Step 2: Replace the standalone download button with a button row**

Find the current download button block (around line 620):

```python
    st.download_button(
        label="Download Change Raster (.tif)",
        data=tiff_bytes,
        file_name=f"change_{index_choice}_{before_scene['datetime'][:10]}_{after_scene['datetime'][:10]}.tif",
        mime="image/tiff",
    )
```

Replace it with:

```python
    # ── Action buttons row ───────────────────────────────────────────────
    maps_url = google_maps_url(bbox)

    # Determine if we have a news link (preset selected with news_url)
    _active_preset = next(
        (p for p in load_presets()
         if p["name"] == st.session_state.get("_last_preset")),
        None,
    )
    _news_url = _active_preset["news_url"] if _active_preset and "news_url" in _active_preset else None

    if _news_url:
        btn_cols = st.columns(3)
    else:
        btn_cols = st.columns(2)

    with btn_cols[0]:
        st.download_button(
            label="Download Change Raster (.tif)",
            data=tiff_bytes,
            file_name=f"change_{index_choice}_{before_scene['datetime'][:10]}_{after_scene['datetime'][:10]}.tif",
            mime="image/tiff",
        )
    with btn_cols[1]:
        st.link_button("See on Google Maps", maps_url)
    if _news_url:
        with btn_cols[2]:
            st.link_button("Read News Article", _news_url)
```

- [ ] **Step 3: Run full unit test suite**

Run: `pytest tests/unit/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Google Maps and news article buttons to results area"
```

---

### Task 6: Run full test suite and verify

**Files:** None (verification only)

- [ ] **Step 1: Run all unit and perf tests**

Run: `pytest tests/unit/ tests/perf/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run a quick smoke test of the app (optional)**

Run: `python -c "from src.visualization import google_maps_url; print(google_maps_url((-156.695, 20.860, -156.660, 20.895)))"`
Expected: `https://www.google.com/maps/@20.8775,-156.6775,13z` (approximately)

- [ ] **Step 3: Final commit if any cleanup needed**

Only if fixups were required in previous steps.
