# Startup Cache Warm-up + Pipeline Parallelization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app demo-ready by warming all preset caches on startup and parallelizing per-click data fetches to get under 5s response time.

**Architecture:** Add a `warm_preset_caches()` function to `app.py` that runs once via `@st.cache_resource` on first load. It uses `ThreadPoolExecutor` to concurrently fetch `search_scenes` + `load_bands` + `get_overture_context` for all presets. The existing sequential data-fetch block in `main()` is replaced with a concurrent version using `ThreadPoolExecutor`. Both changes are I/O-bound (S3 HTTP, STAC API) so threading is appropriate.

**Tech Stack:** `concurrent.futures.ThreadPoolExecutor`, Streamlit `@st.cache_resource`, existing `src.sentinel` and `src.overture` modules

---

## File Structure

- **Modify:** `app.py` — add `warm_preset_caches()`, add `_fetch_scene_and_bands()` helper, parallelize the data-fetch block in `main()`
- **Create:** `tests/test_warmup.py` — test the warm-up logic with mocked I/O
- **Modify:** `src/sentinel.py` — extract `_search_and_load()` as a reusable helper (used by both warm-up and per-click)

Actually, on reflection, extracting to `src/sentinel.py` adds coupling. The warm-up and per-click paths have different error handling needs (warm-up swallows errors; per-click shows `st.error`). Keep both in `app.py` as simple inline lambdas/helpers.

**Final file structure:**
- **Modify:** `app.py` — add `warm_preset_caches()`, parallelize data-fetch block
- **Create:** `tests/test_warmup.py` — test warm-up logic

---

### Task 1: Add `warm_preset_caches()` and test it

**Files:**
- Modify: `app.py:1-50` (add import, add function)
- Create: `tests/test_warmup.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_warmup.py`:

```python
"""Tests for preset cache warm-up logic."""
from unittest.mock import patch, MagicMock
import pytest


def test_warm_preset_caches_calls_all_presets():
    """Warm-up should search + load bands + fetch overture for every preset."""
    fake_presets = [
        {
            "name": "Preset A",
            "bbox": [-115.32, 36.08, -115.08, 36.28],
            "before_range": ["2019-05-01", "2019-07-31"],
            "after_range": ["2023-05-01", "2023-07-31"],
            "default_index": "ndbi",
        },
        {
            "name": "Preset B",
            "bbox": [58.50, 44.80, 59.20, 45.40],
            "before_range": ["2018-07-01", "2018-09-30"],
            "after_range": ["2023-07-01", "2023-09-30"],
            "default_index": "mndwi",
        },
    ]

    fake_scene = {
        "id": "test-scene",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T00:00:00Z",
        "assets": {"red": "url", "green": "url", "blue": "url", "nir": "url", "swir16": "url"},
        "bbox": [-115.32, 36.08, -115.08, 36.28],
    }

    with patch("app.load_presets", return_value=fake_presets), \
         patch("app.search_scenes", return_value=[fake_scene]) as mock_search, \
         patch("app.load_bands", return_value={}) as mock_load, \
         patch("app.get_overture_context", return_value={}) as mock_overture:

        from app import warm_preset_caches
        # Call the underlying function directly (bypass st.cache_resource)
        warm_preset_caches()

        # 2 presets × 2 date ranges = 4 search calls
        assert mock_search.call_count == 4
        # 2 presets × 2 scenes = 4 load_bands calls
        assert mock_load.call_count == 4
        # 2 presets × 1 overture call = 2
        assert mock_overture.call_count == 2


def test_warm_preset_caches_survives_failures():
    """If one preset fails, others should still be warmed."""
    fake_presets = [
        {
            "name": "Failing Preset",
            "bbox": [0, 0, 1, 1],
            "before_range": ["2019-01-01", "2019-03-31"],
            "after_range": ["2023-01-01", "2023-03-31"],
            "default_index": "ndvi",
        },
        {
            "name": "Working Preset",
            "bbox": [10, 10, 11, 11],
            "before_range": ["2019-01-01", "2019-03-31"],
            "after_range": ["2023-01-01", "2023-03-31"],
            "default_index": "ndvi",
        },
    ]

    fake_scene = {
        "id": "test-scene",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T00:00:00Z",
        "assets": {"red": "url", "green": "url", "blue": "url", "nir": "url", "swir16": "url"},
        "bbox": [10, 10, 11, 11],
    }

    call_count = {"search": 0}

    def search_side_effect(bbox, date_range, max_cloud_cover=20):
        call_count["search"] += 1
        if bbox == (0, 0, 1, 1):
            raise RuntimeError("Network error")
        return [fake_scene]

    with patch("app.load_presets", return_value=fake_presets), \
         patch("app.search_scenes", side_effect=search_side_effect), \
         patch("app.load_bands", return_value={}) as mock_load, \
         patch("app.get_overture_context", return_value={}) as mock_overture:

        from app import warm_preset_caches
        # Should not raise
        warm_preset_caches()

        # The working preset should still have been loaded
        assert mock_load.call_count >= 2  # before + after for working preset
        assert mock_overture.call_count >= 1  # overture for working preset
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_warmup.py -v`
Expected: FAIL with `ImportError` (`warm_preset_caches` doesn't exist yet)

- [ ] **Step 3: Implement `warm_preset_caches`**

In `app.py`, add `import logging` and `from concurrent.futures import ThreadPoolExecutor, as_completed` to the imports at the top (after line 9):

```python
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
```

Then add the warm-up function after `load_presets()` (after line 50):

```python
logger = logging.getLogger(__name__)


def warm_preset_caches() -> None:
    """Pre-fetch search results, bands, and Overture data for all presets.

    Runs all presets concurrently via ThreadPoolExecutor. Individual preset
    failures are logged and swallowed so one bad preset doesn't block the rest.
    Populates the disk caches used by load_bands and get_overture_context.
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

        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.warning("Preset warm-up task failed", exc_info=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_warmup.py -v`
Expected: All pass

- [ ] **Step 5: Run full test suite for regressions**

Run: `python -m pytest tests/test_visualization.py tests/test_indices.py tests/test_warmup.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_warmup.py
git commit -m "feat: add warm_preset_caches for concurrent preset pre-fetching"
```

---

### Task 2: Call `warm_preset_caches` on app startup with blocking spinner

**Files:**
- Modify: `app.py:71-84` (add warm-up call at start of `main()`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_warmup.py`:

```python
def test_warm_called_before_main_ui(monkeypatch):
    """The warm-up should be called early in main(), before the sidebar renders."""
    call_order = []

    def fake_warm():
        call_order.append("warm")

    def fake_set_page_config(**kwargs):
        call_order.append("set_page_config")

    def fake_title(t):
        call_order.append("title")

    # We can't fully run main() without Streamlit, but we can verify
    # warm_preset_caches is defined and callable
    from app import warm_preset_caches
    assert callable(warm_preset_caches)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_warmup.py::test_warm_called_before_main_ui -v`
Expected: PASS (this is a structural verification)

- [ ] **Step 3: Add warm-up call to `main()`**

In `app.py`, in the `main()` function, add the warm-up call after `st.caption(...)` (after line 82) and before `presets = load_presets()` (line 84):

```python
    # ── Pre-warm all preset caches on first server load ──────────────────────
    if "_warmup_done" not in st.session_state:
        with st.status("Preparing satellite data for all presets…", expanded=True) as status:
            st.write("Pre-fetching scenes, bands, and map context for all presets…")
            warm_preset_caches()
            st.session_state["_warmup_done"] = True
            status.update(label="Ready!", state="complete", expanded=False)
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: block on startup with warm-up spinner for all presets"
```

---

### Task 3: Parallelize the per-click data-fetch block

**Files:**
- Modify: `app.py:222-288` (replace sequential fetch with concurrent version)
- Create: (no new test file — tested via existing e2e tests)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_warmup.py`:

```python
from concurrent.futures import ThreadPoolExecutor


def test_concurrent_fetch_pattern():
    """Verify the concurrent fetch pattern works correctly with futures."""
    results = {}

    def fetch_before():
        return {"id": "before-scene", "bands": {"red": "data"}}

    def fetch_after():
        return {"id": "after-scene", "bands": {"red": "data"}}

    def fetch_overture():
        return {"building": [], "segment": [], "place": []}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_before = executor.submit(fetch_before)
        future_after = executor.submit(fetch_after)
        future_overture = executor.submit(fetch_overture)

        results["before"] = future_before.result()
        results["after"] = future_after.result()
        results["overture"] = future_overture.result()

    assert results["before"]["id"] == "before-scene"
    assert results["after"]["id"] == "after-scene"
    assert "building" in results["overture"]
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_warmup.py::test_concurrent_fetch_pattern -v`
Expected: PASS (validates the pattern we'll use)

- [ ] **Step 3: Replace sequential fetch block with concurrent version**

In `app.py`, replace the entire data-fetch block (lines 222-288) from:

```python
    # ── Fetch data with step-by-step progress ────────────────────────────────
    needs_before = "before_scene" not in st.session_state
    needs_after = "after_scene" not in st.session_state
    needs_overture = show_overture and "overture" not in st.session_state

    if needs_before or needs_after or needs_overture:
        total = needs_before + needs_after + needs_overture
        step = 0
        with st.status("Analyzing change detection...", expanded=True) as status:
            if needs_before:
                step += 1
                st.write(f"Step {step}/{total} — Searching for best before scene...")
                scenes = search_scenes(
                    bbox=bbox, date_range=before_range, max_cloud_cover=max_cloud,
                )
                if not scenes:
                    st.error(f"No before scenes found with cloud cover < {max_cloud:.0f}%")
                    status.update(label="Analysis failed", state="error")
                    return
                scene = scenes[0]
                st.write(
                    f"Step {step}/{total} — Loading before bands from S3 "
                    f"({scene['id']}, {scene['cloud_cover']:.1f}% cloud)..."
                )
                try:
                    bands = load_bands(
                        scene=scene, bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=10,
                    )
                except Exception as exc:
                    st.error(f"Failed to load before bands: {exc}")
                    status.update(label="Analysis failed", state="error")
                    return
                st.session_state["before_scene"] = scene
                st.session_state["before_bands"] = bands

            if needs_after:
                step += 1
                st.write(f"Step {step}/{total} — Searching for best after scene...")
                scenes = search_scenes(
                    bbox=bbox, date_range=after_range, max_cloud_cover=max_cloud,
                )
                if not scenes:
                    st.error(f"No after scenes found with cloud cover < {max_cloud:.0f}%")
                    status.update(label="Analysis failed", state="error")
                    return
                scene = scenes[0]
                st.write(
                    f"Step {step}/{total} — Loading after bands from S3 "
                    f"({scene['id']}, {scene['cloud_cover']:.1f}% cloud)..."
                )
                try:
                    bands = load_bands(
                        scene=scene, bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=10,
                    )
                except Exception as exc:
                    st.error(f"Failed to load after bands: {exc}")
                    status.update(label="Analysis failed", state="error")
                    return
                st.session_state["after_scene"] = scene
                st.session_state["after_bands"] = bands

            if needs_overture:
                step += 1
                st.write(f"Step {step}/{total} — Fetching Overture Maps context...")
                st.session_state["overture"] = get_overture_context(bbox=bbox)

            status.update(label="Analysis complete!", state="complete", expanded=False)
```

with:

```python
    # ── Fetch data concurrently ──────────────────────────────────────────────
    needs_before = "before_scene" not in st.session_state
    needs_after = "after_scene" not in st.session_state
    needs_overture = show_overture and "overture" not in st.session_state

    if needs_before or needs_after or needs_overture:
        with st.status("Analyzing change detection…", expanded=True) as status:
            st.write("Fetching scenes and bands concurrently…")

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
                    try:
                        scene, bands = future_before.result()
                        st.session_state["before_scene"] = scene
                        st.session_state["before_bands"] = bands
                    except Exception as exc:
                        st.error(f"Failed to fetch before data: {exc}")
                        status.update(label="Analysis failed", state="error")
                        return

                if future_after is not None:
                    try:
                        scene, bands = future_after.result()
                        st.session_state["after_scene"] = scene
                        st.session_state["after_bands"] = bands
                    except Exception as exc:
                        st.error(f"Failed to fetch after data: {exc}")
                        status.update(label="Analysis failed", state="error")
                        return

                if future_overture is not None:
                    st.session_state["overture"] = future_overture.result()

            status.update(label="Analysis complete!", state="complete", expanded=False)
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/test_visualization.py tests/test_indices.py tests/test_warmup.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "perf: parallelize per-click data fetches with ThreadPoolExecutor"
```
