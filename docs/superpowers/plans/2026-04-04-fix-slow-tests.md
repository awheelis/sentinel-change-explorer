# Fix Slow Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the ThreadPoolExecutor hang bug in production and make all three slow tests genuinely fast.

**Architecture:** Three surgical changes — fix executor shutdown in `src/overture.py`, let integration tests use cache, and pre-warm cache for e2e. No new files created.

**Tech Stack:** Python 3.12, geopandas, concurrent.futures, pytest, Playwright

---

### Task 1: Fix ThreadPoolExecutor hang in `src/overture.py`

**Files:**
- Modify: `src/overture.py:33-35` (remove retry constants)
- Modify: `src/overture.py:102-128` (replace retry loop + executor pattern)
- Test: `tests/unit/test_overture.py`

- [ ] **Step 1: Update the unit test for single-attempt behavior**

The existing `test_fetch_overture_layer_retries_on_transient_failure` tests 3 retries. Since we're removing retries, update it to verify single-attempt fail-fast behavior:

In `tests/unit/test_overture.py`, replace the function `test_fetch_overture_layer_retries_on_transient_failure` (lines 156-178) with:

```python
def test_fetch_overture_layer_fails_fast_on_transient_error():
    """Should fail after a single attempt on transient network errors (no retries)."""
    mock_core = MagicMock()
    mock_core.geodataframe.side_effect = ConnectionError("timeout")

    with patch("src.overture._import_overture_core", return_value=mock_core):
        result = fetch_overture_layer(
            "building",
            bbox=(-115.2, 36.1, -115.1, 36.2),
            use_cache=False,
        )
    assert mock_core.geodataframe.call_count == 1
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
```

- [ ] **Step 2: Update the timeout unit test**

The existing `test_fetch_overture_layer_timeout_returns_empty` (lines 119-136) patches `time.sleep` for retry delays that no longer exist. Simplify it:

In `tests/unit/test_overture.py`, replace the function `test_fetch_overture_layer_timeout_returns_empty` with:

```python
def test_fetch_overture_layer_timeout_returns_empty():
    """A slow-but-alive connection should be killed by the timeout wrapper."""
    import time as _time
    mock_core = MagicMock()

    def _slow_fetch(*a, **kw):
        _time.sleep(30)
        return gpd.GeoDataFrame()

    mock_core.geodataframe.side_effect = _slow_fetch

    with patch("src.overture._import_overture_core", return_value=mock_core), \
         patch("src.overture._LAYER_TIMEOUT", 1):
        result = fetch_overture_layer("building", bbox=(-115.2, 36.1, -115.1, 36.2), use_cache=False)

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
```

- [ ] **Step 3: Run unit tests to verify they fail**

Run: `pytest tests/unit/test_overture.py -v`
Expected: `test_fetch_overture_layer_fails_fast_on_transient_error` FAILS (old code still retries 3 times, so `call_count == 3` not `1`). `test_fetch_overture_layer_timeout_returns_empty` may pass or fail depending on whether `time.sleep` patch is needed.

- [ ] **Step 4: Implement the executor fix and remove retry loop**

In `src/overture.py`, remove the retry constants (lines 33-35):

Replace:
```python
_MAX_RETRIES = 3
_RETRY_DELAYS = (2, 4, 8)
_LAYER_TIMEOUT = 15  # seconds per layer fetch
```

With:
```python
_LAYER_TIMEOUT = 15  # seconds per layer fetch
```

Then replace the entire fetch-from-Overture block (lines 102-134) with a single-attempt pattern using explicit shutdown:

Replace everything from `last_exc = None` through the `if last_exc is not None:` block (lines 102-134) with:

```python
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        logger.debug(
            "Fetching Overture layer '%s' for bbox %s",
            layer, bbox,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            future = pool.submit(core.geodataframe, layer, bbox=bbox)
            gdf: gpd.GeoDataFrame = future.result(timeout=_LAYER_TIMEOUT)
    except _TIMEOUT_AND_TRANSIENT as exc:
        logger.warning(
            "Overture fetch failed for layer '%s': %s",
            layer, exc,
        )
        gdf = gpd.GeoDataFrame()
    except (ValueError, TypeError, ArithmeticError) as exc:
        logger.warning("Failed to fetch Overture layer '%s': %s", layer, exc)
        pool.shutdown(wait=False, cancel_futures=True)
        return gpd.GeoDataFrame()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
```

Also remove the now-unused imports/constants. In the `_TIMEOUT_AND_TRANSIENT` line (line 36), it references `_TRANSIENT_ERRORS` which is still used, so keep that. Remove `time` from imports since `time.sleep` was only used for retry delays.

Replace:
```python
import time
```

With:
```python
```

(Remove the `import time` line entirely.)

- [ ] **Step 5: Run unit tests to verify they pass**

Run: `pytest tests/unit/test_overture.py -v`
Expected: ALL PASS. The retry test now expects `call_count == 1`, the timeout test no longer patches `time.sleep`.

- [ ] **Step 6: Commit**

```bash
git add src/overture.py tests/unit/test_overture.py
git commit -m "fix: remove retry loop and fix ThreadPoolExecutor hang in overture fetch"
```

---

### Task 2: Make integration tests use cache

**Files:**
- Modify: `tests/integration/test_overture.py:14-24` (remove `use_cache=False`, relax timeouts)

- [ ] **Step 1: Write the updated integration tests**

In `tests/integration/test_overture.py`, replace `test_fetch_buildings_returns_geodataframe` (lines 14-24) with:

```python
def test_fetch_buildings_returns_geodataframe():
    """Should fetch building footprints for a small Las Vegas bbox."""
    bbox = (-115.20, 36.10, -115.15, 36.15)
    with assert_within(20, "building fetch"):
        gdf = fetch_overture_layer("building", bbox=bbox)
    assert isinstance(gdf, gpd.GeoDataFrame), "Expected GeoDataFrame"
    print(f"\nFetched {len(gdf)} buildings for bbox {bbox}")
    assert len(gdf) > 0, "Expected at least 1 building"
```

And replace `test_get_overture_context_returns_all_layers` (lines 27-37) with:

```python
def test_get_overture_context_returns_all_layers():
    """Should return dict with building, segment, place keys."""
    bbox = (-115.20, 36.10, -115.15, 36.15)
    with assert_within(45, "all overture layers"):
        context = get_overture_context(bbox=bbox)
    assert "building" in context
    assert "segment" in context
    assert "place" in context
    for layer, gdf in context.items():
        assert isinstance(gdf, gpd.GeoDataFrame), f"{layer} should be GeoDataFrame"
    print(f"\nBuildings: {len(context['building'])}, Segments: {len(context['segment'])}, Places: {len(context['place'])}")
```

- [ ] **Step 2: Run integration tests to verify they pass**

Run: `pytest tests/integration/test_overture.py -v -s`
Expected: ALL PASS. With warm cache, tests complete in <1s. With cold cache, single-attempt fetch completes in ≤15s per layer.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_overture.py
git commit -m "fix: let integration tests use cache and relax timeouts"
```

---

### Task 3: Add cache warmup to e2e test

**Files:**
- Modify: `tests/e2e_visual_test.py:44` (reduce `MAX_ANALYSIS_WAIT`)
- Modify: `tests/e2e_visual_test.py:285-303` (add warmup to `test_visual_e2e`)
- Modify: `tests/e2e_visual_test.py:306-329` (add warmup to `__main__` block)

- [ ] **Step 1: Reduce MAX_ANALYSIS_WAIT**

In `tests/e2e_visual_test.py`, replace:

```python
MAX_ANALYSIS_WAIT = 180  # seconds
```

With:

```python
MAX_ANALYSIS_WAIT = 60  # seconds (warm cache: <10s, 60s is safety margin)
```

- [ ] **Step 2: Add warmup to test_visual_e2e**

In `tests/e2e_visual_test.py`, replace `test_visual_e2e` (lines 285-303) with:

```python
def test_visual_e2e():
    """Pytest-compatible entry point."""
    from app import warm_preset_caches
    log("Warming preset caches for e2e test...")
    warm_preset_caches()
    log("Cache warmup complete.")

    proc = start_streamlit()
    try:
        errors = run_visual_test()
    finally:
        stop_streamlit(proc)

    if errors:
        log(f"\nFAILED — {len(errors)} issue(s):")
        for i, e in enumerate(errors, 1):
            log(f"  {i}. {e}")
    else:
        log("\nPASSED — all checks OK")

    if SAVE_SCREENSHOTS:
        log(f"\nScreenshots: {SCREENSHOT_DIR}/")

    assert not errors, f"Visual E2E test failed: {errors}"
```

- [ ] **Step 3: Add warmup to __main__ block**

In `tests/e2e_visual_test.py`, replace the `if __name__ == "__main__":` block (lines 306-329) with:

```python
if __name__ == "__main__":
    from app import warm_preset_caches
    log("Warming preset caches...")
    warm_preset_caches()
    log("Cache warmup complete.")

    proc = start_streamlit()
    try:
        errors = run_visual_test()
    finally:
        stop_streamlit(proc)

    log("")
    log("=" * 60)
    if errors:
        log(f"RESULT: {len(errors)} ISSUE(S)")
        for i, e in enumerate(errors, 1):
            log(f"  {i}. {e}")
    else:
        log("RESULT: ALL CHECKS PASSED")
    log("=" * 60)

    if SAVE_SCREENSHOTS:
        log(f"\nScreenshots: {SCREENSHOT_DIR}/")
        for f in sorted(SCREENSHOT_DIR.iterdir()):
            if f.suffix == ".png":
                log(f"  {f.name}")

    sys.exit(0 if not errors else 1)
```

- [ ] **Step 4: Run unit tests to verify nothing is broken**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e_visual_test.py
git commit -m "fix: pre-warm caches in e2e test and reduce analysis timeout"
```
