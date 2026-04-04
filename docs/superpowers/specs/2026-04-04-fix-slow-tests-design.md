# Fix Slow Tests — Design Spec

Date: 2026-04-04

## Problem

Three tests are unreliable or excessively slow due to external network dependencies:

1. `tests/integration/test_overture.py::test_fetch_buildings_returns_geodataframe` — 60s+ (hits pytest timeout)
2. `tests/e2e_visual_test.py::test_visual_e2e` — 200s+ (hits 180s analysis timeout)
3. `tests/integration/test_overture.py::test_get_overture_context_returns_all_layers` — 30s+ uncached

The root cause of #1 and #3 is a production bug: `ThreadPoolExecutor.__exit__` blocks waiting for the S3 scan thread after timeout, and 3 retries with exponential backoff turn a 15s timeout into a 59s hang. The root cause of #2 is cold band/Overture cache causing 5 Sentinel-2 band downloads during the test.

## Approach

Surgical fix (Approach A): fix the executor hang in production, let integration tests use cache, and pre-warm cache for e2e.

## Changes

### 1. Production Fix — `src/overture.py`

**ThreadPoolExecutor hang fix:**
- Replace `with ThreadPoolExecutor(max_workers=1) as pool:` context manager with explicit `pool.shutdown(wait=False, cancel_futures=True)` in a `finally` block. This prevents the executor from blocking on timed-out S3 scan threads. Requires Python 3.9+ (we're on 3.12).

**Remove retry loop:**
- Change `_MAX_RETRIES` from 3 → 1 and remove `_RETRY_DELAYS`. The Overture S3 scan is inherently slow — retrying a 15s timeout 3 times with 14s of backoff just compounds the hang. Fail fast, return empty GeoDataFrame, let disk cache handle future calls.
- Keep `_LAYER_TIMEOUT = 15` — reasonable for a single attempt.

**Result:** The fetch either succeeds within 15s or fails immediately. No more 59s hangs for users or tests.

### 2. Integration Test Fix — `tests/integration/test_overture.py`

- Remove `use_cache=False` from `test_fetch_buildings_returns_geodataframe`. Unit tests already cover retry/cache logic; the integration test just needs to prove "we can get building data for this bbox," and a cache hit satisfies that.
- Relax `assert_within` timeouts slightly (15→20s for single layer, 30→45s for all layers) as safety margin for cold-cache runs. These rarely fire since cache is normally warm.

**Result:** Tests are instant on warm cache. First cold-cache run takes ≤15s per layer (not 59s) thanks to the executor fix.

### 3. E2E Test Fix — `tests/e2e_visual_test.py`

- Call `warm_preset_caches()` (from `app.py`) at the start of `test_visual_e2e()` before launching Streamlit. This pre-fetches Sentinel-2 bands and Overture data for the Lahaina preset.
- Keep `SKIP_WARMUP=1` for Streamlit — no need to re-warm what was just warmed.
- Reduce `MAX_ANALYSIS_WAIT` from 180→60s. With warm cache, analysis takes <10s; 60s is a generous safety margin.
- Apply the same warmup in the `if __name__ == "__main__"` path.

**Result:** E2E test completes in ~30-40s total (Streamlit startup + browser verification) instead of 200s+.

## Files Modified

| File | Change |
|------|--------|
| `src/overture.py` | Fix executor shutdown, remove retry loop |
| `tests/integration/test_overture.py` | Remove `use_cache=False`, relax timeouts |
| `tests/e2e_visual_test.py` | Add cache warmup, reduce `MAX_ANALYSIS_WAIT` |

## What This Does NOT Change

- No new pytest markers or skip logic — tests stay in the default test suite
- No changes to cache format or cache directory structure
- No changes to `get_overture_context()` (it calls `fetch_overture_layer` which gets the fix)
- No changes to unit tests
- No changes to `conftest.py`
