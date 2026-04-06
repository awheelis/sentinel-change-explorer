# Startup Cache Warm-up + Pipeline Parallelization

## Problem

The Sentinel Change Explorer takes 7-8s per click with warm cache, and 30-60s on cold start (S3 COG reads). For a live demo, the user needs under 5s per click with no manual pre-warming steps.

## Solution

Two changes to `app.py`:

### 1. Startup Cache Warm-up

A `warm_preset_caches()` function decorated with `@st.cache_resource` runs once on first app load. It:

1. Loads all presets from `presets.json`
2. For each preset, submits concurrent futures for:
   - `search_scenes` (before range) + `load_bands`
   - `search_scenes` (after range) + `load_bands`
   - `get_overture_context`
3. All 5 presets warm in parallel using `ThreadPoolExecutor` — each preset's before/after/overture fetches also run concurrently
4. A `st.status` spinner ("Preparing satellite data...") blocks the UI until all futures complete
5. On subsequent Streamlit reruns, `st.cache_resource` returns immediately

The warm-up populates the existing disk caches (`cache/bands/*.npz` and `cache/overture/*.parquet`) so that subsequent `load_bands` and `get_overture_context` calls hit cache.

**Threading model:** One `ThreadPoolExecutor` with `max_workers=15` (5 presets × 3 tasks). Each worker calls `search_scenes` → `load_bands` or `get_overture_context`. These are I/O-bound (S3 HTTP reads), so threading is appropriate.

**Error handling:** Individual preset warm-up failures are logged as warnings but do not block the app — the user will just experience a slower first load for that preset.

### 2. Pipeline Parallelization (per-click)

In the existing data-fetch block of `main()` (lines ~226-287), the before/after/overture fetches currently run sequentially. Change to:

- Submit `search_scenes` + `load_bands` for before and after as concurrent futures
- Submit `get_overture_context` as a concurrent future
- Wait for all to complete

This cuts the warm-cache per-click time from ~7-8s to ~3-4s.

## Files

- **Modify:** `app.py` — add `warm_preset_caches()`, parallelize data-fetch block
- **Create:** `tests/test_warmup.py` — test the warm-up function logic

## Constraints

- `st.cache_resource` ensures warm-up runs exactly once per server lifetime
- The warm-up function itself cannot call `st.` UI functions (it runs outside the main thread via cache_resource), but the calling code in `main()` wraps it in `st.status`
- Custom bboxes are not pre-warmed — they use the normal (slower) path
- `max_cloud_cover` during warm-up uses 20% (the slider default)

## Success Criteria

- All 5 presets are cached on first app load
- Clicking "Analyze Change" on any preset completes in under 5s (warm cache)
- The app shows a blocking spinner during warm-up
- Individual preset failures don't crash the app
