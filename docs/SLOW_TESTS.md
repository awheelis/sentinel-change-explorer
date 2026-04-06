# Slow Tests That Need Attention

Date: 2026-04-04

Tests that are unreliable or excessively slow due to external network dependencies.

---

## 1. `tests/integration/test_overture.py::test_fetch_buildings_returns_geodataframe`

**Time:** 60s+ (hits pytest timeout)

**Root cause:** The Overture Maps API (`overturemaps.core.geodataframe`) is extremely slow for real-time fetches. The building layer for even a small Las Vegas bbox takes >60s to scan from S3. The internal `_LAYER_TIMEOUT` of 15s fires, but the retry logic (3 attempts with 2+4+8s backoff) extends wall time to ~59s. Worse, the `ThreadPoolExecutor.__exit__` blocks waiting for the background thread to finish even after timeout, so the test hangs until the S3 scan completes or the pytest timeout kills it.

**Fix options:**
- Increase `_LAYER_TIMEOUT` and reduce `_MAX_RETRIES` to fail fast instead of retrying a slow API
- Use `cancel_futures=True` in the executor shutdown (Python 3.9+) so timed-out threads don't block
- Mock the Overture API in this test — the unit tests already cover the retry/cache logic; this integration test only needs to verify the real API returns data, which is inherently flaky
- Use a smaller bbox or a region with fewer buildings

---

## 2. `tests/e2e_visual_test.py::test_visual_e2e`

**Time:** 200s+ (hits 180s analysis timeout)

**Root cause:** The e2e test launches the full Streamlit app with the Lahaina preset, which auto-runs analysis on load. This requires downloading 5 Sentinel-2 bands from S3 via COG windowed reads (~30-60s per band uncached) plus Overture context. When the band cache is warm, analysis takes <10s. But STAC search results drift over time (different scene IDs), causing cache misses that force full S3 downloads.

The 180s `MAX_ANALYSIS_WAIT` is often not enough when bands are uncached and the network is slow.

**Fix options:**
- Pre-warm the band cache before running the e2e test (run `warm_preset_caches()` as a fixture)
- Increase `MAX_ANALYSIS_WAIT` to 300s as a quick mitigation
- Use a smaller custom bbox instead of the Lahaina preset to reduce download size
- Add a `--fast` mode that skips the e2e test unless explicitly requested
- Disable Overture fetch during e2e (set `show_overture=False` via environment variable) to eliminate one source of delay

---

## 3. `tests/integration/test_overture.py::test_get_overture_context_returns_all_layers`

**Time:** Depends on cache state; 30s+ uncached

**Status:** Currently passing (because it only checks dict structure, not row counts), but slow when cache is cold. Same underlying Overture API slowness as #1.

**Fix options:** Same as #1 — the two tests share the same bbox and could share a single warm-up fixture.

---

## Recommended Priority

1. **Fix the Overture executor hang** — the `ThreadPoolExecutor` not releasing on timeout is a production bug, not just a test issue. Users of the app also experience this hang.
2. **Add a pre-warm fixture for e2e** — most reliable fix, keeps the test meaningful.
3. **Consider a pytest marker** (e.g. `@pytest.mark.slow`) for all three tests so they can be excluded from fast CI runs with `pytest -m "not slow"`.
