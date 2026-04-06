# Visual End-to-End Test Procedure

**For: Sentinel-2 Change Detection Explorer**
**Audience: Claude (or any AI agent) running tests in a sandboxed environment**

---

## Quick Reference — Full Run Command

```bash
cd /path/to/sentinel-change-explorer

# 1. Install deps (first time only)
uv sync
uv add --dev playwright
uv run playwright install chromium

# 2. Unit tests (always run first — ~7s, no network)
uv run pytest tests/unit/ -v --tb=short

# 3. E2E visual test (~35s with warm cache, ~90s cold)
uv run python tests/e2e_visual_test.py
```

---

## Phase 1: Unit Tests (~7 seconds)

Run these first. If they fail, stop — there's a code-level regression.

```bash
uv run pytest tests/unit/ -v --tb=short
```

**Expected:** 61 tests pass in <10s. No network required.

**What's covered:** index math (NDVI/NDBI/MNDWI), visualization (true color, heatmaps, folium maps), Overture caching logic, sentinel search/load mocking, GeoTIFF export, bbox validation, memory guard, warm-up logic.

**If tests fail:** Fix the failing module before proceeding. The unit tests are self-contained with mocks; failures indicate real bugs, not network issues.

---

## Phase 2: E2E Visual Test (~35–90 seconds)

### What it does
1. Starts Streamlit on port 8599 with `SKIP_WARMUP=1`
2. Opens headless Chromium via Playwright
3. Verifies page title, all sidebar controls, clicks "Analyze Change"
4. Waits for satellite data to load (network-dependent, 10–120s)
5. Checks Panel A (true color images), Panel D (stats), Panel B+C (heatmap/map)
6. Saves 5 screenshots to `/tmp/e2e_screenshots/`

### How to run

```bash
uv run python tests/e2e_visual_test.py
```

### Expected output
```
RESULT: ALL CHECKS PASSED
```

Plus 5 screenshots:
- `01_loaded.png` — app loaded, sidebar visible, info banner shown
- `02_panel_a.png` — two satellite images side by side (before/after)
- `03_panel_d.png` — summary statistics (km², gain%, loss%, unchanged%)
- `04_panel_bc.png` — heatmap/map area (may need manual scroll verification — see Known Issues)
- `05_full_page.png` — final viewport state

### Timing expectations

| Step | Cold cache | Warm cache |
|------|-----------|------------|
| Streamlit startup | ~5s | ~3s |
| Page load + render | ~5s | ~5s |
| "Analyze Change" data fetch | 60–120s | 5–15s |
| Panel checks + screenshots | ~10s | ~10s |
| **Total** | **~90s** | **~35s** |

**FLAG IF:** Total exceeds 3 minutes. This means either the STAC API is slow, S3 band reads are timing out, or the Overture fetch is hanging.

---

## Phase 3: Manual Screenshot Verification

After the E2E test passes, **always visually inspect the screenshots**. The DOM assertions catch structural issues, but only your eyes catch rendering bugs.

### Screenshot checklist

**01_loaded.png:**
- [ ] Title "Sentinel-2 Change Detection Explorer" visible
- [ ] Sidebar shows: Preset location dropdown, Bounding Box inputs, date ranges, Analyze Change button
- [ ] Default preset is "Amazon Deforestation (Rondônia)" (index 3)
- [ ] Blue info banner says "Select a preset location..."

**02_panel_a.png:**
- [ ] "Panel A — True Color Comparison" header visible
- [ ] Two satellite images side by side
- [ ] Left image labeled "Before — YYYY-MM-DD"
- [ ] Right image labeled "After — YYYY-MM-DD"
- [ ] Both images show recognizable satellite imagery (green forest for Amazon preset)
- [ ] "Analysis complete!" status message visible

**03_panel_d.png (scroll down):**
- [ ] "Panel D — Summary Statistics" header visible
- [ ] Four metric cards: Area analyzed (km²), gain %, loss %, Unchanged %
- [ ] Before/After scene metadata with scene ID, date, cloud cover %
- [ ] Histogram visible below stats
- [ ] Overture context line: "X buildings, Y road segments, Z places"

**04_panel_bc.png (scroll to bottom):**
- [ ] "Panel B+C — NDVI Change Heatmap" header visible
- [ ] Folium map iframe rendered (interactive map with tile layer)
- [ ] Layer control visible in map corner
- [ ] Heatmap overlay shows red (loss) and blue (gain) regions

---

## Known Issues & Workarounds

### 1. Screenshots don't capture below-the-fold content
**Problem:** Streamlit uses an internal scrolling container (not document-level scroll), so neither `page.screenshot(full_page=True)` nor `window.scrollBy()` reliably capture Panels D and B+C in screenshots. The viewport is 1400×900 and only shows Panel A and the top of Panel D.

**Workaround:** Increase the Playwright viewport height to capture more content in one shot. In `run_visual_test()`, change the viewport:
```python
page = browser.new_page(viewport={"width": 1400, "height": 3000})
```
This makes the "viewport" tall enough to render the entire page without scrolling. The trade-off is a very tall screenshot image, but it captures everything.

Alternatively, target Streamlit's inner scroll container directly:
```python
page.evaluate('document.querySelector("section.main").scrollTop = 1500')
```

**Impact:** Low for automated testing — the DOM text assertions still verify all content exists. Only pixel-level rendering bugs (blank images, broken map tiles) would be missed without visual inspection.

### 2. Cache directory grows over time
**Problem:** `cache/bands/` stores `.npz` files (~2–10 MB each) for every unique bbox+scene+resolution combo. Over many test runs with different presets, this grows.

**Workaround:** Periodically clear: `rm -rf cache/bands/*.npz`

### 3. STAC API or S3 transient failures
**Problem:** Element84 Earth Search or S3 COG reads occasionally time out or return 503.

**Workaround:** Re-run. The test has a 180s timeout for analysis. If it fails with "Data fetch failed" or "No scenes found," it's almost always transient. Two consecutive failures warrant investigating the STAC endpoint directly:
```bash
python -c "
from pystac_client import Client
c = Client.open('https://earth-search.aws.element84.com/v1')
print(list(c.get_collections()))
"
```

### 4. Memory pressure in constrained environments
**Problem:** The sandbox has ~4 GB RAM. Streamlit + numpy arrays + Playwright Chromium can approach 2 GB peak for large presets.

**Workaround:** The app has a built-in memory guard (rejects bboxes > 500 MB estimated). The default Amazon preset is safe. Avoid testing "Custom" locations with very large bounding boxes.

---

## Testing Additional Presets

The default E2E test only tests the Amazon Deforestation preset (the Streamlit default). To test other presets, modify the test or use this manual procedure:

1. Start Streamlit: `SKIP_WARMUP=1 uv run streamlit run app.py --server.port 8599 --server.headless true`
2. In the Playwright script, after page load, change the preset dropdown:
   ```python
   # Click the preset selectbox and choose a different preset
   page.locator('[data-testid="stSelectbox"]').first.click()
   page.locator('li:has-text("Las Vegas")').click()
   ```
3. Click "Analyze Change" and wait for results.

**Recommended presets to test (in priority order):**

| Preset | Index | Why test it |
|--------|-------|-------------|
| Amazon Deforestation | NDVI | Default; strong signal; reliable imagery |
| Las Vegas Urban Expansion | NDBI | Tests NDBI index path; urban context |
| Aral Sea Retreat | MNDWI | Tests MNDWI index path; water change |
| Tonga Volcanic Eruption | NDVI | Remote area; sparse Overture data |
| Turkish Earthquake | NDBI | Urban density; subtle change signal |

---

## Integration Tests (Optional, ~30–60s)

These hit real AWS/STAC endpoints. Run when you need to verify the full data pipeline works end-to-end, not just the UI.

```bash
uv run pytest tests/integration/ -v -m network --tb=short
```

**Expected:** All pass. Failures usually mean transient network issues — retry once before investigating.

---

## Condensed Checklist (Copy-Paste for Quick Runs)

```
1. uv run pytest tests/unit/ -v --tb=short              # ~7s, expect 61 passed
2. uv run python tests/e2e_visual_test.py               # ~35-90s, expect ALL CHECKS PASSED
3. View /tmp/e2e_screenshots/02_panel_a.png             # Two satellite images visible?
4. View /tmp/e2e_screenshots/05_full_page.png           # Stats and UI intact?
5. (optional) uv run pytest tests/integration/ -v -m network  # Network tests
```

If steps 1–4 all pass, the app is working correctly.
