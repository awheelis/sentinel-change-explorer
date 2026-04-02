# App Feedback — Sentinel-2 Change Detection Explorer

> Reviewed live via headless browser (Playwright). All findings are reproducible.

---

## 🔴 CRITICAL — Breaks the Core Experience

### ~~1. Amazon Deforestation preset is completely non-functional~~ ✅ RESOLVED

**Fix:** Shrunk the Amazon bounding box from `[-62.90, -9.60, -62.50, -9.20]` (0.4° × 0.4°, ~1975 km²) to `[-62.80, -9.50, -62.65, -9.35]` (0.15° × 0.15°, ~274 km²) in `config/presets.json`. Also fixed a JSON serialization error in `src/visualization.py` when rendering Overture GeoDataFrames in folium (stripped complex nested properties to geometry-only). The Amazon preset now loads both scenes, renders all 4 panels, and completes without crashing.

---

### ~~2. App crashes the Streamlit server from memory exhaustion~~ ✅ RESOLVED

**Fix:** Added a pre-flight memory guard in `app.py` that estimates total pixel memory before any S3 loads. The calculation uses bbox dimensions, cos(center_lat) correction, target resolution (10m), 5 bands × 2 dates × 2 bytes/pixel to estimate total MB. If the estimate exceeds 500 MB, the app shows an `st.error()` telling the user to shrink their bounding box and aborts before calling `fetch_scene_data()`. Verified: Amazon preset (0.15°×0.15°, ~52 MB) loads normally; a 1°×1° custom bbox (~1,889 MB) is blocked with a clear error message.

---

### 3. S3 band data is never cached
Overture Maps data is cached to disk (good). Sentinel-2 band arrays are **not**. Every click of "Analyze Change" re-downloads the same COG windows from S3 over a network connection. The spec says: *"Cache fetched data in st.session_state or on disk to avoid re-fetching on every Streamlit rerun."*

In practice: Las Vegas takes ~40s every single time you change the index from NDVI → NDBI and click analyze again. There is zero reuse of previously fetched data within the same session.

---

## 🟠 MAJOR — Severely Degrades Quality

### 4. Broken emoji in the main title heading
The satellite emoji `🛰️` renders as `⊠` (a rectangular replacement glyph) in the H1 title. This is the first thing every reviewer sees. A broken character in the largest text on the page signals sloppiness, not polish. It fails on the headless Chromium renderer used in testing.

---

### 5. Load time exceeds spec requirement — even for the working preset
The spec states: *"keep bounding boxes small enough that data loads in under 30 seconds on a decent connection."* Las Vegas — the smallest working preset — takes **~40 seconds** in testing. That's over the stated target before any of the larger presets are even considered. Tonga, Aral Sea, and Turkish Earthquake presets were not tested but are similarly sized or larger.

---

### 6. Ghost instructional text persists during active analysis
When the user clicks "Analyze Change," the loading spinner appears alongside the info box: *"Select a preset location or enter custom coordinates, choose date ranges, and click Analyze Change to begin."* These two messages coexist on screen simultaneously for 30–40+ seconds. The onboarding prompt should be cleared the moment analysis begins. A user watching both a spinner and "click analyze to begin" will wonder if the app actually registered the click.

---

### 7. `rioxarray` is in the README tech stack table but missing from `requirements.txt`
The README lists `rioxarray` as part of the tech stack. It is not in `requirements.txt`. Any reviewer who runs `pip install -r requirements.txt` in a fresh environment and the code imports `rioxarray` will get an `ImportError` immediately. This is a submission-level defect — it prevents the app from being installed correctly if `rioxarray` is actually used in the source.

---

### 8. Panel D (Summary Statistics) is absent from the rendered output
The spec defines a mandatory **Panel D — Summary Statistics** with specific metrics: total area analyzed (km²), % area with significant vegetation/built-up/water change, scene dates, and cloud cover per scene. In live testing of the Las Vegas preset, the page text contains only "Panel A — True Color Comparison" and "Panel B+C — NDBI — Built-up Change Heatmap." No statistics panel appeared. The spec's most concrete, defensible deliverable — something trivially computed from existing numpy arrays — is missing from the UI.

---

### 9. No real progress feedback during analysis
The only loading feedback is a spinner with sequential text messages: first "Loading before bands from S3…", then "Loading after bands from S3…", then "Fetching Overture Maps context…". There is no progress bar, no percentage complete, no step count (e.g., "Step 2 of 4"), and no time elapsed. For a workflow that takes 40–480 seconds, this is unacceptable UX. Users will kill the tab assuming it's hung — especially because the ghost instructional text (issue #6) is still showing.

---

## 🟡 MODERATE — Meaningful but Fixable Issues

### 10. Default preset is "Custom…" — the wrong first impression
The dropdown opens on "Custom…" by default, which shows four raw coordinate input boxes with values for Las Vegas already pre-populated. A first-time reviewer sees a coordinate form, not a compelling demo. The default should be the most visually dramatic preset (Amazon or Aral Sea) with the coordinates hidden unless explicitly needed. The "wow" moment described in the spec ("within 30 seconds sees... a change heatmap glowing red") never happens automatically.

---

### 11. Satellite true-color imagery is significantly underexposed
In every screenshot captured during testing, the before/after true-color composites appear very dark — almost black in shadow areas with muted midtones. Sentinel-2 surface reflectance values need aggressive brightness stretching for visual display (typical 0–2000 reflectance mapped to 0–255). The current rescaling appears to clip too conservatively, making the imagery look like a night-time shot rather than a daytime satellite view. The "wow" panel is dim and hard to interpret.

---

### 12. Panel B and Panel C are merged into "Panel B+C"
The spec defines these as separate panels:
- **Panel B:** Change detection heatmap
- **Panel C:** Overture Maps context layer (toggle overlay)

The app combines them into a single "Panel B+C" with the Overture layers hardcoded as a checkbox rather than a distinct panel. The spec reviewer will notice this deviation immediately. It also means the Overture layers can't be compared cleanly against the raw heatmap.

---

### 13. Sidebar content overflows the viewport with no indicator
The sidebar's scrollable height is **1421px** against a 900px viewport. The bottom of the sidebar — including the "Analyze Change" button and the "Show Overture Maps layers" checkbox — is not visible without scrolling. There is no visual affordance (fade, scroll indicator, arrow) to tell users more content exists below. On a typical laptop screen at 1366×768, the button may be completely invisible.

---

### 14. Streamlit's "Stop" button is exposed in the header
The top-right of the app shows the "Stop" button — a Streamlit developer control that terminates the server process. This is exposed in what's meant to be a polished take-home submission. It signals that no custom theme or config was applied. A `[theme]` section in `.streamlit/config.toml` and `client.toolbarMode = "minimal"` would suppress this.

---

### 15. `pytest` listed as a runtime dependency
`requirements.txt` includes `pytest>=8.1.0`. This is a dev/test dependency, not a runtime one. Anyone doing `pip install -r requirements.txt` in production pulls in the test framework unnecessarily. A separate `requirements-dev.txt` should isolate test tooling.

---

## 🔵 MINOR — Polish and Accuracy Issues

### 16. No bounding box validation for custom coordinates
The custom coordinate inputs accept any float values. There is no validation that west < east, south < north, or that the box is within valid WGS84 ranges. Entering a flipped bbox or 0.0/0.0 for all values will silently trigger a (likely failing or nonsensical) STAC query.

### 17. The app offers no "zero-click demo" state
The landing page is blank except for the instructional info box. Loading the page and immediately seeing a result (pre-computed or cached) for the most dramatic preset would be far more compelling. The spec's description of what "Done" looks like assumes someone actively drives the UI, but a first impression with zero interaction should also be strong.

### 18. "Panel A" image labels are captions, not overlays
"Before — 2019-07-10" and "After — 2023-07-14" appear as small captions below the imagery, not as overlaid headers on the images. At a glance, it's not immediately obvious which image is which, especially when the images look similar in color tone. Bold overlaid labels with dates would be clearer.

### 19. The sidebar section headers ("Location & Dates", "Display") are not visually distinct from control labels
"Location & Dates" and "Display" use the same font weight and size as the control labels ("Preset location", "Change index"). The visual hierarchy is flat. Users scan sidebars quickly; stronger section header treatment would reduce cognitive load.

### 20. No `.streamlit/config.toml` — no page title, icon, or layout tuning
There is no Streamlit config file. The browser tab shows "localhost" as the page title. The app uses the default wide layout, but without explicit `layout = "wide"` in config there's no guarantee of consistent rendering. The page title in the browser tab should say something meaningful for a submission.

---

## Summary Scorecard

| Category | Verdict |
|---|---|
| Primary demo (Amazon) works | ❌ Crashes server |
| Load time meets spec (<30s) | ❌ 40s for best case, ∞ for worst |
| S3 band caching | ❌ Not implemented |
| All spec panels present (A, B, C, D) | ❌ Panel D missing |
| No server crashes | ❌ OOM kills process |
| Title renders correctly | ❌ Broken emoji |
| `requirements.txt` complete | ⚠️ rioxarray missing |
| UX during loading | ⚠️ Ghost text + no progress |
| Image quality | ⚠️ Underexposed |
| Code runs at all | ✅ Las Vegas preset works |
| Overture Maps integration | ✅ Data fetches and caches |
| STAC search works | ✅ Correct scenes selected |
