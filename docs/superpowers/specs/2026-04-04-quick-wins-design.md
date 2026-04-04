# Design Spec: Quick Wins (Enhancements 1, 2, 3, 4, 6)

Date: 2026-04-04

## Overview

Five low-complexity enhancements from the Enhancement Roadmap, targeting scientific credibility and UX polish. All are independent and can be implemented in any order.

---

## 1. Enhanced Vegetation Index (EVI)

**What:** Add EVI as a fourth spectral index option alongside NDVI, NDBI, MNDWI.

**Formula:** `EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)`

**Changes:**
- `src/indices.py`: Add `compute_evi(nir, red, blue)` — returns float32 array clipped to [-1, 1]. Handle division by zero same as existing functions.
- `app.py`: Register in `INDEX_FUNCTIONS` as `"evi": ("EVI", compute_evi, ["nir", "red", "blue"])`.
- `app.py`: Change `compute_index_for_bands()` to unpack band args dynamically: `fn(*[bands[k] for k in band_order])` instead of hardcoding two positional args.
- `app.py`: Add EVI to `_INDEX_LABELS` dict.

**Edge cases:**
- Denominator `NIR + 6*Red - 7.5*Blue + 1` can be zero or negative with certain reflectance values. Use `np.where(denom == 0, 0.0, ...)` and clip result to [-1, 1].
- All required bands (nir, red, blue) are already in `ALL_BAND_KEYS`.

**Tests:**
- Known-value test: specific NIR/Red/Blue → expected EVI
- Zero-denominator safety
- Output dtype float32, range [-1, 1]
- uint16 input acceptance

---

## 2. Adaptive Thresholding (Otsu's Method)

**What:** Add an "Auto threshold" checkbox that computes the optimal change/no-change boundary from the delta histogram using Otsu's method.

**Changes:**
- `src/indices.py`: Add `compute_adaptive_threshold(delta)` that applies Otsu's method to `np.abs(delta)`. Returns a float threshold value.
- Implementation: Use `skimage.filters.threshold_otsu` if available, otherwise manual implementation (~20 lines). Check if skimage is already a dependency; if not, implement manually to avoid a new dep.
- `app.py` sidebar: Add checkbox "Auto threshold" below the threshold slider. When checked, compute threshold from delta and display it as the slider value (disabled). When unchecked, manual slider is active.
- `app.py` Panel D: Show "Threshold: X.XX (auto)" or "Threshold: X.XX (manual)" in the summary stats.

**Algorithm (manual Otsu on absolute delta):**
1. Take `np.abs(delta)` — we're separating "changed" from "unchanged" pixels
2. Compute histogram (256 bins over [0, max])
3. Find threshold that maximizes inter-class variance between "unchanged" (below) and "changed" (above)

**Edge cases:**
- Uniform delta (no change anywhere) → Otsu returns 0. Fall back to default 0.10.
- All-NaN delta → return default 0.10.

**Tests:**
- Bimodal distribution → threshold between the two peaks
- Uniform distribution → returns fallback default
- Known synthetic data with clear change/no-change clusters

---

## 3. Index-Specific Colormaps

**What:** Auto-select a semantically appropriate colormap based on the active index. User can still override.

**Default mappings:**
| Index | Colormap | Rationale |
|-------|----------|-----------|
| NDVI  | RdYlGn   | Red=loss, green=gain (vegetation) |
| EVI   | RdYlGn   | Same as NDVI |
| MNDWI | BrBG     | Brown=dry, blue-green=water |
| NDBI  | PuOr     | Neutral, no semantic conflict |

**Changes:**
- `app.py`: Add `DEFAULT_COLORMAPS` dict mapping index key → colormap name.
- `app.py`: Expand colormap selectbox options to: `["RdYlGn", "BrBG", "PuOr", "RdBu", "RdYlBu", "PiYG", "PRGn", "coolwarm"]`.
- `app.py`: Track whether user has manually changed the colormap via session state. When index changes and user hasn't manually overridden, auto-select the default for that index.
- Logic: On index change, if `colormap == DEFAULT_COLORMAPS[previous_index]` (user didn't touch it), update to `DEFAULT_COLORMAPS[new_index]`. If user picked something else, leave it.

**Tests:**
- Unit test: verify `DEFAULT_COLORMAPS` maps all keys in `INDEX_FUNCTIONS`.

---

## 4. Confidence / Quality Indicator

**What:** Display data quality metrics in Panel D so users can assess reliability of detected changes.

**Metrics:**
- Cloud cover % for both scenes (already in scene dict)
- Sun elevation angle (extract from STAC `view:sun_elevation`)
- Time gap between scenes (compute from `scene['datetime']`)

**Quality thresholds:**
| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Cloud cover | < 15% | 15-30% | > 30% |
| Sun elevation | > 30° | 20-30° | < 20° |
| Time gap | < 1 year | 1-2 years | > 2 years |

**Changes:**
- `src/sentinel.py`: In `search_scenes()`, add `sun_elevation` to the returned scene dict from `item.properties.get("view:sun_elevation")`. May be None for some scenes.
- `app.py` Panel D: Add "Data Quality" expander with color-coded metrics using `st.metric` or styled markdown.
- Display format: colored circles (🟢🟡🔴) + metric value + explanation.

**Edge cases:**
- `view:sun_elevation` may not exist in all STAC items → display "N/A" gracefully.
- Time gap calculation: parse ISO datetime strings, compute delta in days, display as "X months" or "X years".

**Tests:**
- Unit test for quality classification function (given cloud/sun/gap → returns green/yellow/red)
- Test graceful handling of missing sun_elevation

---

## 5. NDVI Saturation Warning (Enhancement #6 from roadmap)

**What:** Warn users when NDVI appears saturated (high-LAI regions) and suggest switching to EVI.

**Trigger:** After computing `before_index` for NDVI, check `np.percentile(before_index, 90) > 0.75`.

**Display:** Inline `st.warning()` above Panel B (the heatmap), with text:
> "⚠️ NDVI appears saturated in this region (90th percentile: X.XX). Dense vegetation compresses NDVI's dynamic range. Consider switching to **EVI** for better sensitivity to canopy changes."

**Changes:**
- `app.py`: After computing `before_index`, add saturation check. Only for NDVI index.
- Only show warning once per analysis (not on every rerun).

**Tests:**
- Test that high-value array (90th pct > 0.75) triggers warning condition
- Test that moderate-value array does not trigger
- Test only applies to NDVI, not other indices

---

## Implementation Order

1 (EVI) → 5 (NDVI Saturation Warning, depends on EVI existing) → 2 (Adaptive Threshold) → 3 (Colormaps) → 4 (Quality Indicator)

This order ensures EVI is available before the saturation warning references it, and groups the index-related changes before the UI-focused ones.
