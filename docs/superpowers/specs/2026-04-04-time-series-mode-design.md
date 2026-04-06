# Time-Series Mode — Design Spec

**Date:** 2026-04-04
**Status:** Draft
**Backlog item:** #6 — Time-Series Mode (Tier 3, Medium-High complexity)

---

## Overview

Add a temporal trend chart to the existing bi-temporal analysis workflow. For each analysis run, the app fetches all available Sentinel-2 scenes across the full date span (before-start → after-end), computes the mean value of the selected spectral index per scene at reduced resolution, and plots a time-series chart with anomaly detection. The chart appears below the folium map as an integrated part of every analysis — no mode toggle needed.

## Goals

- Show the temporal trajectory of the selected index across the AOI
- Flag anomalous scenes (> 2σ from rolling mean) that may indicate events (fire, flood, harvest)
- Contextualize the bi-temporal before/after comparison within a broader trend
- Add minimal latency to the existing workflow (target: < 10s first run)

## Non-Goals

- Per-pixel temporal analysis or animated heatmap sliders (future Approach B/C)
- Multi-index overlay on a single chart (only the selected index is plotted)
- Separate date range controls (date span is auto-derived from existing before/after pickers)

---

## Data Pipeline

### New module: `src/timeseries.py`

#### `fetch_time_series(bbox, date_span, index_name, max_cloud_cover, apply_scl_mask) → list[dict]`

1. Call `search_scenes(bbox, date_span, max_cloud_cover, max_items=20)` to get available scenes. Note: `search_scenes` returns results sorted by cloud cover ascending. After fetching, **re-sort by datetime** to produce a chronological series. If the date span is long (> 6 months), scenes may cluster in clear-sky seasons — this is acceptable since cloudy seasons contribute less-reliable data anyway.
2. For each scene, in parallel via `ThreadPoolExecutor(max_workers=6)`:
   a. Determine required bands from the index name (see band mapping below).
   b. Call `load_bands(scene, bbox, band_keys, target_res=60)` — reduced resolution for speed.
   c. If `apply_scl_mask` is True, also load the `scl` band and apply cloud/shadow masking via existing `build_scl_mask()` + `apply_mask()`.
   d. Compute the index using existing functions (`compute_ndvi`, `compute_ndbi`, `compute_mndwi`, `compute_evi`).
   e. Compute `np.nanmean()` of the index array for the mean value.
   f. Compute `valid_pixel_pct` as the percentage of non-NaN pixels after masking.
3. Return a list of dicts sorted by datetime:
   ```python
   {
       "datetime": str,       # ISO 8601
       "scene_id": str,
       "mean_index": float,
       "cloud_cover": float,
       "valid_pixel_pct": float,
   }
   ```

#### Band mapping per index

| Index | Bands loaded             |
|-------|--------------------------|
| NDVI  | `["nir", "red"]`         |
| NDBI  | `["swir16", "nir"]`      |
| MNDWI | `["green", "swir16"]`    |
| EVI   | `["nir", "red", "blue"]` |

If `apply_scl_mask` is True, `"scl"` is appended to the band list for every index.

#### Caching

Results cached as JSON to `cache/timeseries/` keyed on `md5(bbox + date_span + index_name + max_cloud_cover + apply_scl_mask)`. The cached payload is the list of scene dicts — small (< 1 KB typically). Raw low-res bands are not persisted.

---

## Anomaly Detection & Statistics

#### `compute_anomalies(series) → dict`

Takes the list of scene dicts from `fetch_time_series()` and returns:

```python
{
    "rolling_mean": list[float],     # per-scene rolling mean
    "rolling_std": list[float],      # per-scene rolling std
    "is_anomaly": list[bool],        # True if |value - rolling_mean| > 2 * rolling_std
    "trend_slope": float,            # linear regression slope (index units per month)
    "trend_direction": str,          # "increasing", "decreasing", or "stable"
    "volatility": float,             # std of all mean_index values
    "anomaly_count": int,
    "max_jump": float,               # largest absolute scene-to-scene change
    "max_jump_date": str,            # datetime of the scene after the largest jump
}
```

**Rolling window:** 3 scenes. If fewer than 4 total usable scenes, skip anomaly detection (set all `is_anomaly` to False) and still compute trend/volatility.

**Cloudy scenes:** Scenes with `valid_pixel_pct < 30` are flagged as "mostly clouded." They are excluded from rolling statistics and trend computation but included in the chart as gray markers so the user sees the temporal gap context.

**Edge case — zero usable scenes:** Return `None`. The app shows `st.info("No scenes with sufficient cloud-free coverage found for temporal analysis.")` instead of the chart.

---

## Visualization

### New function in `src/visualization.py`: `time_series_chart(series, anomalies, index_name, before_date, after_date) → matplotlib.figure.Figure`

**Chart elements:**
- **Data line:** Solid line connecting data points with circular markers (radius 5). Color by index: NDVI=#4CAF50, NDBI=#FF9800, MNDWI=#2196F3, EVI=#2E7D32.
- **Rolling mean:** Thin dashed line in a lighter shade of the index color.
- **±2σ confidence band:** Shaded region around the rolling mean (same color, 20% opacity).
- **Anomaly markers:** Red dots (#E53935), larger radius (8), with annotation showing sigma deviation (e.g., "−2.3σ").
- **Cloudy scenes:** Gray (#888) hollow dots (unfilled circle, dashed edge), not connected to the main line.
- **Before/after reference lines:** Vertical dashed gray lines with text labels at top ("Before (Jun 2023)", "After (Jun 2024)").
- **Legend:** Upper-right corner — Mean {INDEX}, Rolling mean, ±2σ band, Anomaly.
- **Title:** "{INDEX} Temporal Trend — {start_date} to {end_date} ({n} scenes)"
- **Axes:** X = scene dates, Y = "Mean {INDEX}"

**Figure size:** `(12, 4)` — full-width, compact height.

### Summary caption

Rendered as `st.caption()` below the chart:

```
Trend: ↑ increasing (+0.018/month) · Volatility: 0.045 · 1 anomaly detected · 1 scene excluded (cloud cover)
```

Arrow direction: ↑ for increasing, ↓ for decreasing, → for stable (|slope| < 0.005/month).

---

## App Integration

### Placement

Full-width section below the folium map and above the statistics/histogram section. Renders inside a `st.status()` container while loading.

### Execution flow in `app.py`

1. Existing bi-temporal fetch and rendering runs unchanged.
2. After the folium map renders, derive `date_span` from `before_start` and `after_end`.
3. Check session state cache for existing time-series result.
4. If not cached, show `st.status("Loading temporal trend…")` and call `fetch_time_series()`.
5. On success, call `compute_anomalies()`, then render `time_series_chart()` via `st.pyplot()` and summary via `st.caption()`.
6. Cache result in `st.session_state["_results"][cache_key]` alongside existing bi-temporal data.

### Session state cache key

```python
ts_cache_key = f"{bbox}|{before_range}|{after_range}|{max_cloud}|ts|{index_name}|{scl_enabled}"
```

### No new sidebar controls

Time-series uses the existing index selector, before/after date pickers, cloud cover slider, and SCL masking toggle. No mode toggle — it always runs.

---

## File Changes

| File | Change |
|------|--------|
| `src/timeseries.py` | **New.** `fetch_time_series()`, `compute_anomalies()`, band mapping, caching logic. |
| `src/visualization.py` | **Add** `time_series_chart()` function. |
| `app.py` | **Add** time-series section after folium map. Import new functions, derive date span, render chart + caption. |
| `tests/test_timeseries.py` | **New.** Unit tests for `fetch_time_series()` (mocked STAC), `compute_anomalies()` (synthetic data). |
| `tests/test_visualization.py` | **Add** test for `time_series_chart()` returning a valid Figure. |

---

## Testing Strategy

- **`compute_anomalies()`:** Synthetic series with known anomaly → verify detection. Series with < 4 points → verify anomaly detection is skipped. All-cloudy series → verify None return.
- **`time_series_chart()`:** Verify returns `matplotlib.figure.Figure`. Verify correct number of plotted lines/markers.
- **`fetch_time_series()`:** Mock `search_scenes` + `load_bands`, verify correct band selection per index, verify cloud masking integration, verify caching writes and reads.
- **Integration:** Run app with a preset location, verify chart appears below map, verify anomaly markers render for known event locations.
