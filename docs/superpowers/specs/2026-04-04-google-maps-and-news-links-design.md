# Google Maps Button & Preset News Links

**Date:** 2026-04-04
**Status:** Approved

## Overview

Two new features for the post-analysis results area:

1. **"See on Google Maps" button** — opens the analyzed bounding box region in Google Maps in a new tab.
2. **Preset news article link** — each preset includes a link to a relevant news article.

## Feature 1: See on Google Maps

### URL Construction

A pure function `google_maps_url(bbox: tuple[float, float, float, float]) -> str` that:

- Computes center latitude and longitude from the bbox `(west, south, east, north)`
- Estimates a reasonable Google Maps zoom level from the bbox dimensions
- Returns a URL in the format: `https://www.google.com/maps/@{center_lat},{center_lng},{zoom}z`

Zoom estimation: Google Maps zoom is roughly `log2(360 / bbox_width)`, clamped to a reasonable range (2–18).

### UI Placement

After the existing download button (app.py ~line 625), in a row of action buttons using `st.columns`. Uses `st.link_button("See on Google Maps", url)` which opens in a new tab.

Available for **all** analyses — both preset and custom bbox.

### Module Location

Function lives in `src/visualization.py` alongside other display helpers.

## Feature 2: Preset News Article Link

### Data Model

Add a `"news_url"` string field to each preset object in `config/presets.json`:

```json
{
  "name": "Lahaina Wildfire, Maui",
  "news_url": "https://...",
  ...
}
```

News URLs for each preset:

| Preset | Source |
|--------|--------|
| Lahaina Wildfire, Maui | AP News or similar major outlet covering Aug 2023 wildfire |
| Pakistan Mega-Flood, Sindh | Reuters/BBC coverage of Aug 2022 flooding |
| Gigafactory Berlin | Reuters/BBC coverage of Tesla Gruenheide factory |
| Black Summer Bushfires, Australia | Major outlet covering 2019-2020 fire season |
| Egypt's New Capital | Major outlet covering the new administrative capital |

### UI Placement

Same button row as the Google Maps button. Uses `st.link_button("Read News Article", url)`. Only displayed when a preset is selected (not for custom bbox).

## Button Row Layout

```
[Download .tif]  [See on Google Maps]  [Read News Article*]
```

*News article button only present when a preset with `news_url` is active.

Using `st.columns(3)` when news link is available, `st.columns(2)` otherwise. The download button moves into this row.

## Testing Strategy (TDD)

### Unit Tests

1. **`test_google_maps_url_center`** — verifies center lat/lng are correctly computed from bbox
2. **`test_google_maps_url_zoom`** — verifies zoom is reasonable for different bbox sizes
3. **`test_google_maps_url_format`** — verifies URL starts with `https://www.google.com/maps/@`
4. **`test_presets_have_news_url`** — every preset in `presets.json` has a `news_url` key
5. **`test_preset_news_urls_are_valid`** — each `news_url` is a non-empty string starting with `https://`

### Files Modified

- `config/presets.json` — add `news_url` field to each preset
- `src/visualization.py` — add `google_maps_url()` function
- `app.py` — add button row with Google Maps + news link buttons
- `tests/unit/test_visualization.py` — tests for `google_maps_url()`
- `tests/unit/test_app_logic.py` — tests for preset news URL data integrity
