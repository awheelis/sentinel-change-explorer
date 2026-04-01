# Sentinel-2 Change Detection Explorer — Phase 1 Spec

## Purpose

A Streamlit application that lets users visually explore how a location on Earth has changed between two points in time, using free Sentinel-2 satellite imagery for analysis and free Overture Maps data for geographic context. The app should feel like a professional internal tool at a satellite analytics company — clean, fast, and insightful.

This is a take-home assessment for a Senior Software Engineer role at Planet (Bedrock Research team). The audience evaluating this cares about: geospatial AI workflows, clean engineering, and practical satellite imagery tooling. The submission is a zip file.

---

## User Experience (what the user does)

### 1. Choose a Location

The user selects an area of interest in one of two ways:

- **Preset locations** (dropdown): A curated list of ~5 locations where interesting, well-known changes have occurred. Each preset includes a name, description of what changed, bounding box coordinates, and suggested date ranges. Presets should load instantly and demonstrate the app's value without any user configuration.
- **Custom coordinates** (manual input): The user enters a bounding box (west, south, east, north) or a center point + radius. This is the "power user" path.

Suggested presets (pick locations where change is visually dramatic in Sentinel-2 true color or NDVI):

| Name | What Changed | Approx Dates |
|---|---|---|
| Las Vegas Urban Expansion | Desert → suburban development | 2019 vs 2023 |
| Tonga Volcanic Eruption | Hunga Tonga island before/after | 2021 vs 2022 |
| Amazon Deforestation Front | Forest clearing in Rondônia, Brazil | 2019 vs 2023 |
| Aral Sea Retreat | Water body shrinkage | 2018 vs 2023 |
| Turkish Earthquake (Hatay) | Urban destruction Feb 2023 | Dec 2022 vs Mar 2023 |

The builder should validate these work with Sentinel-2 availability and swap if needed.

### 2. Choose Dates

The user selects:

- **"Before" date range**: A month or date range for the baseline image.
- **"After" date range**: A month or date range for the comparison image.
- **Max cloud cover %**: Slider, default 20%. Filters out cloudy scenes.

The app finds the least-cloudy scene within each date range and uses that.

### 3. View Results

The app displays a results dashboard with these panels:

#### Panel A — True Color Comparison

Side-by-side view of the "before" and "after" true color satellite images (RGB from Sentinel-2 bands B04, B03, B02). Use two Streamlit columns with matched maps or images. This is the "wow" visual — users should immediately see the change.

#### Panel B — Change Detection Heatmap

A single map showing WHERE change occurred, colored by intensity. This is computed from one or more spectral indices (see Indices section below). The heatmap should use a diverging colormap (e.g., red = loss, blue = gain, transparent = no change).

#### Panel C — Overture Maps Context Layer

Overlaid on the change heatmap (or as a toggle), show relevant Overture Maps features for the same bounding box:

- **Buildings** (polygons): Show building footprints so users can see if change happened near structures.
- **Transportation** (lines): Roads and infrastructure as reference.
- **Places** (points): Named POIs for orientation.

This layer answers the question: "What's actually on the ground where this change happened?"

#### Panel D — Summary Statistics

A sidebar or card section showing:

- Total area analyzed (km²)
- % of area with significant vegetation change (NDVI delta beyond threshold)
- % of area with significant built-up change (NDBI delta beyond threshold)
- % of area with significant water change (MNDWI delta beyond threshold)
- Dates of the specific scenes selected
- Cloud cover % of each scene used

---

## Spectral Indices to Compute

For each date (before and after), compute the following from Sentinel-2 bands, then compute the difference (after minus before):

### NDVI — Normalized Difference Vegetation Index
- Formula: (NIR - Red) / (NIR + Red)
- Sentinel-2 bands: (B08 - B04) / (B08 + B04)
- Measures: Vegetation health/density
- Negative diff = vegetation loss, Positive diff = vegetation gain

### NDBI — Normalized Difference Built-up Index
- Formula: (SWIR - NIR) / (SWIR + NIR)
- Sentinel-2 bands: (B11 - B08) / (B11 + B08)
- Measures: Built-up/urban areas
- Positive diff = new construction/urbanization

### MNDWI — Modified Normalized Difference Water Index
- Formula: (Green - SWIR) / (Green + SWIR)
- Sentinel-2 bands: (B03 - B11) / (B03 + B11)
- Measures: Water bodies
- Negative diff = water loss, Positive diff = water gain

The user should be able to toggle which index is displayed on the heatmap. Default to NDVI.

---

## Data Sources & Access Patterns

### Sentinel-2 L2A Imagery

- **Source**: Element84 Earth Search STAC API
- **Endpoint**: `https://earth-search.aws.element84.com/v1`
- **Collection**: `sentinel-2-l2a`
- **Auth**: None required
- **Library**: `pystac-client` for search, `rasterio`/`rioxarray` for reading COGs
- **Bands needed**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR) — all 10m or 20m resolution
- **Query pattern**: Search by bounding box + datetime range + cloud cover filter. Sort by cloud cover ascending. Take the first (least cloudy) result.
- **Data loading**: Read band GeoTIFFs directly from S3 URLs in the STAC item assets. Use windowed reads for the bounding box area only — do NOT download full tiles.
- **Important**: The STAC asset keys for Earth Search use lowercase names like `red`, `green`, `blue`, `nir`, `swir16` — NOT `B04`, `B03`, etc. The builder must check actual asset key names.

### Overture Maps

- **Source**: Overture Maps Foundation, hosted on AWS S3
- **Library**: `overturemaps` Python package (pip install overturemaps)
- **Auth**: None required
- **Types to fetch**: `building`, `segment` (transportation), `place`
- **Query pattern**: Download by bounding box, output as GeoJSON. This can be done via CLI or Python API.
- **Caching**: Overture data should be cached locally after first fetch for a given bounding box, since it doesn't change between before/after dates.

---

## Technical Requirements

### Stack
- **Python 3.10+**
- **Streamlit** for the frontend
- **pystac-client** for STAC search
- **rasterio / rioxarray** for reading Sentinel-2 COGs
- **geopandas** for vector data handling
- **folium** for interactive maps (preferred — better Streamlit integration via streamlit-folium)
- **matplotlib** or **plotly** for charts/statistics
- **overturemaps** for Overture data
- **numpy** for index computation

### Performance Considerations
- Sentinel-2 tiles are large. The app MUST use windowed reads (only fetch pixels within the bounding box) rather than downloading full tiles.
- For the preset locations, keep bounding boxes small enough that data loads in under 30 seconds on a decent connection.
- Cache fetched data in `st.session_state` or on disk to avoid re-fetching on every Streamlit rerun.
- Show a progress indicator during data loading.

### Code Quality
- Clean, modular code. Separate data fetching, computation, and visualization.
- Type hints on all functions.
- Docstrings on all public functions.
- Error handling for network failures, missing data, no cloud-free scenes found.
- A clear README.md explaining: what it does, how to install, how to run, what to look at, and where it could go next (Phase 2 teaser).

### Project Structure
```
sentinel-change-explorer/
├── app.py                     # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── sentinel.py            # STAC search + Sentinel-2 band loading
│   ├── indices.py             # Spectral index computation (NDVI, NDBI, MNDWI)
│   ├── overture.py            # Overture Maps data fetching + caching
│   └── visualization.py       # Map rendering, comparison views, heatmaps
├── config/
│   └── presets.json           # Preset locations with bbox, dates, descriptions
├── requirements.txt
├── README.md
└── .gitignore
```

---

## What "Done" Looks Like

A reviewer unzips the file, runs `pip install -r requirements.txt && streamlit run app.py`, selects "Amazon Deforestation Front" from the dropdown, and within 30 seconds sees:

1. A side-by-side true color satellite view showing forest in 2019 and cleared land in 2023.
2. A change heatmap glowing red over the deforested areas.
3. Building footprints and roads from Overture Maps overlaid, showing the infrastructure that replaced forest.
4. A stats panel saying something like "38% of the analyzed area shows significant vegetation loss."

The code is clean, the README is clear, and the reviewer thinks "this person understands geospatial data pipelines."

---

## Out of Scope for Phase 1

- Any ML models or embeddings (Phase 2)
- User authentication
- Persistent storage / database
- Deployment (runs locally only)
- Time-lapse animation (nice-to-have if time allows, not required)
- Drawing tools for custom polygons (bounding box input is sufficient)

---

## Phase 2 Teaser (for README and interview discussion)

The README should mention future directions:

- **Embedding-based change detection**: Use a vision foundation model (e.g., LeJEPA, DINOv2) to compute patch embeddings for before/after imagery. Embedding distance captures semantic change beyond what spectral indices can detect — e.g., distinguishing "new parking lot" from "new building" even when both show similar NDBI increase.
- **Temporal sequences**: Extend from two-date comparison to multi-date time series, enabling trend detection and anomaly flagging.
- **Integration with Planet imagery**: Replace Sentinel-2 (10m, 5-day revisit) with Planet's SkySat or PlanetScope (0.5-3m, daily revisit) for higher resolution and temporal density.
