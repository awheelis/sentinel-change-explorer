# Claude Code Prompt — Sentinel Change Explorer

## Instructions for use

1. Create a new project directory: `mkdir sentinel-change-explorer && cd sentinel-change-explorer`
2. Copy the spec file into it: `cp SPEC-phase1-sentinel-change-explorer.md .`
3. Install plugins:
```
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers
/plugin install context7
```
4. Paste the prompt below into Claude Code.

---

## The Prompt

```
Read the file SPEC-phase1-sentinel-change-explorer.md in this directory. That is the complete specification for what we are building.

Before writing any code, do the following:

1. Use context7 to look up current documentation for: pystac-client, rasterio, streamlit-folium, and the overturemaps Python package. Understand their current APIs before coding against them.

2. Use superpowers to search for "Element84 Earth Search STAC API sentinel-2-l2a asset key names" — we need to know the exact asset keys (they use names like "red", "nir", "swir16" instead of "B04", "B08", "B11"). Get this right before writing sentinel.py.

Now build the application following these rules:

WORKING STYLE:
- Work incrementally. Build and TEST each module before moving to the next.
- After completing each module, run a quick smoke test to verify it works. For example, after writing sentinel.py, write a small test script that fetches one scene and prints its metadata. Run it. Fix any issues before proceeding.
- Git commit after each working milestone with a clear commit message.
- If something fails, debug it yourself — read the error, check the docs, fix it. Don't ask me unless you're truly stuck.

BUILD ORDER:
1. Set up project structure, requirements.txt, and install dependencies.
2. src/sentinel.py — STAC search + band loading. Smoke test: fetch a Sentinel-2 scene for Las Vegas (bbox: -115.32, 36.08, -115.08, 36.28) for 2023-06, print scene ID and cloud cover. Then load the red, green, blue bands as numpy arrays and save a test PNG. THIS MUST WORK before you proceed.
3. src/indices.py — NDVI, NDBI, MNDWI computation from band arrays. Smoke test: compute NDVI for the Las Vegas scene and print min/max/mean values.
4. src/overture.py — Fetch Overture Maps buildings, transportation segments, and places for a bounding box. Cache results to disk. Smoke test: fetch buildings for the Las Vegas bbox and print the count.
5. src/visualization.py — Helper functions for rendering true-color images, change heatmaps, and folium maps with overlays.
6. config/presets.json — The preset locations from the spec.
7. app.py — The Streamlit application that ties everything together.
8. README.md — Clear documentation covering: what it does, a screenshot description, install instructions, how to run, tech stack, and a "Future Directions" section mentioning Phase 2 (embedding-based change detection with LeJEPA/vision foundation models, temporal sequences, Planet imagery integration).

TECHNICAL GOTCHAS TO WATCH FOR:
- Sentinel-2 bands have different resolutions (B02/B03/B04/B08 are 10m, B11 is 20m). When computing indices that use B11 (NDBI, MNDWI), you need to resample B11 to match the 10m bands.
- When reading COGs from S3, use rasterio with windowed reads for the bounding box. Do NOT download entire tiles. Use the STAC item's asset href directly.
- The Sentinel-2 COGs on Earth Search may need unsigned S3 access. Set the AWS_NO_SIGN_REQUEST=YES environment variable or configure rasterio accordingly.
- streamlit-folium is the package for embedding folium maps in Streamlit. Make sure to use st_folium() not st.map().
- For true color display, Sentinel-2 surface reflectance values need to be rescaled for visualization (they're uint16, typically 0-10000 range). Clip to a reasonable range and normalize to 0-255 for display.
- The overturemaps CLI writes GeoJSON. For programmatic use, you may need to shell out to the CLI or use DuckDB directly to query the parquet files. Check what the overturemaps Python package actually exposes as a Python API vs CLI-only.
- Ensure the app handles edge cases: no scenes found for a date range, all scenes too cloudy, Overture data empty for remote areas, network timeouts.

QUALITY:
- Type hints on all functions.
- Docstrings on all public functions (Google style).
- Clean imports, no unused imports.
- No hardcoded credentials or API keys (there shouldn't be any — all data sources are free and open).
- Do not include any personal identifying information in the code (the grading is anonymized).

When you're done, run the full app with `streamlit run app.py`, select a preset, and verify the end-to-end flow works. Fix any issues.
```

---

## Notes

- If Claude Code hits rate limits on the STAC API or S3 during testing, add a small delay between requests or reduce the bounding box size.
- If the overturemaps Python package doesn't have a clean Python API (it may be CLI-only), Claude Code should use subprocess to call the CLI, or switch to using DuckDB with the Overture S3 parquet paths directly.
- The whole build should take Claude Code about 30-60 minutes of compute time across the sessions. Your main job is reviewing what it produces and steering if something looks off.
