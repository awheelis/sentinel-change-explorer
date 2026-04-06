# Sentinel-2 Change Detection Explorer

A Streamlit application that lets you visually compare two dates of Sentinel-2
satellite imagery for any location on Earth — detecting vegetation loss,
urbanization, and water change using spectral indices. Overture Maps building
footprints, roads, and POIs provide ground-truth context.

## What It Does

1. **Choose a location** — 5 curated presets (Lahaina wildfire, Pakistan
   mega-flood, Gigafactory Berlin, Black Summer bushfires, Egypt's new
   capital) or enter a custom bounding box.
2. **Choose two date ranges** — the app finds the least-cloudy Sentinel-2 scene
   in each range automatically.
3. **View results**:
   - Side-by-side true-color satellite images
   - NDVI / NDBI / MNDWI / EVI change heatmap (red = loss, blue = gain)
   - Classified change map (vegetation loss, urban conversion, flooding, etc.)
   - Overture Maps buildings, roads, and POIs overlaid on the change map
   - Time-series anomaly chart for the selected index
   - Summary statistics (% area changed, scene metadata)
   - GeoTIFF export of change results

## Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency and
environment management. Install uv, then:

```bash
uv sync
```

This creates `.venv/` and installs all dependencies (including dev tools)
from `pyproject.toml` / `uv.lock`. Requires Python 3.10+ (uv will fetch
one automatically if needed). No API keys needed — all data sources are
free and open.

## Run

```bash
uv run streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## What to Look At

- **Lahaina Wildfire, Maui** — select NDVI. Red patches = vegetation
  incinerated by the August 2023 wildfire. Overture buildings show the
  dense residential fabric that was destroyed.
- **Gigafactory Berlin** — select NDBI. Red = new impervious surfaces
  (factory roof, parking lots) carved out of pine forest.
- **Pakistan Mega-Flood, Sindh** — select MNDWI. Blue = floodwater that
  submerged farmland and villages in August 2022.

## Tests

Tests are organized into three tiers. Unit and performance tests run offline in
under 5 seconds; integration tests hit real AWS/STAC endpoints and need internet.

```bash
uv run pytest tests/unit/ tests/perf/     # fast, no network (~3s)
uv run pytest -m network                  # integration tests (requires internet)
uv run pytest                             # everything
uv run python tests/e2e_visual_test.py    # visual E2E via headless Chromium (~35-90s)
```

For a detailed visual E2E testing procedure (install steps, timing expectations,
screenshot verification checklist, known issues) see
[VISUAL_E2E_TEST_PROCEDURE.md](VISUAL_E2E_TEST_PROCEDURE.md).

## Tech Stack

| Component | Library |
|-----------|---------|
| Frontend | Streamlit + streamlit-folium |
| STAC search | pystac-client (Element84 Earth Search v1) |
| Band loading | rasterio (windowed COG reads over S3) |
| Index computation | numpy (NDVI, NDBI, MNDWI, EVI) |
| Vector data | Overture Maps + geopandas |
| Maps | folium |

Data sources: [Sentinel-2 L2A on AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/),
[Overture Maps Foundation](https://overturemaps.org/).

## Experimental: Foundation Model (PoC)

An opt-in sidebar toggle runs the same before/after tiles through a small
self-supervised **5-channel ViT-Tiny/8 with register tokens**, pretrained
with the [LeJEPA](https://arxiv.org/abs/2511.08544) objective on a curated
Sentinel-2 chip dataset. The 16×16 patch-token grid (256 positions) plus
register tokens ([Darcet et al. 2024](https://arxiv.org/abs/2309.16588))
give the PCA→RGB feature visualizations the crisp DINOv2-style look that
coarser conv backbones can't produce. The panel shows PCA→RGB feature
images for the before and after scenes plus a learned cosine-distance
change map alongside the classical NDVI-delta heatmap.

- **Dataset:** [falafel-hockey/sentinel2-lejepa-preset-biased-small](https://huggingface.co/datasets/falafel-hockey/sentinel2-lejepa-preset-biased-small)
- **Model (gold):** [falafel-hockey/lejepa-vit-tiny-patch8-sentinel2-5band](https://huggingface.co/falafel-hockey/lejepa-vit-tiny-patch8-sentinel2-5band)
- **Model (legacy):** [falafel-hockey/lejepa-resnet18-sentinel2-5band](https://huggingface.co/falafel-hockey/lejepa-resnet18-sentinel2-5band) — first PoC, kept for comparison
- **Full walkthrough:** [`src/experimental/README.md`](src/experimental/README.md)

This is a proof of concept, not a SOTA system — the goal is to demonstrate
an end-to-end foundation-model pipeline (dataset curation → publication →
SSL pretraining → model publication → inference UI) on the same
infrastructure the classical app already uses. Enable with:

```bash
uv sync --extra experimental
```

then toggle "Experimental: Foundation Model" in the app sidebar.

## Future Directions (Phase 2)

**Embedding-based change detection** — Use a vision foundation model (LeJEPA,
DINOv2) to compute patch embeddings for before/after imagery. Embedding distance
captures *semantic* change beyond spectral indices — distinguishing "new parking
lot" from "new building" even when both show similar NDBI increases.

**Temporal sequences** — Extend from two-date comparison to multi-date time
series, enabling trend detection and anomaly flagging over 12+ months.

**Planet imagery integration** — Replace Sentinel-2 (10m resolution, 5-day
revisit) with Planet's PlanetScope or SkySat (0.5–3m, daily revisit) for higher
spatial and temporal fidelity. This would integrate directly with Planet's
Tasking API for on-demand collection.
