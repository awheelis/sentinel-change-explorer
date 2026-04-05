# Prioritized Backlog: Sentinel-2 Change Detection Explorer

Consolidated from ISSUES.md, NEXT_FEATURES.md, and ENHANCEMENT_ROADMAP.md. Items already implemented have been removed. Prioritized by impact and effort. Interview critique context from Kevin LaTourette (CEO) and Dr. Matt Reisman (CTO) at Bedrock Research / Planet Labs.

---

## Tier 1 — High Impact, Moderate Effort

### 1. ~~Relative Radiometric Normalization~~ DONE

**Why (Dr. Reisman):** Comparing two scenes captured under different atmospheric conditions, sun angles, and sensor geometries introduces systematic false positives. L2A surface reflectance helps but doesn't eliminate relative differences between dates. This was "one of the first problems we solved" at Lockheed.

**Algorithm (PIF-based):**
1. Identify pseudo-invariant features (PIFs) — pixels unlikely to have changed (deep water, bare rock, urban pavement)
2. Fit per-band linear regression between before/after reflectance on PIF pixels
3. Apply correction to after-scene bands before index computation

**Implementation:**
- Add `src/normalization.py` with `normalize_pif(before_bands, after_bands)`
- PIF selection: pixels where all indices show < 0.02 absolute change
- Add sidebar toggle: "Radiometric normalization" checkbox (default on)
- Unit tests verifying normalization reduces variance on synthetic shifted data

**Complexity:** Medium.

**Reference:** Schott, J.R., Salvaggio, C., & Volchok, W.J. (1988). "Radiometric scene normalization using pseudo-invariant features." *Remote Sensing of Environment*, 26(1), 1-16.

---

### 2. ~~SCL Cloud/Shadow Masking~~ DONE

**Why:** Unmasked clouds create false positives that look like dramatic vegetation loss or gain. Sentinel-2's Scene Classification Layer (SCL) provides per-pixel classification that can mask out problematic pixels before index computation.

**Implementation:**
- Load the `scl` band from the same STAC item using existing `load_bands` infrastructure
- Mask out SCL values: 3 (cloud shadow), 8 (cloud medium), 9 (cloud high), 10 (thin cirrus)
- Apply mask to all band arrays before index computation (set to `np.nan`)
- Add sidebar metric showing "% pixels masked" for each date
- If > 50% of AOI is masked, warn suggesting a different date range

**Complexity:** Medium. Requires careful handling of NaN propagation through index computation.

---

### 3. ~~Multi-Index Change Classification~~ DONE

**Why (Dr. Reisman):** Single-index thresholding catches obvious changes but can't distinguish *why* an area changed. Combining multiple indices enables semantic classification — urbanization vs. drought stress vs. flooding.

**Classification rules (numpy threshold logic):**
| Condition | Category | Color |
|-----------|----------|-------|
| NDVI down AND NDBI up | Urban Conversion | Orange |
| NDVI down AND MNDWI stable | Vegetation Loss / Deforestation | Red |
| MNDWI up AND NDVI down | Flooding / Water Gain | Blue |
| NDVI up | Vegetation Gain / Recovery | Green |
| Everything else | Unchanged | Transparent |

**Implementation:**
- Compute all three indices for both dates (already done for Panel D summary)
- Apply classification rules as numpy boolean operations
- Render as RGBA overlay on folium map (separate toggleable layer)
- Add a legend to the map

**Complexity:** Medium. Logic is straightforward; UI integration needs a new layer toggle.

---

## Tier 2 — Medium Impact, Low-Medium Effort

### 4. ~~Folium Map Blank Rendering Fix~~ OBE

**Status:** Overtaken by events — a JS resize workaround is already in `build_folium_map()` and remaining blank-flash is acceptable.

---

### 5. ~~Configurable S3/Rasterio Environment Variables~~ DONE

**Status:** Already implemented — `_RASTERIO_ENV` in `sentinel.py` reads all values from `os.environ` with sensible defaults. No further work needed.

---

## Tier 3 — High Impact, High Effort

### 6. ~~Time-Series Mode~~ DONE

**Status:** Implemented in `src/timeseries.py` + `time_series_chart()` in `src/visualization.py`. Fetches scenes across the full date span at 60m resolution, computes mean index per scene, detects anomalies with a rolling 2σ window, and renders a temporal trend chart below the folium map. Results cached on disk and in session state.

---

### 7. Anomaly Detection Layer

```
INPUT FROM ALEX: 
I DON'T WANT THIS FEATURE
```

**Why (Dr. Reisman):** A pixel might show moderate NDVI decrease AND moderate NDBI increase — individually below threshold, but together indicating clear urbanization. Multi-index anomaly detection catches compound changes that single-index thresholding misses.

**Implementation:**
- Stack NDVI, NDBI, MNDWI deltas into a 3-channel change vector per pixel
- Compute Mahalanobis distance or Z-score from mean change vector
- High-distance pixels are anomalous regardless of which index drives it
- Display as an additional "Anomaly" heatmap option alongside single-index maps

**Complexity:** Medium.

**Reference:** Bruzzone, L. & Prieto, D.F. (2000). "Automatic analysis of the difference image for unsupervised change detection." *IEEE Transactions on Geoscience and Remote Sensing*, 38(3), 1171-1182.

---

### 8. Foundation Model Integration (Proof of Concept) — **IN PROGRESS**

**Status (2026-04-04):** Phases 1-6 implemented behind the `experimental`
extras. Dataset published at
[alexw0/sentinel2-lejepa-preset-biased-small](https://huggingface.co/datasets/alexw0/sentinel2-lejepa-preset-biased-small),
small PoC model trained on M1 CPU and published to
[alexw0/lejepa-resnet18-sentinel2-5band](https://huggingface.co/alexw0/lejepa-resnet18-sentinel2-5band),
inference panel wired into the Streamlit app. Remaining: full GPU training run
on lightning.ai to replace the PoC weights. See `src/experimental/README.md`.

```
INPUT FROM ALEX

What if we trained our own small LeJEPA model and used that. THe LeWM paper on had a model with only 15m parameters. Perhaps that wouldn't be hard to store / pull from hugging face or run???

Possible implementation? 
Start experimental LeJEPA (in experimental folder)
- Download Sentinel data to make a small dataset
- use stable pertaining repo to train a lejepa on data
-- Model must be small enough to run on a M1 mac w 8 gb of ram. 
-- push model to hugging face after training is done 
- run LeJEPA on input data
-- pull model from hugging face
- create RGB image using 1, 2, 3rd principle components of features 
- paste images side by side 
- user sees "experimental" tab on app main screen. Note that required specs. add details on the LeJEPA experimental feature. 

For context, look at LeJEPA and LeWorldModel papers
```
**Why (Kevin):** "Every serious geospatial company is investing in foundation models and deep learning for change detection. Your approach is the methodology from 2010. Show me you know where the field is going — even a branch with a simple experiment — and the conversation changes significantly."

**Possible approaches:**
- **Prithvi feature extraction:** Load NASA/IBM `ibm-nasa-geospatial/Prithvi-100M` from HuggingFace, extract embeddings for before/after scenes, compute cosine distance as "learned change" heatmap
- **Siamese CNN:** Small Siamese network on labeled change detection datasets (LEVIR-CD, WHU-CD)
- **ChangeFormer:** Transformer-based Siamese architecture (Bandara & Patel, 2022)

**Implementation (Prithvi PoC):**
- Optional dependency: `pip install transformers torch`
- Extract features for 6-band inputs, compute cosine distance between feature maps
- Display alongside traditional index-based heatmap for comparison

**Complexity:** High. Requires GPU for reasonable inference times, large model download, careful input normalization. Best as a separate branch or experimental feature.

**References:**
- Jakubik, J. et al. (2023). "Foundation Models for Generalist Geospatial Artificial Intelligence." *arXiv:2310.18660.*
- Daudt, R.C., Le Saux, B., & Boulch, A. (2018). "Fully Convolutional Siamese Networks for Change Detection." *IEEE ICIP.*
- Bandara, W.G.C. & Patel, V.M. (2022). "A Transformer-Based Siamese Network for Change Detection." *IEEE IGARSS.*

---

## Summary

| # | Item | Tier | Complexity | Impact |
|---|------|------|-----------|--------|
| 1 | ~~Radiometric Normalization~~ DONE | 1 | Medium | High — reduces false positives from illumination |
| 2 | ~~SCL Cloud/Shadow Masking~~ DONE | 1 | Medium | High — eliminates cloud-driven false positives |
| 3 | ~~Multi-Index Change Classification~~ DONE | 1 | Medium | High — semantic "why" behind changes |
| 4 | ~~Folium Blank Map Fix~~ OBE | 2 | Low | Medium — first-impression polish |
| 5 | ~~Configurable S3 Env Vars~~ DONE | 2 | Low | Low — operational flexibility |
| 6 | ~~Time-Series Mode~~ DONE | 3 | Med-High | High — temporal analysis capability |
| 7 | Anomaly Detection Layer | 3 | Medium | Medium — multi-index compound insight |
| 8 | Foundation Model PoC | 3 | High | High — demonstrates ML awareness |

**Recommended order:** ~~1~~ → ~~2~~ → ~~3~~ → ~~4~~ → ~~5~~ → ~~6~~ → 7 → 8

Radiometric normalization and cloud masking improve scientific rigor of existing output. Multi-index classification adds a new analytical dimension with moderate effort. Time-series and ML work are longer-term investments that would move the project score from 7.5 toward 9+.
