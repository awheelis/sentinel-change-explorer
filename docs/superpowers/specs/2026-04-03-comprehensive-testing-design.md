# Comprehensive Testing Design — Sentinel Change Explorer

## Goal

Audit the existing test suite and fill all gaps so that passing tests guarantee the app is **functional** (every public function covered, full pipeline validated, error paths handled) and **fast** (explicit timing thresholds at every layer — computation, network, and cache).

## Constraints

- Non-network tests (`tests/unit/` + `tests/perf/`) complete in under 5 seconds total.
- Mocked unit tests for CI on every commit; real network tests with timing for local/nightly runs.
- Performance guardrails on both computation (ms thresholds on synthetic data) and network (second-level thresholds on real S3/STAC calls).
- Existing passing tests are trusted and relocated, not rewritten.

---

## Architecture: Three-Tier Test Organization

```
tests/
├── conftest.py              # Shared fixtures (synthetic bands, bboxes, mock scenes)
├── unit/                    # All mocked, target < 5s total
│   ├── test_indices.py      # Move existing 12 tests
│   ├── test_visualization.py # Move existing 9 + add 6 new (folium map)
│   ├── test_sentinel.py     # NEW — 7 tests, mocked STAC client + cache logic
│   ├── test_overture.py     # NEW — 6 tests, mocked fetch + cache paths
│   └── test_app_logic.py    # NEW — 11 tests, bbox/memory guard/index dispatch/warmup
├── integration/             # Real network, marked @pytest.mark.network
│   ├── test_sentinel.py     # Move existing 4 + add timing assertions
│   ├── test_overture.py     # Move existing 2 + add 1 new + timing
│   ├── test_presets_e2e.py  # Move existing + add 1 new (compute-phase timing)
│   └── test_reprojection.py # Move existing 3 tests
└── perf/                    # Computation benchmarks, marked @pytest.mark.perf
    └── test_benchmarks.py   # NEW — 9 benchmarks with ms thresholds
```

### Running tiers

- `pytest tests/unit/` — CI, every commit, < 5s.
- `pytest tests/perf/` — CI, every commit, < 10s (synthetic data, no network).
- `pytest -m network tests/integration/` — local/nightly, with timing guardrails.

### Pytest markers

- `@pytest.mark.network` — requires internet access to AWS S3 / STAC / Overture.
- `@pytest.mark.perf` — performance benchmarks with timing assertions.

Configure in `pyproject.toml` or `conftest.py` so unmarked tests default to the unit tier.

---

## Shared Fixtures (`tests/conftest.py`)

- **`small_bands()`** — dict of 5 synthetic uint16 arrays (64x64) with realistic Sentinel-2 reflectance ranges (100-3000). Keys: red, green, blue, nir, swir16.
- **`large_bands()`** — same but 2000x2000. Scope: session (allocated once).
- **`sample_bbox()`** — `(-115.20, 36.10, -115.15, 36.15)` (Las Vegas).
- **`mock_scene()`** — fake scene dict with plausible id, datetime, cloud_cover, and assets pointing to dummy URLs.
- **`mock_overture_context()`** — dict with small GeoDataFrames (5 buildings, 3 segments, 2 places) with valid Shapely geometries.
- **`sample_presets()`** — loads and returns `config/presets.json`.

No network calls in any fixture. `large_bands` is `@pytest.fixture(scope="session")`.

---

## Tier 1: Unit Tests (`tests/unit/`)

### `test_indices.py` — move existing, no changes

12 existing tests cover all index functions, edge cases (division by zero, clipping, uint16 input), chunked computation, and change delta. Move as-is.

### `test_visualization.py` — move existing + 6 new

**Existing (9 tests):** true_color_image (5), downscale_array (3), index_to_rgba (1). Move as-is.

**New tests:**

| Test | What it validates |
|---|---|
| `test_build_folium_map_returns_map` | Pass synthetic images + bbox, returns `folium.Map` |
| `test_build_folium_map_layer_count` | before + after + heatmap + overture = expected child count |
| `test_build_folium_map_no_overture` | `show_overture=False` omits vector layers |
| `test_build_folium_map_none_images` | All images None, still returns valid Map |
| `test_build_folium_map_overture_sampling` | 10k buildings input, verify caps at 5000 |
| `test_image_to_bounds_overlay_encoding` | Returns `ImageOverlay`, bounds match bbox |

### `test_sentinel.py` — all new, fully mocked

| Test | What it validates |
|---|---|
| `test_search_scenes_calls_stac_client` | Mock `pystac_client.Client.open`, verify search receives correct bbox/datetime/query |
| `test_search_scenes_returns_sorted_by_cloud` | 3 mock items with varying cloud cover, assert output order |
| `test_search_scenes_empty_results` | Mock returns no items, assert empty list |
| `test_load_bands_cache_hit` | Fake `.npz` on disk, assert no rasterio calls |
| `test_load_bands_cache_miss_writes_file` | Mock rasterio reads, assert `.npz` created |
| `test_load_bands_missing_band_key` | Scene assets missing a requested band, assert `KeyError` |
| `test_load_bands_invalid_target_res` | `target_res=0`, assert `ValueError` |

### `test_overture.py` — all new, fully mocked

| Test | What it validates |
|---|---|
| `test_fetch_layer_cache_hit` | Write parquet to tmp, assert no `overturemaps.core` import |
| `test_fetch_layer_cache_miss_writes_parquet` | Mock `core.geodataframe`, assert parquet created |
| `test_fetch_layer_corrupt_cache_refetches` | Garbage in cache file, assert falls through to fetch |
| `test_fetch_layer_network_failure_returns_empty` | Mock raises, assert empty GeoDataFrame |
| `test_get_overture_context_returns_all_three_keys` | Mock `fetch_overture_layer`, assert building/segment/place keys |
| `test_cache_path_deterministic` | Same args = same path, different args = different path |

### `test_app_logic.py` — all new

| Test | What it validates |
|---|---|
| `test_compute_index_for_bands_ndvi` | Pass bands dict, assert calls `compute_ndvi` with correct arrays |
| `test_compute_index_for_bands_ndbi` | Same for NDBI |
| `test_compute_index_for_bands_mndwi` | Same for MNDWI |
| `test_compute_index_for_bands_invalid_key` | Unknown index key, assert `KeyError` |
| `test_bbox_west_gte_east_rejected` | Validate the west >= east check logic |
| `test_bbox_south_gte_north_rejected` | Validate the south >= north check logic |
| `test_memory_guard_small_bbox_passes` | Small bbox estimated MB < 500 |
| `test_memory_guard_huge_bbox_exceeds` | Large bbox estimated MB > 500 |
| `test_warm_preset_caches_calls_all_presets` | (from existing test_warmup.py) |
| `test_warm_preset_caches_survives_failures` | (from existing test_warmup.py) |
| `test_warm_preset_caches_progress_callback` | (from existing test_warmup.py) |

---

## Tier 2: Integration Tests (`tests/integration/`)

All marked `@pytest.mark.network`. Timing enforced via `time.perf_counter` + assert in a small `assert_within` helper in conftest.

### `test_sentinel.py` — move existing 4 + timing

| Test | Timing budget |
|---|---|
| `test_search_scenes_returns_results` | < 10s |
| `test_load_bands_returns_numpy_arrays` | < 30s |
| `test_load_bands_returns_wgs84_aligned` | < 30s |
| `test_disk_cache_creates_and_reuses` | Second call < 1s |

### `test_overture.py` — move existing 2 + 1 new + timing

| Test | Timing budget |
|---|---|
| `test_fetch_buildings_returns_geodataframe` | < 15s |
| `test_get_overture_context_returns_all_layers` | < 30s |
| `test_overture_cache_hit_fast` (new) | Second call < 1s |

### `test_presets_e2e.py` — move existing + 1 new

| Test | Timing budget |
|---|---|
| `test_preset_full_pipeline` (parametrized, all presets) | < 90s per preset |
| `test_preset_computation_phase_fast` (new) | < 5s per preset (cached bands only) |

### `test_reprojection.py` — move existing 3

- `test_no_nodata_border_las_vegas`
- `test_before_after_shapes_match_las_vegas`
- `test_before_after_shapes_match_all_presets`

No timing needed — correctness-only tests.

---

## Tier 3: Performance Benchmarks (`tests/perf/`)

All use synthetic data from `large_bands` session fixture. No network. Marked `@pytest.mark.perf`.

### `test_benchmarks.py`

| Benchmark | Array size | Threshold |
|---|---|---|
| `test_ndvi_2000x2000_under_100ms` | 2000x2000 | < 100ms |
| `test_ndbi_2000x2000_under_100ms` | 2000x2000 | < 100ms |
| `test_mndwi_2000x2000_under_100ms` | 2000x2000 | < 100ms |
| `test_compute_change_2000x2000_under_50ms` | 2000x2000 | < 50ms |
| `test_chunked_ndvi_not_slower_than_2x` | 2000x2000 | < 2x single-pass time |
| `test_true_color_2000x2000_under_200ms` | 2000x2000 | < 200ms |
| `test_index_to_rgba_2000x2000_under_200ms` | 2000x2000 | < 200ms |
| `test_downscale_2000x2000_to_800_under_50ms` | 2000x2000 | < 50ms |
| `test_full_compute_pipeline_under_500ms` | 2000x2000 | < 500ms |

The `test_full_compute_pipeline_under_500ms` benchmark runs the complete compute path: 3 indices, 3 change deltas, 2 true-color images, 1 heatmap. This is the key "app feels fast" guarantee.

Thresholds are generous — failure means a genuine regression (wrong dtype, accidental copy, algorithmic change), not normal machine variance.

---

## What Passing Tests Guarantee

- **Functional**: Every public function in every module has unit coverage. Full pipeline (search -> load -> reproject -> compute -> render -> map) validated per-preset with real data. Cache read/write/corruption paths tested. Error paths (bad bbox, missing bands, network failure) return sensible results.
- **Fast (computation)**: Index math, rendering, and downscaling on 2000x2000 arrays stay under explicit ms thresholds. Full compute pipeline < 500ms.
- **Fast (network)**: STAC search < 10s, band loading < 30s, Overture < 30s, cache hits < 1s, full preset pipeline < 90s.
- **Fast (tests themselves)**: `pytest tests/unit/ tests/perf/` < 15s with zero network calls.

---

## Migration Plan

1. Create `tests/unit/`, `tests/integration/`, `tests/perf/` directories with `__init__.py`.
2. Move existing test files into their new homes (file-by-file `git mv`).
3. Update imports if any tests reference sibling test files.
4. Add shared fixtures to `tests/conftest.py`.
5. Write new tests per the tables above.
6. Add pytest markers to `conftest.py` or `pyproject.toml`.
7. Verify `pytest tests/unit/` passes < 5s, `pytest tests/perf/` passes < 10s.
8. Verify `pytest -m network` passes with timing assertions.
9. Remove old top-level test files once migration is confirmed.
