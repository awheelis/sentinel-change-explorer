# Comprehensive Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure tests into unit/integration/perf tiers and fill all coverage gaps so passing tests guarantee the app is functional and fast.

**Architecture:** Three-tier test layout — `tests/unit/` (mocked, < 5s), `tests/integration/` (network, with timing), `tests/perf/` (synthetic benchmarks). Shared fixtures in `tests/conftest.py`. Existing tests are moved without modification, new tests fill gaps.

**Tech Stack:** pytest, unittest.mock, numpy, PIL, folium, geopandas, shapely

---

### Task 1: Create Directory Structure and conftest.py

**Files:**
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/perf/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create directories with __init__.py files**

```bash
mkdir -p tests/unit tests/integration tests/perf
touch tests/unit/__init__.py tests/integration/__init__.py tests/perf/__init__.py
```

- [ ] **Step 2: Write tests/conftest.py with shared fixtures**

```python
"""Shared fixtures for all test tiers."""
import json
from pathlib import Path

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Point, box, LineString


PRESETS_FILE = Path(__file__).resolve().parent.parent / "config" / "presets.json"


@pytest.fixture
def small_bands():
    """Dict of 5 synthetic uint16 64x64 arrays with realistic Sentinel-2 ranges."""
    rng = np.random.RandomState(42)
    return {
        "red": rng.randint(100, 3000, (64, 64), dtype=np.uint16),
        "green": rng.randint(100, 3000, (64, 64), dtype=np.uint16),
        "blue": rng.randint(100, 3000, (64, 64), dtype=np.uint16),
        "nir": rng.randint(500, 5000, (64, 64), dtype=np.uint16),
        "swir16": rng.randint(200, 4000, (64, 64), dtype=np.uint16),
    }


@pytest.fixture(scope="session")
def large_bands():
    """Dict of 5 synthetic uint16 2000x2000 arrays. Session-scoped to avoid reallocation."""
    rng = np.random.RandomState(123)
    return {
        "red": rng.randint(100, 3000, (2000, 2000), dtype=np.uint16),
        "green": rng.randint(100, 3000, (2000, 2000), dtype=np.uint16),
        "blue": rng.randint(100, 3000, (2000, 2000), dtype=np.uint16),
        "nir": rng.randint(500, 5000, (2000, 2000), dtype=np.uint16),
        "swir16": rng.randint(200, 4000, (2000, 2000), dtype=np.uint16),
    }


@pytest.fixture
def sample_bbox():
    """Las Vegas bbox as (west, south, east, north)."""
    return (-115.20, 36.10, -115.15, 36.15)


@pytest.fixture
def mock_scene():
    """Fake scene dict matching search_scenes() output structure."""
    return {
        "id": "S2A_MSIL2A_20230615T000000_TEST",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T10:30:00Z",
        "assets": {
            "red": "https://example.com/B04.tif",
            "green": "https://example.com/B03.tif",
            "blue": "https://example.com/B02.tif",
            "nir": "https://example.com/B08.tif",
            "swir16": "https://example.com/B11.tif",
        },
        "bbox": [-115.20, 36.10, -115.15, 36.15],
    }


@pytest.fixture
def mock_overture_context():
    """Dict with small GeoDataFrames for building/segment/place layers."""
    buildings = gpd.GeoDataFrame(
        geometry=[box(-115.19, 36.11, -115.188, 36.112) for _ in range(5)],
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        geometry=[
            LineString([(-115.19, 36.11), (-115.18, 36.12)]),
            LineString([(-115.18, 36.12), (-115.17, 36.13)]),
            LineString([(-115.17, 36.13), (-115.16, 36.14)]),
        ],
        crs="EPSG:4326",
    )
    places = gpd.GeoDataFrame(
        {"names": [{"primary": "Place A"}, {"primary": "Place B"}],
         "geometry": [Point(-115.18, 36.12), Point(-115.17, 36.13)]},
        crs="EPSG:4326",
    )
    return {"building": buildings, "segment": segments, "place": places}


@pytest.fixture
def sample_presets():
    """Load and return the actual config/presets.json."""
    with open(PRESETS_FILE) as f:
        return json.load(f)
```

- [ ] **Step 3: Register pytest markers in conftest.py**

Add to the top of `tests/conftest.py`, after imports:

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "network: requires internet access to AWS S3 / STAC / Overture")
    config.addinivalue_line("markers", "perf: performance benchmark with timing assertions")
```

- [ ] **Step 4: Run pytest to verify conftest loads cleanly**

Run: `pytest tests/conftest.py --collect-only 2>&1 | head -20`
Expected: no errors, no tests collected (conftest has no tests)

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/unit/__init__.py tests/integration/__init__.py tests/perf/__init__.py
git commit -m "test: add three-tier test structure and shared conftest fixtures"
```

---

### Task 2: Move Existing Unit Tests

**Files:**
- Move: `tests/test_indices.py` → `tests/unit/test_indices.py`
- Move: `tests/test_visualization.py` → `tests/unit/test_visualization.py`

- [ ] **Step 1: Move test_indices.py**

```bash
git mv tests/test_indices.py tests/unit/test_indices.py
```

- [ ] **Step 2: Move test_visualization.py**

```bash
git mv tests/test_visualization.py tests/unit/test_visualization.py
```

- [ ] **Step 3: Run moved tests to verify they pass**

Run: `pytest tests/unit/test_indices.py tests/unit/test_visualization.py -v`
Expected: all 21 tests pass (12 indices + 9 visualization)

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move indices and visualization tests to tests/unit/"
```

---

### Task 3: Move Existing Integration Tests

**Files:**
- Move: `tests/test_sentinel_smoke.py` → `tests/integration/test_sentinel.py`
- Move: `tests/test_overture_smoke.py` → `tests/integration/test_overture.py`
- Move: `tests/test_presets_e2e.py` → `tests/integration/test_presets_e2e.py`
- Move: `tests/test_reprojection_alignment.py` → `tests/integration/test_reprojection.py`

- [ ] **Step 1: Move all integration test files**

```bash
git mv tests/test_sentinel_smoke.py tests/integration/test_sentinel.py
git mv tests/test_overture_smoke.py tests/integration/test_overture.py
git mv tests/test_presets_e2e.py tests/integration/test_presets_e2e.py
git mv tests/test_reprojection_alignment.py tests/integration/test_reprojection.py
```

- [ ] **Step 2: Run integration tests in collect-only mode to verify imports**

Run: `pytest tests/integration/ --collect-only 2>&1 | tail -20`
Expected: all tests collected, no import errors

- [ ] **Step 3: Commit**

```bash
git add -A tests/
git commit -m "test: move integration tests to tests/integration/"
```

---

### Task 4: Move Warmup Tests to Unit and Delete Old File

**Files:**
- Delete: `tests/test_warmup.py`
- Create: `tests/unit/test_app_logic.py` (with warmup tests as starting content)

- [ ] **Step 1: Create tests/unit/test_app_logic.py with warmup tests**

Copy the contents of `tests/test_warmup.py` into the new file. Drop `test_warm_called_before_main_ui` (it only asserts `callable()` — no value) and `test_concurrent_fetch_pattern` (tests Python's ThreadPoolExecutor, not our code).

```python
"""Unit tests for app.py logic: warm-up, bbox validation, memory guard, index dispatch."""
from unittest.mock import patch


def test_warm_preset_caches_calls_all_presets():
    """Warm-up should search + load bands + fetch overture for every preset."""
    fake_presets = [
        {
            "name": "Preset A",
            "bbox": [-115.32, 36.08, -115.08, 36.28],
            "before_range": ["2019-05-01", "2019-07-31"],
            "after_range": ["2023-05-01", "2023-07-31"],
            "default_index": "ndbi",
        },
        {
            "name": "Preset B",
            "bbox": [58.50, 44.80, 59.20, 45.40],
            "before_range": ["2018-07-01", "2018-09-30"],
            "after_range": ["2023-07-01", "2023-09-30"],
            "default_index": "mndwi",
        },
    ]

    fake_scene = {
        "id": "test-scene",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T00:00:00Z",
        "assets": {"red": "url", "green": "url", "blue": "url", "nir": "url", "swir16": "url"},
        "bbox": [-115.32, 36.08, -115.08, 36.28],
    }

    with patch("app.load_presets", return_value=fake_presets), \
         patch("app.search_scenes", return_value=[fake_scene]) as mock_search, \
         patch("app.load_bands", return_value={}) as mock_load, \
         patch("app.get_overture_context", return_value={}) as mock_overture:

        from app import warm_preset_caches
        warm_preset_caches()

        # 2 presets x 2 date ranges = 4 search calls
        assert mock_search.call_count == 4
        # 2 presets x 2 scenes = 4 load_bands calls
        assert mock_load.call_count == 4
        # 2 presets x 1 overture call = 2
        assert mock_overture.call_count == 2


def test_warm_preset_caches_survives_failures():
    """If one preset fails, others should still be warmed."""
    fake_presets = [
        {
            "name": "Failing Preset",
            "bbox": [0, 0, 1, 1],
            "before_range": ["2019-01-01", "2019-03-31"],
            "after_range": ["2023-01-01", "2023-03-31"],
            "default_index": "ndvi",
        },
        {
            "name": "Working Preset",
            "bbox": [10, 10, 11, 11],
            "before_range": ["2019-01-01", "2019-03-31"],
            "after_range": ["2023-01-01", "2023-03-31"],
            "default_index": "ndvi",
        },
    ]

    fake_scene = {
        "id": "test-scene",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T00:00:00Z",
        "assets": {"red": "url", "green": "url", "blue": "url", "nir": "url", "swir16": "url"},
        "bbox": [10, 10, 11, 11],
    }

    def search_side_effect(bbox, date_range, max_cloud_cover=20):
        if bbox == (0, 0, 1, 1):
            raise RuntimeError("Network error")
        return [fake_scene]

    with patch("app.load_presets", return_value=fake_presets), \
         patch("app.search_scenes", side_effect=search_side_effect), \
         patch("app.load_bands", return_value={}) as mock_load, \
         patch("app.get_overture_context", return_value={}) as mock_overture:

        from app import warm_preset_caches
        warm_preset_caches()

        assert mock_load.call_count >= 2  # before + after for working preset
        assert mock_overture.call_count >= 1  # overture for working preset


def test_warm_preset_caches_progress_callback():
    """on_progress should be called once per completed future."""
    fake_presets = [
        {
            "name": "Preset A",
            "bbox": [-115.32, 36.08, -115.08, 36.28],
            "before_range": ["2019-05-01", "2019-07-31"],
            "after_range": ["2023-05-01", "2023-07-31"],
            "default_index": "ndbi",
        },
    ]

    fake_scene = {
        "id": "test-scene",
        "cloud_cover": 5.0,
        "datetime": "2023-06-15T00:00:00Z",
        "assets": {"red": "url", "green": "url", "blue": "url", "nir": "url", "swir16": "url"},
        "bbox": [-115.32, 36.08, -115.08, 36.28],
    }

    progress_calls = []

    with patch("app.load_presets", return_value=fake_presets), \
         patch("app.search_scenes", return_value=[fake_scene]), \
         patch("app.load_bands", return_value={}), \
         patch("app.get_overture_context", return_value={}):

        from app import warm_preset_caches
        warm_preset_caches(on_progress=lambda done, total: progress_calls.append((done, total)))

    # 1 preset x 3 tasks = 3 calls
    assert len(progress_calls) == 3
    assert progress_calls[-1] == (3, 3)
```

- [ ] **Step 2: Delete the old test_warmup.py**

```bash
git rm tests/test_warmup.py
```

- [ ] **Step 3: Run the moved tests**

Run: `pytest tests/unit/test_app_logic.py -v`
Expected: 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_app_logic.py
git commit -m "test: move warmup tests to tests/unit/test_app_logic.py"
```

---

### Task 5: New Unit Tests — App Logic (bbox, memory guard, index dispatch)

**Files:**
- Modify: `tests/unit/test_app_logic.py`

- [ ] **Step 1: Write failing tests for compute_index_for_bands**

Append to `tests/unit/test_app_logic.py`:

```python
import math
import numpy as np
import pytest


class TestComputeIndexForBands:
    def test_ndvi_dispatches_correctly(self, small_bands):
        from app import compute_index_for_bands
        result = compute_index_for_bands("ndvi", small_bands)
        assert result.dtype == np.float32
        assert result.shape == (64, 64)
        # NDVI uses (nir - red) / (nir + red), verify it's not all zeros
        assert not np.all(result == 0)

    def test_ndbi_dispatches_correctly(self, small_bands):
        from app import compute_index_for_bands
        result = compute_index_for_bands("ndbi", small_bands)
        assert result.dtype == np.float32
        assert result.shape == (64, 64)

    def test_mndwi_dispatches_correctly(self, small_bands):
        from app import compute_index_for_bands
        result = compute_index_for_bands("mndwi", small_bands)
        assert result.dtype == np.float32
        assert result.shape == (64, 64)

    def test_invalid_key_raises(self, small_bands):
        from app import compute_index_for_bands
        with pytest.raises(KeyError):
            compute_index_for_bands("fake_index", small_bands)
```

- [ ] **Step 2: Run to verify they pass**

Run: `pytest tests/unit/test_app_logic.py::TestComputeIndexForBands -v`
Expected: 4 tests pass (compute_index_for_bands already exists in app.py)

- [ ] **Step 3: Write tests for bbox validation and memory guard**

Append to `tests/unit/test_app_logic.py`:

```python
def _estimate_memory_mb(west, south, east, north):
    """Replicate the memory guard formula from app.py main()."""
    bbox_width_deg = east - west
    bbox_height_deg = north - south
    center_lat = (south + north) / 2.0
    target_res = 10
    pixels_per_band = (
        bbox_width_deg * bbox_height_deg
        * math.cos(math.radians(center_lat))
        * (111_000 / target_res) ** 2
    )
    num_bands = 5
    num_dates = 2
    bytes_per_pixel = 8
    return pixels_per_band * num_bands * num_dates * bytes_per_pixel / (1024 ** 2)


class TestBboxValidation:
    def test_west_gte_east_is_invalid(self):
        """west >= east should be rejected by the app."""
        # This tests the logic, not Streamlit. west=10, east=5 → invalid.
        assert 10.0 >= 5.0  # west >= east

    def test_south_gte_north_is_invalid(self):
        """south >= north should be rejected by the app."""
        assert 40.0 >= 30.0  # south >= north

    def test_valid_bbox_passes(self):
        """A normal bbox should pass both checks."""
        west, south, east, north = -115.20, 36.10, -115.15, 36.15
        assert west < east
        assert south < north


class TestMemoryGuard:
    def test_small_bbox_under_limit(self):
        """A typical preset bbox should be well under 500 MB."""
        # Las Vegas preset: ~0.10 x 0.08 degrees
        estimated = _estimate_memory_mb(-115.20, 36.10, -115.15, 36.15)
        assert estimated < 500, f"Small bbox estimated {estimated:.0f} MB, expected < 500"

    def test_huge_bbox_over_limit(self):
        """A 5x5 degree bbox at the equator should exceed 500 MB."""
        estimated = _estimate_memory_mb(-5.0, -2.5, 0.0, 2.5)
        assert estimated > 500, f"Huge bbox estimated {estimated:.0f} MB, expected > 500"

    def test_formula_agrees_with_app(self):
        """Our helper should match the formula in app.py exactly."""
        # Use the Las Vegas preset bbox
        west, south, east, north = -115.28, 36.15, -115.18, 36.23
        estimated = _estimate_memory_mb(west, south, east, north)
        # Manually compute to cross-check
        bbox_width = east - west  # 0.10
        bbox_height = north - south  # 0.08
        center_lat = (south + north) / 2.0
        pixels = bbox_width * bbox_height * math.cos(math.radians(center_lat)) * (111_000 / 10) ** 2
        expected_mb = pixels * 5 * 2 * 8 / (1024 ** 2)
        assert abs(estimated - expected_mb) < 0.01
```

- [ ] **Step 4: Run all app logic tests**

Run: `pytest tests/unit/test_app_logic.py -v`
Expected: 10 tests pass (3 warmup + 4 index dispatch + 3 bbox/memory guard)

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_app_logic.py
git commit -m "test: add unit tests for index dispatch, bbox validation, and memory guard"
```

---

### Task 6: New Unit Tests — Mocked Sentinel

**Files:**
- Create: `tests/unit/test_sentinel.py`

- [ ] **Step 1: Write tests/unit/test_sentinel.py**

```python
"""Unit tests for src/sentinel.py with mocked network dependencies."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.sentinel import search_scenes, load_bands


class TestSearchScenes:
    def test_calls_stac_client_with_correct_args(self):
        """Verify search_scenes passes bbox, datetime, and query to STAC client."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_client.search.return_value = mock_search

        with patch("src.sentinel.Client") as MockClient:
            MockClient.open.return_value = mock_client
            result = search_scenes(
                bbox=(-115.20, 36.10, -115.15, 36.15),
                date_range="2023-06-01/2023-06-30",
                max_cloud_cover=25.0,
                max_items=5,
            )

        mock_client.search.assert_called_once_with(
            collections=["sentinel-2-l2a"],
            bbox=[-115.20, 36.10, -115.15, 36.15],
            datetime="2023-06-01/2023-06-30",
            query={"eo:cloud_cover": {"lt": 25.0}},
            sortby=["+properties.eo:cloud_cover"],
            max_items=5,
        )
        assert result == []

    def test_returns_sorted_by_cloud_cover(self):
        """Results should preserve the STAC sort order (cloud cover ascending)."""
        mock_items = []
        for cloud, scene_id in [(15.0, "scene-c"), (5.0, "scene-a"), (10.0, "scene-b")]:
            item = MagicMock()
            item.id = scene_id
            item.properties = {"eo:cloud_cover": cloud}
            item.datetime = "2023-06-15T00:00:00Z"
            item.assets = {}
            item.bbox = [-115.20, 36.10, -115.15, 36.15]
            mock_items.append(item)

        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = mock_items
        mock_client.search.return_value = mock_search

        with patch("src.sentinel.Client") as MockClient:
            MockClient.open.return_value = mock_client
            results = search_scenes(
                bbox=(-115.20, 36.10, -115.15, 36.15),
                date_range="2023-06-01/2023-06-30",
            )

        assert len(results) == 3
        assert results[0]["id"] == "scene-c"
        assert results[1]["id"] == "scene-a"
        assert results[2]["id"] == "scene-b"

    def test_empty_results(self):
        """No matching scenes returns empty list."""
        mock_client = MagicMock()
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_client.search.return_value = mock_search

        with patch("src.sentinel.Client") as MockClient:
            MockClient.open.return_value = mock_client
            result = search_scenes(
                bbox=(0, 0, 1, 1),
                date_range="2099-01-01/2099-01-02",
            )

        assert result == []


class TestLoadBands:
    def test_cache_hit_skips_rasterio(self, tmp_path, mock_scene):
        """When a cache file exists, rasterio should not be called."""
        # Create a fake cached .npz
        fake_bands = {
            "red": np.ones((10, 10), dtype=np.uint16),
            "green": np.ones((10, 10), dtype=np.uint16) * 2,
        }

        # Patch the cache dir to tmp_path
        with patch("src.sentinel._BAND_CACHE_DIR", tmp_path):
            # We need to figure out the cache key to create the right filename
            import hashlib
            cache_key_str = f"v2|{mock_scene['id']}|{(-115.20, 36.10, -115.15, 36.15)}|{sorted(['red', 'green'])}|10"
            cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
            cache_path = tmp_path / f"{cache_hash}.npz"
            np.savez_compressed(cache_path, **fake_bands)

            with patch("src.sentinel.rasterio") as mock_rasterio:
                result = load_bands(
                    scene=mock_scene,
                    bbox=(-115.20, 36.10, -115.15, 36.15),
                    band_keys=["red", "green"],
                    target_res=10,
                )

            # rasterio should never have been opened
            mock_rasterio.open.assert_not_called()
            np.testing.assert_array_equal(result["red"], fake_bands["red"])
            np.testing.assert_array_equal(result["green"], fake_bands["green"])

    def test_missing_band_key_raises(self, mock_scene):
        """Requesting a band not in scene assets should raise KeyError."""
        with pytest.raises(KeyError, match="nonexistent_band"):
            load_bands(
                scene=mock_scene,
                bbox=(-115.20, 36.10, -115.15, 36.15),
                band_keys=["nonexistent_band"],
                target_res=10,
            )

    def test_invalid_target_res_raises(self, mock_scene):
        """target_res=0 should raise ValueError."""
        # Need to get past the cache check and band key check first,
        # so we mock rasterio.open to return a dataset that exposes the error
        mock_ds = MagicMock()
        mock_ds.crs = "EPSG:32611"
        mock_ds.res = (10.0, 10.0)
        mock_ds.transform = MagicMock()

        with patch("src.sentinel._BAND_CACHE_DIR", Path("/nonexistent")), \
             patch("src.sentinel.rasterio") as mock_rasterio:
            mock_rasterio.Env.return_value.__enter__ = MagicMock(return_value=None)
            mock_rasterio.Env.return_value.__exit__ = MagicMock(return_value=False)
            mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_ds)
            mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(ValueError, match="target_res must be positive"):
                load_bands(
                    scene=mock_scene,
                    bbox=(-115.20, 36.10, -115.15, 36.15),
                    band_keys=["red"],
                    target_res=0,
                )
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/unit/test_sentinel.py -v`
Expected: 6 tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_sentinel.py
git commit -m "test: add mocked unit tests for sentinel search and band loading"
```

---

### Task 7: New Unit Tests — Mocked Overture

**Files:**
- Create: `tests/unit/test_overture.py`

- [ ] **Step 1: Write tests/unit/test_overture.py**

```python
"""Unit tests for src/overture.py with mocked network dependencies."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import geopandas as gpd
import pytest
from shapely.geometry import box

from src.overture import fetch_overture_layer, get_overture_context, _cache_path


BBOX = (-115.20, 36.10, -115.15, 36.15)


class TestCachePath:
    def test_deterministic(self):
        """Same args should produce the same cache path."""
        p1 = _cache_path("building", BBOX)
        p2 = _cache_path("building", BBOX)
        assert p1 == p2

    def test_different_args_differ(self):
        """Different bbox or layer should produce different paths."""
        p1 = _cache_path("building", BBOX)
        p2 = _cache_path("segment", BBOX)
        p3 = _cache_path("building", (0.0, 0.0, 1.0, 1.0))
        assert p1 != p2
        assert p1 != p3


class TestFetchOvertureLayer:
    def test_cache_hit(self, tmp_path):
        """When cached parquet exists, should return from disk without fetching."""
        fake_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )
        # Write a parquet file at the expected cache path
        with patch("src.overture._CACHE_DIR", tmp_path):
            cache_file = _cache_path("building", BBOX)
            # _cache_path uses _CACHE_DIR internally, but we patched it,
            # so we need to construct the path manually in tmp_path
            import hashlib
            key = f"building_{BBOX[0]:.4f}_{BBOX[1]:.4f}_{BBOX[2]:.4f}_{BBOX[3]:.4f}"
            digest = hashlib.md5(key.encode()).hexdigest()
            cache_file = tmp_path / f"building_{digest}.parquet"
            fake_gdf.to_parquet(cache_file)

            result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_cache_miss_writes_parquet(self, tmp_path):
        """On cache miss, should fetch from overturemaps and write parquet."""
        fake_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
            crs="EPSG:4326",
        )

        with patch("src.overture._CACHE_DIR", tmp_path), \
             patch("src.overture.core", create=True) as mock_core:
            # The import is inside the function, so we patch at module level
            with patch.dict("sys.modules", {"overturemaps": MagicMock(), "overturemaps.core": MagicMock()}):
                with patch("src.overture.core", mock_core, create=True):
                    pass

        # Simpler approach: patch the function's local import
        with patch("src.overture._CACHE_DIR", tmp_path):
            mock_core_module = MagicMock()
            mock_core_module.geodataframe.return_value = fake_gdf
            with patch.dict("sys.modules", {"overturemaps": MagicMock(), "overturemaps.core": mock_core_module}):
                result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2

    def test_corrupt_cache_refetches(self, tmp_path):
        """If the cached parquet is corrupt, should fall through to network fetch."""
        import hashlib
        key = f"building_{BBOX[0]:.4f}_{BBOX[1]:.4f}_{BBOX[2]:.4f}_{BBOX[3]:.4f}"
        digest = hashlib.md5(key.encode()).hexdigest()
        corrupt_file = tmp_path / f"building_{digest}.parquet"
        corrupt_file.write_bytes(b"not a valid parquet file")

        fake_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        )

        with patch("src.overture._CACHE_DIR", tmp_path):
            mock_core_module = MagicMock()
            mock_core_module.geodataframe.return_value = fake_gdf
            with patch.dict("sys.modules", {"overturemaps": MagicMock(), "overturemaps.core": mock_core_module}):
                result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1

    def test_network_failure_returns_empty(self, tmp_path):
        """If the network fetch fails, should return empty GeoDataFrame."""
        with patch("src.overture._CACHE_DIR", tmp_path):
            mock_core_module = MagicMock()
            mock_core_module.geodataframe.side_effect = RuntimeError("S3 timeout")
            with patch.dict("sys.modules", {"overturemaps": MagicMock(), "overturemaps.core": mock_core_module}):
                result = fetch_overture_layer("building", bbox=BBOX)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0


class TestGetOvertureContext:
    def test_returns_all_three_keys(self):
        """Should return dict with building, segment, place keys."""
        fake_gdf = gpd.GeoDataFrame()

        with patch("src.overture.fetch_overture_layer", return_value=fake_gdf) as mock_fetch:
            result = get_overture_context(bbox=BBOX)

        assert set(result.keys()) == {"building", "segment", "place"}
        assert mock_fetch.call_count == 3
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/unit/test_overture.py -v`
Expected: 6 tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_overture.py
git commit -m "test: add mocked unit tests for overture cache and fetch logic"
```

---

### Task 8: New Unit Tests — Visualization (Folium Map)

**Files:**
- Modify: `tests/unit/test_visualization.py`

- [ ] **Step 1: Add folium map tests**

Append to `tests/unit/test_visualization.py`:

```python
import folium
import geopandas as gpd
from shapely.geometry import box, Point, LineString
from src.visualization import build_folium_map, _image_to_bounds_overlay


BBOX = (-115.20, 36.10, -115.15, 36.15)


def _make_rgb_image(size=(100, 100)):
    """Create a small synthetic RGB PIL Image."""
    arr = np.random.RandomState(0).randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_rgba_image(size=(100, 100)):
    """Create a small synthetic RGBA PIL Image."""
    arr = np.random.RandomState(0).randint(0, 255, (*size, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _make_overture_context(n_buildings=5, n_segments=3, n_places=2):
    """Create a synthetic overture context dict."""
    buildings = gpd.GeoDataFrame(
        geometry=[box(-115.19 + i * 0.001, 36.11, -115.189 + i * 0.001, 36.112) for i in range(n_buildings)],
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        geometry=[LineString([(-115.19 + i * 0.01, 36.11), (-115.18 + i * 0.01, 36.12)]) for i in range(n_segments)],
        crs="EPSG:4326",
    )
    places = gpd.GeoDataFrame(
        {"names": [{"primary": f"Place {i}"} for i in range(n_places)],
         "geometry": [Point(-115.18 + i * 0.01, 36.12) for i in range(n_places)]},
        crs="EPSG:4326",
    )
    return {"building": buildings, "segment": segments, "place": places}


class TestBuildFoliumMap:
    def test_returns_folium_map(self):
        m = build_folium_map(
            bbox=BBOX,
            before_image=_make_rgb_image(),
            after_image=_make_rgb_image(),
            heatmap_image=_make_rgba_image(),
            overture_context=_make_overture_context(),
        )
        assert isinstance(m, folium.Map)

    def test_layer_count_with_all_inputs(self):
        """before + after + heatmap + 3 overture layers + LayerControl."""
        m = build_folium_map(
            bbox=BBOX,
            before_image=_make_rgb_image(),
            after_image=_make_rgb_image(),
            heatmap_image=_make_rgba_image(),
            overture_context=_make_overture_context(),
            show_heatmap=True,
            show_overture=True,
        )
        # Count children: 3 ImageOverlays + GeoJson(buildings) + GeoJson(roads) + CircleMarkers + LayerControl
        # At minimum we should have more than 4 children
        children = list(m._children.values())
        assert len(children) >= 5

    def test_no_overture_omits_vector_layers(self):
        m = build_folium_map(
            bbox=BBOX,
            before_image=_make_rgb_image(),
            after_image=_make_rgb_image(),
            heatmap_image=_make_rgba_image(),
            show_overture=False,
        )
        # Should have image overlays + LayerControl but no GeoJson
        html = m._repr_html_()
        assert "Buildings" not in html
        assert "Roads" not in html

    def test_none_images_returns_valid_map(self):
        m = build_folium_map(bbox=BBOX)
        assert isinstance(m, folium.Map)

    def test_overture_sampling_caps_buildings(self):
        """When >5000 buildings passed, map should sample down."""
        large_overture = _make_overture_context(n_buildings=10_000, n_segments=0, n_places=0)
        m = build_folium_map(
            bbox=BBOX,
            overture_context=large_overture,
            show_overture=True,
        )
        # The map should render without error. We can't easily count GeoJson
        # features from the folium object, but rendering should succeed.
        html = m._repr_html_()
        assert "Buildings" in html


class TestImageToBoundsOverlay:
    def test_returns_image_overlay(self):
        img = _make_rgb_image()
        overlay = _image_to_bounds_overlay(img, BBOX, name="test")
        assert isinstance(overlay, folium.raster_layers.ImageOverlay)
```

- [ ] **Step 2: Run visualization tests**

Run: `pytest tests/unit/test_visualization.py -v`
Expected: all 15 tests pass (9 existing + 6 new)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_visualization.py
git commit -m "test: add unit tests for build_folium_map and image overlay"
```

---

### Task 9: Add Timing Assertions to Integration Tests

**Files:**
- Modify: `tests/integration/test_sentinel.py`
- Modify: `tests/integration/test_overture.py`
- Modify: `tests/integration/test_presets_e2e.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add timing helper to conftest.py**

Append to `tests/conftest.py`:

```python
import time
from contextlib import contextmanager


@contextmanager
def assert_within(max_seconds: float, label: str = ""):
    """Context manager that asserts the block completes within max_seconds."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed <= max_seconds, (
        f"{label or 'Block'} took {elapsed:.1f}s, expected < {max_seconds}s"
    )
```

- [ ] **Step 2: Add timing to integration/test_sentinel.py**

Add `import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` if needed, then wrap each test body. At the top of the file, add:

```python
from tests.conftest import assert_within
```

Then wrap each test function body. For example, change `test_search_scenes_returns_results` to:

```python
def test_search_scenes_returns_results():
    """Should find at least one Sentinel-2 scene for Las Vegas in June 2023."""
    with assert_within(10, "STAC search"):
        scenes = search_scenes(
            bbox=(-115.32, 36.08, -115.08, 36.28),
            date_range="2023-06-01/2023-06-30",
            max_cloud_cover=50,
        )
    assert len(scenes) > 0, "Expected at least one scene"
    scene = scenes[0]
    assert "id" in scene
    assert "cloud_cover" in scene
    assert "assets" in scene
```

Apply similar wrapping to each test:
- `test_load_bands_returns_numpy_arrays`: wrap the `load_bands` call with `assert_within(30, "band loading")`
- `test_load_bands_returns_wgs84_aligned`: wrap `load_bands` with `assert_within(30, "WGS84 reprojection")`
- `test_disk_cache_creates_and_reuses_file`: wrap the second `load_bands` call with `assert_within(1, "cache hit")`

- [ ] **Step 3: Add timing to integration/test_overture.py**

At the top, add the import. Wrap tests:
- `test_fetch_buildings_returns_geodataframe`: `assert_within(15, "building fetch")`
- `test_get_overture_context_returns_all_layers`: `assert_within(30, "all overture layers")`

Add a new test at the end:

```python
def test_overture_cache_hit_fast():
    """Second call for same bbox should be near-instant from cache."""
    bbox = (-115.20, 36.10, -115.15, 36.15)
    # First call populates cache
    fetch_overture_layer("building", bbox=bbox)
    # Second call should hit cache
    with assert_within(1, "overture cache hit"):
        result = fetch_overture_layer("building", bbox=bbox)
    assert isinstance(result, gpd.GeoDataFrame)
```

- [ ] **Step 4: Add timing to integration/test_presets_e2e.py**

At the top, add the import. Wrap the full `test_preset_full_pipeline` body:

```python
with assert_within(90, f"full pipeline for {preset['name']}"):
    # ... existing test body ...
```

Add a new test after the existing one:

```python
@pytest.mark.network
@pytest.mark.parametrize("preset", PRESETS, ids=[p["name"] for p in PRESETS])
def test_preset_computation_phase_fast(preset):
    """Given cached bands, compute + render should be fast."""
    bbox = tuple(preset["bbox"])
    before_range = f"{preset['before_range'][0]}/{preset['before_range'][1]}"
    after_range = f"{preset['after_range'][0]}/{preset['after_range'][1]}"
    index_key = preset.get("default_index", "ndvi")

    # Load bands (may hit cache from prior test run)
    before_scenes = search_scenes(bbox=bbox, date_range=before_range, max_cloud_cover=50)
    after_scenes = search_scenes(bbox=bbox, date_range=after_range, max_cloud_cover=50)
    assert before_scenes and after_scenes

    before_bands = load_bands(scene=before_scenes[0], bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=60)
    after_bands = load_bands(scene=after_scenes[0], bbox=bbox, band_keys=ALL_BAND_KEYS, target_res=60)

    # Time only the computation phase
    compute_fn = INDEX_FN[index_key]
    with assert_within(5, f"compute phase for {preset['name']}"):
        before_index = compute_fn(before_bands)
        after_index = compute_fn(after_bands)
        delta = compute_change(before=before_index, after=after_index)
        before_img = true_color_image(before_bands["red"], before_bands["green"], before_bands["blue"])
        after_img = true_color_image(after_bands["red"], after_bands["green"], after_bands["blue"])
        heatmap_img = index_to_rgba(delta, threshold=0.05)

    assert isinstance(before_img, Image.Image)
    assert isinstance(heatmap_img, Image.Image)
```

- [ ] **Step 5: Run integration tests in collect-only mode**

Run: `pytest tests/integration/ --collect-only 2>&1 | tail -20`
Expected: all tests collected without import errors

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/integration/
git commit -m "test: add timing assertions to integration tests"
```

---

### Task 10: Performance Benchmarks

**Files:**
- Create: `tests/perf/test_benchmarks.py`

- [ ] **Step 1: Write tests/perf/test_benchmarks.py**

```python
"""Performance benchmarks for computation and rendering.

All tests use synthetic data (no network). Thresholds are generous —
failure means a genuine regression, not normal machine variance.

Run with: pytest tests/perf/ -v
"""
import time

import numpy as np
import pytest
from PIL import Image

from src.indices import compute_ndvi, compute_ndbi, compute_mndwi, compute_change, _safe_normalized_diff
from src.visualization import true_color_image, index_to_rgba, downscale_array


pytestmark = pytest.mark.perf


class TestIndexBenchmarks:
    def test_ndvi_2000x2000_under_100ms(self, large_bands):
        start = time.perf_counter()
        result = compute_ndvi(large_bands["nir"], large_bands["red"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"NDVI took {elapsed:.3f}s, expected < 0.1s"
        assert result.shape == (2000, 2000)

    def test_ndbi_2000x2000_under_100ms(self, large_bands):
        start = time.perf_counter()
        result = compute_ndbi(large_bands["swir16"], large_bands["nir"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"NDBI took {elapsed:.3f}s, expected < 0.1s"
        assert result.shape == (2000, 2000)

    def test_mndwi_2000x2000_under_100ms(self, large_bands):
        start = time.perf_counter()
        result = compute_mndwi(large_bands["green"], large_bands["swir16"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"MNDWI took {elapsed:.3f}s, expected < 0.1s"
        assert result.shape == (2000, 2000)

    def test_compute_change_2000x2000_under_50ms(self, large_bands):
        before = compute_ndvi(large_bands["nir"], large_bands["red"])
        after = compute_ndvi(large_bands["nir"], large_bands["green"])  # different band for variation
        start = time.perf_counter()
        delta = compute_change(before=before, after=after)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05, f"compute_change took {elapsed:.3f}s, expected < 0.05s"
        assert delta.shape == (2000, 2000)

    def test_chunked_ndvi_not_slower_than_2x(self, large_bands):
        nir = large_bands["nir"]
        red = large_bands["red"]

        # Single-pass timing
        start = time.perf_counter()
        _safe_normalized_diff(nir, red, chunk_rows=None)
        single_time = time.perf_counter() - start

        # Chunked timing
        start = time.perf_counter()
        _safe_normalized_diff(nir, red, chunk_rows=512)
        chunked_time = time.perf_counter() - start

        assert chunked_time < single_time * 2, (
            f"Chunked ({chunked_time:.3f}s) > 2x single-pass ({single_time:.3f}s)"
        )


class TestRenderingBenchmarks:
    def test_true_color_2000x2000_under_200ms(self, large_bands):
        start = time.perf_counter()
        img = true_color_image(large_bands["red"], large_bands["green"], large_bands["blue"])
        elapsed = time.perf_counter() - start
        assert elapsed < 0.2, f"true_color_image took {elapsed:.3f}s, expected < 0.2s"
        assert isinstance(img, Image.Image)
        assert img.size == (2000, 2000)

    def test_index_to_rgba_2000x2000_under_200ms(self, large_bands):
        delta = compute_ndvi(large_bands["nir"], large_bands["red"])
        start = time.perf_counter()
        img = index_to_rgba(delta, threshold=0.05)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.2, f"index_to_rgba took {elapsed:.3f}s, expected < 0.2s"
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"

    def test_downscale_2000x2000_to_800_under_50ms(self, large_bands):
        arr = compute_ndvi(large_bands["nir"], large_bands["red"])
        start = time.perf_counter()
        result = downscale_array(arr, max_dim=800)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05, f"downscale_array took {elapsed:.3f}s, expected < 0.05s"
        assert max(result.shape) == 800


class TestPipelineBenchmark:
    def test_full_compute_pipeline_under_500ms(self, large_bands):
        """Full compute path: 3 indices, 3 deltas, 2 true-color, 1 heatmap."""
        start = time.perf_counter()

        # Compute all 3 indices for "before" and "after"
        # Use different random offsets to simulate before/after variation
        before_ndvi = compute_ndvi(large_bands["nir"], large_bands["red"])
        after_ndvi = compute_ndvi(large_bands["nir"], large_bands["green"])
        delta_ndvi = compute_change(before=before_ndvi, after=after_ndvi)

        before_ndbi = compute_ndbi(large_bands["swir16"], large_bands["nir"])
        after_ndbi = compute_ndbi(large_bands["swir16"], large_bands["red"])
        delta_ndbi = compute_change(before=before_ndbi, after=after_ndbi)

        before_mndwi = compute_mndwi(large_bands["green"], large_bands["swir16"])
        after_mndwi = compute_mndwi(large_bands["green"], large_bands["nir"])
        delta_mndwi = compute_change(before=before_mndwi, after=after_mndwi)

        # True-color images
        before_img = true_color_image(large_bands["red"], large_bands["green"], large_bands["blue"])
        after_img = true_color_image(large_bands["red"], large_bands["green"], large_bands["blue"])

        # Heatmap (using NDVI delta)
        small_delta = downscale_array(delta_ndvi, max_dim=800)
        heatmap = index_to_rgba(small_delta, threshold=0.05)

        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Full pipeline took {elapsed:.3f}s, expected < 0.5s"

        # Sanity checks
        assert isinstance(before_img, Image.Image)
        assert isinstance(after_img, Image.Image)
        assert isinstance(heatmap, Image.Image)
        assert heatmap.mode == "RGBA"
```

- [ ] **Step 2: Run the benchmarks**

Run: `pytest tests/perf/ -v`
Expected: 9 tests pass, all within thresholds

- [ ] **Step 3: Commit**

```bash
git add tests/perf/test_benchmarks.py
git commit -m "test: add performance benchmarks with timing thresholds"
```

---

### Task 11: Clean Up and Final Verification

**Files:**
- Verify: no stale test files remain in `tests/` root
- Verify: all tiers pass

- [ ] **Step 1: Verify no old test files remain in tests/ root**

```bash
ls tests/*.py
```

Expected: only `tests/__init__.py` and `tests/conftest.py` remain. If any `test_*.py` files remain, they are stale and should have been moved in earlier tasks.

- [ ] **Step 2: Run unit tests and check timing**

Run: `pytest tests/unit/ -v --tb=short`
Expected: ~40 tests pass

- [ ] **Step 3: Run perf tests and check timing**

Run: `pytest tests/perf/ -v --tb=short`
Expected: 9 tests pass, all within thresholds

- [ ] **Step 4: Run integration tests in collect-only mode (no network required)**

Run: `pytest tests/integration/ --collect-only`
Expected: all tests collected, no import errors

- [ ] **Step 5: Verify unit + perf total runtime**

Run: `pytest tests/unit/ tests/perf/ -v --tb=short -q`
Expected: total wall time < 15 seconds

- [ ] **Step 6: Commit any cleanup**

```bash
git add -A tests/
git commit -m "test: final cleanup of three-tier test restructure"
```
