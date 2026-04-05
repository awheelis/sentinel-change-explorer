"""Curate a preset-biased Sentinel-2 chip dataset for LeJEPA pretraining.

High-level pipeline (when the full module is built out):

    1. Loop through each preset in config/presets.json, search S2 scenes in
       both before and after date ranges, load the 5 reflectance bands + SCL
       for the AOI via src.sentinel.load_bands, tile-crop into non-overlapping
       128x128 chips, reject chips with >25% cloud/shadow or >10% fill.

    2. Loop through a hand-curated list of diverse global points (deserts,
       forests, cities, coasts, ice, agriculture), same tile-and-filter flow.

    3. Assemble kept chips into a HuggingFace `datasets.Dataset` with a typed
       schema, compute per-band global mean/std over the train split, save
       to disk, and optionally push to the Hub.

This module is runnable as:

    uv run python -m src.experimental.build_dataset --output cache/lejepa_dataset

Phase 2 lands the pure helpers (tile cropping, chip rejection, bbox math) and
their tests. Subsequent tasks fill in the STAC-fetching pipeline, dataset
assembly, norm stats, and the CLI. Heavy imports (datasets, huggingface_hub)
are deferred into the functions that use them so the pure helpers can be
unit-tested with just numpy — no experimental extras required at test time.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.masking import build_scl_mask

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_NORM_STATS_REPO_PATH = _REPO_ROOT / "src" / "experimental" / "norm_stats.json"

# ── Constants ────────────────────────────────────────────────────────────────

#: Side length (pixels) of each training chip. At 10 m/px this is 1.28 km.
CHIP_SIZE: int = 128

#: Target pixel resolution in meters. Matches the rest of the app's default
#: so chips are directly comparable to what load_bands() produces.
TARGET_RES_M: int = 10

#: Reflectance band order for the 5-channel model input. This ordering is
#: what the ResNet-18 first conv expects at train and inference time —
#: changing it requires retraining.
REFLECTANCE_BANDS: tuple[str, ...] = ("red", "green", "blue", "nir", "swir16")

#: All bands fetched from STAC per chip. Includes SCL for cloud masking,
#: which is dropped from the stored reflectance tensor but used for the
#: cloud-fraction rejection check.
FETCH_BANDS: tuple[str, ...] = (*REFLECTANCE_BANDS, "scl")

#: Rejection thresholds. Chips with SCL cloud/shadow coverage above this
#: fraction are dropped.
MAX_CLOUD_FRACTION: float = 0.25

#: Chips where more than this fraction of pixels are all-bands-zero (true
#: no-data fill) are dropped.
MAX_FILL_FRACTION: float = 0.10

#: Edge length (km) of the expanded square AOI used when collecting preset
#: chips. Presets in ``config/presets.json`` have tight demo bboxes (~3 km
#: across); we expand around their center to a 10 km square to get enough
#: non-overlapping 128 px chips per scene for SSL pretraining.
PRESET_AOI_SIZE_KM: float = 10.0

#: Edge length (km) of the square AOI used for each global diversity point.
#: 5.12 km at 10 m/px → 512 px side → 4×4 = 16 chips per scene (modulo
#: reprojection rounding).
GLOBAL_AOI_SIZE_KM: float = 5.12

#: Max cloud cover (%) for STAC search on preset scenes. Stricter than
#: global because we want the preset demo tiles to be clean.
PRESET_MAX_CLOUD_COVER: float = 20.0

#: Max cloud cover (%) for STAC search on global diversity scenes. A bit
#: looser because many global points live in cloudy regions and the chip
#: rejection step drops the actually-cloudy pixels anyway.
GLOBAL_MAX_CLOUD_COVER: float = 30.0


#: Hand-curated list of diverse global points for the ~30% global split.
#: Each entry specifies a center lon/lat and a small set of date ranges
#: spread across seasons. Chosen to cover deserts, forests, croplands,
#: urban cores, coasts, ice/mountain, and wetlands on every inhabited
#: continent. This list is intentionally kept short and readable — if
#: you need more diversity, add entries here, don't bloat the existing
#: ones.
GLOBAL_POINTS: tuple[dict[str, Any], ...] = (
    # ── Deserts ─────────────────────────────────────────────────────────
    {"name": "sahara_algeria", "lon": 2.00, "lat": 25.00, "dates": (
        "2023-01-15/2023-02-15", "2023-06-01/2023-06-30", "2023-11-01/2023-11-30")},
    {"name": "gobi_mongolia", "lon": 104.00, "lat": 43.50, "dates": (
        "2023-04-01/2023-04-30", "2023-07-15/2023-08-15", "2023-10-01/2023-10-31")},
    {"name": "atacama_chile", "lon": -69.30, "lat": -23.80, "dates": (
        "2023-03-01/2023-03-31", "2023-07-01/2023-07-31", "2023-11-01/2023-11-30")},
    {"name": "namib_namibia", "lon": 15.00, "lat": -23.50, "dates": (
        "2023-02-01/2023-02-28", "2023-06-01/2023-06-30", "2023-09-01/2023-09-30")},
    {"name": "simpson_australia", "lon": 137.50, "lat": -25.50, "dates": (
        "2023-03-01/2023-03-31", "2023-08-01/2023-08-31", "2023-12-01/2023-12-31")},
    # ── Forests ─────────────────────────────────────────────────────────
    {"name": "amazon_brazil", "lon": -60.00, "lat": -3.50, "dates": (
        "2023-07-01/2023-07-31", "2023-09-01/2023-09-30", "2024-01-01/2024-01-31")},
    {"name": "congo_drc", "lon": 21.00, "lat": -1.00, "dates": (
        "2023-06-01/2023-06-30", "2023-12-01/2023-12-31", "2024-03-01/2024-03-31")},
    {"name": "boreal_canada", "lon": -95.00, "lat": 54.00, "dates": (
        "2023-06-01/2023-06-30", "2023-08-01/2023-08-31", "2023-09-15/2023-10-15")},
    {"name": "siberia_taiga", "lon": 105.00, "lat": 62.00, "dates": (
        "2023-07-01/2023-07-31", "2023-08-15/2023-09-15")},
    {"name": "pnw_usa", "lon": -123.50, "lat": 47.50, "dates": (
        "2023-06-01/2023-06-30", "2023-08-15/2023-09-15", "2023-11-01/2023-11-30")},
    # ── Croplands ───────────────────────────────────────────────────────
    {"name": "iowa_corn_belt", "lon": -93.50, "lat": 42.00, "dates": (
        "2023-05-01/2023-05-31", "2023-07-15/2023-08-15", "2023-10-01/2023-10-31")},
    {"name": "pampas_argentina", "lon": -62.00, "lat": -35.00, "dates": (
        "2023-02-01/2023-02-28", "2023-06-01/2023-06-30", "2023-11-01/2023-11-30")},
    {"name": "po_valley_italy", "lon": 10.50, "lat": 45.00, "dates": (
        "2023-05-01/2023-05-31", "2023-07-01/2023-07-31", "2023-10-01/2023-10-31")},
    {"name": "punjab_india", "lon": 75.50, "lat": 30.70, "dates": (
        "2023-03-01/2023-03-31", "2023-07-01/2023-07-31", "2023-11-01/2023-11-30")},
    # ── Urban ───────────────────────────────────────────────────────────
    {"name": "tokyo_japan", "lon": 139.75, "lat": 35.70, "dates": (
        "2023-04-01/2023-04-30", "2023-10-01/2023-10-31")},
    {"name": "nyc_usa", "lon": -73.95, "lat": 40.75, "dates": (
        "2023-05-01/2023-05-31", "2023-10-01/2023-10-31")},
    {"name": "lagos_nigeria", "lon": 3.40, "lat": 6.50, "dates": (
        "2023-02-01/2023-02-28", "2023-11-01/2023-11-30")},
    {"name": "sao_paulo_brazil", "lon": -46.63, "lat": -23.55, "dates": (
        "2023-06-01/2023-06-30", "2023-09-01/2023-09-30")},
    {"name": "cairo_egypt", "lon": 31.25, "lat": 30.05, "dates": (
        "2023-03-01/2023-03-31", "2023-10-01/2023-10-31")},
    {"name": "shanghai_china", "lon": 121.47, "lat": 31.23, "dates": (
        "2023-04-01/2023-04-30", "2023-10-01/2023-10-31")},
    # ── Coasts + water ──────────────────────────────────────────────────
    {"name": "chesapeake_bay", "lon": -76.20, "lat": 38.50, "dates": (
        "2023-05-01/2023-05-31", "2023-09-01/2023-09-30")},
    {"name": "dutch_coast", "lon": 4.50, "lat": 52.50, "dates": (
        "2023-05-01/2023-05-31", "2023-08-01/2023-08-31")},
    {"name": "normandy_france", "lon": -0.50, "lat": 49.30, "dates": (
        "2023-06-01/2023-06-30", "2023-09-01/2023-09-30")},
    # ── Ice / mountain ──────────────────────────────────────────────────
    {"name": "greenland_glacier", "lon": -49.70, "lat": 69.20, "dates": (
        "2023-07-01/2023-07-31", "2023-08-15/2023-09-15")},
    {"name": "alps_switzerland", "lon": 8.00, "lat": 46.50, "dates": (
        "2023-07-01/2023-07-31", "2023-09-01/2023-09-30")},
    {"name": "andes_peru", "lon": -72.00, "lat": -13.50, "dates": (
        "2023-05-01/2023-05-31", "2023-08-01/2023-08-31")},
    {"name": "himalaya_nepal", "lon": 86.50, "lat": 27.80, "dates": (
        "2023-10-01/2023-10-31", "2023-11-15/2023-12-15")},
    # ── Wetlands / mixed ────────────────────────────────────────────────
    {"name": "everglades_usa", "lon": -80.80, "lat": 25.80, "dates": (
        "2023-02-01/2023-02-28", "2023-09-01/2023-09-30")},
    {"name": "pantanal_brazil", "lon": -56.00, "lat": -17.50, "dates": (
        "2023-04-01/2023-04-30", "2023-09-01/2023-09-30")},
    {"name": "okavango_botswana", "lon": 22.80, "lat": -19.30, "dates": (
        "2023-04-01/2023-04-30", "2023-09-01/2023-09-30")},
)


# ── Pure numpy helpers (no network, no datasets import) ──────────────────────


def tile_crop(arr: np.ndarray, chip_size: int = CHIP_SIZE) -> list[np.ndarray]:
    """Split a (C, H, W) array into non-overlapping (C, chip_size, chip_size) tiles.

    Chips are emitted in row-major order: left-to-right within a row, then
    top-to-bottom. The remainder along either axis (when H or W is not a
    multiple of chip_size) is silently dropped — for S2 chip extraction this
    is the desired behavior because remainder strips produce partial chips.

    Args:
        arr: Input array of shape (C, H, W). C is typically the number of
            reflectance bands; the channel axis is preserved per-chip.
        chip_size: Side length of each output chip in pixels.

    Returns:
        List of arrays each of shape (C, chip_size, chip_size) with the
        same dtype as ``arr``. Empty list if H or W is smaller than
        ``chip_size``.
    """
    if arr.ndim != 3:
        raise ValueError(f"tile_crop expects a (C, H, W) array, got shape {arr.shape}")
    _, h, w = arr.shape
    n_rows = h // chip_size
    n_cols = w // chip_size
    chips: list[np.ndarray] = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * chip_size, (r + 1) * chip_size
            x0, x1 = c * chip_size, (c + 1) * chip_size
            chips.append(arr[:, y0:y1, x0:x1].copy())
    return chips


def compute_reject_stats(
    reflectance_chip: np.ndarray, scl_chip: np.ndarray
) -> dict[str, float]:
    """Compute cloud and fill fractions for a candidate chip.

    Cloud fraction reuses ``src.masking.build_scl_mask`` so the definition
    of "cloud/shadow" stays consistent with the rest of the app (SCL
    classes 3, 8, 9, 10).

    Fill fraction counts only pixels where ALL reflectance bands are zero.
    Sentinel-2 L2A tiles pad with zeros outside the valid footprint, so
    all-bands-zero is a reliable fill signal. A pixel that is zero in a
    single band is just dark in that band and is NOT treated as fill.

    Args:
        reflectance_chip: Array of shape (C, H, W) where C is the number
            of reflectance bands (expected 5 for the LeJEPA pipeline).
        scl_chip: 2D SCL band array of shape (H, W) aligned with
            ``reflectance_chip``'s spatial dimensions.

    Returns:
        Dict with keys ``cloud_fraction`` and ``fill_fraction``, each a
        float in [0.0, 1.0].
    """
    cloud_mask = build_scl_mask(scl_chip)
    cloud_fraction = float(np.mean(cloud_mask))

    fill_mask = np.all(reflectance_chip == 0, axis=0)
    fill_fraction = float(np.mean(fill_mask))

    return {"cloud_fraction": cloud_fraction, "fill_fraction": fill_fraction}


def should_keep_chip(
    stats: dict[str, float],
    max_cloud: float = MAX_CLOUD_FRACTION,
    max_fill: float = MAX_FILL_FRACTION,
) -> bool:
    """Apply rejection rules to a chip's statistics.

    A chip is kept when both cloud and fill fractions are <= the respective
    thresholds. The <= comparison is deliberate: chips exactly at the
    threshold are kept, so the thresholds are inclusive upper bounds.

    Args:
        stats: Output of ``compute_reject_stats``.
        max_cloud: Maximum allowed cloud/shadow fraction.
        max_fill: Maximum allowed fill fraction.

    Returns:
        True if the chip should be kept, False if it should be dropped.
    """
    return (
        stats["cloud_fraction"] <= max_cloud
        and stats["fill_fraction"] <= max_fill
    )


def bbox_around_point(
    lon: float, lat: float, size_km: float
) -> tuple[float, float, float, float]:
    """Build a square WGS84 bbox of edge length ``size_km`` centered on (lon, lat).

    Uses the small-angle approximation: 111.32 km per degree of longitude
    at the equator (scaled by cos(lat) elsewhere), and 110.54 km per degree
    of latitude (approximately constant). Good to well under a percent at
    bbox sizes of a few km, which is plenty for STAC search + tile
    windowing.

    Args:
        lon: Center longitude in degrees.
        lat: Center latitude in degrees.
        size_km: Edge length of the square bbox in kilometers.

    Returns:
        Tuple ``(west, south, east, north)`` in WGS84 degrees, matching
        the (west, south, east, north) convention used by the rest of the
        codebase.
    """
    half_size_m = size_km * 500.0  # half the edge, in meters
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    half_lat = half_size_m / m_per_deg_lat
    half_lon = half_size_m / m_per_deg_lon
    return (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def expand_preset_aoi(
    preset: dict[str, Any], size_km: float = PRESET_AOI_SIZE_KM
) -> tuple[float, float, float, float]:
    """Return a ``size_km`` square bbox centered on a preset's original bbox.

    The presets in ``config/presets.json`` use tight 3–4 km demo bboxes
    framed on the specific event (a burn scar, a flood, a factory). For
    LeJEPA pretraining we want *nearby terrain*, not just the event, so
    we expand to ``size_km`` around the preset's centroid to yield enough
    non-overlapping chips per scene.
    """
    west, south, east, north = preset["bbox"]
    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    return bbox_around_point(center_lon, center_lat, size_km=size_km)


def _extract_chips_from_bands(
    bands: dict[str, np.ndarray],
    bbox: tuple[float, float, float, float],
    scene_id: str,
    acquisition_date: str,
    source: str,
    preset_name: str | None,
) -> Iterator[dict[str, Any]]:
    """Tile-crop one band-dict into chip records, applying rejection rules.

    Stacks the 5 reflectance bands in ``REFLECTANCE_BANDS`` order into a
    (5, H, W) tensor, slices into non-overlapping 128x128 chips in parallel
    with the SCL band, computes cloud + fill fractions per chip, and yields
    a record dict for every chip that passes the rejection thresholds.
    """
    try:
        reflectance = np.stack([bands[k] for k in REFLECTANCE_BANDS], axis=0)
    except KeyError as exc:
        logger.warning("Scene %s missing required band %s — skipping", scene_id, exc)
        return
    scl = bands.get("scl")
    if scl is None:
        logger.warning("Scene %s missing SCL band — skipping", scene_id)
        return

    refl_chips = tile_crop(reflectance, chip_size=CHIP_SIZE)
    # Wrap SCL with a channel axis so tile_crop accepts it (it expects CxHxW),
    # then unwrap after cropping. This keeps tile_crop's signature strict.
    scl_chips_wrapped = tile_crop(scl[np.newaxis, ...], chip_size=CHIP_SIZE)

    for refl_chip, scl_chip_wrapped in zip(refl_chips, scl_chips_wrapped):
        scl_chip = scl_chip_wrapped[0]
        stats = compute_reject_stats(refl_chip, scl_chip)
        if not should_keep_chip(stats):
            continue
        yield {
            "bands": refl_chip.astype(np.uint16),
            "bbox": list(bbox),
            "acquisition_date": acquisition_date,
            "scene_id": scene_id,
            "source": source,
            "preset_name": preset_name,
        }


def collect_preset_chips(
    presets: list[dict[str, Any]],
    max_scenes_per_range: int = 6,
    max_chips: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield chip records for every preset in ``presets``.

    For each preset, searches both ``before_range`` and ``after_range`` via
    the app's existing ``src.sentinel.search_scenes``, fetches up to
    ``max_scenes_per_range`` scenes per range, loads the 5+1 bands for the
    expanded AOI via ``src.sentinel.load_bands`` (which handles the disk
    cache and reprojection automatically), tile-crops, rejects bad chips,
    and yields the survivors.

    Args:
        presets: List of preset dicts loaded from ``config/presets.json``.
        max_scenes_per_range: Cap on STAC results per date range per preset.
            Raising this increases temporal diversity at the cost of more
            S3 I/O. 6 gives ~60 scenes total across 5 presets × 2 ranges.
        max_chips: Stop yielding after this many kept chips. None = exhaust
            all scenes. Used by the CLI to honor ``--n-preset``.

    Yields:
        Chip record dicts matching the schema in ``_extract_chips_from_bands``.
    """
    # Local import keeps the pure-helper unit tests network-free.
    from src.sentinel import load_bands, search_scenes

    yielded = 0
    for preset in presets:
        aoi_bbox = expand_preset_aoi(preset)
        for range_key in ("before_range", "after_range"):
            dr = preset[range_key]
            date_range = f"{dr[0]}/{dr[1]}"
            try:
                scenes = search_scenes(
                    bbox=aoi_bbox,
                    date_range=date_range,
                    max_cloud_cover=PRESET_MAX_CLOUD_COVER,
                    max_items=max_scenes_per_range,
                )
            except Exception as exc:  # noqa: BLE001 — STAC errors vary
                logger.warning("STAC search failed for %s %s: %s", preset["name"], range_key, exc)
                continue

            for scene in scenes:
                try:
                    bands = load_bands(
                        scene=scene,
                        bbox=aoi_bbox,
                        band_keys=list(FETCH_BANDS),
                        target_res=TARGET_RES_M,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("load_bands failed for scene %s: %s", scene["id"], exc)
                    continue

                for chip in _extract_chips_from_bands(
                    bands=bands,
                    bbox=aoi_bbox,
                    scene_id=scene["id"],
                    acquisition_date=scene["datetime"][:10],
                    source="preset",
                    preset_name=preset["name"],
                ):
                    yield chip
                    yielded += 1
                    if max_chips is not None and yielded >= max_chips:
                        return


def collect_global_chips(
    max_chips: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield chip records from the hand-curated GLOBAL_POINTS list.

    Iterates every (point, date_range) pair, grabs the lowest-cloud scene
    per pair, tiles it, and yields survivors. Stops early if ``max_chips``
    is reached.
    """
    from src.sentinel import load_bands, search_scenes

    yielded = 0
    for point in GLOBAL_POINTS:
        aoi_bbox = bbox_around_point(
            lon=point["lon"], lat=point["lat"], size_km=GLOBAL_AOI_SIZE_KM
        )
        for date_range in point["dates"]:
            try:
                scenes = search_scenes(
                    bbox=aoi_bbox,
                    date_range=date_range,
                    max_cloud_cover=GLOBAL_MAX_CLOUD_COVER,
                    max_items=1,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("STAC search failed for %s %s: %s", point["name"], date_range, exc)
                continue
            if not scenes:
                continue
            scene = scenes[0]
            try:
                bands = load_bands(
                    scene=scene,
                    bbox=aoi_bbox,
                    band_keys=list(FETCH_BANDS),
                    target_res=TARGET_RES_M,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("load_bands failed for %s %s: %s", point["name"], date_range, exc)
                continue

            for chip in _extract_chips_from_bands(
                bands=bands,
                bbox=aoi_bbox,
                scene_id=scene["id"],
                acquisition_date=scene["datetime"][:10],
                source="global",
                preset_name=None,
            ):
                yield chip
                yielded += 1
                if max_chips is not None and yielded >= max_chips:
                    return


# ── HuggingFace dataset assembly ─────────────────────────────────────────────


def _build_features_spec():
    """Return the typed ``datasets.Features`` spec for one chip record.

    Imported lazily because ``datasets`` is only in the experimental extra.
    """
    from datasets import Array3D, ClassLabel, Features, Sequence, Value

    return Features({
        "bands": Array3D(shape=(len(REFLECTANCE_BANDS), CHIP_SIZE, CHIP_SIZE), dtype="uint16"),
        "bbox": Sequence(Value("float64"), length=4),
        "acquisition_date": Value("string"),
        "scene_id": Value("string"),
        "source": ClassLabel(names=["preset", "global"]),
        "preset_name": Value("string"),
    })


def chips_to_dataset(chips: list[dict[str, Any]]):
    """Assemble a list of chip dicts into a ``datasets.Dataset`` with a typed schema.

    ``preset_name`` is coerced to an empty string for global chips because
    the ``Value("string")`` field can't hold None and ClassLabel would be
    overkill (the set of preset names is not fixed).
    """
    from datasets import Dataset

    if not chips:
        raise ValueError("chips_to_dataset called with an empty list")

    normalized = [
        {
            "bands": chip["bands"],
            "bbox": list(chip["bbox"]),
            "acquisition_date": chip["acquisition_date"],
            "scene_id": chip["scene_id"],
            "source": chip["source"],
            "preset_name": chip["preset_name"] or "",
        }
        for chip in chips
    ]

    return Dataset.from_list(normalized, features=_build_features_spec())


def split_train_val(dataset, val_fraction: float = 0.10, seed: int = 42):
    """Return a ``DatasetDict`` with 'train' and 'validation' splits.

    Uses ``datasets.Dataset.train_test_split`` internally and renames the
    'test' split to 'validation' for clarity (this is a pretraining dataset,
    there is no held-out test set).
    """
    from datasets import DatasetDict

    split = dataset.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def compute_norm_stats(train_dataset) -> dict[str, Any]:
    """Compute per-band global mean and std across every chip in the train split.

    Iterates the dataset once, accumulating sum and sum-of-squares per band
    in float64 to avoid uint16 overflow. Variance is clipped to >= 0 before
    sqrt to tolerate tiny negative drift from floating-point roundoff.
    """
    n_bands = len(REFLECTANCE_BANDS)
    total_sum = np.zeros(n_bands, dtype=np.float64)
    total_sq = np.zeros(n_bands, dtype=np.float64)
    total_count = 0

    for record in train_dataset:
        bands = np.asarray(record["bands"], dtype=np.float64)
        n_pixels = bands.shape[1] * bands.shape[2]
        total_sum += bands.sum(axis=(1, 2))
        total_sq += (bands ** 2).sum(axis=(1, 2))
        total_count += n_pixels

    if total_count == 0:
        raise ValueError("compute_norm_stats called with empty train split")

    mean = total_sum / total_count
    var = (total_sq / total_count) - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    return {
        "bands": list(REFLECTANCE_BANDS),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pixel_count": int(total_count),
        "chip_count": int(len(train_dataset)),
    }


def save_dataset_bundle(
    dataset_dict,
    norm_stats: dict[str, Any],
    output_dir: Path | str,
    copy_norm_stats_to_repo: bool = False,
) -> Path:
    """Write dataset splits + norm_stats.json to disk.

    Args:
        dataset_dict: DatasetDict from ``split_train_val``.
        norm_stats: Dict from ``compute_norm_stats``.
        output_dir: Directory to write into. Created if missing.
        copy_norm_stats_to_repo: When True, also writes a copy to
            ``src/experimental/norm_stats.json`` — the canonical path
            inference code will read from if it's committed to the repo.
            Defaults to False so smoke builds don't clobber the real file;
            pass True explicitly on the final production build. In the long
            run Phase 5 will pack norm stats into the HF model repo and
            Phase 6 will pull them from there, making the committed copy
            optional.

    Returns:
        Absolute path to the output directory.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict.save_to_disk(str(output_dir))

    norm_path = output_dir / "norm_stats.json"
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info("Wrote norm stats to %s", norm_path)

    if copy_norm_stats_to_repo:
        _NORM_STATS_REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_NORM_STATS_REPO_PATH, "w") as f:
            json.dump(norm_stats, f, indent=2)
        logger.info("Copied norm stats to repo at %s", _NORM_STATS_REPO_PATH)

    return output_dir


# ── CLI ──────────────────────────────────────────────────────────────────────


def run_build(
    output_dir: Path | str,
    n_preset: int,
    n_global: int,
    max_scenes_per_range: int = 6,
    seed: int = 42,
    push_to_hub: str | None = None,
    copy_norm_stats_to_repo: bool = False,
) -> Path:
    """End-to-end dataset build: collect chips, assemble, split, save.

    Args:
        output_dir: Directory to write the DatasetDict into.
        n_preset: Target number of preset-sourced chips.
        n_global: Target number of global-sourced chips.
        max_scenes_per_range: Cap on STAC results per preset per range.
        seed: Random seed for the train/val split.
        push_to_hub: Optional HuggingFace repo id (e.g. ``user/dataset-name``).
            Phase 3 wires this through to ``Dataset.push_to_hub``. For now,
            passing it raises NotImplementedError so the CLI surface is
            stable but the actual upload lands in Phase 3.

    Returns:
        Absolute path to the saved output directory.
    """
    from src.experimental.build_dataset import (  # self-import for __main__
        collect_global_chips,
        collect_preset_chips,
    )

    # Loading presets through the app's existing loader keeps the source of
    # truth consistent with the Streamlit sidebar.
    presets_path = _REPO_ROOT / "config" / "presets.json"
    with open(presets_path) as f:
        presets = json.load(f)

    logger.info("Collecting up to %d preset chips …", n_preset)
    preset_chips: list[dict[str, Any]] = []
    for chip in collect_preset_chips(
        presets=presets,
        max_scenes_per_range=max_scenes_per_range,
        max_chips=n_preset,
    ):
        preset_chips.append(chip)
        if len(preset_chips) % 100 == 0:
            logger.info("  preset progress: %d / %d", len(preset_chips), n_preset)
    logger.info("Collected %d preset chips", len(preset_chips))

    logger.info("Collecting up to %d global chips …", n_global)
    global_chips: list[dict[str, Any]] = []
    for chip in collect_global_chips(max_chips=n_global):
        global_chips.append(chip)
        if len(global_chips) % 100 == 0:
            logger.info("  global progress: %d / %d", len(global_chips), n_global)
    logger.info("Collected %d global chips", len(global_chips))

    all_chips = preset_chips + global_chips
    if not all_chips:
        raise RuntimeError("No chips collected — check network and STAC availability")
    logger.info("Total chips: %d (preset: %d, global: %d)",
                len(all_chips), len(preset_chips), len(global_chips))

    ds = chips_to_dataset(all_chips)
    dd = split_train_val(ds, val_fraction=0.10, seed=seed)
    stats = compute_norm_stats(dd["train"])
    logger.info("Norm stats: mean=%s std=%s",
                [round(m, 1) for m in stats["mean"]],
                [round(s, 1) for s in stats["std"]])

    out_path = save_dataset_bundle(
        dd, stats, output_dir=output_dir,
        copy_norm_stats_to_repo=copy_norm_stats_to_repo,
    )

    if push_to_hub is not None:
        raise NotImplementedError(
            "push_to_hub lands in Phase 3. For now build locally, then push "
            "in a separate step."
        )

    return out_path


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.experimental.build_dataset",
        description="Build a preset-biased Sentinel-2 chip dataset for LeJEPA pretraining.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cache/lejepa_dataset"),
        help="Directory to save the DatasetDict into.",
    )
    parser.add_argument(
        "--n-preset", type=int, default=5000,
        help="Target number of preset-sourced chips (~70%% of dataset).",
    )
    parser.add_argument(
        "--n-global", type=int, default=2000,
        help="Target number of global-sourced chips (~30%% of dataset).",
    )
    parser.add_argument(
        "--max-scenes-per-range", type=int, default=6,
        help="Max STAC scenes to pull per preset per date range.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None,
        help="(Phase 3) HuggingFace repo id to push the built dataset to.",
    )
    parser.add_argument(
        "--copy-norm-stats-to-repo", action="store_true",
        help="Overwrite src/experimental/norm_stats.json with the computed "
             "stats. Use only on production builds, not smoke tests.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out = run_build(
        output_dir=args.output,
        n_preset=args.n_preset,
        n_global=args.n_global,
        max_scenes_per_range=args.max_scenes_per_range,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        copy_norm_stats_to_repo=args.copy_norm_stats_to_repo,
    )
    logger.info("Done. Dataset saved to: %s", out)


if __name__ == "__main__":
    _cli()


__all__ = [
    "CHIP_SIZE",
    "TARGET_RES_M",
    "REFLECTANCE_BANDS",
    "FETCH_BANDS",
    "MAX_CLOUD_FRACTION",
    "MAX_FILL_FRACTION",
    "PRESET_AOI_SIZE_KM",
    "GLOBAL_AOI_SIZE_KM",
    "GLOBAL_POINTS",
    "tile_crop",
    "compute_reject_stats",
    "should_keep_chip",
    "bbox_around_point",
    "expand_preset_aoi",
    "collect_preset_chips",
    "collect_global_chips",
    "chips_to_dataset",
    "split_train_val",
    "compute_norm_stats",
    "save_dataset_bundle",
    "run_build",
]
