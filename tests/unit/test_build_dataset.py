"""Unit tests for src.experimental.build_dataset pure logic.

Tests only the numpy-native helpers (tile_crop, chip rejection, bbox math).
Network-dependent pipeline functions are covered by a separate smoke test.
These tests do NOT require the `datasets` or `torch` extras to run.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.experimental.build_dataset import (
    bbox_around_point,
    compute_reject_stats,
    should_keep_chip,
    tile_crop,
)


# ── tile_crop ────────────────────────────────────────────────────────────────


def test_tile_crop_evenly_divisible():
    """256x256 input with chip_size=128 yields 4 non-overlapping chips."""
    arr = np.arange(5 * 256 * 256, dtype=np.uint16).reshape(5, 256, 256)
    chips = tile_crop(arr, chip_size=128)
    assert len(chips) == 4
    for chip in chips:
        assert chip.shape == (5, 128, 128)
        assert chip.dtype == np.uint16


def test_tile_crop_preserves_content_top_left():
    """First chip should be the top-left 128x128 slice."""
    arr = np.arange(5 * 256 * 256, dtype=np.uint16).reshape(5, 256, 256)
    chips = tile_crop(arr, chip_size=128)
    np.testing.assert_array_equal(chips[0], arr[:, :128, :128])


def test_tile_crop_preserves_content_bottom_right():
    """Last chip should be the bottom-right 128x128 slice (row-major order)."""
    arr = np.arange(5 * 256 * 256, dtype=np.uint16).reshape(5, 256, 256)
    chips = tile_crop(arr, chip_size=128)
    # Row-major: chips are [(0,0),(0,1),(1,0),(1,1)]
    np.testing.assert_array_equal(chips[-1], arr[:, 128:256, 128:256])


def test_tile_crop_drops_remainder():
    """200x200 with chip_size=128 yields a single 128x128 chip (remainder dropped)."""
    arr = np.zeros((5, 200, 200), dtype=np.uint16)
    chips = tile_crop(arr, chip_size=128)
    assert len(chips) == 1
    assert chips[0].shape == (5, 128, 128)


def test_tile_crop_too_small_returns_empty():
    """Input smaller than chip_size yields no chips (not an error)."""
    arr = np.zeros((5, 100, 100), dtype=np.uint16)
    chips = tile_crop(arr, chip_size=128)
    assert chips == []


def test_tile_crop_non_square_array():
    """Rectangular input: 128x384 with chip_size=128 yields 3 chips in a row."""
    arr = np.zeros((5, 128, 384), dtype=np.uint16)
    chips = tile_crop(arr, chip_size=128)
    assert len(chips) == 3
    for chip in chips:
        assert chip.shape == (5, 128, 128)


def test_tile_crop_preserves_band_channel_order():
    """Per-band values should remain contiguous after cropping."""
    arr = np.zeros((5, 128, 128), dtype=np.uint16)
    for c in range(5):
        arr[c] = c + 1  # band 0 = all 1s, band 1 = all 2s, ...
    chips = tile_crop(arr, chip_size=128)
    assert len(chips) == 1
    chip = chips[0]
    for c in range(5):
        assert np.all(chip[c] == c + 1)


# ── compute_reject_stats ─────────────────────────────────────────────────────


def test_reject_stats_clean_chip():
    """Fully clean chip: 0 cloud, 0 fill."""
    reflectance = np.full((5, 128, 128), 1500, dtype=np.uint16)
    scl = np.full((128, 128), 4, dtype=np.uint8)  # 4 = vegetation
    stats = compute_reject_stats(reflectance, scl)
    assert stats["cloud_fraction"] == 0.0
    assert stats["fill_fraction"] == 0.0


def test_reject_stats_fully_clouded_chip():
    """Entire chip is SCL class 9 (cloud high): cloud_fraction = 1.0."""
    reflectance = np.full((5, 128, 128), 1500, dtype=np.uint16)
    scl = np.full((128, 128), 9, dtype=np.uint8)
    stats = compute_reject_stats(reflectance, scl)
    assert stats["cloud_fraction"] == 1.0


def test_reject_stats_half_clouded_chip():
    """Top half is SCL class 3 (cloud shadow), bottom is vegetation."""
    reflectance = np.full((5, 128, 128), 1500, dtype=np.uint16)
    scl = np.full((128, 128), 4, dtype=np.uint8)
    scl[:64, :] = 3  # cloud shadow
    stats = compute_reject_stats(reflectance, scl)
    assert stats["cloud_fraction"] == pytest.approx(0.5)


def test_reject_stats_mixed_mask_classes():
    """SCL classes 3, 8, 9, 10 should all count toward cloud_fraction."""
    reflectance = np.full((5, 128, 128), 1500, dtype=np.uint16)
    scl = np.full((128, 128), 4, dtype=np.uint8)
    scl[0:32, :] = 3   # cloud shadow
    scl[32:64, :] = 8  # cloud med
    scl[64:96, :] = 9  # cloud high
    scl[96:128, 0:64] = 10  # thin cirrus (half row)
    # Total masked: 32*128 + 32*128 + 32*128 + 32*64 = 14336
    # Total pixels: 128*128 = 16384
    expected = 14336 / 16384
    stats = compute_reject_stats(reflectance, scl)
    assert stats["cloud_fraction"] == pytest.approx(expected)


def test_reject_stats_fill_detection():
    """Pixels where ALL 5 bands are zero count as fill."""
    reflectance = np.full((5, 128, 128), 1500, dtype=np.uint16)
    reflectance[:, 0:16, :] = 0  # top 16 rows are fill (all bands zero)
    scl = np.full((128, 128), 4, dtype=np.uint8)
    stats = compute_reject_stats(reflectance, scl)
    expected_fill = (16 * 128) / (128 * 128)
    assert stats["fill_fraction"] == pytest.approx(expected_fill)


def test_reject_stats_partial_zero_is_not_fill():
    """A pixel where only one band is zero is NOT fill (just dark in one band)."""
    reflectance = np.full((5, 128, 128), 1500, dtype=np.uint16)
    reflectance[0, :, :] = 0  # red band entirely zero
    # but other 4 bands are 1500, so no pixel has all-bands-zero
    scl = np.full((128, 128), 4, dtype=np.uint8)
    stats = compute_reject_stats(reflectance, scl)
    assert stats["fill_fraction"] == 0.0


# ── should_keep_chip ─────────────────────────────────────────────────────────


def test_should_keep_clean_chip():
    assert should_keep_chip(
        {"cloud_fraction": 0.0, "fill_fraction": 0.0},
        max_cloud=0.25,
        max_fill=0.10,
    ) is True


def test_should_keep_at_threshold_boundary():
    """Exactly at the threshold: keep (inclusive)."""
    assert should_keep_chip(
        {"cloud_fraction": 0.25, "fill_fraction": 0.10},
        max_cloud=0.25,
        max_fill=0.10,
    ) is True


def test_should_reject_over_cloud_threshold():
    assert should_keep_chip(
        {"cloud_fraction": 0.26, "fill_fraction": 0.0},
        max_cloud=0.25,
        max_fill=0.10,
    ) is False


def test_should_reject_over_fill_threshold():
    assert should_keep_chip(
        {"cloud_fraction": 0.0, "fill_fraction": 0.11},
        max_cloud=0.25,
        max_fill=0.10,
    ) is False


# ── bbox_around_point ────────────────────────────────────────────────────────


def test_bbox_around_point_order_is_wsen():
    """Returns (west, south, east, north) tuple in WGS84."""
    west, south, east, north = bbox_around_point(lon=0.0, lat=0.0, size_km=2.0)
    assert west < east
    assert south < north


def test_bbox_around_point_equator_square_width():
    """At the equator, a 2 km square bbox has ~0.018° longitude width."""
    west, south, east, north = bbox_around_point(lon=0.0, lat=0.0, size_km=2.0)
    width = east - west
    # 2000 m / 111320 m/deg ≈ 0.01796
    assert width == pytest.approx(0.01796, abs=0.001)


def test_bbox_around_point_centered():
    """Output bbox is centered on the requested point."""
    west, south, east, north = bbox_around_point(lon=-115.0, lat=36.0, size_km=3.0)
    assert (west + east) / 2 == pytest.approx(-115.0)
    assert (south + north) / 2 == pytest.approx(36.0)


def test_bbox_around_point_latitude_stretches_longitude_degrees():
    """At 60°N, the same physical 2 km span TWICE as many degrees of longitude.

    Degrees of longitude are physically narrower at higher latitudes
    (cos(lat) factor), so the same kilometer span requires more degrees.
    """
    _, _, east_eq, _ = bbox_around_point(lon=0.0, lat=0.0, size_km=2.0)
    _, _, east_60, _ = bbox_around_point(lon=0.0, lat=60.0, size_km=2.0)
    width_eq = 2 * east_eq  # symmetric around 0
    width_60 = 2 * east_60
    # cos(60°) = 0.5 exactly, so width_60 = width_eq / 0.5 = 2 * width_eq
    ratio = width_60 / width_eq
    assert ratio == pytest.approx(2.0, abs=0.01)


def test_bbox_around_point_latitude_invariance():
    """Latitude span is independent of the input latitude (110.54 km/deg constant)."""
    _, south_eq, _, north_eq = bbox_around_point(lon=0.0, lat=0.0, size_km=2.0)
    _, south_60, _, north_60 = bbox_around_point(lon=0.0, lat=60.0, size_km=2.0)
    assert (north_eq - south_eq) == pytest.approx(north_60 - south_60, abs=1e-6)


# ── render_dataset_card ──────────────────────────────────────────────────────


from src.experimental.build_dataset import render_dataset_card  # noqa: E402


def _fake_presets():
    return [
        {
            "name": "Demo Preset",
            "bbox": [-156.695, 20.860, -156.660, 20.895],
            "before_range": ["2023-05-01", "2023-07-31"],
            "after_range": ["2023-09-01", "2023-11-30"],
        }
    ]


def _fake_stats():
    return {
        "bands": ["red", "green", "blue", "nir", "swir16"],
        "mean": [1500.0, 1200.0, 900.0, 2200.0, 3100.0],
        "std": [400.0, 350.0, 300.0, 500.0, 600.0],
        "chip_count": 100,
        "pixel_count": 100 * 128 * 128,
    }


def test_render_dataset_card_substitutes_all_placeholders():
    """Rendered card contains substituted stats and no leftover braces."""
    card = render_dataset_card(
        repo_id="alexw0/sentinel2-lejepa-test",
        presets=_fake_presets(),
        n_preset_chips=70,
        n_global_chips=30,
        train_size=90,
        val_size=10,
        norm_stats=_fake_stats(),
        build_date="2026-04-04",
    )
    # Dynamic substitutions
    assert "alexw0/sentinel2-lejepa-test" in card
    assert "2026-04-04" in card
    assert "Demo Preset" in card
    assert "sahara_algeria" in card  # global points list rendered
    assert "1500.00" in card         # norm stats mean for red
    assert "600.00" in card          # norm stats std for swir16
    assert "| 70 " in card or "70 " in card  # n_preset_chips
    # No unsubstituted {name} tokens left behind (doubled braces render as single)
    assert "{build_date}" not in card
    assert "{repo_id}" not in card
    assert "{norm_stats_table}" not in card


def test_render_dataset_card_defaults_build_date_to_today():
    """Omitting build_date falls back to today's date."""
    from datetime import date
    card = render_dataset_card(
        repo_id="u/x",
        presets=_fake_presets(),
        n_preset_chips=1,
        n_global_chips=1,
        train_size=2,
        val_size=0,
        norm_stats=_fake_stats(),
    )
    assert date.today().isoformat() in card


def test_render_dataset_card_has_yaml_frontmatter():
    """Card starts with YAML frontmatter so HF Hub tags the dataset."""
    card = render_dataset_card(
        repo_id="u/x",
        presets=_fake_presets(),
        n_preset_chips=1,
        n_global_chips=1,
        train_size=2,
        val_size=0,
        norm_stats=_fake_stats(),
    )
    assert card.startswith("---\n")
    # frontmatter block closes before the H1
    head, _, _ = card[4:].partition("\n---\n")
    assert "license: cc-by-sa-4.0" in head
    assert "sentinel-2" in head


# ── Dataset assembly (requires the `datasets` extra) ─────────────────────────
# These tests are skipped automatically when the experimental extras aren't
# installed, so the pure-helper tests above still run in a minimal env.

datasets_lib = pytest.importorskip("datasets")

from src.experimental.build_dataset import (  # noqa: E402
    chips_to_dataset,
    compute_norm_stats,
    save_dataset_bundle,
    split_train_val,
)


def _make_fake_chip(value: int, source: str = "preset", preset_name: str | None = "test") -> dict:
    """Build a chip record with uniform per-band values for assembly tests."""
    bands = np.full((5, 128, 128), value, dtype=np.uint16)
    return {
        "bands": bands,
        "bbox": (-115.20, 36.10, -115.15, 36.15),
        "acquisition_date": "2023-06-15",
        "scene_id": f"S2A_TEST_{value}",
        "source": source,
        "preset_name": preset_name,
    }


def test_chips_to_dataset_schema():
    """Assembled dataset has the expected typed schema."""
    chips = [_make_fake_chip(1000), _make_fake_chip(2000)]
    ds = chips_to_dataset(chips)
    assert len(ds) == 2
    # Check feature names
    assert set(ds.features.keys()) == {
        "bands", "bbox", "acquisition_date", "scene_id", "source", "preset_name"
    }
    # source is a ClassLabel with two classes
    assert ds.features["source"].names == ["preset", "global"]
    # Bands shape round-trips
    sample = ds[0]
    assert np.asarray(sample["bands"]).shape == (5, 128, 128)


def test_chips_to_dataset_global_preset_name_coerced_to_empty_string():
    """Global chips (preset_name=None) become empty string in the typed field."""
    chips = [_make_fake_chip(1500, source="global", preset_name=None)]
    ds = chips_to_dataset(chips)
    assert ds[0]["preset_name"] == ""
    assert ds[0]["source"] == 1  # ClassLabel index for "global"


def test_chips_to_dataset_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        chips_to_dataset([])


def test_split_train_val_fractions():
    """90/10 split on 20 chips yields 18/2."""
    chips = [_make_fake_chip(i * 100) for i in range(1, 21)]
    ds = chips_to_dataset(chips)
    dd = split_train_val(ds, val_fraction=0.10, seed=42)
    assert set(dd.keys()) == {"train", "validation"}
    assert len(dd["train"]) == 18
    assert len(dd["validation"]) == 2


def test_split_train_val_deterministic_seed():
    """Same seed produces the same split."""
    chips = [_make_fake_chip(i * 100) for i in range(1, 21)]
    ds = chips_to_dataset(chips)
    dd1 = split_train_val(ds, seed=42)
    dd2 = split_train_val(ds, seed=42)
    assert dd1["train"]["scene_id"] == dd2["train"]["scene_id"]


def test_compute_norm_stats_uniform_chips():
    """Chips with uniform value per band yield mean=value, std=0."""
    chips = [_make_fake_chip(1500) for _ in range(5)]
    ds = chips_to_dataset(chips)
    stats = compute_norm_stats(ds)
    assert stats["bands"] == ["red", "green", "blue", "nir", "swir16"]
    assert stats["mean"] == pytest.approx([1500.0] * 5)
    assert stats["std"] == pytest.approx([0.0] * 5, abs=1e-6)
    assert stats["chip_count"] == 5
    assert stats["pixel_count"] == 5 * 128 * 128


def test_compute_norm_stats_varied_chips():
    """Mean across chips with values {1000, 2000} is 1500 per band."""
    chips = [_make_fake_chip(1000), _make_fake_chip(2000)]
    ds = chips_to_dataset(chips)
    stats = compute_norm_stats(ds)
    assert stats["mean"] == pytest.approx([1500.0] * 5)
    # Std of {1000, 2000} with n weighting is 500 per band
    assert stats["std"] == pytest.approx([500.0] * 5, abs=1e-3)


def test_save_dataset_bundle_round_trip(tmp_path):
    """save_dataset_bundle writes a directory that load_from_disk can read."""
    from datasets import load_from_disk

    chips = [_make_fake_chip(i * 100) for i in range(1, 21)]
    ds = chips_to_dataset(chips)
    dd = split_train_val(ds, seed=42)
    stats = compute_norm_stats(dd["train"])

    output_dir = tmp_path / "bundle"
    save_dataset_bundle(dd, stats, output_dir, copy_norm_stats_to_repo=False)

    # Round trip
    loaded = load_from_disk(str(output_dir))
    assert set(loaded.keys()) == {"train", "validation"}
    assert len(loaded["train"]) + len(loaded["validation"]) == 20

    # Norm stats file written alongside
    norm_path = output_dir / "norm_stats.json"
    assert norm_path.exists()
    import json as _json
    with open(norm_path) as f:
        saved = _json.load(f)
    assert saved["mean"] == stats["mean"]
    assert saved["std"] == stats["std"]
