---
language:
- en
license: cc-by-sa-4.0
pretty_name: Sentinel-2 LeJEPA Preset-Biased (Small)
tags:
- earth-observation
- sentinel-2
- self-supervised-learning
- satellite-imagery
- pretraining
- remote-sensing
size_categories:
- 1K<n<10K
task_categories:
- image-feature-extraction
---

# Sentinel-2 LeJEPA Preset-Biased (Small)

A small, preset-biased Sentinel-2 L2A chip dataset curated for self-supervised
pretraining of a [LeJEPA](https://arxiv.org/abs/2511.08544) ResNet-18 encoder.
Built as a reproducibility artifact for the
[Sentinel Change Explorer](https://github.com/falafel-hockey/sentinel-change-explorer)
proof-of-concept foundation-model change-detection feature.

**This is a proof of concept, not a general-purpose EO pretraining corpus.** It
is intentionally tiny (~thousands of chips) and biased toward the five demo
AOIs the Sentinel Change Explorer app highlights. Use it to reproduce that
specific PoC, not as a substitute for SSL4EO-S12, Clay, or Prithvi.

## Dataset snapshot

| Field               | Value                            |
|---------------------|----------------------------------|
| Build date          | {build_date}                     |
| Total chips         | {total_chips}                    |
| Preset chips (~70%) | {n_preset_chips}                 |
| Global chips (~30%) | {n_global_chips}                 |
| Train split         | {train_size}                     |
| Validation split    | {val_size}                       |
| Chip size           | 128 x 128 px @ 10 m/px (1.28 km) |
| Bands               | red, green, blue, nir, swir16    |
| Dtype               | uint16 (raw L2A reflectance)     |

## Sampling methodology

Chips are drawn from two sources in roughly a 70/30 mix:

1. **Preset AOIs (~70%).** For each of the 5 demo presets in the Sentinel
   Change Explorer app, the builder expands the tight demo bbox into a 10 km
   square centered on the preset's centroid, searches STAC (Element84 Earth
   Search v1) for Sentinel-2 L2A scenes in both `before_range` and
   `after_range`, loads the 5 reflectance bands + SCL via the same
   `src.sentinel.load_bands` the app uses, and tile-crops into non-overlapping
   128x128 chips.
2. **Global diversity points (~30%).** A hand-curated list of 30 globally
   diverse points (deserts, forests, croplands, urban cores, coasts, ice,
   wetlands) across every inhabited continent, each sampled at 2-3 dates
   spread across seasons. Same fetch-and-tile flow with a 5.12 km AOI.

### Rejection filters

Every candidate chip is tested against two filters and dropped if it fails
either:

- **Cloud/shadow fraction > 25%**, computed from the Sentinel-2 Scene
  Classification Layer (SCL classes 3, 8, 9, 10).
- **Fill fraction > 10%**, defined as pixels where all 5 reflectance bands
  equal zero (true no-data, not just a single dark band).

### Preset AOIs

{preset_aoi_list}

### Global diversity points

{global_points_list}

## Schema

Each row is:

```
{{
    "bands": Array3D(shape=(5, 128, 128), dtype=uint16),
    "bbox": Sequence(float32, length=4),        # (west, south, east, north) WGS84
    "acquisition_date": Value(string),          # ISO date of the source scene
    "scene_id": Value(string),                  # STAC item id
    "source": ClassLabel(names=["preset", "global"]),
    "preset_name": Value(string),               # "" for global chips
}}
```

## Normalization stats

Per-band mean and standard deviation computed over the **training split**
(uint16 reflectance, before any scaling):

| Band   | Mean       | Std        |
|--------|------------|------------|
{norm_stats_table}

These are also shipped as `norm_stats.json` in the dataset bundle. The
matching LeJEPA model repo embeds a copy so inference doesn't need to pull
the dataset.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds)
# DatasetDict with "train" and "validation" splits

sample = ds["train"][0]
print(sample["bands"].shape)   # (5, 128, 128)
print(sample["source"])        # 0 = preset, 1 = global
```

The companion pretrained LeJEPA ResNet-18 (5-band) is published separately
and consumes these chips at native resolution without further resizing.

## Limitations

- **Tiny scale.** Thousands of chips, not millions. A real SSL corpus for
  remote sensing is 2-3 orders of magnitude larger. Expect the resulting
  features to overfit to the sampled AOIs and date windows.
- **Preset bias by design.** 70% of chips come from 5 specific locations
  chosen because they are the demo AOIs in the companion app. This is
  intentional for the PoC but makes the features a poor fit for
  general-purpose EO tasks.
- **Single sensor, single level.** Sentinel-2 L2A only. No Sentinel-1, no
  Landsat, no other modalities.
- **5 bands only.** B02, B03, B04, B08, B11. The red-edge, cirrus, and SWIR22
  bands are intentionally excluded to keep the model compact for M1
  inference.
- **No deduplication across dates.** Chips from the same AOI across different
  acquisition dates are both kept. This is a feature for temporal-invariance
  pretraining, but means chips are not i.i.d.

## License and attribution

- Chips are released under **CC-BY-SA-4.0**, matching Copernicus Sentinel
  data's terms for derived products.
- **Contains modified Copernicus Sentinel data [2023-{build_year}], ESA.**
  Source imagery: Sentinel-2 L2A via [Element84 Earth Search v1](https://registry.opendata.aws/sentinel-2-l2a-cogs/).

## Citation

```bibtex
@misc{{sentinel2_lejepa_preset_biased_small,
  title  = {{Sentinel-2 LeJEPA Preset-Biased (Small)}},
  author = {{Wheelis, Alex}},
  year   = {{{build_year}}},
  url    = {{https://huggingface.co/datasets/{repo_id}}}
}}

@misc{{balestriero2025lejepa,
  title  = {{LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}},
  author = {{Balestriero, Randall and LeCun, Yann}},
  year   = {{2025}},
  eprint = {{2511.08544}},
  archivePrefix = {{arXiv}}
}}
```
