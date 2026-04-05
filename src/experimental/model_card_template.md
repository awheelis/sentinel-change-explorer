---
language:
- en
license: cc-by-sa-4.0
library_name: pytorch
pipeline_tag: image-feature-extraction
tags:
- earth-observation
- sentinel-2
- self-supervised-learning
- lejepa
- jepa
- vision-transformer
- satellite-imagery
- feature-extraction
- pretraining
datasets:
- {dataset_repo_id}
---

# LeJEPA {architecture_name} (5-band Sentinel-2, PoC)

A small self-supervised feature extractor for Sentinel-2 L2A imagery: a
5-channel {architecture_name} pretrained with the
[LeJEPA](https://arxiv.org/abs/2511.08544) objective (Balestriero & LeCun,
2025) on the
[{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}) chip
dataset. Built as the end-to-end reproducibility artifact for the
[Sentinel Change Explorer](https://github.com/awheelis/sentinel-change-explorer)
foundation-model change-detection proof of concept.

**This is a proof of concept, not a general-purpose EO model.** The training
set is tiny, the training budget is small, and the features are only as
strong as ~{n_steps} gradient steps on ~{train_chips} chips can make them.
Use it to reproduce the companion app's "Experimental" panel, not as a
substitute for Clay, Prithvi, or SSL4EO-S12 models.

## Architecture

| Component       | Details                                                 |
|-----------------|---------------------------------------------------------|
| Encoder kind    | `{encoder_kind}`                                        |
| Backbone        | {architecture_description}                              |
| Input           | `[B, 5, 128, 128]` uint16-derived reflectance (10 m/px) |
| Output          | `[B, {embed_dim}, {grid_side}, {grid_side}]` feature map |
| Feature grid    | {grid_side}x{grid_side} = {n_positions} positions       |
| Parameter count | ~{encoder_params_m:.1f}M                                |
| Band order      | red, green, blue, nir, swir16 (S2 B02/B03/B04/B08/B11)  |

{architecture_note}

A small MLP predictor head is used during training and is **not** included
in this release; only the encoder is.

## Training recipe

| Hyperparameter        | Value            |
|-----------------------|------------------|
| Objective             | LeJEPA (predictive smooth-L1 + SIGReg) |
| Mask scheme           | {grid_side}x{grid_side} feature grid, disjoint context/target subsets |
| Context aggregation   | mean-pool over visible positions |
| Target encoder        | EMA of online, momentum ramp 0.996 → 1.0 |
| Optimizer             | AdamW (lr={lr}, weight_decay=0.05) |
| LR schedule           | Cosine annealing over {n_steps} steps |
| SIGReg weight (α)     | {alpha_sigreg} |
| Batch size            | {batch_size} |
| Epochs                | {epochs} |
| Training chips        | {train_chips} |
| Precision             | {precision} |
| Device                | {device} |
| Final loss            | {final_loss:.4f} |
| Training date         | {train_date} |

### Normalization

The encoder expects inputs normalized with the per-band statistics computed
over the training split of the companion dataset. These stats ship with the
checkpoint (`.pt` file key `norm_stats`) so inference does not need to pull
the dataset:

| Band   | Mean      | Std       |
|--------|-----------|-----------|
{norm_stats_table}

## Intended use

- **Yes:** feature extraction for small-area Sentinel-2 L2A tiles (one or a
  few 128x128 chips), PCA→RGB visualization, per-patch cosine-distance
  change detection against another time step of the same AOI.
- **No:** high-recall general EO feature extraction, downstream
  classification without fine-tuning, deployment to workloads where feature
  quality matters.

## Limitations

- **Tiny training budget.** This PoC was trained on the hardware noted
  above; a full GPU run is a planned follow-up that will swap the weights
  without any API change.
- **Preset bias.** 70% of training chips came from 5 specific demo AOIs in
  the companion app, so the encoder is best on geography resembling those
  (coastal/tropical, desert-construction, central-European industrial,
  agricultural-flood-plain, forested-burn).
- **Single sensor, single level.** Sentinel-2 L2A only. No S1 SAR, no
  Landsat, no atmospheric TOA inputs.
- **5 bands only.** Red-edge, cirrus, and SWIR22 are intentionally
  excluded to keep the model compact for CPU inference.
- **No downstream supervised head.** Use `mean(dim=[2,3])` on the feature
  map to get a global descriptor, or operate on the raw spatial map.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="{hub_filename}",
)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# The encoder factory lives in the companion repo:
#   git clone https://github.com/awheelis/sentinel-change-explorer
#   from src.experimental.encoders import build_encoder
from src.experimental.encoders import build_encoder

encoder = build_encoder(ckpt["config"]["encoder_kind"])
encoder.load_state_dict(ckpt["encoder_state"])
encoder.eval()

# Normalize with the stats packed into the checkpoint
norm = ckpt["norm_stats"]
mean = torch.tensor(norm["mean"]).view(1, 5, 1, 1)
std = torch.tensor(norm["std"]).view(1, 5, 1, 1).clamp(min=1.0)

# bands: [B, 5, 128, 128] float tensor in raw uint16-derived reflectance
# feat:  [B, {embed_dim}, {grid_side}, {grid_side}]
with torch.no_grad():
    x = (bands - mean) / std
    feat = encoder(x)
```

## License and attribution

- Model weights released under **CC-BY-SA-4.0**.
- Training data: [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}).
- **Contains modified Copernicus Sentinel data [2023-{build_year}], ESA.**

## Citation

```bibtex
@misc{{lejepa_{encoder_kind}_sentinel2_5band,
  title  = {{LeJEPA {architecture_name} (5-band Sentinel-2, PoC)}},
  author = {{Wheelis, Alex}},
  year   = {{{build_year}}},
  url    = {{https://huggingface.co/{repo_id}}}
}}

@misc{{balestriero2025lejepa,
  title  = {{LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}},
  author = {{Balestriero, Randall and LeCun, Yann}},
  year   = {{2025}},
  eprint = {{2511.08544}},
  archivePrefix = {{arXiv}}
}}

@misc{{darcet2024registers,
  title  = {{Vision Transformers Need Registers}},
  author = {{Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr}},
  year   = {{2024}},
  eprint = {{2309.16588}},
  archivePrefix = {{arXiv}}
}}
```
