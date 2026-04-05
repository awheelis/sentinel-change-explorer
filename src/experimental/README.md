# Experimental: LeJEPA Foundation-Model Change Detection

An honest, end-to-end proof-of-concept that applies a small self-supervised
vision model (ResNet-18 pretrained with the
[LeJEPA](https://arxiv.org/abs/2511.08544) objective) to the same Sentinel-2
tiles the main app already loads. It produces PCA→RGB feature
visualizations and a learned per-patch change heatmap shown side-by-side
with the existing NDVI-delta map.

**This is a proof of concept, not a SOTA system.** The model is tiny
(~11M params), the training set is tiny (~5k–8k chips biased toward the
app's preset AOIs), and the main goal is to demonstrate the full pipeline:
dataset curation → publishing → SSL pretraining → model publishing →
inference UI.

## Hardware notes

| Stage                  | Where it runs             | Why                                     |
|------------------------|---------------------------|------------------------------------------|
| Dataset build (Phase 2)| M1 Mac 8GB                | I/O-bound, reuses existing COG cache     |
| Training (Phase 4)     | M1 CPU (small) / GPU (real)| CPU is fine for <1k chips smoke runs; real runs need CUDA. MPS is intentionally avoided |
| Inference (Phase 6)    | M1 Mac 8GB, CPU           | Single tile, ~20-50 ms, fits easily      |

The repo currently ships a small **proof-of-concept checkpoint** trained on
M1 CPU for fast iteration. A real run on lightning.ai GPU is a planned
follow-up that swaps the weights without any API changes.

## Install

```bash
uv sync --extra experimental
huggingface-cli login   # write-token for publishing, read-token for pulling
```

### HuggingFace token setup

Publishing the dataset (Phase 3) and the trained model (Phase 5) both
require a HuggingFace account and an access token with **write** scope:

1. Create an account at https://huggingface.co if you don't have one.
2. Go to https://huggingface.co/settings/tokens and create a token with
   the `write` role. A fine-grained token scoped to your own namespace
   with `repo.write` permission is sufficient and safer than a broad
   `write` token.
3. Run `huggingface-cli login` and paste the token. It gets stored in
   `~/.cache/huggingface/token` and is picked up automatically by both
   `datasets.push_to_hub` and `huggingface_hub.HfApi`.
4. Confirm with `huggingface-cli whoami` — you should see your username.

Read-only pulls (running inference in Phase 6 against the published
model) work with an unauthenticated token or a `read`-scoped token; only
publishing needs `write`.

After install, toggle the "Experimental: Foundation Model" checkbox in
the app sidebar. If the extras aren't installed the panel shows an
install hint instead of trying to import torch.

## Reproduction steps

1. **Build + publish dataset** (Phase 2-3):
   ```bash
   uv run python -m src.experimental.build_dataset \
       --push-to-hub <user>/sentinel2-lejepa-preset-biased-small
   ```
   The published dataset for this repo lives at
   [alexw0/sentinel2-lejepa-preset-biased-small](https://huggingface.co/datasets/alexw0/sentinel2-lejepa-preset-biased-small).

2. **Train** (Phase 4). For a quick CPU-only sanity run:
   ```bash
   uv run python -m src.experimental.train_lejepa \
       --dataset cache/lejepa_dataset \
       --limit-train-chips 500 --batch-size 16 --epochs 15 \
       --wandb-project sentinel-change-lejepa
   ```
   For the real run on lightning.ai (once wired to hub-dataset loading):
   ```bash
   python -m src.experimental.train_lejepa \
       --dataset <user>/sentinel2-lejepa-preset-biased-small \
       --epochs 50 --batch-size 128
   ```

3. **Publish model** (Phase 5):
   ```bash
   uv run python -m src.experimental.upload_model \
       --checkpoint checkpoints/lejepa_resnet18_5band.pt \
       --repo-id <user>/lejepa-resnet18-sentinel2-5band
   ```

4. **Run inference** (Phase 6): open the app, toggle the sidebar checkbox.
   The panel downloads the model from HF on first render and caches it with
   `st.cache_resource`.

### Training telemetry

Pass `--wandb-project <name>` to stream per-step `loss/total`,
`loss/predictive`, `loss/sigreg`, `lr`, and `ema_momentum` metrics to
Weights & Biases. Authenticate once with `wandb login` before the first run.

## Status

Implementation proceeds in 7 phases. Current status lives in
`PRIORITIZED_BACKLOG.md` item #8. Each phase ends in a shippable state.

## References

- Balestriero & LeCun (2025). *LeJEPA: Provable and Scalable Self-Supervised
  Learning Without the Heuristics.* arXiv:2511.08544.
- Maes et al. (2026). *LeWorldModel: A Minimal Latent World Model.*
  arXiv:2603.19312.
- Contains modified Copernicus Sentinel data [2024–2026], ESA.
