# Sharper LeJEPA Feature Maps — DINOv2-Style Visualizations

**Status:** Design. Not yet implemented.
**Author:** Alex + Claude
**Date:** 2026-04-04
**Prerequisite:** `feat/lejepa-experimental` branch (ViT-Tiny/8, 128×128, 1200 steps) already published at `alexw0/lejepa-vit-tiny-patch8-sentinel2-5band`.

---

## Goal

Move the experimental foundation-model panel from "clearly better than ResNet-18" to "visually comparable to DINOv2 feature maps on EO imagery". The user's acceptance criterion is visual: PCA→RGB panels should look like smoothly-varying semantic colorings, not blocky upsampled grids.

## Root cause of the current "blocky" look

The current gold path produces a `[192, 16, 16]` feature map per 128×128 chip: **256 spatial positions, bilinear-upsampled to ~250 px for display**. Each visible block in the screenshot is one patch token. This is an architectural ceiling — no amount of additional training or data changes the block size. Evidence:

- Predictive loss has already converged (0.001 at step 1200).
- Features are non-degenerate (clear color separation, coherent spatial structure, no collapse).
- The block boundaries in the image are at token boundaries, not noise boundaries.

**Neither more data nor a bigger model at the same input/patch combo will sharpen this.** The only thing that sharpens it is *more tokens per image*, which means one of:

1. **Higher input resolution** at the same patch size (e.g. 256×256 with patch 8 → 32×32 = 1024 positions)
2. **Smaller patch size** at the same input (e.g. 128×128 with patch 4 → 32×32 = 1024 positions)

Both give the same token count and similar attention compute. Higher input resolution is strictly preferred because it also gives the encoder more *pixel-level* signal per chip, not just a finer tokenization of the same 128² pixels. Smaller patches on 128² is just a rearrangement of the same information.

Parallel second axis: **richer features per token** via ViT-Small (384-dim) over ViT-Tiny (192-dim). This doesn't move the block boundary but makes the per-token feature more semantically informative, which improves PCA color separation. Free to combine with axis #1.

## Proposed configuration

**Target architecture:** ViT-Small/8 at 256×256 input with 4 register tokens.

| Spec             | Current (Tiny/8 @ 128) | Proposed (Small/8 @ 256) | Multiplier |
|------------------|------------------------|--------------------------|-----------|
| Input resolution | 128 × 128              | 256 × 256                | 4× pixels |
| Patch size       | 8                      | 8                        | —         |
| Feature grid     | 16 × 16 (256 tokens)   | 32 × 32 (1024 tokens)    | 4× tokens |
| Embed dim        | 192                    | 384                      | 2× width  |
| Params           | ~5.5M                  | ~22M                     | 4× params |
| Attention cost   | O(256²·192)            | O(1024²·384)             | ~32×      |
| M1-trainable?    | Yes (CPU, slow)        | No — GPU only            | —         |

The 32× attention cost jump is the real budget story. On an L4 or A10G at batch 32, this fits and trains in a few hours for 50 epochs — this is still a single lightning.ai session.

## What needs to change

### 1. Encoder factory — `src/experimental/encoders.py`

Plumb `img_size` through:

- `FiveChannelViTPatch8.__init__` already accepts `img_size` as a parameter (`src/experimental/encoders.py:101`). timm's `VisionTransformer` auto-sizes the positional embedding from it. No change needed in the class itself.
- `build_encoder(kind)` currently hardcodes `img_size=128` implicitly (default). Add an `img_size` keyword:

```python
def build_encoder(kind: EncoderKind, *, img_size: int = 128) -> nn.Module:
    if kind == "resnet18":
        return FiveChannelResNet18(in_channels=5)  # grid is fixed
    if kind == "vit_tiny_patch8":
        return FiveChannelViTPatch8(variant="tiny", img_size=img_size)
    if kind == "vit_small_patch8":
        return FiveChannelViTPatch8(variant="small", img_size=img_size)
    ...
```

Callers (`train_lejepa`, `inference`, `upload_model`) that want a non-default grid pass `img_size=256`. Old 128-input checkpoints still instantiate cleanly with the default.

### 2. Training — `src/experimental/train_lejepa.py`

- `TrainConfig` gains `img_size: int = 128` (default preserves current behavior).
- CLI: `--img-size 256` flag.
- `build_encoder(cfg.encoder_kind, img_size=cfg.img_size)` at both online and target encoder construction sites.
- `checkpoint["config"]["img_size"]` gets written out so inference can round-trip.
- Mask schedule (`default_mask_schedule`) already auto-scales from `n_positions`: at 1024 positions it picks (640, 256) for context/target. Verify this produces enough gradient signal; if the predictor struggles with 256 target positions per step, reduce to (512, 128).
- **Batch size will drop.** At 256 input + Small variant, effective batch fits around 32 on a 24GB L4. Explicit `--batch-size 32` in the launch script. Loss will be noisier; gradient accumulation (new optional flag, 4 steps) is a safety valve.

### 3. Inference — `src/experimental/inference.py`

- `_CHIP_SIZE` constant must become dynamic. Read `img_size` from the checkpoint config in `load_model_cached` and store it on the returned dict:

```python
chip_size = ckpt.get("config", {}).get("img_size", 128)
return {"encoder": ..., "mean": ..., "std": ..., "kind": kind,
        "chip_size": chip_size, "source": source}
```

- `extract_features` replaces `_CHIP_SIZE` with `model["chip_size"]`. `_center_crop_or_pad` already takes an explicit `size` argument — just thread it through.
- Everything downstream (`features_to_rgb`, `features_to_change_map`) is already shape-agnostic. The existing `display_size` upsample handles 32×32 → 256 cleanly.
- **One caveat for inference tile coverage.** At 256×256 × 10m/px the encoder sees a 2.56 km tile. Most app presets cover 5–20 km, so a single center-crop still captures less of the AOI than desired. This is the same PoC limitation we have today, just at double the footprint. Document it; don't fix it (multi-tile inference is explicitly out of scope for the PoC per the original plan).

### 4. Dataset rebuild — `src/experimental/build_dataset.py` — **the real work**

This is where the "1–2 hours" estimate I gave earlier was wrong. The code change is trivial (`CHIP_SIZE: int = 256`), but the consequences aren't:

- **Chip yield drops ~4×.** A 10 km preset AOI at 128 px tiles gives ~6400 non-overlapping chips; at 256 px tiles it gives ~1600. To maintain ~3000 training chips the expansion is one of:
  - Grow `PRESET_AOI_SIZE_KM` from 10 to 16–20 km (risks straying from the demo's change region, reducing preset bias).
  - Lower the stride below tile size (overlapping chips — dilutes the "independent samples" assumption SIGReg depends on).
  - Accept fewer total chips and lean on more epochs instead (probably fine; 50 epochs × 1600 chips is still 80k forward passes).
- **Global chips at 2 km bboxes don't fit a 256-px tile.** The hand-curated global-diversity points in the current builder use ~2 km boxes, which at 10 m/px is ~200 px — smaller than one 256 chip. Either expand all global bboxes to ≥3 km, or drop the global chips entirely for the 256 run and accept pure preset-biased training.
- **Republish as a new HF dataset.** `alexw0/sentinel2-lejepa-preset-biased-small-256` (or similar). The existing 128 dataset stays published for reproducibility of the current ViT-Tiny checkpoint.
- **Re-run norm stats computation.** Per-band mean/std are pixel-level statistics, so they should be numerically very close to the current values — but recomputing them on the new chip set is the honest path, and the values get baked into the new checkpoint anyway.

### 5. Unit tests — `tests/unit/test_train_lejepa.py`, `tests/unit/test_inference.py`

Existing tests assert shape `(192, 16, 16)` or `(2, 192, 16, 16)`. Make them parameterize over `img_size`:

```python
@pytest.mark.parametrize("img_size,grid", [(128, 16), (256, 32)])
def test_vit_feature_grid_scales_with_img_size(img_size, grid): ...
```

No new test philosophy — just widen coverage so the 256 path is exercised in CI alongside 128.

### 6. Docs — `src/experimental/README.md`, root `README.md`

- README architecture note: update the "16×16 = 256 patch-token grid" phrasing to "16×16 (Tiny @ 128) or 32×32 (Small @ 256)".
- Reproduction steps: add a third training invocation for the 256 path.
- Model card template: already parameterized on `embed_dim`/`grid_side` from the live encoder — no template edits needed; the values will be read off the new encoder automatically.

### 7. Uploader — `src/experimental/upload_model.py`

- `_ARCH_METADATA` is already keyed by `encoder_kind`. The 256 run still uses `vit_small_patch8` as its kind. Add a note to the `vit_small_patch8` entry mentioning the 256 input if present.
- `checkpoint_filename` derives from `encoder_kind` only — *not* from `img_size*. To disambiguate 128 vs 256 Small checkpoints in the hub, either:
  - Embed `img_size` in the filename: `lejepa_vit_small_patch8_256_5band.pt`.
  - Or publish them to *different* hub repo IDs (`-small` vs `-small-256`).

  Different repo IDs are cleaner (one repo, one model card, one set of norm stats, one story per artifact). Go with that. `checkpoint_filename` stays as-is.

## Phase breakdown

Each phase ends in a shippable state. Phases 1–2 run on M1; Phase 3 is the only GPU phase.

1. **Code plumbing for `img_size`** — all the file edits in sections 1-3 above, plus widened tests. Verifiable on M1 with the existing 128 dataset (train 1 smoke epoch at img_size=128, confirm identical behavior to today). ~half a day of focused work.
2. **Dataset rebuild at 256** — run `build_dataset.py` with `CHIP_SIZE=256` and expanded AOI sizing. Publish to HF as a new dataset repo. Spot-check 10 random chips by plotting RGB. Verify `datasets.load_dataset` round-trip. This is the step that spends real time — it's I/O bound on COG reads, not CPU, and reuses the existing band cache as much as possible.
3. **Train on lightning.ai** — one session, ViT-Small/8 @ 256, 50 epochs, batch 32, `--wandb-project sentinel-change-lejepa`. Expected wall clock 2–4 hours on L4. Target predictive loss <0.005 and stable SIGReg.
4. **Upload + verify in app** — new HF model repo, then toggle the "Experimental" panel and compare side-by-side with the current 128 Tiny checkpoint. Subjective visual check is the deciding factor. If 32×32 with richer features does not look visibly sharper, stop and investigate before further scaling.

## Risks and open questions

- **Preset bias at 256.** Bigger tiles eat into the margin of preset AOIs. A change region that was centered in a 128 chip might straddle the edge of a 256 chip. Mitigation: when AOI expansion is applied, ensure the original preset bbox stays within the central ~60% of the tile grid.
- **Gradient signal with 256 target positions.** Predictor MLP is a 2-layer head with position embeddings; predicting 256 positions per step from 640 context positions is ~4× the work per forward pass. Target encoder EMA momentum may need to ramp faster. Smoke-test on M1 with 1024-position grid before the GPU run, even if the smoke test is painfully slow.
- **HF upload bandwidth.** The 256 Small checkpoint is ~90 MB (vs ~22 MB for Tiny @ 128). Still trivial, just noting it.
- **No guaranteed visual improvement.** If the encoder is undertrained at 22M params on ~1600 chips, the features may be noisier than the current Tiny despite the finer grid. The honest fallback is "Tiny @ 256" — same model, 4× tokens, still M1-loadable at inference. Include this as a branch point in Phase 3 if Small @ 256 disappoints.
- **Pipeline end-to-end failure modes.** The new one-shot `lightning_train.sh` (see below) runs dataset pull → smoke → full train → upload in sequence. If training fails mid-run, the upload step should not execute. This is handled by `set -euo pipefail` and `wait $TRAIN_PID` with exit-code propagation — documented in the script comment header.

## Pipeline automation

`src/experimental/lightning_train.sh` is being updated as part of this design so the lightning.ai session becomes single-command: **pull dataset → smoke test → full training → publish checkpoint to HF**. Parameters (encoder, img_size, epochs, batch size, dataset repo, model repo id) are hoisted to environment variables at the top of the script so the same file orchestrates both the current Tiny @ 128 run and the future Small @ 256 run without editing the body.

The script preserves the current pattern of running training under `nohup` with `train.log` for `tail -f`, but now blocks on `wait ${TRAIN_PID}` before running the upload step. If the training process exits non-zero, `pipefail` propagates the failure and the upload is skipped.

Exact script diff lives in the commit that accompanies this design doc.

## Out of scope for this design

- Multi-tile inference for large AOIs. Still a future direction, same as before.
- Fine-tuning on labeled change-detection data.
- Comparison benchmarks against Clay, Prithvi, SSL4EO-S12.
- Patch 4 path (equivalent outcome to 256 input, more complex to implement, no upside).
- Feature-attention click-to-inspect UI.

## Recommendation

Do **Phase 1 only** first — the plumbing work — and verify the existing 128 Tiny checkpoint still trains and loads identically through the new parameterized code path. That unblocks the path without spending the dataset rebuild budget. Then pause and look at the 128 output one more time with fresh eyes: if the user's interview narrative is served well enough by the current output, *stop*. If they want to push for the sharper demo, Phases 2–4 are the real work.
