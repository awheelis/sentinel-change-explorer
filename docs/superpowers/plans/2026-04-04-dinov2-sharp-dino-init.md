# DINOv2-Sharp LeJEPA with DINO Pretrained Init — Implementation Plan

> **Context:** This plan supersedes the direction of `docs/superpowers/specs/2026-04-04-dinov2-sharp-features-design.md` in two major ways: (1) initialize ViT-Small/8 from the actual DINO paper weights (`vit_small_patch8_224.dino` in timm) instead of training from scratch, and (2) train on a globally-diverse Sentinel-2 dataset rather than a preset-biased one. Both changes were motivated by a 1-hour H100 wall-clock budget that makes from-scratch convergence risky.

**Goal:** Produce a LeJEPA ViT-Small/8 checkpoint whose PCA→RGB feature visualizations look substantively like DINO-on-EO, within ~1 hour of H100 time.

**Architecture:** Initialize a 5-band ViT-Small/8 with 4 register tokens from DINO's ImageNet weights (transformer blocks, norms, pos-embed-interpolated-to-32×32, RGB stem copied + NIR/SWIR from mean-of-RGB). Fine-tune with LeJEPA self-supervised objective on ~5k globally-stratified Sentinel-2 chips at 256×256 input on an H100 for ~150 epochs.

**Tech Stack:** PyTorch, timm (DINO weights + ViT backbone), HuggingFace Datasets (dataset publish), planetary computer STAC (global S2 scene query), W&B (metrics), lightning.ai (H100).

---

## Key decisions recorded

1. **DINO init over from-scratch.** Verified present: `timm.create_model('vit_small_patch8_224.dino', pretrained=True)`. Loaded weights have `patch_embed.proj.weight: [384, 3, 8, 8]`, `pos_embed: [1, 785, 384]` (1 CLS + 784 = 28×28 patches at 224), `num_prefix_tokens: 1` (CLS only, NO registers).
2. **Architecture mismatch handling.** Our target encoder has `class_token=False, reg_tokens=4`. We load DINO into it by: copying all transformer blocks + final norm + patch-embed conv (with 3→5 band expansion), interpolating pos embed 28×28→32×32 via bicubic, **discarding DINO's CLS token**, leaving the 4 register tokens at timm's default random init. Transformer attention is order-invariant over the prefix-to-patch interface after position embeddings, so register tokens learn their role during fine-tuning without disrupting the pretrained patch representations.
3. **5-band stem adaptation.** RGB channel weights copy 1:1 into (R, G, B) slots; NIR and SWIR16 channels initialize from `mean(RGB_weights)` across the input-channel axis. Standard EO adaptation from Clay / SatMAE / Prithvi.
4. **Global diversity over preset bias.** Since DINO-init already encodes "what the world looks like" from ImageNet, the fine-tuning data should maximize Sentinel-2 *spectral* diversity (biomes, seasons), not memorize demo regions. 50 globally stratified scenes × ~100 chips each ≈ 5k chips.
5. **256×256 input, not 384.** 384 is ~2× more attention compute and risks blowing the 1-hour budget. 32×32 token grid (256/8) is already 4× the current 16×16 — visually the key jump.
6. **Batch 192 on H100 80GB, 150 epochs, cosine LR.** Fine-tuning converges much faster than from-scratch; 150 epochs is defensive. Cosine schedule already in `train_lejepa.py`.

## File-level plan

### `src/experimental/encoders.py` — Phase 1a + 1b

**Changes:**

1. `FiveChannelViTPatch8.__init__` already accepts `img_size`; no change.
2. Add new function `load_dino_weights_into_vit(model: FiveChannelViTPatch8) -> None` that:
   - Calls `timm.create_model('vit_small_patch8_224.dino', pretrained=True, num_classes=0)` to get the DINO state dict.
   - Builds a new state dict mapping into our `model.vit`:
     - `patch_embed.proj.weight`: expand `[384, 3, 8, 8]` → `[384, 5, 8, 8]`. Channels 0,1,2 = RGB copy; channels 3,4 = mean of RGB across dim=1, broadcast.
     - `patch_embed.proj.bias`: copy verbatim.
     - `pos_embed`: DINO is `[1, 1+784, 384]` (CLS + 28×28). Drop the CLS row. Reshape remaining to `[1, 28, 28, 384]`, bicubic-interpolate spatial dims to `[1, 32, 32, 384]`, flatten back to `[1, 1024, 384]`. Our model's `pos_embed` shape is `[1, 1024, 384]` since register tokens use a separate `reg_token` parameter in timm.
     - All `blocks.*` (attention, mlp, norms): copy verbatim.
     - `norm.*`: copy verbatim.
     - Skip DINO's `cls_token` (we don't have one).
     - Leave our `reg_token` at its default timm init.
   - `model.vit.load_state_dict(adapted, strict=False)` — strict=False to permit our `reg_token` not being in the DINO dict, and to surface any unexpected shape mismatches loudly.
   - Log summary: "loaded N / M parameters from DINO".
3. Extend `build_encoder` signature to `build_encoder(kind: EncoderKind, *, img_size: int = 128, pretrained: bool = False) -> nn.Module`. When `kind == "vit_small_patch8"` and `pretrained=True`, call `load_dino_weights_into_vit` after construction. Raise a clear error if `pretrained=True` with any other kind (DINO weights are Small-specific).

**Why this shape for `pos_embed`:** Verify empirically after the change by running `uv run python -c "from src.experimental.encoders import FiveChannelViTPatch8; m = FiveChannelViTPatch8(variant='small', img_size=256); print(m.vit.pos_embed.shape)"`. If timm stores registers inside `pos_embed` (some versions do), adapt the slicing accordingly.

### `src/experimental/train_lejepa.py` — Phase 1a

**Changes:**

1. Add to `TrainConfig`:
   ```python
   img_size: int = 128
   pretrained: bool = False  # Load DINO init for ViT-Small/8
   ```
2. At encoder construction, pass them through:
   ```python
   online = build_encoder(cfg.encoder_kind, img_size=cfg.img_size, pretrained=cfg.pretrained).to(device)
   target_enc = build_encoder(cfg.encoder_kind, img_size=cfg.img_size, pretrained=cfg.pretrained).to(device)
   target_enc.load_state_dict(online.state_dict())
   ```
   (Target must load the same weights as online — the pretrained flag is idempotent since `load_state_dict` then overwrites.)
3. CLI: add `--img-size` (int, default 128) and `--pretrained` (flag, default False).
4. `save_checkpoint` already serializes `asdict(cfg)` which includes the new fields automatically. Good.

### `src/experimental/inference.py` — Phase 1a

**Changes:**

1. Remove the `_CHIP_SIZE: int = 128` module constant (or keep as fallback default).
2. In `load_model_cached`, read `chip_size = ckpt.get("config", {}).get("img_size", 128)`. Add it to the returned dict: `{"encoder": ..., ..., "chip_size": chip_size}`.
3. Also read `img_size` and pass it into `build_encoder(kind, img_size=chip_size)` so old Tiny @ 128 checkpoints still load unchanged.
4. `extract_features` uses `model["chip_size"]` instead of `_CHIP_SIZE` when calling `_center_crop_or_pad`.
5. Add the new checkpoint to `_LOCAL_CANDIDATES` in priority order: `checkpoints/lejepa_vit_small_patch8_256_5band.pt` (new top), keep the existing entries below.
6. `DEFAULT_REPO_ID` stays — we'll publish the new model to a *different* repo id (`falafel-hockey/lejepa-vit-small-patch8-256-sentinel2-5band`), which the hub-fallback branch will pick up via a new env var or default change. Actually — simpler: leave the default Tiny, add a second sidebar toggle later if we want hub A/B. For this PoC run, local checkpoint takes precedence so the app automatically prefers the new one after the upload syncs to the dev machine.

### `src/experimental/build_dataset.py` — Phase 2a

**Changes:**

1. Replace the hardcoded `CHIP_SIZE: int = 128` with a parameter. Add a module-level `DEFAULT_CHIP_SIZE: int = 128` and thread `chip_size` through `tile_crop`, `chips_from_preset`, the preset/global loops, and the `build` orchestrator. Callers that don't pass it get the default 128 → no regression for the existing pipeline.
2. Add a new top-level function `build_global_dataset(chip_size: int, output_dir: Path, push_to_hub: str | None, private: bool, target_chips: int = 5000) -> Path`. Body:
   - Define `GLOBAL_STRATIFIED_SCENES`: a list of ~50 `(name, lat, lon, date_range, biome)` tuples stratified across Köppen climate zones (tropical rainforest, savanna, desert, mediterranean, temperate forest, boreal, tundra, highland, coastal, urban). I'll hand-curate this list with a mix of well-known landmarks and diverse biomes so we get visually interesting features (Amazon, Sahara, Siberian taiga, Himalayas, Great Barrier Reef coast, Tokyo urban, Iowa cropland, Alps, Svalbard, Andes altiplano, Nile delta, etc.).
   - For each scene: query S2 L2A via existing `src.sentinel.search_scenes`, pick the first cloud-free result in the date window, call `load_bands` for a ~25 km bounding box centered at lat/lon.
   - Stack reflectance bands, tile-crop into non-overlapping `chip_size × chip_size` tiles, SCL-filter for cloud/shadow, keep up to `target_chips // 50 ≈ 100` chips per scene.
   - Emit a `datasets.Dataset` with the same schema as the existing builder, 90/10 train/val split.
   - Compute per-band mean/std across the final chip set, save `norm_stats.json` alongside.
   - `save_to_disk` and optionally `push_to_hub`.
3. CLI: add `--global-diverse` flag to `_cli` that calls `build_global_dataset` instead of the current preset builder.

**Scale estimate:** ~50 scenes × 5 bands per scene = 250 COG reads. At the existing builder's throughput, this is minutes, not hours. Tile I/O is the dominant cost; with `load_bands` already caching by tile-hash, any re-run is near-instant.

### `tests/unit/test_encoders.py` — Phase 1c

Add:

```python
@pytest.mark.parametrize("img_size,grid", [(128, 16), (256, 32)])
def test_vit_feature_grid_scales_with_img_size(img_size, grid):
    from src.experimental.encoders import FiveChannelViTPatch8
    m = FiveChannelViTPatch8(variant="tiny", img_size=img_size)
    x = torch.randn(1, 5, img_size, img_size)
    out = m(x)
    assert out.shape == (1, 192, grid, grid)


def test_build_encoder_dino_pretrained_loads_weights():
    """Smoke: ViT-Small/8 with pretrained=True should load DINO weights without
    raising, and the resulting patch_embed.proj.weight must differ from a
    freshly-initialized model (proving weights were actually transferred)."""
    from src.experimental.encoders import build_encoder
    fresh = build_encoder("vit_small_patch8", img_size=256, pretrained=False)
    loaded = build_encoder("vit_small_patch8", img_size=256, pretrained=True)
    w_fresh = fresh.vit.patch_embed.proj.weight
    w_loaded = loaded.vit.patch_embed.proj.weight
    assert w_fresh.shape == w_loaded.shape == (384, 5, 8, 8)
    # Different init = weights genuinely differ somewhere
    assert not torch.allclose(w_fresh, w_loaded)
    # 5-band stem: channels 3,4 should be close to mean(channels 0,1,2)
    # since that's how we adapted the 3-band DINO stem.
    expected_extra = w_loaded[:, :3, :, :].mean(dim=1, keepdim=True)
    assert torch.allclose(w_loaded[:, 3:4, :, :], expected_extra, atol=1e-6)
    assert torch.allclose(w_loaded[:, 4:5, :, :], expected_extra, atol=1e-6)
```

### `tests/unit/test_train_lejepa.py` and `tests/unit/test_inference.py`

Parameterize any existing shape assertions over `img_size` so both the 128 and 256 paths get CI coverage. Exact edits: grep for `16, 16` and `(192, 16, 16)` and replace with parameterized fixtures.

### `src/experimental/lightning_train.sh` — Phase 2c

**Changes:**

1. Add env vars at the top:
   ```bash
   PRETRAINED="${PRETRAINED:-0}"        # 1 → pass --pretrained flag
   ```
2. Update `CHECKPOINT_PATH` to embed img_size when not 128:
   ```bash
   if [ "${IMG_SIZE}" = "128" ]; then
       CHECKPOINT_PATH="checkpoints/lejepa_${ENCODER}_5band.pt"
   else
       CHECKPOINT_PATH="checkpoints/lejepa_${ENCODER}_${IMG_SIZE}_5band.pt"
   fi
   ```
   And mirror that in `checkpoint_filename()` in `train_lejepa.py` so the two agree.
3. Pass `--img-size "${IMG_SIZE}"` and conditionally `--pretrained` to both the smoke test and the real training invocation.
4. The `norm_stats.json` block that hardcodes the old 128 dataset stats must become a passthrough — the new dataset ships its own `norm_stats.json` inside the `save_to_disk` bundle, so the inline hardcoded stats block should be removed and replaced with a verification that `${DATASET_DIR}/norm_stats.json` exists after the pull (and written only if missing, as a safety net).

### `src/experimental/train_lejepa.py` — `checkpoint_filename` update

Match the lightning script convention:

```python
def checkpoint_filename(encoder_kind: EncoderKind, img_size: int = 128) -> str:
    if img_size == 128:
        return f"lejepa_{encoder_kind}_5band.pt"
    return f"lejepa_{encoder_kind}_{img_size}_5band.pt"
```

Callers pass `img_size=cfg.img_size` where relevant (epoch checkpoints + final save + upload_model's hub_filename derivation).

### `src/experimental/upload_model.py`

- `checkpoint_filename(encoder_kind)` call sites become `checkpoint_filename(encoder_kind, img_size)` — read `img_size` from the checkpoint config.
- `_ARCH_METADATA["vit_small_patch8"]["note"]` extended to mention DINO init + 256 input when those apply. Actually — the note is architecture-level not run-level, so keep it generic and let the config section of the card surface DINO init.
- Default `--dataset-repo-id` updated to point at the new global dataset when appropriate. Or leave it and pass explicitly from the lightning script. Prefer the latter to avoid overloading defaults.

## Execution sequence

**Phase 1 (M1, ~45 min):**
1. Implement encoder changes (img_size threading + DINO loader + 5-band stem).
2. Implement train_lejepa changes (TrainConfig, CLI, checkpoint_filename).
3. Implement inference changes (dynamic chip_size).
4. Update tests; run `uv run pytest tests/unit/test_encoders.py tests/unit/test_train_lejepa.py tests/unit/test_inference.py -v`.
5. Smoke-test the 128 path: `uv run python -m src.experimental.train_lejepa --smoke-test --encoder vit_tiny_patch8 --dataset cache/lejepa_dataset` — should match pre-change behavior.
6. Smoke-test the DINO-init 256 path on CPU, 1 step only, just to verify weights load and forward pass runs without error. Expect painfully slow; only need one successful step.
7. Commit.

**Phase 2 (M1, ~30-60 min mostly I/O):**
1. Implement `build_global_dataset` + CLI flag.
2. Run: `uv run python -m src.experimental.build_dataset --global-diverse --chip-size 256 --target-chips 5000 --output cache/lejepa_dataset_global_256 --push-to-hub falafel-hockey/sentinel2-lejepa-global-diverse-256`.
3. Spot-check: plot 10 random chips as RGB, verify global diversity.
4. Commit.

**Phase 2.5: Update lightning_train.sh and commit.**

**Phase 3 (H100, ~45 min):** The user runs on lightning.ai:
```bash
ENCODER=vit_small_patch8 \
IMG_SIZE=256 \
PRETRAINED=1 \
BATCH_SIZE=192 \
EPOCHS=150 \
DATASET_REPO=falafel-hockey/sentinel2-lejepa-global-diverse-256 \
MODEL_REPO_ID=falafel-hockey/lejepa-vit-small-patch8-256-sentinel2-5band \
WANDB_RUN_NAME=h100-dino-init-global-256 \
bash src/experimental/lightning_train.sh
```

**Phase 4:** Upload is auto-chained by the script. App verification by pulling the new checkpoint locally and eyeballing.

## Risks and open items

- **timm `pos_embed` shape nuance.** Some timm versions lay out register tokens inside `pos_embed`, others use a separate `reg_token` parameter. Will verify empirically in Phase 1b and adapt the slicing if needed.
- **DINO weights on a no-CLS-token architecture.** `load_state_dict(strict=False)` will drop DINO's `cls_token` silently; the register tokens stay at their default init. This is the intended behavior.
- **Global-scene STAC failures.** If a given scene has no cloud-free image in its date window, fall back to the next in the list or widen the date window. Must not silently drop below ~40 scenes of coverage.
- **1-hour H100 cap.** If fine-tuning at batch 192 × 150 epochs is too slow, drop to 100 epochs (DINO init converges fast — should be plenty). Smoke test on lightning's GPU early to measure steps/sec before committing the full run.
- **No guarantee of better visuals.** Primary fallback is still Tiny @ 256 from scratch if Small @ 256 with DINO init disappoints. Not planning for this mid-stream — the DINO init is a strong bet.
