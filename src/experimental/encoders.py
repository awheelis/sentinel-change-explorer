"""Encoder backbones for LeJEPA pretraining + inference.

Two encoder families live here so that train_lejepa and inference can share
the same instantiation code and a checkpoint carries enough metadata
(``config.encoder_kind``) to be rehydrated unambiguously by the inference
loader.

- ``resnet18``: legacy 5-band ResNet-18 returning a ``[B, 512, 4, 4]`` feature
  map. Kept for backward compatibility with the first published PoC model.
  Discouraged for new runs — its 4×4 grid (only 16 feature positions) produces
  blurry PCA→RGB visualizations.

- ``vit_tiny_patch8`` / ``vit_small_patch8``: 5-band ViT with patch size 8 at
  128×128 input, producing a **16×16 = 256** patch-token grid. Uses 4
  register tokens (Darcet et al. 2024) to suppress high-norm artifact tokens
  that otherwise dominate ViT feature PCA projections. This is the gold-
  standard architecture for clean DINOv2-style feature visualizations. Tiny
  is M1-trainable (~5.5M params), Small needs a GPU (~22M params).

All encoders expose the same interface: ``forward(x: [B, 5, 128, 128])``
returns a spatial feature map in NCHW format ``[B, D, H, W]``. Sharing the
NCHW shape lets the downstream masking, SIGReg, and visualization code stay
encoder-agnostic.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import resnet18


EncoderKind = Literal["resnet18", "vit_tiny_patch8", "vit_small_patch8"]

#: Canonical list of supported encoder kinds. Used by the CLI choices and by
#: the inference loader to reject unknown checkpoint metadata cleanly.
ENCODER_KINDS: tuple[str, ...] = ("resnet18", "vit_tiny_patch8", "vit_small_patch8")


class FiveChannelResNet18(nn.Module):
    """ResNet-18 with a 5-channel first conv, returning the pre-avgpool feature
    map ``[B, 512, 4, 4]`` at 128×128 input.

    Legacy architecture kept for backward compatibility with the first
    published LeJEPA checkpoint. New runs should prefer the ViT variants for
    their higher feature-grid resolution.
    """

    grid_side: int = 4
    embed_dim: int = 512

    def __init__(self, in_channels: int = 5) -> None:
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 512, 4, 4]


class FiveChannelViTPatch8(nn.Module):
    """timm ViT wrapper: 5 input channels, patch size 8, 128×128 input,
    register tokens, no class token, no pooling. Emits a ``[B, D, 16, 16]``
    patch-token feature map.

    Why this architecture (vs ResNet-18 for the same task):

    - **16× denser feature grid** (256 vs 16 positions). Directly visible in
      the PCA→RGB output — you get real spatial structure instead of a
      bilinear-upsampled 4×4 watercolor.
    - **Register tokens** (Darcet et al. 2024, "Vision Transformers Need
      Registers", arXiv:2309.16588). Without them, ViTs develop high-norm
      artifact tokens in background regions that dominate PCA projections
      and produce ugly "hot spot" noise. 4 registers soak up those
      artifacts, leaving the patch tokens clean for visualization.
    - **Native variable input channels** via ``in_chans=5``. No stem
      surgery needed the way ResNet requires.

    Two sizes are supported:
      - ``tiny``:  dim 192, depth 12, heads 3, ~5.5M params — M1 CPU-trainable.
      - ``small``: dim 384, depth 12, heads 6, ~22M params — GPU-only.
    """

    def __init__(
        self,
        variant: Literal["tiny", "small"] = "tiny",
        in_channels: int = 5,
        img_size: int = 128,
        patch_size: int = 8,
        reg_tokens: int = 4,
    ) -> None:
        super().__init__()
        # Import locally so the module stays importable even when timm is only
        # pulled in through the `experimental` extra.
        from timm.models.vision_transformer import VisionTransformer

        if variant == "tiny":
            embed_dim, depth, num_heads = 192, 12, 3
        elif variant == "small":
            embed_dim, depth, num_heads = 384, 12, 6
        else:  # pragma: no cover - guarded by Literal + factory
            raise ValueError(f"unknown variant: {variant!r}")

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
            class_token=False,
            reg_tokens=reg_tokens,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.variant = variant
        self.embed_dim = embed_dim
        self.grid_side = img_size // patch_size
        # timm exposes this so we don't have to hand-count class + register
        # prefix tokens. Robust if timm's token layout changes under us.
        self.num_prefix_tokens = self.vit.num_prefix_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` float tensor of shape ``[B, 5, 128, 128]``.

        Returns: ``[B, embed_dim, grid_side, grid_side]`` — the patch tokens
        only, reshaped to NCHW so downstream code (masking, PCA→RGB) is
        identical across encoder families.
        """
        feats = self.vit.forward_features(x)                 # [B, N_all, D]
        feats = feats[:, self.num_prefix_tokens :, :]        # drop register/CLS
        B, N, D = feats.shape
        gs = self.grid_side
        assert N == gs * gs, f"expected {gs*gs} patch tokens, got {N}"
        return feats.transpose(1, 2).reshape(B, D, gs, gs)


def load_dino_weights_into_vit_small_patch8(model: FiveChannelViTPatch8) -> None:
    """Load DINO's `vit_small_patch8_224.dino` ImageNet-pretrained weights into
    our 5-band, 4-register ViT-Small/8 wrapper, adapting the architectural
    deltas:

    - **3→5 channel stem:** DINO's `patch_embed.proj.weight` is
      ``[384, 3, 8, 8]``. We copy the RGB slice into channels 0–2 of our
      ``[384, 5, 8, 8]`` target and initialize channels 3–4 (NIR, SWIR16)
      from the per-filter mean of the RGB weights. Standard EO-adaptation
      trick (Clay, SatMAE, Prithvi).
    - **Positional embedding:** DINO's pos_embed is ``[1, 785, 384]``
      (1 CLS + 28×28 patches at 224/8). Ours is ``[1, 1028, 384]``
      (4 register slots + 32×32 patches at 256/8). We drop DINO's CLS row,
      reshape the remaining 784 patch embeddings to ``[1, 28, 28, 384]``,
      bicubic-interpolate to ``[1, 32, 32, 384]``, flatten, and splice into
      positions [4:1028] of our pos_embed. Positions [0:4] (the register
      slots) remain at timm's default init — they learn their role during
      fine-tuning while the patch positions carry DINO's spatial prior.
    - **CLS token:** DINO has one; we don't (``class_token=False``).
      Silently dropped.
    - **Register tokens:** We have four; DINO has none. Left at default
      init — the transformer blocks still carry DINO's semantic weights
      and register tokens rapidly adapt during SSL fine-tuning.
    - **Transformer blocks, final norm:** copied verbatim (shape-identical).

    Raises:
        RuntimeError: If the target model isn't a Small/8 variant or if
            timm's layout has drifted such that the expected shapes don't
            match at runtime (fail loud rather than silently mis-init).
    """
    import timm
    import torch.nn.functional as F

    if model.variant != "small":
        raise RuntimeError(
            f"DINO pretrained init is only available for ViT-Small/8; "
            f"got variant={model.variant!r}"
        )
    if model.vit.patch_embed.proj.weight.shape != (384, 5, 8, 8):
        raise RuntimeError(
            f"unexpected patch_embed.proj.weight shape "
            f"{tuple(model.vit.patch_embed.proj.weight.shape)}, expected (384, 5, 8, 8)"
        )

    # Download DINO weights. timm caches these in ~/.cache/huggingface so
    # subsequent calls are offline.
    dino = timm.create_model(
        "vit_small_patch8_224.dino", pretrained=True, num_classes=0
    )
    src = dino.state_dict()

    target_sd = model.vit.state_dict()

    # ── 1. Patch embed stem: 3 → 5 channels ─────────────────────────────
    src_stem_w = src["patch_embed.proj.weight"]  # [384, 3, 8, 8]
    assert src_stem_w.shape == (384, 3, 8, 8), src_stem_w.shape
    rgb_mean = src_stem_w.mean(dim=1, keepdim=True)  # [384, 1, 8, 8]
    new_stem_w = torch.cat(
        [src_stem_w, rgb_mean, rgb_mean], dim=1
    )  # [384, 5, 8, 8]
    target_sd["patch_embed.proj.weight"] = new_stem_w
    target_sd["patch_embed.proj.bias"] = src["patch_embed.proj.bias"].clone()

    # ── 2. Positional embedding: 1+784 → 4+1024 ─────────────────────────
    src_pos = src["pos_embed"]  # [1, 785, 384]
    assert src_pos.shape == (1, 785, 384), src_pos.shape
    src_patch_pos = src_pos[:, 1:, :]  # drop CLS → [1, 784, 384]
    # Reshape to 2D grid for bicubic interpolation
    src_grid = src_patch_pos.reshape(1, 28, 28, 384).permute(0, 3, 1, 2)  # [1, 384, 28, 28]
    tgt_grid = F.interpolate(
        src_grid, size=(32, 32), mode="bicubic", align_corners=False
    )  # [1, 384, 32, 32]
    tgt_patch_pos = tgt_grid.permute(0, 2, 3, 1).reshape(1, 1024, 384)

    # Our pos_embed is [1, 1028, 384] = [1, reg_slots(4) + patch_slots(1024), 384].
    # Overwrite the patch slots; leave the first 4 register slots at their
    # existing timm default init.
    cur_pos = target_sd["pos_embed"].clone()
    if cur_pos.shape != (1, 1028, 384):
        raise RuntimeError(
            f"unexpected target pos_embed shape {tuple(cur_pos.shape)}, "
            f"expected (1, 1028, 384). timm layout may have changed."
        )
    cur_pos[:, 4:, :] = tgt_patch_pos
    target_sd["pos_embed"] = cur_pos

    # ── 3. Transformer blocks + final norm (copy verbatim) ──────────────
    copied = 0
    skipped_shape = []
    for k, v in src.items():
        if k in ("patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed"):
            continue  # already handled
        if k in ("cls_token",):
            continue  # DINO has it, we don't
        if k in target_sd:
            if target_sd[k].shape == v.shape:
                target_sd[k] = v.clone()
                copied += 1
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(target_sd[k].shape)))

    # Load the merged state dict. strict=False permits our `reg_token` (not
    # in DINO) and any other register-related keys to stay at default init.
    missing, unexpected = model.vit.load_state_dict(target_sd, strict=False)

    # Log a summary so the training log shows exactly what happened.
    import logging
    log = logging.getLogger(__name__)
    log.info(
        "DINO init: copied %d block tensors, adapted stem (3→5ch) and "
        "pos_embed (28×28→32×32). missing=%d unexpected=%d skipped_shape=%d",
        copied, len(missing), len(unexpected), len(skipped_shape),
    )
    if skipped_shape:
        log.warning("DINO init: shape-mismatched tensors skipped: %s", skipped_shape)


def build_encoder(
    kind: EncoderKind,
    *,
    img_size: int = 128,
    pretrained: bool = False,
) -> nn.Module:
    """Instantiate an encoder by its registered kind string.

    Kept as a tiny factory rather than a dict so callers get a real function
    signature and type checking, and so checkpoint round-trips work: train
    stores ``cfg.encoder_kind`` + ``cfg.img_size``, inference reads them back
    and calls this.

    Args:
        kind: Encoder family ("resnet18", "vit_tiny_patch8", "vit_small_patch8").
        img_size: Input spatial size. ResNet-18 ignores this (fixed 4×4 grid at
            128 input); ViT variants scale their patch-token grid as
            ``img_size // patch_size`` so 256 input gives a 32×32 grid.
        pretrained: If True and kind=="vit_small_patch8", initialize from
            DINO's ImageNet weights via ``load_dino_weights_into_vit_small_patch8``.
            Only supported for the Small variant (DINO didn't publish patch-8
            Tiny weights).
    """
    if kind == "resnet18":
        if pretrained:
            raise ValueError("pretrained=True is only supported for vit_small_patch8")
        return FiveChannelResNet18(in_channels=5)
    if kind == "vit_tiny_patch8":
        if pretrained:
            raise ValueError("pretrained=True is only supported for vit_small_patch8")
        return FiveChannelViTPatch8(variant="tiny", img_size=img_size)
    if kind == "vit_small_patch8":
        model = FiveChannelViTPatch8(variant="small", img_size=img_size)
        if pretrained:
            load_dino_weights_into_vit_small_patch8(model)
        return model
    raise ValueError(
        f"unknown encoder_kind {kind!r}; expected one of {ENCODER_KINDS}"
    )


__all__ = [
    "ENCODER_KINDS",
    "EncoderKind",
    "FiveChannelResNet18",
    "FiveChannelViTPatch8",
    "build_encoder",
    "load_dino_weights_into_vit_small_patch8",
]
