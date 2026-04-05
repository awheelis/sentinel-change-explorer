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


def build_encoder(kind: EncoderKind) -> nn.Module:
    """Instantiate an encoder by its registered kind string.

    Kept as a tiny factory rather than a dict so callers get a real function
    signature and type checking, and so checkpoint round-trips work: train
    stores ``cfg.encoder_kind``, inference reads it back and calls this.
    """
    if kind == "resnet18":
        return FiveChannelResNet18(in_channels=5)
    if kind == "vit_tiny_patch8":
        return FiveChannelViTPatch8(variant="tiny")
    if kind == "vit_small_patch8":
        return FiveChannelViTPatch8(variant="small")
    raise ValueError(
        f"unknown encoder_kind {kind!r}; expected one of {ENCODER_KINDS}"
    )


__all__ = [
    "ENCODER_KINDS",
    "EncoderKind",
    "FiveChannelResNet18",
    "FiveChannelViTPatch8",
    "build_encoder",
]
