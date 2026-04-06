"""Publish a trained LeJEPA checkpoint to the HuggingFace Hub.

Given a ``.pt`` file produced by ``train_lejepa.save_checkpoint`` — which
already packages the encoder state dict, the training config (including
``encoder_kind``), the dataset norm stats, and a short training summary —
this module renders a model card from the checkpoint metadata and pushes
both to a HuggingFace model repo.

The hub filename is derived automatically from the checkpoint's
``encoder_kind`` field, so the same uploader handles the legacy ResNet-18,
the current ViT-Tiny/8, and any future encoder added to the factory.

Usage:
    uv run python -m src.experimental.upload_model \\
        --checkpoint checkpoints/lejepa_vit_tiny_patch8_5band.pt \\
        --repo-id ANONYMOUS/lejepa-vit-tiny-patch8-sentinel2-5band

Phase 5 of the experimental LeJEPA feature. The companion dataset publish
lives in ``build_dataset.py``; inference that pulls the result of this
module lives in ``inference.py``.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_CARD_TEMPLATE_PATH = (
    _REPO_ROOT / "src" / "experimental" / "model_card_template.md"
)

#: Band list used for the norm stats table row labels.
_DEFAULT_BANDS = ("red", "green", "blue", "nir", "swir16")


#: Architecture-specific metadata used to fill the model card template. The
#: values here are what make the card honest about grid size, param count,
#: and the register-token trick — callers read from the checkpoint and the
#: instantiated encoder rather than hardcoding them in the card body.
_ARCH_METADATA: dict[str, dict[str, Any]] = {
    "resnet18": {
        "name": "ResNet-18",
        "description": (
            "ResNet-18 with a 5-channel first conv; `avgpool` and `fc` "
            "dropped so the spatial feature map is the output."
        ),
        "note": (
            "**Why this architecture.** Kept as a legacy baseline for the "
            "first published PoC. The 4x4 feature grid is coarse, so PCA→RGB "
            "visualizations are dominated by bilinear smoothing. New runs "
            "should prefer the ViT variants."
        ),
    },
    "vit_tiny_patch8": {
        "name": "ViT-Tiny/8",
        "description": (
            "Vision Transformer (ViT-Tiny, patch size 8) with 4 register "
            "tokens, trained via timm's `VisionTransformer` with "
            "`in_chans=5`, `class_token=False`, `reg_tokens=4`."
        ),
        "note": (
            "**Why this architecture.** The 16×16 patch-token grid (256 "
            "positions) is 16× denser than the legacy ResNet's 4×4, which "
            "is what makes the PCA→RGB projections readable. Register "
            "tokens (Darcet et al. 2024, arXiv:2309.16588) absorb the "
            "high-norm artifact tokens that otherwise dominate ViT feature "
            "PCA — they are the single biggest quality trick for clean "
            "feature visualizations."
        ),
    },
    "vit_small_patch8": {
        "name": "ViT-Small/8",
        "description": (
            "Vision Transformer (ViT-Small, patch size 8) with 4 register "
            "tokens. Same recipe as the Tiny variant but with 384-dim "
            "embeddings and 6 heads, trained on GPU for higher feature "
            "quality."
        ),
        "note": (
            "**Why this architecture.** The 384-dim 16×16 grid is the gold "
            "visualization target for this project; see the Tiny card for "
            "the full rationale on register tokens."
        ),
    },
}


def _format_norm_stats_table(norm_stats: dict[str, Any]) -> str:
    """Render the per-band mean/std table exactly like the dataset card."""
    bands = norm_stats.get("bands", list(_DEFAULT_BANDS))
    means = norm_stats["mean"]
    stds = norm_stats["std"]
    return "\n".join(
        f"| {b:<6} | {m:>9.2f} | {s:>9.2f} |"
        for b, m, s in zip(bands, means, stds)
    )


def render_model_card(
    *,
    repo_id: str,
    dataset_repo_id: str,
    checkpoint: dict[str, Any],
    build_date: str | None = None,
) -> str:
    """Render the model card markdown from a loaded checkpoint dict.

    Args:
        repo_id: Target HF model repo id.
        dataset_repo_id: HF dataset repo id used during training (shown in the
            card so readers can retrace reproduction).
        checkpoint: The dict saved by ``train_lejepa.save_checkpoint``. Must
            contain ``config``, ``norm_stats``, and ``training_summary``.
        build_date: ISO date string; defaults to today.

    Returns:
        The fully rendered README.md markdown.
    """
    if build_date is None:
        build_date = date.today().isoformat()
    build_year = build_date.split("-")[0]

    cfg = checkpoint.get("config", {})
    stats = checkpoint.get("norm_stats", {})
    summary = checkpoint.get("training_summary", {})

    # Precision / device are inferred — we only run BF16 on CUDA in the
    # current training loop, everything else is FP32 on CPU. That's what the
    # card needs to say so readers understand what the weights were trained
    # in.
    precision = "BF16 mixed" if torch.cuda.is_available() else "FP32"
    device = "CUDA" if torch.cuda.is_available() else "CPU (M1)"

    # Architecture-specific placeholders. We instantiate the encoder once
    # (on CPU, no weights loaded) just to read its published dim / grid_side
    # attributes, which keeps the card in sync with the real code instead of
    # risking a drift between a hand-edited template and the factory.
    from src.experimental.encoders import build_encoder
    from src.experimental.train_lejepa import checkpoint_filename

    encoder_kind = cfg.get("encoder_kind", "resnet18")
    img_size = cfg.get("img_size", 128)
    arch = _ARCH_METADATA.get(encoder_kind, _ARCH_METADATA["resnet18"])
    # pretrained=False: only probing for shapes/param counts, no weights loaded.
    enc_probe = build_encoder(encoder_kind, img_size=img_size, pretrained=False)
    embed_dim = enc_probe.embed_dim
    grid_side = enc_probe.grid_side
    encoder_params_m = sum(p.numel() for p in enc_probe.parameters()) / 1e6
    hub_filename = checkpoint_filename(encoder_kind, img_size)

    template = _MODEL_CARD_TEMPLATE_PATH.read_text()
    return template.format(
        repo_id=repo_id,
        dataset_repo_id=dataset_repo_id,
        build_date=build_date,
        build_year=build_year,
        train_date=build_date,
        epochs=cfg.get("epochs", "?"),
        batch_size=cfg.get("batch_size", "?"),
        lr=cfg.get("lr", "?"),
        alpha_sigreg=cfg.get("alpha_sigreg", "?"),
        train_chips=cfg.get("limit_train_chips") or "(full dataset)",
        precision=precision,
        device=device,
        n_steps=summary.get("n_steps", "?"),
        final_loss=float(summary.get("final_loss") or 0.0),
        norm_stats_table=_format_norm_stats_table(stats),
        encoder_kind=encoder_kind,
        architecture_name=arch["name"],
        architecture_description=arch["description"],
        architecture_note=arch["note"],
        embed_dim=embed_dim,
        grid_side=grid_side,
        n_positions=grid_side * grid_side,
        encoder_params_m=encoder_params_m,
        hub_filename=hub_filename,
    )


def upload(
    checkpoint_path: Path | str,
    repo_id: str,
    dataset_repo_id: str,
    private: bool = False,
) -> str:
    """Push the checkpoint + rendered model card to the HuggingFace Hub.

    The checkpoint file is uploaded verbatim under the name
    ``lejepa_resnet18_5band.pt`` (matching the Phase 6 inference loader's
    expected filename). A sidecar ``norm_stats.json`` is also written so
    downstream consumers that don't want to unpickle the .pt can still pull
    the normalization stats.

    Args:
        checkpoint_path: Path to the local ``.pt`` file.
        repo_id: Target HF model repo id.
        dataset_repo_id: HF dataset repo id used during training.
        private: If True, create a private repo.

    Returns:
        The public URL of the model on the Hub.
    """
    from huggingface_hub import HfApi

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} does not exist")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Derive the hub filename from the checkpoint's own encoder_kind so the
    # uploader works for every architecture in the factory without a flag.
    # Legacy ResNet checkpoints predate the encoder_kind field and fall back
    # to the historical name for backward compatibility with the first
    # published PoC model.
    from src.experimental.train_lejepa import checkpoint_filename

    ckpt_cfg = ckpt.get("config", {})
    encoder_kind = ckpt_cfg.get("encoder_kind", "resnet18")
    img_size = ckpt_cfg.get("img_size", 128)
    hub_filename = checkpoint_filename(encoder_kind, img_size)

    logger.info("Rendering model card …")
    card_md = render_model_card(
        repo_id=repo_id,
        dataset_repo_id=dataset_repo_id,
        checkpoint=ckpt,
    )

    api = HfApi()
    logger.info("Creating / updating repo %s …", repo_id)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    logger.info("Uploading checkpoint as %s …", hub_filename)
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=hub_filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add LeJEPA {encoder_kind} checkpoint",
    )

    # Sidecar norm stats as JSON so downstream consumers don't need to load
    # the pickle just to read a handful of floats.
    norm_stats_bytes = json.dumps(ckpt["norm_stats"], indent=2).encode("utf-8")
    logger.info("Uploading norm_stats.json …")
    api.upload_file(
        path_or_fileobj=norm_stats_bytes,
        path_in_repo="norm_stats.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add norm stats",
    )

    logger.info("Uploading model card …")
    api.upload_file(
        path_or_fileobj=card_md.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Model published: %s", url)
    return url


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.experimental.upload_model",
        description="Publish a LeJEPA ResNet-18 checkpoint to HuggingFace Hub.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/lejepa_vit_tiny_patch8_5band.pt"),
        help="Path to the .pt file produced by train_lejepa.",
    )
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help=(
            "Target HF model repo (e.g. "
            "'ANONYMOUS/lejepa-vit-tiny-patch8-sentinel2-5band')."
        ),
    )
    parser.add_argument(
        "--dataset-repo-id", type=str,
        default="ANONYMOUS/sentinel2-lejepa-preset-biased-small",
        help="HF dataset repo id used during training (shown in the model card).",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private model repo.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    url = upload(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
        dataset_repo_id=args.dataset_repo_id,
        private=args.private,
    )
    logger.info("Done. %s", url)


if __name__ == "__main__":
    _cli()


__all__ = ["render_model_card", "upload"]
