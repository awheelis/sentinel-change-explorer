"""Publish a trained LeJEPA ResNet-18 checkpoint to the HuggingFace Hub.

Given a ``.pt`` file produced by ``train_lejepa.save_checkpoint`` — which
already packages the encoder state dict, the training config, the dataset
norm stats, and a short training summary — this module renders a model card
from the checkpoint metadata and pushes both to a HuggingFace model repo.

Usage:
    uv run python -m src.experimental.upload_model \\
        --checkpoint checkpoints/lejepa_resnet18_5band.pt \\
        --repo-id alexw0/lejepa-resnet18-sentinel2-5band \\
        --dataset-repo-id alexw0/sentinel2-lejepa-preset-biased-small

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

    # Upload the checkpoint under a stable name so the Phase 6 loader can
    # `hf_hub_download(filename="lejepa_resnet18_5band.pt")` without
    # guesswork about the original local path.
    logger.info("Uploading checkpoint …")
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="lejepa_resnet18_5band.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add LeJEPA ResNet-18 checkpoint",
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
        default=Path("checkpoints/lejepa_resnet18_5band.pt"),
        help="Path to the .pt file produced by train_lejepa.",
    )
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="Target HF model repo (e.g. 'alexw0/lejepa-resnet18-sentinel2-5band').",
    )
    parser.add_argument(
        "--dataset-repo-id", type=str,
        default="alexw0/sentinel2-lejepa-preset-biased-small",
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
