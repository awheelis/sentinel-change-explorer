"""LeJEPA pretraining for 5-channel Sentinel-2 encoders.

A minimal, single-file implementation of the LeJEPA objective (Balestriero &
LeCun, 2025, arXiv:2511.08544) adapted for small preset-biased Sentinel-2
datasets. The JEPA masking lives on the encoder's feature grid (not the
image input) so the same training loop works for both conv backbones and
ViTs without per-architecture masking plumbing. We train the online encoder
to predict target-encoder features at masked grid positions from a pooled
context representation, and regularize the full feature batch with SIGReg.

Two encoder families are supported and selected by ``--encoder``:

- ``vit_tiny_patch8`` (default). ViT-Tiny with patch size 8 and 4 register
  tokens, producing a **16×16 = 256**-position feature grid. This is the
  architecture that yields the DINOv2-style crisp PCA→RGB visualizations;
  ~5.5M params, M1-trainable.
- ``vit_small_patch8``: same recipe with ViT-Small dimensions, ~22M params,
  GPU-only for real runs.
- ``resnet18``: legacy 5-band ResNet-18, 4×4 grid (16 positions). Retained
  for backward compatibility with the first PoC checkpoint; not recommended
  for new runs because the coarse grid yields blurry visualizations.

Design tradeoffs chosen deliberately:

- Masking at the feature grid rather than at the image input. For conv
  backbones this is the only option; for ViTs it keeps the training loop
  architecture-agnostic. The SSL signal comes from the context-to-target
  prediction asymmetry, which survives feature-level masking either way.
- Mean-pooled context. A single mean over visible positions replaces
  I-JEPA's transformer-over-context, keeping the predictor tiny.
- BF16 mixed precision on CUDA, FP32 on CPU. MPS is explicitly avoided.

Usage:
    # Smoke test on M1 CPU against a local dataset (~3 min)
    uv run python -m src.experimental.train_lejepa --smoke-test

    # ViT-Tiny/8 training on M1 (PoC checkpoint)
    uv run python -m src.experimental.train_lejepa \\
        --encoder vit_tiny_patch8 --epochs 15 \\
        --limit-train-chips 500 --batch-size 16

    # ViT-Small/8 training on lightning.ai GPU (gold)
    uv run python -m src.experimental.train_lejepa \\
        --encoder vit_small_patch8 --epochs 50 --batch-size 128

    # Feature-collapse sanity check after training
    uv run python -m src.experimental.train_lejepa \\
        --analyze checkpoints/lejepa_vit_tiny_patch8_5band.pt
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# torch lives behind the `experimental` extra. Fail loud and early if missing
# so the app import path never accidentally pulls this module in minimal envs.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
except ImportError as e:  # pragma: no cover - guarded by the extra
    raise ImportError(
        "train_lejepa requires the 'experimental' extras. "
        "Install with: uv sync --extra experimental"
    ) from e

from src.experimental.encoders import (
    ENCODER_KINDS,
    EncoderKind,
    FiveChannelResNet18,  # re-exported for test backcompat
    build_encoder,
)


logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """All hyperparameters in one place. Round-trips into the checkpoint so a
    later inference run can see exactly how the encoder was trained."""

    dataset: str = "cache/lejepa_dataset"      # local path or HF repo id
    output_dir: Path = Path("checkpoints")
    encoder_kind: EncoderKind = "vit_tiny_patch8"  # default to the gold viz path
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    alpha_sigreg: float = 1.0                  # SIGReg loss weight
    ema_base: float = 0.996                    # target-encoder momentum start
    # Context / target counts. None → auto-pick a sensible fraction of the
    # encoder's grid (~62% visible, ~25% predicted). Override on the CLI only
    # when you want to sweep the masking schedule.
    n_context: int | None = None
    n_target: int | None = None
    checkpoint_every: int = 10                 # epochs
    seed: int = 42
    num_workers: int = 2
    smoke_test: bool = False
    limit_train_chips: int | None = None       # cap training subset (CPU runs)
    wandb_project: str | None = None           # e.g. "sentinel-change-lejepa"
    wandb_run_name: str | None = None


def default_mask_schedule(n_positions: int) -> tuple[int, int]:
    """Pick (n_context, n_target) for a given feature-grid size.

    Defaults to ~62% context / ~25% predicted, chosen empirically for I-JEPA-
    style dense prediction: enough visible signal for the predictor to have a
    chance, enough targets that the gradient signal isn't sparse.

    For the 16-position ResNet grid this gives (10, 4); for the 256-position
    ViT grid it gives (160, 64).
    """
    n_context = max(1, int(round(n_positions * 0.625)))
    n_target = max(1, int(round(n_positions * 0.25)))
    # Guarantee disjoint sets even in tiny-grid edge cases.
    if n_context + n_target > n_positions:
        n_target = max(1, n_positions - n_context)
    return n_context, n_target


def checkpoint_filename(encoder_kind: EncoderKind) -> str:
    """Canonical checkpoint filename for a given encoder kind."""
    return f"lejepa_{encoder_kind}_5band.pt"


def pick_device() -> torch.device:
    """CUDA if available, else CPU. MPS is intentionally avoided."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Predictor ────────────────────────────────────────────────────────────────


class Predictor(nn.Module):
    """Per-position predictor. Takes the pooled context embedding plus a
    learnable target-position embedding (one row per flattened grid cell) and
    predicts the target-encoder embedding at that position.

    Architecture: 2-layer MLP with GELU. Param count scales with ``dim`` (and
    therefore with encoder choice).
    """

    def __init__(self, dim: int = 512, n_positions: int = 16) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(n_positions, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self, context: torch.Tensor, target_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context: ``[B, D]`` pooled context representation.
            target_positions: ``[B, K]`` long tensor of flattened grid indices.

        Returns:
            Predicted target embeddings, shape ``[B, K, D]``.
        """
        B, K = target_positions.shape
        D = context.shape[-1]
        ctx = context.unsqueeze(1).expand(B, K, D)
        pos = self.pos_embed[target_positions]        # [B, K, D]
        return self.mlp(torch.cat([ctx, pos], dim=-1))  # [B, K, D]


# ── Masking + losses ─────────────────────────────────────────────────────────


def sample_masks(
    batch_size: int,
    n_context: int,
    n_target: int,
    n_positions: int = 16,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example disjoint context / target position sets via random perm.

    Returns:
        context_idx: ``[B, n_context]`` long — positions visible to the encoder.
        target_idx:  ``[B, n_target]`` long — positions the predictor must reach.
    """
    assert n_context + n_target <= n_positions, "context + target would overlap"
    perms = torch.argsort(
        torch.rand(batch_size, n_positions, generator=generator), dim=1
    )
    context_idx = perms[:, :n_context]
    target_idx = perms[:, n_context : n_context + n_target]
    return context_idx, target_idx


def sigreg_loss(embeddings: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """SIGReg: push the batch covariance toward the identity.

    Implements the simplified form ``‖Σ − I‖_F² + λ·|tr(Σ)/d − 1|`` from the
    LeJEPA paper, using centered embeddings and unbiased covariance.

    Args:
        embeddings: ``[N, D]`` float tensor of feature vectors (can be flattened
            spatial embeddings from the encoder feature map).
        lam: weight on the trace-matching term.
    """
    N, D = embeddings.shape
    mu = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mu
    cov = centered.T @ centered / max(N - 1, 1)
    eye = torch.eye(D, device=cov.device, dtype=cov.dtype)
    frob = (cov - eye).pow(2).sum()
    trace_term = (cov.diagonal().mean() - 1.0).abs()
    return frob + lam * trace_term


def predictive_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 between predictor output and EMA target at target positions."""
    return F.smooth_l1_loss(pred, target)


# ── Dataset wrapper ──────────────────────────────────────────────────────────


class LeJEPADataset(Dataset):
    """Wrap a HuggingFace ``Dataset`` of S2 chips as torch tensors with
    satellite-appropriate augmentation: horizontal flip + random 90° rotation
    only, no color jitter on calibrated reflectance."""

    def __init__(self, hf_dataset, mean: list[float], std: list[float]) -> None:
        self.ds = hf_dataset
        self.mean = torch.tensor(mean, dtype=torch.float32).view(5, 1, 1)
        # clamp std at 1.0 to avoid dividing by near-zero channels if any band
        # happened to be degenerate on a tiny smoke build
        self.std = (
            torch.tensor(std, dtype=torch.float32).view(5, 1, 1).clamp(min=1.0)
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        bands = np.asarray(self.ds[idx]["bands"], dtype=np.float32)
        x = torch.from_numpy(bands)
        x = (x - self.mean) / self.std
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k=k, dims=[1, 2])
        return x


def load_dataset_and_stats(dataset_arg: str) -> tuple[Any, dict[str, Any]]:
    """Load DatasetDict + norm stats from either a local ``save_to_disk`` dir
    or (future) a HF Hub repo id.

    Hub loading requires norm stats to be fetched separately; for now we only
    support local paths so the training loop can run end-to-end without a
    network dependency. Pulling from the Hub lands alongside inference work.
    """
    from datasets import load_from_disk

    p = Path(dataset_arg)
    if not p.exists():
        raise FileNotFoundError(
            f"{dataset_arg} is not a local dataset directory. "
            f"Hub loading is not yet wired into training — run "
            f"`datasets.load_dataset(...).save_to_disk(...)` first, then "
            f"point --dataset at the local path."
        )
    dd = load_from_disk(str(p))
    stats_path = p / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found alongside dataset")
    stats = json.loads(stats_path.read_text())
    return dd, stats


# ── EMA helper ───────────────────────────────────────────────────────────────


def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """In place: ``target = m * target + (1 - m) * online``."""
    with torch.no_grad():
        for t, o in zip(target.parameters(), online.parameters()):
            t.data.mul_(momentum).add_(o.data, alpha=1.0 - momentum)


def ema_schedule(step: int, total_steps: int, base: float) -> float:
    """Linear ramp from ``base`` to 1.0 over training."""
    if total_steps <= 1:
        return base
    return base + (1.0 - base) * (step / (total_steps - 1))


# ── Training loop ────────────────────────────────────────────────────────────


def train(cfg: TrainConfig) -> Path:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device()
    logger.info("Device: %s", device)

    dd, stats = load_dataset_and_stats(cfg.dataset)
    full_train = LeJEPADataset(dd["train"], stats["mean"], stats["std"])
    logger.info("Train chips available: %d", len(full_train))

    if cfg.smoke_test:
        train_ds: Dataset = Subset(full_train, list(range(min(20, len(full_train)))))
        logger.info("Smoke test: using %d chips", len(train_ds))
    elif cfg.limit_train_chips is not None:
        n = min(cfg.limit_train_chips, len(full_train))
        train_ds = Subset(full_train, list(range(n)))
        logger.info("Limited training to %d chips", n)
    else:
        train_ds = full_train

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader is empty: {len(train_ds)} chips, batch {cfg.batch_size}. "
            f"Lower --batch-size or supply a larger dataset."
        )

    online = build_encoder(cfg.encoder_kind).to(device)
    target_enc = build_encoder(cfg.encoder_kind).to(device)
    target_enc.load_state_dict(online.state_dict())
    for p in target_enc.parameters():
        p.requires_grad_(False)

    # Encoders expose these as attributes so downstream code stays shape-free.
    dim = online.embed_dim
    grid_side = online.grid_side
    n_positions = grid_side * grid_side
    predictor = Predictor(dim=dim, n_positions=n_positions).to(device)

    # Resolve masking schedule now that we know the grid size.
    if cfg.n_context is None or cfg.n_target is None:
        auto_ctx, auto_tgt = default_mask_schedule(n_positions)
        n_context = cfg.n_context if cfg.n_context is not None else auto_ctx
        n_target = cfg.n_target if cfg.n_target is not None else auto_tgt
    else:
        n_context, n_target = cfg.n_context, cfg.n_target
    if n_context + n_target > n_positions:
        raise ValueError(
            f"n_context ({n_context}) + n_target ({n_target}) exceeds "
            f"encoder grid size ({n_positions})"
        )
    logger.info(
        "Encoder: %s (dim=%d, grid=%dx%d, params=%.2fM). Masking: "
        "n_context=%d n_target=%d",
        cfg.encoder_kind, dim, grid_side, grid_side,
        sum(p.numel() for p in online.parameters()) / 1e6,
        n_context, n_target,
    )

    params = list(online.parameters()) + list(predictor.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps, 1))

    use_bf16 = device.type == "cuda"

    # Optional W&B init. Skipped silently if wandb isn't installed or no
    # project was passed — training proceeds either way.
    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={k: (str(v) if isinstance(v, Path) else v)
                        for k, v in asdict(cfg).items()},
            )
            logger.info("W&B run: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed; skipping W&B logging")
        except Exception as e:  # pragma: no cover - network / auth failures
            logger.warning("W&B init failed (%s); continuing without it", e)

    step = 0
    loss_history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        online.train()
        predictor.train()
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            B = x.shape[0]
            ctx_idx, tgt_idx = sample_masks(
                B, n_context, n_target, n_positions=n_positions
            )
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                feat = online(x)                                # [B, D, H, W]
                Bf, D, H, W = feat.shape
                flat = feat.flatten(2).transpose(1, 2)          # [B, HW, D]

                ctx_embed = torch.gather(
                    flat, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, D)
                ).mean(dim=1)                                    # [B, D]

                with torch.no_grad():
                    tgt_feat = target_enc(x)
                    tgt_flat = tgt_feat.flatten(2).transpose(1, 2)
                    target_embed = torch.gather(
                        tgt_flat, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, D)
                    )                                            # [B, K, D]

                pred = predictor(ctx_embed, tgt_idx)             # [B, K, D]
                l_pred = predictive_loss(pred, target_embed)

                emb_for_reg = flat.reshape(-1, D).float()        # [B*HW, D]
                l_reg = sigreg_loss(emb_for_reg)
                loss = l_pred + cfg.alpha_sigreg * l_reg

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()

            m = ema_schedule(step, total_steps, cfg.ema_base)
            ema_update(target_enc, online, m)

            loss_history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "l_pred": float(l_pred.item()),
                    "l_reg": float(l_reg.item()),
                }
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss/total": float(loss.item()),
                        "loss/predictive": float(l_pred.item()),
                        "loss/sigreg": float(l_reg.item()),
                        "lr": sched.get_last_lr()[0],
                        "ema_momentum": m,
                        "epoch": epoch,
                    },
                    step=step,
                )
            if step % 10 == 0:
                logger.info(
                    "epoch %d step %d loss=%.4f pred=%.4f reg=%.4f lr=%.2e ema=%.4f",
                    epoch, step, loss.item(), l_pred.item(), l_reg.item(),
                    sched.get_last_lr()[0], m,
                )
            step += 1

        if (epoch + 1) % cfg.checkpoint_every == 0 or epoch == cfg.epochs - 1:
            stem = checkpoint_filename(cfg.encoder_kind).removesuffix(".pt")
            ckpt_path = cfg.output_dir / f"{stem}_e{epoch+1:03d}.pt"
            save_checkpoint(ckpt_path, online, cfg, stats, loss_history)

    final_path = cfg.output_dir / checkpoint_filename(cfg.encoder_kind)
    save_checkpoint(final_path, online, cfg, stats, loss_history)

    if wandb_run is not None:
        wandb_run.finish()

    return final_path


def save_checkpoint(
    path: Path,
    encoder: nn.Module,
    cfg: TrainConfig,
    norm_stats: dict[str, Any],
    loss_history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()
    }
    payload = {
        "encoder_state": encoder.state_dict(),
        "config": cfg_dict,
        "norm_stats": norm_stats,
        "training_summary": {
            "final_loss": loss_history[-1]["loss"] if loss_history else None,
            "n_steps": len(loss_history),
            "loss_tail": loss_history[-50:],
        },
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint: %s", path)


# ── Collapse analyzer ────────────────────────────────────────────────────────


def _simple_kmeans(
    x: np.ndarray, k: int, n_iter: int = 30, seed: int = 0
) -> np.ndarray:
    """Tiny numpy k-means (Lloyd's algorithm). Purpose-built for the collapse
    sanity probe — we only need a rough cluster assignment, not an optimized
    implementation, so we avoid a scikit-learn dependency."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    # init: random distinct points
    idx = rng.choice(n, size=k, replace=False)
    centers = x[idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        # assign: nearest center by squared L2
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update: mean of assigned points, leave empty clusters where they are
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = x[mask].mean(axis=0)
    return labels


def analyze(checkpoint_path: str, dataset_arg: str, k: int = 10) -> dict[str, Any]:
    """Quick feature-collapse sanity check. Extracts globally-pooled features
    for up to 200 val chips, clusters them with k-means, and logs the cluster
    distribution. A healthy run should spread points across clusters; if one
    cluster holds >90% that's a collapse signal."""
    device = pick_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_kind = ckpt.get("config", {}).get("encoder_kind", "resnet18")
    enc = build_encoder(encoder_kind).to(device)
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()

    dd, stats = load_dataset_and_stats(dataset_arg)
    ds = LeJEPADataset(dd["validation"], stats["mean"], stats["std"])
    n = min(200, len(ds))
    feats = []
    with torch.no_grad():
        for i in range(n):
            x = ds[i].unsqueeze(0).to(device)
            f = enc(x).mean(dim=[2, 3])  # global avg pool → [1, 512]
            feats.append(f.cpu().numpy()[0])
    feats_np = np.stack(feats)
    labels = _simple_kmeans(feats_np, k=k)
    counts = {int(i): int((labels == i).sum()) for i in range(k)}
    logger.info("Cluster distribution (k=%d, N=%d): %s", k, n, counts)
    if max(counts.values()) > 0.9 * n:
        logger.warning("Possible feature collapse: one cluster holds >90%% of points")
    return {"n": n, "k": k, "counts": counts}


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.experimental.train_lejepa",
        description="Pretrain a 5-band ResNet-18 with the LeJEPA objective.",
    )
    parser.add_argument("--dataset", type=str, default="cache/lejepa_dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--encoder",
        type=str,
        default="vit_tiny_patch8",
        choices=list(ENCODER_KINDS),
        help=(
            "Encoder backbone. vit_tiny_patch8 (default) → 16×16 feature "
            "grid with registers, M1-trainable. vit_small_patch8 → same "
            "but GPU-only. resnet18 → legacy 4×4 grid."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-sigreg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--limit-train-chips", type=int, default=None,
        help="Cap the number of training chips (mostly for CPU / M1 runs).",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="If set, log training metrics to this W&B project. Requires "
             "`wandb login` first.",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="1 epoch on 20 chips, batch 4, no workers — verifies the full graph",
    )
    parser.add_argument(
        "--analyze", type=str, default=None, metavar="CKPT",
        help="Path to a checkpoint; runs k-means collapse check instead of training",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.analyze:
        analyze(args.analyze, args.dataset)
        return

    cfg = TrainConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        encoder_kind=args.encoder,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=4 if args.smoke_test else args.batch_size,
        lr=args.lr,
        alpha_sigreg=args.alpha_sigreg,
        seed=args.seed,
        num_workers=0 if args.smoke_test else args.num_workers,
        smoke_test=args.smoke_test,
        limit_train_chips=args.limit_train_chips,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    train(cfg)


if __name__ == "__main__":
    _cli()


__all__ = [
    "TrainConfig",
    "FiveChannelResNet18",
    "Predictor",
    "sample_masks",
    "sigreg_loss",
    "predictive_loss",
    "LeJEPADataset",
    "ema_update",
    "ema_schedule",
    "train",
    "save_checkpoint",
    "analyze",
    "default_mask_schedule",
    "checkpoint_filename",
]
