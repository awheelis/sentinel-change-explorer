"""Unit tests for src.experimental.train_lejepa pure pieces.

These tests only cover shape + invariant properties of the model, masking,
and loss functions — they do NOT exercise the full training loop, which is
verified end-to-end by the CLI smoke test (`--smoke-test`).

All tests are skipped automatically when the `experimental` extras are not
installed, so a plain `uv sync` env still passes the suite.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.experimental.encoders import (  # noqa: E402
    ENCODER_KINDS,
    FiveChannelViTPatch8,
    build_encoder,
)
from src.experimental.train_lejepa import (  # noqa: E402
    FiveChannelResNet18,
    Predictor,
    checkpoint_filename,
    default_mask_schedule,
    ema_schedule,
    ema_update,
    predictive_loss,
    sample_masks,
    sigreg_loss,
)


# ── Encoder ──────────────────────────────────────────────────────────────────


def test_encoder_output_shape():
    """128x128 input with 5 bands produces [B, 512, 4, 4] before avgpool."""
    enc = FiveChannelResNet18(in_channels=5)
    x = torch.zeros(2, 5, 128, 128)
    y = enc(x)
    assert y.shape == (2, 512, 4, 4)


def test_encoder_first_conv_accepts_five_channels():
    enc = FiveChannelResNet18(in_channels=5)
    # Accessing the first conv via the stem Sequential; it must be 5-channel.
    conv1 = enc.stem[0]
    assert conv1.in_channels == 5
    assert conv1.out_channels == 64


# ── Predictor ────────────────────────────────────────────────────────────────


def test_predictor_output_shape():
    pred = Predictor(dim=512, n_positions=16)
    context = torch.randn(3, 512)
    target_positions = torch.tensor([[0, 5, 10, 15], [1, 2, 3, 4], [6, 7, 8, 9]])
    out = pred(context, target_positions)
    assert out.shape == (3, 4, 512)


def test_predictor_pos_embed_shape():
    pred = Predictor(dim=512, n_positions=16)
    assert pred.pos_embed.shape == (16, 512)


# ── Masking ──────────────────────────────────────────────────────────────────


def test_sample_masks_shapes():
    ctx, tgt = sample_masks(batch_size=5, n_context=7, n_target=4)
    assert ctx.shape == (5, 7)
    assert tgt.shape == (5, 4)
    assert ctx.dtype == torch.long
    assert tgt.dtype == torch.long


def test_sample_masks_disjoint_per_example():
    """Context and target sets must share zero positions within an example."""
    ctx, tgt = sample_masks(batch_size=20, n_context=7, n_target=4)
    for i in range(20):
        ctx_set = set(ctx[i].tolist())
        tgt_set = set(tgt[i].tolist())
        assert ctx_set.isdisjoint(tgt_set), f"overlap in example {i}"


def test_sample_masks_positions_in_range():
    ctx, tgt = sample_masks(batch_size=10, n_context=7, n_target=4, n_positions=16)
    assert ctx.min() >= 0 and ctx.max() < 16
    assert tgt.min() >= 0 and tgt.max() < 16


def test_sample_masks_rejects_oversized_request():
    with pytest.raises(AssertionError):
        sample_masks(batch_size=1, n_context=10, n_target=10, n_positions=16)


# ── SIGReg loss ──────────────────────────────────────────────────────────────


def test_sigreg_loss_low_on_isotropic_gaussian():
    """Near-isotropic unit-variance features should give a small loss."""
    torch.manual_seed(0)
    emb = torch.randn(4096, 32)  # large N so sample cov ≈ I
    loss = sigreg_loss(emb).item()
    # Trace term ~|1 − 1| = 0, frob ~ small-batch noise on 32x32 cov
    assert loss < 5.0


def test_sigreg_loss_high_on_collapsed_features():
    """A collapsed (rank-1) batch should give a much larger loss than iso."""
    torch.manual_seed(0)
    iso = torch.randn(4096, 32)
    iso_loss = sigreg_loss(iso).item()

    # All embeddings identical → covariance is zero → frob = ‖−I‖² = D
    collapsed = torch.ones(4096, 32) * 3.0
    collapsed_loss = sigreg_loss(collapsed).item()

    assert collapsed_loss > iso_loss
    # Explicit lower bound: frob of -I_32 is 32, plus trace term of 1
    assert collapsed_loss >= 32.0


# ── Predictive loss ──────────────────────────────────────────────────────────


def test_predictive_loss_zero_when_pred_equals_target():
    x = torch.randn(4, 4, 512)
    assert predictive_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_predictive_loss_positive_on_mismatch():
    pred = torch.zeros(4, 4, 512)
    tgt = torch.ones(4, 4, 512)
    assert predictive_loss(pred, tgt).item() > 0.0


# ── EMA ──────────────────────────────────────────────────────────────────────


def test_ema_update_moves_toward_online():
    online = torch.nn.Linear(4, 4)
    target = torch.nn.Linear(4, 4)

    # Put target at zeros, online at ones; after one step with m=0.9 the target
    # weight should be 0.9*0 + 0.1*1 = 0.1.
    with torch.no_grad():
        for p in target.parameters():
            p.zero_()
        for p in online.parameters():
            p.fill_(1.0)

    ema_update(target, online, momentum=0.9)
    for p in target.parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 0.1))


def test_ema_schedule_ramps_to_one():
    assert ema_schedule(0, 100, 0.996) == pytest.approx(0.996)
    # At the final step the momentum has ramped to 1.0
    assert ema_schedule(99, 100, 0.996) == pytest.approx(1.0)


def test_ema_schedule_degenerate_single_step():
    assert ema_schedule(0, 1, 0.996) == pytest.approx(0.996)


# ── End-to-end shape smoke (no training) ─────────────────────────────────────


# ── Encoder factory + ViT ────────────────────────────────────────────────────


def test_encoder_factory_kinds_registered():
    """All advertised kinds must round-trip through the factory."""
    for kind in ENCODER_KINDS:
        # vit_small is 22M params; skip the slow build here, we only need
        # to check it's wired. resnet18 + vit_tiny are exercised below.
        if kind == "vit_small_patch8":
            continue
        enc = build_encoder(kind)
        y = enc(torch.zeros(1, 5, 128, 128))
        assert y.ndim == 4, f"{kind} should return NCHW"
        assert y.shape[0] == 1 and y.shape[2] == y.shape[3]


def test_encoder_factory_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown encoder_kind"):
        build_encoder("definitely_not_a_real_encoder")  # type: ignore[arg-type]


def test_vit_tiny_feature_grid_is_16x16():
    """The whole point of the ViT upgrade — dense 16x16 feature grid for
    clean PCA→RGB visualizations."""
    enc = FiveChannelViTPatch8(variant="tiny")
    y = enc(torch.zeros(2, 5, 128, 128))
    assert y.shape == (2, 192, 16, 16)
    assert enc.embed_dim == 192
    assert enc.grid_side == 16


def test_vit_register_tokens_stripped_before_reshape():
    """Forward must drop the 4 register tokens before reshaping to NCHW —
    otherwise the grid would be 260 tokens and the reshape would fail."""
    enc = FiveChannelViTPatch8(variant="tiny", reg_tokens=4)
    assert enc.num_prefix_tokens == 4
    # A forward pass that doesn't crash on the assertion inside forward()
    # is already the test — but spot-check the output grid anyway.
    y = enc(torch.zeros(1, 5, 128, 128))
    assert y.shape[-2:] == (16, 16)


# ── Mask schedule + checkpoint filename helpers ──────────────────────────────


def test_default_mask_schedule_resnet_grid():
    ctx, tgt = default_mask_schedule(16)
    assert ctx + tgt <= 16
    assert ctx > tgt  # more context than targets


def test_default_mask_schedule_vit_grid():
    ctx, tgt = default_mask_schedule(256)
    assert ctx + tgt <= 256
    # The default ratios should scale linearly with grid size
    assert ctx == 160  # round(256 * 0.625)
    assert tgt == 64   # round(256 * 0.25)


def test_checkpoint_filename_roundtrip():
    assert checkpoint_filename("resnet18") == "lejepa_resnet18_5band.pt"
    assert (
        checkpoint_filename("vit_tiny_patch8") == "lejepa_vit_tiny_patch8_5band.pt"
    )


# ── ViT end-to-end shape flow (mirrors the training inner loop) ──────────────


def test_vit_training_inner_loop_shapes():
    """Feed ViT features through masking + predictor exactly like train() does.

    If this passes, the training loop's gather / mean-pool / predictor path
    is shape-correct for the 256-position grid.
    """
    torch.manual_seed(0)
    enc = FiveChannelViTPatch8(variant="tiny")
    pred = Predictor(dim=enc.embed_dim, n_positions=enc.grid_side ** 2)
    x = torch.randn(2, 5, 128, 128)

    feat = enc(x)                                   # [2, 192, 16, 16]
    flat = feat.flatten(2).transpose(1, 2)          # [2, 256, 192]
    assert flat.shape == (2, 256, 192)

    n_ctx, n_tgt = default_mask_schedule(256)
    ctx_idx, tgt_idx = sample_masks(2, n_ctx, n_tgt, n_positions=256)
    D = feat.shape[1]
    ctx_embed = torch.gather(
        flat, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, D)
    ).mean(dim=1)
    assert ctx_embed.shape == (2, 192)

    out = pred(ctx_embed, tgt_idx)
    assert out.shape == (2, n_tgt, 192)


def test_forward_pass_shapes_match_training_loop():
    """One fake batch through encoder + masking + predictor — shape sanity."""
    torch.manual_seed(0)
    enc = FiveChannelResNet18()
    pred = Predictor(dim=512, n_positions=16)
    x = torch.randn(2, 5, 128, 128)

    feat = enc(x)                                  # [2, 512, 4, 4]
    flat = feat.flatten(2).transpose(1, 2)         # [2, 16, 512]
    assert flat.shape == (2, 16, 512)

    ctx_idx, tgt_idx = sample_masks(2, 7, 4)
    D = feat.shape[1]
    ctx_embed = torch.gather(
        flat, 1, ctx_idx.unsqueeze(-1).expand(-1, -1, D)
    ).mean(dim=1)                                   # [2, 512]
    assert ctx_embed.shape == (2, 512)

    out = pred(ctx_embed, tgt_idx)                  # [2, 4, 512]
    assert out.shape == (2, 4, 512)
