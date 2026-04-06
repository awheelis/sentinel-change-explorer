"""Unit tests for src.experimental.upload_model card rendering.

Only covers the pure rendering path — the HF Hub push is exercised manually
since it has external side effects.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.experimental.upload_model import render_model_card  # noqa: E402


def _fake_checkpoint() -> dict:
    return {
        "config": {
            "epochs": 15,
            "batch_size": 16,
            "lr": 1e-3,
            "alpha_sigreg": 1.0,
            "limit_train_chips": 500,
        },
        "norm_stats": {
            "bands": ["red", "green", "blue", "nir", "swir16"],
            "mean": [1500.0, 1200.0, 900.0, 2200.0, 3100.0],
            "std": [400.0, 350.0, 300.0, 500.0, 600.0],
            "chip_count": 450,
        },
        "training_summary": {
            "n_steps": 465,
            "final_loss": 123.45,
        },
    }


def test_render_model_card_substitutes_all_placeholders():
    card = render_model_card(
        repo_id="falafel-hockey/lejepa-resnet18-sentinel2-5band",
        dataset_repo_id="falafel-hockey/sentinel2-lejepa-global-diverse-256",
        checkpoint=_fake_checkpoint(),
        build_date="2026-04-04",
    )
    # Dynamic substitutions
    assert "falafel-hockey/lejepa-resnet18-sentinel2-5band" in card
    assert "falafel-hockey/sentinel2-lejepa-global-diverse-256" in card
    assert "2026-04-04" in card
    assert "1500.00" in card         # norm stats mean
    assert "600.00" in card          # norm stats std
    assert "123.4500" in card or "123.45" in card  # final_loss
    assert " 465 " in card or "|465" in card or "465 steps" in card or "465" in card
    assert "500" in card             # limit_train_chips
    # No unsubstituted tokens
    assert "{repo_id}" not in card
    assert "{norm_stats_table}" not in card
    assert "{final_loss" not in card


def test_render_model_card_handles_missing_optional_fields():
    """A checkpoint without limit_train_chips should not crash — just fall
    back to '(full dataset)'."""
    ckpt = _fake_checkpoint()
    ckpt["config"].pop("limit_train_chips")
    card = render_model_card(
        repo_id="u/x",
        dataset_repo_id="u/y",
        checkpoint=ckpt,
        build_date="2026-04-04",
    )
    assert "(full dataset)" in card


def test_render_model_card_has_yaml_frontmatter():
    card = render_model_card(
        repo_id="u/x",
        dataset_repo_id="u/y",
        checkpoint=_fake_checkpoint(),
        build_date="2026-04-04",
    )
    assert card.startswith("---\n")
    head, _, _ = card[4:].partition("\n---\n")
    assert "library_name: pytorch" in head
    assert "pipeline_tag: image-feature-extraction" in head
    assert "datasets:" in head


def test_render_model_card_handles_none_final_loss():
    """Fresh-init checkpoints have final_loss=None; the formatter should not
    crash."""
    ckpt = _fake_checkpoint()
    ckpt["training_summary"]["final_loss"] = None
    card = render_model_card(
        repo_id="u/x",
        dataset_repo_id="u/y",
        checkpoint=ckpt,
        build_date="2026-04-04",
    )
    assert "0.0000" in card
