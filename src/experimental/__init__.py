"""Experimental LeJEPA foundation-model feature for Sentinel Change Explorer.

This package holds a self-contained proof-of-concept foundation-model pipeline:

    1. build_dataset.py  — curate a preset-biased Sentinel-2 chip dataset and
       publish it to HuggingFace Hub.
    2. train_lejepa.py   — pretrain a small 5-channel ResNet-18 with the
       LeJEPA objective (JEPA predictive loss + SIGReg regularizer).
    3. inference.py      — load the trained model, extract features, render
       PCA->RGB feature visualizations and a learned change heatmap in the
       Streamlit app's experimental panel.

Everything here is gated behind an optional dependency extra so a plain
`uv sync` keeps the main app lean. To enable:

    uv sync --extra experimental
    huggingface-cli login   # needed for publishing / pulling private artifacts

See ``src/experimental/README.md`` for the full reproduction story and
``docs/`` (or the planning doc at
``~/.claude/plans/iridescent-inventing-stearns.md``) for the design
rationale.
"""
from __future__ import annotations


def _has_torch() -> bool:
    """Return True iff the experimental extras are installed.

    Used by the Streamlit app to gate the experimental panel and render an
    install hint if the user has enabled the feature without installing the
    extras. We check for torch specifically because it's the heaviest and
    most likely missing dependency; if torch imports, timm/datasets are
    overwhelmingly likely to be present too (they are in the same extra).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["_has_torch"]
