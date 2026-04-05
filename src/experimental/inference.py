"""Inference + PCA→RGB visualization for the experimental LeJEPA feature.

Phase 6 of the experimental feature. Loads a trained LeJEPA ResNet-18 encoder
from the HuggingFace Hub (or a local checkpoint), extracts features for the
before/after scenes already present in the Streamlit app's session state, and
renders three visual outputs:

    1. PCA→RGB feature image of the before scene (4x4 → display size).
    2. PCA→RGB feature image of the after scene.
    3. Per-position cosine-distance change map with a matplotlib `magma`
       colormap, shown side-by-side with the existing classical NDVI-delta
       heatmap so the user can eyeball learned-vs-classical agreement.

The encoder class itself lives in ``train_lejepa`` — we import it from there
so the weight keys match by construction. ``torch`` is pulled in through the
``experimental`` extras; this module only gets imported by ``app.py`` after
the sidebar toggle + ``_has_torch()`` gate, so its import failure is
impossible in normal runs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


#: Band order expected by the 5-channel encoder. Must match the training data
#: ordering in ``build_dataset.REFLECTANCE_BANDS``.
_REFLECTANCE_BANDS: tuple[str, ...] = ("red", "green", "blue", "nir", "swir16")

#: Model-input spatial dimensions. The encoder was trained at exactly this
#: size and emits a 4x4 feature grid at this resolution.
_CHIP_SIZE: int = 128


# ── Model loading ────────────────────────────────────────────────────────────


@st.cache_resource
def load_model_cached(
    repo_id: str | None = None,
    local_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Load the encoder + norm stats either from the HF Hub or a local path.

    Cached with ``@st.cache_resource`` so the download + load happens exactly
    once per Streamlit session even as the user clicks between presets.

    Args:
        repo_id: HF Hub model repo id (e.g.
            ``alexw0/lejepa-resnet18-sentinel2-5band``). Used if
            ``local_checkpoint`` is not set.
        local_checkpoint: Absolute path to a local ``.pt`` file. Takes
            precedence over ``repo_id`` if provided — useful for developing
            the app against an as-yet-unpublished checkpoint.

    Returns:
        Dict with keys ``encoder`` (nn.Module in eval mode, CPU), ``mean``
        (torch float tensor [1,5,1,1]), ``std`` (same shape), and ``source``
        (human-readable string for the UI caption).
    """
    import torch

    from src.experimental.train_lejepa import FiveChannelResNet18

    if local_checkpoint is not None:
        ckpt_path = Path(local_checkpoint)
        source = f"local: {ckpt_path}"
    elif repo_id is not None:
        from huggingface_hub import hf_hub_download

        ckpt_path = Path(
            hf_hub_download(repo_id=repo_id, filename="lejepa_resnet18_5band.pt")
        )
        source = f"hub: {repo_id}"
    else:
        raise ValueError("Must provide either repo_id or local_checkpoint")

    logger.info("Loading LeJEPA encoder from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    encoder = FiveChannelResNet18(in_channels=5)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    norm = ckpt["norm_stats"]
    mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, 5, 1, 1)
    std = (
        torch.tensor(norm["std"], dtype=torch.float32).view(1, 5, 1, 1).clamp(min=1.0)
    )
    return {"encoder": encoder, "mean": mean, "std": std, "source": source}


# ── Preprocessing ────────────────────────────────────────────────────────────


def _stack_bands(bands_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the 5 reflectance bands from a load_bands() dict into ``[5, H, W]``.

    Raises KeyError if any expected band is missing. The SCL band (if present
    in the dict) is ignored — the encoder was only trained on reflectance.
    """
    missing = [b for b in _REFLECTANCE_BANDS if b not in bands_dict]
    if missing:
        raise KeyError(f"bands_dict missing required bands: {missing}")
    arrays = [bands_dict[b].astype(np.float32) for b in _REFLECTANCE_BANDS]
    H = min(a.shape[0] for a in arrays)
    W = min(a.shape[1] for a in arrays)
    # bands from load_bands should already share shape, but crop defensively
    # in case a reprojection rounding gave different sizes.
    stacked = np.stack([a[:H, :W] for a in arrays], axis=0)
    return stacked  # [5, H, W]


def _center_crop_or_pad(x: np.ndarray, size: int = _CHIP_SIZE) -> np.ndarray:
    """Force a ``[5, H, W]`` stack to ``[5, size, size]`` via center crop or
    symmetric reflect padding.

    This is a deliberate PoC simplification — typical app AOIs land within
    0.5-3 km at 10 m/px (50-300 px wide), so one centered crop captures the
    change region for the 5 demo presets. Multi-tile inference for larger
    drawn AOIs is explicitly out of scope for the PoC.
    """
    C, H, W = x.shape

    def _fit(arr: np.ndarray, dim: int, axis: int) -> np.ndarray:
        cur = arr.shape[axis]
        if cur == dim:
            return arr
        if cur > dim:
            start = (cur - dim) // 2
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(start, start + dim)
            return arr[tuple(sl)]
        # pad (cur < dim)
        total = dim - cur
        before = total // 2
        after = total - before
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (before, after)
        return np.pad(arr, pad_width, mode="reflect")

    x = _fit(x, size, axis=1)
    x = _fit(x, size, axis=2)
    return x


def extract_features(
    bands_dict: dict[str, np.ndarray], model: dict[str, Any]
) -> np.ndarray:
    """Run a dict of load_bands() arrays through the LeJEPA encoder.

    Returns:
        ``[512, 4, 4]`` numpy float32 feature map for a single tile.
    """
    import torch

    stacked = _stack_bands(bands_dict)                 # [5, H, W]
    stacked = _center_crop_or_pad(stacked, _CHIP_SIZE) # [5, 128, 128]
    x = torch.from_numpy(stacked).unsqueeze(0)         # [1, 5, 128, 128]
    x = (x - model["mean"]) / model["std"]
    with torch.no_grad():
        feat = model["encoder"](x)                     # [1, 512, 4, 4]
    return feat.squeeze(0).cpu().numpy()               # [512, 4, 4]


# ── Feature visualization ────────────────────────────────────────────────────


def features_to_rgb(feat_map: np.ndarray, display_size: int = 256) -> np.ndarray:
    """Reduce a ``[C, H, W]`` feature map to an RGB image via PCA(3) on the
    channel dimension.

    Steps:
      1. Flatten to ``[H*W, C]``.
      2. Center, take top-3 right singular vectors (numpy SVD).
      3. Project → ``[H*W, 3]``, per-channel min-max normalize.
      4. Reshape to ``[H, W, 3]`` and upsample to display_size via PIL
         bilinear so the user sees a smooth image, not a 4x4 postage stamp.

    Returns:
        ``[display_size, display_size, 3]`` uint8 array.
    """
    from PIL import Image

    C, H, W = feat_map.shape
    flat = feat_map.reshape(C, H * W).T                # [HW, C]
    centered = flat - flat.mean(axis=0, keepdims=True)
    # SVD on the centered data: V[:, :3] are the top-3 principal axes
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]                                # [3, C]
    projected = centered @ components.T                # [HW, 3]
    # Per-channel min-max to [0, 1] for display
    lo = projected.min(axis=0, keepdims=True)
    hi = projected.max(axis=0, keepdims=True)
    rng = np.where(hi - lo < 1e-8, 1.0, hi - lo)
    rgb = (projected - lo) / rng                       # [HW, 3] in [0, 1]
    rgb_img = (rgb.reshape(H, W, 3) * 255.0).astype(np.uint8)
    # Upsample 4x4 → display_size with bilinear interpolation
    pil = Image.fromarray(rgb_img).resize(
        (display_size, display_size), resample=Image.BILINEAR
    )
    return np.asarray(pil)


def features_to_change_map(
    before_feat: np.ndarray, after_feat: np.ndarray, display_size: int = 256
) -> np.ndarray:
    """Per-position cosine distance between two ``[C, H, W]`` feature maps.

    Returns an RGBA uint8 image of shape ``[display_size, display_size, 4]``
    colored with the matplotlib ``magma`` colormap (bright = high change).
    """
    import matplotlib.cm as cm
    from PIL import Image

    assert before_feat.shape == after_feat.shape, "feature maps must match"
    C, H, W = before_feat.shape
    a = before_feat.reshape(C, H * W)                  # [C, HW]
    b = after_feat.reshape(C, H * W)
    na = np.linalg.norm(a, axis=0) + 1e-8
    nb = np.linalg.norm(b, axis=0) + 1e-8
    cos = (a * b).sum(axis=0) / (na * nb)              # [HW] in [-1, 1]
    dist = 1.0 - cos                                   # [HW] in [0, 2]
    dist_map = dist.reshape(H, W)
    # Normalize to [0, 1] for the colormap; clamp guards against tiny AOIs
    # with only one value.
    lo, hi = float(dist_map.min()), float(dist_map.max())
    if hi - lo < 1e-8:
        norm = np.zeros_like(dist_map)
    else:
        norm = (dist_map - lo) / (hi - lo)
    rgba = (cm.magma(norm) * 255.0).astype(np.uint8)   # [H, W, 4]
    pil = Image.fromarray(rgba).resize(
        (display_size, display_size), resample=Image.NEAREST
    )
    return np.asarray(pil)


# ── Streamlit panel ──────────────────────────────────────────────────────────


#: Defaults for the panel's model source. When the PoC model is published
#: to the Hub, update these or pass overrides via render_experimental_panel.
DEFAULT_REPO_ID = "alexw0/lejepa-resnet18-sentinel2-5band"
DEFAULT_LOCAL_FALLBACK = "checkpoints/lejepa_resnet18_5band.pt"


def _resolve_model() -> dict[str, Any]:
    """Prefer a local checkpoint during development, fall back to the Hub."""
    local = Path(DEFAULT_LOCAL_FALLBACK)
    if local.exists():
        return load_model_cached(local_checkpoint=str(local.resolve()))
    return load_model_cached(repo_id=DEFAULT_REPO_ID)


def render_experimental_panel(
    before_bands: dict[str, np.ndarray],
    after_bands: dict[str, np.ndarray],
) -> None:
    """Render the foundation-model visualization panel inside the Streamlit app.

    Layout:
        row 1: three columns — Before PCA→RGB, After PCA→RGB, learned change.
        row 2: expandable "Methodology" note explaining the PCA projection
               and the small-model disclaimer.
    """
    st.markdown("### Experimental: Foundation Model (PoC)")
    st.caption(
        "LeJEPA ResNet-18 self-supervised features, trained on ~3k "
        "Sentinel-2 chips biased toward the 5 demo presets. **This is a "
        "proof of concept** — features are coarse and PCA projections can "
        "look noisy. Full methodology in the expander below."
    )

    try:
        model = _resolve_model()
    except Exception as e:
        st.error(
            f"Failed to load the LeJEPA encoder: {e}\n\n"
            f"Expected a checkpoint at `{DEFAULT_LOCAL_FALLBACK}` or the HF "
            f"repo `{DEFAULT_REPO_ID}`. Run Phase 4 training or Phase 5 "
            f"upload first."
        )
        return

    try:
        before_feat = extract_features(before_bands, model)
        after_feat = extract_features(after_bands, model)
    except KeyError as e:
        st.error(f"Band dict is missing a required reflectance band: {e}")
        return

    before_rgb = features_to_rgb(before_feat)
    after_rgb = features_to_rgb(after_feat)
    change_rgba = features_to_change_map(before_feat, after_feat)

    col_b, col_a, col_c = st.columns(3)
    col_b.image(before_rgb, caption="Before features (PCA→RGB)", use_container_width=True)
    col_a.image(after_rgb, caption="After features (PCA→RGB)", use_container_width=True)
    col_c.image(change_rgba, caption="Learned change (cos-dist)", use_container_width=True)

    with st.expander("How to read these panels"):
        st.markdown(
            """
**PCA→RGB feature images** project the encoder's 512-dimensional
per-location features down to 3 principal components and show them as
an RGB image. Similar land-cover types get similar colors; the
interpretation is unsupervised, so "red means forest" is not a fixed
legend — it's whatever the top three principal directions happen to
capture for this specific tile.

**Learned change (cos-dist)** computes the cosine distance between
before/after feature vectors at each of the 16 patch positions on the
feature grid, then upsamples to display size. Bright = high change.

**What to trust.** The model is tiny (~11M params) and trained on a
tiny dataset (~5k chips biased toward these 5 presets). The change
signal is directional — it agrees with the classical NDVI/MNDWI/NDBI
maps for large, obvious changes — but absolute magnitudes are noisy.
This is a proof-of-concept pipeline demonstration, not a SOTA change
detector. A companion training run on real GPUs (lightning.ai) will
replace these weights with a stronger checkpoint without any API
changes.
"""
        )
    st.caption(f"Model source: {model['source']}")
