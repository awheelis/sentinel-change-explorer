#!/usr/bin/env bash
# One-shot LeJEPA pipeline for a lightning.ai GPU studio:
#   pull dataset from HF → write norm_stats → GPU smoke test →
#   full training run (backgrounded, logs to train.log) → block on training →
#   publish the resulting checkpoint + model card to HF.
#
# Assumes: repo already cloned, you are in the repo root, branch checked out.
# Prereqs you must do interactively BEFORE running this script:
#   1. curl -LsSf https://astral.sh/uv/install.sh | sh  (and source the env)
#   2. uv sync --extra experimental
#   3. uv run hf auth login     (HF write token — needed for publish step)
#   4. uv run wandb login       (W&B API key)
#
# Run with defaults (current ViT-Tiny/8 @ 128 path):
#   bash src/experimental/lightning_train.sh
#
# Or override any variable via env for the DINOv2-sharp path once the 256
# dataset exists (see docs/superpowers/specs/2026-04-04-dinov2-sharp-features-design.md):
#   ENCODER=vit_small_patch8 IMG_SIZE=256 BATCH_SIZE=32 \
#     DATASET_REPO=alexw0/sentinel2-lejepa-preset-biased-small-256 \
#     MODEL_REPO_ID=alexw0/lejepa-vit-small-patch8-256-sentinel2-5band \
#     bash src/experimental/lightning_train.sh
#
# `set -euo pipefail` + `wait` propagation means the publish step only runs
# if training exits cleanly. A failed smoke test or training run aborts the
# whole pipeline before touching HF.
set -euo pipefail

# ── Tunables ────────────────────────────────────────────────────────────────
ENCODER="${ENCODER:-vit_tiny_patch8}"
IMG_SIZE="${IMG_SIZE:-128}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"

DATASET_REPO="${DATASET_REPO:-alexw0/sentinel2-lejepa-preset-biased-small}"
DATASET_DIR="${DATASET_DIR:-cache/lejepa_dataset}"

MODEL_REPO_ID="${MODEL_REPO_ID:-alexw0/lejepa-vit-tiny-patch8-sentinel2-5band}"
MODEL_PRIVATE="${MODEL_PRIVATE:-0}"  # set to 1 to publish as a private repo

WANDB_PROJECT="${WANDB_PROJECT:-sentinel-change-lejepa}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-lightning-${ENCODER}-${IMG_SIZE}}"

CHECKPOINT_PATH="checkpoints/lejepa_${ENCODER}_5band.pt"

echo "==> Config"
echo "    encoder     : ${ENCODER}"
echo "    img_size    : ${IMG_SIZE}"
echo "    epochs      : ${EPOCHS}"
echo "    batch_size  : ${BATCH_SIZE}"
echo "    dataset     : ${DATASET_REPO} → ${DATASET_DIR}"
echo "    model repo  : ${MODEL_REPO_ID}"
echo "    wandb run   : ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
echo "    checkpoint  : ${CHECKPOINT_PATH}"

echo "==> Verifying GPU is visible to torch"
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "==> Pulling dataset from HF Hub → ${DATASET_DIR}"
uv run python - <<PY
from datasets import load_dataset
dd = load_dataset("${DATASET_REPO}")
dd.save_to_disk("${DATASET_DIR}")
print("saved:", dd)
PY

echo "==> Writing norm_stats.json alongside dataset"
uv run python - <<'PY'
import json, pathlib
pathlib.Path("cache/lejepa_dataset/norm_stats.json").write_text(json.dumps({
    "bands": ["red", "green", "blue", "nir", "swir16"],
    "mean": [
        1375.7761320782836,
        1197.4988547276673,
        907.2157714102511,
        2513.2625277925504,
        2350.3002480067703,
    ],
    "std": [
        1323.1470322308767,
        1111.3852482659968,
        991.6236564555267,
        1211.1511007885777,
        1463.2444364598798,
    ],
    "pixel_count": 51265536,
    "chip_count": 3129,
}, indent=2))
print("wrote norm_stats.json")
PY

echo "==> Smoke test on GPU (1 epoch, 20 chips) before the real run"
uv run python -m src.experimental.train_lejepa \
    --smoke-test \
    --encoder "${ENCODER}" \
    --dataset "${DATASET_DIR}"

echo "==> Launching real training run in the background (watch train.log or W&B)"
nohup uv run python -m src.experimental.train_lejepa \
    --encoder "${ENCODER}" \
    --dataset "${DATASET_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    > train.log 2>&1 &

TRAIN_PID=$!
echo "    training PID: ${TRAIN_PID}"
echo "    tail logs:    tail -f train.log"
echo "    stop early:   kill ${TRAIN_PID}"

echo "==> Waiting for training to finish before publishing"
# `wait` returns the exit status of the background job; `set -e` then aborts
# the script (and skips the upload step) if training exited non-zero.
wait "${TRAIN_PID}"
echo "==> Training exited cleanly"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "!! expected checkpoint ${CHECKPOINT_PATH} not found — aborting publish" >&2
    exit 1
fi

echo "==> Publishing ${CHECKPOINT_PATH} → https://huggingface.co/${MODEL_REPO_ID}"
PRIVATE_FLAG=()
if [ "${MODEL_PRIVATE}" = "1" ]; then
    PRIVATE_FLAG=(--private)
fi
uv run python -m src.experimental.upload_model \
    --checkpoint "${CHECKPOINT_PATH}" \
    --repo-id "${MODEL_REPO_ID}" \
    --dataset-repo-id "${DATASET_REPO}" \
    "${PRIVATE_FLAG[@]}"

echo "==> Pipeline complete"
echo "    model: https://huggingface.co/${MODEL_REPO_ID}"
echo "    wandb: https://wandb.ai/?project=${WANDB_PROJECT}"
