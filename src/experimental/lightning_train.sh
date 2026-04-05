#!/usr/bin/env bash
# Bootstrap + run LeJEPA training on a lightning.ai GPU studio.
#
# Assumes: repo already cloned, you are in the repo root, branch checked out.
# Prereqs you must do interactively BEFORE running this script:
#   1. curl -LsSf https://astral.sh/uv/install.sh | sh  (and source the env)
#   2. uv sync --extra experimental
#   3. uv run hf auth login     (HF write token)
#   4. uv run wandb login       (W&B API key)
#
# Then: bash src/experimental/lightning_train.sh
set -euo pipefail

DATASET_REPO="alexw0/sentinel2-lejepa-preset-biased-small"
DATASET_DIR="cache/lejepa_dataset"
WANDB_PROJECT="sentinel-change-lejepa"
WANDB_RUN_NAME="lightning-gpu-full"

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
uv run python -m src.experimental.train_lejepa --smoke-test --dataset "${DATASET_DIR}"

echo "==> Launching real training run in the background (watch train.log or W&B)"
nohup uv run python -m src.experimental.train_lejepa \
    --dataset "${DATASET_DIR}" \
    --epochs 50 --batch-size 128 --num-workers 4 \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    > train.log 2>&1 &

TRAIN_PID=$!
echo "Training PID: ${TRAIN_PID}"
echo "Tail logs with:  tail -f train.log"
echo "Stop with:       kill ${TRAIN_PID}"
