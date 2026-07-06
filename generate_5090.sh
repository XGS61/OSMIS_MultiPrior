#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EXP_NAME="${EXP_NAME:-test2_online_minimal_5090}"
DATASET_NAME="${DATASET_NAME:-rendered_us_test2_online}"
NUM_GENERATED="${NUM_GENERATED:-50}"
EPOCH="${1:-}"

if [[ -z "${EPOCH}" ]]; then
  EPOCH="$(cat "checkpoints/${EXP_NAME}/models/latest_epoch.txt")"
fi

python -u test.py \
  --exp_name "${EXP_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --which_epoch "${EPOCH}" \
  --num_generated "${NUM_GENERATED}" \
  --batch_size 1 \
  --num_workers 0

echo "Saved to checkpoints/${EXP_NAME}/evaluation/${EPOCH}/"
