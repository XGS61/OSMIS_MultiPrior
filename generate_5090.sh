#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EXP_NAME="${EXP_NAME:-test2_fullsean_hierarchical_5090}"
DATASET_NAME="${DATASET_NAME:-rendered_us_test2_fullsean}"
NUM_GENERATED="${NUM_GENERATED:-50}"
EPOCH="${1:-}"

if [[ ! -f "datasets/${DATASET_NAME}/metadata.json" ]]; then
  python -u prepare_hierarchical_case.py \
    --image "datasets/rendered_us_test2_source/image/00000.jpg" \
    --indexed-mask "datasets/rendered_us_test2_multilevel_draft/multilevel_mask_indexed.png" \
    --conditions "datasets/rendered_us_test2_multilevel_draft/hierarchical_conditions.npz" \
    --output "datasets/${DATASET_NAME}" \
    --crop-top 20 \
    --overwrite
fi

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
