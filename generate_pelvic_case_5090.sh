#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CASE_ID="${1:-test6}"
EPOCH="${2:-}"
NUM_GENERATED="${NUM_GENERATED:-50}"
CROP_TOP="${CROP_TOP:-0}"

SOURCE_DIR="datasets/pelvic_${CASE_ID}_source"
IMAGE_PATH="${SOURCE_DIR}/image/00000.jpg"
BINARY_MASK_PATH="${SOURCE_DIR}/mask/00000.png"
DRAFT_DIR="datasets/pelvic_${CASE_ID}_multilevel_draft"
DATASET_NAME="pelvic_${CASE_ID}_hierspade"
EXP_NAME="${EXP_NAME:-pelvic_${CASE_ID}_hierspade_stable_support_100k}"

if [[ ! -f "${BINARY_MASK_PATH}" ]]; then
  echo "Missing binary target mask: ${BINARY_MASK_PATH}" >&2
  exit 1
fi

if [[ ! -f "${DRAFT_DIR}/multilevel_mask_indexed.png" ]]; then
  python -u tools/create_multilevel_mask_draft.py \
    --image "${IMAGE_PATH}" \
    --mask "${BINARY_MASK_PATH}" \
    --output "${DRAFT_DIR}"
fi

if [[ ! -f "datasets/${DATASET_NAME}/metadata.json" ]]; then
  python -u prepare_online_case.py \
    --image "${IMAGE_PATH}" \
    --mask "${DRAFT_DIR}/multilevel_mask_indexed.png" \
    --output "datasets/${DATASET_NAME}" \
    --crop-top "${CROP_TOP}" \
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
