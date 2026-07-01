#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EXP_NAME="${1:-rendered_us_atg_osmis_v1}"
IMAGE_PATH="${IMAGE_PATH:-datasets/rendered_us_3d_1/image/00000.png}"
MASK_PATH="${MASK_PATH:-datasets/rendered_us_3d_1/mask/00000.png}"
DATASET_NAME="${DATASET_NAME:-rendered_us_3d_1_anatomy}"
NUM_VARIANTS="${NUM_VARIANTS:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-150000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

python -u prepare_anatomy_dataset.py \
  --image "${IMAGE_PATH}" \
  --mask "${MASK_PATH}" \
  --output "datasets/${DATASET_NAME}" \
  --num-variants "${NUM_VARIANTS}" \
  --overwrite

mkdir -p "run_logs/${EXP_NAME}"
python -u train.py \
  --exp_name "${EXP_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --num_epochs "${NUM_EPOCHS}" \
  --max_size 330 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --use_kornia_augm \
  --prob_augm 0.35 \
  --prob_FA_con 0.15 \
  --prob_FA_lay 0.0 \
  --lambda_DR 0.05 \
  --lambda_seg 10.0 \
  --lambda_boundary 2.0 \
  --lambda_lowfreq 2.0 \
  --lambda_texture 1.0 \
  --freq_print 1000 \
  --freq_save_loss 1000 \
  --freq_save_ckpt 1000 \
  2>&1 | tee "run_logs/${EXP_NAME}/train.log"
