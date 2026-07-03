#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u verify_5090.py

EXP_NAME="${EXP_NAME:-test2_multiprior_5090}"
DATASET_NAME="${DATASET_NAME:-rendered_us_test2_multiprior}"
IMAGE_PATH="${IMAGE_PATH:-datasets/rendered_us_test2_source/image/00000.jpg}"
MASK_PATH="${MASK_PATH:-datasets/rendered_us_test2_source/mask/00000.png}"
NUM_MASK_PRIORS="${NUM_MASK_PRIORS:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

python -u prepare_single_case.py \
  --image "${IMAGE_PATH}" \
  --mask "${MASK_PATH}" \
  --output "datasets/${DATASET_NAME}" \
  --num-mask-priors "${NUM_MASK_PRIORS}" \
  --crop-top 20 \
  --overwrite

python -u validate_single_case.py \
  --dataset "datasets/${DATASET_NAME}" \
  --expected-priors "${NUM_MASK_PRIORS}"

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
  --lambda_DR 0.03 \
  --lambda_texture 1.0 \
  --lambda_frequency 0.25 \
  --lambda_alignment 2.0 \
  --lambda_same_mask_div 0.2 \
  --lambda_anchor 0.5 \
  --style_dim 32 \
  --freq_print 1000 \
  --freq_save_loss 1000 \
  --freq_save_ckpt 1000 \
  2>&1 | tee "run_logs/${EXP_NAME}/train.log"
