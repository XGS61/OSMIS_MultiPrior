#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u verify_5090.py

CASES="${CASES:-test6 test7}"
NUM_EPOCHS="${NUM_EPOCHS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CROP_TOP="${CROP_TOP:-0}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
INIT_FROM_31000_DIR="${INIT_FROM_31000_DIR:-checkpoints/test2_online_minimal_31000_imported/models}"
ANATOMY_MAX_DISPLACEMENT="${ANATOMY_MAX_DISPLACEMENT:-0.04}"
SUPPORT_MAX_DISPLACEMENT="${SUPPORT_MAX_DISPLACEMENT:-0.010}"
SUPPORT_MAX_ROTATION="${SUPPORT_MAX_ROTATION:-1.5}"
SUPPORT_MAX_TRANSLATION="${SUPPORT_MAX_TRANSLATION:-0.010}"
CONTINUE="${CONTINUE:-0}"

train_case() {
  local case_id="$1"
  local source_dir="datasets/pelvic_${case_id}_source"
  local image_path="${source_dir}/image/00000.jpg"
  local binary_mask_path="${source_dir}/mask/00000.png"
  local draft_dir="datasets/pelvic_${case_id}_multilevel_draft"
  local dataset_name="pelvic_${case_id}_hierspade"
  local exp_name="pelvic_${case_id}_hierspade_stable_support_100k"

  if [[ ! -f "${image_path}" ]]; then
    echo "Missing image: ${image_path}" >&2
    exit 1
  fi
  if [[ ! -f "${binary_mask_path}" ]]; then
    echo "Missing binary target mask: ${binary_mask_path}" >&2
    echo "Create it as a black-white PNG with white = levator-hiatus target." >&2
    exit 1
  fi

  python -u tools/create_multilevel_mask_draft.py \
    --image "${image_path}" \
    --mask "${binary_mask_path}" \
    --output "${draft_dir}"

  python -u prepare_online_case.py \
    --image "${image_path}" \
    --mask "${draft_dir}/multilevel_mask_indexed.png" \
    --output "datasets/${dataset_name}" \
    --crop-top "${CROP_TOP}" \
    --overwrite

  python -u validate_single_case.py --dataset "datasets/${dataset_name}"

  mkdir -p "run_logs/${exp_name}"

  local init_args=()
  local continue_args=()
  if [[ "${CONTINUE}" == "1" ]]; then
    continue_args+=(--continue_train)
    echo "Continuing ${exp_name} from latest checkpoint."
  elif [[ -d "${INIT_FROM_31000_DIR}" ]]; then
    init_args+=(--init_from_31000_dir "${INIT_FROM_31000_DIR}")
    echo "Using optional 31000 initialization from ${INIT_FROM_31000_DIR}"
  else
    echo "No initialization directory found; training ${case_id} from scratch."
  fi

  python -u train.py \
    "${continue_args[@]}" \
    --exp_name "${exp_name}" \
    --dataset_name "${dataset_name}" \
    --num_epochs "${NUM_EPOCHS}" \
    --max_size 330 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --use_kornia_augm \
    --prob_augm 0.35 \
    --prob_FA_con 0.15 \
    --prob_FA_lay 0.0 \
    --global_noise_dim 32 \
    --texture_noise_dim 32 \
    --lambda_content 0.5 \
    --lambda_layout 0.15 \
    --lambda_structure 2.0 \
    --lambda_latent 1.0 \
    --lambda_anchor 0.5 \
    --anchor_decay_start 5000 \
    --anchor_decay_end 20000 \
    --anchor_final_ratio 0.10 \
    --anatomy_max_displacement "${ANATOMY_MAX_DISPLACEMENT}" \
    --support_max_displacement "${SUPPORT_MAX_DISPLACEMENT}" \
    --support_max_rotation "${SUPPORT_MAX_ROTATION}" \
    --support_max_translation "${SUPPORT_MAX_TRANSLATION}" \
    --style_dim 32 \
    --freq_print "${SAVE_FREQ}" \
    --freq_save_loss "${SAVE_FREQ}" \
    --freq_save_ckpt "${SAVE_FREQ}" \
    "${init_args[@]}" \
    2>&1 | tee "run_logs/${exp_name}/train.log"
}

for case_id in ${CASES}; do
  train_case "${case_id}"
done
