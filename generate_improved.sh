#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EXP_NAME="${1:-rendered_us_atg_osmis_v1}"
EPOCH="${2:-150000}"
NUM_GENERATED="${3:-50}"

python -u test.py \
  --exp_name "${EXP_NAME}" \
  --which_epoch "${EPOCH}" \
  --num_generated "${NUM_GENERATED}"

echo "Results: checkpoints/${EXP_NAME}/evaluation/${EPOCH}/"
