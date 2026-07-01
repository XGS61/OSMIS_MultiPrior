#!/usr/bin/env bash
set -euo pipefail

NUM_EPOCHS=2 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
NUM_VARIANTS=8 \
bash train_improved.sh smoke_atg_osmis
