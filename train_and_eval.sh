#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_ROOT="${SCRIPT_DIR}/data/PathMNIST"
OUTPUT_ROOT="${SCRIPT_DIR}/output"

python ./train_and_eval.py \
    --download \
    --output_root ${OUTPUT_ROOT} \
    --gpu_ids 0 \
    --dataset_root ${DATASET_ROOT}