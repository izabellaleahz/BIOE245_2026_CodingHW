#!/bin/bash
#SBATCH --job-name=dermamnist_resnet18
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_derma_%j.out
#SBATCH --error=output/slurm_derma_%j.err

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate torch_gpu_env

cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
DATASET_ROOT="${SCRIPT_DIR}/data/DermaMNIST"
OUTPUT_ROOT="${SCRIPT_DIR}/output"

echo "=== DermaMNIST Training ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"

python ./train_and_eval.py \
    --data_flag dermamnist \
    --download \
    --output_root ${OUTPUT_ROOT} \
    --gpu_ids 0 \
    --dataset_root ${DATASET_ROOT}

echo "=== Done: $(date) ==="
