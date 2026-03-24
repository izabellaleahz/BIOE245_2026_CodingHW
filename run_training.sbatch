#!/bin/bash
#SBATCH --job-name=pathmnist_resnet18
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_%j.out
#SBATCH --error=output/slurm_%j.err

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate torch_gpu_env

cd "$(dirname "$0")"

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "================"

bash train_and_eval.sh

echo "=== Done: $(date) ==="
