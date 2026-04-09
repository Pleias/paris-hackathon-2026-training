#!/bin/bash
#SBATCH --job-name=p-train
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=logs/%j_%N.out
#SBATCH --error=logs/%j_%N.err

set -euo pipefail

mkdir -p logs

NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
echo "Single-node run | GPUs/node: $NPROC_PER_NODE"

source .venv/bin/activate

# ── Single-node launch ───────────────────────────────────────────────────────
torchrun --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    train_yannick.py \
        --data_dir      /home/data/ \
        --checkpoint_path checkpoint.pt \
        --vocab_size       32000 \
        --max_steps        5000 \
        --time_limit_min   10 \
        --wandb_project gpumode


