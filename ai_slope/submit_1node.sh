#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:08:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source /home/pleias/yarik-slope/ai_slope/.venv/bin/activate

VENV=/home/pleias/yarik-slope/ai_slope/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$VENV/cu13/lib:$VENV/cublas/lib:$VENV/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

# NCCL tuning
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Master: $MASTER_ADDR | Nodes: $SLURM_NNODES | GPUs: 8"
echo "TP=4, DP=2, world_size=8"

mkdir -p logs

# 1 node x 8 GPUs: TP=4, DP=2
# global_batch = micro_batch * dp_size * grad_accum = 4 * 2 * 16 = 128
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id=$SLURM_JOB_ID \
    /home/pleias/yarik-slope/ai_slope/train.py \
        --tp-size 4 \
        --data-dir /home/data/ \
        --checkpoint-path /home/pleias/yarik-slope/ai_slope/checkpoint.pt \
        --metrics-dir /home/pleias/yarik-slope/ai_slope/metrics \
        --time-limit-min 8 \
        --max-steps 20 \
        --micro-batch 4 \
        --global-batch 128 \
        --max-lr 6e-4 \
        --wandb-project gpumode \
        --wandb-run-name llama7b-fp8-1node

echo "[submit] job $SLURM_JOB_ID complete"
