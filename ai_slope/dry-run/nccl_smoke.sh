#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j_nccl.out
#SBATCH --error=logs/%j_nccl.err

source /home/pleias/yarik-slope/ai_slope/.venv/bin/activate
export LD_LIBRARY_PATH=/home/pleias/yarik-slope/ai_slope/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "Nodes: $SLURM_NNODES, GPUs per node: 8, World size: $((SLURM_NNODES * 8))"

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    /home/pleias/yarik-slope/ai_slope/dry-run/nccl_test.py

echo "=== DONE ==="
