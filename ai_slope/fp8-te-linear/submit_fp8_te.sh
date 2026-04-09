#!/bin/bash
#SBATCH --job-name=fp8-te-linear
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_fp8_te.out
#SBATCH --error=logs/%j_fp8_te.err

source /home/pleias/yarik-slope/ai_slope/.venv/bin/activate

VENV=/home/pleias/yarik-slope/ai_slope/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$VENV/cu13/lib:$VENV/cublas/lib:$VENV/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
echo "Master: $MASTER_ADDR | Nodes: $SLURM_NNODES | GPUs: 8"

srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    /home/pleias/yarik-slope/ai_slope/fp8-te-linear/train.py \
        --data_dir      /home/data/ \
        --checkpoint_path /home/pleias/yarik-slope/ai_slope/fp8-te-linear/checkpoint.pt \
        --vocab_size    32000 \
        --n_layer       18 \
        --n_head        16 \
        --n_embd        2048 \
        --max_steps     50 \
        --time_limit_min 8 \
        --wandb_project gpumode \
        --wandb_run_name fp8-te-linear-1b-8gpu

echo "=== DONE ==="
