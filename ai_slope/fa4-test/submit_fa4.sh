#!/bin/bash
#SBATCH --job-name=fa4-test
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_fa4.out
#SBATCH --error=logs/%j_fa4.err

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
    /home/pleias/yarik-slope/ai_slope/fa4-test/train.py \
        --data_dir      /home/data/ \
        --checkpoint_path /home/pleias/yarik-slope/ai_slope/fa4-test/checkpoint.pt \
        --vocab_size    32000 \
        --max_steps     50 \
        --time_limit_min 8 \
        --require_fa4 true \
        --wandb_project paris-hackathon \
        --wandb_run_name fa4-test-8gpu

echo "=== DONE ==="
