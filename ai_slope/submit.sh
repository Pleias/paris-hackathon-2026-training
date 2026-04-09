#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:12:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Activate shared venv
# ---------------------------------------------------------------------------
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 2. Stage data to local NVMe on each node (parallel, idempotent)
# ---------------------------------------------------------------------------
srun --ntasks-per-node=1 bash -c \
    'rsync -a --ignore-existing /home/data/ /mnt/local_disk/data/ && \
     echo "[node $SLURM_NODEID] data ready"'

# ---------------------------------------------------------------------------
# 3. Verify max token ID and decide vocab_size (should be <= 32767)
# ---------------------------------------------------------------------------
MAX_TOKEN_ID=$(srun --ntasks=1 --nodes=1 python3 -c "
import numpy as np, glob
files = glob.glob('/mnt/local_disk/data/*.bin')
if not files:
    print(32767)
else:
    print(max(int(np.memmap(f, dtype='uint16', mode='r').max()) for f in files[:5]))
")
echo "[staging] max token ID in first 5 shards: ${MAX_TOKEN_ID}"
# We always use vocab_size=32768 (safe upper bound per design spec)

# ---------------------------------------------------------------------------
# 4. Set rendezvous variables
# ---------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

# NCCL tuning for InfiniBand + NVLink
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# ---------------------------------------------------------------------------
# 5. Launch training — one srun task per node, torchrun spawns 8 workers each
#    Use rendezvous (c10d) instead of --node-rank to avoid SLURM env coupling.
# ---------------------------------------------------------------------------
mkdir -p logs

srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="$SLURM_JOB_ID" \
    /home/tphung/projects/paris-hackathon-2026-training/ai_slope/train.py \
        --tp-size 4 \
        --data-dir /mnt/local_disk/data/ \
        --checkpoint-path "/home/${USER}/checkpoint.pt" \
        --metrics-dir "/home/${USER}/metrics" \
        --time-limit-min 10 \
        --max-steps 1700 \
        --micro-batch 4 \
        --global-batch 512 \
        --max-lr 6e-4

# ---------------------------------------------------------------------------
# 6. Checkpoint is now on shared NFS — training is complete
# ---------------------------------------------------------------------------
echo "[submit] job $SLURM_JOB_ID complete"
