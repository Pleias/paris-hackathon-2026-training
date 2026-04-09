#!/bin/bash
# Single H100 1B GPT training — plain Ubuntu server, no SLURM

set -e

echo "=== Single H100 1B GPT training ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py \
    --data_dir      /home/data/ \
    --checkpoint_path ./checkpoint.pt \
    --vocab_size    32000 \
    --n_layer       18 \
    --n_head        16 \
    --n_embd        2048 \
    --batch_size    4 \
    --grad_accum_steps 8 \
    --max_steps     50 \
    --time_limit_min 25 \
    --wandb_project gpumode \
    --wandb_run_name h100-1b-single-fp8

echo "=== DONE ==="
