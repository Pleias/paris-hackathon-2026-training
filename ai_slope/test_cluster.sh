#!/bin/bash
#SBATCH --job-name=test-cluster
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:05:00
#SBATCH --output=logs_yarik/%j_test.out
#SBATCH --error=logs_yarik/%j_test.err

set -euo pipefail
mkdir -p logs

echo "=== Node: $(hostname) ==="
echo "=== Date: $(date) ==="
echo ""

echo "=== GPU Info ==="
nvidia-smi
echo ""

echo "=== GPU Topology ==="
nvidia-smi topo -m
echo ""

echo "=== InfiniBand ==="
ibstat 2>/dev/null || echo "No InfiniBand found"
echo ""

echo "=== CUDA version ==="
nvcc --version 2>/dev/null || echo "nvcc not found"
echo ""

echo "=== Python ==="
which python 2>/dev/null && python --version || echo "No python in PATH"
echo ""

echo "=== Data check ==="
ls /home/data/chunk*.bin | wc -l
ls -lh /home/data/chunk_0001.bin
echo ""

echo "=== Local disk ==="
df -h /mnt/local_disk
echo ""

echo "=== Network interfaces ==="
ip link show | grep -E '^[0-9]+:|state'
echo ""

echo "=== Done ==="
