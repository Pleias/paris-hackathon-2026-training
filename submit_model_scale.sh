#!/bin/bash
#SBATCH --job-name=model-scale-2m
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:35:00
#SBATCH --output=logs/%j_model-scale.out
#SBATCH --error=logs/%j_model-scale.err

set -euo pipefail

mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

echo "Master: $MASTER_ADDR:$MASTER_PORT | Nodes: $NNODES | GPUs/node: $NPROC_PER_NODE"

date_tag=$(date +%Y%m%d_%H%M%S)
out_dir="experiments/model_scale_${date_tag}"
mkdir -p "$out_dir"

gpu_total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p')
gpu_total_gib=$(python3 - <<PY
print(${gpu_total_mib} / 1024.0)
PY
)

echo "GPU total memory (rank0 device): ${gpu_total_gib} GiB"

echo "run_name,n_layer,n_head,n_embd,params,steps_completed,final_loss,tokens_per_sec_avg,compute_usage_pct_avg,mem_peak_reserved_gib_max,mem_util_pct_max,summary_path" > "$out_dir/results.csv"

echo "[]" > "$out_dir/results.json"

source .venv/bin/activate

runs=(
  "m110 12 12 768"
  "m261 18 16 1024"
  "m389 22 16 1152"
  "m666 26 22 1408"
  "m958 32 24 1536"
)

for entry in "${runs[@]}"; do
  read -r run_name n_layer n_head n_embd <<< "$entry"
  ckpt_path="$out_dir/${run_name}.pt"

  params=$(python - <<PY
from model import GPT
m = GPT(32768, 1024, ${n_layer}, ${n_head}, ${n_embd}, 0.0)
print(sum(p.numel() for p in m.parameters()))
PY
)

  echo "\n=== Running ${run_name}: layers=${n_layer}, heads=${n_head}, embd=${n_embd}, params=${params} ==="

  srun python -m torch.distributed.run \
      --nnodes="$NNODES" \
      --nproc_per_node="$NPROC_PER_NODE" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
      --rdzv_id="${SLURM_JOB_ID}_${run_name}" \
      train.py \
          --data_dir /home/data/ \
          --checkpoint_path "$ckpt_path" \
          --seq_len 1024 \
          --vocab_size 32768 \
          --n_layer "$n_layer" \
          --n_head "$n_head" \
          --n_embd "$n_embd" \
          --batch_size 8 \
          --grad_accum_steps 4 \
          --max_steps 1000000 \
          --time_limit_min 2

  summary_path=$(ls -1t "$out_dir/${run_name}_summary"*.json 2>/dev/null | head -n1)
  if [[ -z "$summary_path" ]]; then
    echo "Missing summary for ${run_name}" >&2
    exit 1
  fi

  python - <<PY
import json
from pathlib import Path

summary_path = Path("${summary_path}")
out_dir = Path("${out_dir}")
params = int("${params}")
run_name = "${run_name}"
n_layer = int("${n_layer}")
n_head = int("${n_head}")
n_embd = int("${n_embd}")
gpu_total_gib = float("${gpu_total_gib}")

summary = json.loads(summary_path.read_text())["summary"]
mem_peak = float(summary.get("mem_peak_reserved_gib_max", 0.0))
mem_util_pct = (100.0 * mem_peak / gpu_total_gib) if gpu_total_gib > 0 else 0.0

row = {
    "run_name": run_name,
    "n_layer": n_layer,
    "n_head": n_head,
    "n_embd": n_embd,
    "params": params,
    "steps_completed": int(summary.get("steps_completed", 0)),
    "final_loss": summary.get("final_loss", None),
    "tokens_per_sec_avg": float(summary.get("tokens_per_sec_avg", 0.0)),
    "compute_usage_pct_avg": float(summary.get("compute_usage_pct_avg", 0.0)),
    "mem_peak_reserved_gib_max": mem_peak,
    "mem_util_pct_max": mem_util_pct,
    "summary_path": str(summary_path),
}

csv_path = out_dir / "results.csv"
with csv_path.open("a", encoding="utf-8") as f:
    f.write(
        f"{row['run_name']},{row['n_layer']},{row['n_head']},{row['n_embd']},{row['params']},"
        f"{row['steps_completed']},{row['final_loss']},{row['tokens_per_sec_avg']},"
        f"{row['compute_usage_pct_avg']},{row['mem_peak_reserved_gib_max']},"
        f"{row['mem_util_pct_max']},{row['summary_path']}\n"
    )

json_path = out_dir / "results.json"
records = json.loads(json_path.read_text())
records.append(row)
json_path.write_text(json.dumps(records, indent=2))

print("Recorded", row)
PY

done

echo "\nAll runs complete."
echo "Results CSV: $out_dir/results.csv"
echo "Results JSON: $out_dir/results.json"
