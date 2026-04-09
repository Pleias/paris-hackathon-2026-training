#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j_dryrun.out
#SBATCH --error=logs/%j_dryrun.err

source /home/pleias/yarik-slope/ai_slope/.venv/bin/activate
VENV=/home/pleias/yarik-slope/ai_slope/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$VENV/cu13/lib:$VENV/cublas/lib:$VENV/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

echo "=== Python/PyTorch ==="
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0)}')"

echo "=== Megatron-Core ==="
python -c "from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding; print('megatron-core imports OK')"

echo "=== Transformer Engine ==="
python -c "
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
r = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)
try:
    ctx = te.fp8_autocast(enabled=True, fp8_recipe=r)
    print('TE fp8_autocast works with fp8_recipe=')
except TypeError:
    ctx = te.fp8_autocast(enabled=True, recipe=r)
    print('TE fp8_autocast works with recipe=')
print(f'TE version: {te.__version__ if hasattr(te, \"__version__\") else \"unknown\"}')
"

echo "=== Model on GPU ==="
cd /home/pleias/yarik-slope
python -c "
import sys; sys.path.insert(0, 'ai_slope')
from model import get_model
import torch
cfg = {'hidden_size': 256, 'num_layers': 2, 'num_attention_heads': 4,
'num_key_value_heads': 2, 'intermediate_size': 512, 'vocab_size': 1024, 'seq_len': 64}
m = get_model(cfg).cuda().bfloat16()
x = torch.randint(0, 1024, (2, 64), device='cuda')
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits, loss = m(x, targets=x)
print(f'GPU forward: logits={logits.shape}, loss={loss.item():.4f}')
print(f'GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GiB')
"

echo "=== Data check ==="
python -c "
import numpy as np, glob
files = sorted(glob.glob('/home/data/chunk*.bin'))
print(f'Data shards: {len(files)}')
d = np.memmap(files[0], dtype='uint16', mode='r')
print(f'First shard: {len(d):,} tokens, max_id={d.max()}, min_id={d.min()}')
"

echo "=== torch.compile ==="
python -c "
import torch
m = torch.nn.Linear(256, 256).cuda()
m = torch.compile(m, mode='max-autotune', fullgraph=False)
x = torch.randn(2, 256, device='cuda')
y = m(x)
print(f'torch.compile OK, output={y.shape}')
"

echo "=== DONE ==="
