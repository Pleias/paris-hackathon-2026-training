# Megatron-Core Llama-7B Training Design

## Context

Paris Hackathon 2026 — train the best LLM possible in 10 minutes of GPU time on a 32-GPU cluster (4 nodes x 8x B300 SXM6, 275 GiB VRAM each). Evaluation: validation loss (perplexity), HellaSwag accuracy as tiebreaker.

No direct SSH access to workers — all execution via SLURM from the hub host.

## Approach

**Megatron-Core directly** (without the full NeMo wrapper). Megatron-Core provides battle-tested tensor parallelism, FP8 via Transformer Engine, and Llama architecture support with fewer dependencies than full NeMo.

## Deliverables

| File | Purpose |
|---|---|
| `requirements.txt` | PyTorch (nightly/2.7+), megatron-core, transformer-engine |
| `model.py` | Llama-7B architecture via Megatron-Core, exposes `get_model(config) -> nn.Module` with `forward(idx, targets=None) -> (logits, loss)` |
| `train.py` | Imports model from `model.py` via `get_model()`, runs custom Megatron-Core training loop |
| `submit.sh` | SLURM job: data staging + training launch |

### model.py / train.py Contract

`train.py` imports `get_model` from `model.py` to create the model. This keeps architecture definition separate from training logic and satisfies the competition evaluation contract.

```python
# train.py usage:
from model import get_model
model = get_model(config)
# ... wrap with Megatron-Core parallelism, train, save checkpoint
```

### Training Loop Strategy

We write a **custom training loop** (not `megatron.core.training.pretrain()`) for two reasons:

1. `pretrain()` requires a specific dataset format (Megatron indexed `.bin`) and a tightly coupled dataset provider API. Converting our uint16 shards would cost wall time we cannot afford.
2. A custom loop lets us plug in our existing `BinDataset` memory-mapped reader directly, use our own time-limit logic, and implement per-step NaN detection without fighting against Megatron's callback system.

Megatron-Core is still worth it because we use it for its **parallelism primitives** (`ColumnParallelLinear`, `RowParallelLinear`, `TensorParallelAttention`) and **Transformer Engine FP8** integration — these are the performance-critical components. The training loop itself is straightforward to write.

## Model Architecture

Llama-2 7B configuration with competition vocab:

| Parameter | Value |
|---|---|
| `hidden_size` | 4096 |
| `num_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA, 4:1 ratio) |
| `intermediate_size` | 11008 (SwiGLU) |
| `vocab_size` | 32,000 (per README) |
| `seq_len` | 1024 (matches evaluation seq_len) |
| `positional_encoding` | RoPE (theta=10000) |
| `normalization` | RMSNorm |
| `activation` | SiLU (SwiGLU MLP) |
| `attention_bias` | false |
| `tie_word_embeddings` | false |

**Vocab size note:** The README specifies 32,000. The baseline `train.py` uses 32,768. Before training, verify the actual maximum token ID in the data shards (e.g., `numpy.max` over a few shards). Use 32,000 as specified in the README but pad to 32,768 if any token ID >= 32,000 is found, to avoid index errors. This check costs negligible time and should happen during data staging.

**Why seq_len=1024 and not longer:** The evaluation set uses seq_len=1024. Training at a longer sequence length (2048 or 4096) wastes FLOPs on attention over positions that are never tested, and the model learns positional patterns that don't transfer to the shorter eval context. Training at seq_len=1024 means every FLOP directly targets what is evaluated. It also halves tokens-per-step vs seq_len=2048, which halves step time and roughly doubles the number of gradient updates in 10 minutes — a significant advantage for convergence.

## Parallelism Strategy

- **Tensor Parallel (TP=4)** within each node — uses NV18 all-to-all NVLink (~900 GB/s intra-node). Splits attention heads and FFN columns across 4 GPUs.
- **Data Parallel (DP=8)** across 2 GPUs per node x 4 nodes — gradient all-reduce over InfiniBand (800 Gb/s per GPU, rail-optimized).
- No pipeline parallelism or FSDP — unnecessary at this scale with 275 GiB/GPU.

**Why TP=4 not TP=8:** With 8 KV heads and TP=8, each GPU holds exactly 1 KV head, which is degenerate GQA — the grouped-query attention optimization provides no benefit (no sharing across heads). With TP=4, each GPU holds 2 KV heads and 8 query heads per group, which preserves the intended 4:1 GQA ratio. TP=4 also yields larger GEMM tiles (better GPU utilization) and halves the TP all-reduce communication volume. DP=8 still gives strong gradient averaging across 8 independent data streams.

## Precision

- **FP8 via Transformer Engine** — primary mode. Nearly doubles throughput vs BF16 on Blackwell B300.
- **Fallback: BF16 AMP** — if Transformer Engine doesn't support CUDA 13.0 / Blackwell yet. See BF16 fallback config section below.

## Training Configuration

| Parameter | Value |
|---|---|
| Global batch size (GBS) | 512 sequences (512K tokens/step at seq_len=1024) |
| Micro-batch size per GPU | 4 sequences |
| Gradient accumulation steps | GBS / (micro_batch x DP) = 512 / (4 x 8) = 16 steps |
| Optimizer | Megatron-Core distributed Adam (`use_distributed_optimizer=True`) |
| Weight decay | 0.1 (on params with dim >= 2) |
| LR schedule | Cosine decay with linear warmup |
| Peak LR | 6e-4 |
| Min LR | 6e-5 |
| Warmup | 100 steps |
| Gradient clipping | 1.0 |
| Time limit | 10 minutes (hard stop from first forward pass) |
| Max wall time | 12 minutes (SLURM, allows for NCCL init + checkpoint) |
| Estimated max_steps | ~1,700 conservative / ~2,200 optimistic (see throughput section) |

**Micro-batch and gradient accumulation rationale:** micro_batch=4 keeps activation memory well within budget even with activation checkpointing. 16 accumulation steps per DP group achieves GBS=512 without requiring a synchronize every micro-step. `use_distributed_optimizer=True` shards the Adam state across DP ranks, reducing per-GPU optimizer memory from ~24 GiB to ~3 GiB — freeing headroom for larger batches or future scaling.

**LR rationale:** Peak LR of 6e-4 is appropriate for the severely undertrained regime (<<1 token per parameter). The model trains from scratch with a very short schedule (~1,700-2,200 steps); a higher LR of 6e-4-1e-3 accelerates early loss descent, which is where we spend most of our compute budget. Values below 3e-4 leave loss improvement on the table in this window.

**Warmup rationale:** 100-step warmup allows the optimizer to warm up across ~51M tokens before hitting peak LR. A 50-step warmup is too aggressive for a 7B model — large initial gradient spikes can cause early divergence. 100 steps adds only ~6% overhead on a ~1,700-step run.

### Activation Checkpointing

**Required.** With GBS=512, micro_batch=4, seq_len=1024, and TP=4, activation memory per GPU is still material. Use **selective recompute** (Megatron-Core `--recompute-granularity=selective`): recompute only the attention softmax and dropout activations (the memory-intensive but cheap-to-recompute parts), retaining all linear layer outputs. This recovers ~40% of activation memory at ~5% throughput cost.

If OOM is observed at micro_batch=4, fall back to `--recompute-granularity=full` (full layer recompute) at a ~15% throughput cost, or reduce micro_batch to 2.

## Throughput Estimation & Go/No-Go

### 7B Model FLOPs

The standard estimate for transformer training FLOPs per token is 6 × N (forward + backward):

- N = 6.7B parameters (Llama-7B)
- FLOPs per token = 6 × 6.7e9 ≈ 40.2 TFLOPs/token

### Cluster Peak Throughput

- B300 FP8 peak: ~5 PFLOPS = 5,000 TFLOPs/GPU
- 32 GPUs: 160,000 TFLOPs/s theoretical peak
- **Conservative MFU (primary estimate): 40%** — first runs on new hardware with Megatron-Core typically land here due to communication overhead, kernel tuning, and suboptimal scheduling
- **Optimistic MFU: 50%** — achievable after profiling and tuning, or with ideal conditions
- **Primary achieved throughput (40% MFU):** 64,000 TFLOPs/s
- **Optimistic achieved throughput (50% MFU):** 80,000 TFLOPs/s

### Tokens per Second

**Conservative (40% MFU):**
- tokens/s = 64,000 / 40.2 ≈ **1,592,000 tokens/s** (~1.6M tok/s globally)
- Per step (GBS=512, seq_len=1024): 512 × 1024 = 524,288 tokens
- Step time: 524,288 / 1,592,000 ≈ **~329 ms/step**

**Optimistic (50% MFU):**
- tokens/s = 80,000 / 40.2 ≈ **2,000,000 tokens/s** (~2M tok/s globally)
- Step time: 524,288 / 2,000,000 ≈ **~262 ms/step**

### Total Steps in 10 Minutes

- Effective training time: 600s - 45s (torch.compile warmup on first ~4 steps) = **~555s** of productive training

**Conservative (40% MFU, primary):**
- 555s / 0.329s ≈ **~1,687 steps** → use max_steps=1,700 as the primary configured ceiling
- Total tokens: 1,700 × 524,288 ≈ **891 million tokens**
- Tokens per parameter: 0.89B / 6.7B ≈ 0.13

**Optimistic (50% MFU):**
- 555s / 0.262s ≈ **~2,118 steps** → max_steps=2,200 ceiling if throughput is confirmed higher

**Why this is better than seq_len=2048:** Same total token budget (~890M tokens conservative), but nearly **double the gradient updates** (~1,700 vs ~850 steps). Each update carries an independent gradient signal, which is more valuable for convergence than processing the same tokens in fewer, larger steps. Additionally, every training FLOP targets the exact sequence length used at evaluation time — no wasted computation on positions 1024-2047 that are never tested.

**Cosine schedule:** LR cosine decay runs from warmup (100 steps) to max_steps (1,700 conservative / 2,200 optimistic). The schedule end is set to max_steps; if training hits the time limit early, the last LR used is whatever step it reached. Warmup represents 100/1,700 ≈ 6% of steps at the conservative estimate.

**torch.compile overhead:** The first few forward passes trigger compilation (~30-45s spread across steps 1-4). This is included in the 600s training budget (the clock starts at first forward pass). An alternative is to run a dummy forward pass during data staging (before the timer starts) to pre-compile, which would recover ~45s and increase conservative max_steps to ~1,824.

### Go/No-Go Criteria

**Measure step time from steps 5-10** (after torch.compile warmup completes on steps 1-4). If the median step time across steps 5-10 exceeds **450 ms** (at seq_len=1024), the 7B model will complete fewer than 1,200 useful steps in 10 minutes, which is likely insufficient for competitive loss. In that case, abort and relaunch with the 3B fallback config immediately. This check costs only ~5-7 seconds of training time (negligible), unlike a separate 60-second benchmark that would consume 10% of the training budget.

```python
if step == 10:
    median_step_ms = sorted(step_times[4:10])[3]  # median of steps 5-10
    if median_step_ms > 450:
        log(f"[go/no-go] median step time {median_step_ms:.0f}ms > 450ms, aborting 7B → 3B fallback")
        abort_and_launch_3b()
```

### 3B Fallback Configuration

| Parameter | Value |
|---|---|
| `hidden_size` | 3072 |
| `num_layers` | 26 |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 8 |
| `intermediate_size` | 8192 |
| `vocab_size` | 32,000 |
| Micro-batch size | 8 |
| Global batch size | 1024 |
| Peak LR | 8e-4 |

3B FLOPs/token ≈ 18 TFLOPs. At seq_len=1024 with GBS=1024: 1,048,576 tokens/step. tokens/s = 64,000/18 ≈ 3.56M tok/s at 40% MFU. Step time: 1,048,576 / 3,556,000 ≈ ~295 ms/step → ~1,881 steps in 555s (after compile warmup) → ~1.97B tokens → 0.66 tok/param. Meaningfully better trained.

## Data Loading

Custom memory-mapped data loader (not Megatron's native indexed format):

- Reads raw pre-tokenized uint16 `.bin` shards directly from `/mnt/local_disk/data/`
- 49 shards, ~92 GiB total, ~49 billion tokens
- Random sampling with shard weighting (proportional to shard size)
- Draws (seq_len + 1)-token windows, splits into input/target pairs
- Based on the existing `BinDataset` pattern in the baseline `train.py`
- **Vectorized batch sampling:** draw `batch_size` random (shard, offset) pairs in a single `numpy.random.choice` call, then stack with `numpy.stack` before converting to torch — avoids a Python loop over batch items

This avoids a data conversion step that would eat into wall time.

## Data Staging

Each node copies training data from NFS to local NVMe before training starts (off the clock):

```bash
# In submit.sh, before launching training
srun --ntasks-per-node=1 bash -c 'rsync -a --ignore-existing /home/data/ /mnt/local_disk/data/'
```

- Source: `/home/data/` (shared NFS, 92 GiB)
- Destination: `/mnt/local_disk/data/` (local NVMe, 7 TiB available)
- Runs in parallel across all 4 nodes
- `--ignore-existing` skips files already copied (safe to re-run, idempotent)
- Completes before first forward pass — does not count against 10-min limit
- Also: verify max token ID during staging: `python -c "import numpy as np, glob; print(max(np.memmap(f,'uint16','r').max() for f in glob.glob('/mnt/local_disk/data/*.bin')))"`

## Checkpoint & Evaluation

### Saving

- During training, the best checkpoint is **not written to NFS immediately** on each validation improvement. Writing 13 GiB to NFS on every improvement interrupts training for several seconds. Instead, **every TP rank** saves its own shard of the best checkpoint `state_dict` to **CPU RAM** (a `{k: v.cpu() ...}` copy, ~13 GiB / TP_size ≈ 3.3 GiB per rank, well within the 200 GiB available per node).
- At end of training, all TP ranks in DP group 0 gather their shards to rank 0 via `dist.gather` or by rank 0 receiving each shard over the TP process group. Rank 0 then concatenates the TP-sharded weights back into full tensors, applies the Megatron-to-Llama weight remapping (see below), and writes the consolidated checkpoint.
- Both the final checkpoint and the best checkpoint are written to shared NFS in a single pass at the end: `{"step": int, "model": state_dict, "config": dict}`.
- This means NFS I/O happens exactly once, at teardown, and never competes with the training loop.

### Weight Mapping (Megatron-Core → Vanilla Llama)

Megatron-Core uses different parameter name conventions than vanilla Llama (used by the competition's `get_model()` / `load_state_dict()` contract). A remapping step is required at checkpoint save time.

Concrete mapping (Megatron-Core name → vanilla Llama name):

```python
MEGATRON_TO_LLAMA = {
    # Embeddings
    "embedding.word_embeddings.weight":                              "tok_embeddings.weight",
    # Transformer layers (template, apply for i in range(num_layers))
    # linear_qkv.weight is fused in Megatron — must be split into wq, wk, wv
    "decoder.layers.{i}.self_attention.linear_qkv.weight":          "layers.{i}.attention.wq.weight",   # after split [0]
    #                                                                  "layers.{i}.attention.wk.weight",  # after split [1]
    #                                                                  "layers.{i}.attention.wv.weight",  # after split [2]
    "decoder.layers.{i}.self_attention.linear_proj.weight":         "layers.{i}.attention.wo.weight",
    # linear_fc1.weight is fused gate+up in Megatron — must be split into w1, w3
    "decoder.layers.{i}.mlp.linear_fc1.weight":                     "layers.{i}.feed_forward.w1.weight", # after split [0]
    #                                                                  "layers.{i}.feed_forward.w3.weight", # after split [1]
    "decoder.layers.{i}.mlp.linear_fc2.weight":                     "layers.{i}.feed_forward.w2.weight",
    "decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight":"layers.{i}.attention_norm.weight",
    "decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight":          "layers.{i}.ffn_norm.weight",
    # Final norm and output
    "decoder.final_layernorm.weight":                                "norm.weight",
    "output_layer.weight":                                           "output.weight",
}
```

**model.py parameter naming:** `model.py` uses **separate Q, K, V weight matrices** (standard Llama convention), not a fused QKV tensor. The exact parameter names in `model.py` for each attention layer are:
- `layers.{i}.attention.wq` — `nn.Linear(hidden_size, num_heads * head_dim, bias=False)`
- `layers.{i}.attention.wk` — `nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)`
- `layers.{i}.attention.wv` — `nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)`
- `layers.{i}.attention.wo` — `nn.Linear(num_heads * head_dim, hidden_size, bias=False)`

For the MLP (SwiGLU), `model.py` uses separate gate and up-projection matrices:
- `layers.{i}.feed_forward.w1` — gate projection, `nn.Linear(hidden_size, intermediate_size, bias=False)`
- `layers.{i}.feed_forward.w3` — up projection, `nn.Linear(hidden_size, intermediate_size, bias=False)`
- `layers.{i}.feed_forward.w2` — down projection, `nn.Linear(intermediate_size, hidden_size, bias=False)`

**Split operations at checkpoint save time:**

- `linear_qkv.weight` (Megatron fused, shape `[(H_q + 2*H_kv), H]` for GQA with H_q=4096, H_kv=1024) must be split along dim 0 into `wq.weight` (shape `[4096, 4096]`), `wk.weight` (shape `[1024, 4096]`), `wv.weight` (shape `[1024, 4096]`).
- `linear_fc1.weight` (Megatron fused, shape `[2*FFN, H]` = `[22016, 4096]`) must be split along dim 0 into `w1.weight` (`[11008, 4096]`) and `w3.weight` (`[11008, 4096]`).

This remapping is implemented as a `consolidate_and_remap(ckpt_dir) -> state_dict` function called on rank 0 before saving the final `checkpoint.pt`. The `config` dict saved alongside uses the vanilla Llama key names so `get_model(config)` reconstructs correctly.

## Validation & NaN Detection

### Periodic Validation Loss (every 100 steps)

Reserve ~1% of the data shards (first half of `chunk_0001.bin`) as a held-out validation set. Every 100 training steps, run a validation pass on **all ranks within the first TP group (DP rank 0)**. With TP=4, rank 0 alone only holds 1/4 of the model weights — a forward pass on rank 0 in isolation will crash. The validation forward pass must execute across all 4 ranks of the TP group together. Only the first DP group (ranks 0–3, i.e., TP group 0) runs validation; other DP groups skip. The loss is reduced across TP ranks and logged by global rank 0:

```python
if step % 100 == 0:
    # All ranks in TP group 0 (dp_rank == 0) participate; other DP groups skip
    if dp_rank == 0:
        model.eval()
        with torch.no_grad():
            val_loss = compute_val_loss(val_dataset, model, n_batches=20)
            # val_loss is already reduced across TP ranks inside compute_val_loss
        model.train()
        # ALL TP ranks save their shard to CPU RAM (each rank holds 1/TP of the model)
        if tp_rank == 0:
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
        # Broadcast the is_best decision from tp_rank 0 to all TP ranks
        is_best_tensor = torch.tensor([int(is_best)] if tp_rank == 0 else [0], device=device)
        dist.broadcast(is_best_tensor, src=0, group=tp_group)
        if is_best_tensor.item():
            # Every TP rank saves its own shard to CPU RAM
            best_ckpt_shard = {k: v.cpu() for k, v in model.state_dict().items()}
            best_ckpt_step = step
        if tp_rank == 0:
            log(f"step={step} val_loss={val_loss:.4f}{' (new best)' if is_best_tensor.item() else ''}")
```

`compute_val_loss` runs `n_batches=20` micro-batches through the TP-sharded model (all TP ranks process each batch together), then does `dist.all_reduce(loss, group=tp_group)` / `tp_size` to get the mean loss across TP ranks.

The best checkpoint by validation loss is held in CPU RAM during training and written to NFS only at the end (see Checkpoint section). The evaluator will load the best checkpoint if it exists.

### NaN / Divergence Detection

Check training loss every step. If loss is NaN or Inf, or if loss > 20.0 after step 50 (indicates divergence):

1. Log the event with step number and last good loss
2. Reload the last saved checkpoint
3. Halve the learning rate (`peak_lr *= 0.5`)
4. Resume training from that checkpoint

```python
if math.isnan(loss) or math.isinf(loss) or (step > 50 and loss > 20.0):
    print(f"[warn] NaN/divergence at step {step}, reloading ckpt, halving LR")
    load_checkpoint(model, optimizer, last_ckpt_path)
    peak_lr *= 0.5
    # update optimizer param groups
    continue
```

Allow at most 2 recovery attempts; if divergence persists after 2 halvings, log and continue (may stabilize).

## torch.compile

Add `torch.compile` to the training model before the first step:

```python
model = torch.compile(model, mode="max-autotune", fullgraph=False)
```

**Compatibility notes with Megatron-Core:**

- Use `fullgraph=False` — Megatron-Core uses Python control flow that breaks full-graph tracing.
- Compile after DDP/TP wrapping (compile the wrapped model, not the base module).
- On first step, compilation may take 30-90 seconds — this happens before the 10-minute clock starts only if placed before the first `optimizer.step()`. Adjust the timer start point accordingly, or accept that compilation is part of the 10-minute budget (it is: the clock starts at first forward pass, and compilation happens at first forward pass).
- If `torch.compile` causes errors with Transformer Engine FP8 ops, wrap only the non-TE layers and leave TE ops uncompiled (`torch._dynamo.disable` on TE modules).
- Expected benefit: 10-20% throughput improvement on Blackwell beyond Megatron-Core's existing kernel optimizations.

## SLURM Job Flow (`submit.sh`)

```bash
#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:12:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# 1. Activate shared .venv
source /home/.venv/bin/activate

# 2. Stage data to local NVMe on each node (parallel, idempotent)
srun --ntasks-per-node=1 bash -c \
  'rsync -a --ignore-existing /home/data/ /mnt/local_disk/data/ && echo "[$SLURM_NODEID] data ready"'

# 3. Launch training: one srun task per node, torchrun spawns 8 GPU workers per node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=8 \
  --node-rank=$SLURM_NODEID \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
  train.py \
    --tp-size 4 \
    --data-dir /mnt/local_disk/data/ \
    --checkpoint-path /home/$USER/checkpoint.pt \
    --time-limit-min 10

# 4. Checkpoint is saved to shared NFS before time expires
```

**SLURM launch strategy:** Using `--ntasks-per-node=1 --gpus-per-node=8` places exactly one SLURM task per node. `srun torchrun` then runs on all 4 nodes simultaneously (one torchrun per node), and each torchrun spawns 8 GPU worker processes. Without `srun`, bare `torchrun` only executes on the head node — the other 3 nodes sit idle. `$SLURM_NODEID` gives the per-node rank (0–3), which is the correct value for `--node-rank`.

## BF16 Fallback Configuration

If Transformer Engine FP8 is not functional on CUDA 13.0 / Blackwell at competition time:

| Parameter | Adjusted Value |
|---|---|
| Precision | BF16 AMP (`torch.bfloat16`) |
| Global batch size | 256 sequences |
| Micro-batch size | 2 |
| Gradient accumulation | 256 / (2 × 8) = 16 steps |
| Peak LR | 6e-4 (unchanged) |
| Model | 3B fallback (7B throughput in BF16 will be too slow for meaningful steps) |

BF16 throughput is ~50% of FP8 on Blackwell. The 3B model at BF16 gives approximately the same token throughput as the 7B at FP8, making 3B+BF16 the rational fallback. The BF16 fallback is activated by removing the Transformer Engine import and using standard `torch.amp.autocast`.

## Cluster Specs (Reference)

| Component | Details |
|---|---|
| GPUs | 8x NVIDIA B300 SXM6 per node, 275 GiB VRAM each |
| Intra-node | NV18 all-to-all NVLink (full bisection bandwidth) |
| Inter-node | 8x 800 Gb/s InfiniBand per node (6.4 Tb/s total, rail-optimized) |
| Local storage | 7 TiB NVMe per worker (`/mnt/local_disk`) |
| CUDA | 13.0 (driver 580.126.09) |
| Workers | 5 available, competition uses 4 |

## Logging & Monitoring

Since training runs via SLURM with no interactive access, all metrics go to both **stdout** (captured by SLURM `--output` log) and a **metrics file** on shared NFS.

### Per-step metrics (stdout + file, every step):
- Training loss
- Validation loss (every 100 steps)
- Learning rate
- Gradient norm
- Step time (ms)
- Tokens/sec (global)
- NaN/divergence events

### Periodic metrics (every 50 steps):
- MFU (model FLOPs utilization)
- GPU memory used / allocated
- Forward / backward / optimizer / communication time breakdown
- Cumulative tokens processed

### Logging implementation:
- **Console:** `print()` or `logging.info()` on rank 0 — SLURM captures via `--output=logs/%j.out` and `--error=logs/%j.err`
- **Metrics file:** CSV or JSON lines to `/home/$USER/metrics/{job_id}.jsonl` on shared NFS — queryable after the run
- **TensorBoard (optional):** Write to `/home/$USER/tb_logs/` if tensorboard is available in the venv
- Log GPU temperature at start and every 2 minutes (B300 at 1100W TDP can thermal throttle)

### Example log line:
```
step=100 loss=5.432 val_loss=5.891 lr=6.00e-04 grad_norm=1.23 tokens/s=1.6M step_ms=329 mem_used=45.2G mfu=0.40
```

## Risks & Fallbacks

1. **Megatron-Core / Transformer Engine not compatible with CUDA 13.0 / Blackwell**
   - Fallback: BF16 without Transformer Engine (see BF16 Fallback Configuration section)
   - Fallback 2: Switch to TorchTitan (native PyTorch TP + manual FP8)

2. **Megatron-Core data loader incompatibility**
   - Mitigation: Custom `BinDataset` plugged into custom training loop (not Megatron's pretrain())

3. **7B throughput insufficient (step_time > 450ms)**
   - Mitigation: Reduce to 3B (see 3B Fallback Configuration and Go/No-Go criteria above)

4. **NFS venv performance**
   - Mitigation: If Python import times are slow over NFS, add a setup step to copy venv to local NVMe

5. **torch.compile incompatible with Megatron-Core TP ops**
   - Mitigation: Disable compile on Megatron TP modules, compile only non-parallel layers
