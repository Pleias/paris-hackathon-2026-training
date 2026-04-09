# Cluster Review: verda-hackathon

## Cluster Topology

| Component | Details |
|---|---|
| **Jumphost** | `verda-hackathon-jumphost` (10.13.114.100) — 32 vCPUs, 125 GiB RAM, no GPUs |
| **Worker nodes** | 5 nodes (`verda-hackathon-1` through `verda-hackathon-5`) |
| **CPUs per worker** | 240 vCPUs (2 sockets x 120 cores x 1 thread) |
| **RAM per worker** | ~200 GiB (204000 MB reported by SLURM) |
| **Scheduler** | SLURM, single partition `gpus` (default) |
| **Shared filesystem** | `/home` (NFS-mounted across all nodes) |
| **Local storage** | `/mnt/local_disk` — 290 GiB total, ~235 GiB free (on jumphost; likely NVMe on workers) |

## GPU Info

| Property | Value |
|---|---|
| **GPU model** | NVIDIA B300 (Blackwell architecture) |
| **GPUs per node** | 8 |
| **Nodes used** | 4 (verda-hackathon-1 through verda-hackathon-4) |
| **Total GPUs** | 32 |
| **VRAM per GPU** | ~192 GiB HBM3e (estimated for B300) |
| **Total VRAM** | ~6 TiB |
| **Key features** | FP4/FP8 tensor cores, NVLink 5.0 (intra-node), 2nd-gen Transformer Engine |

The B300 is NVIDIA's latest Blackwell GPU — significantly faster than H100 for transformer training, especially with FP8/BF16 mixed precision. With 32 of them and 10 minutes, we have enormous compute budget.

## Network / Interconnect

Intra-node: **NVLink 5.0** (1.8 TB/s bisection bandwidth between 8 GPUs per node via NVSwitch).
Inter-node: Likely high-speed networking (InfiniBand or RoCE) given this is a purpose-built GPU cluster. Still worth confirming from a worker node with `ibstat`.

## Training Data

| Property | Value |
|---|---|
| **Location** | `/home/data/chunk_*.bin` (shared filesystem) |
| **Shards** | 49 files (`chunk_0001.bin` to `chunk_0049.bin`) |
| **Total size** | 92 GiB |
| **Format** | Pre-tokenized `uint16` binary |
| **Vocab size** | 32,000 (per README) |

At 2 bytes per token (uint16), this is approximately **~49 billion tokens** of training data — far more than can be consumed in 10 minutes, so data is not a bottleneck.

## Software Environment

- **No Python or PyTorch pre-installed** on the jumphost (or at least not in default PATH).
- A helper script at `/home/pytorch.setup.sh` sets up a uv-based venv with PyTorch + CUDA. Must be run on a worker node (needs `nvcc`).
- SLURM is configured and operational.

## Node Status (at time of check)

| Node | State |
|---|---|
| verda-hackathon-1 | **allocated** (someone is using it) |
| verda-hackathon-2 | **mixed** (partially allocated) |
| verda-hackathon-3 | idle |
| verda-hackathon-4 | idle |
| verda-hackathon-5 | idle |

## Remaining Gaps

1. **Inter-node interconnect** — Confirm InfiniBand/RoCE from a worker node (`ibstat`). Affects NCCL backend choice.
2. **PyTorch environment** — Need to run the setup script on a worker or confirm an existing venv with Blackwell support (PyTorch 2.6+ needed for B300).
3. **SLURM GPU GRES config** — Run `scontrol show node verda-hackathon-3` to verify SLURM sees the GPUs.

## Strategic Implications

### Compute Budget
With 32x B300 GPUs for 10 minutes, we have a massive compute envelope. The baseline GPT-2 Small (124M params) is **far too small** for this hardware — it would underutilize the GPUs dramatically. We should scale up significantly.

### Model Sizing
- **Minimum viable:** ~1B params (would still underutilize 32 B300s)
- **Sweet spot:** ~3-7B params — large enough to use the compute well, small enough to converge meaningfully in 10 min
- **Aggressive:** 10B+ params — risky, may not converge enough in time

### Key Optimizations to Consider
1. **FP8 training** — B300 Transformer Engine supports FP8, which nearly doubles throughput vs BF16. This is the single biggest lever.
2. **FSDP or tensor parallelism** — With 32 GPUs, plain DDP may OOM on larger models. FSDP (ZeRO-3) or TP+DP hybrid is likely needed.
3. **Data staging to local NVMe** — Copy shards to `/mnt/local_disk` before training starts (free time before the clock).
4. **Large batch sizes** — B300's massive VRAM allows very large micro-batches, reducing communication overhead.
5. **Aggressive learning rate + warmup** — With only 10 min, we need fast convergence. Large batch + high LR + short warmup.
6. **Sequence length** — Consider increasing beyond 1024 if the model can handle it; more tokens per step = faster learning.
7. **torch.compile** — Can significantly speed up on Blackwell with the latest PyTorch.

### Data Throughput
- 92 GiB = ~49B tokens. At 32 GPUs with large batches, we might process 1-5B tokens in 10 min depending on model size.
- Data is not a bottleneck, but NFS read speed could be. Stage to local disk.

### Architecture Changes to Consider
- Replace learned positional embeddings with **RoPE** (better for scaling)
- Use **SwiGLU** activation instead of GELU (better quality per FLOP)
- Use **GQA** (grouped-query attention) to reduce memory and increase throughput
- Consider **flash attention 3** if available for Blackwell
