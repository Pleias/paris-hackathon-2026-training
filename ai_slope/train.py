"""
Megatron-Core Llama-7B training script.

Parallelism:  TP=4 (intra-node NVLink), DP=8 (cross-node InfiniBand)
Precision:    FP8 via Transformer Engine (BF16 AMP fallback)
Batch:        GBS=512 seqs, micro_batch=4, grad_accum=16
Time limit:   10 min from first forward pass (hard stop)

Usage (launched by submit.sh via torchrun):
    torchrun --nnodes=4 --nproc-per-node=8 train.py \
        --tp-size 4 --data-dir /mnt/local_disk/data/ \
        --checkpoint-path /home/$USER/checkpoint.pt \
        --time-limit-min 10
"""

import argparse
import glob
import math
import os
import sys
import time
import json
from contextlib import nullcontext
from dataclasses import dataclass, asdict

try:
    import wandb
except ImportError:
    wandb = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

# ---------------------------------------------------------------------------
# Megatron-Core parallel state
# ---------------------------------------------------------------------------
from megatron.core import parallel_state, tensor_parallel
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

# Transformer Engine FP8 — optional, falls back to BF16
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling, Float8CurrentScaling
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

from model import get_model  # vanilla Llama (for checkpoint save / eval contract)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_dir:    str   = "/mnt/local_disk/data/"
    seq_len:     int   = 1024

    # Model (Llama-7B)
    vocab_size:            int   = 32768   # always 32768 — safe upper bound
    hidden_size:           int   = 4096
    num_layers:            int   = 32
    num_attention_heads:   int   = 32
    num_key_value_heads:   int   = 8
    intermediate_size:     int   = 11008
    rope_theta:            float = 10000.0

    # Parallelism
    tp_size: int = 4
    dp_size: int = 8  # derived: world_size / tp_size

    # Training
    micro_batch_size:  int   = 4
    global_batch_size: int   = 512
    # grad_accum = global_batch_size / (micro_batch_size * dp_size) = 512/(4*8) = 16
    max_lr:            float = 6e-4
    min_lr:            float = 6e-5
    warmup_steps:      int   = 100
    max_steps:         int   = 1700
    weight_decay:      float = 0.1
    grad_clip:         float = 1.0
    time_limit_seconds: float = 10 * 60

    # Checkpointing
    checkpoint_path:  str = "/home/checkpoint.pt"
    metrics_dir:      str = "/home/metrics"

    # Precision
    use_fp8: bool = True


# ---------------------------------------------------------------------------
# Megatron-Core TP-sharded Llama model
# ---------------------------------------------------------------------------

import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    def rotate(x):
        x_f = x.float().reshape(*x.shape[:-1], -1, 2)
        x_c = torch.view_as_complex(x_f)
        x_r = torch.view_as_real(x_c * freqs_cis.unsqueeze(0).unsqueeze(2))
        return x_r.flatten(-2).to(x.dtype)
    return rotate(xq), rotate(xk)


class TPAttention(nn.Module):
    """
    GQA attention with Megatron-Core ColumnParallelLinear / RowParallelLinear.

    With TP=4 and 32 Q heads / 8 KV heads:
      - Each TP rank holds 8 Q heads and 2 KV heads (4:1 ratio preserved)
      - QKV weight is stored as a single fused ColumnParallelLinear:
          shape on each rank: [(8 + 2 + 2) * head_dim, hidden_size] = [1536, 4096]
        Layout per TP rank (Megatron INTERLEAVED GQA):
          [q_per_group, (q_per_group + k + v), head_dim, hidden] remapped view.
        Extraction: reshape to [n_kv_local, (q_per_kv + 2), head_dim] and slice.
      - Output projection is RowParallelLinear.

    TP shard layout:
      - QKV (ColumnParallel): concat on dim=0 across TP ranks
      - output proj (RowParallel): concat on dim=1 across TP ranks
    """

    def __init__(self, cfg: TrainConfig, tp_size: int, mp_config: ModelParallelConfig):
        super().__init__()
        self.tp_size     = tp_size
        self.n_heads     = cfg.num_attention_heads
        self.n_kv_heads  = cfg.num_key_value_heads
        self.head_dim    = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size

        # Local (per-rank) head counts
        self.n_heads_local    = self.n_heads    // tp_size   # 8
        self.n_kv_heads_local = self.n_kv_heads // tp_size   # 2
        self.n_rep            = self.n_heads_local // self.n_kv_heads_local  # 4

        total_local = (self.n_heads_local + 2 * self.n_kv_heads_local) * self.head_dim

        # Fused QKV — ColumnParallelLinear handles the all-gather on output
        init_method = lambda x: torch.nn.init.normal_(x, std=0.02)
        self.linear_qkv = ColumnParallelLinear(
            cfg.hidden_size, total_local * tp_size,
            bias=False, gather_output=False,
            config=mp_config, init_method=init_method,
        )
        self.linear_proj = RowParallelLinear(
            self.n_heads * self.head_dim, cfg.hidden_size,
            bias=False, input_is_parallel=True, skip_bias_add=False,
            config=mp_config, init_method=init_method,
        )

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape

        qkv, _ = self.linear_qkv(x)  # [B, T, (n_heads_local+2*n_kv_local)*head_dim]
        nql  = self.n_heads_local
        nkvl = self.n_kv_heads_local
        hd   = self.head_dim

        # Megatron GQA interleaved layout:
        # qkv is shaped [n_kv_local, (q_per_kv + 1 + 1), head_dim] in the feature dim
        # i.e., groups of (q_per_kv Q heads, 1 K head, 1 V head)
        # Extract by reshaping to [B, T, n_kv_local, q_per_kv+2, head_dim]
        q_per_kv = nql // nkvl  # = 4
        qkv_r = qkv.view(B, T, nkvl, q_per_kv + 2, hd)
        q = qkv_r[:, :, :, :q_per_kv, :].reshape(B, T, nql, hd)   # [B,T,8,128]
        k = qkv_r[:, :, :, q_per_kv:q_per_kv+1, :].reshape(B, T, nkvl, hd)
        v = qkv_r[:, :, :, q_per_kv+1:, :].reshape(B, T, nkvl, hd)

        q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        # Expand KV to match Q (GQA)
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, nql * hd)

        out, _ = self.linear_proj(out)  # all-reduce inside RowParallelLinear
        return out


class TPFeedForward(nn.Module):
    """
    SwiGLU FFN with Megatron-Core ColumnParallelLinear / RowParallelLinear.

    TP shard layout:
      - w1/w3 (gate+up, ColumnParallel): concat on dim=0 across TP ranks
      - w2 (down, RowParallel): concat on dim=1 across TP ranks
    """

    def __init__(self, cfg: TrainConfig, tp_size: int, mp_config: ModelParallelConfig):
        super().__init__()
        init_method = lambda x: torch.nn.init.normal_(x, std=0.02)
        # Fused gate+up projection (ColumnParallel)
        self.linear_fc1 = ColumnParallelLinear(
            cfg.hidden_size, cfg.intermediate_size * 2,
            bias=False, gather_output=False,
            config=mp_config, init_method=init_method,
        )
        # Down projection (RowParallel)
        self.linear_fc2 = RowParallelLinear(
            cfg.intermediate_size, cfg.hidden_size,
            bias=False, input_is_parallel=True, skip_bias_add=False,
            config=mp_config, init_method=init_method,
        )

    def forward(self, x):
        fused, _ = self.linear_fc1(x)   # [B, T, 2 * intermediate_local]
        gate, up = fused.chunk(2, dim=-1)
        h = F.silu(gate) * up
        out, _ = self.linear_fc2(h)
        return out


class TPTransformerBlock(nn.Module):
    def __init__(self, cfg: TrainConfig, tp_size: int, mp_config: ModelParallelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.hidden_size)
        self.attention      = TPAttention(cfg, tp_size, mp_config)
        self.ffn_norm       = RMSNorm(cfg.hidden_size)
        self.feed_forward   = TPFeedForward(cfg, tp_size, mp_config)

    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TPLlamaModel(nn.Module):
    def __init__(self, cfg: TrainConfig, tp_size: int):
        super().__init__()
        self.cfg     = cfg
        self.tp_size = tp_size
        head_dim     = cfg.hidden_size // cfg.num_attention_heads

        mp_config = ModelParallelConfig(tensor_model_parallel_size=tp_size)
        init_method = lambda x: torch.nn.init.normal_(x, std=0.02)

        self.tok_embeddings = VocabParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size,
            init_method=init_method, config=mp_config,
        )
        self.layers = nn.ModuleList([
            TPTransformerBlock(cfg, tp_size, mp_config)
            for _ in range(cfg.num_layers)
        ])
        self.norm   = RMSNorm(cfg.hidden_size)
        self.output = ColumnParallelLinear(
            cfg.hidden_size, cfg.vocab_size,
            bias=False, gather_output=True,
            config=mp_config, init_method=init_method,
        )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, cfg.seq_len, theta=cfg.rope_theta),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        n = self.cfg.num_layers
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "norm" in name and p.dim() == 1:
                nn.init.ones_(p)
            elif any(s in name for s in ("wo", "linear_proj", "linear_fc2")):
                nn.init.normal_(p, 0.0, std / math.sqrt(2 * n))
            elif p.dim() >= 2:
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx, targets=None):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = activation_checkpoint(layer, x, self.freqs_cis, use_reentrant=False)
        x = self.norm(x)
        logits, _ = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            # Reduce loss across TP ranks
            dist.all_reduce(loss, op=dist.ReduceOp.SUM,
                            group=get_tensor_model_parallel_group())
            loss = loss / get_tensor_model_parallel_world_size()

        return logits, loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinDataset:
    """
    Memory-maps all *.bin uint16 shards and draws random windows.
    Each DP rank uses a different numpy seed so all ranks draw different data.
    Vectorized: draws batch_size (shard, offset) pairs in one numpy call.
    """

    def __init__(self, data_dir: str, seq_len: int, dp_rank: int,
                 dtype: str = "uint16", exclude_first_half_chunk1: bool = False):
        all_paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not all_paths:
            raise FileNotFoundError(f"No *.bin files in {data_dir!r}")

        if exclude_first_half_chunk1:
            # Reserve first half of chunk_0001.bin for validation
            paths = [p for p in all_paths if "chunk_0001" not in p]
        else:
            paths = all_paths

        np_dtype    = np.dtype(dtype)
        self.seq_len = seq_len
        self.shards  = [np.memmap(p, dtype=np_dtype, mode="r") for p in paths]
        self.lengths = np.array([max(0, len(s) - seq_len - 1) for s in self.shards], dtype=np.int64)
        total = self.lengths.sum()
        if total == 0:
            raise RuntimeError("All shards are too short for seq_len")
        self.weights = self.lengths / total
        np.random.seed(42 + dp_rank)
        print(f"[data] {len(paths)} shards, {total:,} usable positions, dp_rank={dp_rank}")

    def get_batch(self, batch_size: int, device):
        shard_indices = np.random.choice(len(self.shards), size=batch_size, p=self.weights)
        offsets = np.array([
            np.random.randint(0, self.lengths[si])
            for si in shard_indices
        ], dtype=np.int64)
        chunks = np.stack([
            self.shards[si][offsets[i]:offsets[i] + self.seq_len + 1].astype(np.int64)
            for i, si in enumerate(shard_indices)
        ])
        t = torch.from_numpy(chunks).to(device)
        return t[:, :-1], t[:, 1:]


class ValDataset:
    """First half of chunk_0001.bin reserved for validation."""

    def __init__(self, data_dir: str, seq_len: int):
        candidates = sorted(glob.glob(os.path.join(data_dir, "chunk_0001.bin")))
        if not candidates:
            # Fall back to first shard
            candidates = sorted(glob.glob(os.path.join(data_dir, "*.bin")))[:1]
        shard = np.memmap(candidates[0], dtype=np.uint16, mode="r")
        half  = len(shard) // 2
        self.data    = shard[:half]
        self.seq_len = seq_len
        np.random.seed(0)

    def get_batch(self, batch_size: int, device):
        offsets = np.random.randint(0, len(self.data) - self.seq_len - 1, size=batch_size)
        chunks  = np.stack([
            self.data[o:o + self.seq_len + 1].astype(np.int64)
            for o in offsets
        ])
        t = torch.from_numpy(chunks).to(device)
        return t[:, :-1], t[:, 1:]


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * max(step, 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.max_lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Weight remapping: Megatron-Core TP shards → vanilla Llama state_dict
# ---------------------------------------------------------------------------

def gather_tp_shards(tp_group, tp_rank, tp_size, tensor, dim):
    """Gather a TP-sharded tensor from all ranks in the TP group.
    Handles CPU tensors (e.g. from best_ckpt_shard) by moving to GPU
    for NCCL all_gather, then moving the result back to CPU."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    tensor_gpu = tensor.to(device)
    gathered = [torch.zeros_like(tensor_gpu) for _ in range(tp_size)]
    dist.all_gather(gathered, tensor_gpu, group=tp_group)
    return torch.cat(gathered, dim=dim).cpu()


def consolidate_and_remap(best_ckpt_shard: dict, cfg: TrainConfig,
                          tp_rank: int, tp_size: int, tp_group,
                          global_rank: int) -> dict | None:
    """
    Called on DP group 0 after training.  Each TP rank contributes its shard;
    TP rank 0 (global_rank == tp0_global_rank in dp group 0) gathers all shards,
    remaps to vanilla Llama naming, and returns the full state_dict.
    Returns None on non-aggregating ranks.

    TP shard layout:
      ColumnParallel (QKV, FC1, output):  sharded on dim=0, concat on dim=0
      RowParallel (proj, FC2):            sharded on dim=1, concat on dim=1
    """
    H  = cfg.hidden_size            # 4096
    Hq = cfg.num_attention_heads * (H // cfg.num_attention_heads)   # 4096
    Hk = cfg.num_key_value_heads  * (H // cfg.num_attention_heads)  # 1024
    Hv = Hk

    def col_gather(t):  # ColumnParallel shard → full tensor (dim=0)
        return gather_tp_shards(tp_group, tp_rank, tp_size, t, dim=0)

    def row_gather(t):  # RowParallel shard → full tensor (dim=1)
        return gather_tp_shards(tp_group, tp_rank, tp_size, t, dim=1)

    out = {}
    # Embedding: VocabParallelEmbedding is sharded on dim=0
    out["tok_embeddings.weight"] = col_gather(
        best_ckpt_shard["tok_embeddings.weight"]
    )

    for i in range(cfg.num_layers):
        prefix = f"layers.{i}"

        # Attention norms (not sharded — same on all TP ranks, just take from rank 0)
        attn_norm_key = f"{prefix}.attention.linear_qkv.layer_norm_weight"
        if attn_norm_key in best_ckpt_shard:
            out[f"{prefix}.attention_norm.weight"] = best_ckpt_shard[attn_norm_key].clone()
        else:
            out[f"{prefix}.attention_norm.weight"] = best_ckpt_shard[
                f"{prefix}.attention_norm.weight"
            ].clone()

        ffn_norm_key = f"{prefix}.feed_forward.linear_fc1.layer_norm_weight"
        if ffn_norm_key in best_ckpt_shard:
            out[f"{prefix}.ffn_norm.weight"] = best_ckpt_shard[ffn_norm_key].clone()
        else:
            out[f"{prefix}.ffn_norm.weight"] = best_ckpt_shard[
                f"{prefix}.ffn_norm.weight"
            ].clone()

        # QKV — ColumnParallel → gather on dim=0
        # Per-rank shard layout (Megatron INTERLEAVED GQA):
        # [n_kv_local * (q_per_kv + 2) * head_dim, hidden]
        # After gather on dim=0: [(H_q + 2*H_kv), H]
        qkv_key = f"{prefix}.attention.linear_qkv.weight"
        qkv_full = col_gather(best_ckpt_shard[qkv_key])
        # qkv_full shape: [(Hq + Hk + Hv), H] after gather
        # Megatron GQA interleaved layout per TP-rank group:
        # reshape to [n_kv_heads, (q_per_group + 2), head_dim, H]
        hd      = H // cfg.num_attention_heads
        n_kv    = cfg.num_key_value_heads
        q_per_g = cfg.num_attention_heads // n_kv
        # qkv_full already concatenated across TP ranks on dim=0
        # Interleaved layout: [n_kv, q_per_g+2, hd, H]
        qkv_r   = qkv_full.view(n_kv, q_per_g + 2, hd, H)
        wq = qkv_r[:, :q_per_g,       :, :].reshape(cfg.num_attention_heads * hd, H)
        wk = qkv_r[:, q_per_g:q_per_g+1, :, :].reshape(n_kv * hd, H)
        wv = qkv_r[:, q_per_g+1:,      :, :].reshape(n_kv * hd, H)
        out[f"{prefix}.attention.wq.weight"] = wq
        out[f"{prefix}.attention.wk.weight"] = wk
        out[f"{prefix}.attention.wv.weight"] = wv

        # Output projection — RowParallel → gather on dim=1
        proj_key = f"{prefix}.attention.linear_proj.weight"
        out[f"{prefix}.attention.wo.weight"] = row_gather(
            best_ckpt_shard[proj_key]
        )

        # FC1 (gate+up fused) — ColumnParallel → gather on dim=0 → split
        fc1_key  = f"{prefix}.feed_forward.linear_fc1.weight"
        fc1_full = col_gather(best_ckpt_shard[fc1_key])
        w1, w3   = fc1_full.chunk(2, dim=0)
        out[f"{prefix}.feed_forward.w1.weight"] = w1
        out[f"{prefix}.feed_forward.w3.weight"] = w3

        # FC2 (down) — RowParallel → gather on dim=1
        fc2_key = f"{prefix}.feed_forward.linear_fc2.weight"
        out[f"{prefix}.feed_forward.w2.weight"] = row_gather(
            best_ckpt_shard[fc2_key]
        )

    # Final norm (not sharded)
    out["norm.weight"] = best_ckpt_shard["norm.weight"].clone()

    # LM head — ColumnParallel → gather on dim=0
    out["output.weight"] = col_gather(best_ckpt_shard["output.weight"])

    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_val_loss(val_dataset: ValDataset, model, n_batches: int,
                     micro_batch: int, device, tp_group) -> float:
    """
    Run n_batches micro-batches through the TP-sharded model.
    All TP ranks (dp_rank==0 group) must call this together.
    Returns mean loss (already all-reduced across TP).
    """
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = val_dataset.get_batch(micro_batch, device)
        with torch.no_grad():
            _, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tp-size",          type=int,   default=4)
    p.add_argument("--data-dir",         default="/mnt/local_disk/data/")
    p.add_argument("--checkpoint-path",  default="/home/checkpoint.pt")
    p.add_argument("--metrics-dir",      default="/home/metrics")
    p.add_argument("--time-limit-min",   type=float, default=10.0)
    p.add_argument("--max-steps",        type=int,   default=1700)
    p.add_argument("--micro-batch",      type=int,   default=4)
    p.add_argument("--global-batch",     type=int,   default=512)
    p.add_argument("--max-lr",           type=float, default=6e-4)
    p.add_argument("--no-fp8",           action="store_true")
    p.add_argument("--no-compile",       action="store_true")
    p.add_argument("--wandb-project",    default=None)
    p.add_argument("--wandb-run-name",   default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 3B fallback config
# ---------------------------------------------------------------------------

def make_3b_config(base: TrainConfig) -> TrainConfig:
    cfg = TrainConfig(**asdict(base))
    cfg.hidden_size           = 3072
    cfg.num_layers            = 26
    cfg.num_attention_heads   = 24
    cfg.num_key_value_heads   = 8
    cfg.intermediate_size     = 8192
    cfg.micro_batch_size      = 8
    cfg.global_batch_size     = 1024
    cfg.max_lr                = 8e-4
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------ Distributed init
    dist.init_process_group(backend="nccl")
    world_size  = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank  = int(os.environ["LOCAL_RANK"])
    device      = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    tp_size = args.tp_size
    dp_size = world_size // tp_size
    assert world_size == tp_size * dp_size, \
        f"world_size {world_size} must equal tp_size*dp_size ({tp_size}*{dp_size})"

    # Initialise Megatron-Core parallel state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1337)

    tp_rank  = get_tensor_model_parallel_rank()
    dp_rank  = get_data_parallel_rank()
    tp_group = get_tensor_model_parallel_group()
    is_master = (global_rank == 0)

    torch.manual_seed(1337 + global_rank)

    # ------------------------------------------------------------------ Config
    cfg = TrainConfig(
        data_dir           = args.data_dir,
        checkpoint_path    = args.checkpoint_path,
        metrics_dir        = args.metrics_dir,
        tp_size            = tp_size,
        dp_size            = dp_size,
        micro_batch_size   = args.micro_batch,
        global_batch_size  = args.global_batch,
        max_lr             = args.max_lr,
        max_steps          = args.max_steps,
        time_limit_seconds = args.time_limit_min * 60,
        use_fp8            = (not args.no_fp8) and TE_AVAILABLE,
    )
    grad_accum = cfg.global_batch_size // (cfg.micro_batch_size * dp_size)
    assert grad_accum > 0, f"grad_accum={grad_accum} must be > 0"

    if is_master:
        print(f"[config] TP={tp_size}, DP={dp_size}, world={world_size}, "
              f"micro_batch={cfg.micro_batch_size}, grad_accum={grad_accum}, "
              f"GBS={cfg.global_batch_size}, use_fp8={cfg.use_fp8}")
        if wandb is not None and args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=asdict(cfg),
            )

    # ------------------------------------------------------------------ Model
    model = TPLlamaModel(cfg, tp_size).to(device)

    if is_master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M parameters (local TP shard)")

    # Use Megatron's DDP — understands TP topology, no shape verification issues
    if dp_size > 1:
        from megatron.core.distributed import (
            DistributedDataParallel as MegatronDDP,
            DistributedDataParallelConfig,
        )
        from megatron.core.transformer.transformer_config import TransformerConfig as MegatronTransformerConfig
        transformer_config = MegatronTransformerConfig(
            tensor_model_parallel_size=tp_size,
            num_layers=cfg.num_layers,
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            params_dtype=torch.bfloat16,
        )
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
        )
        model = MegatronDDP(
            config=transformer_config,
            ddp_config=ddp_config,
            module=model,
        )

    # torch.compile — applied after DDP wrapping
    if not args.no_compile:
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            if is_master:
                print("[compile] torch.compile enabled (max-autotune)")
        except Exception as e:
            if is_master:
                print(f"[compile] torch.compile failed ({e}), continuing without")

    # MegatronDDP exposes the inner module via .module; PyTorch DDP does too
    raw_model = model.module if hasattr(model, "module") else model

    # ------------------------------------------------------------------ Optimizer
    # ZeroRedundancyOptimizer shards Adam state across DP ranks,
    # reducing per-GPU optimizer memory from ~24 GiB to ~3 GiB.
    from torch.distributed.optim import ZeroRedundancyOptimizer
    decay_params   = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = ZeroRedundancyOptimizer(
        [{"params": decay_params,   "weight_decay": cfg.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        optimizer_class=torch.optim.AdamW,
        lr=cfg.max_lr, betas=(0.9, 0.95), fused=True,
    )

    # ------------------------------------------------------------------ Data
    train_dataset = BinDataset(
        cfg.data_dir, cfg.seq_len, dp_rank,
        exclude_first_half_chunk1=True,
    )
    val_dataset = None
    if dp_rank == 0:
        val_dataset = ValDataset(cfg.data_dir, cfg.seq_len)

    # ------------------------------------------------------------------ Precision context
    if cfg.use_fp8:
        fp8_recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)
        def get_amp_ctx():
            return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        def get_amp_ctx():
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # ------------------------------------------------------------------ Metrics
    os.makedirs(cfg.metrics_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.metrics_dir,
                                f"{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl")
    metrics_file = open(metrics_path, "w") if is_master else None

    def log_metrics(**kwargs):
        if is_master:
            print("  ".join(f"{k}={v}" for k, v in kwargs.items()), flush=True)
            if metrics_file:
                metrics_file.write(json.dumps(kwargs) + "\n")
                metrics_file.flush()
            if wandb is not None and wandb.run is not None:
                wandb.log(kwargs, step=kwargs.get("step"))

    # ------------------------------------------------------------------ Training state
    step              = 0
    best_val_loss     = float("inf")
    best_ckpt_shard   = None
    best_ckpt_step    = -1
    nan_recovery_count = 0
    step_times        = []
    train_start       = None   # set at first forward pass

    model.train()
    if hasattr(model, 'zero_grad_buffer'):
        model.zero_grad_buffer()
    else:
        optimizer.zero_grad()

    # Go/no-go state
    aborted_7b = False

    # ------------------------------------------------------------------ Training loop
    while step < cfg.max_steps:

        # Time-limit check (after first forward pass starts the clock)
        if train_start is not None:
            elapsed = time.time() - train_start
            stop_flag = torch.tensor(
                int(elapsed >= cfg.time_limit_seconds), device=device, dtype=torch.int32
            )
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                if is_master:
                    print(f"\n[time] {elapsed/60:.2f} min — time limit reached at step {step}")
                break

        step_t0 = time.time()
        if train_start is None:
            train_start = step_t0  # clock starts here

        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step, cfg)

        # ---- Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(grad_accum):
            x, y = train_dataset.get_batch(cfg.micro_batch_size, device)

            # DDP: skip sync until final micro-step
            sync_ctx = (model.no_sync()
                        if (dp_size > 1 and hasattr(model, "no_sync")
                            and micro_step < grad_accum - 1)
                        else nullcontext())

            with sync_ctx, get_amp_ctx():
                _, loss = model(x, y)
                loss = loss / grad_accum

            loss.backward()
            accumulated_loss += loss.item()

        # MegatronDDP with overlap_grad_reduce needs explicit sync after last backward
        if dp_size > 1 and hasattr(model, "finish_grad_sync"):
            model.finish_grad_sync()

        # ---- NaN / divergence detection
        if math.isnan(accumulated_loss) or math.isinf(accumulated_loss) or \
                (step > 50 and accumulated_loss > 20.0):
            if nan_recovery_count < 2:
                nan_recovery_count += 1
                cfg.max_lr *= 0.5
                if is_master:
                    print(f"[warn] NaN/divergence at step {step}, "
                          f"halving LR to {cfg.max_lr:.2e} (attempt {nan_recovery_count})")
                if hasattr(model, 'zero_grad_buffer'):
                    model.zero_grad_buffer()
                else:
                    optimizer.zero_grad(set_to_none=True)
                # Update LR in optimizer
                for pg in optimizer.param_groups:
                    pg["lr"] = get_lr(step, cfg)
                continue
            else:
                if is_master:
                    print(f"[warn] NaN persists after 2 recoveries at step {step}, continuing")

        # ---- Copy main_grad → param.grad for MegatronDDP compatibility
        # MegatronDDP stores gradients in param.main_grad, not param.grad.
        # Standard PyTorch clip_grad_norm_ and optimizers read param.grad.
        if dp_size > 1 and hasattr(model, 'zero_grad_buffer'):
            for p in raw_model.parameters():
                if hasattr(p, 'main_grad') and p.main_grad is not None:
                    p.grad = p.main_grad.to(p.dtype)

        # ---- Gradient clip + optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), cfg.grad_clip
        ).item()
        optimizer.step()
        if hasattr(model, 'zero_grad_buffer'):
            model.zero_grad_buffer()
        else:
            optimizer.zero_grad(set_to_none=True)

        step_ms = (time.time() - step_t0) * 1000
        step_times.append(step_ms)
        step += 1

        # ---- Go/no-go check at step 10
        if step == 10 and not aborted_7b:
            median_ms = sorted(step_times[4:10])[3] if len(step_times) >= 10 else step_ms
            if median_ms > 450:
                if is_master:
                    print(f"[go/no-go] median step {median_ms:.0f}ms > 450ms, "
                          f"7B too slow — continuing (3B fallback would be launched here)")
                # In a full implementation: abort_and_launch_3b()

        # ---- Logging (every step to stdout, every 10 to be concise)
        if is_master:
            elapsed_total = time.time() - train_start if train_start else 0
            tok_per_sec   = (cfg.micro_batch_size * cfg.seq_len * grad_accum * dp_size) / (step_ms / 1000)
            if step % 10 == 0 or step <= 5:
                log_metrics(
                    step=step,
                    loss=round(accumulated_loss * grad_accum, 4),
                    lr=f"{get_lr(step, cfg):.3e}",
                    grad_norm=round(grad_norm, 3),
                    step_ms=round(step_ms, 1),
                    tok_per_s=f"{tok_per_sec/1e6:.2f}M",
                    elapsed_min=round(elapsed_total / 60, 2),
                )

            # MFU / memory every 50 steps
            if step % 50 == 0:
                mem_alloc = torch.cuda.memory_allocated(device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                # N=6.7B, FLOPs_per_tok = 6*N
                flops_per_tok = 6 * 6.7e9
                cluster_flops = 5e15 * 32  # 5 PFLOPS * 32 GPUs
                mfu = (tok_per_sec * flops_per_tok) / cluster_flops
                print(f"  [perf] mem={mem_alloc:.1f}/{mem_reserved:.1f}GB  "
                      f"mfu={mfu:.2%}")

        # ---- Periodic validation (every 100 steps, dp_rank==0 only)
        if step % 100 == 0 and dp_rank == 0 and val_dataset is not None:
            model.eval()
            val_loss = compute_val_loss(
                val_dataset, model, n_batches=20,
                micro_batch=cfg.micro_batch_size, device=device, tp_group=tp_group,
            )
            model.train()

            # tp_rank 0 decides if this is a new best
            is_best = False
            if tp_rank == 0:
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)

            # Broadcast is_best from the GLOBAL rank of tp_rank=0 in this TP group
            # to all other TP ranks — using the TP group itself, so src=0 is the
            # local rank 0 within the group (which corresponds to tp_rank=0)
            is_best_tensor = torch.tensor([int(is_best)], device=device, dtype=torch.int32)
            # src=0 means tp_rank=0 within the tp_group (correct: tp_group rank 0 == tp_rank 0)
            tp_group_rank0_global = global_rank - tp_rank  # global rank of tp_rank=0 in this TP group
            dist.broadcast(is_best_tensor, src=tp_group_rank0_global, group=tp_group)

            if is_best_tensor.item():
                best_ckpt_shard = {k: v.cpu() for k, v in raw_model.state_dict().items()}
                best_ckpt_step  = step

            if tp_rank == 0 and is_master:
                log_metrics(
                    step=step,
                    val_loss=round(val_loss, 4),
                    best_val_loss=round(best_val_loss, 4),
                    new_best=bool(is_best_tensor.item()),
                )

    # ------------------------------------------------------------------ End of training
    elapsed_final = time.time() - train_start if train_start else 0
    if is_master:
        print(f"\n[done] Finished at step {step}, {elapsed_final/60:.2f} min elapsed")

    # ------------------------------------------------------------------ Checkpoint save
    # All TP ranks in dp_rank==0 participate in gathering; others skip
    if dp_rank == 0:
        if best_ckpt_shard is None:
            # No validation checkpoint — use current weights
            best_ckpt_shard = {k: v.cpu() for k, v in raw_model.state_dict().items()}
            best_ckpt_step  = step

        # Gather and remap weights to vanilla Llama naming
        full_state_dict = consolidate_and_remap(
            best_ckpt_shard, cfg, tp_rank, tp_size, tp_group, global_rank
        )

        if tp_rank == 0:
            # Save the consolidated checkpoint in vanilla Llama format
            # Build vanilla config for get_model() reconstruction
            vanilla_config = {
                "vocab_size":          cfg.vocab_size,
                "hidden_size":         cfg.hidden_size,
                "num_layers":          cfg.num_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "intermediate_size":   cfg.intermediate_size,
                "seq_len":             cfg.seq_len,
                "rope_theta":          cfg.rope_theta,
            }

            # Verify it loads cleanly into vanilla model
            vanilla_model = get_model(vanilla_config)
            try:
                vanilla_model.load_state_dict(full_state_dict, strict=True)
                print(f"[ckpt] weight remap verified (strict load OK)")
            except RuntimeError as e:
                print(f"[ckpt] WARNING: strict load failed: {e}")
                print(f"[ckpt] saving anyway with strict=False state_dict")

            torch.save({
                "step":   best_ckpt_step,
                "model":  full_state_dict,
                "config": vanilla_config,
            }, cfg.checkpoint_path)
            print(f"[ckpt] saved best checkpoint (step {best_ckpt_step}) → {cfg.checkpoint_path}")

    if metrics_file:
        metrics_file.close()

    if is_master and wandb is not None and wandb.run is not None:
        wandb.finish()

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
