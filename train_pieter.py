"""
Starter training script for the gpu-mode Paris hackathon training track
"""

import os
import time
import glob
import math
import json
import argparse
import importlib
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    import wandb
except ImportError:
    wandb = None

try:
    from muon import MuonWithAuxAdam
except ImportError:
    MuonWithAuxAdam = None

try:
    import torchao  # noqa: F401 — presence check; float8 API used lazily in get_model
    _torchao_available = True
except ImportError:
    _torchao_available = False

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from model_plus_plus import get_model
# from model_pieter import get_model


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    data_dir:    str   = "data"
    token_dtype: str   = "uint16"
    seq_len:     int   = 1024

    # Model (passed through to get_model — add arch-specific keys in model.py)
    vocab_size: int   = 32000
    n_layer:    int   = 12
    n_head:     int   = 6
    n_embd:     int   = 384
    dropout:    float = 0.0

    # Training
    batch_size:       int   = 64
    grad_accum_steps: int   = 1
    max_lr:           float = 5e-4
    min_lr:           float = 6e-5
    warmup_steps:     int   = 10
    max_steps:        int   = 1_000
    weight_decay:     float = 0.01
    grad_clip:        float = 1.0
    time_limit_seconds: float = 10 * 60

    # Optimizer
    muon:    bool  = True
    muon_lr: float = 0.02

    # FP8
    fp8: bool = False

    # Checkpointing
    checkpoint_path: str = "checkpoint.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinDataset:
    """Memory-maps all *.bin files and draws random (seq_len+1)-token windows."""

    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16"):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(f"No *.bin files found in '{data_dir}'")
        self.seq_len  = seq_len
        np_dtype      = np.dtype(dtype)
        self.shards   = [np.memmap(p, dtype=np_dtype, mode="r") for p in paths]
        self.lengths  = [len(s) for s in self.shards]
        self.total    = sum(self.lengths)
        self.weights  = [l / self.total for l in self.lengths]
        print(f"[data] {len(paths)} shard(s), {self.total:,} tokens total")

    def get_batch(self, batch_size: int, device):
        xs, ys = [], []
        for _ in range(batch_size):
            shard = self.shards[np.random.choice(len(self.shards), p=self.weights)]
            start = np.random.randint(0, len(shard) - self.seq_len - 1)
            chunk = torch.from_numpy(shard[start:start + self.seq_len + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay → min_lr
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.max_lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, step: int, cfg: Config):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "step":   step,
        "model":  raw_model.state_dict(),
        "config": asdict(cfg),
    }, cfg.checkpoint_path)
    print(f"[ckpt] saved → {cfg.checkpoint_path}  (step {step})")


def get_memory_stats(device: str) -> dict:
    if "cuda" not in device or not torch.cuda.is_available():
        return {
            "mem_allocated_gib": 0.0,
            "mem_reserved_gib": 0.0,
            "mem_peak_allocated_gib": 0.0,
            "mem_peak_reserved_gib": 0.0,
        }

    gib = 1024 ** 3
    return {
        "mem_allocated_gib": torch.cuda.memory_allocated() / gib,
        "mem_reserved_gib": torch.cuda.memory_reserved() / gib,
        "mem_peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "mem_peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


def aggregate_distributed_metrics(ddp: bool, device: str, metrics: dict[str, float]) -> dict[str, float]:
    """Aggregate per-rank scalars into world average and world max metrics."""
    if not ddp:
        out = dict(metrics)
        for key, value in metrics.items():
            out[f"{key}_max"] = value
        return out

    keys = list(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float32)
    sum_values = values.clone()
    max_values = values.clone()

    dist.all_reduce(sum_values, op=dist.ReduceOp.SUM)
    dist.all_reduce(max_values, op=dist.ReduceOp.MAX)

    world_size = dist.get_world_size()
    out = {}
    for i, key in enumerate(keys):
        out[key] = (sum_values[i] / world_size).item()
        out[f"{key}_max"] = max_values[i].item()
    return out


def _next_available_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{base}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def save_metrics_reports(cfg: Config, summary: dict):
    base, _ = os.path.splitext(cfg.checkpoint_path)
    json_path = _next_available_path(f"{base}_summary.json")

    with open(json_path, "x", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, indent=2)

    print(f"[metrics] saved → {json_path}")


def setup_attention_backend(device: str, master: bool, require_fa4: bool):
    """Configure attention backend and return context manager factory for forward passes."""
    if "cuda" not in device:
        if master:
            print("[attn] CUDA not detected; using default non-flash attention path.")
        return nullcontext

    flash_attn_version = None
    try:
        flash_attn = importlib.import_module("flash_attn")
        flash_attn_version = getattr(flash_attn, "__version__", None)
    except Exception:
        flash_attn_version = None

    if flash_attn_version is None:
        try:
            import importlib.metadata as importlib_metadata
            flash_attn_version = importlib_metadata.version("flash-attn-4")
        except Exception:
            flash_attn_version = None

    fa4_available = False
    if flash_attn_version is not None:
        try:
            fa4_available = int(str(flash_attn_version).split(".")[0]) >= 4
        except Exception:
            fa4_available = False

    if require_fa4 and not fa4_available:
        raise RuntimeError(
            "FlashAttention-4 was requested, but package flash_attn>=4 is not available in this environment. "
            "Install it first, or run with --require_fa4 false to use PyTorch flash SDPA kernels."
        )

    # Force PyTorch SDPA flash backend (math and mem-efficient kernels disabled).
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    if master:
        if fa4_available:
            print(f"[attn] flash_attn {flash_attn_version} detected (FA4+). Forcing PyTorch flash SDPA backend.")
        else:
            print("[attn] flash_attn>=4 not detected. Forcing PyTorch flash SDPA backend.")

    return lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          default="data")
    parser.add_argument("--checkpoint_path",   default="checkpoint.pt")
    parser.add_argument("--seq_len",           type=int,   default=1024)
    parser.add_argument("--vocab_size",        type=int,   default=32768)
    parser.add_argument("--n_layer",           type=int,   default=12)
    parser.add_argument("--n_head",            type=int,   default=12)
    parser.add_argument("--n_embd",            type=int,   default=1536) # 1536
    parser.add_argument("--batch_size",        type=int,   default=256)
    parser.add_argument("--grad_accum_steps",  type=int,   default=1)
    parser.add_argument("--max_steps",         type=int,   default=1_000)
    parser.add_argument("--time_limit_min",    type=float, default=10.0)
    parser.add_argument("--require_fa4", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (omit to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--muon", type=lambda x: x.lower() not in ("0", "false", "no"), default=True,
                        help="Use Muon optimizer for hidden weights (default: true)")
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--fp8", type=lambda x: x.lower() in ("1", "true", "yes"), default=False,
                        help="Enable FP8 training via torchao (B300/Blackwell recommended)")
    args = parser.parse_args()

    cfg = Config(
        data_dir           = args.data_dir,
        checkpoint_path    = args.checkpoint_path,
        seq_len            = args.seq_len,
        vocab_size         = args.vocab_size,
        n_layer            = args.n_layer,
        n_head             = args.n_head,
        n_embd             = args.n_embd,
        batch_size         = args.batch_size,
        grad_accum_steps   = args.grad_accum_steps,
        max_steps          = args.max_steps,
        time_limit_seconds = args.time_limit_min * 60,
        muon               = args.muon,
        muon_lr            = args.muon_lr,
        fp8                = args.fp8,
    )

    # ------------------------------------------------------------------ DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device     = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master     = rank == 0
    else:
        rank = 0; master = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337 + rank)
    # Keep bf16 autocast: fp8 linears handle their own casting internally;
    # bf16 covers norms, embeddings, activations, and the lm_head.
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
              if "cuda" in device else nullcontext()
    attn_ctx_factory = setup_attention_backend(device=device, master=master, require_fa4=args.require_fa4)

    # ------------------------------------------------------------------ Model
    model = get_model(asdict(cfg)).to(device)
    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M parameters")
        if cfg.fp8:
            if not _torchao_available:
                raise RuntimeError("fp8=True but torchao is not installed. Run: pip install torchao")
            print("[fp8] FP8 training enabled via torchao (ROWWISE dynamic scaling — B300 optimised)")
        if wandb is not None and args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=asdict(cfg),
            )
            wandb.run.summary["n_params"] = n_params

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------ Optimizer
    raw_model = model.module if ddp else model
    # Unwrap torch.compile's OptimizedModule to get stable parameter names
    underlying = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
    if cfg.muon and MuonWithAuxAdam is not None:
        # Split: hidden weight matrices (Muon) vs embeddings/head/norms/biases (AdamW)
        seen = set()
        hidden_weights, nonhidden_params = [], []
        for param in underlying.transformer.h.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                (hidden_weights if param.ndim >= 2 else nonhidden_params).append(param)
        for param in underlying.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                nonhidden_params.append(param)
        optimizer = MuonWithAuxAdam([
            dict(params=hidden_weights,   use_muon=True,  lr=cfg.muon_lr, weight_decay=cfg.weight_decay),
            dict(params=nonhidden_params, use_muon=False, lr=cfg.max_lr,  betas=(0.9, 0.95), weight_decay=cfg.weight_decay),
        ])
        if master:
            print(f"[optim] Muon: {len(hidden_weights)} hidden tensors | AdamW: {len(nonhidden_params)} other tensors")
    else:
        if cfg.muon and MuonWithAuxAdam is None:
            print("[optim] WARNING: muon=True but muon package not found, falling back to AdamW")
        decay_params   = [p for n, p in underlying.named_parameters()
                          if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in underlying.named_parameters()
                          if p.requires_grad and p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [{"params": decay_params,   "weight_decay": cfg.weight_decay},
             {"params": nodecay_params, "weight_decay": 0.0}],
            lr=cfg.max_lr, betas=(0.9, 0.95), fused="cuda" in device,
        )

    # ------------------------------------------------------------------ Data
    dataset = BinDataset(cfg.data_dir, cfg.seq_len, cfg.token_dtype)

    # ------------------------------------------------------------------ Train
    step        = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    world_size = dist.get_world_size() if ddp else 1
    total_tokens = 0
    metrics_count = 0
    tokens_per_sec_sum = 0.0
    compute_usage_sum = 0.0
    peak_mem_reserved = 0.0
    final_loss = None
    stop_reason = "max_steps"

    while step < cfg.max_steps:

        # Time-limit check — never starts a new step after the deadline
        elapsed = time.time() - train_start
        stop = torch.tensor(int(elapsed >= cfg.time_limit_seconds), device=device)
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item():
            if master:
                print(f"\n[time] {elapsed/60:.1f} min elapsed — time limit reached.")
                save_checkpoint(model, step, cfg)
            stop_reason = "time_limit"
            break

        step_start = time.time()
        data_time = 0.0
        compute_time = 0.0
        if "cuda" in device:
            torch.cuda.reset_peak_memory_stats()
        adamw_lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.muon_lr * (adamw_lr / cfg.max_lr) if pg.get("use_muon", False) else adamw_lr

        # Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            t_data = time.time()
            x, y     = dataset.get_batch(cfg.batch_size, device)
            data_time += time.time() - t_data

            sync_ctx = model.no_sync() if (ddp and micro_step < cfg.grad_accum_steps - 1) \
                       else nullcontext()
            t_compute = time.time()
            with sync_ctx, amp_ctx, attn_ctx_factory():
                _, loss = model(x, y)
                loss    = loss / cfg.grad_accum_steps
            loss.backward()
            compute_time += time.time() - t_compute
            accumulated_loss += loss.item()

        t_opt = time.time()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        compute_time += time.time() - t_opt

        step += 1
        step_time = time.time() - step_start
        tokens_this_step = cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len * world_size
        total_tokens += tokens_this_step
        tokens_per_sec = tokens_this_step / max(step_time, 1e-9)
        compute_usage_pct_local = 100.0 * compute_time / max(step_time, 1e-9)
        mem_stats_local = get_memory_stats(device)
        aggregated = aggregate_distributed_metrics(
            ddp=ddp,
            device=device,
            metrics={
                "compute_usage_pct": compute_usage_pct_local,
                "mem_allocated_gib": mem_stats_local["mem_allocated_gib"],
                "mem_reserved_gib": mem_stats_local["mem_reserved_gib"],
                "mem_peak_allocated_gib": mem_stats_local["mem_peak_allocated_gib"],
                "mem_peak_reserved_gib": mem_stats_local["mem_peak_reserved_gib"],
            },
        )

        if master:
            elapsed_total = time.time() - train_start
            remaining = max(0, cfg.time_limit_seconds - elapsed_total)
            step_metrics = {
                "step": step,
                "loss": accumulated_loss,
                "lr": adamw_lr,
                "step_time_sec": step_time,
                "data_time_sec": data_time,
                "compute_time_sec": compute_time,
                "compute_usage_pct": aggregated["compute_usage_pct"],
                "tokens_this_step": tokens_this_step,
                "tokens_total": total_tokens,
                "tokens_per_sec": tokens_per_sec,
                "mem_allocated_gib": aggregated["mem_allocated_gib"],
                "mem_allocated_gib_max": aggregated["mem_allocated_gib_max"],
                "mem_reserved_gib": aggregated["mem_reserved_gib"],
                "mem_reserved_gib_max": aggregated["mem_reserved_gib_max"],
                "mem_peak_allocated_gib": aggregated["mem_peak_allocated_gib"],
                "mem_peak_allocated_gib_max": aggregated["mem_peak_allocated_gib_max"],
                "mem_peak_reserved_gib": aggregated["mem_peak_reserved_gib"],
                "mem_peak_reserved_gib_max": aggregated["mem_peak_reserved_gib_max"],
                "elapsed_minutes": elapsed_total / 60.0,
                "remaining_minutes": remaining / 60.0,
            }
            metrics_count += 1
            tokens_per_sec_sum += step_metrics["tokens_per_sec"]
            compute_usage_sum += step_metrics["compute_usage_pct"]
            peak_mem_reserved = max(peak_mem_reserved, step_metrics["mem_peak_reserved_gib_max"])
            final_loss = step_metrics["loss"]

        if master and wandb is not None and wandb.run is not None:
            wandb.log(step_metrics, step=step)

        if master and step % 10 == 0:
            elapsed_total = time.time() - train_start
            remaining     = max(0, cfg.time_limit_seconds - elapsed_total)
            print(f"step {step:6d} | loss {accumulated_loss:.4f} | "
                  f"lr {adamw_lr:.2e} | "
                  f"{step_time*1000:.0f}ms/step | "
                  f"tok/s {tokens_per_sec:,.0f} | "
                  f"tok total {total_tokens:,} | "
                f"compute avg/max {aggregated['compute_usage_pct']:5.1f}%/{aggregated['compute_usage_pct_max']:5.1f}% | "
                f"mem peak avg/max {aggregated['mem_peak_reserved_gib']:.2f}/{aggregated['mem_peak_reserved_gib_max']:.2f}GiB | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"time left {remaining/60:.1f}m")

    # max_steps reached cleanly
    if step >= cfg.max_steps and master:
        print(f"\n[done] Reached max_steps={cfg.max_steps}.")
        save_checkpoint(model, step, cfg)

    if master:
        elapsed_total = time.time() - train_start
        if metrics_count > 0:
            avg_tokens_per_sec = tokens_per_sec_sum / metrics_count
            avg_compute_usage = compute_usage_sum / metrics_count
        else:
            avg_tokens_per_sec = 0.0
            avg_compute_usage = 0.0
            peak_mem_reserved = 0.0
            final_loss = None

        summary = {
            "stop_reason": stop_reason,
            "steps_completed": step,
            "elapsed_seconds": elapsed_total,
            "tokens_total": total_tokens,
            "tokens_per_sec_avg": avg_tokens_per_sec,
            "compute_usage_pct_avg": avg_compute_usage,
            "mem_peak_reserved_gib_max": peak_mem_reserved,
            "final_loss": final_loss,
            "world_size": world_size,
            "config": asdict(cfg),
        }
        save_metrics_reports(cfg, summary)
        if wandb is not None and wandb.run is not None:
            wandb.run.summary.update(summary)
            wandb.finish()

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

