"""
Single-GPU training script for H100 80GB — based on fp8-fa4-combined with all DA-approved fixes.
No DDP/FSDP — just one GPU, clean and simple.
"""

import os
import time
import glob
import math
import json
import argparse
import importlib
import queue
import threading
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    import wandb
except ImportError:
    wandb = None

from model import get_model

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    data_dir:    str   = "data"
    token_dtype: str   = "uint16"
    seq_len:     int   = 1024

    # Model — 1B GPT defaults
    vocab_size: int   = 32000
    n_layer:    int   = 18
    n_head:     int   = 16
    n_embd:     int   = 2048
    dropout:    float = 0.0

    # Training
    batch_size:       int   = 4
    grad_accum_steps: int   = 8
    max_lr:           float = 3e-4
    min_lr:           float = 3e-5
    warmup_steps:     int   = 100
    max_steps:        int   = 10_000
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    time_limit_seconds: float = 30 * 60

    # Checkpointing
    checkpoint_path: str = "checkpoint.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinDataset:
    """Memory-maps all *.bin files and draws random (seq_len+1)-token windows.
    Prefetches batches in a background thread so GPU never waits on CPU data loading."""

    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16", prefetch: int = 2):
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

        self._device     = None
        self._batch_size = None
        self._queue      = queue.Queue(maxsize=prefetch)
        self._thread     = None

    def _worker(self):
        while True:
            item = self._build_batch(self._batch_size)
            self._queue.put(item)

    def _build_batch(self, batch_size):
        xs, ys = [], []
        for _ in range(batch_size):
            shard = self.shards[np.random.choice(len(self.shards), p=self.weights)]
            start = np.random.randint(0, len(shard) - self.seq_len - 1)
            chunk = torch.from_numpy(shard[start:start + self.seq_len + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.stack(xs).pin_memory()
        y = torch.stack(ys).pin_memory()
        return x, y

    def start_prefetch(self, batch_size: int, device):
        self._batch_size = batch_size
        self._device     = device
        self._thread     = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def get_batch(self, batch_size: int, device):
        if self._thread is None:
            self.start_prefetch(batch_size, device)
        x, y = self._queue.get()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay -> min_lr
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
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "step":   step,
        "model":  inner.state_dict(),
        "config": asdict(cfg),
    }, cfg.checkpoint_path)
    print(f"[ckpt] saved -> {cfg.checkpoint_path}  (step {step})")


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

    print(f"[metrics] saved -> {json_path}")


def setup_attention_backend(device: str):
    """Configure attention backend and return context manager factory."""
    if "cuda" not in device:
        print("[attn] CUDA not detected; using default attention path.")
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

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    if fa4_available:
        print(f"[attn] flash_attn {flash_attn_version} detected (FA4+). Flash SDPA forced.")
    else:
        print("[attn] Using PyTorch flash SDPA backend.")

    return lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          default="data")
    parser.add_argument("--checkpoint_path",   default="checkpoint.pt")
    parser.add_argument("--seq_len",           type=int,   default=1024)
    parser.add_argument("--vocab_size",        type=int,   default=32000)
    parser.add_argument("--n_layer",           type=int,   default=18)
    parser.add_argument("--n_head",            type=int,   default=16)
    parser.add_argument("--n_embd",            type=int,   default=2048)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--grad_accum_steps",  type=int,   default=8)
    parser.add_argument("--max_steps",         type=int,   default=10_000)
    parser.add_argument("--time_limit_min",    type=float, default=30.0)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
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
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.set_float32_matmul_precision("high")

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
              if "cuda" in device else nullcontext()
    attn_ctx_factory = setup_attention_backend(device)

    if TE_AVAILABLE and "cuda" in device:
        fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)
        fp8_ctx_factory = lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
        print("[fp8] Transformer Engine FP8 enabled (HYBRID format)")
    else:
        fp8_ctx_factory = nullcontext
        print("[fp8] TE not available, running in bfloat16 only")

    # ------------------------------------------------------------------ Model
    model = get_model(asdict(cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.1f}M parameters on {device}")

    model = torch.compile(model, mode="max-autotune")

    if wandb is not None and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=asdict(cfg),
        )
        wandb.run.summary["n_params"] = n_params

    # ------------------------------------------------------------------ Optimizer
    decay_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
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
    total_tokens = 0
    metrics_count = 0
    tokens_per_sec_sum = 0.0
    compute_usage_sum = 0.0
    peak_mem_reserved = 0.0
    final_loss = None
    stop_reason = "max_steps"

    while step < cfg.max_steps:

        elapsed = time.time() - train_start
        if elapsed >= cfg.time_limit_seconds:
            print(f"\n[time] {elapsed/60:.1f} min elapsed — time limit reached.")
            save_checkpoint(model, step, cfg)
            stop_reason = "time_limit"
            break

        step_start = time.time()
        data_time = 0.0
        compute_time = 0.0
        if "cuda" in device:
            torch.cuda.reset_peak_memory_stats()
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step, cfg)

        # Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            t_data = time.time()
            x, y = dataset.get_batch(cfg.batch_size, device)
            data_time += time.time() - t_data

            t_compute = time.time()
            with amp_ctx, fp8_ctx_factory(), attn_ctx_factory():
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
        tokens_this_step = cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len
        total_tokens += tokens_this_step
        tokens_per_sec = tokens_this_step / max(step_time, 1e-9)
        compute_usage_pct = 100.0 * compute_time / max(step_time, 1e-9)
        mem_stats = get_memory_stats(device)

        elapsed_total = time.time() - train_start
        remaining = max(0, cfg.time_limit_seconds - elapsed_total)
        step_metrics = {
            "step": step,
            "loss": accumulated_loss,
            "lr": get_lr(step, cfg),
            "step_time_sec": step_time,
            "data_time_sec": data_time,
            "compute_time_sec": compute_time,
            "compute_usage_pct": compute_usage_pct,
            "tokens_this_step": tokens_this_step,
            "tokens_total": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "mem_allocated_gib": mem_stats["mem_allocated_gib"],
            "mem_reserved_gib": mem_stats["mem_reserved_gib"],
            "mem_peak_allocated_gib": mem_stats["mem_peak_allocated_gib"],
            "mem_peak_reserved_gib": mem_stats["mem_peak_reserved_gib"],
            "elapsed_minutes": elapsed_total / 60.0,
            "remaining_minutes": remaining / 60.0,
        }
        metrics_count += 1
        tokens_per_sec_sum += tokens_per_sec
        compute_usage_sum += compute_usage_pct
        peak_mem_reserved = max(peak_mem_reserved, mem_stats["mem_peak_reserved_gib"])
        final_loss = accumulated_loss

        if wandb is not None and wandb.run is not None:
            wandb.log(step_metrics, step=step)

        if step % 10 == 0:
            print(f"step {step:6d} | loss {accumulated_loss:.4f} | "
                  f"lr {get_lr(step, cfg):.2e} | "
                  f"{step_time*1000:.0f}ms/step | "
                  f"tok/s {tokens_per_sec:,.0f} | "
                  f"tok total {total_tokens:,} | "
                  f"compute {compute_usage_pct:5.1f}% | "
                  f"mem peak {mem_stats['mem_peak_reserved_gib']:.2f}GiB | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"time left {remaining/60:.1f}m")

    # max_steps reached
    if step >= cfg.max_steps:
        print(f"\n[done] Reached max_steps={cfg.max_steps}.")
        save_checkpoint(model, step, cfg)

    elapsed_total = time.time() - train_start
    if metrics_count > 0:
        avg_tokens_per_sec = tokens_per_sec_sum / metrics_count
        avg_compute_usage = compute_usage_sum / metrics_count
    else:
        avg_tokens_per_sec = 0.0
        avg_compute_usage = 0.0

    summary = {
        "stop_reason": stop_reason,
        "steps_completed": step,
        "elapsed_seconds": elapsed_total,
        "tokens_total": total_tokens,
        "tokens_per_sec_avg": avg_tokens_per_sec,
        "compute_usage_pct_avg": avg_compute_usage,
        "mem_peak_reserved_gib": peak_mem_reserved,
        "final_loss": final_loss,
        "world_size": 1,
        "config": asdict(cfg),
    }
    save_metrics_reports(cfg, summary)
    if wandb is not None and wandb.run is not None:
        wandb.run.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
