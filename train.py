"""
Starter training script for the gpu-mode Paris hackathon training track
"""

import os
import time
import glob
import math
import csv
import json
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from model import get_model


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
    vocab_size: int   = 32768
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    dropout:    float = 0.0

    # Training
    batch_size:       int   = 8
    grad_accum_steps: int   = 4
    max_lr:           float = 6e-4
    min_lr:           float = 6e-5
    warmup_steps:     int   = 100
    max_steps:        int   = 10_000
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    time_limit_seconds: float = 10 * 60

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


def save_metrics_reports(cfg: Config, records: list[dict], summary: dict):
    base, _ = os.path.splitext(cfg.checkpoint_path)
    json_path = f"{base}_metrics.json"
    csv_path = f"{base}_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    fields = [
        "step",
        "loss",
        "lr",
        "step_time_sec",
        "data_time_sec",
        "compute_time_sec",
        "compute_usage_pct",
        "tokens_this_step",
        "tokens_total",
        "tokens_per_sec",
        "mem_allocated_gib",
        "mem_reserved_gib",
        "mem_peak_allocated_gib",
        "mem_peak_reserved_gib",
        "elapsed_minutes",
        "remaining_minutes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"[metrics] saved → {json_path}")
    print(f"[metrics] saved → {csv_path}")


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
    parser.add_argument("--n_embd",            type=int,   default=768)
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--grad_accum_steps",  type=int,   default=4)
    parser.add_argument("--max_steps",         type=int,   default=10_000)
    parser.add_argument("--time_limit_min",    type=float, default=10.0)
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
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
              if "cuda" in device else nullcontext()

    # ------------------------------------------------------------------ Model
    model = get_model(asdict(cfg)).to(device)
    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M parameters")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ------------------------------------------------------------------ Optimizer
    raw_model      = model.module if ddp else model
    decay_params   = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in raw_model.named_parameters()
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
    metrics_records = []
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
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step, cfg)

        # Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            t_data = time.time()
            x, y     = dataset.get_batch(cfg.batch_size, device)
            data_time += time.time() - t_data

            sync_ctx = model.no_sync() if (ddp and micro_step < cfg.grad_accum_steps - 1) \
                       else nullcontext()
            t_compute = time.time()
            with sync_ctx, amp_ctx:
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
        compute_usage_pct = 100.0 * compute_time / max(step_time, 1e-9)
        mem_stats = get_memory_stats(device)

        if master:
            elapsed_total = time.time() - train_start
            remaining = max(0, cfg.time_limit_seconds - elapsed_total)
            metrics_records.append({
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
            })

        if master and step % 10 == 0:
            elapsed_total = time.time() - train_start
            remaining     = max(0, cfg.time_limit_seconds - elapsed_total)
            print(f"step {step:6d} | loss {accumulated_loss:.4f} | "
                  f"lr {get_lr(step, cfg):.2e} | "
                  f"{step_time*1000:.0f}ms/step | "
                  f"tok/s {tokens_per_sec:,.0f} | "
                  f"tok total {total_tokens:,} | "
                  f"compute {compute_usage_pct:5.1f}% | "
                  f"mem {mem_stats['mem_peak_reserved_gib']:.2f}GiB peak | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"time left {remaining/60:.1f}m")

    # max_steps reached cleanly
    if step >= cfg.max_steps and master:
        print(f"\n[done] Reached max_steps={cfg.max_steps}.")
        save_checkpoint(model, step, cfg)

    if master:
        elapsed_total = time.time() - train_start
        if metrics_records:
            avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in metrics_records) / len(metrics_records)
            avg_compute_usage = sum(r["compute_usage_pct"] for r in metrics_records) / len(metrics_records)
            peak_mem_reserved = max(r["mem_peak_reserved_gib"] for r in metrics_records)
            final_loss = metrics_records[-1]["loss"]
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
        save_metrics_reports(cfg, metrics_records, summary)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

