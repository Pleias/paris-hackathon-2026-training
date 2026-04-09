"""
Starter training script for the gpu-mode Paris hackathon training track
"""

import os
import time
import glob
import math
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
            break

        step_start = time.time()
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step, cfg)

        # Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            x, y     = dataset.get_batch(cfg.batch_size, device)
            sync_ctx = model.no_sync() if (ddp and micro_step < cfg.grad_accum_steps - 1) \
                       else nullcontext()
            with sync_ctx, amp_ctx:
                _, loss = model(x, y)
                loss    = loss / cfg.grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1

        if master and step % 10 == 0:
            elapsed_total = time.time() - train_start
            remaining     = max(0, cfg.time_limit_seconds - elapsed_total)
            print(f"step {step:6d} | loss {accumulated_loss:.4f} | "
                  f"lr {get_lr(step, cfg):.2e} | "
                  f"{(time.time()-step_start)*1000:.0f}ms/step | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"time left {remaining/60:.1f}m")

    # max_steps reached cleanly
    if step >= cfg.max_steps and master:
        print(f"\n[done] Reached max_steps={cfg.max_steps}.")
        save_checkpoint(model, step, cfg)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

