"""
TorchTitan-based variant of the starter training script using Qwen3 ~0.8B.

Checkpoint contract: checkpoint.pt contains {"step", "model", "config"}.
"""

import os
import time
import glob
import json
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

try:
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.components.lr_scheduler import LRSchedulersContainer
    from torchtitan.models.common import Embedding, Linear, RoPE
    from torchtitan.models.common.attention import ScaledDotProductAttention
    from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
    from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
    from torchtitan.models.common.rmsnorm import RMSNorm
except ImportError as exc:
    raise ImportError(
        "TorchTitan is not installed. Install from source, e.g. "
        "pip install git+https://github.com/pytorch/torchtitan.git"
    ) from exc

from qwen import Qwen3Model, Qwen3TransformerBlock


# ---------------------------------------------------------------------------
# Qwen3 ~0.8B model builder
# ---------------------------------------------------------------------------

_EPS = 1e-6
_LINEAR_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02), "bias": nn.init.zeros_}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


def _output_init(dim: int) -> dict:
    s = dim ** -0.5
    return {"weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s)}


def _depth_init(layer_id: int) -> dict:
    return {"weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id))}


def _norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def build_qwen3_0_8b(vocab_size: int, seq_len: int) -> Qwen3Model:
    """Qwen3-style dense model targeting ~0.8B parameters.

    Architecture (matches Qwen3 family conventions):
      dim=1280, n_layers=32, n_heads=16, n_kv_heads=8, head_dim=128, hidden_dim=4096
    Parameter count (with weight tying):
      32 × (attention ~7.9M + FFN ~15.7M) + embeddings (~vocab×1280) ≈ 0.80B @ vocab=32768
    """
    dim = 1280
    head_dim = 128
    n_heads = 16
    n_kv_heads = 8
    hidden_dim = 4096
    n_layers = 32

    layers = []
    for layer_id in range(n_layers):
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_norm(dim),
                ffn_norm=_norm(dim),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=ScaledDotProductAttention.Config(),
                    mask_type="causal",
                    rope_backend="cos_sin",
                    q_norm=_norm(head_dim),
                    k_norm=_norm(head_dim),
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )

    config = Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        enable_weight_tying=True,
        norm=_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=seq_len,
            theta=1_000_000.0,
            backend="cos_sin",
        ),
        layers=layers,
    )
    model = Qwen3Model(config)
    model.init_states()
    return model


class Qwen3LM(nn.Module):
    """Thin wrapper giving Qwen3Model the (idx, targets) -> (logits, loss) interface."""

    def __init__(self, vocab_size: int, seq_len: int):
        super().__init__()
        self.model = build_qwen3_0_8b(vocab_size, seq_len)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.model(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    data_dir: str = "data"
    token_dtype: str = "uint16"
    seq_len: int = 1024

    # Model — architecture is fixed at Qwen3 ~0.8B; only vocab_size is configurable
    vocab_size: int = 32768

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 4
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 10_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
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
        self.seq_len = seq_len
        np_dtype = np.dtype(dtype)
        self.shards = [np.memmap(p, dtype=np_dtype, mode="r") for p in paths]
        self.lengths = [len(s) for s in self.shards]
        self.total = sum(self.lengths)
        self.weights = [l / self.total for l in self.lengths]
        print(f"[data] {len(paths)} shard(s), {self.total:,} tokens total")

    def get_batch(self, batch_size: int, device):
        xs, ys = [], []
        for _ in range(batch_size):
            shard = self.shards[np.random.choice(len(self.shards), p=self.weights)]
            start = np.random.randint(0, len(shard) - self.seq_len - 1)
            chunk = torch.from_numpy(shard[start : start + self.seq_len + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, step: int, cfg: Config):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "config": asdict(cfg),
        },
        cfg.checkpoint_path,
    )
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

    print(f"[metrics] saved -> {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--checkpoint_path", default="checkpoint.pt")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--time_limit_min", type=float, default=10.0)
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (omit to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        time_limit_seconds=args.time_limit_min * 60,
    )

    # ------------------------------------------------------------------ DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        rank = 0
        master = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337 + rank)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if "cuda" in device
        else nullcontext()
    )

    # ------------------------------------------------------------------ Model
    model = Qwen3LM(cfg.vocab_size, cfg.seq_len).to(device)
    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M parameters")
        if wandb is not None and args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=asdict(cfg),
            )
            wandb.run.summary["n_params"] = n_params

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if ddp else model

    # ------------------------------------------------------------------ TorchTitan Optimizer + Scheduler
    optimizer_impl = "fused" if "cuda" in device else "foreach"
    tt_optimizer_cfg = OptimizersContainer.Config(
        name="AdamW",
        lr=cfg.max_lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=cfg.weight_decay,
        implementation=optimizer_impl,
    )
    optimizers = tt_optimizer_cfg.build(model_parts=[raw_model])

    min_lr_factor = cfg.min_lr / cfg.max_lr if cfg.max_lr > 0 else 0.0
    tt_lr_cfg = LRSchedulersContainer.Config(
        warmup_steps=cfg.warmup_steps,
        decay_ratio=None,
        decay_type="cosine",
        min_lr_factor=min_lr_factor,
    )
    lr_schedulers = tt_lr_cfg.build(optimizers=optimizers, training_steps=cfg.max_steps)

    # ------------------------------------------------------------------ Data
    dataset = BinDataset(cfg.data_dir, cfg.seq_len, cfg.token_dtype)

    # ------------------------------------------------------------------ Train
    step = 0
    train_start = time.time()
    model.train()
    optimizers.zero_grad()
    world_size = dist.get_world_size() if ddp else 1
    total_tokens = 0
    metrics_count = 0
    tokens_per_sec_sum = 0.0
    compute_usage_sum = 0.0
    peak_mem_reserved = 0.0
    final_loss = None
    stop_reason = "max_steps"

    while step < cfg.max_steps:
        # Time-limit check: do not start a new step after deadline.
        elapsed = time.time() - train_start
        stop = torch.tensor(int(elapsed >= cfg.time_limit_seconds), device=device)
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item():
            if master:
                print(f"\n[time] {elapsed/60:.1f} min elapsed - time limit reached.")
                save_checkpoint(model, step, cfg)
            stop_reason = "time_limit"
            break

        step_start = time.time()
        data_time = 0.0
        compute_time = 0.0
        if "cuda" in device:
            torch.cuda.reset_peak_memory_stats()

        # Gradient accumulation
        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            t_data = time.time()
            x, y = dataset.get_batch(cfg.batch_size, device)
            data_time += time.time() - t_data

            sync_ctx = (
                model.no_sync() if (ddp and micro_step < cfg.grad_accum_steps - 1) else nullcontext()
            )
            t_compute = time.time()
            with sync_ctx, amp_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            loss.backward()
            compute_time += time.time() - t_compute
            accumulated_loss += loss.item()

        t_opt = time.time()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizers.step()
        lr_schedulers.step()
        optimizers.zero_grad(set_to_none=True)
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
            remaining = max(0.0, cfg.time_limit_seconds - elapsed_total)
            current_lr = optimizers.optimizers[0].param_groups[0]["lr"]
            step_metrics = {
                "step": step,
                "loss": accumulated_loss,
                "lr": current_lr,
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
            remaining = max(0.0, cfg.time_limit_seconds - elapsed_total)
            current_lr = optimizers.optimizers[0].param_groups[0]["lr"]
            print(
                f"step {step:6d} | loss {accumulated_loss:.4f} | "
                f"lr {current_lr:.2e} | "
                f"{step_time*1000:.0f}ms/step | "
                f"tok/s {tokens_per_sec:,.0f} | "
                f"tok total {total_tokens:,} | "
                f"compute avg/max {aggregated['compute_usage_pct']:5.1f}%/{aggregated['compute_usage_pct_max']:5.1f}% | "
                f"mem peak avg/max {aggregated['mem_peak_reserved_gib']:.2f}/{aggregated['mem_peak_reserved_gib_max']:.2f}GiB | "
                f"elapsed {elapsed_total/60:.1f}m | "
                f"time left {remaining/60:.1f}m"
            )

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
