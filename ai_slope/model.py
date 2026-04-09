"""
Vanilla PyTorch Llama-7B architecture.

No Megatron-Core dependencies — this file is loaded by both train.py (for
checkpoint remapping) and by the evaluator (via get_model / load_state_dict).

CONTRACT
--------
    get_model(config: dict) -> nn.Module

The returned model's forward must be:
    forward(idx: LongTensor[B, T], targets: LongTensor[B, T] | None = None)
        -> (logits: Tensor[B, T, vocab_size], loss: Tensor | None)

Parameter naming used by train.py's weight remapper
----------------------------------------------------
Each transformer layer i has:
  layers.{i}.attention_norm.weight          -- pre-attention RMSNorm
  layers.{i}.attention.wq                   -- Q projection  [n_heads*head_dim, hidden]
  layers.{i}.attention.wk                   -- K projection  [n_kv_heads*head_dim, hidden]
  layers.{i}.attention.wv                   -- V projection  [n_kv_heads*head_dim, hidden]
  layers.{i}.attention.wo                   -- output proj   [hidden, n_heads*head_dim]
  layers.{i}.ffn_norm.weight                -- pre-FFN RMSNorm
  layers.{i}.feed_forward.w1                -- gate proj     [intermediate, hidden]
  layers.{i}.feed_forward.w3                -- up proj       [intermediate, hidden]
  layers.{i}.feed_forward.w2                -- down proj     [hidden, intermediate]
Top-level:
  tok_embeddings.weight                     -- token embedding
  norm.weight                               -- final RMSNorm
  output.weight                             -- LM head (not tied)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor):
    """xq/xk: [B, T, n_heads, head_dim]"""
    def rotate(x):
        x_f = x.float().reshape(*x.shape[:-1], -1, 2)
        x_c = torch.view_as_complex(x_f)
        x_r = torch.view_as_real(x_c * freqs_cis.unsqueeze(0).unsqueeze(2))
        return x_r.flatten(-2).to(x.dtype)
    return rotate(xq), rotate(xk)


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int,
                 head_dim: int):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.n_rep      = n_heads // n_kv_heads  # groups per KV head

        self.wq = nn.Linear(hidden_size, n_heads    * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden_size,     bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        # Expand KV heads to match Q heads (GQA)
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # [B, n_heads, T, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, intermediate_size: int):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size)
        self.attention      = Attention(hidden_size, n_heads, n_kv_heads, head_dim)
        self.ffn_norm       = RMSNorm(hidden_size)
        self.feed_forward   = FeedForward(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Llama model
# ---------------------------------------------------------------------------

class Llama(nn.Module):
    def __init__(
        self,
        vocab_size:        int,
        hidden_size:       int,
        num_layers:        int,
        num_attention_heads:   int,
        num_key_value_heads:   int,
        intermediate_size: int,
        seq_len:           int,
        rope_theta:        float = 10000.0,
    ):
        super().__init__()
        head_dim = hidden_size // num_attention_heads

        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_attention_heads,
                             num_key_value_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm   = RMSNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        # No weight tying — per spec (tie_word_embeddings: false)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, seq_len, theta=rope_theta),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        n_layers = len(self.layers)
        for name, p in self.named_parameters():
            if "tok_embeddings" in name or "output" in name:
                nn.init.normal_(p, mean=0.0, std=std)
            elif p.dim() >= 2:
                if any(s in name for s in ("wo.weight", "w2.weight")):
                    # residual projection scaling
                    nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * n_layers))
                else:
                    nn.init.normal_(p, mean=0.0, std=std)
            else:
                nn.init.ones_(p)  # RMSNorm weights

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = self.norm(x)
        logits = self.output(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


# ---------------------------------------------------------------------------
# Competition interface
# ---------------------------------------------------------------------------

def get_model(config: dict) -> nn.Module:
    """
    Reconstruct the Llama-7B model from a config dict.
    Called by train.py (before training) and by the evaluator after loading
    checkpoint.pt.

    Expected keys (with Llama-7B defaults):
        vocab_size            int   32768
        hidden_size           int   4096
        num_layers            int   32
        num_attention_heads   int   32
        num_key_value_heads   int   8
        intermediate_size     int   11008
        seq_len               int   1024
        rope_theta            float 10000.0
    """
    return Llama(
        vocab_size          = config.get("vocab_size",          32768),
        hidden_size         = config.get("hidden_size",         4096),
        num_layers          = config.get("num_layers",          32),
        num_attention_heads = config.get("num_attention_heads", 32),
        num_key_value_heads = config.get("num_key_value_heads", 8),
        intermediate_size   = config.get("intermediate_size",   11008),
        seq_len             = config.get("seq_len",             1024),
        rope_theta          = config.get("rope_theta",          10000.0),
    )
