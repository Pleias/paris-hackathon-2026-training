"""
Starter reference model definition.

CONTRACT
--------------------
Must expose exactly one function:

    get_model(config: dict) -> torch.nn.Module

The returned model's forward method must have the signature:

    forward(idx: LongTensor[B, T], targets: LongTensor[B, T] | None = None)
        -> (logits: Tensor[B, T, vocab_size], loss: Tensor | None)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as _fa4_func
    _FA4_AVAILABLE = True
except ImportError:
    _FA4_AVAILABLE = False

torch.set_float32_matmul_precision("high")


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, seq_len, dropout, attn_backend="auto"):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj  = nn.Linear(n_embd, n_embd,     bias=False)
        self.dropout = dropout

        if attn_backend == "fa4":
            if not _FA4_AVAILABLE:
                raise RuntimeError("attn_backend='fa4' requested but flash_attn is not installed")
            self.use_fa4 = True
        elif attn_backend == "sdpa":
            self.use_fa4 = False
        else:  # "auto"
            self.use_fa4 = _FA4_AVAILABLE

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head

        if self.use_fa4:
            # FA4 expects (B, T, H, D)
            q = q.view(B, T, self.n_head, hd)
            k = k.view(B, T, self.n_head, hd)
            v = v.view(B, T, self.n_head, hd)
            y = _fa4_func(q, k, v, causal=True)
            y = y.contiguous().view(B, T, C)
        else:
            # SDPA expects (B, H, T, D)
            q = q.view(B, T, self.n_head, hd).transpose(1, 2)
            k = k.view(B, T, self.n_head, hd).transpose(1, 2)
            v = v.view(B, T, self.n_head, hd).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0.0)
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc   = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, seq_len, dropout, attn_backend="auto"):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, seq_len, dropout, attn_backend=attn_backend)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_embd, dropout=0.0, attn_backend="auto"):
        super().__init__()
        self.seq_len = seq_len
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, n_embd),
            wpe  = nn.Embedding(seq_len,    n_embd),
            drop = nn.Dropout(dropout),
            h    = nn.ModuleList([Block(n_embd, n_head, seq_len, dropout, attn_backend=attn_backend)
                                  for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Competition interface — participants must implement this
# ---------------------------------------------------------------------------

# torch 
def get_model(config: dict) -> nn.Module:
    """
    Instantiate and return the model from a config dict.
    Called by both train.py (before training) and eval.py (to load a checkpoint).
    """
    return torch.compile(GPT(
        vocab_size   = config.get("vocab_size", 32768),
        seq_len      = config.get("seq_len",    1024),
        n_layer      = config.get("n_layer",    12),
        n_head       = config.get("n_head",     12),
        n_embd       = config.get("n_embd",     768),
        dropout      = config.get("dropout",    0.0),
        attn_backend = config.get("attn_backend", "auto"),
    ), mode="max-autotune")

