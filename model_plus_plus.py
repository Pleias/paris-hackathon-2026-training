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

class CausalConv1d(nn.Module):
    """Depthwise 1D Convolution for Canon Layers."""
    def __init__(self, channels, kernel_size=4):
        super().__init__()
        # groups=channels makes it depthwise (very fast)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size - 1, groups=channels)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        T = x.size(1)
        x = x.transpose(1, 2)
        # Apply conv and slice to truncate padding to maintain causality
        x = self.conv(x)[..., :T]
        return x.transpose(1, 2).contiguous()

class EngramModule(nn.Module):
    """O(1) Conditional Memory via Scalable Lookup (DeepSeek Engram)."""
    def __init__(self, n_embd, table_size=131072, num_heads=8, n_gram=3):
        super().__init__()
        self.n_gram = n_gram
        self.num_heads = num_heads
        self.table_size = table_size
        self.head_dim = n_embd // num_heads
        
        # Flattened memory table for fast F.embedding lookups
        self.memory = nn.Parameter(torch.empty(num_heads * table_size, self.head_dim))
        nn.init.normal_(self.memory, mean=0.0, std=0.02)
        
        # Large prime seeds for vectorized hashing
        self.register_buffer("hash_seeds", torch.randint(1, 2**31, (num_heads, n_gram), dtype=torch.int64))
        
        # Conditioning gate for fusion
        self.gate = nn.Linear(n_embd * 2, n_embd)
        
    def forward(self, x, idx):
        B, T = idx.shape
        
        # Fast causal N-gram extraction using unfold (B, T, n_gram)
        padded_idx = F.pad(idx, (self.n_gram - 1, 0), value=0)
        ngrams = padded_idx.unfold(1, self.n_gram, 1)
        
        # Vectorized polynomial hashing
        ngrams_exp = ngrams.unsqueeze(1)  # (B, 1, T, n_gram)
        seeds_exp = self.hash_seeds.view(1, self.num_heads, 1, self.n_gram)
        
        # Hash modulo table limits
        hash_idx = (ngrams_exp * seeds_exp).sum(dim=-1).abs() % self.table_size
        
        # Offset each head's hashes for the flattened memory lookup
        offset = torch.arange(self.num_heads, device=idx.device).view(1, self.num_heads, 1) * self.table_size
        flat_hash_idx = hash_idx + offset
        
        # O(1) Retrieval
        retrieved = F.embedding(flat_hash_idx, self.memory)  # (B, num_heads, T, head_dim)
        retrieved = retrieved.transpose(1, 2).contiguous().view(B, T, -1)
        
        # Conditioned fusion
        fusion_input = torch.cat([x, retrieved], dim=-1)
        g = torch.sigmoid(self.gate(fusion_input))
        return x + g * retrieved

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, seq_len, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj  = nn.Linear(n_embd, n_embd,     bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
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
        self.canon_d = CausalConv1d(4 * n_embd, kernel_size=4) # Canon at Position D
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.fc(x))
        h = self.canon_d(h)
        return self.drop(self.proj(h))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, seq_len, dropout, use_engram=False):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.canon_a = CausalConv1d(n_embd, kernel_size=4) # Canon at Position A
        self.attn = CausalSelfAttention(n_embd, n_head, seq_len, dropout)
        
        self.ln2  = nn.LayerNorm(n_embd)
        self.canon_c = CausalConv1d(n_embd, kernel_size=4) # Canon at Position C
        self.mlp  = MLP(n_embd, dropout)
        
        # Engram injection
        self.engram = EngramModule(n_embd) if use_engram else None

    def forward(self, x, idx):
        x = x + self.attn(self.canon_a(self.ln1(x)))
        x = x + self.mlp(self.canon_c(self.ln2(x)))
        if self.engram is not None:
            x = self.engram(x, idx)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layer, n_head, n_embd, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(vocab_size, n_embd),
            wpe  = nn.Embedding(seq_len,    n_embd),
            drop = nn.Dropout(dropout),
            # Injecting Engram into the first 4 layers (indices 0, 1, 2, 3)
            # h    = nn.ModuleList([Block(n_embd, n_head, seq_len, dropout, use_engram=(i < 4))
            #                       for i in range(n_layer)]),
            h    = nn.ModuleList([Block(n_embd, n_head, seq_len, dropout, use_engram=False)
                                  for i in range(n_layer)]),
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
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x, idx)
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

def get_model(config: dict) -> nn.Module:
    """
    Instantiate and return the model from a config dict.
    Called by both train.py (before training) and eval.py (to load a checkpoint).
    """
    model = GPT(
        vocab_size = config.get("vocab_size", 32768),
        seq_len    = config.get("seq_len",    1024),
        n_layer    = config.get("n_layer",    12),
        n_head     = config.get("n_head",     12),
        n_embd     = config.get("n_embd",     768),
        dropout    = config.get("dropout",    0.0),
    )

    if config.get("fp8", False):
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        from torchao.float8.config import Float8LinearRecipeName

        # ROWWISE uses per-row dynamic scaling — best for B300 (Blackwell)
        fp8_config = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE)

        def _fp8_filter(mod: nn.Module, fqn: str) -> bool:
            # Skip lm_head: weight-tied to wte embedding; converting it would
            # break the tie and the embedding itself is not a Linear.
            if fqn == "lm_head":
                return False
            return isinstance(mod, nn.Linear)

        convert_to_float8_training(model, config=fp8_config, module_filter_fn=_fp8_filter)

    return torch.compile(model)