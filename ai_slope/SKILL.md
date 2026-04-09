# Helion Kernel Writing — Comprehensive Skill Guide

## 1. Quick Reference

### What is Helion?

Helion is a Python-embedded domain-specific language (DSL) for authoring GPU machine-learning kernels. It compiles down to [Triton](https://github.com/triton-lang/triton), itself a performant GPU programming backend. You write kernels in ordinary PyTorch idioms, annotated with `hl.tile()` loops that divide work into tiles. Helion then automatically handles tensor indexing strategies, masking, grid sizes, PID calculations, loop ordering, L2 swizzling, persistent kernel strategies, and warp specialization — all surfaced as a rich autotuning search space. A single Helion kernel always compiles to exactly **one** GPU kernel.

### When to Use Helion vs Alternatives

| Situation | Recommended tool |
|---|---|
| Need a new fused op (e.g. fused softmax + norm) with minimal GPU code | **Helion** — highest productivity, auto-tunes 100s of Triton variants |
| Have a fixed, hand-crafted Triton kernel and don't need re-tuning | **Raw Triton** — direct control, zero abstraction overhead |
| Simple elementwise/pointwise fusion, no custom reduction logic | **`torch.compile`** — easiest, zero new code |
| Prototype then optimize | Start with `torch.compile`, move to Helion when perf matters |
| Maximum fine-grained GPU control (inline PTX, warp-level primitives) | **Raw Triton** |

### Installation

```bash
pip install helion
```

For development from source:

```bash
git clone https://github.com/pytorch/helion.git
cd helion
uv venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

**Requirements:** Linux, Python 3.10–3.14, PyTorch ≥ 2.9, Triton ≥ 3.5.

---

## 2. Core API

### `@helion.kernel()` decorator

```python
import helion
import helion.language as hl

@helion.kernel(
    # ── Configs ──────────────────────────────────────────────────────────────
    config=helion.Config(...),           # pin a single config, skip autotuning
    configs=[helion.Config(...), ...],   # pick best from these; lighter search

    # ── Autotuning effort ────────────────────────────────────────────────────
    autotune_effort="full",              # "none" | "quick" | "full" (default)
    # "none"  → use default config; fast iteration but NOT production quality
    # "quick" → small search; minutes instead of ~10 min
    # "full"  → full DE search, ~10 min, best perf

    # ── Shape specialization ─────────────────────────────────────────────────
    static_shapes=True,                  # True (default): autotune per shape
                                         # False: share config across shapes

    # ── Debugging ────────────────────────────────────────────────────────────
    print_output_code=True,              # print generated Triton to stderr
    print_repro=True,                    # print standalone repro script to stderr

    # ── Override specific config search dims ────────────────────────────────
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "num_warps": 4,
    },

    # ── Suppress specific warnings ───────────────────────────────────────────
    ignore_warnings=[helion.exc.TensorOperationInWrapper],

    # ── Other settings (all have env-var equivalents) ────────────────────────
    fast_math=False,
    index_dtype=None,                    # auto: int32 or int64 based on size
    dot_precision="tf32",                # "tf32" | "tf32x3" | "ieee"
    backend="triton",                    # "triton" | "pallas" | "cute" | "metal"
)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

**Without parentheses** (uses all defaults):

```python
@helion.kernel          # valid — equivalent to @helion.kernel()
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

**Environment variable equivalents** (useful in CI/debugging):

```bash
HELION_AUTOTUNE_EFFORT=none          # skip autotuning
HELION_PRINT_OUTPUT_CODE=1           # print Triton code to stderr
HELION_PRINT_REPRO=1                 # print repro script to stderr
HELION_FORCE_AUTOTUNE=1              # force re-tune even if config provided
HELION_STATIC_SHAPES=0               # disable per-shape specialization
HELION_AUTOTUNE_CACHE=LocalAutotuneCache
TRITON_INTERPRET=1                   # CPU interpreter for print/breakpoint
HELION_INTERPRET=1                   # Helion eager mode for print/breakpoint
```

### `hl.tile()` — tiling the iteration space

`hl.tile` is the central primitive. It divides a size (or list of sizes) into tiles that are executed **in parallel** on the GPU.

```python
# Single dimension → one Tile object per CTA
for tile_m in hl.tile(m):
    out[tile_m] = x[tile_m] + 1

# Two dimensions → one (tile_m, tile_n) pair per CTA
for tile_m, tile_n in hl.tile([m, n]):
    out[tile_m, tile_n] = x[tile_m, tile_n] * 2

# Explicit block size (bypasses autotuning for this dim)
bs = hl.register_block_size(m)
for tile_m in hl.tile(m, block_size=bs):
    ...

# Range variant — iterate sub-range of a pre-registered block
for tile_m in hl.tile(m, block_size=m_block):
    for sub_m in hl.tile(tile_m.begin, tile_m.end):
        ...
```

**Tile attributes:**

```python
tile.index        # 1-D int tensor of absolute offsets (e.g. [64, 65, ..., 127])
tile.begin        # scalar int — start of this tile
tile.end          # scalar int — end of this tile (may be < begin + block_size on last tile)
tile.block_size   # scalar int — nominal tile size (from config/autotune)
tile.id           # scalar int — tile index (= begin // block_size)
tile.count        # scalar int — number of tiles along this dimension
```

### `hl.zeros()` — accumulator pattern

Use inside a tile loop to create a **register-level** accumulator:

```python
for tile_m, tile_n in hl.tile([m, n]):
    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)  # fp32 accumulator
    for tile_k in hl.tile(k):
        acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
    out[tile_m, tile_n] = acc.to(out.dtype)
```

`hl.zeros` shape argument accepts tile indices; they are automatically converted to block sizes.

### `hl.full()` — fill with arbitrary value

```python
m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
```

### `hl.grid()` — scalar per-element iteration

`hl.grid` gives **scalar** integer indices rather than Tile objects. Use it when you need one thread per element:

```python
for i in hl.grid(n):
    out[i] = x[i] * 2.0          # scalar access

for i, j in hl.grid([m, n]):
    out[i, j] = x[i, j] + y[i, j]
```

### `hl.specialize()` — compile-time constants

Promote a dynamic shape to a compile-time constant so Triton can unroll loops and optimize:

```python
head_dim = hl.specialize(q_in.size(-1))   # bakes head_dim into kernel
```

### `hl.register_block_size()` — shared block size

Register a block size that multiple `hl.tile` calls should share:

```python
block_size_m = hl.register_block_size(m)
block_size_n = hl.register_block_size(n)
for tile_m in hl.tile(m, block_size=block_size_m):
    for tile_n in hl.tile(n, block_size=block_size_n):
        ...
```

### `hl.load()` / `hl.store()` — explicit indexed memory ops

For gather/scatter patterns or when you need an extra mask:

```python
# Gather: load arbitrary indices from a flat tensor
values = hl.load(flat_tensor, [index_tensor])

# Store with extra mask beyond tile bounds
hl.store(output, [row_idx, col_idx], value, extra_mask=some_bool_tensor)
```

### `hl.constexpr` — kernel parameter specialization

Mark a parameter to create one kernel per distinct value:

```python
@helion.kernel
def norm(x: torch.Tensor, compute_bias: hl.constexpr = True) -> torch.Tensor:
    if compute_bias:
        ...
```

### Atomic operations

```python
hl.atomic_add(tensor, [indices], values)
hl.atomic_max(tensor, [indices], values)
hl.atomic_min(tensor, [indices], values)
hl.atomic_and(tensor, [indices], values)
hl.atomic_or(tensor, [indices], values)
hl.atomic_xor(tensor, [indices], values)
hl.atomic_xchg(tensor, [indices], values)
hl.atomic_cas(tensor, [indices], cmp, val)
```

### Scan / Reduction ops

```python
hl.cumsum(tensor, dim=0)
hl.cumprod(tensor, dim=0)
hl.associative_scan(combine_fn, inputs, dim=0)
hl.reduce(combine_fn, input, dim=0)
```

### Other hl.* functions

```python
hl.dot(a, b)                   # low-level matmul (wraps tl.dot)
hl.dot_scaled(a, a_scale, b, b_scale)  # FP8 scaled matmul
hl.arange(start, end)          # arange on the kernel device
hl.rand(shape, ...)            # random float [0,1)
hl.randint(shape, ...)         # random integers
hl.join(a, b, dim=0)           # join two tensors along dim
hl.split(tensor, dim=0)        # split tensor along dim
hl.subscript(tensor, indices)  # advanced indexing helper
hl.static_range(n)             # compile-time unrolled loop
hl.jagged_tile(...)            # tiling for ragged/jagged tensors
hl.barrier()                   # explicit barrier (persistent kernels)
hl.inline_asm_elementwise(...) # inline PTX/AMDGCN
hl.inline_triton(...)          # embed raw Triton code
hl.triton_kernel(...)          # call existing Triton kernel
hl.device_print(fmt, *args)    # GPU-side printf
hl.StackTensor / hl.stacktensor_like(...)   # tensor on the stack
```

---

## 3. Patterns — Complete Code Examples

### 3.1 Vector Operation (Elementwise Add)

The simplest possible kernel: element-wise addition with broadcasting.

```python
import torch
import helion
import helion.language as hl


@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition with broadcasting."""
    # Align shapes according to PyTorch broadcasting rules
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    # hl.tile receives the full shape; Helion picks 1D or 2D tiling
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


# Usage
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
result = add(x, y)
```

### 3.2 Matrix Multiplication

Classic matmul with an optional epilogue (fused bias add, ReLU, etc.).

```python
import torch
import helion
import helion.language as hl


@helion.kernel(
    static_shapes=True,                  # static shapes give a speedup for matmul
    autotune_config_overrides={          # tl.dot is already pipelined
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul(
    x: torch.Tensor,
    y: torch.Tensor,
    epilogue=lambda acc, tile: acc,      # identity by default
) -> torch.Tensor:
    """Matrix multiplication x @ y, with an optional epilogue on the accumulator."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))

    return out


# Plain matmul
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = matmul(a, b)

# Matmul + fused bias + ReLU
bias = torch.randn(1024, device="cuda", dtype=torch.float16)
c = matmul(a, b, lambda acc, tile: torch.relu(acc + bias[tile[1]]))
```

### 3.3 Softmax

Three variants: simple wrapper, decomposed, and numerically stable two-pass.

```python
import torch
import helion
import helion.language as hl


# --- Simple wrapper (delegates to PyTorch's softmax) ---
@helion.kernel()
def softmax_simple(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


# --- Numerically stable two-pass (online softmax) ---
@helion.kernel()
def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
    """
    Online softmax: accumulate running (max, sum) then normalize.
    Keeps fully in registers — no separate passes over DRAM.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)

    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)

        # Pass 1: accumulate running max and sum
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(values - mi_next[:, None]).sum(dim=1)
            mi = mi_next

        # Pass 2: normalize
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]

    return out


# Usage
x = torch.randn(4096, 2560, device="cuda", dtype=torch.float16)
out = softmax_two_pass(x)
```

### 3.4 Attention Kernel (Flash-Attention style)

Full scaled dot-product attention with online softmax, supporting batched multi-head inputs.

```python
import math
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def attention(
    q_in: torch.Tensor,   # [..., seq_len_q, head_dim]
    k_in: torch.Tensor,   # [..., seq_len_k, head_dim]
    v_in: torch.Tensor,   # [..., seq_len_k, head_dim]
) -> torch.Tensor:
    """
    Computes: Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    Uses base-2 log trick for numerical stability.
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))  # bake head_dim into kernel

    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)

    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1 / log(2) for exp2 trick

    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i  = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i  = torch.full_like(m_i, 1.0)
        acc  = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)

        q = q_view[tile_b, tile_m, :]

        for tile_n in hl.tile(v_view.size(1)):
            k   = k_view[tile_b, :, tile_n]
            qk  = torch.bmm(q, k)

            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk   = qk * qk_scale - m_ij[:, :, None]
            p    = torch.exp2(qk)
            l_ij = torch.sum(p, -1)

            alpha = torch.exp2(m_i - m_ij)
            l_i   = l_i * alpha + l_ij
            acc   = acc * alpha[:, :, None]

            v   = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(v.dtype), v)
            m_i = m_ij

        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)

    return out.view(q_in.size())


# Usage
z, h, n_ctx, head_dim = 2, 32, 1024, 64
q = torch.randn(z, h, n_ctx, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(z, h, n_ctx, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(z, h, n_ctx, head_dim, device="cuda", dtype=torch.float16)
out = attention(q, k, v)
```

### 3.5 Fused Operations (SwiGLU)

Fuse two operations (SiLU activation + elementwise multiply) into a single kernel to avoid multiple DRAM passes.

```python
import torch
import helion
import helion.language as hl


@helion.kernel()
def swiglu_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU: SiLU(a) * b, where SiLU(x) = x * sigmoid(x).
    Fuses activation + multiply into one GPU kernel — no intermediate tensor.
    """
    assert a.shape == b.shape
    out = torch.empty_like(a, dtype=torch.promote_types(a.dtype, b.dtype))

    # Flatten for 1D tiling — works for any input shape
    total = a.numel()
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    out_flat = out.view(-1)

    for tile_idx in hl.tile(total):
        a_vals  = a_flat[tile_idx].to(torch.float32)
        b_vals  = b_flat[tile_idx]
        silu_a  = a_vals * torch.sigmoid(a_vals)
        out_flat[tile_idx] = silu_a.to(b_vals.dtype) * b_vals

    return out


# Usage — typical in LLaMA-style MLP
gate = torch.randn(4, 8192, 4096, device="cuda", dtype=torch.bfloat16)
up   = torch.randn(4, 8192, 4096, device="cuda", dtype=torch.bfloat16)
result = swiglu_fwd(gate, up)
```

**Epilogue pattern for fused matmul + op:**

```python
@helion.kernel(static_shapes=True)
def matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matmul with fused ReLU — no separate DRAM pass for activation."""
    m, k = x.size()
    _, n  = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu(acc).to(out.dtype)
    return out
```

### 3.6 Reduction Pattern (Row Sum / Layer Norm)

Reductions operate over a non-tiled dimension within a tile loop.

```python
import torch
import helion
import helion.language as hl


@helion.kernel()
def sum_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum each row of a 2D tensor."""
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)   # full row loaded, summed in registers
    return out


@helion.kernel()
def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """1D layer normalization fused into a single kernel."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        # Mean
        mean_val = torch.sum(acc, dim=-1) / n
        # Variance
        centered = acc - mean_val[:, None]
        var_val  = torch.sum(centered * centered, dim=-1) / n
        rstd_val = torch.rsqrt(var_val + eps)
        # Normalize + scale
        normalized = centered * rstd_val[:, None]
        result = normalized * weight[:].to(torch.float32)
        if bias is not None:
            result = result + bias[:].to(torch.float32)
        out[tile_m, :] = result.to(x.dtype)

    return out


# Usage
x      = torch.randn(4096, 10240, device="cuda", dtype=torch.float16)
weight = torch.ones(10240, device="cuda", dtype=torch.float16)
bias   = torch.zeros(10240, device="cuda", dtype=torch.float16)
out    = layer_norm_fwd(x, weight, bias)
```

---

## 4. Autotuning

### How it works

When a Helion kernel runs for the first time without a pinned config, it launches a **Differential Evolution Search**:

1. It generates an initial population of ~40 random configs (covering block sizes, loop orders, indexing strategies, warp counts, pipeline stages, etc.).
2. Over ~20 generations it evaluates ~1500 configs, keeping the fastest.
3. Results are cached locally (default: `~/.cache/helion/`).

Typical output:

```
[0s] Starting DifferentialEvolutionSearch population=40, generations=20
[20s] Initial population: min=0.027 mid=0.158 max=1.239 best=Config(block_sizes=[64,32,64], ...)
[586s] Autotuning complete in 586s after searching 1520 configs.
One can hardcode the best config and skip autotuning with:
    @helion.kernel(config=helion.Config(block_sizes=[64, 64, 64], ...))
```

### Controlling autotuning effort

```python
# Skip autotuning — uses slow default config, OK for development
@helion.kernel(autotune_effort="none")
# or: HELION_AUTOTUNE_EFFORT=none python script.py

# Quick search (~1–2 min, good enough for most ops)
@helion.kernel(autotune_effort="quick")

# Full search (~10 min, production quality)
@helion.kernel(autotune_effort="full")   # default
```

### Pinning a config (skip autotuning on re-runs)

After autotuning, copy the printed config into the decorator:

```python
@helion.kernel(config=helion.Config(
    block_sizes=[64, 64, 64],
    loop_orders=[[0, 1]],
    l2_groupings=[4],
    range_unroll_factors=[0, 1],
    range_warp_specializes=[None, False],
    range_num_stages=[0, 3],
    range_multi_buffers=[None, False],
    range_flattens=[None, None],
    num_warps=8,
    num_stages=6,
    indexing="block_ptr",
    pid_type="flat",
))
def matmul(x, y):
    ...
```

### Multi-config lightweight search

Provide a list of candidate configs; Helion benchmarks them and picks the fastest:

```python
@helion.kernel(configs=[
    helion.Config(block_sizes=[64, 64, 64], num_warps=4, ...),
    helion.Config(block_sizes=[128, 64, 32], num_warps=8, ...),
])
def matmul(x, y):
    ...
```

### Saving and loading configs programmatically

```python
# Run autotuning and save the best config
best = matmul.autotune((a, b), force=True)
best.save("matmul_best.json")

# Load and reuse
import helion
cfg = helion.Config.load("matmul_best.json")

@helion.kernel(config=cfg)
def matmul(x, y):
    ...
```

### Inspecting generated Triton code

```bash
HELION_PRINT_OUTPUT_CODE=1 python my_script.py 2>kernel.py
```

Or programmatically:

```python
# Get Triton source as a string
bound = my_kernel.bind((x, y))
triton_src = bound.to_triton_code(config)
print(triton_src)
```

Or in the decorator:

```python
@helion.kernel(print_output_code=True)
def my_kernel(x):
    ...
```

### Config parameters reference

| Parameter | Type | Effect |
|---|---|---|
| `block_sizes` | `list[int]` | Tile size for each `hl.tile` dimension |
| `loop_orders` | `list[list[int]]` | Permute iteration order of multi-dim tiles |
| `flatten_loops` | `list[bool]` | Flatten multi-dim tile into 1D |
| `range_unroll_factors` | `list[int]` | Loop unroll factor per dimension |
| `range_num_stages` | `list[int]` | Pipeline stages per loop (`tl.range` param) |
| `range_multi_buffers` | `list[bool\|None]` | Allow/disallow multi-buffering per loop |
| `range_flattens` | `list[bool\|None]` | Flatten the loop range |
| `range_warp_specializes` | `list[bool\|None]` | Warp specialization (Blackwell+) |
| `static_ranges` | `list[bool]` | Use `tl.static_range` (compile-time loop) |
| `reduction_loops` | `list[int\|None]` | Block size for looped reductions (`None` = persistent) |
| `l2_groupings` | `list[int]` | L2 swizzle grouping (`1` = disabled) |
| `indexing` | `str\|list[str]` | `"pointer"`, `"block_ptr"`, `"tensor_descriptor"` |
| `pid_type` | `str` | `"flat"`, `"xyz"`, `"persistent_blocked"`, `"persistent_interleaved"` |
| `num_warps` | `int` | Warps per CTA |
| `num_stages` | `int` | Triton pipeline stages |
| `load_eviction_policies` | `list[str]` | `""`, `"first"`, `"last"` per load site |

---

## 5. Integration with Training Loop

### Replacing a PyTorch op with a Helion kernel

```python
import torch
import torch.nn as nn
import helion
import helion.language as hl


# ── 1. Define the Helion kernel ────────────────────────────────────────────
@helion.kernel(autotune_effort="none")   # use "full" in production
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    m, n = x.size()
    out     = torch.empty_like(x)
    inv_rms = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        x_fp32    = x[tile_m, :].to(torch.float32)
        rms_val   = torch.rsqrt(x_fp32.pow(2).mean(-1) + eps)
        out[tile_m, :] = (x_fp32 * rms_val[:, None] * weight[:].to(torch.float32)).to(x.dtype)
    return out


# ── 2. Wrap with autograd so gradients flow ────────────────────────────────
class RMSNormHelion(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


# ── 3. Drop-in replacement in a model ─────────────────────────────────────
class MyModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        # Before:  self.norm = nn.RMSNorm(dim)
        self.norm = RMSNormHelion(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# ── 4. Training loop (unchanged) ───────────────────────────────────────────
model  = MyModel(1024).cuda().bfloat16()
opt    = torch.optim.Adam(model.parameters(), lr=1e-4)
data   = torch.randn(32, 512, 1024, device="cuda", dtype=torch.bfloat16)
labels = torch.randint(0, 1024, (32 * 512,), device="cuda")

for step, x in enumerate(data.unbind(0)):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x).view(-1, 1024), labels[:512])
    loss.backward()
    opt.step()
    if step % 10 == 0:
        print(f"step {step}, loss {loss.item():.4f}")
```

### Benchmarking: before/after comparison

```python
import torch
import time


def benchmark(fn, args, warmup=10, iters=100, desc=""):
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / iters * 1000
    print(f"{desc:40s}: {elapsed_ms:.3f} ms")
    return elapsed_ms


# Setup
x      = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

# Warm up the Helion kernel (triggers autotuning if no config cached)
_ = rms_norm(x, weight)

# Benchmark
t_torch  = benchmark(torch.nn.functional.rms_norm, (x, (4096,), weight), desc="torch.rms_norm")
t_helion = benchmark(rms_norm, (x, weight), desc="helion rms_norm")
print(f"Speedup: {t_torch / t_helion:.2f}x")
```

---

## 6. Common Pitfalls and Debugging

### 6.1 Calling `hl.*` outside a kernel

```
helion.exc.NotInsideKernel: Functions found in helion.language.* must be called from inside a kernel.
Did you forget the @helion.kernel decorator?
```

**Fix:** Add `@helion.kernel` to the function, or only call `hl.*` functions inside a decorated function.

### 6.2 Using `*args`, `**kwargs`, or keyword-only arguments

```
TypeError: Kernel(my_fn) cannot have *args, **kwargs, or keyword-only arguments
```

**Fix:** Use positional-or-keyword parameters only; use `hl.constexpr` for optional specializations.

### 6.3 Closures

```
helion.exc.ClosuresNotSupported: A closure ('bias') was found in the kernel. Closures are not supported.
```

**Fix:** Pass captured variables as explicit kernel arguments. Alternatively, closures are allowed in epilogue functions passed as arguments — Helion lifts them automatically.

### 6.4 Untraceable code (`make_fx` requirement)

```
torch._dynamo.exc.Unsupported: ...
```

Any code **inside** a tile loop must be traceable by `torch.fx.experimental.proxy_tensor.make_fx`. Things that break tracing:
- Python data structures mutated conditionally on tensor values
- Non-PyTorch third-party ops without a `torch.library` registration
- `print()` inside loops without `TRITON_INTERPRET=1`

**Fix:** Move untraceable code outside the loops (to the host section), or use `hl.device_print` for GPU-side printing.

### 6.5 Nested device loops with the same block size

```
helion.exc.NestedDeviceLoopsConflict: Nested device loops must have distinct block sizes.
```

**Fix:** Use `hl.register_block_size` to assign distinct block sizes:

```python
bs_m = hl.register_block_size(m)
bs_n = hl.register_block_size(n)
for tile_m in hl.tile(m, block_size=bs_m):
    for tile_n in hl.tile(n, block_size=bs_n):
        ...
```

### 6.6 `for...else` in device loops

```
helion.exc.DeviceLoopElseBlock: for...else block is not allowed in a hl.tile() device loop.
```

**Fix:** Remove `else` clauses from tile loops.

### 6.7 Host tensor used directly inside a tile loop

```
helion.exc.HostTensorDirectUsage: ...
```

**Fix:** Pass the tensor as a kernel argument so Helion can manage its lifecycle, or access it before the loop and pass the result in.

### 6.8 Barrier without persistent kernel

```
helion.exc.BarrierRequiresPersistent: ...
```

**Fix:** Use `pid_type="persistent_blocked"` or `"persistent_interleaved"` in the config.

### 6.9 Autotuning with `autotune_effort="none"` gives wrong performance

The default config is deliberately conservative and **not** production-quality. Always use `"full"` (or provide a hand-tuned config) before benchmarking.

### 6.10 Unsupported operations inside kernels

Operations that do **not** work inside `hl.tile` loops:
- Dynamic control flow that depends on tensor values (use `torch.where` instead of `if tensor_val:`)
- `torch.nn.Module` forward passes that contain non-traceable code
- `random.random()` — use `hl.rand()` / `hl.randint()` instead
- In-place ops on host tensors (allocate inside the kernel with `hl.zeros`)
- Data-dependent output shapes (e.g. `torch.nonzero`) — not supported

### 6.11 Printing / debugging inside kernels

```python
# CPU interpreter — allows Python print() and breakpoint() inside tile loops
TRITON_INTERPRET=1 python my_script.py

# Helion eager mode — runs as plain PyTorch, no GPU compilation
HELION_INTERPRET=1 python my_script.py
```

### 6.12 Repro script for bug reports

```bash
HELION_PRINT_REPRO=1 python my_script.py 2>repro.py
python repro.py   # self-contained reproduction
```

---

## 7. Links for Further Reading

| Resource | Description |
|---|---|
| **GitHub repo** — https://github.com/pytorch/helion | Source code, issues, CI, examples directory |
| **Official docs** — https://helionlang.com | API reference, tutorials, design documentation |
| **PyTorch blog post** — https://pytorch.org/blog/helion/ | High-level design rationale and motivation |
| **Colab notebook** — https://colab.research.google.com/github/pytorch/helion/blob/main/notebooks/softmax.ipynb | Interactive introduction: run a softmax kernel in-browser; no local setup needed |
| **AMD DevCloud notebook** — https://amd-ai-academy.com/github/pytorch/helion/blob/main/notebooks/softmax.ipynb | Same softmax tutorial but on AMD GPUs |
| **PyTorch Conference talk** — https://youtu.be/BW-Ht-5IxgM | Video walkthrough of Helion's design and demo by the core team |
| **DeepWiki** — https://deepwiki.com/pytorch-labs/helion | Auto-generated codebase map: modules, call graphs, data-flow |
| **Examples directory** — https://github.com/pytorch/helion/tree/main/examples | 40+ complete runnable kernels: matmul, attention, softmax, layer_norm, RMS norm, SwiGLU, FP8, grouped GEMM, Mamba, etc. |
| **Triton docs** — https://triton-lang.org | What Helion compiles to; useful for understanding `indexing`, `num_stages`, and `num_warps` |
| **GPU MODE Discord** `#helion` channel | Community Q&A, direct access to developers |
