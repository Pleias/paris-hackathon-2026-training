"""
Weight mapping unit tests for Megatron-Core -> Llama checkpoint conversion.

Standalone: does NOT require Megatron-Core or GPUs installed.
Run with: pytest ai_slope/test_weight_mapping.py -v
"""

import os
import sys
import importlib
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: GQA QKV split logic (mirrors what consolidate_and_remap() does)
# ---------------------------------------------------------------------------

def split_gqa_qkv(fused: torch.Tensor, num_groups: int, q_per_group: int, kv_per_group: int, head_dim: int):
    """
    Split Megatron's interleaved GQA QKV weight into separate Q, K, V tensors.

    Megatron stores the fused QKV as shape [num_groups * (q_per_group + kv_per_group) * head_dim, hidden],
    with the interleaved layout: for each group, q_per_group Q heads, then 1 K head, then 1 V head.

    Steps:
      1. Reshape to [num_groups, q_per_group + kv_per_group, head_dim, hidden]
      2. Slice: Q=[:, 0:q_per_group], K=[:, q_per_group:q_per_group+1], V=[:, q_per_group+1:]
      3. Reshape each slice back to 2D
    """
    hidden = fused.shape[1]
    heads_per_group = q_per_group + kv_per_group  # kv_per_group = k(1) + v(1) = 2
    # Reshape: [num_groups, heads_per_group, head_dim, hidden]
    reshaped = fused.reshape(num_groups, heads_per_group, head_dim, hidden)

    q_slice = reshaped[:, 0:q_per_group, :, :]          # [num_groups, q_per_group, head_dim, hidden]
    k_slice = reshaped[:, q_per_group:q_per_group+1, :, :]  # [num_groups, 1, head_dim, hidden]
    v_slice = reshaped[:, q_per_group+1:, :, :]         # [num_groups, 1, head_dim, hidden]

    wq = q_slice.reshape(num_groups * q_per_group * head_dim, hidden)
    wk = k_slice.reshape(num_groups * 1 * head_dim, hidden)
    wv = v_slice.reshape(num_groups * 1 * head_dim, hidden)

    return wq, wk, wv


def split_swiglu(fused: torch.Tensor):
    """Split fused gate+up (SwiGLU) weight along dim=0."""
    assert fused.shape[0] % 2 == 0, "SwiGLU fused weight dim0 must be even"
    half = fused.shape[0] // 2
    return fused[:half], fused[half:]


# ---------------------------------------------------------------------------
# Test 1: GQA QKV split
# ---------------------------------------------------------------------------

class TestGQAQKVSplit:
    # Llama-7B GQA config
    NUM_GROUPS = 8       # num_kv_heads
    Q_PER_GROUP = 4      # num_heads / num_kv_heads = 32 / 8
    KV_PER_GROUP = 2     # K=1 + V=1
    HEAD_DIM = 128
    HIDDEN = 4096

    def _make_fused(self, fill_value=None):
        rows = self.NUM_GROUPS * (self.Q_PER_GROUP + self.KV_PER_GROUP) * self.HEAD_DIM
        # rows = 8 * 6 * 128 = 6144
        shape = (rows, self.HIDDEN)
        if fill_value is not None:
            return torch.full(shape, fill_value, dtype=torch.float32)
        return torch.randn(shape, dtype=torch.float32)

    def test_fused_shape(self):
        """Fused tensor has shape [6144, 4096]."""
        fused = self._make_fused()
        assert fused.shape == (6144, self.HIDDEN)

    def test_split_output_shapes(self):
        """After split: Q=[4096,4096], K=[1024,4096], V=[1024,4096]."""
        fused = self._make_fused()
        wq, wk, wv = split_gqa_qkv(
            fused, self.NUM_GROUPS, self.Q_PER_GROUP, self.KV_PER_GROUP, self.HEAD_DIM
        )
        assert wq.shape == (4096, 4096), f"wq shape {wq.shape}"
        assert wk.shape == (1024, 4096), f"wk shape {wk.shape}"
        assert wv.shape == (1024, 4096), f"wv shape {wv.shape}"

    def test_split_roundtrip_values(self):
        """Values from split match the original fused tensor (round-trip)."""
        fused = self._make_fused()
        wq, wk, wv = split_gqa_qkv(
            fused, self.NUM_GROUPS, self.Q_PER_GROUP, self.KV_PER_GROUP, self.HEAD_DIM
        )
        # Reconstruct fused by reversing the reshape+slice
        wq_r = wq.reshape(self.NUM_GROUPS, self.Q_PER_GROUP, self.HEAD_DIM, self.HIDDEN)
        wk_r = wk.reshape(self.NUM_GROUPS, 1, self.HEAD_DIM, self.HIDDEN)
        wv_r = wv.reshape(self.NUM_GROUPS, 1, self.HEAD_DIM, self.HIDDEN)
        reconstructed = torch.cat([wq_r, wk_r, wv_r], dim=1)
        reconstructed = reconstructed.reshape(6144, self.HIDDEN)
        assert torch.allclose(fused, reconstructed), "Round-trip reconstruction mismatch"

    def test_split_values_distinct_groups(self):
        """Each group's Q/K/V values are correctly separated (no cross-contamination)."""
        # Build fused with known per-group values
        fused = torch.zeros(6144, self.HIDDEN)
        reshaped = fused.reshape(self.NUM_GROUPS, 6, self.HEAD_DIM, self.HIDDEN)
        for g in range(self.NUM_GROUPS):
            reshaped[g, :4, :, :] = float(g + 1)       # Q slots for group g
            reshaped[g, 4:5, :, :] = float(g + 100)    # K slot
            reshaped[g, 5:6, :, :] = float(g + 200)    # V slot
        fused = reshaped.reshape(6144, self.HIDDEN)

        wq, wk, wv = split_gqa_qkv(fused, self.NUM_GROUPS, self.Q_PER_GROUP, self.KV_PER_GROUP, self.HEAD_DIM)

        wq_r = wq.reshape(self.NUM_GROUPS, self.Q_PER_GROUP, self.HEAD_DIM, self.HIDDEN)
        wk_r = wk.reshape(self.NUM_GROUPS, 1, self.HEAD_DIM, self.HIDDEN)
        wv_r = wv.reshape(self.NUM_GROUPS, 1, self.HEAD_DIM, self.HIDDEN)

        for g in range(self.NUM_GROUPS):
            assert (wq_r[g] == float(g + 1)).all(), f"Q group {g} value wrong"
            assert (wk_r[g] == float(g + 100)).all(), f"K group {g} value wrong"
            assert (wv_r[g] == float(g + 200)).all(), f"V group {g} value wrong"


# ---------------------------------------------------------------------------
# Test 2: SwiGLU split
# ---------------------------------------------------------------------------

class TestSwiGLUSplit:
    HIDDEN = 4096
    FFN = 11008

    def test_split_shapes(self):
        """SwiGLU fused [22016, 4096] -> w1 [11008, 4096] + w3 [11008, 4096]."""
        fused = torch.randn(self.FFN * 2, self.HIDDEN)
        w1, w3 = split_swiglu(fused)
        assert w1.shape == (self.FFN, self.HIDDEN), f"w1 shape {w1.shape}"
        assert w3.shape == (self.FFN, self.HIDDEN), f"w3 shape {w3.shape}"

    def test_split_values(self):
        """w1 = first half, w3 = second half."""
        fused = torch.randn(self.FFN * 2, self.HIDDEN)
        w1, w3 = split_swiglu(fused)
        assert torch.equal(w1, fused[:self.FFN])
        assert torch.equal(w3, fused[self.FFN:])

    def test_split_odd_raises(self):
        """Odd dim0 should raise."""
        with pytest.raises(AssertionError):
            split_swiglu(torch.randn(999, self.HIDDEN))


# ---------------------------------------------------------------------------
# Test 3: TP shard concatenation
# ---------------------------------------------------------------------------

class TestTPShardConcat:
    HIDDEN = 4096
    TP = 4

    def test_qkv_column_parallel_concat(self):
        """QKV: column-parallel (dim=0) shards concat back to original."""
        original = torch.randn(6144, self.HIDDEN)
        shards = torch.chunk(original, self.TP, dim=0)
        assert len(shards) == self.TP
        assert shards[0].shape == (6144 // self.TP, self.HIDDEN)
        reconstructed = torch.cat(shards, dim=0)
        assert torch.equal(reconstructed, original)

    def test_fc1_column_parallel_concat(self):
        """FC1 (SwiGLU fused): column-parallel (dim=0) shards concat back to original."""
        original = torch.randn(22016, self.HIDDEN)
        shards = torch.chunk(original, self.TP, dim=0)
        assert shards[0].shape == (22016 // self.TP, self.HIDDEN)
        reconstructed = torch.cat(shards, dim=0)
        assert torch.equal(reconstructed, original)

    def test_proj_row_parallel_concat(self):
        """Attention output proj: row-parallel (dim=1) shards concat back to original."""
        original = torch.randn(self.HIDDEN, self.HIDDEN)
        shards = torch.chunk(original, self.TP, dim=1)
        assert shards[0].shape == (self.HIDDEN, self.HIDDEN // self.TP)
        reconstructed = torch.cat(shards, dim=1)
        assert torch.equal(reconstructed, original)

    def test_fc2_row_parallel_concat(self):
        """FC2 (down proj): row-parallel (dim=1) shards concat back to original."""
        original = torch.randn(self.HIDDEN, 11008)
        shards = torch.chunk(original, self.TP, dim=1)
        assert shards[0].shape == (self.HIDDEN, 11008 // self.TP)
        reconstructed = torch.cat(shards, dim=1)
        assert torch.equal(reconstructed, original)

    def test_shard_values_are_subsets(self):
        """Each shard contains exactly the rows/cols of the original."""
        original = torch.randn(6144, self.HIDDEN)
        shards = torch.chunk(original, self.TP, dim=0)
        chunk_size = 6144 // self.TP
        for i, shard in enumerate(shards):
            expected = original[i * chunk_size:(i + 1) * chunk_size]
            assert torch.equal(shard, expected), f"Shard {i} mismatch"


# ---------------------------------------------------------------------------
# Helper: remap pipeline (Megatron names -> Llama names)
# ---------------------------------------------------------------------------

def consolidate_and_remap(tp_shards_per_layer, num_layers,
                          num_kv_heads=None, num_heads=None, head_dim=None):
    """
    Simulate the full remap pipeline for a single-layer or multi-layer fake checkpoint.

    tp_shards_per_layer: list of dicts, each dict is one TP rank's state_dict for all layers.
    num_kv_heads / num_heads / head_dim: if not provided, inferred from the fused QKV weight shape
    using the GQA 4:1 ratio defaults (num_groups=8, q_per_group=4, head_dim=128, hidden=4096).
    Returns a Llama-convention state_dict.
    """
    state_dict = {}

    # Embeddings (replicated across TP ranks — just take rank 0)
    state_dict["tok_embeddings.weight"] = tp_shards_per_layer[0]["embedding.word_embeddings.weight"]
    state_dict["norm.weight"] = tp_shards_per_layer[0]["decoder.final_layernorm.weight"]
    # output_layer is column-parallel (dim=0)
    state_dict["output.weight"] = torch.cat(
        [s["output_layer.weight"] for s in tp_shards_per_layer], dim=0
    )

    # Infer GQA config from the fused QKV shape if not provided
    # After TP concat, QKV shape is [num_groups*(q_per_group+2)*head_dim, hidden]
    # We need num_heads, num_kv_heads, head_dim to split correctly.
    # If caller provides them, use those; otherwise fall back to 7B defaults.
    if num_heads is not None and num_kv_heads is not None and head_dim is not None:
        NUM_GROUPS = num_kv_heads
        Q_PER_GROUP = num_heads // num_kv_heads
        KV_PER_GROUP = 2
        HEAD_DIM = head_dim
    else:
        NUM_GROUPS = 8
        Q_PER_GROUP = 4
        KV_PER_GROUP = 2
        HEAD_DIM = 128

    for i in range(num_layers):
        prefix = f"decoder.layers.{i}"
        out_prefix = f"layers.{i}"

        # --- QKV: column-parallel, dim=0 ---
        fused_qkv = torch.cat(
            [s[f"{prefix}.self_attention.linear_qkv.weight"] for s in tp_shards_per_layer], dim=0
        )
        wq, wk, wv = split_gqa_qkv(fused_qkv, NUM_GROUPS, Q_PER_GROUP, KV_PER_GROUP, HEAD_DIM)
        state_dict[f"{out_prefix}.attention.wq.weight"] = wq
        state_dict[f"{out_prefix}.attention.wk.weight"] = wk
        state_dict[f"{out_prefix}.attention.wv.weight"] = wv

        # --- Attention output proj: row-parallel, dim=1 ---
        state_dict[f"{out_prefix}.attention.wo.weight"] = torch.cat(
            [s[f"{prefix}.self_attention.linear_proj.weight"] for s in tp_shards_per_layer], dim=1
        )

        # --- FC1 (SwiGLU gate+up): column-parallel, dim=0 ---
        fused_fc1 = torch.cat(
            [s[f"{prefix}.mlp.linear_fc1.weight"] for s in tp_shards_per_layer], dim=0
        )
        w1, w3 = split_swiglu(fused_fc1)
        state_dict[f"{out_prefix}.feed_forward.w1.weight"] = w1
        state_dict[f"{out_prefix}.feed_forward.w3.weight"] = w3

        # --- FC2 (down proj): row-parallel, dim=1 ---
        state_dict[f"{out_prefix}.feed_forward.w2.weight"] = torch.cat(
            [s[f"{prefix}.mlp.linear_fc2.weight"] for s in tp_shards_per_layer], dim=1
        )

        # --- Norms: replicated, take rank 0 ---
        state_dict[f"{out_prefix}.attention_norm.weight"] = (
            tp_shards_per_layer[0][f"{prefix}.self_attention.linear_qkv.layer_norm_weight"]
        )
        state_dict[f"{out_prefix}.ffn_norm.weight"] = (
            tp_shards_per_layer[0][f"{prefix}.mlp.linear_fc1.layer_norm_weight"]
        )

    return state_dict


def make_fake_megatron_shards(num_layers, hidden, num_heads, num_kv_heads, ffn, vocab_size, tp):
    """
    Build fake Megatron-style TP-sharded checkpoint (list of tp dicts, one per TP rank).

    Column-parallel weights are split along dim=0.
    Row-parallel weights are split along dim=1.
    """
    head_dim = hidden // num_heads
    num_groups = num_kv_heads
    q_per_group = num_heads // num_kv_heads

    # Full (unsharded) weights — we'll shard them
    full = {}
    full["embedding.word_embeddings.weight"] = torch.randn(vocab_size, hidden)
    full["decoder.final_layernorm.weight"] = torch.randn(hidden)
    full["output_layer.weight"] = torch.randn(vocab_size, hidden)  # col-parallel

    for i in range(num_layers):
        p = f"decoder.layers.{i}"
        # QKV fused: col-parallel, shape [num_groups*(q_per_group+2)*head_dim, hidden]
        qkv_rows = num_groups * (q_per_group + 2) * head_dim
        full[f"{p}.self_attention.linear_qkv.weight"] = torch.randn(qkv_rows, hidden)
        # proj: row-parallel, shape [hidden, hidden]
        full[f"{p}.self_attention.linear_proj.weight"] = torch.randn(hidden, hidden)
        # fc1 fused gate+up: col-parallel, shape [2*ffn, hidden]
        full[f"{p}.mlp.linear_fc1.weight"] = torch.randn(ffn * 2, hidden)
        # fc2 down: row-parallel, shape [hidden, ffn]
        full[f"{p}.mlp.linear_fc2.weight"] = torch.randn(hidden, ffn)
        # Norms: replicated
        full[f"{p}.self_attention.linear_qkv.layer_norm_weight"] = torch.randn(hidden)
        full[f"{p}.mlp.linear_fc1.layer_norm_weight"] = torch.randn(hidden)

    # Shard
    shards = [{} for _ in range(tp)]
    col_parallel_keys = (
        ["output_layer.weight"] +
        [f"decoder.layers.{i}.self_attention.linear_qkv.weight" for i in range(num_layers)] +
        [f"decoder.layers.{i}.mlp.linear_fc1.weight" for i in range(num_layers)]
    )
    row_parallel_keys = (
        [f"decoder.layers.{i}.self_attention.linear_proj.weight" for i in range(num_layers)] +
        [f"decoder.layers.{i}.mlp.linear_fc2.weight" for i in range(num_layers)]
    )
    replicated_keys = (
        ["embedding.word_embeddings.weight", "decoder.final_layernorm.weight"] +
        [f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight" for i in range(num_layers)] +
        [f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight" for i in range(num_layers)]
    )

    for key in col_parallel_keys:
        chunks = torch.chunk(full[key], tp, dim=0)
        for rank, shard_dict in enumerate(shards):
            shard_dict[key] = chunks[rank].clone()

    for key in row_parallel_keys:
        chunks = torch.chunk(full[key], tp, dim=1)
        for rank, shard_dict in enumerate(shards):
            shard_dict[key] = chunks[rank].clone()

    for key in replicated_keys:
        for shard_dict in shards:
            shard_dict[key] = full[key].clone()

    return shards, full


# ---------------------------------------------------------------------------
# Test 4a: Full remap pipeline (no model.py)
# ---------------------------------------------------------------------------

class TestFullRemapPipeline:
    """Full remap: concat TP shards -> split fused QKV/SwiGLU -> rename to Llama convention."""

    HIDDEN = 256
    LAYERS = 2
    NUM_HEADS = 8
    NUM_KV_HEADS = 2
    FFN = 512
    VOCAB = 512
    TP = 4

    @property
    def HEAD_DIM(self):
        return self.HIDDEN // self.NUM_HEADS

    def _run(self):
        shards, full = make_fake_megatron_shards(
            self.LAYERS, self.HIDDEN, self.NUM_HEADS, self.NUM_KV_HEADS, self.FFN, self.VOCAB, self.TP
        )
        remapped = consolidate_and_remap(
            shards, self.LAYERS,
            num_kv_heads=self.NUM_KV_HEADS,
            num_heads=self.NUM_HEADS,
            head_dim=self.HEAD_DIM,
        )
        return remapped, full

    def test_remapped_keys_present(self):
        """All expected Llama keys are present after remap."""
        remapped, _ = self._run()
        expected_keys = (
            ["tok_embeddings.weight", "norm.weight", "output.weight"] +
            [f"layers.{i}.attention.wq.weight" for i in range(self.LAYERS)] +
            [f"layers.{i}.attention.wk.weight" for i in range(self.LAYERS)] +
            [f"layers.{i}.attention.wv.weight" for i in range(self.LAYERS)] +
            [f"layers.{i}.attention.wo.weight" for i in range(self.LAYERS)] +
            [f"layers.{i}.feed_forward.w1.weight" for i in range(self.LAYERS)] +
            [f"layers.{i}.feed_forward.w2.weight" for i in range(self.LAYERS)] +
            [f"layers.{i}.feed_forward.w3.weight" for i in range(self.LAYERS)] +
            [f"layers.{i}.attention_norm.weight" for i in range(self.LAYERS)] +
            [f"layers.{i}.ffn_norm.weight" for i in range(self.LAYERS)]
        )
        for key in expected_keys:
            assert key in remapped, f"Missing key: {key}"

    def test_no_megatron_keys_remain(self):
        """No Megatron-style keys remain in remapped dict."""
        remapped, _ = self._run()
        for key in remapped:
            assert "decoder" not in key and "linear_qkv" not in key and "linear_fc1" not in key, (
                f"Megatron key leaked into remapped dict: {key}"
            )

    def test_wq_shape(self):
        remapped, _ = self._run()
        head_dim = self.HIDDEN // self.NUM_HEADS
        wq_rows = self.NUM_HEADS * head_dim  # = HIDDEN
        assert remapped["layers.0.attention.wq.weight"].shape == (wq_rows, self.HIDDEN)

    def test_wk_shape(self):
        remapped, _ = self._run()
        head_dim = self.HIDDEN // self.NUM_HEADS
        wk_rows = self.NUM_KV_HEADS * head_dim
        assert remapped["layers.0.attention.wk.weight"].shape == (wk_rows, self.HIDDEN)

    def test_wv_shape(self):
        remapped, _ = self._run()
        head_dim = self.HIDDEN // self.NUM_HEADS
        wv_rows = self.NUM_KV_HEADS * head_dim
        assert remapped["layers.0.attention.wv.weight"].shape == (wv_rows, self.HIDDEN)

    def test_wo_shape(self):
        remapped, _ = self._run()
        assert remapped["layers.0.attention.wo.weight"].shape == (self.HIDDEN, self.HIDDEN)

    def test_w1_w3_shapes(self):
        remapped, _ = self._run()
        assert remapped["layers.0.feed_forward.w1.weight"].shape == (self.FFN, self.HIDDEN)
        assert remapped["layers.0.feed_forward.w3.weight"].shape == (self.FFN, self.HIDDEN)

    def test_w2_shape(self):
        remapped, _ = self._run()
        assert remapped["layers.0.feed_forward.w2.weight"].shape == (self.HIDDEN, self.FFN)

    def test_concat_roundtrip_wo(self):
        """wo after concat matches original full wo (row-parallel concat is lossless)."""
        shards, full = make_fake_megatron_shards(
            self.LAYERS, self.HIDDEN, self.NUM_HEADS, self.NUM_KV_HEADS, self.FFN, self.VOCAB, self.TP
        )
        remapped = consolidate_and_remap(
            shards, self.LAYERS,
            num_kv_heads=self.NUM_KV_HEADS, num_heads=self.NUM_HEADS, head_dim=self.HEAD_DIM,
        )
        assert torch.allclose(remapped["layers.0.attention.wo.weight"],
                              full["decoder.layers.0.self_attention.linear_proj.weight"])

    def test_concat_roundtrip_w2(self):
        """w2 after concat matches original full w2."""
        shards, full = make_fake_megatron_shards(
            self.LAYERS, self.HIDDEN, self.NUM_HEADS, self.NUM_KV_HEADS, self.FFN, self.VOCAB, self.TP
        )
        remapped = consolidate_and_remap(
            shards, self.LAYERS,
            num_kv_heads=self.NUM_KV_HEADS, num_heads=self.NUM_HEADS, head_dim=self.HEAD_DIM,
        )
        assert torch.allclose(remapped["layers.0.feed_forward.w2.weight"],
                              full["decoder.layers.0.mlp.linear_fc2.weight"])

    def test_no_nans(self):
        """Remapped weights contain no NaN values."""
        remapped, _ = self._run()
        for key, val in remapped.items():
            assert not torch.isnan(val).any(), f"NaN found in {key}"


# ---------------------------------------------------------------------------
# Test 4b: Full round-trip with model.py (skipped if model.py not present)
# ---------------------------------------------------------------------------

MODEL_PY_PATH = os.path.join(os.path.dirname(__file__), "model.py")
MODEL_AVAILABLE = os.path.exists(MODEL_PY_PATH)

@pytest.mark.skipif(not MODEL_AVAILABLE, reason="model.py not yet available")
class TestFullRoundTripWithModel:
    """
    End-to-end: fake Megatron checkpoint -> remap -> load_state_dict -> forward pass.
    Only runs when ai_slope/model.py exists.
    """

    HIDDEN = 256
    LAYERS = 2
    NUM_HEADS = 4
    NUM_KV_HEADS = 2
    FFN = 512
    VOCAB = 512
    TP = 4
    SEQ_LEN = 16
    HEAD_DIM = HIDDEN // NUM_HEADS  # 64

    def _load_model_module(self):
        spec = importlib.util.spec_from_file_location("model", MODEL_PY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_forward_pass_shape_and_no_nan(self):
        """
        Full round-trip:
          1. Build fake Megatron TP-sharded checkpoint
          2. Remap to Llama state_dict
          3. load_state_dict into model
          4. Run forward pass
          5. Check output shape and no NaN
        """
        mod = self._load_model_module()

        config = {
            "hidden_size": self.HIDDEN,
            "num_layers": self.LAYERS,
            "num_attention_heads": self.NUM_HEADS,
            "num_key_value_heads": self.NUM_KV_HEADS,
            "intermediate_size": self.FFN,
            "vocab_size": self.VOCAB,
            "seq_len": self.SEQ_LEN,
        }
        model = mod.get_model(config)
        model.eval()

        # Build fake checkpoint
        shards, _ = make_fake_megatron_shards(
            self.LAYERS, self.HIDDEN, self.NUM_HEADS, self.NUM_KV_HEADS, self.FFN, self.VOCAB, self.TP
        )
        remapped = consolidate_and_remap(
            shards, self.LAYERS,
            num_kv_heads=self.NUM_KV_HEADS,
            num_heads=self.NUM_HEADS,
            head_dim=self.HEAD_DIM,
        )

        # load_state_dict
        result = model.load_state_dict(remapped, strict=True)
        assert not result.missing_keys, f"Missing keys: {result.missing_keys}"
        assert not result.unexpected_keys, f"Unexpected keys: {result.unexpected_keys}"

        # Forward pass
        batch_size = 2
        idx = torch.randint(0, self.VOCAB, (batch_size, self.SEQ_LEN))
        with torch.no_grad():
            logits, loss = model(idx)

        # Shape check: [batch, seq_len, vocab]
        assert logits.shape == (batch_size, self.SEQ_LEN, self.VOCAB), (
            f"Unexpected logits shape: {logits.shape}"
        )

        # No NaN
        assert not torch.isnan(logits).any(), "NaN in logits"
