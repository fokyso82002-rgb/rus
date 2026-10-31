"""Custom Triton GPU kernels for Qwen2.5-1.5B-Instruct QLoRA training on NVIDIA Tesla T4 (sm_75).

Implements 9 specialized @triton.jit kernels:
1. nf4_dequant_turing_kernel: Register-muxed NF4 + double-quantization dequantizer to coalesced fp16x2
   (supports single launch over concatenated [q|k|v] and [gate|up] with per-segment offsets).
2. rms_norm_residual_fwd_kernel: Fused residual addition + FP32 variance RMSNorm + FP16 weight scaling.
3. rms_norm_bwd_kernel: Fused RMSNorm backward + residual gradient accumulation.
4. qkv_lora_rope_fwd_kernel: Fused QKV bias/LoRA delta addition + NeoX RoPE rotary embedding + head layout.
5. qkv_lora_rope_bwd_kernel: Fused inverse NeoX RoPE rotation + head gradient packing for Q, K, V.
6. tiled_mlp_glu_fwd_kernel: 2D multi-tile SwiGLU forward over fused [gate|up] or separate projections.
7. tiled_mlp_glu_bwd_kernel: 2D multi-tile SwiGLU backward writing in-place to fused [d_gate|d_up] or separate buffers.
8. lm_head_piece_kernel: Vocabulary-chunked online logsumexp, Shannon entropy, argmax, target logit, and backward gradient.
9. lm_head_finish_kernel: Finalizes per-token NLL loss, exact Shannon entropy, and accuracy statistics.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# 1. Turing-optimized NF4 Double-Quantization Dequantizer (@triton.jit)
# -----------------------------------------------------------------------------

@triton.jit
def nf4_dequant_turing_kernel(
    packed_ptr,
    absmax_ptr,
    code2_ptr,
    absmax2_ptr,
    out_u32_ptr,
    offset_0,
    offset_1,
    offset_2,
    split_byte_1,
    split_byte_2,
    n_bytes,
    HALF_BLOCK: tl.constexpr,
    SUPER_BLOCK: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Dequantize packed uint8 NF4 weights (2 weights per byte) with double quantization
    using register-level constant binary-tree selection (no shared/global LUT traffic)
    and coalesced 32-bit (fp16x2) stores. Supports 1, 2 ([gate|up]), or 3 ([q|k|v]) fused
    segments in a single kernel launch with exact per-segment double-quantization offsets."""
    pid = tl.program_id(0)
    byte_idx = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    mask = byte_idx < n_bytes

    packed = tl.load(packed_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
    block_idx = byte_idx // HALF_BLOCK
    q_abs = tl.load(absmax_ptr + block_idx, mask=mask, other=0).to(tl.int32)
    code_val = tl.load(code2_ptr + q_abs, mask=mask, other=0.0).to(tl.float32)
    super_idx = block_idx // SUPER_BLOCK
    abs2_val = tl.load(absmax2_ptr + super_idx, mask=mask, other=0.0).to(tl.float32)

    off_val = tl.where(
        byte_idx < split_byte_1,
        offset_0,
        tl.where(byte_idx < split_byte_2, offset_1, offset_2),
    )
    scale = code_val * abs2_val + off_val

    # High nibble -> even element (2 * byte_idx), Low nibble -> odd element (2 * byte_idx + 1)
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F

    # Binary mux tree for high nibble (exact bitsandbytes NF4 codebook constants)
    hi_b0 = (hi & 1) != 0
    hi_b1 = (hi & 2) != 0
    hi_b2 = (hi & 4) != 0
    hi_b3 = (hi & 8) != 0

    hi_p00 = tl.where(hi_b0, -0.6961928009986877, -1.0)
    hi_p01 = tl.where(hi_b0, -0.39491748809814453, -0.5250730514526367)
    hi_p02 = tl.where(hi_b0, -0.18477343022823334, -0.28444138169288635)
    hi_p03 = tl.where(hi_b0, 0.0, -0.09105003625154495)
    hi_p04 = tl.where(hi_b0, 0.16093020141124725, 0.07958029955625534)
    hi_p05 = tl.where(hi_b0, 0.33791524171829224, 0.24611230194568634)
    hi_p06 = tl.where(hi_b0, 0.5626170039176941, 0.44070982933044434)
    hi_p07 = tl.where(hi_b0, 1.0, 0.7229568362236023)

    hi_q0 = tl.where(hi_b1, hi_p01, hi_p00)
    hi_q1 = tl.where(hi_b1, hi_p03, hi_p02)
    hi_q2 = tl.where(hi_b1, hi_p05, hi_p04)
    hi_q3 = tl.where(hi_b1, hi_p07, hi_p06)

    val_hi = tl.where(hi_b3, tl.where(hi_b2, hi_q3, hi_q2), tl.where(hi_b2, hi_q1, hi_q0)) * scale

    # Binary mux tree for low nibble
    lo_b0 = (lo & 1) != 0
    lo_b1 = (lo & 2) != 0
    lo_b2 = (lo & 4) != 0
    lo_b3 = (lo & 8) != 0

    lo_p00 = tl.where(lo_b0, -0.6961928009986877, -1.0)
    lo_p01 = tl.where(lo_b0, -0.39491748809814453, -0.5250730514526367)
    lo_p02 = tl.where(lo_b0, -0.18477343022823334, -0.28444138169288635)
    lo_p03 = tl.where(lo_b0, 0.0, -0.09105003625154495)
    lo_p04 = tl.where(lo_b0, 0.16093020141124725, 0.07958029955625534)
    lo_p05 = tl.where(lo_b0, 0.33791524171829224, 0.24611230194568634)
    lo_p06 = tl.where(lo_b0, 0.5626170039176941, 0.44070982933044434)
    lo_p07 = tl.where(lo_b0, 1.0, 0.7229568362236023)

    lo_q0 = tl.where(lo_b1, lo_p01, lo_p00)
    lo_q1 = tl.where(lo_b1, lo_p03, lo_p02)
    lo_q2 = tl.where(lo_b1, lo_p05, lo_p04)
    lo_q3 = tl.where(lo_b1, lo_p07, lo_p06)

    val_lo = tl.where(lo_b3, tl.where(lo_b2, lo_q3, lo_q2), tl.where(lo_b2, lo_q1, lo_q0)) * scale

    # Pack two adjacent fp16 values into one uint32 word (little-endian: hi at bits 0..15, lo at bits 16..31)
    u16_hi = val_hi.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    u16_lo = val_lo.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    packed_u32 = u16_hi | (u16_lo << 16)
    tl.store(out_u32_ptr + byte_idx, packed_u32, mask=mask)


# -----------------------------------------------------------------------------
# 2. Fused Residual + RMSNorm Forward & Backward Kernels (@triton.jit)
# -----------------------------------------------------------------------------

@triton.jit
def rms_norm_residual_fwd_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    sum_out_ptr,
    norm_out_ptr,
    rstd_ptr,
    stride_row,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused residual addition (h = x + res) and RMSNorm (y = (h * rstd(h)).to(dtype) * w).
    One Triton program handles one token row, so grid = (n_rows,) requires no row-count specialization."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < N_COLS
    row_offset = row * stride_row + cols

    x = tl.load(x_ptr + row_offset, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        res = tl.load(res_ptr + row_offset, mask=mask, other=0.0)
        h = (x + res).to(x.dtype)
        tl.store(sum_out_ptr + row_offset, h, mask=mask)
    else:
        h = x

    h_fp32 = h.to(tl.float32)
    var = tl.sum(tl.where(mask, h_fp32 * h_fp32, 0.0), axis=0) / N_COLS
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # Match Qwen2RMSNorm exact casting order: self.weight * hidden_states.to(input_dtype)
    normed = (h_fp32 * rstd).to(x.dtype)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(x.dtype)
    tl.store(norm_out_ptr + row_offset, out, mask=mask)


@triton.jit
def rms_norm_bwd_kernel(
    dy_ptr,
    h_ptr,
    w_ptr,
    rstd_ptr,
    dres_ptr,
    dx_ptr,
    stride_row,
    HAS_DRESIDUAL: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused RMSNorm backward pass with optional upstream residual gradient accumulation."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < N_COLS
    row_offset = row * stride_row + cols

    dy = tl.load(dy_ptr + row_offset, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(h_ptr + row_offset, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    x_hat = h * rstd
    wdy = w * dy
    c1 = tl.sum(tl.where(mask, x_hat * wdy, 0.0), axis=0) / N_COLS
    dx = (wdy - x_hat * c1) * rstd

    if HAS_DRESIDUAL:
        dres = tl.load(dres_ptr + row_offset, mask=mask, other=0.0).to(tl.float32)
        dx = dx + dres

    tl.store(dx_ptr + row_offset, dx.to(dx_ptr.dtype.element_ty), mask=mask)


# -----------------------------------------------------------------------------
# 3. Fused QKV + LoRA Delta + NeoX RoPE Forward & Backward Kernels (@triton.jit)
# -----------------------------------------------------------------------------

@triton.jit
def qkv_lora_rope_fwd_kernel(
    qkv_ptr,
    q_lora_ptr,
    v_lora_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    stride_qkv,
    stride_ql,
    stride_vl,
    stride_cos,
    scale_q,
    scale_v,
    HAS_LORA: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    """Fuses LoRA scaling/addition on Q and V with NeoX rotary position embeddings on Q and K.
    Grid: (n_tokens, H_Q + 2 * H_KV)."""
    tok = tl.program_id(0)
    head_id = tl.program_id(1)
    d_idx = tl.arange(0, HALF_DIM)
    head_dim = HALF_DIM * 2

    if head_id < H_Q:
        pos = tl.load(pos_ptr + tok)
        cos_lo = tl.load(cos_ptr + pos * stride_cos + d_idx).to(tl.float32)
        cos_hi = tl.load(cos_ptr + pos * stride_cos + HALF_DIM + d_idx).to(tl.float32)
        sin_lo = tl.load(sin_ptr + pos * stride_cos + d_idx).to(tl.float32)
        sin_hi = tl.load(sin_ptr + pos * stride_cos + HALF_DIM + d_idx).to(tl.float32)

        base_off = tok * stride_qkv + head_id * head_dim
        q_lo = tl.load(qkv_ptr + base_off + d_idx).to(tl.float32)
        q_hi = tl.load(qkv_ptr + base_off + HALF_DIM + d_idx).to(tl.float32)
        if HAS_LORA:
            ql_off = tok * stride_ql + head_id * head_dim
            q_lo += tl.load(q_lora_ptr + ql_off + d_idx).to(tl.float32) * scale_q
            q_hi += tl.load(q_lora_ptr + ql_off + HALF_DIM + d_idx).to(tl.float32) * scale_q

        out_lo = q_lo * cos_lo - q_hi * sin_lo
        out_hi = q_hi * cos_hi + q_lo * sin_hi
        out_off = (tok * H_Q + head_id) * head_dim
        tl.store(q_out_ptr + out_off + d_idx, out_lo.to(q_out_ptr.dtype.element_ty))
        tl.store(q_out_ptr + out_off + HALF_DIM + d_idx, out_hi.to(q_out_ptr.dtype.element_ty))
    elif head_id < H_Q + H_KV:
        kv_h = head_id - H_Q
        pos = tl.load(pos_ptr + tok)
        cos_lo = tl.load(cos_ptr + pos * stride_cos + d_idx).to(tl.float32)
        cos_hi = tl.load(cos_ptr + pos * stride_cos + HALF_DIM + d_idx).to(tl.float32)
        sin_lo = tl.load(sin_ptr + pos * stride_cos + d_idx).to(tl.float32)
        sin_hi = tl.load(sin_ptr + pos * stride_cos + HALF_DIM + d_idx).to(tl.float32)

        base_off = tok * stride_qkv + (H_Q + kv_h) * head_dim
        k_lo = tl.load(qkv_ptr + base_off + d_idx).to(tl.float32)
        k_hi = tl.load(qkv_ptr + base_off + HALF_DIM + d_idx).to(tl.float32)

        out_lo = k_lo * cos_lo - k_hi * sin_lo
        out_hi = k_hi * cos_hi + k_lo * sin_hi
        out_off = (tok * H_KV + kv_h) * head_dim
        tl.store(k_out_ptr + out_off + d_idx, out_lo.to(k_out_ptr.dtype.element_ty))
        tl.store(k_out_ptr + out_off + HALF_DIM + d_idx, out_hi.to(k_out_ptr.dtype.element_ty))
    else:
        kv_h = head_id - H_Q - H_KV
        base_off = tok * stride_qkv + (H_Q + H_KV + kv_h) * head_dim
        v_lo = tl.load(qkv_ptr + base_off + d_idx).to(tl.float32)
        v_hi = tl.load(qkv_ptr + base_off + HALF_DIM + d_idx).to(tl.float32)
        if HAS_LORA:
            vl_off = tok * stride_vl + kv_h * head_dim
            v_lo += tl.load(v_lora_ptr + vl_off + d_idx).to(tl.float32) * scale_v
            v_hi += tl.load(v_lora_ptr + vl_off + HALF_DIM + d_idx).to(tl.float32) * scale_v
        out_off = (tok * H_KV + kv_h) * head_dim
        tl.store(v_out_ptr + out_off + d_idx, v_lo.to(v_out_ptr.dtype.element_ty))
        tl.store(v_out_ptr + out_off + HALF_DIM + d_idx, v_hi.to(v_out_ptr.dtype.element_ty))


@triton.jit
def qkv_lora_rope_bwd_kernel(
    dq_rot_ptr,
    dk_rot_ptr,
    dv_out_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    dqkv_ptr,
    stride_dqkv,
    stride_cos,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    """Inverse NeoX RoPE rotation on dQ and dK and packing of [dQ | dK | dV] into a single
    contiguous gradient buffer for the fused QKV backward GEMM. Grid: (n_tokens, H_Q + 2 * H_KV)."""
    tok = tl.program_id(0)
    head_id = tl.program_id(1)
    d_idx = tl.arange(0, HALF_DIM)
    head_dim = HALF_DIM * 2

    if head_id < H_Q:
        pos = tl.load(pos_ptr + tok)
        cos_lo = tl.load(cos_ptr + pos * stride_cos + d_idx).to(tl.float32)
        cos_hi = tl.load(cos_ptr + pos * stride_cos + HALF_DIM + d_idx).to(tl.float32)
        sin_lo = tl.load(sin_ptr + pos * stride_cos + d_idx).to(tl.float32)
        sin_hi = tl.load(sin_ptr + pos * stride_cos + HALF_DIM + d_idx).to(tl.float32)

        in_off = (tok * H_Q + head_id) * head_dim
        dy_lo = tl.load(dq_rot_ptr + in_off + d_idx).to(tl.float32)
        dy_hi = tl.load(dq_rot_ptr + in_off + HALF_DIM + d_idx).to(tl.float32)

        dx_lo = dy_lo * cos_lo + dy_hi * sin_hi
        dx_hi = dy_hi * cos_hi - dy_lo * sin_lo
        out_off = tok * stride_dqkv + head_id * head_dim
        tl.store(dqkv_ptr + out_off + d_idx, dx_lo.to(dqkv_ptr.dtype.element_ty))
        tl.store(dqkv_ptr + out_off + HALF_DIM + d_idx, dx_hi.to(dqkv_ptr.dtype.element_ty))
    elif head_id < H_Q + H_KV:
        kv_h = head_id - H_Q
        pos = tl.load(pos_ptr + tok)
        cos_lo = tl.load(cos_ptr + pos * stride_cos + d_idx).to(tl.float32)
        cos_hi = tl.load(cos_ptr + pos * stride_cos + HALF_DIM + d_idx).to(tl.float32)
        sin_lo = tl.load(sin_ptr + pos * stride_cos + d_idx).to(tl.float32)
        sin_hi = tl.load(sin_ptr + pos * stride_cos + HALF_DIM + d_idx).to(tl.float32)

        in_off = (tok * H_KV + kv_h) * head_dim
        dy_lo = tl.load(dk_rot_ptr + in_off + d_idx).to(tl.float32)
        dy_hi = tl.load(dk_rot_ptr + in_off + HALF_DIM + d_idx).to(tl.float32)

        dx_lo = dy_lo * cos_lo + dy_hi * sin_hi
        dx_hi = dy_hi * cos_hi - dy_lo * sin_lo
        out_off = tok * stride_dqkv + (H_Q + kv_h) * head_dim
        tl.store(dqkv_ptr + out_off + d_idx, dx_lo.to(dqkv_ptr.dtype.element_ty))
        tl.store(dqkv_ptr + out_off + HALF_DIM + d_idx, dx_hi.to(dqkv_ptr.dtype.element_ty))
    else:
        kv_h = head_id - H_Q - H_KV
        in_off = (tok * H_KV + kv_h) * head_dim
        dv_lo = tl.load(dv_out_ptr + in_off + d_idx)
        dv_hi = tl.load(dv_out_ptr + in_off + HALF_DIM + d_idx)
        out_off = tok * stride_dqkv + (H_Q + H_KV + kv_h) * head_dim
        tl.store(dqkv_ptr + out_off + d_idx, dv_lo.to(dqkv_ptr.dtype.element_ty))
        tl.store(dqkv_ptr + out_off + HALF_DIM + d_idx, dv_hi.to(dqkv_ptr.dtype.element_ty))


# -----------------------------------------------------------------------------
# 4. 2D Multi-Tile MLP Gated Linear Unit Forward & Backward Kernels (@triton.jit)
# -----------------------------------------------------------------------------

@triton.jit
def tiled_mlp_glu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    stride_g_row,
    stride_u_row,
    stride_o_row,
    n_cols,
    FUSED_GU: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    """2D row-and-column multi-tile gated activation over fused [gate | up] or separate matrices.
    Each program processes TILES_PER_PROG column blocks along a token row to amortize address math."""
    row_id = tl.program_id(0)
    group_id = tl.program_id(1)
    g_base = gate_ptr + row_id * stride_g_row
    u_base = up_ptr + row_id * stride_u_row
    o_base = out_ptr + row_id * stride_o_row

    step = 0
    while step < TILES_PER_PROG:
        tile_id = group_id * TILES_PER_PROG + step
        col_idx = tile_id * BLOCK_COL + tl.arange(0, BLOCK_COL)
        col_mask = col_idx < n_cols

        g_val = tl.load(g_base + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        if FUSED_GU:
            u_val = tl.load(g_base + n_cols + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        else:
            u_val = tl.load(u_base + col_idx, mask=col_mask, other=0.0).to(tl.float32)

        inv_denom = 1.0 / (1.0 + tl.exp(-g_val))
        silu_g = g_val * inv_denom
        prod = silu_g * u_val
        tl.store(o_base + col_idx, prod.to(out_ptr.dtype.element_ty), mask=col_mask)
        step += 1


@triton.jit
def tiled_mlp_glu_bwd_kernel(
    dout_ptr,
    gate_ptr,
    up_ptr,
    dgate_ptr,
    dup_ptr,
    stride_do_row,
    stride_g_row,
    stride_u_row,
    stride_dg_row,
    stride_du_row,
    n_cols,
    FUSED_GU: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    """2D row-and-column multi-tile backward pass for gated MLP activation. When FUSED_GU is True,
    reads [gate | up] from gate_ptr and writes [d_gate | d_up] in-place into dgate_ptr."""
    row_id = tl.program_id(0)
    group_id = tl.program_id(1)
    do_base = dout_ptr + row_id * stride_do_row
    g_base = gate_ptr + row_id * stride_g_row
    u_base = up_ptr + row_id * stride_u_row
    dg_base = dgate_ptr + row_id * stride_dg_row
    du_base = dup_ptr + row_id * stride_du_row

    step = 0
    while step < TILES_PER_PROG:
        tile_id = group_id * TILES_PER_PROG + step
        col_idx = tile_id * BLOCK_COL + tl.arange(0, BLOCK_COL)
        col_mask = col_idx < n_cols

        do_val = tl.load(do_base + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        g_val = tl.load(g_base + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        if FUSED_GU:
            u_val = tl.load(g_base + n_cols + col_idx, mask=col_mask, other=0.0).to(tl.float32)
        else:
            u_val = tl.load(u_base + col_idx, mask=col_mask, other=0.0).to(tl.float32)

        inv_denom = 1.0 / (1.0 + tl.exp(-g_val))
        silu_g = g_val * inv_denom
        grad_u = do_val * silu_g
        dsilu = inv_denom + silu_g * (1.0 - inv_denom)
        grad_g = do_val * u_val * dsilu

        tl.store(dg_base + col_idx, grad_g.to(dgate_ptr.dtype.element_ty), mask=col_mask)
        if FUSED_GU:
            tl.store(dg_base + n_cols + col_idx, grad_u.to(dgate_ptr.dtype.element_ty), mask=col_mask)
        else:
            tl.store(du_base + col_idx, grad_u.to(dup_ptr.dtype.element_ty), mask=col_mask)
        step += 1


# -----------------------------------------------------------------------------
# 5. Chunked Vocabulary Head Kernels: Online LSE, Entropy, Argmax & Grad (@triton.jit)
# -----------------------------------------------------------------------------

@triton.jit
def lm_head_piece_kernel(
    logits_ptr,
    labels_ptr,
    m_ptr,
    s_ptr,
    w_ptr,
    target_logit_ptr,
    best_val_ptr,
    best_idx_ptr,
    lse_ptr,
    stride_logits_row,
    vocab_start,
    chunk_cols,
    grad_scale,
    MODE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Processes one vocabulary chunk [vocab_start, vocab_start + chunk_cols) per token row.
    MODE == 0 (forward stats pass): updates online max (m), exp-sum (s), weighted exp-sum (w for
    Shannon entropy), global argmax (best_val, best_idx), and extracts target_logit.
    MODE == 1 (backward grad pass): overwrites logits_ptr in-place with scaled softmax cross-entropy
    gradient: grad_scale * (exp(z - lse) - 1_{v == label}) for active tokens, 0 for masked tokens."""
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row).to(tl.int64)
    row_base = logits_ptr + row * stride_logits_row

    if MODE == 0:
        m_curr = tl.load(m_ptr + row).to(tl.float32)
        s_curr = tl.load(s_ptr + row).to(tl.float32)
        w_curr = tl.load(w_ptr + row).to(tl.float32)
        b_val = tl.load(best_val_ptr + row).to(tl.float32)
        b_idx = tl.load(best_idx_ptr + row).to(tl.int64)

        col_start = 0
        while col_start < chunk_cols:
            cols = col_start + tl.arange(0, BLOCK_V)
            mask = cols < chunk_cols
            z = tl.load(row_base + cols, mask=mask, other=-1.0e30).to(tl.float32)
            z_safe = tl.where(mask, z, 0.0)

            tile_max = tl.max(z, axis=0)
            if tile_max > b_val:
                tile_arg = tl.argmax(z, axis=0).to(tl.int64)
                b_val = tile_max
                b_idx = vocab_start + col_start + tile_arg

            m_new = tl.maximum(m_curr, tile_max)
            alpha = tl.exp(m_curr - m_new)
            exp_z = tl.where(mask, tl.exp(z - m_new), 0.0)
            s_curr = s_curr * alpha + tl.sum(exp_z, axis=0)
            w_curr = w_curr * alpha + tl.sum(exp_z * z_safe, axis=0)
            m_curr = m_new
            col_start += BLOCK_V

        tl.store(m_ptr + row, m_curr)
        tl.store(s_ptr + row, s_curr)
        tl.store(w_ptr + row, w_curr)
        tl.store(best_val_ptr + row, b_val)
        tl.store(best_idx_ptr + row, b_idx)

        if label >= vocab_start and label < vocab_start + chunk_cols:
            t_val = tl.load(row_base + (label - vocab_start)).to(tl.float32)
            tl.store(target_logit_ptr + row, t_val)
    else:
        is_valid = label != -100
        lse_val = tl.load(lse_ptr + row).to(tl.float32)
        scale = tl.where(is_valid, grad_scale, 0.0)
        col_start = 0
        while col_start < chunk_cols:
            cols = col_start + tl.arange(0, BLOCK_V)
            mask = cols < chunk_cols
            z = tl.load(row_base + cols, mask=mask, other=-1.0e30).to(tl.float32)
            prob = tl.where(mask, tl.exp(z - lse_val), 0.0)
            v_global = vocab_start + cols
            indicator = tl.where(v_global == label, 1.0, 0.0)
            grad = (prob - indicator) * scale
            tl.store(row_base + cols, grad.to(logits_ptr.dtype.element_ty), mask=mask)
            col_start += BLOCK_V


@triton.jit
def lm_head_finish_kernel(
    m_ptr,
    s_ptr,
    w_ptr,
    target_logit_ptr,
    labels_ptr,
    lse_out_ptr,
    nll_out_ptr,
    entropy_out_ptr,
    n_tokens,
    BLOCK_T: tl.constexpr,
):
    """Finalizes online accumulators (m, s, w, target_logit) into exact logsumexp,
    per-token cross-entropy NLL, and per-token Shannon entropy H = lse - w / s."""
    pid = tl.program_id(0)
    idx = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = idx < n_tokens

    m = tl.load(m_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    t_logit = tl.load(target_logit_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    label = tl.load(labels_ptr + idx, mask=mask, other=-100).to(tl.int64)

    lse = m + tl.log(s)
    entropy = lse - (w / s)
    nll = tl.where(label != -100, lse - t_logit, 0.0)

    tl.store(lse_out_ptr + idx, lse, mask=mask)
    tl.store(nll_out_ptr + idx, nll, mask=mask)
    tl.store(entropy_out_ptr + idx, entropy, mask=mask)


# -----------------------------------------------------------------------------
# Python Launchers & Autograd Wrappers for Standalone Testing and Layer Execution
# -----------------------------------------------------------------------------

def dequant_nf4_tensor(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    code2: torch.Tensor,
    absmax2: torch.Tensor,
    offset: float | tuple[float, float, float],
    out_shape: tuple[int, int],
    out_buf: torch.Tensor | None = None,
    split_bytes: tuple[int, int] | None = None,
    blocksize: int = 64,
    super_blocksize: int = 256,
) -> torch.Tensor:
    """Launch nf4_dequant_turing_kernel into `out_buf` (or a new FP16 tensor of `out_shape`)."""
    n_elements = out_shape[0] * out_shape[1]
    n_bytes = n_elements // 2
    if out_buf is None:
        out = torch.empty(out_shape, dtype=torch.float16, device=packed.device)
    else:
        out = out_buf.view(-1)[:n_elements].view(out_shape)
    out_u32 = out.view(torch.uint32).view(-1)
    if isinstance(offset, (tuple, list)):
        off0, off1, off2 = float(offset[0]), float(offset[1]), float(offset[2])
    else:
        off0 = off1 = off2 = float(offset)
    if split_bytes is not None:
        sb1, sb2 = int(split_bytes[0]), int(split_bytes[1])
    else:
        sb1, sb2 = n_bytes, n_bytes

    BLOCK_BYTES = 512
    grid = (triton.cdiv(n_bytes, BLOCK_BYTES),)
    nf4_dequant_turing_kernel[grid](
        packed.view(-1),
        absmax.view(-1),
        code2.view(-1),
        absmax2.view(-1),
        out_u32,
        off0,
        off1,
        off2,
        sb1,
        sb2,
        n_bytes,
        HALF_BLOCK=blocksize // 2,
        SUPER_BLOCK=super_blocksize,
        BLOCK_BYTES=BLOCK_BYTES,
    )
    return out


def rms_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused residual + RMSNorm forward kernel. Returns (norm_out, sum_out_or_x, rstd)."""
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    x_2d = x.contiguous().view(-1, n_cols)
    n_rows = x_2d.shape[0]
    norm_out = torch.empty_like(x_2d)
    rstd = torch.empty((n_rows,), dtype=torch.float32, device=x.device)
    BLOCK_D = triton.next_power_of_2(n_cols)

    if residual is not None:
        res_2d = residual.contiguous().view(-1, n_cols)
        sum_out = torch.empty_like(x_2d)
        rms_norm_residual_fwd_kernel[(n_rows,)](
            x_2d,
            res_2d,
            weight,
            sum_out,
            norm_out,
            rstd,
            x_2d.stride(0),
            float(eps),
            HAS_RESIDUAL=True,
            N_COLS=n_cols,
            BLOCK_D=BLOCK_D,
        )
        return norm_out.view(orig_shape), sum_out.view(orig_shape), rstd
    else:
        rms_norm_residual_fwd_kernel[(n_rows,)](
            x_2d,
            x_2d,
            weight,
            x_2d,
            norm_out,
            rstd,
            x_2d.stride(0),
            float(eps),
            HAS_RESIDUAL=False,
            N_COLS=n_cols,
            BLOCK_D=BLOCK_D,
        )
        return norm_out.view(orig_shape), x, rstd


def rms_norm_bwd(
    dy: torch.Tensor,
    h: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    dresidual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused RMSNorm backward kernel with optional residual gradient addition."""
    orig_shape = dy.shape
    n_cols = orig_shape[-1]
    dy_2d = dy.contiguous().view(-1, n_cols)
    h_2d = h.contiguous().view(-1, n_cols)
    n_rows = dy_2d.shape[0]
    dx = torch.empty_like(h_2d)
    BLOCK_D = triton.next_power_of_2(n_cols)

    if dresidual is not None:
        dres_2d = dresidual.contiguous().view(-1, n_cols)
        rms_norm_bwd_kernel[(n_rows,)](
            dy_2d,
            h_2d,
            weight,
            rstd,
            dres_2d,
            dx,
            dy_2d.stride(0),
            HAS_DRESIDUAL=True,
            N_COLS=n_cols,
            BLOCK_D=BLOCK_D,
        )
    else:
        rms_norm_bwd_kernel[(n_rows,)](
            dy_2d,
            h_2d,
            weight,
            rstd,
            dy_2d,
            dx,
            dy_2d.stride(0),
            HAS_DRESIDUAL=False,
            N_COLS=n_cols,
            BLOCK_D=BLOCK_D,
        )
    return dx.view(orig_shape)


def qkv_rope_fwd(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_ids: torch.Tensor,
    q_lora: torch.Tensor | None = None,
    v_lora: torch.Tensor | None = None,
    scale_q: float = 2.0,
    scale_v: float = 2.0,
    h_q: int = 12,
    h_kv: int = 2,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run qkv_lora_rope_fwd_kernel to produce (q_rot, k_rot, v_out)."""
    n_tokens = qkv.shape[0]
    q_out = torch.empty((n_tokens, h_q, head_dim), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty((n_tokens, h_kv, head_dim), dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty((n_tokens, h_kv, head_dim), dtype=qkv.dtype, device=qkv.device)
    has_lora = q_lora is not None and v_lora is not None
    ql = q_lora if has_lora else qkv
    vl = v_lora if has_lora else qkv

    grid = (n_tokens, h_q + 2 * h_kv)
    qkv_lora_rope_fwd_kernel[grid](
        qkv,
        ql,
        vl,
        cos,
        sin,
        pos_ids,
        q_out,
        k_out,
        v_out,
        qkv.stride(0),
        ql.stride(0),
        vl.stride(0),
        cos.stride(0),
        float(scale_q),
        float(scale_v),
        HAS_LORA=has_lora,
        H_Q=h_q,
        H_KV=h_kv,
        HALF_DIM=head_dim // 2,
    )
    return q_out, k_out, v_out


def qkv_rope_bwd(
    dq_rot: torch.Tensor,
    dk_rot: torch.Tensor,
    dv_out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_ids: torch.Tensor,
    h_q: int = 12,
    h_kv: int = 2,
    head_dim: int = 128,
) -> torch.Tensor:
    """Run qkv_lora_rope_bwd_kernel to produce packed dqkv of shape [n_tokens, (h_q + 2 * h_kv) * head_dim]."""
    n_tokens = dq_rot.shape[0]
    total_dim = (h_q + 2 * h_kv) * head_dim
    dqkv = torch.empty((n_tokens, total_dim), dtype=dq_rot.dtype, device=dq_rot.device)
    dq_c = dq_rot.contiguous()
    dk_c = dk_rot.contiguous()
    dv_c = dv_out.contiguous()

    grid = (n_tokens, h_q + 2 * h_kv)
    qkv_lora_rope_bwd_kernel[grid](
        dq_c,
        dk_c,
        dv_c,
        cos,
        sin,
        pos_ids,
        dqkv,
        dqkv.stride(0),
        cos.stride(0),
        H_Q=h_q,
        H_KV=h_kv,
        HALF_DIM=head_dim // 2,
    )
    return dqkv


def mlp_glu_fwd(gate_or_gu: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    """Run 2D tiled_mlp_glu_fwd_kernel on either fused [gate|up] (up=None) or separate (gate, up)."""
    BLOCK_COL = 1024
    TILES_PER_PROG = 2
    if up is None:
        orig_shape = gate_or_gu.shape
        n_cols = orig_shape[-1] // 2
        gu_2d = gate_or_gu.view(-1, 2 * n_cols)
        n_rows = gu_2d.shape[0]
        out_2d = torch.empty((n_rows, n_cols), dtype=gate_or_gu.dtype, device=gate_or_gu.device)
        n_groups = triton.cdiv(n_cols, BLOCK_COL * TILES_PER_PROG)
        grid = (n_rows, n_groups)
        tiled_mlp_glu_fwd_kernel[grid](
            gu_2d,
            gu_2d,
            out_2d,
            gu_2d.stride(0),
            gu_2d.stride(0),
            out_2d.stride(0),
            n_cols,
            FUSED_GU=True,
            TILES_PER_PROG=TILES_PER_PROG,
            BLOCK_COL=BLOCK_COL,
        )
        return out_2d.view(*orig_shape[:-1], n_cols)
    else:
        orig_shape = gate_or_gu.shape
        n_cols = orig_shape[-1]
        gate_2d = gate_or_gu.view(-1, n_cols)
        up_2d = up.view(-1, n_cols)
        n_rows = gate_2d.shape[0]
        out_2d = torch.empty((n_rows, n_cols), dtype=gate_or_gu.dtype, device=gate_or_gu.device)
        n_groups = triton.cdiv(n_cols, BLOCK_COL * TILES_PER_PROG)
        grid = (n_rows, n_groups)
        tiled_mlp_glu_fwd_kernel[grid](
            gate_2d,
            up_2d,
            out_2d,
            gate_2d.stride(0),
            up_2d.stride(0),
            out_2d.stride(0),
            n_cols,
            FUSED_GU=False,
            TILES_PER_PROG=TILES_PER_PROG,
            BLOCK_COL=BLOCK_COL,
        )
        return out_2d.view(orig_shape)


def mlp_glu_bwd(
    dout: torch.Tensor,
    gate_or_gu: torch.Tensor,
    up: torch.Tensor | None = None,
    dgu_out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run 2D tiled_mlp_glu_bwd_kernel on fused [gate|up] (up=None, writing in-place to dgu_out or gate_or_gu)
    or separate (gate, up)."""
    BLOCK_COL = 1024
    TILES_PER_PROG = 2
    if up is None:
        n_cols = dout.shape[-1]
        dout_2d = dout.contiguous().view(-1, n_cols)
        gu_2d = gate_or_gu.view(-1, 2 * n_cols)
        n_rows = gu_2d.shape[0]
        dgu_2d = gu_2d if dgu_out is None else dgu_out.view(-1, 2 * n_cols)
        n_groups = triton.cdiv(n_cols, BLOCK_COL * TILES_PER_PROG)
        grid = (n_rows, n_groups)
        tiled_mlp_glu_bwd_kernel[grid](
            dout_2d,
            gu_2d,
            gu_2d,
            dgu_2d,
            dgu_2d,
            dout_2d.stride(0),
            gu_2d.stride(0),
            gu_2d.stride(0),
            dgu_2d.stride(0),
            dgu_2d.stride(0),
            n_cols,
            FUSED_GU=True,
            TILES_PER_PROG=TILES_PER_PROG,
            BLOCK_COL=BLOCK_COL,
        )
        return dgu_2d
    else:
        orig_shape = gate_or_gu.shape
        n_cols = orig_shape[-1]
        dout_2d = dout.contiguous().view(-1, n_cols)
        gate_2d = gate_or_gu.view(-1, n_cols)
        up_2d = up.view(-1, n_cols)
        n_rows = gate_2d.shape[0]
        dg_2d = torch.empty_like(dout_2d)
        du_2d = torch.empty_like(dout_2d)
        n_groups = triton.cdiv(n_cols, BLOCK_COL * TILES_PER_PROG)
        grid = (n_rows, n_groups)
        tiled_mlp_glu_bwd_kernel[grid](
            dout_2d,
            gate_2d,
            up_2d,
            dg_2d,
            du_2d,
            dout_2d.stride(0),
            gate_2d.stride(0),
            up_2d.stride(0),
            dg_2d.stride(0),
            du_2d.stride(0),
            n_cols,
            FUSED_GU=False,
            TILES_PER_PROG=TILES_PER_PROG,
            BLOCK_COL=BLOCK_COL,
        )
        return dg_2d.view(orig_shape), du_2d.view(orig_shape)
