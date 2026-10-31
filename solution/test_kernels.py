"""Unit tests for all 9 custom Triton kernels and autograd operations in `solution/`.

Run on a CUDA GPU: `python -m pytest solution -q`
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


# -----------------------------------------------------------------------------
# 1. Test Turing NF4 Double-Quantization Dequantizer Kernel
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1536, 1536), (2048, 1536), (8960, 1536)])
def test_nf4_dequant_matches_bitsandbytes(shape):
    import bitsandbytes.functional as bnb_F
    from solution.kernels import dequant_nf4_tensor

    torch.manual_seed(42)
    w_fp16 = torch.randn(shape, device="cuda", dtype=torch.float16)
    packed, qs = bnb_F.quantize_4bit(
        w_fp16, blocksize=64, compress_statistics=True, quant_type="nf4", quant_storage=torch.uint8
    )
    ref = bnb_F.dequantize_4bit(packed, qs, quant_type="nf4").to(torch.float16)

    offset = float(qs.offset.item()) if isinstance(qs.offset, torch.Tensor) else float(qs.offset or 0.0)
    out = dequant_nf4_tensor(
        packed=packed.view(-1),
        absmax=qs.absmax.view(-1),
        code2=qs.state2.code.view(-1).to(torch.float32),
        absmax2=qs.state2.absmax.view(-1).to(torch.float32),
        offset=offset,
        out_shape=shape,
        blocksize=int(qs.blocksize),
        super_blocksize=int(qs.state2.blocksize),
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


# -----------------------------------------------------------------------------
# 2. Test Fused Residual + RMSNorm Forward & Backward Kernels
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("shape", [(4, 128, 1536), (273, 1536)])
def test_rms_norm_matches_qwen2_pytorch(dtype, has_residual, shape):
    from solution.ops import rms_norm

    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn((shape[-1],), device="cuda", dtype=dtype)
    res = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) if has_residual else None
    grad_y = torch.randn(shape, device="cuda", dtype=dtype)

    # Reference Qwen2RMSNorm + optional residual
    h_ref = (x + res) if has_residual else x
    h_fp32 = h_ref.to(torch.float32)
    var = h_fp32.pow(2).mean(-1, keepdim=True)
    y_ref = w * (h_fp32 * torch.rsqrt(var + eps)).to(dtype)
    y_ref.backward(grad_y)
    ref_dx = x.grad.float().clone()
    x.grad = None
    if res is not None:
        res.grad = None

    if has_residual:
        y_out, _ = rms_norm(x, w, eps=eps, residual=res)
    else:
        y_out = rms_norm(x, w, eps=eps, residual=None)
    y_out.backward(grad_y)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(y_out.float(), y_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)


# -----------------------------------------------------------------------------
# 3. Test Fused QKV + LoRA + NeoX RoPE Forward & Backward Kernels
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("n_tokens", [64, 311])
def test_qkv_lora_rope_matches_pytorch(dtype, n_tokens):
    from solution.ops import qkv_lora_rope

    torch.manual_seed(1)
    h_q, h_kv, head_dim = 12, 2, 128
    q_dim, kv_dim = h_q * head_dim, h_kv * head_dim
    scale_q, scale_v = 2.0, 2.0

    qkv = torch.randn((n_tokens, q_dim + 2 * kv_dim), device="cuda", dtype=dtype, requires_grad=True)
    q_lora = torch.randn((n_tokens, q_dim), device="cuda", dtype=dtype, requires_grad=True)
    v_lora = torch.randn((n_tokens, kv_dim), device="cuda", dtype=dtype, requires_grad=True)
    cos = torch.randn((512, head_dim), device="cuda", dtype=dtype)
    sin = torch.randn((512, head_dim), device="cuda", dtype=dtype)
    pos_ids = torch.randint(0, 512, (n_tokens,), device="cuda", dtype=torch.long)

    dq = torch.randn((n_tokens, h_q, head_dim), device="cuda", dtype=dtype)
    dk = torch.randn((n_tokens, h_kv, head_dim), device="cuda", dtype=dtype)
    dv = torch.randn((n_tokens, h_kv, head_dim), device="cuda", dtype=dtype)

    def _rotate_half(t):
        t1 = t[..., : head_dim // 2]
        t2 = t[..., head_dim // 2 :]
        return torch.cat((-t2, t1), dim=-1)

    # Reference PyTorch
    q_ref = (qkv[:, :q_dim].float() + q_lora.float() * scale_q).view(n_tokens, h_q, head_dim)
    k_ref = qkv[:, q_dim : q_dim + kv_dim].float().view(n_tokens, h_kv, head_dim)
    v_ref = (qkv[:, q_dim + kv_dim :].float() + v_lora.float() * scale_v).view(n_tokens, h_kv, head_dim)
    cos_p = cos[pos_ids].float().unsqueeze(1)
    sin_p = sin[pos_ids].float().unsqueeze(1)
    q_rot_ref = (q_ref * cos_p + _rotate_half(q_ref) * sin_p).to(dtype)
    k_rot_ref = (k_ref * cos_p + _rotate_half(k_ref) * sin_p).to(dtype)
    v_out_ref = v_ref.to(dtype)

    loss_ref = (q_rot_ref.float() * dq.float()).sum() + (k_rot_ref.float() * dk.float()).sum() + (v_out_ref.float() * dv.float()).sum()
    loss_ref.backward()
    ref_dqkv = qkv.grad.float().clone()
    ref_dql = q_lora.grad.float().clone()
    ref_dvl = v_lora.grad.float().clone()
    qkv.grad = q_lora.grad = v_lora.grad = None

    q_out, k_out, v_out = qkv_lora_rope(
        qkv, cos, sin, pos_ids, q_lora=q_lora, v_lora=v_lora, scale_q=scale_q, scale_v=scale_v
    )
    loss_out = (q_out.float() * dq.float()).sum() + (k_out.float() * dk.float()).sum() + (v_out.float() * dv.float()).sum()
    loss_out.backward()

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(q_out.float(), q_rot_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(k_out.float(), k_rot_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(v_out.float(), v_out_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(qkv.grad.float(), ref_dqkv, rtol=tol, atol=tol)
    torch.testing.assert_close(q_lora.grad.float(), ref_dql, rtol=tol, atol=tol)
    torch.testing.assert_close(v_lora.grad.float(), ref_dvl, rtol=tol, atol=tol)


# -----------------------------------------------------------------------------
# 4. Test 2D Multi-Tile MLP GLU Forward & Backward Kernels (Separate & Fused [gate|up])
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("shape", [(4, 64, 8960), (3, 77, 1001)])
def test_tiled_mlp_glu_matches_pytorch(dtype, fused, shape):
    from solution.ops import mlp_glu

    torch.manual_seed(2)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = F.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float().clone(), up.grad.float().clone()
    gate.grad = up.grad = None

    if fused:
        gu = torch.cat([gate.detach(), up.detach()], dim=-1).requires_grad_(True)
        out = mlp_glu(gu, None)
        out.backward(grad)
        dgate_got = gu.grad[..., : shape[-1]].float()
        dup_got = gu.grad[..., shape[-1] :].float()
    else:
        out = mlp_glu(gate, up)
        out.backward(grad)
        dgate_got, dup_got = gate.grad.float(), up.grad.float()

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(dgate_got, ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(dup_got, ref_dup, rtol=tol, atol=tol)


# -----------------------------------------------------------------------------
# 5. Test Chunked Vocabulary Head (lm_head_piece_kernel + lm_head_finish_kernel + Proxy)
# -----------------------------------------------------------------------------

def test_chunked_lm_head_and_proxy_match_pytorch():
    from solution.ops import ChunkedLogitsProxy, chunked_lm_head

    torch.manual_seed(3)
    B, S, D, V = 2, 32, 256, 4099
    hidden = torch.randn((B, S, D), device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn((V, D), device="cuda", dtype=torch.float16) * 0.02
    labels = torch.randint(0, V, (B, S), device="cuda", dtype=torch.long)
    labels[:, :8] = -100  # mask prompt tokens

    shift_labels = torch.full((B, S), -100, dtype=torch.long, device="cuda")
    shift_labels[:, :-1] = labels[:, 1:]

    # Reference full-logits Cross-Entropy, Shannon Entropy, and Argmax
    logits_ref = F.linear(hidden, weight).float()
    loss_ref = F.cross_entropy(
        logits_ref[:, :-1].reshape(-1, V),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    loss_ref.backward()
    ref_dhidden = hidden.grad.float().clone()
    hidden.grad = None

    probs_ref = F.softmax(logits_ref, dim=-1)
    ent_ref = torch.logsumexp(logits_ref, dim=-1) - torch.sum(probs_ref * logits_ref, dim=-1)
    argmax_ref = logits_ref.argmax(dim=-1)
    row_loss_0_ref = F.cross_entropy(logits_ref[0, :-1], labels[0, 1:], ignore_index=-100)

    # Chunked LM Head (v_chunk = 1024 across V = 4099)
    loss_out, nll_1d, ent_1d, argmax_1d = chunked_lm_head(hidden, shift_labels, weight, v_chunk=1024)
    loss_out.backward()

    torch.testing.assert_close(loss_out.float(), loss_ref.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(hidden.grad.float(), ref_dhidden, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(ent_1d.view(B, S), ent_ref, rtol=2e-3, atol=2e-3)
    assert torch.equal(argmax_1d.view(B, S), argmax_ref)

    # Verify ChunkedLogitsProxy compatibility with TRL & Soup Rewind
    proxy = ChunkedLogitsProxy(
        per_token_nll=nll_1d.view(B, S),
        per_token_entropy=ent_1d.view(B, S),
        per_token_argmax=argmax_1d.view(B, S),
        vocab_size=V,
    )
    assert proxy.shape == (B, S, V)
    proxy_ent = torch.logsumexp(proxy, dim=-1) - torch.sum(F.softmax(proxy, dim=-1) * proxy, dim=-1)
    torch.testing.assert_close(proxy_ent, ent_ref, rtol=2e-3, atol=2e-3)
    assert torch.equal(proxy[:, :-1].argmax(dim=-1), argmax_ref[:, :-1])
    assert torch.equal(proxy.argmax(dim=-1)[:, :-1], argmax_ref[:, :-1])
    proxy_row_0 = F.cross_entropy(proxy[0, :-1].float(), labels[0, 1:], ignore_index=-100)
    torch.testing.assert_close(proxy_row_0, row_loss_0_ref, rtol=2e-3, atol=2e-3)
