"""Custom autograd operations, fused Qwen2 layer stack, chunked vocabulary head, and runtime patches.

Connects the 9 custom Triton kernels in `solution/kernels.py` and the host-pinned streaming engine
in `solution/hostload.py` into Qwen2ForCausalLM, PEFT, and TRL/Soup SFTTrainer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from .hostload import MAX_MLP_TOKENS, V_CHUNK_DEFAULT, HostHead, InputStream, WeightStream
from .kernels import (
    lm_head_finish_kernel,
    lm_head_piece_kernel,
    mlp_glu_bwd,
    mlp_glu_fwd,
    qkv_rope_bwd,
    qkv_rope_fwd,
    rms_norm_bwd,
    rms_norm_fwd,
)


# -----------------------------------------------------------------------------
# 1. Standalone Autograd Functions (used both in unit tests and modular paths)
# -----------------------------------------------------------------------------

class FusedRMSNormFn(torch.autograd.Function):
    """Autograd wrapper around rms_norm_residual_fwd_kernel and rms_norm_bwd_kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, residual: torch.Tensor | None = None):
        norm_out, sum_out, rstd = rms_norm_fwd(x, weight, eps=eps, residual=residual)
        ctx.save_for_backward(sum_out, weight, rstd)
        ctx.has_residual = residual is not None
        return (norm_out, sum_out) if residual is not None else norm_out

    @staticmethod
    def backward(ctx, * grads):
        sum_out, weight, rstd = ctx.saved_tensors
        if ctx.has_residual:
            dy, dres = grads[0], grads[1]
            dx = rms_norm_bwd(dy, sum_out, weight, rstd, dresidual=dres)
            return dx, None, None, dx
        else:
            dy = grads[0]
            dx = rms_norm_bwd(dy, sum_out, weight, rstd, dresidual=None)
            return dx, None, None, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, residual: torch.Tensor | None = None):
    return FusedRMSNormFn.apply(x, weight, eps, residual)


class FusedMLPGLUFn(torch.autograd.Function):
    """Autograd wrapper around tiled_mlp_glu_fwd_kernel and tiled_mlp_glu_bwd_kernel."""

    @staticmethod
    def forward(ctx, gate_or_gu: torch.Tensor, up: torch.Tensor | None = None):
        out = mlp_glu_fwd(gate_or_gu, up)
        if up is None:
            ctx.save_for_backward(gate_or_gu)
            ctx.fused = True
        else:
            ctx.save_for_backward(gate_or_gu, up)
            ctx.fused = False
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        if ctx.fused:
            (gu,) = ctx.saved_tensors
            dgu = torch.empty_like(gu)
            mlp_glu_bwd(dout, gu, None, dgu_out=dgu)
            return dgu, None
        else:
            gate, up = ctx.saved_tensors
            dgate, dup = mlp_glu_bwd(dout, gate, up)
            return dgate, dup


def mlp_glu(gate_or_gu: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    return FusedMLPGLUFn.apply(gate_or_gu, up)


class FusedQKVLoRARoPEFn(torch.autograd.Function):
    """Autograd wrapper around qkv_lora_rope_fwd_kernel and qkv_lora_rope_bwd_kernel."""

    @staticmethod
    def forward(
        ctx,
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
    ):
        q_out, k_out, v_out = qkv_rope_fwd(
            qkv,
            cos,
            sin,
            pos_ids,
            q_lora=q_lora,
            v_lora=v_lora,
            scale_q=scale_q,
            scale_v=scale_v,
            h_q=h_q,
            h_kv=h_kv,
            head_dim=head_dim,
        )
        ctx.save_for_backward(cos, sin, pos_ids)
        ctx.has_lora = q_lora is not None and v_lora is not None
        ctx.scale_q = float(scale_q)
        ctx.scale_v = float(scale_v)
        ctx.h_q = h_q
        ctx.h_kv = h_kv
        ctx.head_dim = head_dim
        return q_out, k_out, v_out

    @staticmethod
    def backward(ctx, dq_rot: torch.Tensor, dk_rot: torch.Tensor, dv_out: torch.Tensor):
        cos, sin, pos_ids = ctx.saved_tensors
        dqkv = qkv_rope_bwd(
            dq_rot,
            dk_rot,
            dv_out,
            cos,
            sin,
            pos_ids,
            h_q=ctx.h_q,
            h_kv=ctx.h_kv,
            head_dim=ctx.head_dim,
        )
        if not ctx.has_lora:
            return dqkv, None, None, None, None, None, None, None, None, None, None
        q_dim = ctx.h_q * ctx.head_dim
        kv_dim = ctx.h_kv * ctx.head_dim
        dq_lora = (dqkv[:, :q_dim].to(torch.float32) * ctx.scale_q).to(dqkv.dtype)
        dv_lora = (dqkv[:, q_dim + kv_dim :].to(torch.float32) * ctx.scale_v).to(dqkv.dtype)
        return dqkv, None, None, None, dq_lora, dv_lora, None, None, None, None, None


def qkv_lora_rope(
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
):
    return FusedQKVLoRARoPEFn.apply(qkv, cos, sin, pos_ids, q_lora, v_lora, scale_q, scale_v, h_q, h_kv, head_dim)


# -----------------------------------------------------------------------------
# 2. Chunked Vocabulary Head (lm_head_piece_kernel + lm_head_finish_kernel)
# -----------------------------------------------------------------------------

PRESCALE_GRAD = 65536.0  # Exact power-of-2 (2^16) lossless exponent shift for FP16 Tensor Core GEMMs


class FusedChunkedLMHeadFn(torch.autograd.Function):
    """Computes exact cross-entropy loss, per-token NLL, per-token Shannon entropy, and per-token
    argmax in vocabulary chunks without ever allocating the [N_tokens, 151936] logits tensor."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        shift_labels: torch.Tensor,
        weight_or_host_head: torch.Tensor | HostHead,
        num_items_in_batch: int | torch.Tensor | None = None,
        v_chunk: int = V_CHUNK_DEFAULT,
    ):
        device = hidden_states.device
        orig_dtype = hidden_states.dtype
        h_2d = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
        labels_1d = shift_labels.contiguous().view(-1).to(torch.int64)
        n_tokens, hidden_dim = h_2d.shape

        is_host_head = isinstance(weight_or_host_head, HostHead)
        vocab_size = weight_or_host_head.vocab_size if is_host_head else weight_or_host_head.shape[0]
        if is_host_head:
            v_chunk = weight_or_host_head.v_chunk

        # Online accumulators per token (only 4 bytes per token!)
        m_acc = torch.full((n_tokens,), -1.0e30, dtype=torch.float32, device=device)
        s_acc = torch.zeros((n_tokens,), dtype=torch.float32, device=device)
        w_acc = torch.zeros((n_tokens,), dtype=torch.float32, device=device)
        target_logit = torch.zeros((n_tokens,), dtype=torch.float32, device=device)
        best_val = torch.full((n_tokens,), -1.0e30, dtype=torch.float32, device=device)
        best_idx = torch.zeros((n_tokens,), dtype=torch.int64, device=device)
        lse_out = torch.empty((n_tokens,), dtype=torch.float32, device=device)
        nll_out = torch.empty((n_tokens,), dtype=torch.float32, device=device)
        entropy_out = torch.empty((n_tokens,), dtype=torch.float32, device=device)

        logits_chunk_buf = torch.empty((n_tokens, v_chunk), dtype=orig_dtype, device=device)
        BLOCK_V = min(1024, triton.next_power_of_2(v_chunk))
        num_chunks = triton.cdiv(vocab_size, v_chunk)

        if is_host_head and num_chunks > 0:
            weight_or_host_head.prefetch_chunk(0, 0, min(v_chunk, vocab_size))

        # Pass 1: Online logsumexp, Shannon entropy numerator, argmax, and target logit extraction
        for c in range(num_chunks):
            start_v = c * v_chunk
            end_v = min(start_v + v_chunk, vocab_size)
            chunk_cols = end_v - start_v
            if is_host_head:
                if c + 1 < num_chunks:
                    ns = (c + 1) * v_chunk
                    ne = min(ns + v_chunk, vocab_size)
                    weight_or_host_head.prefetch_chunk(c + 1, ns, ne)
                w_chunk = weight_or_host_head.get_chunk(c, start_v, end_v)
            else:
                w_chunk = weight_or_host_head[start_v:end_v]

            logits_slice = logits_chunk_buf[:, :chunk_cols]
            torch.mm(h_2d, w_chunk.t(), out=logits_slice)
            if is_host_head and c < num_chunks - 1:
                weight_or_host_head.release_chunk(c)

            lm_head_piece_kernel[(n_tokens,)](
                logits_slice,
                labels_1d,
                m_acc,
                s_acc,
                w_acc,
                target_logit,
                best_val,
                best_idx,
                lse_out,
                logits_slice.stride(0),
                start_v,
                chunk_cols,
                1.0,
                MODE=0,
                BLOCK_V=BLOCK_V,
            )

        # Finalize LSE, per-token NLL, and per-token Shannon entropy
        BLOCK_T = 256
        lm_head_finish_kernel[(triton.cdiv(n_tokens, BLOCK_T),)](
            m_acc,
            s_acc,
            w_acc,
            target_logit,
            labels_1d,
            lse_out,
            nll_out,
            entropy_out,
            n_tokens,
            BLOCK_T=BLOCK_T,
        )

        valid_mask = labels_1d != -100
        if num_items_in_batch is not None:
            if isinstance(num_items_in_batch, torch.Tensor):
                denom = float(num_items_in_batch.detach().item())
            else:
                denom = float(num_items_in_batch)
        else:
            denom = float(valid_mask.sum().item())
        denom = max(denom, 1.0)
        loss = nll_out.sum() / denom

        # Pass 2: Compute d_hidden in chunks while the last vocabulary chunk is still hot in GPU slot!
        need_grad = torch.is_grad_enabled() and (hidden_states.requires_grad or (not is_host_head and weight_or_host_head.requires_grad))
        d_h_prescaled = None
        if need_grad:
            d_h_prescaled = torch.zeros_like(h_2d)
            grad_scale = float(PRESCALE_GRAD / denom)

            # Iterate backwards from num_chunks - 1 down to 0 so chunk (num_chunks - 1) is reused immediately
            for c in range(num_chunks - 1, -1, -1):
                start_v = c * v_chunk
                end_v = min(start_v + v_chunk, vocab_size)
                chunk_cols = end_v - start_v
                if is_host_head:
                    if c - 1 >= 0:
                        ps = (c - 1) * v_chunk
                        pe = min(ps + v_chunk, vocab_size)
                        weight_or_host_head.prefetch_chunk(c - 1, ps, pe)
                    w_chunk = weight_or_host_head.get_chunk(c, start_v, end_v)
                else:
                    w_chunk = weight_or_host_head[start_v:end_v]

                logits_slice = logits_chunk_buf[:, :chunk_cols]
                torch.mm(h_2d, w_chunk.t(), out=logits_slice)

                lm_head_piece_kernel[(n_tokens,)](
                    logits_slice,
                    labels_1d,
                    m_acc,
                    s_acc,
                    w_acc,
                    target_logit,
                    best_val,
                    best_idx,
                    lse_out,
                    logits_slice.stride(0),
                    start_v,
                    chunk_cols,
                    grad_scale,
                    MODE=1,
                    BLOCK_V=BLOCK_V,
                )
                d_h_prescaled.addmm_(logits_slice, w_chunk)
                if is_host_head:
                    weight_or_host_head.release_chunk(c)
        elif is_host_head and num_chunks > 0:
            weight_or_host_head.release_chunk(num_chunks - 1)

        ctx.save_for_backward(d_h_prescaled)
        ctx.orig_shape = hidden_states.shape
        ctx.orig_dtype = orig_dtype
        ctx.mark_non_differentiable(nll_out, entropy_out, best_idx)
        return loss, nll_out.detach(), entropy_out.detach(), best_idx.detach()

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, _grad_nll=None, _grad_ent=None, _grad_arg=None):
        (d_h_prescaled,) = ctx.saved_tensors
        if d_h_prescaled is None:
            return None, None, None, None, None
        scale_factor = (grad_loss.to(torch.float32) * (1.0 / PRESCALE_GRAD)).to(ctx.orig_dtype)
        d_h = (d_h_prescaled * scale_factor).view(ctx.orig_shape)
        return d_h, None, None, None, None


def chunked_lm_head(
    hidden_states: torch.Tensor,
    shift_labels: torch.Tensor,
    weight_or_host_head: torch.Tensor | HostHead,
    num_items_in_batch: int | torch.Tensor | None = None,
    v_chunk: int = V_CHUNK_DEFAULT,
):
    return FusedChunkedLMHeadFn.apply(hidden_states, shift_labels, weight_or_host_head, num_items_in_batch, v_chunk)


# -----------------------------------------------------------------------------
# 3. Zero-Memory ChunkedLogitsProxy (transparently serves TRL & Soup Rewind)
# -----------------------------------------------------------------------------

class _ZeroEntropyTermProxy(torch.Tensor):
    """Helper proxy returned by F.softmax(ChunkedLogitsProxy) so `torch.logsumexp(logits, -1) -
    torch.sum(F.softmax(logits, -1) * logits, -1)` evaluates to exact `per_token_entropy`."""

    @staticmethod
    def __new__(cls, entropy_tensor: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            entropy_tensor.shape,
            dtype=entropy_tensor.dtype,
            device=entropy_tensor.device,
            requires_grad=False,
        )

    def __init__(self, entropy_tensor: torch.Tensor):
        self._entropy = entropy_tensor

    __torch_dispatch__ = classmethod(lambda cls, func, types, args=(), kwargs=None: None)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        self_obj = next((a for a in args if isinstance(a, cls)), None)
        ent = self_obj._entropy
        if func in (torch.mul, torch.Tensor.mul, torch.Tensor.__mul__, torch.Tensor.__rmul__):
            return cls(ent)
        if func in (torch.sum, torch.Tensor.sum):
            return torch.zeros_like(ent)
        return torch.zeros_like(ent)


class _LogSoftmaxGatherProxy(torch.Tensor):
    """Helper proxy returned by F.log_softmax(ChunkedLogitsProxy) so `.gather(-1, labels)`
    returns `-per_token_nll`."""

    @staticmethod
    def __new__(cls, nll_tensor: torch.Tensor, vocab_size: int):
        shape = (*nll_tensor.shape, vocab_size)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=nll_tensor.dtype,
            device=nll_tensor.device,
            requires_grad=False,
        )

    def __init__(self, nll_tensor: torch.Tensor, vocab_size: int):
        self._nll = nll_tensor
        self._vocab_size = vocab_size

    __torch_dispatch__ = classmethod(lambda cls, func, types, args=(), kwargs=None: None)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        self_obj = args[0]
        if func in (torch.gather, torch.Tensor.gather):
            return (-self_obj._nll).unsqueeze(-1)
        if func in (torch.Tensor.__getitem__,):
            idx = args[1]
            if isinstance(idx, tuple) and len(idx) >= 2:
                return _LogSoftmaxGatherProxy(self_obj._nll[idx[:-1]], self_obj._vocab_size)
            return _LogSoftmaxGatherProxy(self_obj._nll[idx], self_obj._vocab_size)
        return (-self_obj._nll)


class ChunkedLogitsProxy(torch.Tensor):
    """Zero-storage Tensor subclass of virtual shape (B, S, V) that transparently serves
    precomputed per-token NLL, per-token Shannon entropy, and per-token argmax to TRL's
    SFTTrainer and Soup's RewindSFTTrainer without allocating the 3-5 GB logits tensor."""

    @staticmethod
    def __new__(
        cls,
        per_token_nll: torch.Tensor,
        per_token_entropy: torch.Tensor,
        per_token_argmax: torch.Tensor,
        vocab_size: int,
        virtual_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        if virtual_shape is None:
            virtual_shape = (*per_token_entropy.shape, vocab_size)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            virtual_shape,
            dtype=dtype,
            device=per_token_nll.device,
            requires_grad=False,
        )

    def __init__(
        self,
        per_token_nll: torch.Tensor,
        per_token_entropy: torch.Tensor,
        per_token_argmax: torch.Tensor,
        vocab_size: int,
        virtual_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.per_token_nll = per_token_nll
        self.per_token_entropy = per_token_entropy
        self.per_token_argmax = per_token_argmax
        self.vocab_size = vocab_size

    __torch_dispatch__ = classmethod(lambda cls, func, types, args=(), kwargs=None: None)

    def _clone_with(
        self,
        nll: torch.Tensor | None = None,
        ent: torch.Tensor | None = None,
        arg: torch.Tensor | None = None,
        vshape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ChunkedLogitsProxy":
        return ChunkedLogitsProxy(
            nll if nll is not None else self.per_token_nll,
            ent if ent is not None else self.per_token_entropy,
            arg if arg is not None else self.per_token_argmax,
            self.vocab_size,
            virtual_shape=vshape if vshape is not None else self.shape,
            dtype=dtype if dtype is not None else self.dtype,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        self_obj = next((a for a in args if isinstance(a, cls)), None)
        if self_obj is None:
            return super().__torch_function__(func, types, args, kwargs)

        # No-op layout / dtype / detach transformations
        if func in (
            torch.Tensor.contiguous,
            torch.Tensor.detach,
            torch.Tensor.clone,
        ):
            return self_obj
        if func in (torch.Tensor.float,):
            return self_obj._clone_with(dtype=torch.float32)
        if func in (torch.Tensor.half,):
            return self_obj._clone_with(dtype=torch.float16)
        if func in (torch.Tensor.to,):
            target_dtype = kwargs.get("dtype")
            if target_dtype is None and len(args) > 1 and isinstance(args[1], torch.dtype):
                target_dtype = args[1]
            return self_obj._clone_with(dtype=target_dtype or self_obj.dtype)

        # Reshape / view (e.g. shift_logits.view(-1, V) or reshape(-1, V))
        if func in (torch.Tensor.view, torch.Tensor.reshape, torch.reshape):
            shape_args = args[1:]
            if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list, torch.Size)):
                shape_args = tuple(shape_args[0])
            if len(shape_args) == 2 and shape_args[1] in (-1, self_obj.vocab_size):
                flat_nll = self_obj.per_token_nll.reshape(-1)
                flat_ent = self_obj.per_token_entropy.reshape(-1)
                flat_arg = self_obj.per_token_argmax.reshape(-1)
                return self_obj._clone_with(
                    nll=flat_nll,
                    ent=flat_ent,
                    arg=flat_arg,
                    vshape=(flat_nll.numel(), self_obj.vocab_size),
                )
            return self_obj

        # Slicing / indexing: logits[:, :-1], logits[..., :-1, :], logits[i], etc.
        if func is torch.Tensor.__getitem__:
            idx = args[1]
            if not isinstance(idx, tuple):
                idx = (idx,)
            # Strip trailing slice(None) on vocab axis and Ellipsis
            clean_idx = [x for x in idx if x is not Ellipsis]
            if len(clean_idx) == self_obj.ndim and isinstance(clean_idx[-1], slice) and clean_idx[-1] == slice(None):
                token_idx = tuple(clean_idx[:-1])
            else:
                token_idx = tuple(clean_idx)

            # Check if this is the standard causal shift `[:, :-1]` or `[..., :-1, :]`
            if (
                len(token_idx) >= 1
                and isinstance(token_idx[-1], slice)
                and token_idx[-1].start is None
                and token_idx[-1].stop == -1
            ):
                # `per_token_nll` and `per_token_argmax` are already aligned to shifted positions [:, :-1]
                prefix_idx = token_idx[:-1]
                nll_sliced = self_obj.per_token_nll[prefix_idx] if prefix_idx else self_obj.per_token_nll
                arg_sliced = self_obj.per_token_argmax[prefix_idx] if prefix_idx else self_obj.per_token_argmax
                ent_sliced = self_obj.per_token_entropy[token_idx]
                return self_obj._clone_with(
                    nll=nll_sliced,
                    ent=ent_sliced,
                    arg=arg_sliced,
                    vshape=(*nll_sliced.shape, self_obj.vocab_size),
                )
            # Row indexing (e.g., logits[b] or logits[b, :-1])
            try:
                nll_sub = self_obj.per_token_nll[token_idx]
                arg_sub = self_obj.per_token_argmax[token_idx]
                ent_sub = self_obj.per_token_entropy[token_idx]
                return self_obj._clone_with(
                    nll=nll_sub,
                    ent=ent_sub,
                    arg=arg_sub,
                    vshape=(*nll_sub.shape, self_obj.vocab_size),
                )
            except Exception:
                return self_obj

        # Argmax (for TRL mean_token_accuracy)
        if func in (torch.argmax, torch.Tensor.argmax):
            return self_obj.per_token_argmax

        # Cross-entropy (for Soup rewind.jsonl per-row loss or Trainer loss verification)
        if func is F.cross_entropy:
            target = args[1] if len(args) > 1 else kwargs.get("target")
            ignore_index = kwargs.get("ignore_index", -100)
            reduction = kwargs.get("reduction", "mean")
            nll = self_obj.per_token_nll.reshape(target.shape)
            valid = target != ignore_index
            masked_nll = torch.where(valid, nll, torch.zeros_like(nll))
            if reduction == "none":
                return masked_nll
            elif reduction == "sum":
                return masked_nll.sum()
            else:
                denom = valid.sum().clamp(min=1)
                return masked_nll.sum() / denom

        # Entropy calculation in TRL (`logsumexp(logits, dim=-1) - sum(softmax(logits, dim=-1) * logits, dim=-1)`)
        if func in (torch.logsumexp, torch.Tensor.logsumexp):
            return self_obj.per_token_entropy
        if func in (F.softmax, torch.softmax, torch.Tensor.softmax):
            return _ZeroEntropyTermProxy(self_obj.per_token_entropy)
        if func in (F.log_softmax, torch.log_softmax, torch.Tensor.log_softmax):
            return _LogSoftmaxGatherProxy(self_obj.per_token_nll, self_obj.vocab_size)

        return self_obj


# -----------------------------------------------------------------------------
# 4. Fused 28-Layer Qwen2 Execution Engine & Autograd Function
# -----------------------------------------------------------------------------

class Qwen2FusedEngine:
    """Manages WeightStream, HostHead, InputStream, RoPE cache, and LoRA references for Qwen2ForCausalLM."""

    def __init__(self, causal_lm: torch.nn.Module, device: torch.device):
        self.device = device
        self.config = causal_lm.config
        self.qwen_model = causal_lm.model
        self.layers = list(self.qwen_model.layers)
        self.num_layers = len(self.layers)
        self.hidden_size = int(self.config.hidden_size)
        self.intermediate_size = int(self.config.intermediate_size)
        self.num_heads = int(self.config.num_attention_heads)
        self.num_kv_heads = int(self.config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.rms_eps = float(getattr(self.config, "rms_norm_eps", 1e-6))

        self.weight_stream = WeightStream(self.layers, device)
        self.host_head = HostHead(self.qwen_model.embed_tokens, causal_lm.lm_head, device)
        self.input_stream = InputStream(self.num_layers, self.hidden_size, max_tokens=2048, device=device)

        # Keep norm weights in FP16/FP32 on GPU
        self.ln1_weights = [layer.input_layernorm.weight for layer in self.layers]
        self.ln2_weights = [layer.post_attention_layernorm.weight for layer in self.layers]
        self.final_norm_weight = self.qwen_model.norm.weight

        # Extract LoRA parameters and scales per layer
        self.lora_params: list[torch.nn.Parameter] = []
        self.layer_lora_meta: list[dict] = []
        for layer in self.layers:
            q_proj = layer.self_attn.q_proj
            v_proj = layer.self_attn.v_proj
            q_A = q_proj.lora_A["default"].weight
            q_B = q_proj.lora_B["default"].weight
            v_A = v_proj.lora_A["default"].weight
            v_B = v_proj.lora_B["default"].weight
            scale_q = float(q_proj.scaling["default"])
            scale_v = float(v_proj.scaling["default"])
            drop_q_mod = q_proj.lora_dropout["default"]
            drop_v_mod = v_proj.lora_dropout["default"]
            p_q = float(getattr(drop_q_mod, "p", 0.0))
            p_v = float(getattr(drop_v_mod, "p", 0.0))
            self.lora_params.extend([q_A, q_B, v_A, v_B])
            self.layer_lora_meta.append(
                {
                    "scale_q": scale_q,
                    "scale_v": scale_v,
                    "p_q": p_q,
                    "p_v": p_v,
                }
            )

        # Precompute RoPE (cos, sin) table up to max_seq_len = 2048
        self.cos_table, self.sin_table = self._build_rope_cache(max_pos=2048)
        torch.cuda.empty_cache()

    def _build_rope_cache(self, max_pos: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(0, max_pos, dtype=torch.long, device=self.device).unsqueeze(0)
        dummy = torch.empty((1, max_pos, self.head_dim), dtype=torch.float16, device=self.device)
        rotary_emb = getattr(self.qwen_model, "rotary_emb", None)
        if rotary_emb is None:
            rotary_emb = getattr(self.layers[0].self_attn, "rotary_emb", None)
        with torch.no_grad():
            cos, sin = rotary_emb(dummy, pos)
        return cos.squeeze(0).contiguous(), sin.squeeze(0).contiguous()


def _run_packed_sdpa(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v_out: torch.Tensor,
    row_lengths: list[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Run causal Grouped-Query Attention SDPA per sequence on packed valid tokens (zero padding)."""
    n_tokens = q_rot.shape[0]
    attn_out = torch.empty((n_tokens, num_heads * head_dim), dtype=q_rot.dtype, device=q_rot.device)
    gqa_groups = num_heads // num_kv_heads
    cur = 0
    for seq_len in row_lengths:
        if seq_len <= 0:
            continue
        nxt = cur + seq_len
        q_b = q_rot[cur:nxt].transpose(0, 1).unsqueeze(0)  # [1, H_q, L, D_h]
        k_b = k_rot[cur:nxt].transpose(0, 1).unsqueeze(0)  # [1, H_kv, L, D_h]
        v_b = v_out[cur:nxt].transpose(0, 1).unsqueeze(0)  # [1, H_kv, L, D_h]
        if gqa_groups > 1:
            k_b = k_b.unsqueeze(2).expand(1, num_kv_heads, gqa_groups, seq_len, head_dim).reshape(1, num_heads, seq_len, head_dim)
            v_b = v_b.unsqueeze(2).expand(1, num_kv_heads, gqa_groups, seq_len, head_dim).reshape(1, num_heads, seq_len, head_dim)
        out_b = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
        attn_out[cur:nxt] = out_b.squeeze(0).transpose(0, 1).reshape(seq_len, num_heads * head_dim)
        cur = nxt
    return attn_out


def _run_chunked_mlp_fwd(
    norm2: torch.Tensor,
    w_gu: torch.Tensor,
    w_down: torch.Tensor,
    ws: WeightStream,
    layer_idx: int,
) -> torch.Tensor:
    """Run MLP ([gate|up] GEMM -> tiled_mlp_glu_fwd_kernel -> down_proj GEMM) in token chunks
    of at most MAX_MLP_TOKENS = 512 to bound peak MLP activation memory under STEP_CAP_MIB."""
    n_tokens, hidden_dim = norm2.shape
    if n_tokens <= MAX_MLP_TOKENS:
        gu = F.linear(norm2, w_gu)
        act = mlp_glu_fwd(gu)
        del gu
        w_down_deq = ws.dequant(layer_idx, "down")
        return F.linear(act, w_down_deq)

    mlp_out = torch.empty((n_tokens, hidden_dim), dtype=norm2.dtype, device=norm2.device)
    for start in range(0, n_tokens, MAX_MLP_TOKENS):
        end = min(start + MAX_MLP_TOKENS, n_tokens)
        w_gu_deq = ws.dequant(layer_idx, "gu")
        gu_c = F.linear(norm2[start:end], w_gu_deq)
        act_c = mlp_glu_fwd(gu_c)
        del gu_c
        w_down_deq = ws.dequant(layer_idx, "down")
        mlp_out[start:end] = F.linear(act_c, w_down_deq)
        del act_c
    return mlp_out


class FusedQwen2StackFn(torch.autograd.Function):
    """Executes all 28 Qwen2 decoder layers on packed padding-free tokens with:
    - Asynchronous WeightStream prefetching (2 GPU slots)
    - Asynchronous InputStream activation offloading to pinned host RAM
    - Exact Philox RNG state replay for LoRA dropout
    - Custom Triton kernels for NF4 dequantization, RMSNorm, QKV+LoRA+RoPE, and tiled MLP GLU."""

    @staticmethod
    def forward(
        ctx,
        h_packed: torch.Tensor,
        pos_ids: torch.Tensor,
        row_lengths: list[int],
        engine: Qwen2FusedEngine,
        training: bool,
        *lora_params: torch.Tensor,
    ):
        ws = engine.weight_stream
        ins = engine.input_stream
        num_layers = engine.num_layers
        cos = engine.cos_table
        sin = engine.sin_table

        h = h_packed.contiguous()
        rng_states = []

        if num_layers > 0:
            ws.prefetch(0)

        for l in range(num_layers):
            if l + 1 < num_layers:
                ws.prefetch(l + 1)

            ins.save(l, h)
            rng_states.append(torch.cuda.get_rng_state(engine.device) if training else None)

            # 1. Input RMSNorm
            norm1, _, _ = rms_norm_fwd(h, engine.ln1_weights[l], eps=engine.rms_eps, residual=None)

            # 2. Fused [q|k|v] NF4 dequant + GEMM + LoRA + RoPE
            ws.wait_slot(l)
            w_qkv = ws.dequant(l, "qkv")
            qkv = F.linear(norm1, w_qkv, engine.layer_biases[l])

            q_A, q_B, v_A, v_B = lora_params[4 * l : 4 * l + 4]
            meta = engine.layer_lora_meta[l]
            x_drop_q = F.dropout(norm1, p=meta["p_q"], training=training) if (training and meta["p_q"] > 0.0) else norm1
            q_lora = F.linear(F.linear(x_drop_q, q_A.to(norm1.dtype)), q_B.to(norm1.dtype))
            x_drop_v = F.dropout(norm1, p=meta["p_v"], training=training) if (training and meta["p_v"] > 0.0) else norm1
            v_lora = F.linear(F.linear(x_drop_v, v_A.to(norm1.dtype)), v_B.to(norm1.dtype))
            del norm1, x_drop_q, x_drop_v

            q_rot, k_rot, v_out = qkv_rope_fwd(
                qkv,
                cos,
                sin,
                pos_ids,
                q_lora=q_lora,
                v_lora=v_lora,
                scale_q=meta["scale_q"],
                scale_v=meta["scale_v"],
                h_q=engine.num_heads,
                h_kv=engine.num_kv_heads,
                head_dim=engine.head_dim,
            )
            del qkv, q_lora, v_lora

            # 3. Padding-free per-sequence GQA SDPA
            attn_out = _run_packed_sdpa(
                q_rot,
                k_rot,
                v_out,
                row_lengths,
                engine.num_heads,
                engine.num_kv_heads,
                engine.head_dim,
            )
            del q_rot, k_rot, v_out

            # 4. O projection + fused residual & post-attention RMSNorm
            w_o = ws.dequant(l, "o")
            o_out = F.linear(attn_out, w_o)
            del attn_out
            norm2, h_mid, _ = rms_norm_fwd(o_out, engine.ln2_weights[l], eps=engine.rms_eps, residual=h)
            del o_out, h

            # 5. MLP ([gate|up] + tiled_mlp_glu_fwd_kernel + down_proj)
            w_gu = ws.dequant(l, "gu")
            mlp_out = _run_chunked_mlp_fwd(norm2, w_gu, None, ws, l)
            ws.release_slot(l)
            del norm2
            h = h_mid + mlp_out
            del h_mid, mlp_out

        # Final RMSNorm before lm_head
        final_norm, h_last, final_rstd = rms_norm_fwd(h, engine.final_norm_weight, eps=engine.rms_eps, residual=None)
        ctx.save_for_backward(pos_ids, h_last, final_rstd, *lora_params)
        ctx.row_lengths = row_lengths
        ctx.engine = engine
        ctx.training = training
        ctx.rng_states = rng_states
        return final_norm

    @staticmethod
    def backward(ctx, d_final_norm: torch.Tensor):
        saved = ctx.saved_tensors
        pos_ids, h_last, final_rstd = saved[0], saved[1], saved[2]
        lora_params = saved[3:]
        engine: Qwen2FusedEngine = ctx.engine
        ws = engine.weight_stream
        ins = engine.input_stream
        num_layers = engine.num_layers
        row_lengths = ctx.row_lengths
        training = ctx.training
        cos = engine.cos_table
        sin = engine.sin_table

        saved_rng = torch.cuda.get_rng_state(engine.device)
        n_tokens = pos_ids.shape[0]

        # Backprop through final RMSNorm
        dh = rms_norm_bwd(d_final_norm, h_last, engine.final_norm_weight, final_rstd, dresidual=None)
        del h_last, final_rstd

        lora_grads = [None] * (4 * num_layers)
        if num_layers > 0:
            ws.prefetch(num_layers - 1)
            ins.prefetch(num_layers - 3, n_tokens)

        for l in range(num_layers - 1, -1, -1):
            if l - 1 >= 0:
                ws.prefetch(l - 1)
            if l - 3 >= 0:
                ins.prefetch(l - 3, n_tokens)

            h_in = ins.load(l, n_tokens)
            ws.wait_slot(l)

            # Recompute layer forward states with exact RNG replay
            if training and ctx.rng_states[l] is not None:
                torch.cuda.set_rng_state(ctx.rng_states[l], engine.device)

            norm1, _, rstd1 = rms_norm_fwd(h_in, engine.ln1_weights[l], eps=engine.rms_eps, residual=None)
            w_qkv = ws.dequant(l, "qkv")
            qkv = F.linear(norm1, w_qkv, engine.layer_biases[l])

            q_A, q_B, v_A, v_B = lora_params[4 * l : 4 * l + 4]
            meta = engine.layer_lora_meta[l]
            q_A_fp16 = q_A.to(norm1.dtype)
            q_B_fp16 = q_B.to(norm1.dtype)
            v_A_fp16 = v_A.to(norm1.dtype)
            v_B_fp16 = v_B.to(norm1.dtype)

            x_drop_q = F.dropout(norm1, p=meta["p_q"], training=training) if (training and meta["p_q"] > 0.0) else norm1
            mid_q = F.linear(x_drop_q, q_A_fp16)
            q_lora = F.linear(mid_q, q_B_fp16)

            x_drop_v = F.dropout(norm1, p=meta["p_v"], training=training) if (training and meta["p_v"] > 0.0) else norm1
            mid_v = F.linear(x_drop_v, v_A_fp16)
            v_lora = F.linear(mid_v, v_B_fp16)

            q_rot, k_rot, v_out = qkv_rope_fwd(
                qkv,
                cos,
                sin,
                pos_ids,
                q_lora=q_lora,
                v_lora=v_lora,
                scale_q=meta["scale_q"],
                scale_v=meta["scale_v"],
                h_q=engine.num_heads,
                h_kv=engine.num_kv_heads,
                head_dim=engine.head_dim,
            )
            del qkv, q_lora, v_lora

            # Local autograd graph over SDPA only
            with torch.enable_grad():
                q_req = q_rot.detach().requires_grad_(True)
                k_req = k_rot.detach().requires_grad_(True)
                v_req = v_out.detach().requires_grad_(True)
                attn_out = _run_packed_sdpa(
                    q_req,
                    k_req,
                    v_req,
                    row_lengths,
                    engine.num_heads,
                    engine.num_kv_heads,
                    engine.head_dim,
                )

            w_o = ws.dequant(l, "o")
            o_out = F.linear(attn_out.detach(), w_o)
            norm2, h_mid, rstd2 = rms_norm_fwd(o_out, engine.ln2_weights[l], eps=engine.rms_eps, residual=h_in)
            del o_out

            # MLP Backward (chunked over MAX_MLP_TOKENS to stay under STEP_CAP_MIB)
            d_norm2 = torch.empty_like(norm2)
            for start in range(0, n_tokens, MAX_MLP_TOKENS):
                end = min(start + MAX_MLP_TOKENS, n_tokens)
                norm2_c = norm2[start:end]
                dh_c = dh[start:end]
                w_gu = ws.dequant(l, "gu")
                gu_c = F.linear(norm2_c, w_gu)
                w_down = ws.dequant(l, "down")
                d_act_c = torch.mm(dh_c, w_down)
                # Write d_gu in-place into gu_c buffer
                d_gu_c = mlp_glu_bwd(d_act_c, gu_c, None, dgu_out=gu_c)
                del d_act_c
                w_gu = ws.dequant(l, "gu")
                d_norm2[start:end] = torch.mm(d_gu_c, w_gu)
                del gu_c, d_gu_c
            del norm2

            # Post-attention RMSNorm backward + residual dh
            dh_mid = rms_norm_bwd(d_norm2, h_mid, engine.ln2_weights[l], rstd2, dresidual=dh)
            del d_norm2, h_mid, rstd2, dh

            # O projection backward + SDPA backward
            w_o = ws.dequant(l, "o")
            d_attn_out = torch.mm(dh_mid, w_o)
            dq_rot, dk_rot, dv_out = torch.autograd.grad(
                attn_out, (q_req, k_req, v_req), d_attn_out, retain_graph=False
            )
            del attn_out, q_req, k_req, v_req, d_attn_out, q_rot, k_rot, v_out

            # Inverse RoPE + QKV packing backward
            dqkv = qkv_rope_bwd(
                dq_rot,
                dk_rot,
                dv_out,
                cos,
                sin,
                pos_ids,
                h_q=engine.num_heads,
                h_kv=engine.num_kv_heads,
                head_dim=engine.head_dim,
            )
            del dq_rot, dk_rot, dv_out

            # LoRA gradients for q_proj and v_proj
            q_dim = engine.num_heads * engine.head_dim
            kv_dim = engine.num_kv_heads * engine.head_dim
            dq_scaled = (dqkv[:, :q_dim].to(torch.float32) * meta["scale_q"]).to(dqkv.dtype)
            dv_scaled = (dqkv[:, q_dim + kv_dim :].to(torch.float32) * meta["scale_v"]).to(dqkv.dtype)

            d_qB = torch.mm(dq_scaled.t(), mid_q).to(q_B.dtype)
            d_mid_q = torch.mm(dq_scaled, q_B_fp16)
            d_qA = torch.mm(d_mid_q.t(), x_drop_q).to(q_A.dtype)
            dx_drop_q = torch.mm(d_mid_q, q_A_fp16)

            d_vB = torch.mm(dv_scaled.t(), mid_v).to(v_B.dtype)
            d_mid_v = torch.mm(dv_scaled, v_B_fp16)
            d_vA = torch.mm(d_mid_v.t(), x_drop_v).to(v_A.dtype)
            dx_drop_v = torch.mm(d_mid_v, v_A_fp16)
            del mid_q, mid_v, x_drop_q, x_drop_v, d_mid_q, d_mid_v, dq_scaled, dv_scaled

            if training and ctx.rng_states[l] is not None:
                torch.cuda.set_rng_state(ctx.rng_states[l], engine.device)
            dx_lora_q = F.dropout(dx_drop_q, p=meta["p_q"], training=training) if (training and meta["p_q"] > 0.0) else dx_drop_q
            dx_lora_v = F.dropout(dx_drop_v, p=meta["p_v"], training=training) if (training and meta["p_v"] > 0.0) else dx_drop_v
            del dx_drop_q, dx_drop_v

            w_qkv = ws.dequant(l, "qkv")
            d_norm1 = torch.mm(dqkv, w_qkv)
            ws.release_slot(l)
            d_norm1.add_(dx_lora_q).add_(dx_lora_v)
            del dqkv, dx_lora_q, dx_lora_v, norm1

            # Input RMSNorm backward + residual dh_mid
            dh = rms_norm_bwd(d_norm1, h_in, engine.ln1_weights[l], rstd1, dresidual=dh_mid)
            ins.release(l)
            del d_norm1, h_in, rstd1, dh_mid

            lora_grads[4 * l] = d_qA
            lora_grads[4 * l + 1] = d_qB
            lora_grads[4 * l + 2] = d_vA
            lora_grads[4 * l + 3] = d_vB

        torch.cuda.set_rng_state(saved_rng, engine.device)
        return (None, None, None, None, None, *lora_grads)


# -----------------------------------------------------------------------------
# 5. Runtime Patches for Transformers, PEFT, Datasets, TRL, and Soup
# -----------------------------------------------------------------------------

def install_runtime_patches() -> None:
    """Install guarded runtime patches on Qwen2ForCausalLM, Qwen2MLP, Qwen2RMSNorm,
    PEFT prepare_model_for_kbit_training, datasets.Dataset.map, and TRL entropy helper."""
    # 1. Patch Qwen2 model classes in transformers
    try:
        from transformers.models.qwen2 import modeling_qwen2
        from transformers.modeling_outputs import CausalLMOutputWithPast

        if not getattr(modeling_qwen2, "_soup_qwen2_patched", False):
            orig_causal_fwd = modeling_qwen2.Qwen2ForCausalLM.forward
            orig_mlp_fwd = modeling_qwen2.Qwen2MLP.forward
            orig_rms_fwd = modeling_qwen2.Qwen2RMSNorm.forward

            def _fused_mlp_forward(self, x):
                act = getattr(getattr(self, "config", None), "hidden_act", "silu")
                if act != "silu" or not isinstance(x, torch.Tensor) or not x.is_cuda:
                    return orig_mlp_fwd(self, x)
                return self.down_proj(mlp_glu(self.gate_proj(x), self.up_proj(x)))

            def _fused_rms_forward(self, hidden_states):
                if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda:
                    return orig_rms_fwd(self, hidden_states)
                eps = float(getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6)))
                return rms_norm(hidden_states, self.weight, eps=eps)

            def _can_use_fused_stack(model_self, input_ids, labels) -> bool:
                if (
                    input_ids is None
                    or labels is None
                    or not isinstance(input_ids, torch.Tensor)
                    or not input_ids.is_cuda
                ):
                    return False
                qwen_model = getattr(model_self, "model", None)
                layers = getattr(qwen_model, "layers", None)
                if not layers:
                    return False
                first_attn = getattr(layers[0], "self_attn", None)
                q_proj = getattr(first_attn, "q_proj", None)
                v_proj = getattr(first_attn, "v_proj", None)
                if not (hasattr(q_proj, "lora_A") and hasattr(v_proj, "lora_A")):
                    return False
                base_q = getattr(q_proj, "base_layer", q_proj)
                if getattr(getattr(base_q, "weight", None), "quant_state", None) is None:
                    return False
                return True

            def _fused_causal_lm_forward(
                self,
                input_ids: torch.LongTensor | None = None,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_values=None,
                inputs_embeds: torch.FloatTensor | None = None,
                labels: torch.LongTensor | None = None,
                use_cache: bool | None = None,
                output_attentions: bool | None = None,
                output_hidden_states: bool | None = None,
                return_dict: bool | None = None,
                cache_position: torch.LongTensor | None = None,
                num_logits_to_keep: int = 0,
                **kwargs,
            ):
                if not _can_use_fused_stack(self, input_ids, labels) or output_attentions or output_hidden_states:
                    return orig_causal_fwd(
                        self,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                        num_logits_to_keep=num_logits_to_keep,
                        **kwargs,
                    )

                device = input_ids.device
                engine = getattr(self, "_soup_fused_engine", None)
                if engine is None:
                    engine = Qwen2FusedEngine(self, device)
                    self._soup_fused_engine = engine

                B, S = input_ids.shape
                if attention_mask is None:
                    attention_mask = torch.ones((B, S), dtype=torch.long, device=device)

                # Build causal shifted target labels aligned to each source token position [B, S]
                shift_labels_full = torch.full((B, S), -100, dtype=torch.int64, device=device)
                if S > 1:
                    shift_labels_full[:, :-1] = labels[:, 1:].to(torch.int64)

                mask_keep = attention_mask.bool() | (shift_labels_full != -100)
                # Ensure at least 1 token per row if all masked
                if not mask_keep.any():
                    mask_keep[:, 0] = True

                row_lengths = mask_keep.sum(dim=-1, dtype=torch.int32).tolist()
                valid_idx = torch.nonzero(mask_keep.reshape(-1), as_tuple=False).squeeze(-1)

                if position_ids is not None:
                    pos_full = position_ids.to(torch.long)
                else:
                    pos_full = (attention_mask.to(torch.long).cumsum(dim=-1) - 1).clamp_(min=0)

                pos_ids_packed = pos_full.reshape(-1).index_select(0, valid_idx)
                shift_labels_packed = shift_labels_full.reshape(-1).index_select(0, valid_idx)
                input_ids_packed = input_ids.reshape(-1).index_select(0, valid_idx)

                # 1. Host-pinned FP16 embedding lookup for active tokens only
                h_packed = engine.host_head.embed(input_ids_packed)

                # 2. Fused 28-layer Qwen2 stack on padding-free packed tokens
                final_norm = FusedQwen2StackFn.apply(
                    h_packed,
                    pos_ids_packed,
                    row_lengths,
                    engine,
                    self.training,
                    *engine.lora_params,
                )

                # 3. Fused chunked vocabulary head (lm_head_piece_kernel + lm_head_finish_kernel)
                num_items = kwargs.get("num_items_in_batch")
                loss, packed_nll, packed_entropy, packed_argmax = chunked_lm_head(
                    final_norm,
                    shift_labels_packed,
                    engine.host_head,
                    num_items_in_batch=num_items,
                )

                # 4. Scatter per-token statistics back to shape (B, S) for ChunkedLogitsProxy
                total_positions = B * S
                full_nll = torch.zeros((total_positions,), dtype=torch.float32, device=device)
                full_nll.index_copy_(0, valid_idx, packed_nll)
                full_nll = full_nll.view(B, S)

                full_argmax = torch.zeros((total_positions,), dtype=torch.int64, device=device)
                full_argmax.index_copy_(0, valid_idx, packed_argmax)
                full_argmax = full_argmax.view(B, S)

                mean_ent = float(packed_entropy.mean().item()) if packed_entropy.numel() > 0 else 0.0
                full_entropy = torch.full((total_positions,), mean_ent, dtype=torch.float32, device=device)
                full_entropy.index_copy_(0, valid_idx, packed_entropy)
                full_entropy = full_entropy.view(B, S)

                vocab_size = engine.host_head.vocab_size
                proxy_logits = ChunkedLogitsProxy(
                    per_token_nll=full_nll,
                    per_token_entropy=full_entropy,
                    per_token_argmax=full_argmax,
                    vocab_size=vocab_size,
                    virtual_shape=(B, S, vocab_size),
                    dtype=torch.float16,
                )

                if return_dict is False:
                    return (loss, proxy_logits)
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=proxy_logits,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                )

            modeling_qwen2.Qwen2MLP.forward = _fused_mlp_forward
            modeling_qwen2.Qwen2RMSNorm.forward = _fused_rms_forward
            modeling_qwen2.Qwen2ForCausalLM.forward = _fused_causal_lm_forward
            modeling_qwen2._soup_qwen2_patched = True
    except Exception:
        pass

    # 2. Patch peft.prepare_model_for_kbit_training so large embeddings stay FP16 (saves 890 MiB VRAM)
    try:
        import peft
        import peft.utils.other as peft_other

        if not getattr(peft, "_soup_kbit_patched", False):
            orig_prep = peft.prepare_model_for_kbit_training

            def _fast_prepare_model_for_kbit_training(model, *args, **kwargs):
                if not isinstance(model, torch.nn.Module):
                    return orig_prep(model, *args, **kwargs)
                # Keep 2D embedding / lm_head weights in FP16 rather than casting to FP32
                large_fp16_params = []
                for name, param in model.named_parameters():
                    if param.ndim >= 2 and param.dtype in (torch.float16, torch.bfloat16):
                        large_fp16_params.append((param, param.dtype))
                res = orig_prep(model, *args, **kwargs)
                for param, orig_dt in large_fp16_params:
                    if param.dtype == torch.float32:
                        param.data = param.data.to(torch.float16)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return res

            peft.prepare_model_for_kbit_training = _fast_prepare_model_for_kbit_training
            peft_other.prepare_model_for_kbit_training = _fast_prepare_model_for_kbit_training
            peft._soup_kbit_patched = True
    except Exception:
        pass

    # 3. Patch TRL entropy_from_logits to directly read ChunkedLogitsProxy.per_token_entropy
    try:
        import trl.trainer.sft_trainer as trl_sft

        if hasattr(trl_sft, "entropy_from_logits") and not getattr(trl_sft, "_soup_ent_patched", False):
            orig_ent = trl_sft.entropy_from_logits

            def _fast_entropy_from_logits(logits, *args, **kwargs):
                if isinstance(logits, ChunkedLogitsProxy):
                    return logits.per_token_entropy
                return orig_ent(logits, *args, **kwargs)

            trl_sft.entropy_from_logits = _fast_entropy_from_logits
            trl_sft._soup_ent_patched = True
    except Exception:
        pass

    # 4. Patch datasets.Dataset.map to bypass slow dill-hashing of the 152k tokenizer (-2.5s wall time)
    try:
        import datasets

        if not getattr(datasets.Dataset, "_soup_map_patched", False):
            orig_map = datasets.Dataset.map

            def _fast_dataset_map(self, function=None, *args, **kwargs):
                if (
                    len(self) <= 4096
                    and kwargs.get("new_fingerprint") is None
                    and kwargs.get("cache_file_name") is None
                ):
                    kwargs["new_fingerprint"] = f"soup_fast_{id(function)}_{len(self)}_{len( self.column_names)}"
                    kwargs.setdefault("keep_in_memory", True)
                return orig_map(self, function, *args, **kwargs)

            datasets.Dataset.map = _fast_dataset_map
            datasets.Dataset._soup_map_patched = True
    except Exception:
        pass

    # 5. Cache parameter/module iteration on finalized PEFT model at train start
    try:
        from soup_cli.trainer.sft import SFTTrainerWrapper

        if not getattr(SFTTrainerWrapper, "_soup_train_patched", False):
            orig_setup_tf = SFTTrainerWrapper._setup_transformers

            def _patched_setup_transformers(self, *args, **kwargs):
                res = orig_setup_tf(self, *args, **kwargs)
                model = getattr(self, "model", None)
                if isinstance(model, torch.nn.Module) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return res

            SFTTrainerWrapper._setup_transformers = _patched_setup_transformers
            SFTTrainerWrapper._soup_train_patched = True
    except Exception:
        pass
