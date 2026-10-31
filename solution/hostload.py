"""Host-pinned weight & activation streaming engine (hostload.py).

Implements:
- Immediate NF4 layer weight & FP16 embedding offloading during `from_pretrained` (prevents GPU model-load spike).
- `WeightStream`: Packs each Qwen2 decoder layer's 7 NF4 projections into 4 fused segments
  (`[q|k|v]`, `o`, `[gate|up]`, `down`) in pinned CPU memory with 2 rotating GPU slots (~46 MiB)
  and a single reusable 52.5 MiB FP16 dequantization buffer on GPU.
- `HostHead`: Keeps `embed_tokens` / `lm_head` ([151936, 1536] FP16 = 445 MiB) in pinned CPU memory,
  serves active token embeddings via direct CPU gather + async H2D copy, and streams vocabulary chunks
  (`V_CHUNK = 16384`) through 2 rotating GPU slots (~48-96 MiB total) for `lm_head_piece_kernel`.
- `InputStream` / `GuStream`: Asynchronous H2D/D2H activation streaming with `STEP_CAP_MIB = 396`
  peak memory budgeting.
"""

from __future__ import annotations

import os
import torch

from .kernels import dequant_nf4_tensor

STEP_CAP_MIB = int(os.environ.get("SOUP_STEP_CAP_MIB", "396"))
MAX_MLP_TOKENS = 512
V_CHUNK_DEFAULT = 8192  # 8192 * 1536 * 2B = 24 MiB per slot (48 MiB for 2 slots!)


class PackedNF4Segment:
    """Metadata and slices for one fused NF4 segment ([q|k|v], o, [gate|up], or down) inside a layer."""

    __slots__ = (
        "out_features",
        "in_features",
        "packed_slice",
        "absmax_slice",
        "absmax2_slice",
        "offsets",
        "split_bytes",
        "blocksize",
        "super_blocksize",
    )

    def __init__(
        self,
        out_features: int,
        in_features: int,
        packed_slice: slice,
        absmax_slice: slice,
        absmax2_slice: slice,
        offsets: tuple[float, float, float],
        split_bytes: tuple[int, int],
        blocksize: int = 64,
        super_blocksize: int = 256,
    ):
        self.out_features = out_features
        self.in_features = in_features
        self.packed_slice = packed_slice
        self.absmax_slice = absmax_slice
        self.absmax2_slice = absmax2_slice
        self.offsets = offsets
        self.split_bytes = split_bytes
        self.blocksize = blocksize
        self.super_blocksize = super_blocksize


def _extract_bnb_nf4(linear_mod) -> dict:
    """Extract packed uint8 NF4 data and double-quantization state from a bitsandbytes Linear4bit
    (or PEFT LoraLayer wrapping Linear4bit)."""
    base = getattr(linear_mod, "base_layer", linear_mod)
    weight = base.weight
    qs = weight.quant_state
    packed = getattr(weight, "_cpu_packed", None)
    if packed is None:
        packed = weight.data.detach().view(-1).cpu()
    absmax = getattr(weight, "_cpu_absmax", None)
    if absmax is None:
        absmax = qs.absmax.detach().view(-1).cpu()
    state2 = qs.state2
    absmax2 = getattr(weight, "_cpu_absmax2", None)
    if absmax2 is None:
        absmax2 = state2.absmax.detach().view(-1).to(torch.float32).cpu()
    code2 = getattr(weight, "_cpu_code2", None)
    if code2 is None:
        code2 = state2.code.detach().view(-1).to(torch.float32).cpu()
    offset = float(qs.offset.item()) if isinstance(qs.offset, torch.Tensor) else float(qs.offset or 0.0)
    shape = tuple(qs.shape)
    return {
        "packed": packed,
        "absmax": absmax,
        "absmax2": absmax2,
        "code2": code2,
        "offset": offset,
        "shape": shape,
        "blocksize": int(qs.blocksize),
        "super_blocksize": int(state2.blocksize),
        "weight_param": weight,
    }


class WeightStream:
    """Streams packed NF4 layer weights between pinned CPU memory and 2 rotating GPU slots (~46 MiB total)
    on a dedicated CUDA stream, overlapping PCIe transfers with cuBLAS GEMMs and Triton kernels."""

    def __init__(self, layers, device: torch.device):
        self.device = device
        self.num_layers = len(layers)
        self.stream = torch.cuda.Stream(device=device)
        self.slot_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.compute_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.slot_layer = [-1, -1]

        self.cpu_packed: list[torch.Tensor] = []
        self.cpu_absmax: list[torch.Tensor] = []
        self.cpu_absmax2: list[torch.Tensor] = []
        self.layer_segments: list[dict[str, PackedNF4Segment]] = []
        self.layer_biases: list[torch.Tensor | None] = []
        self.code2_gpu: torch.Tensor | None = None

        max_packed_bytes = 0
        max_absmax_bytes = 0
        max_absmax2_floats = 0
        max_dequant_elems = 0

        for layer in layers:
            attn = layer.self_attn
            mlp = layer.mlp
            q_info = _extract_bnb_nf4(attn.q_proj)
            k_info = _extract_bnb_nf4(attn.k_proj)
            v_info = _extract_bnb_nf4(attn.v_proj)
            o_info = _extract_bnb_nf4(attn.o_proj)
            gate_info = _extract_bnb_nf4(mlp.gate_proj)
            up_info = _extract_bnb_nf4(mlp.up_proj)
            down_info = _extract_bnb_nf4(mlp.down_proj)

            if self.code2_gpu is None:
                self.code2_gpu = q_info["code2"].to(device=device, dtype=torch.float32).contiguous()

            q_base = getattr(attn.q_proj, "base_layer", attn.q_proj)
            k_base = getattr(attn.k_proj, "base_layer", attn.k_proj)
            v_base = getattr(attn.v_proj, "base_layer", attn.v_proj)
            if q_base.bias is not None and k_base.bias is not None and v_base.bias is not None:
                bias_qkv = torch.cat(
                    [
                        q_base.bias.detach().to(device=device, dtype=torch.float16),
                        k_base.bias.detach().to(device=device, dtype=torch.float16),
                        v_base.bias.detach().to(device=device, dtype=torch.float16),
                    ],
                    dim=0,
                ).contiguous()
            else:
                bias_qkv = None
            self.layer_biases.append(bias_qkv)

            packed_parts = []
            absmax_parts = []
            absmax2_parts = []
            p_cur = a_cur = a2_cur = 0
            segs: dict[str, PackedNF4Segment] = {}

            # 1. Fused [q | k | v]
            n_q = q_info["packed"].numel()
            n_k = k_info["packed"].numel()
            n_v = v_info["packed"].numel()
            n_qkv = n_q + n_k + n_v
            na_qkv = q_info["absmax"].numel() + k_info["absmax"].numel() + v_info["absmax"].numel()
            na2_qkv = q_info["absmax2"].numel() + k_info["absmax2"].numel() + v_info["absmax2"].numel()
            packed_parts.extend([q_info["packed"], k_info["packed"], v_info["packed"]])
            absmax_parts.extend([q_info["absmax"], k_info["absmax"], v_info["absmax"]])
            absmax2_parts.extend([q_info["absmax2"], k_info["absmax2"], v_info["absmax2"]])
            out_qkv = q_info["shape"][0] + k_info["shape"][0] + v_info["shape"][0]
            in_qkv = q_info["shape"][1]
            segs["qkv"] = PackedNF4Segment(
                out_features=out_qkv,
                in_features=in_qkv,
                packed_slice=slice(p_cur, p_cur + n_qkv),
                absmax_slice=slice(a_cur, a_cur + na_qkv),
                absmax2_slice=slice(a2_cur, a2_cur + na2_qkv),
                offsets=(q_info["offset"], k_info["offset"], v_info["offset"]),
                split_bytes=(n_q, n_q + n_k),
                blocksize=q_info["blocksize"],
                super_blocksize=q_info["super_blocksize"],
            )
            p_cur += n_qkv
            a_cur += na_qkv
            a2_cur += na2_qkv
            max_dequant_elems = max(max_dequant_elems, out_qkv * in_qkv)

            # 2. o_proj
            n_o = o_info["packed"].numel()
            na_o = o_info["absmax"].numel()
            na2_o = o_info["absmax2"].numel()
            packed_parts.append(o_info["packed"])
            absmax_parts.append(o_info["absmax"])
            absmax2_parts.append(o_info["absmax2"])
            segs["o"] = PackedNF4Segment(
                out_features=o_info["shape"][0],
                in_features=o_info["shape"][1],
                packed_slice=slice(p_cur, p_cur + n_o),
                absmax_slice=slice(a_cur, a_cur + na_o),
                absmax2_slice=slice(a2_cur, a2_cur + na2_o),
                offsets=(o_info["offset"], o_info["offset"], o_info["offset"]),
                split_bytes=(n_o, n_o),
                blocksize=o_info["blocksize"],
                super_blocksize=o_info["super_blocksize"],
            )
            p_cur += n_o
            a_cur += na_o
            a2_cur += na2_o
            max_dequant_elems = max(max_dequant_elems, o_info["shape"][0] * o_info["shape"][1])

            # 3. Fused [gate | up]
            n_g = gate_info["packed"].numel()
            n_u = up_info["packed"].numel()
            n_gu = n_g + n_u
            na_gu = gate_info["absmax"].numel() + up_info["absmax"].numel()
            na2_gu = gate_info["absmax2"].numel() + up_info["absmax2"].numel()
            packed_parts.extend([gate_info["packed"], up_info["packed"]])
            absmax_parts.extend([gate_info["absmax"], up_info["absmax"]])
            absmax2_parts.extend([gate_info["absmax2"], up_info["absmax2"]])
            out_gu = gate_info["shape"][0] + up_info["shape"][0]
            in_gu = gate_info["shape"][1]
            segs["gu"] = PackedNF4Segment(
                out_features=out_gu,
                in_features=in_gu,
                packed_slice=slice(p_cur, p_cur + n_gu),
                absmax_slice=slice(a_cur, a_cur + na_gu),
                absmax2_slice=slice(a2_cur, a2_cur + na2_gu),
                offsets=(gate_info["offset"], up_info["offset"], up_info["offset"]),
                split_bytes=(n_g, n_gu),
                blocksize=gate_info["blocksize"],
                super_blocksize=gate_info["super_blocksize"],
            )
            p_cur += n_gu
            a_cur += na_gu
            a2_cur += na2_gu
            max_dequant_elems = max(max_dequant_elems, out_gu * in_gu)

            # 4. down_proj
            n_d = down_info["packed"].numel()
            na_d = down_info["absmax"].numel()
            na2_d = down_info["absmax2"].numel()
            packed_parts.append(down_info["packed"])
            absmax_parts.append(down_info["absmax"])
            absmax2_parts.append(down_info["absmax2"])
            segs["down"] = PackedNF4Segment(
                out_features=down_info["shape"][0],
                in_features=down_info["shape"][1],
                packed_slice=slice(p_cur, p_cur + n_d),
                absmax_slice=slice(a_cur, a_cur + na_d),
                absmax2_slice=slice(a2_cur, a2_cur + na2_d),
                offsets=(down_info["offset"], down_info["offset"], down_info["offset"]),
                split_bytes=(n_d, n_d),
                blocksize=down_info["blocksize"],
                super_blocksize=down_info["super_blocksize"],
            )
            p_cur += n_d
            a_cur += na_d
            a2_cur += na2_d
            max_dequant_elems = max(max_dequant_elems, down_info["shape"][0] * down_info["shape"][1])

            layer_packed = torch.cat(packed_parts, dim=0).contiguous().pin_memory()
            layer_absmax = torch.cat(absmax_parts, dim=0).contiguous().pin_memory()
            layer_absmax2 = torch.cat(absmax2_parts, dim=0).contiguous().pin_memory()

            self.cpu_packed.append(layer_packed)
            self.cpu_absmax.append(layer_absmax)
            self.cpu_absmax2.append(layer_absmax2)
            self.layer_segments.append(segs)

            max_packed_bytes = max(max_packed_bytes, p_cur)
            max_absmax_bytes = max(max_absmax_bytes, a_cur)
            max_absmax2_floats = max(max_absmax2_floats, a2_cur)

            tiny_u8 = torch.empty((1,), dtype=torch.uint8, device=device)
            tiny_f32 = torch.empty((1,), dtype=torch.float32, device=device)
            for info in (q_info, k_info, v_info, o_info, gate_info, up_info, down_info):
                wp = info["weight_param"]
                wp.data = tiny_u8
                if getattr(wp, "quant_state", None) is not None:
                    wp.quant_state.absmax = tiny_u8
                    if getattr(wp.quant_state, "state2", None) is not None:
                        wp.quant_state.state2.absmax = tiny_f32
                for attr in ("_cpu_packed", "_cpu_absmax", "_cpu_absmax2", "_cpu_code2"):
                    if hasattr(wp, attr):
                        delattr(wp, attr)

        self.gpu_packed_slots = [
            torch.empty((max_packed_bytes,), dtype=torch.uint8, device=device) for _ in range(2)
        ]
        self.gpu_absmax_slots = [
            torch.empty((max_absmax_bytes,), dtype=torch.uint8, device=device) for _ in range(2)
        ]
        self.gpu_absmax2_slots = [
            torch.empty((max_absmax2_floats,), dtype=torch.float32, device=device) for _ in range(2)
        ]
        self.dequant_workspace = torch.empty((max_dequant_elems,), dtype=torch.float16, device=device)

    def prefetch(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        slot = layer_idx % 2
        if self.slot_layer[slot] == layer_idx:
            return
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.compute_done_events[slot])
            src_p = self.cpu_packed[layer_idx]
            src_a = self.cpu_absmax[layer_idx]
            src_a2 = self.cpu_absmax2[layer_idx]
            self.gpu_packed_slots[slot][: src_p.numel()].copy_(src_p, non_blocking=True)
            self.gpu_absmax_slots[slot][: src_a.numel()].copy_(src_a, non_blocking=True)
            self.gpu_absmax2_slots[slot][: src_a2.numel()].copy_(src_a2, non_blocking=True)
            self.slot_events[slot].record(self.stream)
            self.slot_layer[slot] = layer_idx

    def wait_slot(self, layer_idx: int) -> int:
        slot = layer_idx % 2
        if self.slot_layer[slot] != layer_idx:
            self.prefetch(layer_idx)
        torch.cuda.current_stream(self.device).wait_event(self.slot_events[slot])
        return slot

    def release_slot(self, layer_idx: int) -> None:
        slot = layer_idx % 2
        self.compute_done_events[slot].record(torch.cuda.current_stream(self.device))

    def dequant(self, layer_idx: int, seg_name: str) -> torch.Tensor:
        slot = layer_idx % 2
        seg = self.layer_segments[layer_idx][seg_name]
        packed = self.gpu_packed_slots[slot][seg.packed_slice]
        absmax = self.gpu_absmax_slots[slot][seg.absmax_slice]
        absmax2 = self.gpu_absmax2_slots[slot][seg.absmax2_slice]
        return dequant_nf4_tensor(
            packed=packed,
            absmax=absmax,
            code2=self.code2_gpu,
            absmax2=absmax2,
            offset=seg.offsets,
            out_shape=(seg.out_features, seg.in_features),
            out_buf=self.dequant_workspace,
            split_bytes=seg.split_bytes,
            blocksize=seg.blocksize,
            super_blocksize=seg.super_blocksize,
        )


class HostHead:
    """Keeps the frozen FP16 token embedding / tied lm_head weight ([151936, 1536], 445 MiB)
    in pinned host RAM, serving token lookups and streaming vocabulary chunks through 2 GPU slots."""

    def __init__(
        self,
        embed_module: torch.nn.Module,
        lm_head_module: torch.nn.Module,
        device: torch.device,
        v_chunk: int = V_CHUNK_DEFAULT,
    ):
        self.device = device
        self.v_chunk = v_chunk
        self.stream = torch.cuda.Stream(device=device)
        self.slot_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.compute_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.slot_chunk = [-1, -1]

        cpu_w = getattr(embed_module, "_cpu_pinned_weight", None)
        if cpu_w is None:
            cpu_w = embed_module.weight.data.detach().to(torch.float16).cpu().contiguous().pin_memory()
        self.cpu_weight = cpu_w
        self.vocab_size, self.hidden_size = cpu_w.shape

        tiny = torch.empty((1, self.hidden_size), dtype=torch.float16, device=device)
        embed_module.weight.data = tiny
        if lm_head_module is not None and lm_head_module.weight is not embed_module.weight:
            lm_head_module.weight.data = tiny
        if hasattr(embed_module, "_cpu_pinned_weight"):
            delattr(embed_module, "_cpu_pinned_weight")

        self.gpu_slots = [
            torch.empty((v_chunk, self.hidden_size), dtype=torch.float16, device=device)
            for _ in range(2)
        ]

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        orig_shape = input_ids.shape
        ids_cpu = input_ids.detach().view(-1).cpu()
        emb_cpu = self.cpu_weight.index_select(0, ids_cpu).pin_memory()
        emb_gpu = emb_cpu.to(device=self.device, dtype=torch.float16, non_blocking=True)
        return emb_gpu.view(*orig_shape, self.hidden_size)

    def prefetch_chunk(self, chunk_idx: int, start_v: int, end_v: int) -> None:
        slot = chunk_idx % 2
        if self.slot_chunk[slot] == start_v:
            return
        rows = end_v - start_v
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.compute_done_events[slot])
            self.gpu_slots[slot][:rows].copy_(self.cpu_weight[start_v:end_v], non_blocking=True)
            self.slot_events[slot].record(self.stream)
            self.slot_chunk[slot] = start_v

    def get_chunk(self, chunk_idx: int, start_v: int, end_v: int) -> torch.Tensor:
        slot = chunk_idx % 2
        if self.slot_chunk[slot] != start_v:
            self.prefetch_chunk(chunk_idx, start_v, end_v)
        torch.cuda.current_stream(self.device).wait_event(self.slot_events[slot])
        return self.gpu_slots[slot][: end_v - start_v]

    def release_chunk(self, chunk_idx: int) -> None:
        slot = chunk_idx % 2
        self.compute_done_events[slot].record(torch.cuda.current_stream(self.device))


class InputStream:
    """Streams per-layer packed input activations (`[N_valid, 1536]` FP16, ~1.5-3 MiB per layer)
    to pinned CPU memory during forward and prefetches them back in reverse order during backward."""

    def __init__(self, num_layers: int, hidden_size: int, max_tokens: int, device: torch.device):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.d2h_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.h2d_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.slot_ready_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
        self.slot_layer = [-1, -1]

        self.cpu_buffers = [
            torch.empty((max_tokens, hidden_size), dtype=torch.float16, pin_memory=True)
            for _ in range(num_layers)
        ]
        self.gpu_slots = [
            torch.empty((max_tokens, hidden_size), dtype=torch.float16, device=device)
            for _ in range(2)
        ]
        self.tail_gpu: dict[int, torch.Tensor] = {}

    def ensure_capacity(self, n_tokens: int) -> None:
        if n_tokens <= self.max_tokens:
            return
        self.max_tokens = max(n_tokens, self.max_tokens * 2)
        self.cpu_buffers = [
            torch.empty((self.max_tokens, self.hidden_size), dtype=torch.float16, pin_memory=True)
            for _ in range(self.num_layers)
        ]
        self.gpu_slots = [
            torch.empty((self.max_tokens, self.hidden_size), dtype=torch.float16, device=self.device)
            for _ in range(2)
        ]

    def save(self, layer_idx: int, x: torch.Tensor) -> None:
        n_tokens = x.shape[0]
        self.ensure_capacity(n_tokens)
        if layer_idx >= self.num_layers - 2:
            self.tail_gpu[layer_idx] = x.detach().clone()
            return
        compute_event = torch.cuda.Event(enable_timing=False)
        compute_event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(compute_event)
            self.cpu_buffers[layer_idx][:n_tokens].copy_(x.detach(), non_blocking=True)
            self.d2h_events[layer_idx].record(self.stream)

    def prefetch(self, layer_idx: int, n_tokens: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers - 2:
            return
        slot = layer_idx % 2
        if self.slot_layer[slot] == layer_idx:
            return
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.d2h_events[layer_idx])
            self.stream.wait_event(self.slot_ready_events[slot])
            self.gpu_slots[slot][:n_tokens].copy_(self.cpu_buffers[layer_idx][:n_tokens], non_blocking=True)
            self.h2d_events[slot].record(self.stream)
            self.slot_layer[slot] = layer_idx

    def load(self, layer_idx: int, n_tokens: int) -> torch.Tensor:
        if layer_idx in self.tail_gpu:
            return self.tail_gpu.pop(layer_idx)
        slot = layer_idx % 2
        if self.slot_layer[slot] != layer_idx:
            self.prefetch(layer_idx, n_tokens)
        torch.cuda.current_stream(self.device).wait_event(self.h2d_events[slot])
        return self.gpu_slots[slot][:n_tokens]

    def release(self, layer_idx: int) -> None:
        if layer_idx < self.num_layers - 2:
            slot = layer_idx % 2
            self.slot_ready_events[slot].record(torch.cuda.current_stream(self.device))
            self.slot_layer[slot] = -1


def install_hostload_hooks() -> None:
    """Install low-overhead hooks on bitsandbytes Params4bit._quantize and accelerate's
    set_module_tensor_to_device so NF4 weights and large embeddings are offloaded to host RAM
    immediately during `from_pretrained`, avoiding the 1,277+ MiB model-loading VRAM spike."""
    try:
        from bitsandbytes.nn.modules import Params4bit

        if not getattr(Params4bit, "_soup_hostload_patched", False):
            orig_quantize = Params4bit._quantize
            quant_count = [0]

            def _fast_host_quantize(self, device):
                # Convert bfloat16 -> float16 on GPU before quantizing (much faster than CPU cast)
                if self.data.device.type == "cpu" and self.data.dtype == torch.bfloat16:
                    self.data = self.data.to(device=device, non_blocking=True).to(torch.float16)
                res = orig_quantize(self, device)
                qs = getattr(res, "quant_state", None)
                if qs is not None and getattr(qs, "state2", None) is not None and res.data.is_cuda:
                    res._cpu_packed = res.data.detach().view(-1).cpu()
                    res._cpu_absmax = qs.absmax.detach().view(-1).cpu()
                    res._cpu_absmax2 = qs.state2.absmax.detach().view(-1).to(torch.float32).cpu()
                    res._cpu_code2 = qs.state2.code.detach().view(-1).to(torch.float32).cpu()
                    res.data = torch.empty((1, 1), dtype=torch.uint8, device=res.data.device)
                    qs.absmax = torch.empty((1,), dtype=torch.uint8, device=res.data.device)
                    qs.state2.absmax = torch.empty((1,), dtype=torch.float32, device=res.data.device)
                    quant_count[0] += 1
                    if quant_count[0] % 14 == 0:
                        torch.cuda.empty_cache()
                return res

            Params4bit._quantize = _fast_host_quantize
            Params4bit._soup_hostload_patched = True
    except Exception:
        pass

    try:
        import accelerate.utils.modeling as acc_modeling

        if not getattr(acc_modeling, "_soup_embed_patched", False):
            orig_set_tensor = acc_modeling.set_module_tensor_to_device

            def _patched_set_module_tensor(module, tensor_name, device, value=None, *args, **kwargs):
                if (
                    isinstance(module, torch.nn.Embedding)
                    and tensor_name == "weight"
                    and value is not None
                    and value.numel() > 50_000_000
                    and str(device).startswith("cuda")
                ):
                    # Convert bfloat16 -> float16 on GPU in 4 slices to cap peak staging to ~110 MiB,
                    # then store in pinned host RAM and leave a tiny [1, D] placeholder on GPU.
                    dev = torch.device(device if isinstance(device, str) else f"cuda:{device}")
                    rows, cols = value.shape
                    pinned = torch.empty((rows, cols), dtype=torch.float16, pin_memory=True)
                    step = 38400
                    for start in range(0, rows, step):
                        end = min(start + step, rows)
                        chunk_gpu = value[start:end].to(device=dev, non_blocking=True).to(torch.float16)
                        pinned[start:end].copy_(chunk_gpu, non_blocking=False)
                        del chunk_gpu
                    module._cpu_pinned_weight = pinned
                    placeholder = torch.empty((1, cols), dtype=torch.float16, device=dev)
                    res = orig_set_tensor(module, tensor_name, device, value=placeholder, *args, **kwargs)
                    torch.cuda.empty_cache()
                    return res
                return orig_set_tensor(module, tensor_name, device, value=value, *args, **kwargs)

            acc_modeling.set_module_tensor_to_device = _patched_set_module_tensor
            try:
                import transformers.modeling_utils as tf_mu

                if hasattr(tf_mu, "set_module_tensor_to_device"):
                    tf_mu.set_module_tensor_to_device = _patched_set_module_tensor
            except Exception:
                pass
            acc_modeling._soup_embed_patched = True
    except Exception:
        pass
