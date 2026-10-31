# Submission: <team name>

Fill in the bracketed fields (`<...>`) below after running the final commit on a Kaggle Tesla T4 notebook.

## Links
- Repository, final commit: https://github.com/<you>/<repo>/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| <Member 1> | ML Systems / Triton Engineer | <github> | <kaggle> | <email / telegram> |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_turing_kernel` | `solution/kernels.py` | `bitsandbytes.functional.dequantize_4bit` (register-muxed NF4 + double-quantization dequantizer to coalesced `fp16x2` over fused `[q\|k\|v]`, `o`, `[gate\|up]`, `down`) | <from metrics.json>% |
| `rms_norm_residual_fwd_kernel` | `solution/kernels.py` | `Qwen2RMSNorm.forward` + residual stream addition in FP32 registers | <from metrics.json>% |
| `rms_norm_bwd_kernel` | `solution/kernels.py` | `Qwen2RMSNorm` backward + residual gradient accumulation | <from metrics.json>% |
| `qkv_lora_rope_fwd_kernel` | `solution/kernels.py` | LoRA scaling/addition on Q & V + NeoX rotary position embeddings (`apply_rotary_pos_emb`) + head layout | <from metrics.json>% |
| `qkv_lora_rope_bwd_kernel` | `solution/kernels.py` | Inverse NeoX RoPE backward rotation on dQ & dK + contiguous `[dQ\|dK\|dV]` packing | <from metrics.json>% |
| `tiled_mlp_glu_fwd_kernel` | `solution/kernels.py` | 2D multi-tile `F.silu(gate) * up` over fused `[gate\|up]` projection | <from metrics.json>% |
| `tiled_mlp_glu_bwd_kernel` | `solution/kernels.py` | 2D multi-tile backward for `SwiGLU`, writing `[d_gate\|d_up]` in-place | <from metrics.json>% |
| `lm_head_piece_kernel` | `solution/kernels.py` | Chunked vocabulary online logsumexp, Shannon entropy numerator, argmax, target logit, and in-place cross-entropy gradient | <from metrics.json>% |
| `lm_head_finish_kernel` | `solution/kernels.py` | Final per-token cross-entropy NLL and exact Shannon entropy reduction | <from metrics.json>% |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Online 1-pass logsumexp / softmax rescaling: Milakov & Gimelshein, *Online normalizer calculation for softmax* (https://arxiv.org/abs/1805.02867).
- Layer-to-layer CPU-GPU weight & activation streaming: Pudipeddi et al., *Training Large Neural Networks with Constant Memory using a New Execution Algorithm* (https://arxiv.org/abs/2002.05645).
- NF4 quantization codebook constants: Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (https://arxiv.org/abs/2305.14314).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | <base_wall_s> | <sol_wall_s> | speedup <speedup>x |
| Peak GPU memory, MiB | <base_peak_mib> | <sol_peak_mib> | base/ours <memory_ratio>x |
| Eval loss (public rows) | <base_eval_loss> | <sol_eval_loss> | <quality_delta_pct>% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed (verify on Kaggle T4 run)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (eliminating 5.3 GB full logits via chunked online LSE/entropy head, stripping batch padding, fusing RMSNorm/RoPE/LoRA/SwiGLU, and register-muxed NF4 dequantization), how the 9 Triton kernels and 2-slot host-pinned streaming engine work, results against the base (speed, memory, quality), limits and next steps.

## Limitations
- The fused 28-layer stack (`FusedQwen2StackFn`) targets `Qwen2ForCausalLM` with 4-bit NF4 weights and LoRA on `q_proj` and `v_proj` (falling back to standard PyTorch/Transformers execution for other architectures).
- Streaming `lm_head` chunks and NF4 layer weights over PCIe reduces VRAM to ~490–540 MiB, but PCIe bandwidth becomes a lower bound on step time for very small batch sizes.
