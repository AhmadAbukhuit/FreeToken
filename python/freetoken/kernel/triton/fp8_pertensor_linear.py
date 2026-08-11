"""Per-tensor (per-output-row) FP8 W8A16 dense linear, shared across models.

The nvidia/modelopt ``MIXED_PRECISION`` checkpoints (Qwen3.5/3.6, ...) quantize the attention
and GatedDeltaNet projections to **per-tensor FP8** (``weight`` fp8-e4m3 + a single scalar
``weight_scale``; ``weight_bf16 = weight_fp8 * weight_scale``) while the routed experts are
NVFP4. FreeToken used to *dequantize these dense weights to bf16 at load* and run them through
cuBLAS bf16 GEMV -- which is the dominant decode cost (~54% of the step) because it reads
2x the bytes of the resident FP8 weight. This module keeps the weight FP8 and reads it
directly in a W8A16 kernel (bf16 activation, fp8 weight), halving that traffic.

Fused projections (q/k/v -> ``qkv_proj``; GDN qkv|z -> ``in_proj_qkvz``) concatenate parts
that each carry their *own* per-tensor scalar, so the scalar is broadcast to a per-output-row
vector ``weight_scale`` ``[N]`` (piecewise-constant after fusion) and applied per output row.
This is exactly equal to applying each part's scalar to its own rows.

Numerics: the kernel converts fp8->f32 and accumulates in fp32 (standard W8A16; strictly more
accurate than the previous ``weight.to(bf16) * scale`` materialization, which it replaces).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from freetoken.layers import BaseOP

from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32

FP8 = torch.float8_e4m3fn
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Escape hatch: FREETOKEN_DEBUG_FP8_REF=1 swaps the triton kernels for a pure-torch dequant matmul
# (numeric reference / A-B debugging). Evaluated once; the kernels are the default.
_USE_REF = os.environ.get("FREETOKEN_DEBUG_FP8_REF") == "1"


# ======================================================================================
# Decode (M==1) split-K GEMV: raw fp8 x bf16 reduction in fp32, per-row scale at reduce.
# ======================================================================================
@triton.jit
def _gemv_splitk_kernel(
    a_ptr, w_ptr, part_ptr, N, K, n_kb, kb_per,
    stride_ak, stride_wn, stride_wk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Each (pid_n, pid_k) computes the partial sum over ``kb_per`` BLOCK_K chunks for a
    BLOCK_N slice of outputs. ``kb_per`` ceil-tiles K so K only needs to be a multiple of
    BLOCK_K is *not* required (loads are k-masked). Per-row scale is applied in the reduce."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            acc += tl.sum(w * a[None, :], axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _splitk_reduce_kernel(
    part_ptr, scale_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
    stride_pk, stride_pn, BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + offs * stride_pn, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (acc * scale).to(OUT), mask=mask)


def _gemv(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """M==1 split-K GEMV. ``a`` [K] bf16; ``weight`` [N, K] fp8; ``weight_scale`` [N] fp32."""
    N, K = weight.shape
    BLOCK_K = 128
    n_kb = triton.cdiv(K, BLOCK_K)
    BLOCK_N = 16
    n_tiles = triton.cdiv(N, BLOCK_N)
    split_k = max(1, min(1536 // n_tiles, n_kb))
    split_k = 1 << (split_k.bit_length() - 1)  # pow2 -> stable reduction order
    kb_per = triton.cdiv(n_kb, split_k)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a.device)
    _gemv_splitk_kernel[(n_tiles, split_k)](
        a, weight, part, N, K, n_kb, kb_per,
        a.stride(0), weight.stride(0), weight.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=1,
    )
    out = torch.empty(N, dtype=out_dtype, device=a.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, weight_scale, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16],
        num_warps=2,
    )
    return out


# ======================================================================================
# Prefill (M>1) W8A16 GEMM: fp8 weight read from HBM, upcast to bf16 in-register for the
# tensor-core dot (fp8 e4m3 -> bf16 is lossless), per-row scale applied after accumulation.
# ======================================================================================
@triton.jit
def _gemm_kernel(
    a_ptr, w_ptr, scale_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
        w_mask = n_mask[:, None] & (offs_k[None, :] < k_rem)
        if e4m3_native_cx():
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(a.dtype)
        else:
            w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=w_mask, other=0)).to(a.dtype)
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None] & n_mask[None, :])


def _gemm(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """M>1 W8A16 GEMM. ``a`` [M, K] bf16; ``weight`` [N, K] fp8; ``weight_scale`` [N] fp32."""
    M, K = a.shape
    N = weight.shape[0]
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    out = torch.empty((M, N), dtype=compute, device=a.device)
    BLOCK_M = 64 if M >= 64 else 32
    BLOCK_N, BLOCK_K = 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_kernel[grid](
        a, weight, weight_scale, out, M, N, K,
        a.stride(0), a.stride(1), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, compute_type=_TL_DTYPE[compute],
        num_warps=8 if M >= 64 else 4, num_stages=3,
    )
    return out


def fp8_pertensor_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ (weight_fp8 * weight_scale)^T``. Decode (M=1) -> split-K GEMV; prefill -> GEMM.
    ``weight`` [N, K] fp8-e4m3, ``weight_scale`` [N] fp32 (per output row)."""
    *lead, K = x.shape
    N = weight.shape[0]
    if _USE_REF:  # numeric-reference fallback (debug / A-B)
        w = weight.to(x.dtype) * weight_scale.to(x.dtype)[:, None]
        out = (x.reshape(-1, K) @ w.t()).reshape(*lead, N)
    else:
        w8 = e4m3_kernel_view(weight)
        if x.numel() // K == 1:
            out = _gemv(x.reshape(K), w8, weight_scale, x.dtype).reshape(*lead, N)
        else:
            out = _gemm(x.reshape(-1, K), w8, weight_scale, x.dtype).reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


# ======================================================================================
# BaseOP linear layers (TP=1, replicated). Buffers: fp8 ``weight`` + fp32 ``weight_scale``.
# ======================================================================================
class Fp8PerTensorLinear(BaseOP):
    """Replicated per-tensor-FP8 linear: fp8-e4m3 ``weight`` ``[out, in]`` + per-row fp32
    ``weight_scale`` ``[out]`` (a genuine per-tensor weight stores the same scalar in every
    row; a fused projection stores each part's scalar across its own rows)."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features, dtype=FP8)
        self.weight_scale = torch.empty(out_features, dtype=torch.float32)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_pertensor_linear(x, self.weight, self.weight_scale, self.bias)


class Fp8PerTensorColMerged(Fp8PerTensorLinear):
    """Column-merged per-tensor-FP8 linear (drop-in for ``LinearColParallelMerged`` at TP=1):
    one fp8 weight concatenating several projections along the output dim; the caller splits
    the bf16 output by ``output_sizes`` as before."""

    def __init__(self, in_features: int, output_sizes: list[int], has_bias: bool = False):
        self.output_sizes = list(output_sizes)
        super().__init__(in_features, sum(output_sizes), has_bias)


__all__ = [
    "FP8",
    "Fp8PerTensorLinear",
    "Fp8PerTensorColMerged",
    "fp8_pertensor_linear",
]
