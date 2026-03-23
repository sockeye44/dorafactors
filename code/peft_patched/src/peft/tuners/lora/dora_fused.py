# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fused Triton kernels and custom autograd functions for DoRA composition.

This module provides four optimizations:
1. Fused Element-wise Composition Kernel: replaces 4 sequential element-wise ops
   with a single Triton kernel computing `out = (mag - 1) * base + mag * (scale * lora)`.
2. Fused Norm Assembly Kernel: fuses the final norm computation
   `sqrt(clamp(w_norm_sq + 2*s*cross + s^2*ba_norm_sq, min=0))` into a single kernel.
3. Fused Forward-and-Inner Kernel: dual-output kernel that computes both the
   composition output and `inner = scale * lora + base` in a single pass, used by
   the custom autograd forward to eliminate intermediate VRAM allocations.
4. Custom Autograd Function: wraps the DoRA forward composition with hand-written
   backward pass that fuses gradient computation into a single kernel.

All kernels gracefully fall back to plain PyTorch when Triton is unavailable or
when tensors are not on CUDA.

Canonical evaluation order
--------------------------
The DoRA composition formula is::

    out = (mag - 1) * base + mag * (scale * lora)

**All PyTorch paths must parenthesise the LoRA term as ``mag * (scale * lora)``.**
This means ``scale * lora`` is computed first, then multiplied by ``mag``.
The in-place path achieves this via ``lora.mul_(scale).mul_(mag)``.

This contract matters because bf16/fp16 float multiplication is not associative:
``(mag * scale) * lora`` and ``mag * (scale * lora)`` produce different rounding.
Enforcing a single order across all PyTorch paths (out-of-place, in-place, and
autograd forward-and-inner) guarantees bitwise parity between them when they
operate under the same dtype contract.  Mixed-dtype eager AMP paths additionally
materialize the promoted stable-form result and copy it back into the activation
buffer so they remain bitwise-equal to the eager out-of-place reference.
``DoraLinearLayer`` still casts ``mag`` before the fused-autograd dispatch, so
mixed-dtype eager-vs-fused parity is a separate, higher-level concern.

Triton kernels also use ``scale * lora`` first (explicit ``scaled_lora`` local),
but FMA hardware may still fuse or reorder ops, so Triton-vs-PyTorch agreement
is within O(epsilon) tolerance, not bitwise.
"""

import logging

import torch

try:  # pragma: no cover - torch._dynamo may not be available
    from torch._dynamo import is_compiling as dynamo_is_compiling
except Exception:  # pragma: no cover - older torch
    dynamo_is_compiling = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triton availability
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def is_triton_available() -> bool:
    """Return True if Triton is importable."""
    return _TRITON_AVAILABLE


# ---------------------------------------------------------------------------
# Dual-mode autotune configuration
# ---------------------------------------------------------------------------
# Default mode: slim config lists derived from 6-GPU autotune analysis
# (H100, H200, B200, B300, L40S, RTX PRO 6000).  RPP=1 across all kernels:
# RPP=1 won 96% of forward, 93% of backward, and 96% of compose entries.
#
# Comprehensive mode (DORA_AUTOTUNE_COMPREHENSIVE=1): full grid search over
# block sizes x warps x pipeline stages.  Use for new GPU architectures or
# MoE models with unusual hidden dimensions.
# ---------------------------------------------------------------------------
import os

_DORA_AUTOTUNE_COMPREHENSIVE = os.environ.get("DORA_AUTOTUNE_COMPREHENSIVE", "0") == "1"


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _bucket_num_rows(num_rows):
    """Coarse occupancy bucket for Triton autotune keys.

    The DoRA kernels' launch geometry depends on ``num_rows`` via
    ``ROWS_PER_PROGRAM``. Keying only on ``num_cols`` aliases materially
    different occupancy regimes into the same autotune cache entry.

    A power-of-two bucket keeps the cache small while separating the row-count
    regimes that actually change winners on modern GPUs.
    """
    if num_rows <= 1:
        return 1
    return min(32768, 1 << (num_rows - 1).bit_length())


def _build_triton_configs(meta_options, warp_selector, stage_selector):
    """Materialize a Triton autotune grid from per-shape meta options."""
    invalid_block_sizes = sorted(
        {
            meta["BLOCK_SIZE"]
            for meta in meta_options
            if "BLOCK_SIZE" in meta and not _is_power_of_two(meta["BLOCK_SIZE"])
        }
    )
    if invalid_block_sizes:
        raise ValueError(
            "Triton autotune BLOCK_SIZE must be a power of two because these kernels use "
            f"tl.arange(0, BLOCK_SIZE); got {invalid_block_sizes}"
        )

    configs = []
    for meta in meta_options:
        for warps in warp_selector(meta):
            for stages in stage_selector(meta):
                configs.append(
                    triton.Config(
                        dict(meta),
                        num_warps=warps,
                        num_stages=stages,
                    )
                )
    return configs


def _compose_comprehensive_meta_options():
    """Stage-1 reduced forward space from H200/B200/B300 model-derived wins."""
    return [
        {"BLOCK_SIZE": 64, "ROWS_PER_PROGRAM": 2},
        {"BLOCK_SIZE": 64, "ROWS_PER_PROGRAM": 4},
        {"BLOCK_SIZE": 64, "ROWS_PER_PROGRAM": 8},
        {"BLOCK_SIZE": 128, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 128, "ROWS_PER_PROGRAM": 2},
        {"BLOCK_SIZE": 256, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 1024, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 2048, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 4096, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 8192, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 8192, "ROWS_PER_PROGRAM": 2},
    ]


def _backward_comprehensive_meta_options():
    """Stage-1 reduced backward space from H200/B200/B300 model-derived wins."""
    return [
        {"BLOCK_SIZE": 64, "ROWS_PER_PROGRAM": 2},
        {"BLOCK_SIZE": 64, "ROWS_PER_PROGRAM": 4},
        {"BLOCK_SIZE": 128, "ROWS_PER_PROGRAM": 2},
        {"BLOCK_SIZE": 256, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 256, "ROWS_PER_PROGRAM": 2},
        {"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 1024, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 2048, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 4096, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 8192, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 16384, "ROWS_PER_PROGRAM": 1},
        {"BLOCK_SIZE": 16384, "ROWS_PER_PROGRAM": 2},
    ]


def _compose_or_backward_warps(meta):
    bs = meta["BLOCK_SIZE"]
    rpp = meta.get("ROWS_PER_PROGRAM")
    if bs == 64:
        return [1, 2]
    if bs == 128:
        return [1, 2]
    if bs == 256:
        return [1, 2]
    if bs == 512:
        return [1, 2]
    if bs == 1024:
        return [1, 2, 4, 8] if rpp == 1 else [1, 2, 4]
    if bs == 2048:
        return [2, 4, 8]
    if bs == 4096:
        return [2, 4, 8]
    if bs == 8192:
        return [4, 8]
    if bs == 16384:
        return [8, 16]
    return [4, 8, 16]


def _compose_or_backward_stages(meta):
    bs = meta["BLOCK_SIZE"]
    if bs == 64:
        return [1, 2, 4]
    if bs == 128:
        return [1, 2, 4]
    if bs == 256:
        return [1, 2, 3, 4]
    if bs == 512:
        return [1, 2, 3]
    if bs == 1024:
        return [1, 2, 3, 4]
    if bs == 2048:
        return [2, 3, 4]
    if bs <= 8192:
        return [2, 3, 4, 5]
    return [2, 3, 4]


def _norm_comprehensive_meta_options():
    """Stage-1 reduced norm space from H200/B200/B300 model-derived wins."""
    return [{"BLOCK_SIZE": bs} for bs in [32, 64, 128, 256, 2048]]


def _norm_warps(meta):
    bs = meta["BLOCK_SIZE"]
    if bs == 32:
        return [1, 2]
    if bs == 64:
        return [1, 2]
    if bs in (128, 256):
        return [1, 2, 4]
    return [4, 8]


def _norm_stages(meta):
    bs = meta["BLOCK_SIZE"]
    if bs == 32:
        return [1, 2]
    if bs <= 256:
        return [1, 2, 3]
    return [1, 2, 3, 4]


def _compose_configs():
    """Autotune configs for compose and forward_and_inner kernels (RPP=1).

    6-GPU analysis: BS=4096/8192 dominate forward (195/216 wins of 626),
    BS=2048/1024 are significant secondary winners.  BS=16384 and BS=32768
    scored 0 forward wins and are dropped.  RPP=1 won 96% of entries.

    Warp counts are pinned per block size from the dominant winners:
      BS=512 → W=1 (unanimous B200/B300/H100/H200/RTX6000)
      BS=1024 → W=2 (H200/L40S dominant, covers B200/RTX6000 adequately)
      BS=2048 → W=8 (B300/H100/H200 dominant)
      BS=4096 → W=4,8 (W=8 dominant on B200/B300/H100/H200; W=4 secondary)
      BS=8192 → W=4,8 (W=4 dominant forward; W=8 dominant compose on H200/L40S)
    Stages left at Triton default (S=2) — bandwidth-bound kernels show
    <5% sensitivity to stage count across most shapes.
    """
    if not _TRITON_AVAILABLE:
        return []
    if _DORA_AUTOTUNE_COMPREHENSIVE:
        # Stage-1 reduction: remove obviously dead forward configs while
        # preserving winners seen across H200, B200, and B300 model traffic.
        return _build_triton_configs(
            _compose_comprehensive_meta_options(),
            _compose_or_backward_warps,
            _compose_or_backward_stages,
        )
    # Default: slim list from 6-GPU autotune analysis (7 configs).
    return [
        triton.Config({"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 1}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 1024, "ROWS_PER_PROGRAM": 1}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 2048, "ROWS_PER_PROGRAM": 1}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096, "ROWS_PER_PROGRAM": 1}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096, "ROWS_PER_PROGRAM": 1}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192, "ROWS_PER_PROGRAM": 1}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 8192, "ROWS_PER_PROGRAM": 1}, num_warps=8),
    ]


def _backward_configs():
    """Autotune configs for the backward kernel (RPP=1).

    6-GPU analysis: BS=16384 dominates on H100 (43 wins), BS=8192/4096 are
    strong secondary winners.  BS=32768 scored 0 wins and is dropped.
    BS=512 is marginal (6 total wins) and dropped.  RPP=1 won 93% of entries.

    Warp counts pinned per block size:
      BS=1024 → W=1 (B300 7/9, H100 2/3)
      BS=2048 → W=2 (B300/H100/H200 dominant)
      BS=4096 → W=2,4 (W=2 dominant H200/RTX6000; W=4 dominant B300)
      BS=8192 → W=4 (dominant most GPUs, W=8 secondary)
      BS=16384 → W=8,16 (W=16 unanimous H200, dominant B300; W=8 dominant H100)
    Stages left at Triton default — see compose docstring.
    """
    if not _TRITON_AVAILABLE:
        return []
    if _DORA_AUTOTUNE_COMPREHENSIVE:
        # Stage-1 reduction: keep only backward tiles with real cross-device
        # wins or a plausible adjacent need in current model-derived traffic.
        # 32K remains available in targeted microbenchmarks, but not here.
        return _build_triton_configs(
            _backward_comprehensive_meta_options(),
            _compose_or_backward_warps,
            _compose_or_backward_stages,
        )
    # Default: slim list from 6-GPU autotune analysis (7 configs).
    return [
        triton.Config({"BLOCK_SIZE": 1024, "ROWS_PER_PROGRAM": 1}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 2048, "ROWS_PER_PROGRAM": 1}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 4096, "ROWS_PER_PROGRAM": 1}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 4096, "ROWS_PER_PROGRAM": 1}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 8192, "ROWS_PER_PROGRAM": 1}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 16384, "ROWS_PER_PROGRAM": 1}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 16384, "ROWS_PER_PROGRAM": 1}, num_warps=16),
    ]


def _norm_configs():
    """Autotune configs for norm assembly kernels.

    Returns None in default mode (norm kernels use @triton.jit with fixed
    BS=256 — they are launch-latency bound on modern GPUs and autotune
    cannot differentiate configs).
    """
    if not _TRITON_AVAILABLE:
        return None
    if _DORA_AUTOTUNE_COMPREHENSIVE:
        return _build_triton_configs(
            _norm_comprehensive_meta_options(),
            _norm_warps,
            _norm_stages,
        )
    return None


# ---------------------------------------------------------------------------
# Strategy 1: Fused Element-wise Composition Kernel
# ---------------------------------------------------------------------------
# Computes: out = (mag - 1) * base + mag * (scale * lora)
#
# This is algebraically equivalent to: mag * (scale * lora + base) - base
# but avoids catastrophic cancellation in bf16/fp16 when mag ≈ 1.  The
# original form computes two large-magnitude terms (mag*base and base)
# and subtracts them; the rewritten form keeps (mag-1) as a small value
# near zero, preserving precision.
#
# Replaces 4 separate CUDA kernel launches with 1.
# Memory traffic: 3 reads (lora, base, mag_norm_scale) + 1 write
# vs original: ~12 memory passes across 4 ops.
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.autotune(configs=_compose_configs(), key=["num_cols", "num_rows_bucket"])
    @triton.jit
    def _fused_dora_compose_kernel(
        lora_ptr,
        base_ptr,
        mag_ptr,
        out_ptr,
        scale,
        num_rows,
        num_cols,
        num_rows_bucket,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        """
        Fused DoRA composition kernel (numerically stable form).

        For each element (i, j):
            out[i, j] = (mag[j] - 1) * base[i, j] + mag[j] * (scale * lora[i, j])

        This is algebraically equivalent to ``mag * (scale * lora + base) - base``
        but avoids catastrophic cancellation in bf16/fp16 when mag ≈ 1.

        mag is broadcast along dim 0 (shape [1, num_cols] or [num_cols]).
        lora, base, out are [num_rows, num_cols].
        """
        pid = tl.program_id(0)
        row_start = pid * ROWS_PER_PROGRAM
        col_offsets = tl.arange(0, BLOCK_SIZE)

        for r in range(ROWS_PER_PROGRAM):
            row = row_start + r
            row_mask = row < num_rows
            for col_start in range(0, num_cols, BLOCK_SIZE):
                cols = col_start + col_offsets
                col_mask = cols < num_cols
                mask = row_mask & col_mask

                idx = row * num_cols + cols

                lora_val = tl.load(lora_ptr + idx, mask=mask, other=0.0)
                base_val = tl.load(base_ptr + idx, mask=mask, other=0.0)
                mag_val = tl.load(mag_ptr + cols, mask=col_mask, other=0.0)

                scaled_lora = scale * lora_val
                composed = (mag_val - 1.0) * base_val + mag_val * scaled_lora
                tl.store(out_ptr + idx, composed, mask=mask)

    # No autotune for the in-place kernel: triton.autotune runs the kernel
    # multiple times to benchmark configs, but each trial mutates the same
    # lora buffer.  By the time the best config is selected, the buffer is
    # corrupted from previous trials.  The kernel is bandwidth-bound (3 reads
    # + 1 write), so a fixed config works well for all practical sizes.
    @triton.jit
    def _fused_dora_compose_inplace_kernel(
        lora_ptr,
        base_ptr,
        mag_ptr,
        scale,
        num_rows,
        num_cols,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        """
        In-place fused DoRA composition kernel (numerically stable form).

        For each element (i, j):
            lora[i, j] = (mag[j] - 1) * base[i, j] + mag[j] * (scale * lora[i, j])

        mag is broadcast along dim 0 (shape [1, num_cols] or [num_cols]).
        """
        pid = tl.program_id(0)
        row_start = pid * ROWS_PER_PROGRAM
        col_offsets = tl.arange(0, BLOCK_SIZE)

        for r in range(ROWS_PER_PROGRAM):
            row = row_start + r
            row_mask = row < num_rows
            for col_start in range(0, num_cols, BLOCK_SIZE):
                cols = col_start + col_offsets
                col_mask = cols < num_cols
                mask = row_mask & col_mask

                idx = row * num_cols + cols

                lora_val = tl.load(lora_ptr + idx, mask=mask, other=0.0)
                base_val = tl.load(base_ptr + idx, mask=mask, other=0.0)
                mag_val = tl.load(mag_ptr + cols, mask=col_mask, other=0.0)

                scaled_lora = scale * lora_val
                composed = (mag_val - 1.0) * base_val + mag_val * scaled_lora
                tl.store(lora_ptr + idx, composed, mask=mask)

    @triton.autotune(configs=_compose_configs(), key=["num_cols", "num_rows_bucket"])
    @triton.jit
    def _fused_dora_forward_and_inner_kernel(
        lora_ptr,
        base_ptr,
        mag_ptr,
        out_ptr,
        inner_ptr,
        scale,
        num_rows,
        num_cols,
        num_rows_bucket,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        """
        Dual-output fused DoRA forward kernel for the custom autograd path.

        Computes both outputs in a single pass over the input data:
            out[i, j]   = (mag[j] - 1) * base[i, j] + mag[j] * (scale * lora[i, j])
            inner[i, j] = scale * lora[i, j] + base[i, j]

        The intermediate ``scaled_lora = scale * lora`` lives only in SRAM
        registers and is never materialized as a global-memory tensor.  This
        eliminates the +256 MB forward-pass VRAM spike observed when the
        autograd forward computed these via sequential PyTorch ops.

        mag is broadcast along dim 0 (shape [1, num_cols] or [num_cols]).
        lora, base, out, inner are [num_rows, num_cols].
        """
        pid = tl.program_id(0)
        row_start = pid * ROWS_PER_PROGRAM
        col_offsets = tl.arange(0, BLOCK_SIZE)

        for r in range(ROWS_PER_PROGRAM):
            row = row_start + r
            row_mask = row < num_rows
            for col_start in range(0, num_cols, BLOCK_SIZE):
                cols = col_start + col_offsets
                col_mask = cols < num_cols
                mask = row_mask & col_mask

                idx = row * num_cols + cols

                lora_val = tl.load(lora_ptr + idx, mask=mask, other=0.0)
                base_val = tl.load(base_ptr + idx, mask=mask, other=0.0)
                mag_val = tl.load(mag_ptr + cols, mask=col_mask, other=0.0)

                scaled_lora = scale * lora_val
                inner_val = scaled_lora + base_val
                out_val = (mag_val - 1.0) * base_val + mag_val * scaled_lora

                tl.store(out_ptr + idx, out_val, mask=mask)
                tl.store(inner_ptr + idx, inner_val, mask=mask)


def _is_dynamo_compiling() -> bool:
    """Return True when running under torch.compile graph capture."""
    return bool(dynamo_is_compiling is not None and dynamo_is_compiling())


def _mag_broadcasts_last_dim(mag: torch.Tensor, lora: torch.Tensor) -> bool:
    """Return True if *mag* broadcasts only along the last dimension of *lora*.

    The Triton compose kernel treats mag as a 1-D vector of length
    ``lora.shape[-1]``.  This helper returns False for Conv-style shapes
    like ``[1, C, 1, 1]`` applied to ``[N, C, H, W]`` (where C != W)
    so that those are routed to the PyTorch fallback which handles
    arbitrary broadcasting natively.

    We check both that mag's total element count matches the last dim AND
    that mag's own last dimension matches, ruling out degenerate shapes
    like ``mag=[F]`` applied to ``lora=[B, F, 1]`` where ``numel() == F``
    but the last dim is 1.
    """
    return mag.numel() == lora.shape[-1] and mag.shape[-1] == lora.shape[-1]


def fused_dora_compose(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Fused DoRA composition: out = (mag_norm_scale - 1) * base + mag_norm_scale * (scale * lora).

    Algebraically equivalent to ``mag * (scale * lora + base) - base`` but uses
    the numerically stable form that avoids catastrophic cancellation in bf16/fp16.

    Args:
        lora: LoRA result tensor [..., out_features]
        base: Base layer result tensor, same shape as lora
        mag_norm_scale: Magnitude/norm scale, broadcastable to lora shape.
                        Expected shape [1, out_features] or [out_features].
        scale: Scalar LoRA scaling factor.
        inplace: If True, write result into lora tensor.

    Returns:
        The composed result tensor (may be lora if inplace=True).
    """
    if (
        not _is_dynamo_compiling()
        and _TRITON_AVAILABLE
        and lora.is_cuda
        and lora.is_contiguous()
        and base.is_contiguous()
        and mag_norm_scale.is_contiguous()
        and _mag_broadcasts_last_dim(mag_norm_scale, lora)
        and lora.dtype == base.dtype == mag_norm_scale.dtype
    ):
        return _fused_dora_compose_triton(lora, base, mag_norm_scale, scale, inplace)
    return _fused_dora_compose_torch(lora, base, mag_norm_scale, scale, inplace)


def _fused_dora_compose_torch(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
    inplace: bool = True,
) -> torch.Tensor:
    """Pure PyTorch fallback for fused DoRA composition.

    Uses the numerically stable form ``(mag - 1) * base + mag * (scale * lora)``
    to avoid catastrophic cancellation in bf16/fp16 when mag ≈ 1.

    Note:
        The caller (``dora.py``) is responsible for ensuring ``lora`` is a
        freshly-computed tensor (e.g. ``lora_B(lora_A(x))``) and does not
        share storage with other live tensors when ``inplace=True``.
    """
    if inplace:
        if torch.promote_types(torch.promote_types(lora.dtype, base.dtype), mag_norm_scale.dtype) != lora.dtype:
            result = (mag_norm_scale - 1) * base + mag_norm_scale * (scale * lora)
            lora.copy_(result)
            return lora

        # Numerically stable in-place: lora = (mag-1)*base + mag*(scale*lora)
        # Two mul_ calls preserve canonical evaluation order mag*(scale*lora),
        # matching the out-of-place path and the Triton kernel for same-dtype
        # bitwise parity.
        lora.mul_(scale)
        lora.mul_(mag_norm_scale)
        lora.add_(base * (mag_norm_scale - 1))
        return lora
    else:
        result = (mag_norm_scale - 1) * base + mag_norm_scale * (scale * lora)
        # Cast to lora.dtype for consistent output contract (matches inplace branch).
        if result.dtype != lora.dtype:
            result = result.to(lora.dtype)
        return result


def _fused_dora_compose_triton(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
    inplace: bool = True,
) -> torch.Tensor:
    """Triton implementation of fused DoRA composition."""
    # Early return for empty tensors — avoids Triton launch with grid=(0,).
    if lora.numel() == 0:
        return _fused_dora_compose_torch(lora, base, mag_norm_scale, scale, inplace)

    # Flatten to 2D for the kernel
    orig_shape = lora.shape
    num_cols = orig_shape[-1]
    num_rows = lora.numel() // num_cols
    num_rows_bucket = _bucket_num_rows(num_rows)

    lora_2d = lora.reshape(num_rows, num_cols)
    base_2d = base.reshape(num_rows, num_cols)
    mag_flat = mag_norm_scale.reshape(-1)

    if mag_flat.shape[0] != num_cols:
        raise ValueError(f"mag_norm_scale last dim {mag_flat.shape[0]} != num_cols {num_cols}")

    if inplace:
        # Fixed config (no autotune) — see comment on _fused_dora_compose_inplace_kernel.
        BLOCK_SIZE = 4096
        ROWS_PER_PROGRAM = 1
        grid = ((num_rows + ROWS_PER_PROGRAM - 1) // ROWS_PER_PROGRAM,)
        _fused_dora_compose_inplace_kernel[grid](
            lora_2d,
            base_2d,
            mag_flat,
            scale,
            num_rows,
            num_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        )
        return lora.reshape(orig_shape)
    else:
        # Lambda grid: autotune selects ROWS_PER_PROGRAM per hidden-dim config.
        grid = lambda meta: ((num_rows + meta["ROWS_PER_PROGRAM"] - 1) // meta["ROWS_PER_PROGRAM"],)  # noqa: E731
        out = torch.empty_like(lora_2d)
        _fused_dora_compose_kernel[grid](
            lora_2d,
            base_2d,
            mag_flat,
            out,
            scale,
            num_rows,
            num_cols,
            num_rows_bucket,
        )
        return out.reshape(orig_shape)


def fused_dora_forward_and_inner(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
) -> tuple:
    """
    Compute both the DoRA composition output and the ``inner`` tensor in one pass.

    Returns:
        (out, inner) where:
            out   = (mag_norm_scale - 1) * base + mag_norm_scale * (scale * lora)
            inner = scale * lora + base

    When Triton is available and tensors are on CUDA, a single fused kernel
    computes both outputs simultaneously — the intermediate ``scaled_lora``
    never leaves SRAM registers.  This eliminates the VRAM spike caused by
    sequential PyTorch ops in ``FusedDoRAComposeFunction.forward``.
    """
    if (
        not _is_dynamo_compiling()
        and _TRITON_AVAILABLE
        and lora.is_cuda
        and lora.is_contiguous()
        and base.is_contiguous()
        and mag_norm_scale.is_contiguous()
        and _mag_broadcasts_last_dim(mag_norm_scale, lora)
        and lora.dtype == base.dtype == mag_norm_scale.dtype
    ):
        return _fused_dora_forward_and_inner_triton(lora, base, mag_norm_scale, scale)
    return _fused_dora_forward_and_inner_torch(lora, base, mag_norm_scale, scale)


def _fused_dora_forward_and_inner_torch(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
) -> tuple:
    """Pure PyTorch fallback for fused forward-and-inner computation.

    Computes both outputs without Triton.  ``scaled_lora`` is a temporary
    that is freed after ``inner`` and ``out`` are computed.
    """
    result_dtype = lora.dtype
    scaled_lora = scale * lora
    inner = scaled_lora + base
    out = (mag_norm_scale - 1) * base + mag_norm_scale * scaled_lora
    if out.dtype != result_dtype:
        out = out.to(result_dtype)
    # Cast inner to match the Triton path's output contract (inner in
    # lora.dtype).  Without this, mixed-dtype inputs produce an fp32 inner
    # that doubles the activation VRAM saved for backward (d_mag reduction).
    if inner.dtype != result_dtype:
        inner = inner.to(result_dtype)
    return out, inner


def _fused_dora_forward_and_inner_triton(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
) -> tuple:
    """Triton implementation of fused forward-and-inner computation."""
    # Early return for empty tensors — avoids Triton launch with grid=(0,).
    if lora.numel() == 0:
        return _fused_dora_forward_and_inner_torch(lora, base, mag_norm_scale, scale)

    orig_shape = lora.shape
    num_cols = orig_shape[-1]
    num_rows = lora.numel() // num_cols
    num_rows_bucket = _bucket_num_rows(num_rows)

    lora_2d = lora.reshape(num_rows, num_cols)
    base_2d = base.reshape(num_rows, num_cols)
    mag_flat = mag_norm_scale.reshape(-1)

    if mag_flat.shape[0] != num_cols:
        raise ValueError(f"mag_norm_scale last dim {mag_flat.shape[0]} != num_cols {num_cols}")

    out = torch.empty_like(lora_2d)
    inner = torch.empty_like(lora_2d)

    grid = lambda meta: ((num_rows + meta["ROWS_PER_PROGRAM"] - 1) // meta["ROWS_PER_PROGRAM"],)  # noqa: E731

    _fused_dora_forward_and_inner_kernel[grid](
        lora_2d,
        base_2d,
        mag_flat,
        out,
        inner,
        scale,
        num_rows,
        num_cols,
        num_rows_bucket,
    )

    return out.reshape(orig_shape), inner.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Strategy 2: Fused Norm Assembly Kernel
# ---------------------------------------------------------------------------
# Fuses: norm_sq = w_norm_sq + 2*s*cross + s^2*ba_norm_sq
#         norm_sq = clamp(norm_sq, min=0)
#         weight_norm = sqrt(norm_sq)
# and optionally: mag_norm_scale = magnitude / weight_norm
# into a single kernel pass over vectors of length out_features.
# ---------------------------------------------------------------------------

_norm_cfgs = _norm_configs()  # None in default mode

if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_norm_assembly_kernel_impl(
        w_norm_sq_ptr,
        cross_term_ptr,
        ba_norm_sq_ptr,
        out_ptr,
        two_s,
        s2,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused norm assembly kernel.

        Scalar coefficients ``two_s = 2*scale`` and ``s2 = scale*scale`` are
        precomputed by the launcher in fp64 Python arithmetic, then auto-cast
        to fp32 at the Triton kernel boundary — exactly matching PyTorch's
        scalar-tensor promotion.  Store-reload barriers after each multiply
        and add force intermediate rounding to fp32 (preventing FMA fusion),
        reproducing PyTorch's 4-kernel evaluation order.

        For each element i:
            norm_sq = w_norm_sq[i] + two_s*cross[i] + s2*ba_norm_sq[i]
            norm_sq = max(norm_sq, 0)
            out[i] = sqrt(norm_sq)
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        w = tl.load(w_norm_sq_ptr + offsets, mask=mask, other=0.0)
        c = tl.load(cross_term_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(ba_norm_sq_ptr + offsets, mask=mask, other=0.0)

        # term1: two_s * c — force rounding via store-reload
        term_c = two_s * c
        tl.store(out_ptr + offsets, term_c, mask=mask)
        term_c = tl.load(out_ptr + offsets, mask=mask)

        # partial: w + term_c — force rounding via store-reload
        partial = w + term_c
        tl.store(out_ptr + offsets, partial, mask=mask)
        partial = tl.load(out_ptr + offsets, mask=mask)

        # term2: s2 * b — force rounding via store-reload
        term_b = s2 * b
        tl.store(out_ptr + offsets, term_b, mask=mask)
        term_b = tl.load(out_ptr + offsets, mask=mask)

        # final add (matches PyTorch's 4th separate kernel)
        norm_sq = partial + term_b

        # clamp_min(0), NaN-preserving: tl.where(x > 0, x, 0) would map NaN→0
        # because NaN > 0 is False.  Using a two-step approach preserves NaN
        # for parity with torch.clamp_min (which propagates NaN per IEEE 754).
        norm_sq = tl.where(norm_sq > 0.0, norm_sq, tl.where(norm_sq != norm_sq, norm_sq, 0.0))
        # Force IEEE 754 correctly-rounded sqrt (sqrt.rn.f32) via inline PTX.
        # Both tl.sqrt and tl.math.sqrt compile to sqrt.approx.ftz.f32 on
        # SM90 (Hopper), which diverges from torch.sqrt (sqrt.rn.f32).
        # The .to(tl.float32) cast is needed because the inline asm operates
        # on fp32 registers; bf16/fp16 bit patterns would be misinterpreted.
        # In production inputs are always fp32 (pre-accumulated), so both
        # casts are no-ops.
        result = tl.inline_asm_elementwise(
            "sqrt.rn.f32 $0, $1;",
            "=r,r",
            args=[norm_sq.to(tl.float32)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        ).to(norm_sq.dtype)

        tl.store(out_ptr + offsets, result, mask=mask)

    # Conditional decoration: autotune in comprehensive mode, plain jit otherwise.
    # Norm kernels are launch-latency bound on modern GPUs (identical timings
    # across all input dims on B200), so autotune wastes compilation time in
    # default mode.  Fixed BS=256 is used instead.
    if _norm_cfgs is not None:
        _fused_norm_assembly_kernel = triton.autotune(configs=_norm_cfgs, key=["N"])(_fused_norm_assembly_kernel_impl)
    else:
        _fused_norm_assembly_kernel = _fused_norm_assembly_kernel_impl


def fused_norm_assembly(
    w_norm_sq: torch.Tensor,
    cross_term: torch.Tensor,
    ba_norm_sq: torch.Tensor,
    scale: float,
) -> tuple:
    """
    Fused norm assembly: compute weight_norm from components.

    Args:
        w_norm_sq: ||W||^2 per row, shape [out_features]
        cross_term: <W, BA> per row, shape [out_features]
        ba_norm_sq: ||BA||^2 per row, shape [out_features]
        scale: LoRA scaling factor

    Returns:
        (weight_norm,) — magnitude division is always done in PyTorch by the caller.
    """
    if (
        not _is_dynamo_compiling()
        and _TRITON_AVAILABLE
        and w_norm_sq.is_cuda
        and w_norm_sq.is_contiguous()
        and cross_term.is_contiguous()
        and ba_norm_sq.is_contiguous()
        # The Triton norm kernel uses inline PTX (sqrt.rn.f32) which is
        # NVIDIA-specific.  On ROCm/HIP (torch.version.hip is set), fall
        # back to PyTorch to avoid a crash from NVIDIA-only assembly.
        and torch.version.hip is None
    ):
        return _fused_norm_assembly_triton(
            w_norm_sq,
            cross_term,
            ba_norm_sq,
            scale,
        )
    return _fused_norm_assembly_torch(
        w_norm_sq,
        cross_term,
        ba_norm_sq,
        scale,
    )


def _fused_norm_assembly_torch(
    w_norm_sq: torch.Tensor,
    cross_term: torch.Tensor,
    ba_norm_sq: torch.Tensor,
    scale: float,
) -> tuple:
    """Pure PyTorch fallback for fused norm assembly."""
    # Use Python float directly instead of torch.as_tensor to avoid allocating
    # a new scalar tensor on every call.  PyTorch handles scalar-tensor
    # promotion natively in the arithmetic below.
    s = float(scale)
    norm_sq = w_norm_sq + (2.0 * s) * cross_term + (s * s) * ba_norm_sq
    # Keep this out-of-place to avoid in-place autograd hazards if this helper
    # is called with grad tracking enabled.
    norm_sq = norm_sq.clamp_min(0)
    weight_norm = torch.sqrt(norm_sq)

    return (weight_norm,)


def _fused_norm_assembly_triton(
    w_norm_sq: torch.Tensor,
    cross_term: torch.Tensor,
    ba_norm_sq: torch.Tensor,
    scale: float,
) -> tuple:
    """Triton implementation of fused norm assembly (norm-only)."""
    N = w_norm_sq.shape[0]

    # Early return for empty tensors — avoids Triton launch with grid=(0,).
    if N == 0:
        return _fused_norm_assembly_torch(w_norm_sq, cross_term, ba_norm_sq, scale)

    # Precompute scalar coefficients in fp64 Python arithmetic, matching the
    # PyTorch fallback's scalar-tensor promotion (Python float → fp32 at the
    # tensor multiply boundary).  Triton auto-casts these fp64 Python floats
    # to fp32 at the kernel boundary — same cast PyTorch performs.
    s = float(scale)
    two_s = 2.0 * s  # fp64 arithmetic
    s2 = s * s  # fp64 arithmetic

    # In autotune mode, BLOCK_SIZE is selected by autotune; use lambda grid.
    # In fixed mode (default), use BS=256 directly.
    if _norm_cfgs is not None:
        grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)  # noqa: E731
    else:
        _FIXED_BS = 256
        grid = ((N + _FIXED_BS - 1) // _FIXED_BS,)

    out = torch.empty_like(w_norm_sq)
    if _norm_cfgs is not None:
        _fused_norm_assembly_kernel[grid](
            w_norm_sq,
            cross_term,
            ba_norm_sq,
            out,
            two_s,
            s2,
            N,
        )
    else:
        _fused_norm_assembly_kernel[grid](
            w_norm_sq,
            cross_term,
            ba_norm_sq,
            out,
            two_s,
            s2,
            N,
            BLOCK_SIZE=_FIXED_BS,
        )
    return (out,)


# ---------------------------------------------------------------------------
# Strategy 3: Custom Autograd Function with Fused Backward
# ---------------------------------------------------------------------------
# Wraps the DoRA composition in a torch.autograd.Function with a hand-written
# backward pass that fuses gradient computation.
#
# Forward (stable form): out = (mag - 1) * base + mag * (s * lora)
#   Algebraically equivalent to ``mag * (s * lora + base) - base``, but
#   avoids catastrophic cancellation when ``mag ≈ 1``.
#
# Backward (derived from the equivalent form ``mag * inner - base``
#   where ``inner = s * lora + base``):
#   d_lora = mag * s * d_out
#   d_mag  = (inner * d_out).sum(broadcast_dims)
#   d_base = (mag - 1) * d_out
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.autotune(configs=_backward_configs(), key=["num_cols", "num_rows_bucket"])
    @triton.jit
    def _fused_dora_backward_kernel(
        d_out_ptr,
        mag_ptr,
        d_lora_ptr,
        d_base_ptr,
        scale,
        num_rows,
        num_cols,
        num_rows_bucket,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        """
        Fused backward kernel for DoRA composition (d_lora and d_base only).

        d_mag is computed separately via PyTorch reduction to avoid atomic
        contention that scales poorly with num_rows.

        For each (i, j):
            d_lora[i,j] = mag[j] * scale * d_out[i,j]
            d_base[i,j] = (mag[j] - 1) * d_out[i,j]

        6-GPU autotune analysis shows RPP=1 winning 93% of backward entries.
        The backward kernel has lower total memory traffic than the
        forward+inner kernel (3 matrix-sized passes: 1 read + 2 writes,
        vs forward's 4: 2 reads + 2 writes).
        """
        pid = tl.program_id(0)
        row_start = pid * ROWS_PER_PROGRAM
        col_offsets = tl.arange(0, BLOCK_SIZE)

        for r in range(ROWS_PER_PROGRAM):
            row = row_start + r
            row_mask = row < num_rows
            for col_start in range(0, num_cols, BLOCK_SIZE):
                cols = col_start + col_offsets
                col_mask = cols < num_cols
                mask = row_mask & col_mask

                idx = row * num_cols + cols

                d_out_val = tl.load(d_out_ptr + idx, mask=mask, other=0.0)
                mag_val = tl.load(mag_ptr + cols, mask=col_mask, other=0.0)

                # d_lora = mag * scale * d_out
                d_lora_val = mag_val * scale * d_out_val
                tl.store(d_lora_ptr + idx, d_lora_val, mask=mask)

                # d_base = (mag - 1) * d_out
                d_base_val = (mag_val - 1.0) * d_out_val
                tl.store(d_base_ptr + idx, d_base_val, mask=mask)


class FusedDoRAComposeFunction(torch.autograd.Function):
    """
    Custom autograd function for DoRA composition with fused backward.

    Forward (stable form): out = (mag_norm_scale - 1) * base + mag_norm_scale * (scale * lora)

    Algebraically equivalent to ``mag_norm_scale * (scale * lora + base) - base``,
    but avoids catastrophic cancellation when ``mag_norm_scale ≈ 1``.

    Backward: fused gradient computation for d_lora, d_base, d_mag

    **VRAM tradeoff**: This path saves ``inner = scale * lora + base``
    (one tensor, same size as ``lora``) via ``ctx.save_for_backward`` only
    when ``mag_norm_scale.requires_grad`` is True.  When mag is frozen
    (e.g. during warmup or partial fine-tuning), ``inner`` is never
    allocated, reclaiming 100% of the fused backward VRAM overhead.

    When ``inner`` is needed, the forward pass uses
    ``fused_dora_forward_and_inner`` to compute both ``out`` and ``inner``
    in a single fused kernel (when Triton is available), so the intermediate
    ``scaled_lora = scale * lora`` never leaves SRAM registers and is never
    allocated as a global-memory tensor.  This eliminates the VRAM spike
    from sequential PyTorch ops.  When mag is frozen, a forward-only fused
    compose is used instead (no ``inner`` allocation at all).  Versus the
    in-place unfused baseline, overhead is at most 1x ``lora``-sized
    activation per layer (the saved ``inner``).
    """

    @staticmethod
    def forward(
        ctx,
        lora: torch.Tensor,
        base: torch.Tensor,
        mag_norm_scale: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """
        Args:
            lora: LoRA output tensor [..., out_features]
            base: Base result tensor, same shape as lora
            mag_norm_scale: Magnitude/norm scale [1, out_features] or [out_features]
            scale: Scalar LoRA scaling factor (float)

        Returns:
            out = (mag_norm_scale - 1) * base + mag_norm_scale * (scale * lora)
        """
        # The entire forward body runs under no_grad because gradients are
        # hand-computed in backward().
        with torch.no_grad():
            # Only materialise ``inner`` when we need d_mag in backward.
            # When mag is frozen (requires_grad=False) — e.g. during warmup
            # or partial fine-tuning — this skips a full activation-sized
            # allocation (up to 34 GB on 70B models).
            if mag_norm_scale.requires_grad:
                # fused_dora_forward_and_inner computes both ``out`` and
                # ``inner`` in a single Triton kernel (when available),
                # keeping ``scaled_lora`` in SRAM registers only.
                out, inner = fused_dora_forward_and_inner(lora, base, mag_norm_scale, scale)
                ctx.save_for_backward(inner, mag_norm_scale)
                ctx.needs_mag = True
            else:
                # mag frozen — no inner needed, forward-only compose.
                out = fused_dora_compose(lora, base, mag_norm_scale, scale, inplace=False)
                ctx.save_for_backward(
                    mag_norm_scale,
                )
                ctx.needs_mag = False

        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, d_out):
        """
        Fused backward pass.

        Gradients (derived from ``out = mag * inner - base`` where
        ``inner = scale * lora + base``):
            d_lora = mag * scale * d_out
            d_base = (mag - 1) * d_out
            d_mag  = sum_over_broadcast_dims(inner * d_out)

        Numerical note — bf16/fp16 precision gap:
            The forward pass uses the numerically stable form
            ``out = (mag - 1) * base + mag * (scale * lora)`` to avoid
            catastrophic cancellation when ``mag ≈ 1``.  The backward,
            however, is derived from the algebraically equivalent form
            ``out = mag * inner - base`` (where ``inner = scale * lora + base``).
            In exact arithmetic the two are identical, but in bf16/fp16 they
            differ by O(eps_bf16) per element per layer due to rounding in
            intermediate accumulations.  This gap is expected and benign for
            typical training workloads.
        """
        scale = ctx.scale
        needs_mag = ctx.needs_mag

        if needs_mag:
            inner, mag_norm_scale = ctx.saved_tensors
        else:
            (mag_norm_scale,) = ctx.saved_tensors
            inner = None  # not saved — mag was frozen

        d_lora = d_base = d_mag = None

        needs_lora_grad = ctx.needs_input_grad[0]
        needs_base_grad = ctx.needs_input_grad[1]
        needs_mag_grad = ctx.needs_input_grad[2]

        if (
            not _is_dynamo_compiling()
            and _TRITON_AVAILABLE
            and d_out.is_cuda
            and d_out.is_contiguous()
            and (inner is None or inner.is_contiguous())
            and _mag_broadcasts_last_dim(mag_norm_scale, d_out)
            and d_out.dtype == mag_norm_scale.dtype
            # inner dtype intentionally not checked: Triton backward only
            # reads d_out and mag; inner is used in d_mag reduction which
            # has its own .to() cast.
        ):
            d_lora, d_base, d_mag = _fused_backward_triton(
                d_out,
                inner,
                mag_norm_scale,
                scale,
                needs_lora_grad,
                needs_base_grad,
                needs_mag_grad,
            )
        else:
            d_lora, d_base, d_mag = _fused_backward_torch(
                d_out,
                inner,
                mag_norm_scale,
                scale,
                needs_lora_grad,
                needs_base_grad,
                needs_mag_grad,
            )

        return d_lora, d_base, d_mag, None  # None for scale (not a Tensor)


def _broadcast_reduce_dims(out_shape: torch.Size, mag_shape: torch.Size) -> list:
    """Return the dimensions of *out_shape* that were broadcast from *mag_shape*.

    For Linear:  out ``[B, F]``, mag ``[1, F]`` → reduce ``[0]``
    For Conv2d:  out ``[N, C, H, W]``, mag ``[1, C, 1, 1]`` → reduce ``[0, 2, 3]``
    """
    # Right-align shapes (standard broadcast semantics)
    ndim_out = len(out_shape)
    ndim_mag = len(mag_shape)
    dims: list = []
    for i in range(ndim_out):
        mag_i = i - (ndim_out - ndim_mag)
        if mag_i < 0:
            # mag has fewer dims → this dim was implicitly broadcast
            dims.append(i)
        elif mag_shape[mag_i] == 1 and out_shape[i] != 1:
            dims.append(i)
    return dims


def _fused_backward_torch(d_out, inner, mag_norm_scale, scale, needs_lora_grad, needs_base_grad, needs_mag_grad):
    """Pure PyTorch fallback for fused backward.

    ``inner`` is ``scale * lora + base`` (precomputed in forward).
    """
    d_lora = d_base = d_mag = None

    if needs_lora_grad:
        d_lora = mag_norm_scale * scale * d_out

    if needs_base_grad:
        d_base = (mag_norm_scale - 1) * d_out

    if needs_mag_grad:
        assert inner is not None, "inner must be saved when mag requires grad"
        # inner was computed under no_grad in forward and may have a different
        # dtype than d_out when AMP autocast is active (e.g. inner in fp32
        # while d_out in fp16).  Align dtypes to avoid silent precision loss.
        if inner.dtype != d_out.dtype:
            inner = inner.to(d_out.dtype)
        d_mag_full = inner * d_out
        # Reduce over dimensions that were broadcast from mag_norm_scale.
        # For Linear [B, F] with mag [1, F] → reduce dim 0.
        # For Conv2d [N, C, H, W] with mag [1, C, 1, 1] → reduce dims 0,2,3.
        sum_dims = _broadcast_reduce_dims(d_out.shape, mag_norm_scale.shape)
        if sum_dims:
            d_mag = d_mag_full.sum(dim=sum_dims, keepdim=True)
        else:
            d_mag = d_mag_full
        # Reshape to match mag_norm_scale shape
        d_mag = d_mag.reshape(mag_norm_scale.shape)

    return d_lora, d_base, d_mag


def _fused_backward_triton(d_out, inner, mag_norm_scale, scale, needs_lora_grad, needs_base_grad, needs_mag_grad):
    """Triton implementation of fused backward.

    Uses a Triton kernel for the element-wise d_lora / d_base computation
    and a plain PyTorch ``.sum()`` for the d_mag reduction. This avoids the
    ``tl.atomic_add`` contention that scaled poorly with large num_rows in
    the previous single-kernel approach.

    Only allocates output tensors and launches the kernel when at least one
    of d_lora / d_base is needed.  When only one is needed, we fall back to
    a single PyTorch elementwise op (cheaper than a kernel launch for one
    output).  When neither is needed we skip the kernel entirely.
    """
    # Early return for empty tensors — avoids Triton launch with grid=(0,).
    if d_out.numel() == 0:
        return _fused_backward_torch(
            d_out,
            inner,
            mag_norm_scale,
            scale,
            needs_lora_grad,
            needs_base_grad,
            needs_mag_grad,
        )

    d_lora = d_base = d_mag = None

    # The Triton kernel fuses d_lora + d_base into one pass.  Only worth
    # launching when *both* are needed.  For a single output, a plain
    # PyTorch elementwise op is cheaper than the Triton launch overhead.
    if needs_lora_grad and needs_base_grad:
        orig_shape = d_out.shape
        num_cols = orig_shape[-1]
        num_rows = d_out.numel() // num_cols
        num_rows_bucket = _bucket_num_rows(num_rows)

        d_out_2d = d_out.reshape(num_rows, num_cols)
        mag_flat = mag_norm_scale.reshape(-1)

        d_lora_2d = torch.empty_like(d_out_2d)
        d_base_2d = torch.empty_like(d_out_2d)

        # Autotune selects BLOCK_SIZE and ROWS_PER_PROGRAM from kernel configs.
        grid = lambda meta: ((num_rows + meta["ROWS_PER_PROGRAM"] - 1) // meta["ROWS_PER_PROGRAM"],)  # noqa: E731

        _fused_dora_backward_kernel[grid](
            d_out_2d,
            mag_flat,
            d_lora_2d,
            d_base_2d,
            scale,
            num_rows,
            num_cols,
            num_rows_bucket,
        )

        d_lora = d_lora_2d.reshape(orig_shape)
        d_base = d_base_2d.reshape(orig_shape)
    else:
        # At most one grad needed — use PyTorch (no wasted allocation).
        if needs_lora_grad:
            d_lora = mag_norm_scale * scale * d_out
        if needs_base_grad:
            d_base = (mag_norm_scale - 1) * d_out

    # d_mag via PyTorch reduction — avoids atomic contention in Triton
    if needs_mag_grad:
        assert inner is not None, "inner must be saved when mag requires grad"
        # inner was computed under no_grad in forward and may have a different
        # dtype than d_out when AMP autocast is active.  Align dtypes.
        if inner.dtype != d_out.dtype:
            inner = inner.to(d_out.dtype)
        d_mag_full = inner * d_out  # use original (non-reshaped) tensors
        sum_dims = _broadcast_reduce_dims(d_out.shape, mag_norm_scale.shape)
        if sum_dims:
            d_mag = d_mag_full.sum(dim=sum_dims, keepdim=True)
        else:
            d_mag = d_mag_full
        d_mag = d_mag.reshape(mag_norm_scale.shape)

    return d_lora, d_base, d_mag


# ---------------------------------------------------------------------------
# torch.library.custom_op registration (PyTorch 2.4+)
# ---------------------------------------------------------------------------
# Registering FusedDoRAComposeFunction as a custom op allows torch.compile's
# Dynamo to trace through the DoRA composition as a single opaque node rather
# than graph-breaking.  This lets Inductor fuse surrounding operations (LoRA
# A/B matmuls, activation functions) around it.
#
# The custom op forward computes the composition formula directly in PyTorch
# (required because AOTAutograd traces with FakeTensors that lack real data
# pointers for Triton).  The math is identical to FusedDoRAComposeFunction.
# ---------------------------------------------------------------------------

_HAS_CUSTOM_OP = False

try:
    _custom_op_decorator = getattr(torch.library, "custom_op", None)
    if _custom_op_decorator is not None:

        @_custom_op_decorator("peft::fused_dora_compose", mutates_args=())
        def _fused_dora_compose_custom_op(
            lora: torch.Tensor,
            base: torch.Tensor,
            mag_norm_scale: torch.Tensor,
            scale: float,
        ) -> torch.Tensor:
            """Custom op forward: same formula as FusedDoRAComposeFunction."""
            with torch.no_grad():
                scaled_lora = scale * lora
                out = (mag_norm_scale - 1) * base + mag_norm_scale * scaled_lora
            return out

        @_fused_dora_compose_custom_op.register_fake
        def _fused_dora_compose_fake(lora, base, mag_norm_scale, scale):
            return torch.empty_like(lora)

        def _fused_dora_compose_setup_context(ctx, inputs, output):
            lora, base, mag_norm_scale, scale = inputs
            # Mirror FusedDoRAComposeFunction.forward: only materialise
            # ``inner`` when mag requires grad (d_mag needs it in backward).
            # When mag is frozen this skips an activation-sized allocation
            # per adapted module — up to ~20 GB on 70B models.
            if mag_norm_scale.requires_grad:
                with torch.no_grad():
                    inner = scale * lora + base
                ctx.save_for_backward(inner, mag_norm_scale)
                ctx.needs_mag = True
            else:
                ctx.save_for_backward(
                    mag_norm_scale,
                )
                ctx.needs_mag = False
            ctx.scale = scale

        def _fused_dora_compose_backward(ctx, d_out):
            scale = ctx.scale
            needs_mag = ctx.needs_mag

            if needs_mag:
                inner, mag_norm_scale = ctx.saved_tensors
            else:
                (mag_norm_scale,) = ctx.saved_tensors
                inner = None

            needs_lora_grad = ctx.needs_input_grad[0]
            needs_base_grad = ctx.needs_input_grad[1]
            needs_mag_grad = ctx.needs_input_grad[2]

            # Always use the PyTorch path here.  AOTAutograd traces this
            # function with FakeTensors to build the compiled backward graph,
            # and Triton kernels require real data pointers (.data_ptr()) that
            # FakeTensors cannot provide.  Inductor will fuse and optimize
            # the resulting PyTorch ops when compiling the backward graph.
            #
            # The Triton backward kernel is still used via the
            # FusedDoRAComposeFunction eager autograd path (non-compiled).
            d_lora, d_base, d_mag = _fused_backward_torch(
                d_out,
                inner,
                mag_norm_scale,
                scale,
                needs_lora_grad,
                needs_base_grad,
                needs_mag_grad,
            )

            return d_lora, d_base, d_mag, None

        _fused_dora_compose_custom_op.register_autograd(
            _fused_dora_compose_backward,
            setup_context=_fused_dora_compose_setup_context,
        )

        _HAS_CUSTOM_OP = True
        logger.debug("DoRA: registered peft::fused_dora_compose custom op for torch.compile")
except Exception:
    # torch.library.custom_op not available or registration failed;
    # fall back to the _is_dynamo_compiling() guard in fused_dora_compose_autograd.
    logger.debug("DoRA: custom_op registration failed, falling back", exc_info=True)
    _HAS_CUSTOM_OP = False


def fused_dora_compose_autograd(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    DoRA composition with custom autograd for fused backward.

    This is the main entry point for training-time composition that benefits
    from the fused backward pass.

    When ``torch.compile`` is active and a ``torch.library.custom_op`` is
    registered (PyTorch 2.4+), this routes through ``peft::fused_dora_compose``
    so that Dynamo sees a single opaque graph node.  In eager mode, this
    always uses ``FusedDoRAComposeFunction.apply`` with Triton kernels.

    Args:
        lora: LoRA output tensor [..., out_features]
        base: Base result tensor, same shape as lora
        mag_norm_scale: Magnitude/norm scale [1, out_features]
        scale: Scalar LoRA scaling factor

    Returns:
        out = (mag_norm_scale - 1) * base + mag_norm_scale * (scale * lora)
    """
    if _is_dynamo_compiling():
        if _HAS_CUSTOM_OP:
            return _fused_dora_compose_custom_op(lora, base, mag_norm_scale, scale)
        # Fallback for older PyTorch without custom_op support.
        return (mag_norm_scale - 1) * base + mag_norm_scale * (scale * lora)
    return FusedDoRAComposeFunction.apply(lora, base, mag_norm_scale, scale)
