import torch
import triton
import triton.language as tl


# Model constant used by the reference implementation.
_DORA_SCALE = 0.7


@triton.jit
def _dora_backward_singlepass_kernel(
    d_out_ptr,
    inner_ptr,
    mag_ptr,
    out_ptr,
    stride_do0,
    stride_do1,
    stride_in0,
    stride_in1,
    stride_mag1,
    stride_out0,
    stride_out1,
    num_rows,
    num_cols,
    scale,
    INPUT_IS_BF16: tl.constexpr,
    INPUT_IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused stages:
      1) stream over all row tiles for one column block
      2) compute/store d_lora
      3) compute/store d_base
      4) accumulate d_mag entirely in-register for this column block
      5) store the final packed d_mag row

    We intentionally map one program to one column block and iterate over rows
    inside the kernel. That keeps the whole pipeline fused and avoids fp32
    atomic-add ordering noise on d_mag, which was the source of the test failure.
    A further row-parallel split would require atomics or a separate reduction
    kernel, so this is the maximal practical fusion for exactness here.
    """
    pid_n = tl.program_id(axis=0)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < num_cols

    # mag is logically [1, C]
    mag = tl.load(mag_ptr + cols * stride_mag1, mask=col_mask, other=0.0)
    mag_scaled = mag * scale
    mag_minus_one = mag - 1.0

    d_mag_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    row_offsets = tl.arange(0, BLOCK_M)

    for row_start in tl.range(0, num_rows, BLOCK_M):
        rows = row_start + row_offsets
        row_mask = rows < num_rows
        mask = row_mask[:, None] & col_mask[None, :]

        d_out_ptrs = d_out_ptr + rows[:, None] * stride_do0 + cols[None, :] * stride_do1
        inner_ptrs = inner_ptr + rows[:, None] * stride_in0 + cols[None, :] * stride_in1

        d_out = tl.load(d_out_ptrs, mask=mask, other=0.0)
        inner = tl.load(inner_ptrs, mask=mask, other=0.0)

        # Elementwise outputs.
        d_lora = d_out * mag_scaled[None, :]
        d_base = d_out * mag_minus_one[None, :]

        out_lora_ptrs = out_ptr + rows[:, None] * stride_out0 + cols[None, :] * stride_out1
        out_base_ptrs = out_ptr + (rows[:, None] + num_rows) * stride_out0 + cols[None, :] * stride_out1

        tl.store(out_lora_ptrs, d_lora.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_base_ptrs, d_base.to(out_ptr.dtype.element_ty), mask=mask)

        # Reduction for d_mag.
        # Match the reference more closely by rounding the elementwise product
        # to the input low precision first, then accumulating in fp32.
        prod = d_out * inner
        if INPUT_IS_BF16:
            prod = prod.to(tl.bfloat16)
        elif INPUT_IS_FP16:
            prod = prod.to(tl.float16)

        d_mag_acc += tl.sum(prod.to(tl.float32), axis=0)

    # Final packed d_mag row is row index 2 * num_rows.
    out_mag_ptrs = out_ptr + (2 * num_rows) * stride_out0 + cols * stride_out1
    tl.store(out_mag_ptrs, d_mag_acc.to(out_ptr.dtype.element_ty), mask=col_mask)


def kernel_function(
    d_out: torch.Tensor,
    inner: torch.Tensor,
    mag_norm_scale: torch.Tensor,
) -> torch.Tensor:
    assert d_out.is_cuda and inner.is_cuda and mag_norm_scale.is_cuda
    assert d_out.ndim == 2 and inner.ndim == 2
    assert d_out.shape == inner.shape
    assert d_out.dtype == inner.dtype

    r, c = d_out.shape

    if mag_norm_scale.ndim == 1:
        assert mag_norm_scale.shape[0] == c
        mag_view = mag_norm_scale[None, :]
    else:
        assert mag_norm_scale.ndim == 2
        assert mag_norm_scale.shape[0] == 1 and mag_norm_scale.shape[1] == c
        mag_view = mag_norm_scale

    assert mag_view.dtype == d_out.dtype

    out = torch.empty((2 * r + 1, c), device=d_out.device, dtype=d_out.dtype)

    # Single fused kernel: elementwise outputs + full row reduction.
    BLOCK_M = 64
    BLOCK_N = 32

    if c > 0:
        grid = (triton.cdiv(c, BLOCK_N),)
        _dora_backward_singlepass_kernel[grid](
            d_out,
            inner,
            mag_view,
            out,
            d_out.stride(0),
            d_out.stride(1),
            inner.stride(0),
            inner.stride(1),
            mag_view.stride(1),
            out.stride(0),
            out.stride(1),
            r,
            c,
            _DORA_SCALE,
            INPUT_IS_BF16=d_out.dtype == torch.bfloat16,
            INPUT_IS_FP16=d_out.dtype == torch.float16,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

    return out