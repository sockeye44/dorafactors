import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["num_rows", "num_cols"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["num_rows"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["num_cols"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _dora_backward_stage1_kernel(
    d_out_ptr,
    inner_ptr,
    mag_ptr,
    out_ptr,
    partial_ptr,
    stride_do0,
    stride_do1,
    stride_in0,
    stride_in1,
    stride_mag,
    stride_out0,
    stride_out1,
    stride_part0,
    stride_part1,
    num_rows,
    num_cols,
    scale,
    INPUT_IS_BF16: tl.constexpr,
    INPUT_IS_FP16: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = rows < num_rows
    col_mask = cols < num_cols

    mag = tl.load(mag_ptr + cols * stride_mag, mask=col_mask, other=0.0)
    mag_scaled = mag * scale
    mag_minus_one = mag - 1.0

    d_out_ptrs = d_out_ptr + rows[:, None] * stride_do0 + cols[None, :] * stride_do1
    inner_ptrs = inner_ptr + rows[:, None] * stride_in0 + cols[None, :] * stride_in1
    out_lora_ptrs = out_ptr + rows[:, None] * stride_out0 + cols[None, :] * stride_out1
    out_base_ptrs = out_ptr + (rows[:, None] + num_rows) * stride_out0 + cols[None, :] * stride_out1

    if EVEN_M and EVEN_N:
        d_out = tl.load(d_out_ptrs, cache_modifier=".cg")
        inner = tl.load(inner_ptrs, cache_modifier=".cg")

        d_lora = d_out * mag_scaled[None, :]
        d_base = d_out * mag_minus_one[None, :]

        tl.store(out_lora_ptrs, d_lora.to(out_ptr.dtype.element_ty))
        tl.store(out_base_ptrs, d_base.to(out_ptr.dtype.element_ty))
    else:
        mask = row_mask[:, None] & col_mask[None, :]

        d_out = tl.load(d_out_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        inner = tl.load(inner_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

        d_lora = d_out * mag_scaled[None, :]
        d_base = d_out * mag_minus_one[None, :]

        tl.store(out_lora_ptrs, d_lora.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_base_ptrs, d_base.to(out_ptr.dtype.element_ty), mask=mask)

    prod = d_out * inner
    if INPUT_IS_BF16:
        prod = prod.to(tl.bfloat16)
    elif INPUT_IS_FP16:
        prod = prod.to(tl.float16)

    d_mag_partial = tl.sum(prod.to(tl.float32), axis=0)
    partial_ptrs = partial_ptr + pid_m * stride_part0 + cols * stride_part1
    tl.store(partial_ptrs, d_mag_partial, mask=col_mask)


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["num_cols"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _dora_backward_stage2_kernel(
    partial_ptr,
    out_ptr,
    stride_part0,
    stride_part1,
    stride_out0,
    stride_out1,
    num_rows,
    num_cols,
    num_partials,
    EVEN_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < num_cols

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in tl.range(0, num_partials, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        mask = (ks[:, None] < num_partials) & col_mask[None, :]
        ptrs = partial_ptr + ks[:, None] * stride_part0 + cols[None, :] * stride_part1
        vals = tl.load(ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        acc += tl.sum(vals, axis=0)

    out_ptrs = out_ptr + (2 * num_rows) * stride_out0 + cols * stride_out1
    if EVEN_N:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))
    else:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=col_mask)


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

    assert mag_norm_scale.dtype == d_out.dtype
    if mag_norm_scale.ndim == 1:
        assert mag_norm_scale.shape[0] == c
    else:
        assert mag_norm_scale.ndim == 2
        assert mag_norm_scale.shape[0] == 1 and mag_norm_scale.shape[1] == c

    out = torch.empty((2 * r + 1, c), device=d_out.device, dtype=d_out.dtype)

    BLOCK_M = 64
    num_partials = triton.cdiv(r, BLOCK_M)
    partial = torch.empty((num_partials, c), device=d_out.device, dtype=torch.float32)

    if c > 0:
        if num_partials > 0:
            grid_stage1 = lambda META: (triton.cdiv(c, META["BLOCK_N"]), num_partials)
            _dora_backward_stage1_kernel[grid_stage1](
                d_out,
                inner,
                mag_norm_scale,
                out,
                partial,
                d_out.stride(0),
                d_out.stride(1),
                inner.stride(0),
                inner.stride(1),
                mag_norm_scale.stride(-1),
                out.stride(0),
                out.stride(1),
                partial.stride(0),
                partial.stride(1),
                r,
                c,
                0.7,
                INPUT_IS_BF16=d_out.dtype == torch.bfloat16,
                INPUT_IS_FP16=d_out.dtype == torch.float16,
                BLOCK_M=BLOCK_M,
            )

        REDUCE_BLOCK_N = 128
        _dora_backward_stage2_kernel[(triton.cdiv(c, REDUCE_BLOCK_N),)](
            partial,
            out,
            partial.stride(0),
            partial.stride(1),
            out.stride(0),
            out.stride(1),
            r,
            c,
            num_partials,
            BLOCK_K=32,
            BLOCK_N=REDUCE_BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

    return out