import torch
import triton
import triton.language as tl


# Fused stages:
#   1) load broadcast mag_norm_scale
#   2) load base and lora tiles
#   3) compute DoRA compose in fp32
#   4) store result
#
# There is no additional dependent stage to fuse in this API because the caller
# already provides the materialized base/lora branches as inputs.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    ],
    key=["num_rows", "num_cols"],
)
@triton.jit
def _fused_dora_compose_kernel(
    lora_ptr,
    base_ptr,
    mag_ptr,
    out_ptr,
    num_rows,
    num_cols,
    stride_lora_r,
    stride_lora_c,
    stride_base_r,
    stride_base_c,
    stride_out_r,
    stride_out_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = row_offsets < num_rows
    col_mask = col_offsets < num_cols
    mask = row_mask[:, None] & col_mask[None, :]

    lora_ptrs = lora_ptr + row_offsets[:, None] * stride_lora_r + col_offsets[None, :] * stride_lora_c
    base_ptrs = base_ptr + row_offsets[:, None] * stride_base_r + col_offsets[None, :] * stride_base_c
    out_ptrs = out_ptr + row_offsets[:, None] * stride_out_r + col_offsets[None, :] * stride_out_c

    # Important fix:
    # Do not combine cache_modifier=".ca"/".cg" with eviction_policy on SM90A.
    # ptxas rejects those combinations. Use plain tl.load/tl.store instead.
    mag = tl.load(mag_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    lora = tl.load(lora_ptrs, mask=mask, other=0.0).to(tl.float32)
    base = tl.load(base_ptrs, mask=mask, other=0.0).to(tl.float32)

    # DoRA compose:
    # out = (mag_norm_scale - 1) * base + mag_norm_scale * (0.7 * lora)
    out = (mag[None, :] - 1.0) * base + mag[None, :] * (0.7 * lora)

    tl.store(out_ptrs, out, mask=mask)


def kernel_function(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
) -> torch.Tensor:
    assert lora.is_cuda and base.is_cuda and mag_norm_scale.is_cuda
    assert lora.shape == base.shape
    assert lora.device == base.device == mag_norm_scale.device
    assert lora.dtype == base.dtype

    if lora.numel() == 0:
        return torch.empty_like(lora)

    orig_shape = lora.shape
    num_cols = orig_shape[-1]
    assert num_cols > 0
    num_rows = lora.numel() // num_cols

    assert mag_norm_scale.numel() == num_cols

    # Canonicalize to 2D contiguous views for coalesced accesses.
    # Wrapper only validates/allocates/launches; all math stays in Triton.
    lora_2d = lora.reshape(num_rows, num_cols).contiguous()
    base_2d = base.reshape(num_rows, num_cols).contiguous()
    mag_flat = mag_norm_scale.reshape(num_cols).contiguous()

    out_2d = torch.empty_like(lora_2d)

    grid = lambda META: (
        triton.cdiv(num_rows, META["BLOCK_M"]),
        triton.cdiv(num_cols, META["BLOCK_N"]),
    )

    _fused_dora_compose_kernel[grid](
        lora_2d,
        base_2d,
        mag_flat,
        out_2d,
        num_rows,
        num_cols,
        lora_2d.stride(0),
        lora_2d.stride(1),
        base_2d.stride(0),
        base_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
    )

    return out_2d.reshape(orig_shape)