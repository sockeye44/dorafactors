import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "CHUNK_N": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512, "CHUNK_N": 64, "GROUP_M": 16}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512, "CHUNK_N": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
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
def _fused_dora_compose_kernel(
    lora_ptr,
    base_ptr,
    mag_ptr,
    out_ptr,
    num_rows,
    num_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N % CHUNK_N == 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(num_rows, BLOCK_M)
    num_pid_n = tl.cdiv(num_cols, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_base = rows[:, None] * num_cols
    cols_chunk = tl.arange(0, CHUNK_N)
    start_n = pid_n * BLOCK_N

    if not (EVEN_M and EVEN_N):
        row_mask = rows < num_rows

    for off_n in tl.static_range(0, BLOCK_N, CHUNK_N):
        cols = start_n + off_n + cols_chunk
        idx = row_base + cols[None, :]

        if EVEN_M and EVEN_N:
            mag = tl.load(mag_ptr + cols, cache_modifier=".ca").to(tl.float32)
            lora = tl.load(lora_ptr + idx, cache_modifier=".cg").to(tl.float32)
            base = tl.load(base_ptr + idx, cache_modifier=".cg").to(tl.float32)
            out = tl.fma((mag - 1.0)[None, :], base, (mag * 0.7)[None, :] * lora)
            tl.store(out_ptr + idx, out)
        else:
            col_mask = cols < num_cols
            mask = row_mask[:, None] & col_mask[None, :]
            mag = tl.load(mag_ptr + cols, mask=col_mask, other=0.0, cache_modifier=".ca").to(tl.float32)
            lora = tl.load(lora_ptr + idx, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            base = tl.load(base_ptr + idx, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            out = tl.fma((mag - 1.0)[None, :], base, (mag * 0.7)[None, :] * lora)
            tl.store(out_ptr + idx, out, mask=mask)


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

    lora_2d = lora.reshape(num_rows, num_cols).contiguous()
    base_2d = base.reshape(num_rows, num_cols).contiguous()
    mag_flat = mag_norm_scale.reshape(num_cols).contiguous()
    out_2d = torch.empty_like(lora_2d)

    grid = lambda META: (
        triton.cdiv(num_rows, META["BLOCK_M"]) * triton.cdiv(num_cols, META["BLOCK_N"]),
    )

    _fused_dora_compose_kernel[grid](
        lora_2d,
        base_2d,
        mag_flat,
        out_2d,
        num_rows,
        num_cols,
    )

    return out_2d.reshape(orig_shape)