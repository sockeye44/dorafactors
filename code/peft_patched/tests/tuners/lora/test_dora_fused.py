"""
Comprehensive regression and performance tests for the fused DoRA kernels.

Four fused entry points are tested:
  1. Fused Element-wise Composition Kernel (fused_dora_compose)
  2. Fused Dual-Output Forward-and-Inner Kernel (fused_dora_forward_and_inner)
  3. Fused Norm Assembly Kernel (fused_norm_assembly)
  4. Custom Autograd Function with Fused Backward (FusedDoRAComposeFunction)

Tests are organized into:
  - Unit tests for each fused function (PyTorch fallback path)
  - Regression tests verifying fused == reference across dtypes, shapes, scales
  - Gradient correctness tests (autograd.gradcheck / manual comparison)
  - Integration tests with the full DoraLinearLayer forward pass
  - Edge case tests (zero scale, zero tensors, single element, large shapes)
  - Environment variable control tests
  - GPU-only tests (marked with @pytest.mark.skipif for non-CUDA environments)
  - Performance benchmarks (marked with @pytest.mark.benchmark)

GPU-dependent tests are marked with:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

For final validation with GPU, run:
    pytest tests/tuners/lora/test_dora_fused.py -v -k "cuda or gpu or triton"
"""

import gc
import logging
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F

import peft.tuners.lora.dora as dora_mod
import peft.tuners.lora.dora_fused as dora_fused_mod

logger = logging.getLogger(__name__)
from peft.tuners.lora.dora import DoraLinearLayer, DoraConv2dLayer, _invalidate_fused_cache
from peft.tuners.lora.dora_fused import (
    is_triton_available,
    fused_dora_compose,
    fused_norm_assembly,
    fused_dora_compose_autograd,
    fused_dora_forward_and_inner,
    FusedDoRAComposeFunction,
    _fused_dora_compose_torch,
    _fused_dora_forward_and_inner_torch,
    _fused_norm_assembly_torch,
    _fused_backward_torch,
    _broadcast_reduce_dims,
    _HAS_CUSTOM_OP,
)

# Import the custom op function directly for tests that need to exercise it
# in eager mode (bypassing the _is_dynamo_compiling() routing gate).
if _HAS_CUSTOM_OP:
    from peft.tuners.lora.dora_fused import _fused_dora_compose_custom_op
from peft.utils.other import transpose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()
_HAS_TRITON = is_triton_available()

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
requires_triton = pytest.mark.skipif(not _HAS_TRITON, reason="Triton required")
requires_cuda_triton = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_TRITON),
    reason="CUDA + Triton required",
)


def _device_for_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16) and not _HAS_CUDA:
        pytest.skip(f"{dtype} matmul not reliably supported on CPU")
    # Use CUDA for fp16 and bf16 so Triton kernels are actually exercised.
    return torch.device("cuda" if _HAS_CUDA and dtype in (torch.float16, torch.bfloat16) else "cpu")


def _max_diff(a, b):
    return torch.max(torch.abs(a.float() - b.float())).item()


def _relative_l2_error(a, b):
    diff = (a.to(torch.float64) - b.to(torch.float64)).norm()
    ref = b.to(torch.float64).norm().clamp_min(1e-30)
    return (diff / ref).item()


def _quantization_floor_max_abs(ref, dtype):
    ref64 = ref.to(torch.float64)
    quantized = ref64.to(dtype).to(torch.float64)
    return torch.max(torch.abs(quantized - ref64)).item()


def _max_diff_fp64(a, b):
    return torch.max(torch.abs(a.to(torch.float64) - b.to(torch.float64))).item()


def _flat_cosine_similarity(a, b):
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1), dim=1).item()


def _ref_compose(lora, base, mag, scale):
    """Reference composition (numerically stable form).

    Algebraically: mag * (scale * lora + base) - base
    Rewritten as:  (mag - 1) * base + mag * (scale * lora)
    The rewritten form avoids catastrophic cancellation in bf16/fp16
    when mag ≈ 1.

    Note: evaluation order ``mag * (scale * lora)`` matters in reduced
    precision dtypes — it must match the implementation to avoid different
    intermediate rounding.  The implementation computes ``scale * lora``
    first (matching the forward-and-inner kernel which needs
    ``scaled_lora = scale * lora`` as an intermediate for ``inner``).
    """
    return (mag - 1) * base + mag * (scale * lora)


def _ref_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, scale):
    """Reference norm assembly — matches ``_fused_norm_assembly_torch`` exactly.

    Uses Python fp64 float for scalar coefficients, which PyTorch auto-casts
    to the tensor dtype at the scalar-tensor multiply boundary.  This is
    identical to how the PyTorch fallback and (after the precision fix) the
    Triton kernel evaluate the expression.
    """
    s = float(scale)
    norm_sq = w_norm_sq + (2.0 * s) * cross_term + (s * s) * ba_norm_sq
    norm_sq = norm_sq.clamp_min_(0)
    return torch.sqrt(norm_sq)


def _random_compose_tensors(batch, out_features, dtype, device):
    """Create random tensors for composition tests."""
    lora = torch.randn(batch, out_features, dtype=dtype, device=device)
    base = torch.randn(batch, out_features, dtype=dtype, device=device)
    mag = torch.rand(1, out_features, dtype=dtype, device=device) + 0.5
    return lora, base, mag


def _random_norm_tensors(out_features, dtype, device):
    """Create random tensors for norm assembly tests."""
    w_norm_sq = torch.rand(out_features, dtype=dtype, device=device) * 10
    cross_term = torch.randn(out_features, dtype=dtype, device=device)
    ba_norm_sq = torch.rand(out_features, dtype=dtype, device=device) * 5
    return w_norm_sq, cross_term, ba_norm_sq


class _DummyTensorLike:
    """Minimal tensor-shaped stub for dispatch heuristics tests."""

    def __init__(self, shape, *, is_cuda=True):
        self.shape = torch.Size(shape)
        self.ndim = len(self.shape)
        self.is_cuda = is_cuda

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@pytest.fixture(autouse=True)
def _reset_fused_cache():
    """Invalidate the cached env var results before and after each test.

    This ensures tests that set ``PEFT_DORA_FUSED`` / ``PEFT_DORA_FUSED_BACKWARD``
    via ``monkeypatch.setenv`` see the updated values through the caching layer.
    """
    _invalidate_fused_cache()
    yield
    _invalidate_fused_cache()


class TestAutotuneConfigGrids:
    class _FakeConfig:
        def __init__(self, kwargs, num_warps=None, num_stages=None):
            self.kwargs = dict(kwargs)
            self.num_warps = num_warps
            self.num_stages = num_stages

    def _patch_comprehensive_triton(self, monkeypatch):
        monkeypatch.setattr(dora_fused_mod, "_TRITON_AVAILABLE", True)
        monkeypatch.setattr(
            dora_fused_mod,
            "triton",
            types.SimpleNamespace(Config=self._FakeConfig),
        )
        monkeypatch.setattr(dora_fused_mod, "_DORA_AUTOTUNE_COMPREHENSIVE", True)

    @pytest.mark.parametrize(
        ("num_rows", "expected_bucket"),
        [
            (1, 1),
            (2, 2),
            (3, 4),
            (8, 8),
            (129, 256),
            (32768, 32768),
            (32769, 32768),
        ],
    )
    def test_bucket_num_rows_rounds_up_and_caps(self, num_rows, expected_bucket):
        assert dora_fused_mod._bucket_num_rows(num_rows) == expected_bucket

    def test_compose_comprehensive_grid_covers_midpoint_and_ultrawide_shapes(self, monkeypatch):
        self._patch_comprehensive_triton(monkeypatch)

        configs = dora_fused_mod._compose_configs()
        pairs = {(cfg.kwargs["BLOCK_SIZE"], cfg.kwargs["ROWS_PER_PROGRAM"]) for cfg in configs}
        warps_by_bs = {}
        stages_by_bs = {}
        for cfg in configs:
            bs = cfg.kwargs["BLOCK_SIZE"]
            warps_by_bs.setdefault(bs, set()).add(cfg.num_warps)
            stages_by_bs.setdefault(bs, set()).add(cfg.num_stages)

        for pair in [(64, 2), (64, 8), (128, 1), (128, 2), (256, 1), (8192, 2)]:
            assert pair in pairs
        assert len(configs) == 97
        assert all(dora_fused_mod._is_power_of_two(cfg.kwargs["BLOCK_SIZE"]) for cfg in configs)
        assert {cfg.kwargs["BLOCK_SIZE"] for cfg in configs} == {64, 128, 256, 512, 1024, 2048, 4096, 8192}
        assert warps_by_bs[64] == {1, 2}
        assert warps_by_bs[1024] == {1, 2, 4, 8}
        assert warps_by_bs[8192] == {4, 8}
        assert stages_by_bs[64] == {1, 2, 4}
        assert stages_by_bs[4096] == {2, 3, 4, 5}

    def test_backward_comprehensive_grid_expands_rows_per_program_and_drops_32k(self, monkeypatch):
        self._patch_comprehensive_triton(monkeypatch)

        configs = dora_fused_mod._backward_configs()
        pairs = {(cfg.kwargs["BLOCK_SIZE"], cfg.kwargs["ROWS_PER_PROGRAM"]) for cfg in configs}
        warps_by_bs = {}
        stages_by_bs = {}
        for cfg in configs:
            bs = cfg.kwargs["BLOCK_SIZE"]
            warps_by_bs.setdefault(bs, set()).add(cfg.num_warps)
            stages_by_bs.setdefault(bs, set()).add(cfg.num_stages)

        for pair in [(64, 2), (64, 4), (128, 2), (256, 1), (256, 2), (8192, 1), (16384, 2)]:
            assert pair in pairs
        assert len(configs) == 97
        assert all(dora_fused_mod._is_power_of_two(cfg.kwargs["BLOCK_SIZE"]) for cfg in configs)
        assert {cfg.kwargs["BLOCK_SIZE"] for cfg in configs} == {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
        assert warps_by_bs[64] == {1, 2}
        assert warps_by_bs[16384] == {8, 16}
        assert stages_by_bs[64] == {1, 2, 4}
        assert stages_by_bs[16384] == {2, 3, 4}

    def test_norm_comprehensive_grid_adds_intermediate_block_sizes(self, monkeypatch):
        self._patch_comprehensive_triton(monkeypatch)

        configs = dora_fused_mod._norm_configs()
        block_sizes = {cfg.kwargs["BLOCK_SIZE"] for cfg in configs}
        warps_by_bs = {}
        stages_by_bs = {}
        for cfg in configs:
            bs = cfg.kwargs["BLOCK_SIZE"]
            warps_by_bs.setdefault(bs, set()).add(cfg.num_warps)
            stages_by_bs.setdefault(bs, set()).add(cfg.num_stages)

        for bs in [32, 64, 128, 256, 2048]:
            assert bs in block_sizes
        assert len(configs) == 36
        assert block_sizes == {32, 64, 128, 256, 2048}
        assert all(dora_fused_mod._is_power_of_two(bs) for bs in block_sizes)
        assert warps_by_bs[32] == {1, 2}
        assert warps_by_bs[64] == {1, 2}
        assert warps_by_bs[256] == {1, 2, 4}
        assert warps_by_bs[2048] == {4, 8}
        assert stages_by_bs[32] == {1, 2}
        assert stages_by_bs[256] == {1, 2, 3}
        assert stages_by_bs[2048] == {1, 2, 3, 4}

    def test_build_triton_configs_rejects_non_power_of_two_block_sizes(self, monkeypatch):
        self._patch_comprehensive_triton(monkeypatch)

        with pytest.raises(ValueError, match="power of two"):
            dora_fused_mod._build_triton_configs(
                [{"BLOCK_SIZE": 384}],
                lambda meta: [1],
                lambda meta: [1],
            )


# ===================================================================
# Strategy 1: Fused Element-wise Composition Kernel Tests
# ===================================================================


class TestFusedDoraComposeTorch:
    """Tests for the PyTorch fallback path of fused composition."""

    @pytest.mark.parametrize(
        "dtype,batch,out_features,scale",
        [
            (torch.float32, 1, 12, 0.0),
            (torch.float32, 4, 64, 0.3),
            (torch.float32, 32, 64, 2.5),
            (torch.float32, 32, 256, 1.0),
            (torch.float32, 4, 256, 2.5),
            (torch.float32, 1, 256, 0.3),
            (torch.bfloat16, 1, 12, 0.0),
            (torch.bfloat16, 4, 64, 0.3),
            (torch.bfloat16, 32, 64, 2.5),
            (torch.bfloat16, 32, 256, 1.0),
            (torch.bfloat16, 4, 256, 2.5),
            (torch.bfloat16, 1, 256, 0.3),
        ],
    )
    def test_torch_compose_matches_reference(self, dtype, batch, out_features, scale):
        """Fused torch compose must match reference formula."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        lora, base, mag = _random_compose_tensors(batch, out_features, dtype, device)

        ref = _ref_compose(lora.clone(), base, mag, scale)
        result = _fused_dora_compose_torch(lora.clone(), base, mag, scale, inplace=False)

        # The stable form ``(mag-1)*base + mag*s*lora`` and the reference
        # ``mag*(s*lora+base)-base`` are algebraically identical but differ
        # in bf16 rounding (the stable form is actually *more* precise).
        tol = 1e-5 if dtype == torch.float32 else 5e-3
        assert _max_diff(result, ref) <= tol, f"diff={_max_diff(result, ref)}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_torch_compose_inplace(self, dtype):
        """In-place compose must modify the lora buffer."""
        torch.manual_seed(43)
        lora, base, mag = _random_compose_tensors(8, 32, dtype, torch.device("cpu"))

        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        lora_orig = lora.clone()
        result = _fused_dora_compose_torch(lora, base, mag, 0.5, inplace=True)

        assert result.data_ptr() == lora.data_ptr(), "Should be in-place"
        # In-place now uses the canonical order lora.mul_(s).mul_(mag),
        # matching the out-of-place path — must be bitwise identical.
        assert _max_diff(result, ref) == 0.0, f"In-place diverged from reference: {_max_diff(result, ref)}"

    def test_torch_compose_not_inplace(self):
        """Out-of-place compose must not modify original buffer."""
        torch.manual_seed(44)
        lora, base, mag = _random_compose_tensors(4, 16, torch.float32, torch.device("cpu"))
        lora_copy = lora.clone()

        result = _fused_dora_compose_torch(lora, base, mag, 0.7, inplace=False)

        # Original lora should be unchanged
        assert torch.equal(lora, lora_copy), "lora should not be modified in out-of-place mode"
        ref = _ref_compose(lora_copy, base, mag, 0.7)
        assert _max_diff(result, ref) <= 1e-5

    @pytest.mark.parametrize("scale", [0.0])
    def test_zero_scale_reduces_to_identity_minus_base(self, scale):
        """When scale=0: out = mag * base - base = (mag - 1) * base."""
        torch.manual_seed(45)
        lora, base, mag = _random_compose_tensors(4, 16, torch.float32, torch.device("cpu"))
        result = _fused_dora_compose_torch(lora.clone(), base, mag, scale, inplace=False)
        ref = (mag - 1) * base
        assert _max_diff(result, ref) <= 1e-5

    def test_unit_mag_scale_reduces_to_scaled_lora(self):
        """When mag=1 everywhere: out = scale * lora."""
        torch.manual_seed(46)
        lora, base, _ = _random_compose_tensors(4, 16, torch.float32, torch.device("cpu"))
        mag = torch.ones(1, 16)
        result = _fused_dora_compose_torch(lora.clone(), base, mag, 0.7, inplace=False)
        ref = 0.7 * lora
        assert _max_diff(result, ref) <= 1e-5

    def test_3d_input(self):
        """Composition should handle 3D inputs (batch, seq, features)."""
        torch.manual_seed(47)
        batch, seq, features = 2, 8, 16
        lora = torch.randn(batch, seq, features)
        base = torch.randn(batch, seq, features)
        mag = torch.rand(1, features) + 0.5

        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        result = _fused_dora_compose_torch(lora.clone(), base, mag, 0.5, inplace=False)
        assert _max_diff(result, ref) <= 1e-5

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_reduced_precision_requires_canonical_parenthesization(self, dtype):
        """Reduced precision drifts if ``(mag * scale) * lora`` replaces ``mag * (scale * lora)``."""
        torch.manual_seed(48)
        device = torch.device("cpu")
        lora = (torch.randn(64, 256, device=device) * 256).to(dtype)
        base = (torch.randn(64, 256, device=device) * 64).to(dtype)
        mag = (1 + torch.randn(1, 256, device=device) * 0.05).to(dtype)
        scale = 0.3

        canonical = _ref_compose(lora, base, mag, scale)
        result = _fused_dora_compose_torch(lora.clone(), base, mag, scale, inplace=False)
        noncanonical = (mag - 1) * base + (mag * scale) * lora

        assert _max_diff(result, canonical) == 0.0
        assert _max_diff(canonical, noncanonical) > 0.0


@requires_cuda_triton
class TestFusedDoraComposeTriton:
    """Tests for the Triton kernel path of fused composition."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("batch", [1, 4, 32, 128])
    @pytest.mark.parametrize("out_features", [12, 64, 128, 512])
    @pytest.mark.parametrize("scale", [0.0, 0.3, 1.0, 2.5])
    def test_triton_compose_matches_reference(self, dtype, batch, out_features, scale):
        """Triton compose must match reference formula."""
        torch.manual_seed(50)
        device = torch.device("cuda")
        lora, base, mag = _random_compose_tensors(batch, out_features, dtype, device)

        ref = _ref_compose(lora.clone(), base, mag, scale)
        result = fused_dora_compose(lora.clone(), base, mag, scale, inplace=False)

        # Triton uses the same parenthesization as PyTorch, but FMA contraction
        # and backend scheduling can still change the last-bit rounding.
        # For reduced-precision dtypes, the error scales with max(1, |scale|)
        # because the FMA rounding gap is proportional to the magnitude of the
        # fused multiply-add operands.  Empirical worst-case bf16 diffs:
        #   scale=1.0 → 0.03125 (1 ULP);  scale=2.5 → 0.0625 (1 ULP)
        # 3.5e-2 * max(1, |scale|) gives ~12-40% headroom per scale point.
        tol = 1e-4 if dtype == torch.float32 else 3.5e-2 * max(1.0, abs(scale))
        assert _max_diff(result, ref) <= tol, (
            f"dtype={dtype}, batch={batch}, out={out_features}, scale={scale}, diff={_max_diff(result, ref)}"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_triton_compose_inplace(self, dtype):
        """In-place Triton compose must modify the lora buffer."""
        torch.manual_seed(51)
        device = torch.device("cuda")
        lora, base, mag = _random_compose_tensors(8, 64, dtype, device)

        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        result = fused_dora_compose(lora, base, mag, 0.5, inplace=True)

        assert result.data_ptr() == lora.data_ptr(), "Should be in-place on CUDA"
        tol = 1e-4 if dtype == torch.float32 else 5e-2
        assert _max_diff(result, ref) <= tol

    def test_triton_compose_not_inplace(self):
        """Out-of-place Triton compose must produce a new tensor."""
        torch.manual_seed(52)
        device = torch.device("cuda")
        lora, base, mag = _random_compose_tensors(4, 32, torch.float32, device)
        lora_copy = lora.clone()

        result = fused_dora_compose(lora, base, mag, 0.7, inplace=False)

        assert result.data_ptr() != lora.data_ptr(), "Should be out-of-place"
        ref = _ref_compose(lora_copy, base, mag, 0.7)
        assert _max_diff(result, ref) <= 1e-4

    def test_triton_compose_large_shape(self):
        """Smoke test: large shapes similar to real LLM hidden dims."""
        torch.manual_seed(53)
        device = torch.device("cuda")
        # batch=4, seq=2048, hidden=4096 in bf16 (like a real model layer)
        batch_seq = 4 * 2048
        hidden = 4096
        lora = torch.randn(batch_seq, hidden, dtype=torch.bfloat16, device=device)
        base = torch.randn(batch_seq, hidden, dtype=torch.bfloat16, device=device)
        mag = torch.rand(1, hidden, dtype=torch.bfloat16, device=device) + 0.5

        result = fused_dora_compose(lora.clone(), base, mag, 0.3, inplace=False)
        ref = _ref_compose(lora, base, mag, 0.3)

        assert torch.all(torch.isfinite(result))
        assert _max_diff(result, ref) <= 5e-2  # bf16 tolerance for large tensors

    def test_triton_compose_3d(self):
        """Triton compose with 3D input [batch, seq, features]."""
        torch.manual_seed(54)
        device = torch.device("cuda")
        batch, seq, features = 2, 16, 64
        lora = torch.randn(batch, seq, features, dtype=torch.float32, device=device)
        base = torch.randn(batch, seq, features, dtype=torch.float32, device=device)
        mag = torch.rand(1, features, dtype=torch.float32, device=device) + 0.5

        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        # Make contiguous 2D view for Triton
        result = fused_dora_compose(lora.clone().contiguous(), base.contiguous(), mag, 0.5, inplace=False)
        assert _max_diff(result, ref) <= 1e-4

    def test_triton_compose_large_num_rows_grid(self):
        """Large num_rows should launch correctly with row-tiling grid."""
        torch.manual_seed(55)
        device = torch.device("cuda")
        num_rows, hidden = 131072, 64  # > 65535 rows
        lora = torch.randn(num_rows, hidden, dtype=torch.float16, device=device)
        base = torch.randn(num_rows, hidden, dtype=torch.float16, device=device)
        mag = torch.rand(1, hidden, dtype=torch.float16, device=device) + 0.5

        result = fused_dora_compose(lora.clone(), base, mag, 0.3, inplace=False)
        ref = _ref_compose(lora, base, mag, 0.3)
        assert torch.all(torch.isfinite(result))
        assert _max_diff(result, ref) <= 5e-2

    def test_triton_skipped_when_dynamo_compiling(self, monkeypatch):
        """torch.compile capture should force the PyTorch fallback path."""
        torch.manual_seed(56)
        device = torch.device("cuda")
        lora, base, mag = _random_compose_tensors(8, 64, torch.float32, device)
        ref = _fused_dora_compose_torch(lora.clone(), base, mag, 0.4, inplace=False)

        def _boom(*args, **kwargs):
            raise AssertionError("Triton path should be skipped under Dynamo capture")

        monkeypatch.setattr(dora_fused_mod, "_is_dynamo_compiling", lambda: True)
        monkeypatch.setattr(dora_fused_mod, "_fused_dora_compose_triton", _boom)

        result = fused_dora_compose(lora.clone(), base, mag, 0.4, inplace=False)
        assert _max_diff(result, ref) <= 1e-5


# ===================================================================
# Mixed-dtype Triton dispatch fallback tests
# ===================================================================

# Shared parametrize lists for mixed-dtype tests.
# Tuple order: (base_dtype, lora_dtype, mag_dtype)
# Note: function sigs in dora_fused.py use (lora, base, mag) argument order.
_MIXED_DTYPE_CASES = [
    (torch.float32, torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.float16, torch.float16),
    (torch.bfloat16, torch.float32, torch.bfloat16),
    (torch.float32, torch.bfloat16, torch.float32),
    (torch.float16, torch.bfloat16, torch.float16),
]
_MATCHING_DTYPE_CASES = [
    (torch.float32, torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16, torch.float16),
]


@requires_cuda_triton
class TestTritonMixedDtypeFallback:
    """Verify Triton dispatch falls back to PyTorch when input dtypes differ.

    The Triton kernels allocate outputs as empty_like(lora), which hard-codes
    the output dtype to lora.dtype.  In mixed-dtype scenarios PyTorch would
    promote to a wider dtype, so the Triton path must be skipped to preserve
    numerical parity with the eager fallback.
    """

    @pytest.mark.parametrize("base_dtype,lora_dtype,mag_dtype", _MIXED_DTYPE_CASES)
    def test_compose_mixed_dtype_falls_back_to_torch(self, base_dtype, lora_dtype, mag_dtype, monkeypatch):
        """fused_dora_compose must skip Triton when dtypes differ."""
        torch.manual_seed(900)
        device = torch.device("cuda")
        batch, out_features = 8, 64
        lora = torch.randn(batch, out_features, dtype=lora_dtype, device=device)
        base = torch.randn(batch, out_features, dtype=base_dtype, device=device)
        mag = torch.rand(1, out_features, dtype=mag_dtype, device=device) + 0.5

        # The Triton path must NOT be called — blow up if it is.
        def _boom(*args, **kwargs):
            raise AssertionError("Triton compose should not be called for mixed dtypes")

        monkeypatch.setattr(dora_fused_mod, "_fused_dora_compose_triton", _boom)

        result = fused_dora_compose(lora, base, mag, 0.5, inplace=False)
        ref = _ref_compose(lora, base, mag, 0.5)
        # Output is cast to lora.dtype; cast ref to match before comparing.
        assert result.dtype == lora_dtype, f"Expected {lora_dtype}, got {result.dtype}"
        ref = ref.to(lora_dtype)
        tol = 1e-5 if torch.float32 in (lora_dtype, base_dtype, mag_dtype) else 2e-2
        assert _max_diff(result, ref) <= tol, f"diff={_max_diff(result, ref)}"

    @pytest.mark.parametrize("base_dtype,lora_dtype,mag_dtype", _MIXED_DTYPE_CASES)
    def test_forward_and_inner_mixed_dtype_falls_back_to_torch(self, base_dtype, lora_dtype, mag_dtype, monkeypatch):
        """fused_dora_forward_and_inner must skip Triton when dtypes differ."""
        torch.manual_seed(901)
        device = torch.device("cuda")
        batch, out_features = 8, 64
        lora = torch.randn(batch, out_features, dtype=lora_dtype, device=device)
        base = torch.randn(batch, out_features, dtype=base_dtype, device=device)
        mag = torch.rand(1, out_features, dtype=mag_dtype, device=device) + 0.5

        def _boom(*args, **kwargs):
            raise AssertionError("Triton forward_and_inner should not be called for mixed dtypes")

        monkeypatch.setattr(dora_fused_mod, "_fused_dora_forward_and_inner_triton", _boom)

        out, inner = fused_dora_forward_and_inner(lora, base, mag, 0.5)
        ref_out = _ref_compose(lora.clone(), base, mag, 0.5)
        ref_inner = 0.5 * lora + base
        # out is cast to lora.dtype; cast ref to match before comparing.
        assert out.dtype == lora_dtype, f"Expected out {lora_dtype}, got {out.dtype}"
        # inner must also be cast to lora.dtype (matches Triton contract,
        # prevents 2x VRAM when inner is saved for backward d_mag reduction).
        assert inner.dtype == lora_dtype, f"Expected inner {lora_dtype}, got {inner.dtype}"
        ref_out = ref_out.to(lora_dtype)
        ref_inner = ref_inner.to(lora_dtype)
        tol = 1e-5 if torch.float32 in (lora_dtype, base_dtype, mag_dtype) else 2e-2
        assert _max_diff(out, ref_out) <= tol, f"out diff={_max_diff(out, ref_out)}"
        assert _max_diff(inner, ref_inner) <= tol, f"inner diff={_max_diff(inner, ref_inner)}"

    @pytest.mark.parametrize("base_dtype,lora_dtype,mag_dtype", _MATCHING_DTYPE_CASES)
    def test_compose_matching_dtypes_uses_triton(self, base_dtype, lora_dtype, mag_dtype, monkeypatch):
        """When all dtypes match, Triton path must be dispatched (not just correct)."""
        torch.manual_seed(902)
        device = torch.device("cuda")
        batch, out_features = 8, 64
        lora = torch.randn(batch, out_features, dtype=lora_dtype, device=device)
        base = torch.randn(batch, out_features, dtype=base_dtype, device=device)
        mag = torch.rand(1, out_features, dtype=mag_dtype, device=device) + 0.5

        # PyTorch fallback must NOT be called — blow up if it is.
        def _boom(*args, **kwargs):
            raise AssertionError("PyTorch fallback should not be called when dtypes match")

        monkeypatch.setattr(dora_fused_mod, "_fused_dora_compose_torch", _boom)

        result = fused_dora_compose(lora.clone(), base, mag, 0.5, inplace=False)
        ref = _ref_compose(lora, base, mag, 0.5)
        tol = 1e-4 if lora_dtype == torch.float32 else 2e-2
        assert _max_diff(result, ref) <= tol, f"diff={_max_diff(result, ref)}"

    @pytest.mark.parametrize("base_dtype,lora_dtype,mag_dtype", _MATCHING_DTYPE_CASES)
    def test_forward_and_inner_matching_dtypes_uses_triton(self, base_dtype, lora_dtype, mag_dtype, monkeypatch):
        """When all dtypes match, forward_and_inner Triton path must be dispatched."""
        torch.manual_seed(905)
        device = torch.device("cuda")
        batch, out_features = 8, 64
        lora = torch.randn(batch, out_features, dtype=lora_dtype, device=device)
        base = torch.randn(batch, out_features, dtype=base_dtype, device=device)
        mag = torch.rand(1, out_features, dtype=mag_dtype, device=device) + 0.5

        # PyTorch fallback must NOT be called — blow up if it is.
        def _boom(*args, **kwargs):
            raise AssertionError("PyTorch fallback should not be called when dtypes match")

        monkeypatch.setattr(dora_fused_mod, "_fused_dora_forward_and_inner_torch", _boom)

        out, inner = fused_dora_forward_and_inner(lora.clone(), base, mag, 0.5)
        ref_out = _ref_compose(lora, base, mag, 0.5)
        ref_inner = 0.5 * lora + base
        tol = 1e-4 if lora_dtype == torch.float32 else 2e-2
        assert _max_diff(out, ref_out) <= tol, f"out diff={_max_diff(out, ref_out)}"
        assert _max_diff(inner, ref_inner) <= tol, f"inner diff={_max_diff(inner, ref_inner)}"

    def test_compose_mixed_dtype_output_matches_eager_dtype(self):
        """Mixed-dtype compose output dtype should match the torch fallback's cast-to-lora-dtype."""
        torch.manual_seed(903)
        device = torch.device("cuda")
        batch, out_features = 4, 32
        lora = torch.randn(batch, out_features, dtype=torch.bfloat16, device=device)
        base = torch.randn(batch, out_features, dtype=torch.float32, device=device)
        mag = torch.rand(1, out_features, dtype=torch.bfloat16, device=device) + 0.5

        result = fused_dora_compose(lora, base, mag, 0.5, inplace=False)
        # Both paths cast result to lora.dtype (bf16), overriding PyTorch's
        # default promotion to fp32.
        eager = _fused_dora_compose_torch(lora, base, mag, 0.5, inplace=False)
        assert result.dtype == eager.dtype, f"Output dtype mismatch: got {result.dtype}, expected {eager.dtype}"
        assert _max_diff(result, eager) <= 1e-5

    @pytest.mark.parametrize(
        "base_dtype,lora_dtype,mag_dtype",
        [
            # After forward type-promotion to fp32, d_out is fp32 but mag
            # stays bf16 → backward guard (d_out.dtype == mag.dtype) fails
            # → falls back to _fused_backward_torch.
            (torch.float32, torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32, torch.bfloat16),
        ],
    )
    def test_backward_mixed_dtype_falls_back_to_torch(self, base_dtype, lora_dtype, mag_dtype, monkeypatch):
        """FusedDoRAComposeFunction backward must skip Triton when d_out and mag dtypes differ.

        Note: this tests the backward guard in isolation via direct .apply() call.
        Through _compose_with_dispatch, the mag cast ensures mag is already in
        activation dtype, so d_out.dtype == mag.dtype holds and the backward
        guard never triggers. This test verifies the guard is correct for direct
        callers of FusedDoRAComposeFunction.
        """
        torch.manual_seed(904)
        device = torch.device("cuda")
        batch, out_features = 8, 64
        lora = torch.randn(batch, out_features, dtype=lora_dtype, device=device, requires_grad=True)
        base = torch.randn(batch, out_features, dtype=base_dtype, device=device, requires_grad=True)
        # Create mag as a leaf tensor (no + 0.5 in graph) so .grad is retained.
        mag = (torch.rand(1, out_features, dtype=mag_dtype, device=device) + 0.5).detach().requires_grad_(True)

        # Triton backward must NOT be called — blow up if it is.
        def _boom(*args, **kwargs):
            raise AssertionError("Triton backward should not be called when d_out and mag dtypes differ")

        monkeypatch.setattr(dora_fused_mod, "_fused_backward_triton", _boom)

        out = FusedDoRAComposeFunction.apply(lora, base, mag, 0.5)
        loss = out.sum()
        loss.backward()

        # Sanity: all grads should be populated and finite.
        for name, param in [("lora", lora), ("base", base), ("mag", mag)]:
            assert param.grad is not None, f"{name}.grad is None"
            assert torch.all(torch.isfinite(param.grad)), f"{name}.grad has non-finite values"

        # Correctness: compare against reference autograd through _ref_compose.
        lora_ref = lora.detach().clone().requires_grad_(True)
        base_ref = base.detach().clone().requires_grad_(True)
        mag_ref = mag.detach().clone().requires_grad_(True)
        out_ref = _ref_compose(lora_ref, base_ref, mag_ref, 0.5)
        out_ref.sum().backward()

        # Mixed-dtype means we're comparing fused-backward-torch (which
        # operates in the promoted dtype) against PyTorch autograd through
        # _ref_compose (same promoted dtype). Tolerance accounts for the
        # different evaluation order in the backward vs forward expressions.
        has_reduced = any(d in (torch.bfloat16, torch.float16) for d in (lora_dtype, base_dtype, mag_dtype))
        elem_tol = 2e-2 if has_reduced else 1e-5
        # d_mag involves a sum-reduction over broadcast dims (batch), so
        # rounding differences accumulate — needs wider tolerance.
        mag_tol = 0.1 if has_reduced else 1e-4
        for name, fused_g, ref_g in [
            ("lora", lora.grad, lora_ref.grad),
            ("base", base.grad, base_ref.grad),
            ("mag", mag.grad, mag_ref.grad),
        ]:
            tol = mag_tol if name == "mag" else elem_tol
            diff = _max_diff(fused_g, ref_g)
            assert diff <= tol, f"{name}.grad diff={diff} > tol={tol}"

    def test_amp_autocast_uses_triton_compose(self, monkeypatch):
        """End-to-end AMP test: DoraLinearLayer under autocast must use Triton compose.

        Under AMP, mag_norm_scale is fp32 (computed under _disable_autocast)
        while lora_out and base_result are bf16.  The mag cast in
        _compose_with_dispatch ensures homogeneous dtypes so Triton fires.
        Without the cast, the dtype guard would reject every AMP forward —
        functionally correct but ~2x activation memory.

        The training path (fused_backward=1) goes through
        FusedDoRAComposeFunction which calls fused_dora_forward_and_inner,
        so we monkeypatch _fused_dora_forward_and_inner_torch to verify
        the Triton forward_and_inner kernel is dispatched.
        """
        torch.manual_seed(906)
        device = torch.device("cuda")
        in_features, out_features, rank = 24, 12, 6

        base = nn.Linear(in_features, out_features, bias=True).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        lora_A = nn.Linear(in_features, rank, bias=False).to(device)
        lora_B = nn.Linear(rank, out_features, bias=False).to(device)

        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        # PyTorch forward_and_inner fallback must NOT be called — blow up if it is.
        def _boom(*args, **kwargs):
            raise AssertionError(
                "PyTorch forward_and_inner fallback should not be called under AMP — "
                "mag cast + bias cast should ensure Triton dispatch"
            )

        monkeypatch.setattr(dora_fused_mod, "_fused_dora_forward_and_inner_torch", _boom)

        x = torch.randn(4, in_features, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            base_result = base(x).detach()
            out = layer(
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=0.5,
                base_layer=base,
                base_result=base_result,
            )

        # Output should be bf16 (not promoted to fp32).
        assert out.dtype == torch.bfloat16, f"Expected bf16 output under AMP, got {out.dtype}"
        assert torch.all(torch.isfinite(out)), "Output has non-finite values"

    def test_conv2d_amp_autocast_bias_dtype_preserved(self):
        """Conv2d DoRA under AMP: bias cast must preserve bf16 base_result dtype.

        The conv bias subtraction (base_result - bias) would promote base_result
        from bf16 to fp32 without the bias.to(base_result.dtype) cast. This test
        verifies the bias cast keeps base_result in bf16 so downstream compose
        receives homogeneous activation dtypes.

        Note: Triton never dispatches for conv shapes (_mag_broadcasts_last_dim
        returns False for [C,1,1,...] mag), so this test verifies dtype
        preservation rather than Triton dispatch.
        """
        torch.manual_seed(907)
        device = torch.device("cuda")
        in_channels, out_channels, rank = 4, 5, 3

        base = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        lora_A = nn.Conv2d(in_channels, rank, 3, padding=1, bias=False).to(device)
        lora_B = nn.Conv2d(rank, out_channels, 1, bias=False).to(device)

        layer = DoraConv2dLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(2, in_channels, 8, 8, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            base_result = base(x).detach()
            out = layer(
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=0.5,
                base_layer=base,
                base_result=base_result,
            )

        # Conv output under AMP should stay bf16 (bias cast prevents promotion).
        assert out.dtype == torch.bfloat16, f"Expected bf16 output under AMP, got {out.dtype}"
        assert torch.all(torch.isfinite(out)), "Output has non-finite values"

    def test_conv2d_amp_training_gradients(self):
        """Conv2d DoRA under AMP produces finite gradients for all parameters.

        Conv is always eager (Triton can't broadcast [C,1,1,...] mag).
        This test verifies backward under AMP produces finite, non-None
        gradients for lora_A, lora_B, and magnitude.
        """
        torch.manual_seed(910)
        device = torch.device("cuda")
        in_channels, out_channels, rank = 4, 5, 3

        base = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        lora_A = nn.Conv2d(in_channels, rank, 3, padding=1, bias=False).to(device)
        lora_B = nn.Conv2d(rank, out_channels, 1, bias=False).to(device)

        layer = DoraConv2dLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(2, in_channels, 8, 8, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            base_result = base(x).detach()
            out = layer(
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=0.5,
                base_layer=base,
                base_result=base_result,
            )

        assert out.dtype == torch.bfloat16, f"Expected bf16 output, got {out.dtype}"
        out.sum().backward()

        assert lora_A.weight.grad is not None, "lora_A gradient is None"
        assert lora_B.weight.grad is not None, "lora_B gradient is None"
        assert layer.weight.grad is not None, "magnitude gradient is None"

        assert torch.all(torch.isfinite(lora_A.weight.grad)), "lora_A grad has non-finite values"
        assert torch.all(torch.isfinite(lora_B.weight.grad)), "lora_B grad has non-finite values"
        assert torch.all(torch.isfinite(layer.weight.grad)), "magnitude grad has non-finite values"

    def test_amp_autocast_chunks_path_bf16(self, monkeypatch):
        """The base_result=None chunk path under AMP must also produce bf16 output.

        When base_result is not precomputed, DoraLinearLayer takes the
        _compose_with_base_chunks path which computes F.linear per chunk.
        The eager chunk helper keeps fp32 magnitude and copies the promoted
        stable-form result back into the bf16 activation buffer.
        """
        torch.manual_seed(908)
        device = torch.device("cuda")
        in_features, out_features, rank = 24, 12, 6

        base = nn.Linear(in_features, out_features, bias=True).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        lora_A = nn.Linear(in_features, rank, bias=False).to(device)
        lora_B = nn.Linear(rank, out_features, bias=False).to(device)

        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()

        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(4, in_features, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Pass base_result=None to exercise _compose_with_base_chunks
            out = layer(
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=0.5,
                base_layer=base,
                base_result=None,
            )

        assert out.dtype == torch.bfloat16, f"Expected bf16 output from chunks path under AMP, got {out.dtype}"
        assert torch.all(torch.isfinite(out)), "Output has non-finite values"

    def test_amp_eager_training_path_bf16(self, monkeypatch):
        """Eager training path (fused_backward=0) under AMP must produce bf16 output.

        When PEFT_DORA_FUSED_BACKWARD=0 and the tensor is below the auto-fuse
        threshold, _compose_with_dispatch falls through to the eager training
        path.  That path intentionally keeps fp32 magnitude so the stable form
        evaluates in fp32 before the final cast back to the activation dtype.
        """
        torch.manual_seed(909)
        device = torch.device("cuda")
        in_features, out_features, rank = 24, 12, 6

        base = nn.Linear(in_features, out_features, bias=True).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        lora_A = nn.Linear(in_features, rank, bias=False).to(device)
        lora_B = nn.Linear(rank, out_features, bias=False).to(device)

        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()

        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(4, in_features, device=device, requires_grad=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            base_result = base(x).detach()
            out = layer(
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=0.5,
                base_layer=base,
                base_result=base_result,
            )

        assert out.dtype == torch.bfloat16, f"Expected bf16 output from eager path under AMP, got {out.dtype}"
        assert torch.all(torch.isfinite(out)), "Output has non-finite values"
        # Verify backward still works
        out.sum().backward()
        assert x.grad is not None


# ===================================================================
# Strategy 1b: Fused Forward-and-Inner Kernel Tests
# ===================================================================


class TestFusedForwardAndInnerTorch:
    """Tests for the PyTorch fallback path of fused forward-and-inner."""

    @pytest.mark.parametrize(
        "dtype,batch,out_features,scale",
        [
            (torch.float32, 1, 12, 0.0),
            (torch.float32, 4, 64, 0.3),
            (torch.float32, 32, 256, 1.0),
            (torch.float32, 4, 256, 2.5),
            (torch.bfloat16, 4, 64, 0.3),
            (torch.bfloat16, 32, 256, 1.0),
        ],
    )
    def test_torch_forward_and_inner_matches_reference(self, dtype, batch, out_features, scale):
        """Both out and inner must match reference formulas."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        lora, base, mag = _random_compose_tensors(batch, out_features, dtype, device)

        out, inner = _fused_dora_forward_and_inner_torch(lora, base, mag, scale)
        ref_out = _ref_compose(lora.clone(), base, mag, scale)
        ref_inner = scale * lora + base

        tol = 1e-5 if dtype == torch.float32 else 5e-3
        assert _max_diff(out, ref_out) <= tol, f"out diff={_max_diff(out, ref_out)}"
        assert _max_diff(inner, ref_inner) <= tol, f"inner diff={_max_diff(inner, ref_inner)}"

    def test_torch_forward_and_inner_does_not_modify_inputs(self):
        """Inputs must not be modified."""
        torch.manual_seed(43)
        lora, base, mag = _random_compose_tensors(4, 16, torch.float32, torch.device("cpu"))
        lora_copy = lora.clone()
        base_copy = base.clone()

        _fused_dora_forward_and_inner_torch(lora, base, mag, 0.5)

        assert torch.equal(lora, lora_copy), "lora should not be modified"
        assert torch.equal(base, base_copy), "base should not be modified"

    def test_torch_forward_and_inner_3d_input(self):
        """Should handle 3D inputs [batch, seq, features]."""
        torch.manual_seed(44)
        batch, seq, features = 2, 8, 16
        lora = torch.randn(batch, seq, features)
        base = torch.randn(batch, seq, features)
        mag = torch.rand(1, features) + 0.5

        out, inner = _fused_dora_forward_and_inner_torch(lora, base, mag, 0.5)
        ref_out = _ref_compose(lora.clone(), base, mag, 0.5)
        ref_inner = 0.5 * lora + base

        assert _max_diff(out, ref_out) <= 1e-5
        assert _max_diff(inner, ref_inner) <= 1e-5

    def test_zero_scale_inner_equals_base(self):
        """When scale=0, inner = base and out = (mag-1)*base."""
        torch.manual_seed(45)
        lora, base, mag = _random_compose_tensors(4, 16, torch.float32, torch.device("cpu"))

        out, inner = _fused_dora_forward_and_inner_torch(lora, base, mag, 0.0)
        assert _max_diff(inner, base) <= 1e-5
        assert _max_diff(out, (mag - 1) * base) <= 1e-5

    def test_unit_mag_out_equals_scaled_lora(self):
        """When mag=1, out = scale*lora."""
        torch.manual_seed(46)
        lora, base, _ = _random_compose_tensors(4, 16, torch.float32, torch.device("cpu"))
        mag = torch.ones(1, 16)

        out, inner = _fused_dora_forward_and_inner_torch(lora, base, mag, 0.7)
        assert _max_diff(out, 0.7 * lora) <= 1e-5
        assert _max_diff(inner, 0.7 * lora + base) <= 1e-5


@requires_cuda_triton
class TestFusedForwardAndInnerTriton:
    """Tests for the Triton kernel path of fused forward-and-inner."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("batch", [1, 4, 32, 128])
    @pytest.mark.parametrize("out_features", [12, 64, 128, 512])
    @pytest.mark.parametrize("scale", [0.0, 0.3, 1.0])
    def test_triton_forward_and_inner_matches_reference(self, dtype, batch, out_features, scale):
        """Both out and inner from Triton kernel must match reference."""
        torch.manual_seed(50)
        device = torch.device("cuda")
        lora, base, mag = _random_compose_tensors(batch, out_features, dtype, device)

        out, inner = fused_dora_forward_and_inner(lora, base, mag, scale)
        ref_out = _ref_compose(lora.clone(), base, mag, scale)
        ref_inner = scale * lora + base

        tol = 1e-4 if dtype == torch.float32 else 5e-2
        assert _max_diff(out, ref_out) <= tol, (
            f"out: dtype={dtype}, batch={batch}, out={out_features}, scale={scale}, diff={_max_diff(out, ref_out)}"
        )
        assert _max_diff(inner, ref_inner) <= tol, (
            f"inner: dtype={dtype}, batch={batch}, out={out_features}, scale={scale}, "
            f"diff={_max_diff(inner, ref_inner)}"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_triton_forward_and_inner_does_not_modify_inputs(self, dtype):
        """Inputs must not be modified by the Triton kernel."""
        torch.manual_seed(51)
        device = torch.device("cuda")
        lora, base, mag = _random_compose_tensors(8, 64, dtype, device)
        lora_copy = lora.clone()
        base_copy = base.clone()

        fused_dora_forward_and_inner(lora, base, mag, 0.5)

        assert torch.equal(lora, lora_copy), "lora modified by Triton kernel"
        assert torch.equal(base, base_copy), "base modified by Triton kernel"

    def test_triton_forward_and_inner_large_shape(self):
        """Smoke test with real-LLM-sized tensors."""
        torch.manual_seed(52)
        device = torch.device("cuda")
        batch_seq = 4 * 2048
        hidden = 4096
        lora = torch.randn(batch_seq, hidden, dtype=torch.bfloat16, device=device)
        base = torch.randn(batch_seq, hidden, dtype=torch.bfloat16, device=device)
        mag = torch.rand(1, hidden, dtype=torch.bfloat16, device=device) + 0.5

        out, inner = fused_dora_forward_and_inner(lora, base, mag, 0.3)
        ref_out = _ref_compose(lora, base, mag, 0.3)
        ref_inner = 0.3 * lora + base

        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isfinite(inner))
        assert _max_diff(out, ref_out) <= 5e-2
        assert _max_diff(inner, ref_inner) <= 5e-2

    def test_triton_forward_and_inner_3d(self):
        """Triton forward-and-inner with 3D input [batch, seq, features]."""
        torch.manual_seed(53)
        device = torch.device("cuda")
        batch, seq, features = 2, 16, 64
        lora = torch.randn(batch, seq, features, dtype=torch.float32, device=device)
        base = torch.randn(batch, seq, features, dtype=torch.float32, device=device)
        mag = torch.rand(1, features, dtype=torch.float32, device=device) + 0.5

        out, inner = fused_dora_forward_and_inner(
            lora.contiguous(),
            base.contiguous(),
            mag,
            0.5,
        )
        ref_out = _ref_compose(lora.clone(), base, mag, 0.5)
        ref_inner = 0.5 * lora + base

        assert _max_diff(out, ref_out) <= 1e-4
        assert _max_diff(inner, ref_inner) <= 1e-4

    def test_triton_forward_and_inner_matches_torch_fallback(self):
        """Triton path must match the PyTorch fallback exactly (fp32)."""
        torch.manual_seed(54)
        device = torch.device("cuda")
        lora, base, mag = _random_compose_tensors(16, 128, torch.float32, device)

        out_triton, inner_triton = fused_dora_forward_and_inner(lora, base, mag, 0.7)
        out_torch, inner_torch = _fused_dora_forward_and_inner_torch(lora, base, mag, 0.7)

        assert _max_diff(out_triton, out_torch) <= 1e-4
        assert _max_diff(inner_triton, inner_torch) <= 1e-4

    def test_triton_skipped_when_dynamo_compiling(self, monkeypatch):
        """torch.compile capture should force the PyTorch fallback path."""
        torch.manual_seed(55)
        device = torch.device("cuda")
        lora, base, mag = _random_compose_tensors(8, 64, torch.float32, device)

        ref_out, ref_inner = _fused_dora_forward_and_inner_torch(lora, base, mag, 0.4)

        def _boom(*args, **kwargs):
            raise AssertionError("Triton path should be skipped under Dynamo capture")

        monkeypatch.setattr(dora_fused_mod, "_is_dynamo_compiling", lambda: True)
        monkeypatch.setattr(dora_fused_mod, "_fused_dora_forward_and_inner_triton", _boom)

        out, inner = fused_dora_forward_and_inner(lora, base, mag, 0.4)
        assert _max_diff(out, ref_out) <= 1e-5
        assert _max_diff(inner, ref_inner) <= 1e-5


# ===================================================================
# Strategy 2: Fused Norm Assembly Kernel Tests
# ===================================================================


class TestFusedNormAssemblyTorch:
    """Tests for the PyTorch fallback path of fused norm assembly."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("out_features", [1, 16, 64, 256, 1024])
    @pytest.mark.parametrize("scale", [0.0, 0.1, 0.5, 1.0, 2.0])
    def test_torch_norm_matches_reference(self, dtype, out_features, scale):
        """Torch norm assembly must match reference formula."""
        torch.manual_seed(60)
        device = torch.device("cpu")
        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, dtype, device)

        ref = _ref_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, scale)
        (result,) = _fused_norm_assembly_torch(w_norm_sq, cross_term, ba_norm_sq, scale)

        tol = 1e-5 if dtype == torch.float32 else 1e-10
        assert _max_diff(result, ref) <= tol

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_torch_norm_then_pytorch_division(self, dtype):
        """Norm-only assembly + PyTorch division matches fused reference."""
        torch.manual_seed(61)
        device = torch.device("cpu")
        out_features = 64
        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, dtype, device)
        magnitude = torch.rand(out_features, dtype=dtype, device=device) + 0.1

        eps = 1e-6
        (weight_norm,) = _fused_norm_assembly_torch(
            w_norm_sq,
            cross_term,
            ba_norm_sq,
            0.5,
        )
        # Division always in PyTorch (de-fused path)
        mag_scale = magnitude / weight_norm.clamp_min(eps)

        ref_norm = _ref_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, 0.5)
        ref_scale = magnitude / ref_norm.clamp_min(eps)

        assert _max_diff(weight_norm, ref_norm) <= 1e-5
        assert _max_diff(mag_scale, ref_scale) <= 1e-5

    def test_negative_norm_sq_clamped(self):
        """Negative norm_sq values should be clamped to 0."""
        torch.manual_seed(62)
        out_features = 16
        # Create values where norm_sq would be negative
        w_norm_sq = torch.zeros(out_features)
        cross_term = -torch.ones(out_features) * 100
        ba_norm_sq = torch.ones(out_features)

        (result,) = _fused_norm_assembly_torch(w_norm_sq, cross_term, ba_norm_sq, 0.01)
        assert torch.all(result >= 0), "All norm values should be non-negative"
        assert torch.all(torch.isfinite(result)), "All values should be finite"

    def test_zero_scale_returns_base_norm(self):
        """With scale=0, norm should equal sqrt(w_norm_sq)."""
        torch.manual_seed(63)
        out_features = 32
        w_norm_sq = torch.rand(out_features) * 10 + 0.1
        cross_term = torch.randn(out_features)
        ba_norm_sq = torch.rand(out_features)

        (result,) = _fused_norm_assembly_torch(w_norm_sq, cross_term, ba_norm_sq, 0.0)
        ref = torch.sqrt(w_norm_sq)
        assert _max_diff(result, ref) <= 1e-6


@requires_cuda_triton
class TestFusedNormAssemblyTriton:
    """Tests for the Triton kernel path of fused norm assembly."""

    @pytest.mark.parametrize("out_features", [1, 16, 64, 256, 1024, 4096])
    @pytest.mark.parametrize("scale", [0.0, 0.1, 0.5, 1.0, 2.0])
    def test_triton_norm_matches_reference(self, out_features, scale):
        """Triton norm assembly must match reference formula."""
        torch.manual_seed(70)
        device = torch.device("cuda")
        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, torch.float32, device)

        ref = _ref_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, scale)
        (result,) = fused_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, scale)

        assert _max_diff(result, ref) <= 1e-6, f"out={out_features}, scale={scale}, diff={_max_diff(result, ref)}"

    def test_triton_norm_then_pytorch_division(self):
        """Triton norm-only + PyTorch division matches reference."""
        torch.manual_seed(71)
        device = torch.device("cuda")
        out_features = 256
        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, torch.float32, device)
        magnitude = torch.rand(out_features, dtype=torch.float32, device=device) + 0.1
        eps = 1e-6

        (weight_norm,) = fused_norm_assembly(
            w_norm_sq,
            cross_term,
            ba_norm_sq,
            0.5,
        )
        # Division in PyTorch (de-fused path)
        mag_scale = magnitude / weight_norm.clamp_min(eps)

        ref_norm = _ref_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, 0.5)
        ref_scale = magnitude / ref_norm.clamp_min(eps)

        assert _max_diff(weight_norm, ref_norm) <= 1e-6
        assert _max_diff(mag_scale, ref_scale) <= 1e-6

    def test_triton_norm_then_pytorch_division_bf16(self):
        """Triton norm-only + PyTorch division in bf16 — tighter agreement after de-fusion."""
        if not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA bf16 not supported on this device")

        torch.manual_seed(711)
        device = torch.device("cuda")
        out_features = 256
        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, torch.bfloat16, device)
        magnitude = torch.rand(out_features, dtype=torch.bfloat16, device=device) + 0.1
        eps = 1e-6

        (weight_norm,) = fused_norm_assembly(
            w_norm_sq,
            cross_term,
            ba_norm_sq,
            0.5,
        )
        # Division in PyTorch (identical to eager path after de-fusion)
        mag_scale = magnitude / weight_norm.clamp_min(eps)

        ref_norm = _ref_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, 0.5)
        ref_scale = magnitude / ref_norm.clamp_min(eps)

        assert _max_diff(weight_norm, ref_norm) <= 5e-2
        # After de-fusion, the division is identical in both paths, so tolerance
        # only depends on the norm difference (not amplified by Triton division).
        rel_diff = (mag_scale.float() - ref_scale.float()).abs() / ref_scale.float().abs().clamp_min(1e-6)
        assert rel_diff.max().item() <= 5e-2, f"mag_scale relative diff: {rel_diff.max().item():.4f}"

    def test_triton_norm_large_features(self):
        """Smoke test with large feature dim (like 8192 in LLMs)."""
        torch.manual_seed(72)
        device = torch.device("cuda")
        out_features = 8192
        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, torch.float32, device)

        (result,) = fused_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, 1.0)
        assert torch.all(torch.isfinite(result))
        assert result.shape == (out_features,)

    def test_triton_skipped_when_dynamo_compiling_norm(self, monkeypatch):
        """fused_norm_assembly should fall back to PyTorch under Dynamo."""
        torch.manual_seed(73)
        device = torch.device("cuda")
        out_features = 64
        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, torch.float32, device)

        ref = _ref_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, 0.5)

        def _boom(*args, **kwargs):
            raise AssertionError("Triton norm path should be skipped under Dynamo capture")

        monkeypatch.setattr(dora_fused_mod, "_is_dynamo_compiling", lambda: True)
        monkeypatch.setattr(dora_fused_mod, "_fused_norm_assembly_triton", _boom)

        (result,) = fused_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, 0.5)
        assert _max_diff(result, ref) <= 1e-5


# ===================================================================
# Strategy 3: Custom Autograd Function Tests
# ===================================================================


class TestFusedDoRAAutograd:
    """Tests for the custom autograd function with fused backward."""

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("batch", [1, 4, 16])
    @pytest.mark.parametrize("out_features", [8, 32, 64])
    @pytest.mark.parametrize("scale", [0.3, 1.0])
    def test_autograd_forward_matches_reference(self, dtype, batch, out_features, scale):
        """Autograd forward must match reference."""
        torch.manual_seed(80)
        device = torch.device("cpu")
        lora = torch.randn(batch, out_features, dtype=dtype, device=device, requires_grad=True)
        base = torch.randn(batch, out_features, dtype=dtype, device=device, requires_grad=True)
        mag = (torch.rand(1, out_features, dtype=dtype, device=device) + 0.5).requires_grad_(True)

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        ref = _ref_compose(lora.detach().clone(), base.detach().clone(), mag.detach().clone(), scale)

        assert _max_diff(result, ref) <= 1e-5

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("batch", [1, 4])
    @pytest.mark.parametrize("out_features", [8, 32])
    @pytest.mark.parametrize("scale", [0.3, 1.0])
    def test_autograd_backward_lora_grad(self, dtype, batch, out_features, scale):
        """d_lora = mag * scale * d_out."""
        torch.manual_seed(81)
        device = torch.device("cpu")
        lora = torch.randn(batch, out_features, dtype=dtype, device=device, requires_grad=True)
        base = torch.randn(batch, out_features, dtype=dtype, device=device)
        mag = torch.rand(1, out_features, dtype=dtype, device=device) + 0.5

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        expected_d_lora = mag * scale * d_out
        assert _max_diff(lora.grad, expected_d_lora) <= 1e-5

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_autograd_backward_base_grad(self, dtype):
        """d_base = (mag - 1) * d_out."""
        torch.manual_seed(82)
        device = torch.device("cpu")
        batch, out_features = 4, 16
        lora = torch.randn(batch, out_features, dtype=dtype, device=device)
        base = torch.randn(batch, out_features, dtype=dtype, device=device, requires_grad=True)
        mag = torch.rand(1, out_features, dtype=dtype, device=device) + 0.5

        result = fused_dora_compose_autograd(lora, base, mag, scale=0.7)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        expected_d_base = (mag - 1) * d_out
        assert _max_diff(base.grad, expected_d_base) <= 1e-5

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_autograd_backward_mag_grad(self, dtype):
        """d_mag = sum((scale * lora + base) * d_out, broadcast dims)."""
        torch.manual_seed(83)
        device = torch.device("cpu")
        batch, out_features = 4, 16
        lora = torch.randn(batch, out_features, dtype=dtype, device=device)
        base = torch.randn(batch, out_features, dtype=dtype, device=device)
        mag = (torch.rand(1, out_features, dtype=dtype, device=device) + 0.5).requires_grad_(True)
        scale = 0.7

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        inner = scale * lora + base
        expected_d_mag = (inner * d_out).sum(dim=0, keepdim=True)
        assert _max_diff(mag.grad, expected_d_mag) <= 1e-5

    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_autograd_all_grads_simultaneously(self, dtype):
        """All three gradients should be correct simultaneously."""
        torch.manual_seed(84)
        device = torch.device("cpu")
        batch, out_features = 8, 32
        scale = 0.5

        lora = torch.randn(batch, out_features, dtype=dtype, device=device, requires_grad=True)
        base = torch.randn(batch, out_features, dtype=dtype, device=device, requires_grad=True)
        mag = (torch.rand(1, out_features, dtype=dtype, device=device) + 0.5).requires_grad_(True)

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        # Check all gradients
        assert _max_diff(lora.grad, mag * scale * d_out) <= 1e-5
        assert _max_diff(base.grad, (mag - 1) * d_out) <= 1e-5
        expected_d_mag = ((scale * lora.detach() + base.detach()) * d_out).sum(dim=0, keepdim=True)
        assert _max_diff(mag.grad, expected_d_mag) <= 1e-5

    def test_autograd_no_grad_inputs(self):
        """When no inputs require grad, backward should not error."""
        torch.manual_seed(85)
        lora = torch.randn(4, 16)
        base = torch.randn(4, 16)
        mag = torch.rand(1, 16) + 0.5

        result = fused_dora_compose_autograd(lora, base, mag, 0.5)
        assert not result.requires_grad

    def test_autograd_3d_input(self):
        """Autograd should handle 3D inputs [batch, seq, features]."""
        torch.manual_seed(86)
        batch, seq, features = 2, 8, 16
        lora = torch.randn(batch, seq, features, requires_grad=True)
        base = torch.randn(batch, seq, features, requires_grad=True)
        mag = (torch.rand(1, features) + 0.5).requires_grad_(True)

        result = fused_dora_compose_autograd(lora, base, mag, 0.5)
        loss = result.sum()
        loss.backward()

        assert lora.grad is not None
        assert base.grad is not None
        assert mag.grad is not None
        assert lora.grad.shape == lora.shape
        assert base.grad.shape == base.shape
        assert mag.grad.shape == mag.shape

    def test_autograd_api_falls_back_when_dynamo_compiling(self, monkeypatch):
        """fused_dora_compose_autograd should bypass custom Function under compile.

        When _HAS_CUSTOM_OP is True, the custom_op path is taken instead of
        Function.apply — this test disables it to exercise the legacy fallback.
        """
        torch.manual_seed(86)
        lora = torch.randn(3, 12, requires_grad=True)
        base = torch.randn(3, 12, requires_grad=True)
        mag = (torch.rand(1, 12) + 0.5).requires_grad_(True)

        class _Stub:
            @staticmethod
            def apply(*args, **kwargs):
                raise AssertionError("Custom Function path should be skipped")

        monkeypatch.setattr(dora_fused_mod, "_HAS_CUSTOM_OP", False)
        monkeypatch.setattr(dora_fused_mod, "_is_dynamo_compiling", lambda: True)
        monkeypatch.setattr(dora_fused_mod, "FusedDoRAComposeFunction", _Stub)

        result = fused_dora_compose_autograd(lora, base, mag, 0.5)
        ref = _ref_compose(lora, base, mag, 0.5)
        assert _max_diff(result, ref) <= 1e-5

    def test_autograd_dynamo_guard_backward_produces_correct_grads(self, monkeypatch):
        """When Dynamo is compiling and no custom_op is available, the
        fused_dora_compose_autograd fallback must still produce correct
        gradients via standard autograd.
        """
        torch.manual_seed(88)
        lora = torch.randn(4, 16, requires_grad=True)
        base = torch.randn(4, 16, requires_grad=True)
        mag = (torch.rand(1, 16) + 0.5).requires_grad_(True)
        scale = 0.7

        monkeypatch.setattr(dora_fused_mod, "_HAS_CUSTOM_OP", False)
        monkeypatch.setattr(dora_fused_mod, "_is_dynamo_compiling", lambda: True)

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        # Verify correct gradient values
        expected_d_lora = mag * scale * d_out
        expected_d_base = (mag - 1) * d_out
        inner = scale * lora.detach() + base.detach()
        expected_d_mag = (inner * d_out).sum(dim=0, keepdim=True)

        assert _max_diff(lora.grad, expected_d_lora) <= 1e-5
        assert _max_diff(base.grad, expected_d_base) <= 1e-5
        assert _max_diff(mag.grad, expected_d_mag) <= 1e-5

    @pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op not available")
    def test_custom_op_forward_matches_reference(self):
        """peft::fused_dora_compose custom op must match reference formula.

        Calls _fused_dora_compose_custom_op directly to exercise the custom op's
        forward + setup_context path in eager mode (not routed through
        fused_dora_compose_autograd which always uses FusedDoRAComposeFunction).
        """
        torch.manual_seed(90)
        lora = torch.randn(4, 16, requires_grad=True)
        base = torch.randn(4, 16, requires_grad=True)
        mag = (torch.rand(1, 16) + 0.5).requires_grad_(True)
        scale = 0.5

        result = _fused_dora_compose_custom_op(lora, base, mag, scale)
        ref = _ref_compose(lora.detach(), base.detach(), mag.detach(), scale)
        assert _max_diff(result, ref) <= 1e-5

    @pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op not available")
    def test_custom_op_backward_produces_correct_grads(self):
        """peft::fused_dora_compose custom op backward must produce correct grads.

        Calls _fused_dora_compose_custom_op directly to exercise the custom op's
        registered autograd (setup_context → backward) in eager mode.
        """
        torch.manual_seed(91)
        lora = torch.randn(4, 16, requires_grad=True)
        base = torch.randn(4, 16, requires_grad=True)
        mag = (torch.rand(1, 16) + 0.5).requires_grad_(True)
        scale = 0.7

        result = _fused_dora_compose_custom_op(lora, base, mag, scale)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        expected_d_lora = mag.detach() * scale * d_out
        expected_d_base = (mag.detach() - 1) * d_out
        inner = scale * lora.detach() + base.detach()
        expected_d_mag = (inner * d_out).sum(dim=0, keepdim=True)

        assert _max_diff(lora.grad, expected_d_lora) <= 1e-5
        assert _max_diff(base.grad, expected_d_base) <= 1e-5
        assert _max_diff(mag.grad, expected_d_mag) <= 1e-5

    @pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op not available")
    def test_custom_op_frozen_mag_forward(self):
        """Custom op with frozen mag must match reference (no inner allocation).

        Calls _fused_dora_compose_custom_op directly — verifies setup_context
        skips inner allocation when mag doesn't require grad.
        """
        torch.manual_seed(92)
        lora = torch.randn(4, 16, requires_grad=True)
        base = torch.randn(4, 16, requires_grad=True)
        mag = torch.rand(1, 16) + 0.5  # no requires_grad

        result = _fused_dora_compose_custom_op(lora, base, mag, 0.5)
        ref = _ref_compose(lora.detach(), base.detach(), mag, 0.5)
        assert _max_diff(result, ref) <= 1e-5

    @pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op not available")
    def test_custom_op_frozen_mag_backward(self):
        """Custom op with frozen mag: d_lora and d_base correct, no d_mag.

        Calls _fused_dora_compose_custom_op directly — verifies the registered
        backward handles the frozen-mag codepath (no inner saved, no d_mag).
        """
        torch.manual_seed(93)
        lora = torch.randn(4, 16, requires_grad=True)
        base = torch.randn(4, 16, requires_grad=True)
        mag = torch.rand(1, 16) + 0.5  # frozen
        scale = 0.7

        result = _fused_dora_compose_custom_op(lora, base, mag, scale)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        assert lora.grad is not None
        assert base.grad is not None
        assert _max_diff(lora.grad, mag * scale * d_out) <= 1e-5
        assert _max_diff(base.grad, (mag - 1) * d_out) <= 1e-5

    def test_gradcheck_float64(self):
        """torch.autograd.gradcheck with float64 for numerical accuracy."""
        torch.manual_seed(87)
        # Small tensors for gradcheck (it's expensive)
        lora = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        base = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        mag = (torch.rand(1, 4, dtype=torch.float64) + 0.5).requires_grad_(True)
        scale = 0.7

        assert torch.autograd.gradcheck(
            lambda l, b, m: FusedDoRAComposeFunction.apply(l, b, m, scale),
            (lora, base, mag),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    # -- Frozen-mag path (conditional inner allocation) ----------------------

    def test_frozen_mag_forward_matches_reference(self):
        """Forward must match reference when mag_norm_scale.requires_grad=False."""
        torch.manual_seed(200)
        batch, out_features, scale = 8, 32, 0.7
        lora = torch.randn(batch, out_features, requires_grad=True)
        base = torch.randn(batch, out_features, requires_grad=True)
        mag = torch.rand(1, out_features) + 0.5  # no requires_grad

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        ref = _ref_compose(lora.detach(), base.detach(), mag, scale)

        assert _max_diff(result, ref) <= 1e-5

    def test_frozen_mag_backward_produces_correct_lora_base_grads(self):
        """With frozen mag, d_lora and d_base must be correct and d_mag must be None."""
        torch.manual_seed(201)
        batch, out_features, scale = 8, 32, 0.5
        lora = torch.randn(batch, out_features, requires_grad=True)
        base = torch.randn(batch, out_features, requires_grad=True)
        mag = torch.rand(1, out_features) + 0.5  # frozen

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        assert lora.grad is not None
        assert base.grad is not None
        assert _max_diff(lora.grad, mag * scale * d_out) <= 1e-5
        assert _max_diff(base.grad, (mag - 1) * d_out) <= 1e-5

    def test_frozen_mag_backward_d_mag_is_none(self):
        """d_mag must be None when mag is frozen — inner was never saved."""
        torch.manual_seed(202)
        lora = torch.randn(4, 16, requires_grad=True)
        base = torch.randn(4, 16, requires_grad=True)
        mag = torch.rand(1, 16) + 0.5  # frozen

        result = FusedDoRAComposeFunction.apply(lora, base, mag, 0.5)
        d_out = torch.randn_like(result)
        grads = torch.autograd.grad(result, [lora, base], d_out)
        # mag was not in the grad request (frozen), so we just verify no crash
        assert grads[0] is not None  # d_lora
        assert grads[1] is not None  # d_base

    def test_frozen_mag_3d_input(self):
        """Frozen-mag path should handle 3D inputs [batch, seq, features]."""
        torch.manual_seed(203)
        batch, seq, features = 2, 8, 16
        lora = torch.randn(batch, seq, features, requires_grad=True)
        base = torch.randn(batch, seq, features, requires_grad=True)
        mag = torch.rand(1, features) + 0.5  # frozen

        result = fused_dora_compose_autograd(lora, base, mag, 0.5)
        loss = result.sum()
        loss.backward()

        assert lora.grad is not None and lora.grad.shape == lora.shape
        assert base.grad is not None and base.grad.shape == base.shape

    def test_fused_backward_torch_inner_none_no_mag_grad(self):
        """_fused_backward_torch with inner=None and needs_mag_grad=False must not crash."""
        torch.manual_seed(204)
        d_out = torch.randn(4, 16)
        mag = torch.rand(1, 16) + 0.5

        d_lora, d_base, d_mag = _fused_backward_torch(
            d_out,
            None,
            mag,
            0.5,
            True,
            True,
            False,
        )

        assert d_lora is not None
        assert d_base is not None
        assert d_mag is None
        assert _max_diff(d_lora, mag * 0.5 * d_out) <= 1e-5
        assert _max_diff(d_base, (mag - 1) * d_out) <= 1e-5

    def test_fused_backward_torch_inner_none_asserts_on_mag_grad(self):
        """_fused_backward_torch with inner=None must assert if needs_mag_grad=True."""
        torch.manual_seed(205)
        d_out = torch.randn(4, 16)
        mag = torch.rand(1, 16) + 0.5

        with pytest.raises(AssertionError, match="inner must be saved"):
            _fused_backward_torch(d_out, None, mag, 0.5, True, True, True)


@requires_cuda_triton
class TestFusedDoRAAutogradTriton:
    """GPU-specific autograd tests that exercise the Triton backward kernel."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("batch", [4, 32])
    @pytest.mark.parametrize("out_features", [32, 128])
    def test_triton_backward_matches_torch(self, dtype, batch, out_features):
        """Triton backward should produce same gradients as PyTorch backward."""
        torch.manual_seed(90)
        device = torch.device("cuda")
        scale = 0.5

        # Create inputs and compute inner = scale * lora + base
        lora_t = torch.randn(batch, out_features, dtype=dtype, device=device)
        base_t = torch.randn(batch, out_features, dtype=dtype, device=device)
        mag_t = torch.rand(1, out_features, dtype=dtype, device=device) + 0.5
        d_out = torch.randn(batch, out_features, dtype=dtype, device=device)
        inner = scale * lora_t + base_t

        # Run with PyTorch fallback
        d_lora_t, d_base_t, d_mag_t = _fused_backward_torch(
            d_out,
            inner,
            mag_t,
            scale,
            True,
            True,
            True,
        )

        # Run with Triton
        from peft.tuners.lora.dora_fused import _fused_backward_triton

        d_lora_tr, d_base_tr, d_mag_tr = _fused_backward_triton(
            d_out.contiguous(),
            inner.contiguous(),
            mag_t.contiguous(),
            scale,
            True,
            True,
            True,
        )

        tol = 1e-4 if dtype == torch.float32 else 5e-3
        assert _max_diff(d_lora_tr, d_lora_t) <= tol
        assert _max_diff(d_base_tr, d_base_t) <= tol
        assert _max_diff(d_mag_tr, d_mag_t) <= tol

    def test_triton_autograd_full_backward(self):
        """Full autograd backward with Triton kernels on CUDA."""
        torch.manual_seed(91)
        device = torch.device("cuda")
        batch, out_features = 16, 64
        scale = 0.5

        lora = torch.randn(batch, out_features, dtype=torch.float32, device=device, requires_grad=True)
        base = torch.randn(batch, out_features, dtype=torch.float32, device=device, requires_grad=True)
        mag = (torch.rand(1, out_features, dtype=torch.float32, device=device) + 0.5).requires_grad_(True)

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        loss = result.sum()
        loss.backward()

        assert lora.grad is not None
        assert base.grad is not None
        assert mag.grad is not None
        assert torch.all(torch.isfinite(lora.grad))
        assert torch.all(torch.isfinite(base.grad))
        assert torch.all(torch.isfinite(mag.grad))

    def test_triton_gradcheck_cuda(self):
        """Gradcheck on CUDA with float64."""
        torch.manual_seed(92)
        device = torch.device("cuda")
        lora = torch.randn(2, 8, dtype=torch.float64, device=device, requires_grad=True)
        base = torch.randn(2, 8, dtype=torch.float64, device=device, requires_grad=True)
        mag = (torch.rand(1, 8, dtype=torch.float64, device=device) + 0.5).requires_grad_(True)

        assert torch.autograd.gradcheck(
            lambda l, b, m: FusedDoRAComposeFunction.apply(l, b, m, 0.5),
            (lora, base, mag),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_triton_frozen_mag_backward(self):
        """Triton backward with frozen mag: inner=None, d_lora/d_base correct."""
        torch.manual_seed(210)
        device = torch.device("cuda")
        batch, out_features, scale = 16, 64, 0.5

        lora = torch.randn(batch, out_features, device=device, requires_grad=True)
        base = torch.randn(batch, out_features, device=device, requires_grad=True)
        mag = torch.rand(1, out_features, device=device) + 0.5  # frozen

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        d_out = torch.randn_like(result)
        result.backward(d_out)

        assert lora.grad is not None
        assert base.grad is not None
        assert _max_diff(lora.grad, mag * scale * d_out) <= 1e-4
        assert _max_diff(base.grad, (mag - 1) * d_out) <= 1e-4

    def test_triton_fused_backward_inner_none_no_mag_grad(self):
        """_fused_backward_triton with inner=None must work when needs_mag_grad=False."""
        torch.manual_seed(211)
        device = torch.device("cuda")
        d_out = torch.randn(8, 32, device=device)
        mag = torch.rand(1, 32, device=device) + 0.5

        from peft.tuners.lora.dora_fused import _fused_backward_triton

        d_lora, d_base, d_mag = _fused_backward_triton(
            d_out,
            None,
            mag,
            0.5,
            True,
            True,
            False,
        )

        assert d_lora is not None
        assert d_base is not None
        assert d_mag is None
        assert _max_diff(d_lora, mag * 0.5 * d_out) <= 1e-4
        assert _max_diff(d_base, (mag - 1) * d_out) <= 1e-4

    def test_triton_fused_backward_inner_none_asserts_on_mag_grad(self):
        """_fused_backward_triton with inner=None must assert if needs_mag_grad=True."""
        torch.manual_seed(212)
        device = torch.device("cuda")
        d_out = torch.randn(4, 16, device=device)
        mag = torch.rand(1, 16, device=device) + 0.5

        from peft.tuners.lora.dora_fused import _fused_backward_triton

        with pytest.raises(AssertionError, match="inner must be saved"):
            _fused_backward_triton(d_out, None, mag, 0.5, True, True, True)


# ===================================================================
# Integration Tests: Fused ops through DoraLinearLayer
# ===================================================================


class TestIntegrationDoraLinearFused:
    """Integration tests running full DoraLinearLayer forward with fused kernels."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("use_fused", ["0", "1"])
    @pytest.mark.parametrize("use_fused_backward", ["0", "1"])
    def test_fused_vs_unfused_forward_match(self, dtype, use_fused, use_fused_backward, monkeypatch):
        """Fused and unfused forward should produce identical results."""
        torch.manual_seed(100)
        device = _device_for_dtype(dtype)

        base = nn.Linear(24, 12, bias=True, dtype=dtype).to(device)
        rank = 6
        lora_A = nn.Linear(24, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 12, bias=False, dtype=dtype).to(device)

        # Run unfused
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()

        layer_unfused = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer_unfused.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.7)

        x = torch.randn(5, 24, device=device, dtype=dtype)
        base_result = base(x).detach()

        out_unfused = layer_unfused(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.7,
            base_layer=base,
            base_result=base_result.clone(),
        )

        # Run fused
        monkeypatch.setenv("PEFT_DORA_FUSED", use_fused)
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", use_fused_backward)
        _invalidate_fused_cache()

        layer_fused = DoraLinearLayer(fan_in_fan_out=False).to(device)
        # Copy the magnitude from unfused layer
        layer_fused.weight = nn.Parameter(layer_unfused.weight.clone())

        out_fused = layer_fused(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.7,
            base_layer=base,
            base_result=base_result.clone(),
        )

        tol = 1e-5 if dtype == torch.float32 else 5e-2
        assert _max_diff(out_fused, out_unfused) <= tol, (
            f"fused={use_fused}, backward={use_fused_backward}, diff={_max_diff(out_fused, out_unfused)}"
        )

    @pytest.mark.parametrize("use_fused_backward", ["0", "1"])
    def test_fused_grad_flow(self, use_fused_backward, monkeypatch):
        """Gradient flow must work through fused layers."""
        torch.manual_seed(101)
        dtype = torch.float32
        device = torch.device("cpu")

        base = nn.Linear(16, 8, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        rank = 4
        lora_A = nn.Linear(16, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype).to(device)
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", use_fused_backward)

        x = torch.randn(3, 16, dtype=dtype, device=device)
        base_result = base(x).detach()

        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
            base_layer=base,
            base_result=base_result,
        )
        loss = out.sum()
        loss.backward()

        assert lora_A.weight.grad is not None, "lora_A should have grad"
        assert lora_B.weight.grad is not None, "lora_B should have grad"
        assert layer.weight.grad is not None, "magnitude should have grad"
        assert torch.all(torch.isfinite(lora_A.weight.grad))
        assert torch.all(torch.isfinite(lora_B.weight.grad))
        assert torch.all(torch.isfinite(layer.weight.grad))

    @pytest.mark.parametrize("use_fused_backward", ["0", "1"])
    def test_fused_grad_flow_frozen_magnitude(self, use_fused_backward, monkeypatch):
        """With frozen magnitude, lora grads must flow and magnitude must get no grad."""
        torch.manual_seed(220)
        dtype = torch.float32
        device = torch.device("cpu")

        base = nn.Linear(16, 8, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        rank = 4
        lora_A = nn.Linear(16, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype).to(device)
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)
        # Freeze magnitude
        layer.weight.requires_grad_(False)

        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", use_fused_backward)
        _invalidate_fused_cache()

        x = torch.randn(3, 16, dtype=dtype, device=device)
        base_result = base(x).detach()

        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
            base_layer=base,
            base_result=base_result,
        )
        loss = out.sum()
        loss.backward()

        assert lora_A.weight.grad is not None, "lora_A should have grad"
        assert lora_B.weight.grad is not None, "lora_B should have grad"
        assert layer.weight.grad is None, "frozen magnitude should have no grad"
        assert torch.all(torch.isfinite(lora_A.weight.grad))
        assert torch.all(torch.isfinite(lora_B.weight.grad))

    def test_fused_chunked_forward(self, monkeypatch):
        """Fused composition through the chunked path."""
        torch.manual_seed(102)
        dtype = torch.float32
        device = torch.device("cpu")

        base = nn.Linear(64, 48, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        rank = 16
        scaling = 0.3
        lora_A = nn.Linear(64, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 48, bias=False, dtype=dtype).to(device)
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(
            base_layer=base,
            lora_A=lora_A.weight,
            lora_B=lora_B.weight,
            scaling=scaling,
        )

        x = torch.randn(8, 64, dtype=dtype, device=device)

        # Force chunked path by shrinking the budget (8 rows * 48 cols * 4 bytes = 1536 per col)
        monkeypatch.setattr(dora_mod, "_get_forward_chunk_threshold_bytes", lambda: 128)
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")

        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=base,
            base_result=None,
        )

        # Compute reference
        base_weight = dora_mod.dequantize_module_weight(base).to(dtype)
        weight_norm = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=scaling,
        )
        mag_norm_scale = (layer.weight / weight_norm).view(1, -1)
        base_dense = F.linear(x, transpose(base_weight, False))
        lora_dense = lora_B(lora_A(x))
        ref = _ref_compose(lora_dense * 1.0, base_dense, mag_norm_scale, scaling)

        assert layer._last_forward_chunk_size is not None
        assert layer._last_forward_chunk_size < 48
        # Note: chunked uses unfused compose steps, allow slightly larger tolerance
        assert _max_diff(out, ref) <= 1e-5

    def test_chunked_forward_falls_back_when_fused_backward_enabled(self, monkeypatch):
        """Chunked composition should gracefully fall back to eager compose."""
        torch.manual_seed(103)
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base = nn.Linear(64, 48, bias=True, dtype=dtype).to(device)
        rank = 8
        scaling = 0.3
        lora_A = nn.Linear(64, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 48, bias=False, dtype=dtype).to(device)
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(
            base_layer=base,
            lora_A=lora_A.weight,
            lora_B=lora_B.weight,
            scaling=scaling,
        )

        x = torch.randn(8, 64, dtype=dtype, device=device)

        # Force chunking and enable fused backward.
        monkeypatch.setattr(dora_mod, "_get_forward_chunk_threshold_bytes", lambda: 128)
        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        if device.type == "cuda":

            def _boom(*args, **kwargs):
                raise AssertionError("forward-only fused compose should not run in grad-enabled chunked path")

            monkeypatch.setattr(dora_mod, "fused_dora_compose", _boom)

        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=base,
            base_result=None,
        )
        loss = out.sum()
        loss.backward()

        assert lora_A.weight.grad is not None
        assert lora_B.weight.grad is not None
        assert layer.weight.grad is not None


# ===================================================================
# Environment variable control tests
# ===================================================================


class TestEnvVarControl:
    """Tests for PEFT_DORA_FUSED and PEFT_DORA_FUSED_BACKWARD env vars."""

    def test_fused_enabled_by_default_when_triton_available(self, monkeypatch):
        monkeypatch.delenv("PEFT_DORA_FUSED", raising=False)
        # dora_mod._use_fused_kernels depends on is_triton_available()
        result = dora_mod._use_fused_kernels()
        assert result == _HAS_TRITON

    def test_fused_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        assert not dora_mod._use_fused_kernels()

    def test_fused_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        assert dora_mod._use_fused_kernels()

    def test_fused_enabled_via_env_true(self, monkeypatch):
        monkeypatch.setenv("PEFT_DORA_FUSED", "true")
        assert dora_mod._use_fused_kernels()

    def test_fused_disabled_via_env_false(self, monkeypatch):
        monkeypatch.setenv("PEFT_DORA_FUSED", "false")
        assert not dora_mod._use_fused_kernels()

    def test_fused_backward_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        assert not dora_mod._use_fused_backward()

    def test_fused_backward_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        assert dora_mod._use_fused_backward()

    def test_fused_backward_default_is_enabled(self, monkeypatch):
        """Fused backward defaults to enabled — VRAM spike eliminated by fused forward-and-inner kernel."""
        monkeypatch.delenv("PEFT_DORA_FUSED_BACKWARD", raising=False)
        result = dora_mod._use_fused_backward()
        assert result is True

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            ((512, 512), False),
            ((2048, 2048), False),
            ((2048, 4096), False),
            ((4096, 4096), True),
            ((2048, 6144), True),
            ((2048, 8192), True),
        ],
    )
    def test_auto_fused_backward_shape_heuristic(self, shape, expected):
        assert dora_mod._should_auto_use_fused_backward_shape(*shape) is expected

    def test_auto_fused_backward_gate_uses_benchmark_crossover(self, monkeypatch):
        monkeypatch.delenv("PEFT_DORA_FUSED_BACKWARD", raising=False)

        small = _DummyTensorLike((2048, 4096), is_cuda=True)
        large = _DummyTensorLike((2048, 6144), is_cuda=True)
        mag_small = _DummyTensorLike((1, 4096), is_cuda=True)
        mag_large = _DummyTensorLike((1, 6144), is_cuda=True)

        assert not dora_mod._should_use_fused_backward_for_tensor(small, mag_small)
        assert dora_mod._should_use_fused_backward_for_tensor(large, mag_large)

    def test_fused_backward_env_force_overrides_auto_gate(self, monkeypatch):
        tensor = _DummyTensorLike((512, 512), is_cuda=True)
        mag = _DummyTensorLike((1, 512), is_cuda=True)

        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()
        assert dora_mod._should_use_fused_backward_for_tensor(tensor, mag)

        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()
        assert not dora_mod._should_use_fused_backward_for_tensor(tensor, mag)

    def test_fused_backward_auto_gate_preserves_conv_broadcast_behavior(self, monkeypatch):
        monkeypatch.delenv("PEFT_DORA_FUSED_BACKWARD", raising=False)

        conv_out = _DummyTensorLike((2, 5, 8, 8), is_cuda=True)
        conv_mag = _DummyTensorLike((1, 5, 1, 1), is_cuda=True)

        assert dora_mod._should_use_fused_backward_for_tensor(conv_out, conv_mag)


def test_dispatch_training_obeys_fused_backward_shape_gate(monkeypatch):
    torch.manual_seed(130)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    lora_out = torch.randn(4, 12, requires_grad=True)
    base_result = torch.randn(4, 12)
    mag_norm_scale = (torch.rand(1, 12) + 0.5).requires_grad_(True)

    monkeypatch.setattr(dora_mod, "_should_use_fused_backward_for_tensor", lambda *args, **kwargs: False)

    def _boom(*args, **kwargs):
        raise AssertionError("fused backward compose should not run when the shape gate rejects it")

    monkeypatch.setattr(dora_mod, "fused_dora_compose_autograd", _boom)

    out = layer._compose_with_dispatch(
        lora_out=lora_out,
        base_result=base_result,
        mag_norm_scale=mag_norm_scale,
        scale=0.7,
    )
    ref = _ref_compose(lora_out.detach(), base_result, mag_norm_scale.detach(), 0.7)
    assert _max_diff(out.detach(), ref) <= 1e-6


def test_dispatch_training_uses_fused_backward_when_shape_gate_allows(monkeypatch):
    torch.manual_seed(1301)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    lora_out = torch.randn(4, 12, requires_grad=True)
    base_result = torch.randn(4, 12)
    mag_norm_scale = (torch.rand(1, 12) + 0.5).requires_grad_(True)
    expected = torch.randn_like(lora_out)

    monkeypatch.setattr(dora_mod, "_should_use_fused_backward_for_tensor", lambda *args, **kwargs: True)
    monkeypatch.setattr(dora_mod, "fused_dora_compose_autograd", lambda *args, **kwargs: expected)

    out = layer._compose_with_dispatch(
        lora_out=lora_out,
        base_result=base_result,
        mag_norm_scale=mag_norm_scale,
        scale=0.7,
    )
    assert out is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dispatch_training_skips_forward_only_fused(monkeypatch):
    """With grads enabled, dispatch must avoid inference-only fused compose."""
    torch.manual_seed(131)
    dtype = torch.float32
    device = torch.device("cuda")

    base = nn.Linear(24, 12, bias=True, dtype=dtype).to(device)
    rank = 6
    lora_A = nn.Linear(24, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 12, bias=False, dtype=dtype).to(device)

    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.7)

    x = torch.randn(5, 24, device=device, dtype=dtype)
    base_result = base(x).detach()

    monkeypatch.setenv("PEFT_DORA_FUSED", "1")
    monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
    _invalidate_fused_cache()

    def _boom(*args, **kwargs):
        raise AssertionError("forward-only fused compose must be skipped when grads are required")

    monkeypatch.setattr(dora_mod, "fused_dora_compose", _boom)

    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.7,
        base_layer=base,
        base_result=base_result.clone(),
    )
    loss = out.sum()
    loss.backward()

    assert lora_A.weight.grad is not None
    assert lora_B.weight.grad is not None
    assert layer.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dispatch_training_skips_forward_only_fused_when_only_base_requires_grad(monkeypatch):
    """If only base_result requires grad, dispatch must still avoid forward-only compose."""
    torch.manual_seed(132)
    device = torch.device("cuda")

    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    lora_out = torch.randn(4, 12, device=device)
    base_result = torch.randn(4, 12, device=device, requires_grad=True)
    mag_norm_scale = torch.rand(1, 12, device=device) + 0.5
    scale = 0.7

    monkeypatch.setenv("PEFT_DORA_FUSED", "1")
    monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
    _invalidate_fused_cache()

    def _boom(*args, **kwargs):
        raise AssertionError("forward-only fused compose must be skipped when base_result requires grad")

    monkeypatch.setattr(dora_mod, "fused_dora_compose", _boom)

    out = layer._compose_with_dispatch(
        lora_out=lora_out.clone(),
        base_result=base_result,
        mag_norm_scale=mag_norm_scale,
        scale=scale,
    )
    out.sum().backward()

    expected_grad = (mag_norm_scale - 1).expand_as(base_result)
    assert torch.allclose(base_result.grad, expected_grad, atol=1e-6, rtol=1e-6)


def test_chunked_compose_skips_fused_when_x_requires_grad(monkeypatch):
    """Chunked compose must not use inference-only fused path when x requires grad."""
    torch.manual_seed(133)
    dtype = torch.float32
    device = torch.device("cpu")

    base = nn.Linear(16, 8, bias=False, dtype=dtype).to(device)
    base.weight.requires_grad_(False)
    rank = 4
    lora_A = nn.Linear(16, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype).to(device)
    # Freeze LoRA and magnitude so only x carries requires_grad
    lora_A.weight.requires_grad_(False)
    lora_B.weight.requires_grad_(False)

    monkeypatch.setenv("PEFT_DORA_FUSED", "1")
    monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
    _invalidate_fused_cache()

    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A.weight,
        lora_B=lora_B.weight,
        scaling=0.5,
    )
    layer.weight.requires_grad_(False)

    x = torch.randn(4, 16, dtype=dtype, device=device, requires_grad=True)

    # Patch fused_dora_compose to explode if called
    def _boom(*args, **kwargs):
        raise AssertionError("forward-only fused compose must be skipped when x requires grad")

    monkeypatch.setattr(dora_mod, "fused_dora_compose", _boom)

    # Force chunked path (base_result=None)
    monkeypatch.setattr(dora_mod, "_get_forward_chunk_threshold_bytes", lambda: 128)

    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.5,
        base_layer=base,
        base_result=None,
    )
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "x should receive gradients"
    assert torch.all(torch.isfinite(x.grad))


# ===================================================================
# Edge case tests
# ===================================================================


class TestBroadcastReduceDims:
    """Direct tests for _broadcast_reduce_dims correctness."""

    @pytest.mark.parametrize(
        "out_shape, mag_shape, expected",
        [
            # Linear: [B, F] with mag [1, F] → reduce batch dim
            (torch.Size([4, 16]), torch.Size([1, 16]), [0]),
            # Linear: [B, F] with mag [F] → reduce batch dim (fewer dims)
            (torch.Size([4, 16]), torch.Size([16]), [0]),
            # Conv2d: [N, C, H, W] with mag [1, C, 1, 1] → reduce N, H, W
            (torch.Size([2, 8, 32, 32]), torch.Size([1, 8, 1, 1]), [0, 2, 3]),
            # Conv1d: [N, C, L] with mag [1, C, 1] → reduce N, L
            (torch.Size([2, 8, 64]), torch.Size([1, 8, 1]), [0, 2]),
            # Conv3d: [N, C, D, H, W] with mag [1, C, 1, 1, 1] → reduce N, D, H, W
            (
                torch.Size([2, 4, 8, 8, 8]),
                torch.Size([1, 4, 1, 1, 1]),
                [0, 2, 3, 4],
            ),
            # Same shape (no broadcast) → empty list
            (torch.Size([4, 16]), torch.Size([4, 16]), []),
            # All-ones mag broadcasts everywhere except matching dims
            (torch.Size([2, 3, 4]), torch.Size([1, 1, 1]), [0, 1, 2]),
        ],
        ids=[
            "linear_2d",
            "linear_1d_mag",
            "conv2d",
            "conv1d",
            "conv3d",
            "no_broadcast",
            "all_broadcast",
        ],
    )
    def test_broadcast_reduce_dims(self, out_shape, mag_shape, expected):
        result = _broadcast_reduce_dims(out_shape, mag_shape)
        assert result == expected, f"Expected {expected}, got {result}"


class TestEdgeCases:
    """Edge case tests for fused operations."""

    def test_compose_single_element(self):
        """Composition with a single element."""
        lora = torch.tensor([[1.0]])
        base = torch.tensor([[2.0]])
        mag = torch.tensor([[0.5]])
        result = _fused_dora_compose_torch(lora.clone(), base, mag, 1.0, inplace=False)
        # out = (0.5 - 1)*2 + 0.5*1*1 = -1 + 0.5 = -0.5
        assert abs(result.item() - (-0.5)) < 1e-6

    def test_compose_zero_tensors(self):
        """Composition with all-zero tensors."""
        lora = torch.zeros(4, 16)
        base = torch.zeros(4, 16)
        mag = torch.ones(1, 16)
        result = _fused_dora_compose_torch(lora.clone(), base, mag, 1.0, inplace=False)
        assert torch.all(result == 0)

    def test_compose_negative_scale(self):
        """Composition with negative scale."""
        torch.manual_seed(110)
        lora, base, mag = _random_compose_tensors(4, 16, torch.float32, torch.device("cpu"))
        ref = _ref_compose(lora.clone(), base, mag, -0.5)
        result = _fused_dora_compose_torch(lora.clone(), base, mag, -0.5, inplace=False)
        assert _max_diff(result, ref) <= 1e-5

    def test_compose_very_large_values(self):
        """Composition with large values should not overflow in float32."""
        lora = torch.ones(2, 4) * 1e4
        base = torch.ones(2, 4) * 1e4
        mag = torch.ones(1, 4)
        result = _fused_dora_compose_torch(lora.clone(), base, mag, 1.0, inplace=False)
        # out = (1-1)*1e4 + 1*1*1e4 = 0 + 1e4 = 1e4
        assert torch.allclose(result, torch.ones(2, 4) * 1e4, atol=1)

    def test_norm_assembly_single_element(self):
        """Norm assembly with a single element."""
        w = torch.tensor([4.0])
        c = torch.tensor([1.0])
        b = torch.tensor([1.0])
        (result,) = _fused_norm_assembly_torch(w, c, b, 1.0)
        # norm_sq = 4 + 2*1*1 + 1*1 = 7; sqrt(7) ≈ 2.6458
        expected = torch.sqrt(torch.tensor([7.0]))
        assert abs(result.item() - expected.item()) < 1e-5

    def test_norm_assembly_zero_everything(self):
        """Norm assembly with all zeros should return zeros."""
        n = 8
        (result,) = _fused_norm_assembly_torch(torch.zeros(n), torch.zeros(n), torch.zeros(n), 0.0)
        assert torch.all(result == 0)

    # -- Empty tensor guards (Bug #4) --

    def test_compose_empty_tensor_cpu(self):
        """Compose on empty tensors should return empty without error (CPU path)."""
        lora = torch.empty(0, 16)
        base = torch.empty(0, 16)
        mag = torch.ones(1, 16)
        # Out-of-place
        result = fused_dora_compose(lora.clone(), base, mag, 1.0, inplace=False)
        assert result.shape == (0, 16)
        assert result.numel() == 0
        # In-place
        lora_ip = lora.clone()
        result_ip = fused_dora_compose(lora_ip, base, mag, 1.0, inplace=True)
        assert result_ip.shape == (0, 16)

    def test_forward_and_inner_empty_tensor_cpu(self):
        """Forward-and-inner on empty tensors should return empty pair (CPU path)."""
        lora = torch.empty(0, 16)
        base = torch.empty(0, 16)
        mag = torch.ones(1, 16)
        out, inner = fused_dora_forward_and_inner(lora, base, mag, 1.0)
        assert out.shape == (0, 16)
        assert inner.shape == (0, 16)

    def test_norm_assembly_empty_tensor_cpu(self):
        """Norm assembly on empty tensors should return empty (CPU path)."""
        w = torch.empty(0)
        c = torch.empty(0)
        b = torch.empty(0)
        (result,) = fused_norm_assembly(w, c, b, 1.0)
        assert result.shape == (0,)

    def test_backward_empty_tensor_cpu(self):
        """Backward on empty tensors should return correct-shaped empty grads (CPU path)."""
        d_out = torch.empty(0, 16)
        inner = torch.empty(0, 16)
        mag = torch.ones(1, 16)
        d_lora, d_base, d_mag = _fused_backward_torch(d_out, inner, mag, 1.0, True, True, True)
        assert d_lora.shape == (0, 16)
        assert d_base.shape == (0, 16)
        # d_mag reduction over empty dim 0 → zeros with mag's shape
        assert d_mag is not None
        assert d_mag.shape == mag.shape
        assert torch.all(d_mag == 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_compose_empty_tensor_cuda_triton(self):
        """Compose on empty CUDA tensors must not crash Triton (grid=(0,) guard)."""
        from peft.tuners.lora.dora_fused import _fused_dora_compose_triton

        lora = torch.empty(0, 16, device="cuda")
        base = torch.empty(0, 16, device="cuda")
        mag = torch.ones(1, 16, device="cuda")
        # Out-of-place
        result = _fused_dora_compose_triton(lora.clone(), base, mag, 1.0, inplace=False)
        assert result.shape == (0, 16)
        assert result.numel() == 0
        # In-place
        result_ip = _fused_dora_compose_triton(lora.clone(), base, mag, 1.0, inplace=True)
        assert result_ip.shape == (0, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_forward_and_inner_empty_tensor_cuda_triton(self):
        """Forward-and-inner on empty CUDA tensors must not crash Triton."""
        from peft.tuners.lora.dora_fused import _fused_dora_forward_and_inner_triton

        lora = torch.empty(0, 16, device="cuda")
        base = torch.empty(0, 16, device="cuda")
        mag = torch.ones(1, 16, device="cuda")
        out, inner = _fused_dora_forward_and_inner_triton(lora, base, mag, 1.0)
        assert out.shape == (0, 16)
        assert inner.shape == (0, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_backward_empty_tensor_cuda_triton(self):
        """Backward on empty CUDA tensors must not crash Triton."""
        from peft.tuners.lora.dora_fused import _fused_backward_triton

        d_out = torch.empty(0, 16, device="cuda")
        inner = torch.empty(0, 16, device="cuda")
        mag = torch.ones(1, 16, device="cuda")
        d_lora, d_base, d_mag = _fused_backward_triton(d_out, inner, mag, 1.0, True, True, True)
        assert d_lora.shape == (0, 16)
        assert d_base.shape == (0, 16)
        # d_mag reduction over empty dim 0 → zeros with mag's shape
        assert d_mag is not None
        assert d_mag.shape == mag.shape
        assert torch.all(d_mag == 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_norm_assembly_empty_tensor_cuda_triton(self):
        """Norm assembly on empty CUDA tensors must not crash Triton."""
        from peft.tuners.lora.dora_fused import _fused_norm_assembly_triton

        w = torch.empty(0, device="cuda")
        c = torch.empty(0, device="cuda")
        b = torch.empty(0, device="cuda")
        (result,) = _fused_norm_assembly_triton(w, c, b, 1.0)
        assert result.shape == (0,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_autograd_empty_tensor_cuda(self):
        """FusedDoRAComposeFunction on empty CUDA tensors must not crash."""
        lora = torch.empty(0, 16, device="cuda", requires_grad=True)
        base = torch.empty(0, 16, device="cuda", requires_grad=True)
        mag = torch.ones(1, 16, device="cuda", requires_grad=True)
        out = FusedDoRAComposeFunction.apply(lora, base, mag, 1.0)
        assert out.shape == (0, 16)
        # Backward should also work on empty
        out.sum().backward()
        assert lora.grad.shape == (0, 16)
        assert base.grad.shape == (0, 16)
        assert mag.grad is not None
        assert mag.grad.shape == (1, 16)
        assert torch.all(mag.grad == 0)


# ===================================================================
# Benchmark formula parity: parenthesization must match canonical form
# ===================================================================


class TestBenchmarkFormulaParity:
    """Verify that the composition formula used by benchmark baselines matches
    the canonical eager helper ``_ref_compose``.

    In reduced-precision dtypes (bf16/fp16), float multiplication is not
    associative: ``(mag * scale) * lora != mag * (scale * lora)``.  The
    canonical form is ``mag * (scale * lora)`` — scale first, then mag.
    These tests ensure benchmark reference formulas use the same order so
    that reported accuracy metrics are meaningful.
    """

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_benchmark_compose_formula_matches_ref(self, dtype):
        """The benchmark's compose formula must be bitwise-equal to _ref_compose."""
        torch.manual_seed(9999)
        rows, cols = 64, 128
        lora = torch.randn(rows, cols, dtype=dtype)
        base = torch.randn(rows, cols, dtype=dtype)
        mag = torch.randn(1, cols, dtype=dtype).abs() + 0.5
        scale = 0.5

        # Canonical reference
        ref = _ref_compose(lora, base, mag, scale)

        # Benchmark formula (correctly parenthesized)
        bench = (mag - 1) * base + mag * (scale * lora)

        assert torch.equal(ref, bench), (
            f"Benchmark formula diverges from _ref_compose in {dtype}: max diff = {(ref - bench).abs().max().item()}"
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_wrong_parenthesization_diverges(self, dtype):
        """Confirm that the WRONG multiply order actually produces different
        results in reduced precision, validating that the test above is
        meaningful (not vacuously passing).

        Uses a non-power-of-two scale (0.3) so that ``scale * lora`` involves
        real rounding, and a tensor-typed scale to ensure both multiplies
        go through reduced-precision arithmetic (scalar floats can be
        promoted to fp64 by PyTorch before the multiply).
        """
        torch.manual_seed(9999)
        rows, cols = 256, 256
        lora = torch.randn(rows, cols, dtype=dtype)
        base = torch.randn(rows, cols, dtype=dtype)
        mag = torch.randn(1, cols, dtype=dtype).abs() + 0.5
        # Tensor-typed scale forces reduced-precision intermediate
        scale = torch.tensor(0.3, dtype=dtype)

        correct = (mag - 1) * base + mag * (scale * lora)
        wrong = (mag - 1) * base + mag * scale * lora  # left-to-right: (mag*scale)*lora

        # They should differ in at least some elements in reduced precision
        assert not torch.equal(correct, wrong), (
            f"Expected different results between correct and wrong parenthesization "
            f"in {dtype}, but they matched — test is not meaningful"
        )


# ===================================================================
# Regression tests: fused vs unfused produce identical numerics
# ===================================================================


class TestRegressionFusedVsUnfused:
    """Regression tests comparing fused and unfused paths exactly.

    Uses explicit representative combos instead of a full parametrize grid
    to keep CI wall-time reasonable (~18 cases instead of 324).
    """

    _COMBOS = [
        # (dtype, batch, out_features, in_features, rank, scaling) - representative spread
        pytest.param(torch.float32, 1, 12, 24, 4, 0.1, id="fp32-small-low_scale"),
        pytest.param(torch.float32, 8, 48, 64, 16, 0.5, id="fp32-medium"),
        pytest.param(torch.float32, 64, 128, 256, 4, 1.0, id="fp32-large-scale1"),
        pytest.param(torch.float32, 1, 128, 64, 16, 0.5, id="fp32-batch1-wide"),
        pytest.param(torch.float32, 64, 12, 256, 16, 0.1, id="fp32-batch64-narrow"),
        pytest.param(torch.float32, 8, 128, 256, 16, 1.0, id="fp32-all-large"),
        pytest.param(torch.bfloat16, 1, 12, 24, 4, 0.1, id="bf16-small-low_scale"),
        pytest.param(torch.bfloat16, 8, 48, 64, 16, 0.5, id="bf16-medium"),
        pytest.param(torch.bfloat16, 64, 128, 256, 4, 1.0, id="bf16-large-scale1"),
        pytest.param(torch.bfloat16, 1, 128, 64, 16, 0.5, id="bf16-batch1-wide"),
        pytest.param(torch.bfloat16, 64, 12, 256, 16, 0.1, id="bf16-batch64-narrow"),
        pytest.param(torch.bfloat16, 8, 128, 256, 16, 1.0, id="bf16-all-large"),
    ]

    @pytest.mark.parametrize("dtype,batch_size,out_features,in_features,rank,scaling", _COMBOS)
    def test_full_forward_regression(self, dtype, batch_size, out_features, in_features, rank, scaling, monkeypatch):
        """Full forward pass: fused must match unfused numerically."""
        torch.manual_seed(200)
        device = _device_for_dtype(dtype)

        base = nn.Linear(in_features, out_features, bias=True, dtype=dtype).to(device)
        lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)

        # Unfused
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        base_result = base(x).detach()
        out_unfused = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=base,
            base_result=base_result.clone(),
        )

        # Fused
        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()
        out_fused = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=base,
            base_result=base_result.clone(),
        )

        tol = 1e-5 if dtype == torch.float32 else 5e-2
        diff = _max_diff(out_fused, out_unfused)
        assert diff <= tol, (
            f"Regression failed: dtype={dtype}, batch={batch_size}, "
            f"out={out_features}, in={in_features}, rank={rank}, "
            f"scaling={scaling}, diff={diff}"
        )


class TestRegressionGradients:
    """Regression tests comparing gradients between fused and unfused."""

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("out_features", [8, 32])
    def test_gradient_regression(self, dtype, batch_size, out_features, monkeypatch):
        """Gradients from fused backward must match PyTorch autograd."""
        torch.manual_seed(210)
        device = torch.device("cpu")
        in_features = out_features * 2
        rank = 4
        scaling = 0.5

        base = nn.Linear(in_features, out_features, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

        # Unfused path
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()

        lora_A1 = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B1 = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        layer1 = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer1.update_layer(base_layer=base, lora_A=lora_A1.weight, lora_B=lora_B1.weight, scaling=scaling)

        base_result = base(x).detach()
        out1 = layer1(
            x,
            lora_A=lora_A1,
            lora_B=lora_B1,
            scaling=scaling,
            base_layer=base,
            base_result=base_result.clone(),
        )
        loss1 = out1.sum()
        loss1.backward()

        # Fused backward path
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        lora_A2 = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B2 = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        # Copy weights
        lora_A2.weight.data.copy_(lora_A1.weight.data)
        lora_B2.weight.data.copy_(lora_B1.weight.data)
        layer2 = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer2.weight = nn.Parameter(layer1.weight.clone())

        out2 = layer2(
            x,
            lora_A=lora_A2,
            lora_B=lora_B2,
            scaling=scaling,
            base_layer=base,
            base_result=base_result.clone(),
        )
        loss2 = out2.sum()
        loss2.backward()

        tol = 1e-4
        assert _max_diff(out1, out2) <= tol, f"Forward diff: {_max_diff(out1, out2)}"

        # Compare gradients
        assert _max_diff(lora_A1.weight.grad, lora_A2.weight.grad) <= tol, (
            f"lora_A grad diff: {_max_diff(lora_A1.weight.grad, lora_A2.weight.grad)}"
        )
        assert _max_diff(lora_B1.weight.grad, lora_B2.weight.grad) <= tol, (
            f"lora_B grad diff: {_max_diff(lora_B1.weight.grad, lora_B2.weight.grad)}"
        )
        assert _max_diff(layer1.weight.grad, layer2.weight.grad) <= tol, (
            f"magnitude grad diff: {_max_diff(layer1.weight.grad, layer2.weight.grad)}"
        )


# ===================================================================
# Triton availability / fallback tests
# ===================================================================


class TestFallbackBehavior:
    """Tests for graceful fallback when Triton is unavailable."""

    def test_cpu_tensors_use_torch_fallback(self):
        """CPU tensors should always use the PyTorch fallback."""
        torch.manual_seed(300)
        lora, base, mag = _random_compose_tensors(4, 16, torch.float32, torch.device("cpu"))
        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        result = fused_dora_compose(lora.clone(), base, mag, 0.5, inplace=False)
        assert _max_diff(result, ref) <= 1e-5

    def test_norm_cpu_uses_torch_fallback(self):
        """CPU norm tensors should use PyTorch fallback."""
        torch.manual_seed(301)
        w, c, b = _random_norm_tensors(32, torch.float32, torch.device("cpu"))
        ref = _ref_norm_assembly(w, c, b, 0.5)
        (result,) = fused_norm_assembly(w, c, b, 0.5)
        assert _max_diff(result, ref) <= 1e-5

    def test_is_triton_available_returns_bool(self):
        """is_triton_available should return a boolean."""
        result = is_triton_available()
        assert isinstance(result, bool)


# ===================================================================
# Conv integration tests with fused paths
# ===================================================================


class TestConvFusedIntegration:
    """Integration tests for Conv layers with fused kernels."""

    @pytest.mark.parametrize("use_fused", ["0", "1"])
    def test_conv2d_fused_forward(self, use_fused, monkeypatch):
        """Conv2d DoRA forward with fused/unfused should match."""
        torch.manual_seed(400)
        dtype = torch.float32
        device = torch.device("cpu")

        base = nn.Conv2d(4, 5, 3, padding=1, bias=True, dtype=dtype).to(device)
        rank = 3
        lora_A = nn.Conv2d(4, rank, 3, padding=1, bias=False, dtype=dtype).to(device)
        lora_B = nn.Conv2d(rank, 5, 1, bias=False, dtype=dtype).to(device)

        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()
        layer = DoraConv2dLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.6)

        x = torch.randn(2, 4, 8, 8, dtype=dtype, device=device)
        base_result = base(x).detach()

        out_unfused = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.6,
            base_layer=base,
            base_result=base_result.clone(),
        )

        monkeypatch.setenv("PEFT_DORA_FUSED", use_fused)
        _invalidate_fused_cache()
        out_fused = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.6,
            base_layer=base,
            base_result=base_result.clone(),
        )

        assert _max_diff(out_fused, out_unfused) <= 1e-5

    def test_conv2d_compose_shapes(self):
        """fused_dora_compose with Conv2d-style [N,C,H,W] and mag [1,C,1,1]."""
        torch.manual_seed(410)
        N, C, H, W = 2, 5, 8, 8
        lora = torch.randn(N, C, H, W)
        base = torch.randn(N, C, H, W)
        mag = torch.rand(1, C, 1, 1) + 0.5
        scale = 0.6

        ref = _ref_compose(lora.clone(), base, mag, scale)
        result = fused_dora_compose(lora.clone(), base, mag, scale, inplace=False)
        assert _max_diff(result, ref) <= 1e-5

    def test_conv2d_autograd_compose(self):
        """fused_dora_compose_autograd with Conv2d-style shapes."""
        torch.manual_seed(411)
        N, C, H, W = 2, 5, 8, 8
        lora = torch.randn(N, C, H, W, requires_grad=True)
        base = torch.randn(N, C, H, W, requires_grad=True)
        mag = (torch.rand(1, C, 1, 1) + 0.5).requires_grad_(True)
        scale = 0.6

        result = fused_dora_compose_autograd(lora, base, mag, scale)
        ref = _ref_compose(lora.detach().clone(), base.detach().clone(), mag.detach().clone(), scale)
        assert _max_diff(result, ref) <= 1e-5

        # Test backward gradients
        d_out = torch.randn_like(result)
        result.backward(d_out)

        assert lora.grad is not None and lora.grad.shape == lora.shape
        assert base.grad is not None and base.grad.shape == base.shape
        assert mag.grad is not None and mag.grad.shape == mag.shape

        # Verify gradient values
        expected_d_lora = mag * scale * d_out
        assert _max_diff(lora.grad, expected_d_lora) <= 1e-5

        expected_d_base = (mag - 1) * d_out
        assert _max_diff(base.grad, expected_d_base) <= 1e-5

        # d_mag should sum over dims {0, 2, 3} for Conv [N, C, H, W]
        inner = scale * lora.detach() + base.detach()
        expected_d_mag = (inner * d_out).sum(dim=[0, 2, 3], keepdim=True)
        assert _max_diff(mag.grad, expected_d_mag) <= 1e-5

    @pytest.mark.parametrize("use_fused_backward", ["0", "1"])
    def test_conv2d_fused_backward_grad_flow(self, use_fused_backward, monkeypatch):
        """Conv2d DoRA backward: gradients must flow correctly."""
        torch.manual_seed(412)
        dtype = torch.float32
        device = torch.device("cpu")

        base = nn.Conv2d(4, 5, 3, padding=1, bias=False, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        rank = 3
        lora_A = nn.Conv2d(4, rank, 3, padding=1, bias=False, dtype=dtype).to(device)
        lora_B = nn.Conv2d(rank, 5, 1, bias=False, dtype=dtype).to(device)

        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", use_fused_backward)
        _invalidate_fused_cache()

        layer = DoraConv2dLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.6)

        x = torch.randn(2, 4, 8, 8, dtype=dtype, device=device)
        base_result = base(x).detach()

        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.6,
            base_layer=base,
            base_result=base_result.clone(),
        )
        loss = out.sum()
        loss.backward()

        assert lora_A.weight.grad is not None, "lora_A should have grad"
        assert lora_B.weight.grad is not None, "lora_B should have grad"
        assert layer.weight.grad is not None, "magnitude should have grad"
        assert torch.all(torch.isfinite(lora_A.weight.grad))
        assert torch.all(torch.isfinite(lora_B.weight.grad))
        assert torch.all(torch.isfinite(layer.weight.grad))


# ===================================================================
# Performance benchmark tests (GPU-only)
# ===================================================================


@requires_cuda
class TestPerformanceBenchmarks:
    """
    Performance benchmarks for fused vs unfused DoRA operations.

    These tests measure execution time and should be run on GPU for
    meaningful results. They verify functional correctness and provide
    timing information.

    To run these benchmarks:
        pytest tests/tuners/lora/test_dora_fused.py -v -k "TestPerformanceBenchmarks"
    """

    def _warmup_cuda(self):
        """Warm up CUDA to avoid cold start bias."""
        x = torch.randn(64, 64, device="cuda")
        for _ in range(10):
            x = x @ x.T
        torch.cuda.synchronize()

    def _benchmark_fn(self, fn, warmup=5, iters=50):
        """Benchmark a function, returning median time in microseconds."""
        self._warmup_cuda()

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            start_event.record()
            fn()
            end_event.record()
            end_event.synchronize()
            times.append(start_event.elapsed_time(end_event) * 1e3)  # microseconds

        times.sort()
        return times[len(times) // 2]  # median

    @requires_cuda_triton
    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_seq", [1024, 8192])
    @pytest.mark.parametrize("hidden", [1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compose_benchmark(self, batch_seq, hidden, dtype):
        """
        Benchmark fused vs unfused composition.

        IMPORTANT: Run on GPU with CUDA for meaningful results.
        This is a functional benchmark - it measures correctness AND timing.
        """
        torch.manual_seed(500)
        device = torch.device("cuda")

        lora = torch.randn(batch_seq, hidden, dtype=dtype, device=device)
        base = torch.randn(batch_seq, hidden, dtype=dtype, device=device)
        mag = torch.rand(1, hidden, dtype=dtype, device=device) + 0.5
        scale = 0.5

        # Unfused (numerically stable form)
        def unfused():
            r = lora.clone()
            r.mul_(scale)
            r.mul_(mag)
            r.add_(base * (mag - 1))
            return r

        # Fused
        def fused():
            return fused_dora_compose(lora.clone(), base, mag, scale, inplace=True)

        time_unfused = self._benchmark_fn(unfused)
        time_fused = self._benchmark_fn(fused)

        # Verify correctness (both use stable formula but in-place rounding
        # order differs from Triton kernel order in bf16)
        ref = unfused()
        result = fused()
        tol = 1e-4 if dtype == torch.float32 else 5e-2
        assert _max_diff(ref, result) <= tol

        logger.info(
            "Compose benchmark (%dx%d, %s): unfused=%0.f µs, fused=%.0f µs, speedup=%.2fx",
            batch_seq,
            hidden,
            dtype,
            time_unfused,
            time_fused,
            time_unfused / max(time_fused, 1),
        )

    @requires_cuda_triton
    @pytest.mark.benchmark
    @pytest.mark.parametrize("out_features", [1024, 4096, 8192])
    def test_norm_assembly_benchmark(self, out_features):
        """
        Benchmark fused vs unfused norm assembly.

        IMPORTANT: Run on GPU with CUDA for meaningful results.
        """
        torch.manual_seed(510)
        device = torch.device("cuda")

        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, torch.float32, device)
        scale = 0.5
        scale_t = torch.tensor(scale, device=device, dtype=torch.float32)

        def unfused():
            n = w_norm_sq + 2.0 * scale_t * cross_term + (scale_t**2) * ba_norm_sq
            n = n.clamp_min_(0)
            return torch.sqrt(n)

        def fused():
            (r,) = fused_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, scale)
            return r

        time_unfused = self._benchmark_fn(unfused)
        time_fused = self._benchmark_fn(fused)

        ref = unfused()
        result = fused()
        assert _max_diff(ref, result) <= 1e-4

        logger.info(
            "Norm assembly benchmark (out_features=%d): unfused=%.0f µs, fused=%.0f µs, speedup=%.2fx",
            out_features,
            time_unfused,
            time_fused,
            time_unfused / max(time_fused, 1),
        )

    @requires_cuda_triton
    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_seq", [1024, 4096])
    @pytest.mark.parametrize("hidden", [1024, 4096])
    def test_backward_benchmark(self, batch_seq, hidden):
        """
        Benchmark fused vs unfused backward pass.

        IMPORTANT: Run on GPU with CUDA for meaningful results.
        """
        torch.manual_seed(520)
        device = torch.device("cuda")
        dtype = torch.float32
        scale = 0.5

        # Allocate tensors once outside the timed loop to only benchmark
        # the compose + backward overhead, not tensor allocation.
        lora_src = torch.randn(batch_seq, hidden, dtype=dtype, device=device)
        base_src = torch.randn(batch_seq, hidden, dtype=dtype, device=device)
        mag_src = torch.rand(1, hidden, dtype=dtype, device=device) + 0.5

        def run_unfused():
            lora = lora_src.clone().requires_grad_(True)
            base = base_src.clone().requires_grad_(True)
            mag = mag_src.clone().requires_grad_(True)
            out = (mag - 1) * base + mag * (scale * lora)
            out.sum().backward()

        def run_fused():
            lora = lora_src.clone().requires_grad_(True)
            base = base_src.clone().requires_grad_(True)
            mag = mag_src.clone().requires_grad_(True)
            out = fused_dora_compose_autograd(lora, base, mag, scale)
            out.sum().backward()

        time_unfused = self._benchmark_fn(run_unfused, warmup=3, iters=20)
        time_fused = self._benchmark_fn(run_fused, warmup=3, iters=20)

        logger.info(
            "Backward benchmark (%dx%d): unfused=%.0f µs, fused=%.0f µs, speedup=%.2fx",
            batch_seq,
            hidden,
            time_unfused,
            time_fused,
            time_unfused / max(time_fused, 1),
        )

    @requires_cuda_triton
    @pytest.mark.benchmark
    def test_end_to_end_dora_layer_benchmark(self, monkeypatch):
        """
        End-to-end benchmark of full DoRA layer forward + backward.

        IMPORTANT: Run on GPU with CUDA for meaningful results.
        """
        torch.manual_seed(530)
        device = torch.device("cuda")
        dtype = torch.float32
        in_features, out_features, rank = 4096, 4096, 64
        batch_seq = 2048
        scaling = 0.5

        base = nn.Linear(in_features, out_features, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

        x = torch.randn(batch_seq, in_features, dtype=dtype, device=device)
        base_result = base(x).detach()

        def _run_step():
            lora_A.zero_grad()
            lora_B.zero_grad()
            layer.weight.grad = None
            out = layer(
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=scaling,
                base_layer=base,
                base_result=base_result.clone(),
            )
            out.sum().backward()

        # Set env vars and invalidate cache ONCE before each benchmark
        # to avoid measuring cache invalidation overhead in the timed loop.
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()
        time_unfused = self._benchmark_fn(_run_step, warmup=3, iters=20)

        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()
        time_fused = self._benchmark_fn(_run_step, warmup=3, iters=20)

        logger.info(
            "End-to-end DoRA layer benchmark (%dx%d->%d): unfused=%.0f µs, fused=%.0f µs, speedup=%.2fx",
            batch_seq,
            in_features,
            out_features,
            time_unfused,
            time_fused,
            time_unfused / max(time_fused, 1),
        )


# ===================================================================
# GPU-specific correctness tests
# ===================================================================


@requires_cuda_triton
class TestGPUCorrectness:
    """
    GPU-specific correctness tests that exercise the actual Triton kernels.

    IMPORTANT: These tests require CUDA + Triton. Run for final validation with:
        pytest tests/tuners/lora/test_dora_fused.py -v -k "TestGPUCorrectness"
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_compose_all_dtypes_gpu(self, dtype):
        """Fused composition on GPU across all dtypes."""
        torch.manual_seed(600)
        device = torch.device("cuda")
        batch, out_features = 32, 128
        lora, base, mag = _random_compose_tensors(batch, out_features, dtype, device)

        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        result = fused_dora_compose(lora.clone(), base, mag, 0.5, inplace=False)

        tol = 1e-4 if dtype == torch.float32 else 5e-2
        assert _max_diff(result, ref) <= tol

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_compose_grad_flow_gpu(self, dtype):
        """Gradient flow through fused composition on GPU."""
        torch.manual_seed(601)
        device = torch.device("cuda")
        batch, out_features = 8, 64

        lora = torch.randn(batch, out_features, dtype=dtype, device=device, requires_grad=True)
        base = torch.randn(batch, out_features, dtype=dtype, device=device, requires_grad=True)
        mag = (torch.rand(1, out_features, dtype=dtype, device=device) + 0.5).requires_grad_(True)

        result = fused_dora_compose_autograd(lora, base, mag, 0.5)
        loss = result.float().sum()
        loss.backward()

        assert lora.grad is not None and torch.all(torch.isfinite(lora.grad))
        assert base.grad is not None and torch.all(torch.isfinite(base.grad))
        assert mag.grad is not None and torch.all(torch.isfinite(mag.grad))

    def test_full_dora_layer_gpu(self, monkeypatch):
        """Full DoRA layer forward+backward on GPU with fused kernels."""
        torch.manual_seed(602)
        device = torch.device("cuda")
        dtype = torch.float32

        base = nn.Linear(64, 32, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        rank = 8
        lora_A = nn.Linear(64, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 32, bias=False, dtype=dtype).to(device)
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(16, 64, dtype=dtype, device=device)
        base_result = base(x).detach()

        # Run with fused backward enabled
        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
            base_layer=base,
            base_result=base_result,
        )
        loss = out.sum()
        loss.backward()

        assert lora_A.weight.grad is not None
        assert lora_B.weight.grad is not None
        assert layer.weight.grad is not None
        assert torch.all(torch.isfinite(out))

    @pytest.mark.parametrize("hidden", [128, 512, 2048, 4096])
    def test_compose_various_hidden_dims_gpu(self, hidden):
        """Test composition across various hidden dimensions on GPU."""
        torch.manual_seed(603)
        device = torch.device("cuda")
        batch = 32
        lora, base, mag = _random_compose_tensors(batch, hidden, torch.float32, device)

        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        result = fused_dora_compose(lora.clone(), base, mag, 0.5, inplace=False)

        assert _max_diff(result, ref) <= 1e-4

    def test_norm_assembly_gpu_matches_torch(self):
        """Triton norm-only + PyTorch division must match pure-PyTorch on GPU."""
        torch.manual_seed(604)
        device = torch.device("cuda")
        out_features = 4096
        w, c, b = _random_norm_tensors(out_features, torch.float32, device)
        magnitude = torch.rand(out_features, dtype=torch.float32, device=device) + 0.1
        eps = 1e-6

        # PyTorch reference
        ref_norm = _ref_norm_assembly(w, c, b, 0.5)
        ref_scale = magnitude / ref_norm.clamp_min(eps)

        # Triton norm-only + PyTorch division (de-fused path)
        (weight_norm,) = fused_norm_assembly(w, c, b, 0.5)
        mag_scale = magnitude / weight_norm.clamp_min(eps)

        assert _max_diff(weight_norm, ref_norm) <= 1e-4
        # After de-fusion, mag_scale differences are only from norm differences,
        # not from separate Triton division precision context.
        assert _max_diff(mag_scale, ref_scale) <= 0.1


# ===================================================================
# Non-contiguous input tests
# ===================================================================


class TestNonContiguousInputs:
    """Tests that non-contiguous inputs are handled correctly (via fallback)."""

    def test_compose_non_contiguous_lora(self):
        """Non-contiguous lora should still produce correct results."""
        torch.manual_seed(700)
        lora = torch.randn(8, 32)[:, ::2]  # non-contiguous
        base = torch.randn(8, 16)
        mag = torch.rand(1, 16) + 0.5

        assert not lora.is_contiguous()
        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        result = fused_dora_compose(lora.clone(), base, mag, 0.5, inplace=False)
        assert _max_diff(result, ref) <= 1e-5

    def test_compose_non_contiguous_base(self):
        """Non-contiguous base should still produce correct results."""
        torch.manual_seed(701)
        lora = torch.randn(8, 16)
        base = torch.randn(8, 32)[:, ::2]  # non-contiguous
        mag = torch.rand(1, 16) + 0.5

        assert not base.is_contiguous()
        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        result = fused_dora_compose(lora.clone(), base, mag, 0.5, inplace=False)
        assert _max_diff(result, ref) <= 1e-5

    def test_compose_non_contiguous_mag(self):
        """Non-contiguous mag_norm_scale should still produce correct results."""
        torch.manual_seed(702)
        lora = torch.randn(8, 16)
        base = torch.randn(8, 16)
        mag = (torch.rand(1, 32) + 0.5)[:, ::2]  # non-contiguous

        assert not mag.is_contiguous()
        ref = _ref_compose(lora.clone(), base, mag, 0.5)
        result = fused_dora_compose(lora.clone(), base, mag, 0.5, inplace=False)
        assert _max_diff(result, ref) <= 1e-5

    def test_norm_assembly_non_contiguous(self):
        """Non-contiguous norm inputs should produce correct results."""
        torch.manual_seed(703)
        w = (torch.rand(32) * 10)[::2]
        c = torch.randn(32)[::2]
        b = (torch.rand(32) * 5)[::2]

        assert not w.is_contiguous()
        ref = _ref_norm_assembly(w, c, b, 0.5)
        (result,) = fused_norm_assembly(w, c, b, 0.5)
        assert _max_diff(result, ref) <= 1e-5

    def test_autograd_non_contiguous_inputs(self):
        """Custom autograd should handle non-contiguous inputs."""
        torch.manual_seed(704)
        # Create non-contiguous tensors that are still leaf tensors
        lora_full = torch.randn(8, 32)
        lora = lora_full[:, ::2].clone().requires_grad_(True)  # contiguous copy
        # Instead test non-contiguous base (transpose makes it non-contiguous)
        base_t = torch.randn(16, 8, requires_grad=True)
        base = base_t.t()  # non-contiguous view
        base.retain_grad()
        mag = (torch.rand(1, 16) + 0.5).requires_grad_(True)

        assert not base.is_contiguous()
        result = fused_dora_compose_autograd(lora, base, mag, 0.5)
        loss = result.sum()
        loss.backward()

        assert lora.grad is not None
        assert base_t.grad is not None
        assert mag.grad is not None


# ===================================================================
# ZeRO-3 / FSDP mock tests
# ===================================================================


class TestZeRO3MockIntegration:
    """Mock tests verifying DoRA works in gather-context scenarios."""

    def test_forward_with_mock_gather_context(self, monkeypatch):
        """DoRA layer should work when base weight is gathered from shards."""
        torch.manual_seed(800)
        dtype = torch.float32
        device = torch.device("cpu")

        base = nn.Linear(16, 8, bias=True, dtype=dtype).to(device)
        rank = 4
        lora_A = nn.Linear(16, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype).to(device)

        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()

        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(4, 16, dtype=dtype, device=device)
        base_result = base(x).detach()

        # Simulate ZeRO-3 scenario: base_result provided externally
        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
            base_layer=base,
            base_result=base_result.clone(),
        )
        assert torch.all(torch.isfinite(out))
        assert out.shape == (4, 8)

    @requires_cuda
    @pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
    def test_fused_under_autocast(self, amp_dtype, monkeypatch):
        """Fused DoRA forward+backward should work under torch.amp.autocast."""
        torch.manual_seed(802)
        device = torch.device("cuda")
        dtype = torch.float32  # model weights are fp32, autocast mixes in

        base = nn.Linear(32, 16, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        rank = 4
        lora_A = nn.Linear(32, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 16, bias=False, dtype=dtype).to(device)

        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(4, 32, dtype=dtype, device=device)
        base_result = base(x).detach()

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            out = layer(
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=0.5,
                base_layer=base,
                base_result=base_result.clone(),
            )
            loss = out.float().sum()
        loss.backward()

        assert torch.all(torch.isfinite(out)), "output has non-finite values"
        assert lora_A.weight.grad is not None, "lora_A should have grad"
        assert lora_B.weight.grad is not None, "lora_B should have grad"
        assert layer.weight.grad is not None, "magnitude should have grad"
        # Verify gradient dtypes are consistent with their parameter dtypes
        assert lora_A.weight.grad.dtype == lora_A.weight.dtype, (
            f"lora_A grad dtype {lora_A.weight.grad.dtype} != param dtype {lora_A.weight.dtype}"
        )
        assert lora_B.weight.grad.dtype == lora_B.weight.dtype, (
            f"lora_B grad dtype {lora_B.weight.grad.dtype} != param dtype {lora_B.weight.dtype}"
        )
        assert layer.weight.grad.dtype == layer.weight.dtype, (
            f"magnitude grad dtype {layer.weight.grad.dtype} != param dtype {layer.weight.dtype}"
        )

    def test_forward_without_base_result_gathers_weight(self, monkeypatch):
        """When base_result=None, DoRA should compute base output internally."""
        torch.manual_seed(801)
        dtype = torch.float32
        device = torch.device("cpu")

        base = nn.Linear(16, 8, bias=False, dtype=dtype).to(device)
        rank = 4
        lora_A = nn.Linear(16, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype).to(device)

        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()

        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(4, 16, dtype=dtype, device=device)

        # base_result=None triggers the chunked path
        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
            base_layer=base,
            base_result=None,
        )
        assert torch.all(torch.isfinite(out))
        assert out.shape == (4, 8)


# ===================================================================
# Additional regression tests: reduced-precision gradients
# ===================================================================


class TestRegressionGradientsReducedPrecision:
    """Gradient regression tests for bf16/fp16 (conceptually part of TestRegressionGradients)."""

    @requires_cuda
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("out_features", [8, 32])
    def test_gradient_regression_reduced_precision(self, dtype, batch_size, out_features, monkeypatch):
        """Gradients from fused backward must match PyTorch autograd in bf16/fp16."""
        torch.manual_seed(900)
        device = torch.device("cuda")
        in_features = out_features * 2
        rank = 4
        scaling = 0.5

        base = nn.Linear(in_features, out_features, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

        # Unfused path
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
        _invalidate_fused_cache()

        lora_A1 = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B1 = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        layer1 = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer1.update_layer(base_layer=base, lora_A=lora_A1.weight, lora_B=lora_B1.weight, scaling=scaling)

        base_result = base(x).detach()
        out1 = layer1(
            x,
            lora_A=lora_A1,
            lora_B=lora_B1,
            scaling=scaling,
            base_layer=base,
            base_result=base_result.clone(),
        )
        loss1 = out1.float().sum()
        loss1.backward()

        # Fused backward path
        monkeypatch.setenv("PEFT_DORA_FUSED", "0")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        lora_A2 = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B2 = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        # Copy weights
        lora_A2.weight.data.copy_(lora_A1.weight.data)
        lora_B2.weight.data.copy_(lora_B1.weight.data)
        layer2 = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer2.weight = nn.Parameter(layer1.weight.clone())

        out2 = layer2(
            x,
            lora_A=lora_A2,
            lora_B=lora_B2,
            scaling=scaling,
            base_layer=base,
            base_result=base_result.clone(),
        )
        loss2 = out2.float().sum()
        loss2.backward()

        # 5e-2 tolerance: bf16 has ~1e-2 relative precision and the fused backward
        # uses different intermediate rounding (fused mul-add vs separate ops), so
        # we allow ~5x machine epsilon.  Tighter bounds (e.g. 2e-2) would cause
        # flaky failures on some GPU architectures due to non-deterministic reduction order.
        tol = 5e-2
        assert _max_diff(out1, out2) <= tol, f"Forward diff: {_max_diff(out1, out2)}"

        # Compare gradients
        assert _max_diff(lora_A1.weight.grad, lora_A2.weight.grad) <= tol, (
            f"lora_A grad diff: {_max_diff(lora_A1.weight.grad, lora_A2.weight.grad)}"
        )
        assert _max_diff(lora_B1.weight.grad, lora_B2.weight.grad) <= tol, (
            f"lora_B grad diff: {_max_diff(lora_B1.weight.grad, lora_B2.weight.grad)}"
        )
        assert _max_diff(layer1.weight.grad, layer2.weight.grad) <= tol, (
            f"magnitude grad diff: {_max_diff(layer1.weight.grad, layer2.weight.grad)}"
        )


# ===================================================================
# Triton backward bf16 test (conceptually part of TestFusedDoRAAutogradTriton)
# ===================================================================


@requires_cuda_triton
class TestTritonBackwardBf16:
    """Direct comparison of Triton vs PyTorch backward in bf16."""

    def test_triton_backward_matches_torch_bf16(self):
        """Triton backward should produce same gradients as PyTorch backward in bf16."""
        torch.manual_seed(901)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        batch, out_features = 16, 64
        scale = 0.5

        # Create inputs and compute inner = scale * lora + base
        lora_t = torch.randn(batch, out_features, dtype=dtype, device=device)
        base_t = torch.randn(batch, out_features, dtype=dtype, device=device)
        mag_t = torch.rand(1, out_features, dtype=dtype, device=device) + 0.5
        d_out = torch.randn(batch, out_features, dtype=dtype, device=device)
        inner = scale * lora_t + base_t

        # Run with PyTorch fallback
        d_lora_t, d_base_t, d_mag_t = _fused_backward_torch(
            d_out,
            inner,
            mag_t,
            scale,
            True,
            True,
            True,
        )

        # Run with Triton
        from peft.tuners.lora.dora_fused import _fused_backward_triton

        d_lora_tr, d_base_tr, d_mag_tr = _fused_backward_triton(
            d_out.contiguous(),
            inner.contiguous(),
            mag_t.contiguous(),
            scale,
            True,
            True,
            True,
        )

        # See tolerance note in TestRegressionGradientsReducedPrecision
        tol = 5e-2
        assert _max_diff(d_lora_tr, d_lora_t) <= tol, f"d_lora diff: {_max_diff(d_lora_tr, d_lora_t)}"
        assert _max_diff(d_base_tr, d_base_t) <= tol, f"d_base diff: {_max_diff(d_base_tr, d_base_t)}"
        assert _max_diff(d_mag_tr, d_mag_t) <= tol, f"d_mag diff: {_max_diff(d_mag_tr, d_mag_t)}"


# ===================================================================
# Fused norm assembly near-zero eps clamp test
# ===================================================================


class TestFusedNormAssemblyNearZero:
    """Tests for fused norm assembly with near-zero norm values."""

    @pytest.mark.parametrize(
        "device_str",
        [
            pytest.param("cpu", id="cpu"),
            pytest.param("cuda", id="cuda", marks=requires_cuda),
        ],
    )
    def test_fused_norm_assembly_near_zero_eps_clamp(self, device_str):
        """Norm-only assembly + PyTorch division should not produce inf/NaN near zero."""
        torch.manual_seed(902)
        device = torch.device(device_str)
        out_features = 32
        eps = 1e-6

        # Construct tensors such that assembled norm is near zero:
        # w_norm_sq + 2*s*cross_term + s^2*ba_norm_sq ~ 0
        # Use very small norm components
        w_norm_sq = torch.full((out_features,), 1e-12, dtype=torch.float32, device=device)
        cross_term = torch.full((out_features,), -1e-12, dtype=torch.float32, device=device)
        ba_norm_sq = torch.full((out_features,), 1e-12, dtype=torch.float32, device=device)
        magnitude = torch.rand(out_features, dtype=torch.float32, device=device) + 0.1
        scale = 0.5

        (weight_norm,) = fused_norm_assembly(
            w_norm_sq,
            cross_term,
            ba_norm_sq,
            scale,
        )
        # Division in PyTorch (de-fused path)
        mag_scale = magnitude / weight_norm.clamp_min(eps)

        # weight_norm should be >= 0 (clamped)
        assert torch.all(weight_norm >= 0), "weight_norm has negative values"
        # mag_scale should be finite (no inf/NaN from division by near-zero)
        assert torch.all(torch.isfinite(mag_scale)), f"mag_scale has non-finite values: {mag_scale}"


# ===================================================================
# Gradient accumulation correctness test
# ===================================================================


class TestGradientAccumulationCorrectness:
    """Verify gradient accumulation is correct across multiple fwd/bwd passes."""

    @requires_cuda
    @pytest.mark.parametrize("fused_backward", ["0", "1"])
    def test_gradient_accumulation_correctness(self, fused_backward, monkeypatch):
        """Accumulated grad over 3 passes must equal sum of individual grads."""
        torch.manual_seed(903)
        device = torch.device("cuda")
        dtype = torch.float32
        in_features = 32
        out_features = 16
        rank = 4
        scaling = 0.5
        num_passes = 3

        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", fused_backward)
        _invalidate_fused_cache()

        base = nn.Linear(in_features, out_features, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        # Create separate inputs for each pass (deterministic)
        rng = torch.Generator(device=device)
        rng.manual_seed(904)
        xs = [torch.randn(4, in_features, device=device, dtype=dtype, generator=rng) for _ in range(num_passes)]

        # Create initial LoRA weights and save copies for the second run
        lora_A_init = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B_init = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)

        # Save initial weight data for cloning
        init_A_data = lora_A_init.weight.data.clone()
        init_B_data = lora_B_init.weight.data.clone()

        # --- Accumulated gradients (no zero_grad between passes) ---
        lora_A_acc = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B_acc = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        lora_A_acc.weight.data.copy_(init_A_data)
        lora_B_acc.weight.data.copy_(init_B_data)
        layer_acc = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer_acc.update_layer(base_layer=base, lora_A=lora_A_acc.weight, lora_B=lora_B_acc.weight, scaling=scaling)
        init_mag_data = layer_acc.weight.data.clone()

        for i in range(num_passes):
            base_result = base(xs[i]).detach()
            out = layer_acc(
                xs[i],
                lora_A=lora_A_acc,
                lora_B=lora_B_acc,
                scaling=scaling,
                base_layer=base,
                base_result=base_result.clone(),
            )
            loss = out.sum()
            loss.backward()
            # Do NOT call zero_grad -- accumulate

        grad_A_acc = lora_A_acc.weight.grad.clone()
        grad_B_acc = lora_B_acc.weight.grad.clone()
        grad_mag_acc = layer_acc.weight.grad.clone()

        # --- Sum of individual gradients (zero_grad each time) ---
        lora_A_ind = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B_ind = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        lora_A_ind.weight.data.copy_(init_A_data)
        lora_B_ind.weight.data.copy_(init_B_data)

        layer_ind = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer_ind.update_layer(base_layer=base, lora_A=lora_A_ind.weight, lora_B=lora_B_ind.weight, scaling=scaling)
        layer_ind.weight.data.copy_(init_mag_data)

        grad_A_sum = torch.zeros_like(lora_A_ind.weight)
        grad_B_sum = torch.zeros_like(lora_B_ind.weight)
        grad_mag_sum = torch.zeros_like(layer_ind.weight)

        for i in range(num_passes):
            lora_A_ind.weight.grad = None
            lora_B_ind.weight.grad = None
            layer_ind.weight.grad = None

            base_result = base(xs[i]).detach()
            out = layer_ind(
                xs[i],
                lora_A=lora_A_ind,
                lora_B=lora_B_ind,
                scaling=scaling,
                base_layer=base,
                base_result=base_result.clone(),
            )
            loss = out.sum()
            loss.backward()

            grad_A_sum += lora_A_ind.weight.grad
            grad_B_sum += lora_B_ind.weight.grad
            grad_mag_sum += layer_ind.weight.grad

        tol = 1e-5
        assert _max_diff(grad_A_acc, grad_A_sum) <= tol, (
            f"lora_A accumulated grad diff: {_max_diff(grad_A_acc, grad_A_sum)}"
        )
        assert _max_diff(grad_B_acc, grad_B_sum) <= tol, (
            f"lora_B accumulated grad diff: {_max_diff(grad_B_acc, grad_B_sum)}"
        )
        assert _max_diff(grad_mag_acc, grad_mag_sum) <= tol, (
            f"magnitude accumulated grad diff: {_max_diff(grad_mag_acc, grad_mag_sum)}"
        )


# ===================================================================
# Triton norm assembly with fp16 magnitude (conceptually TestFusedNormAssemblyTriton)
# ===================================================================


@requires_cuda_triton
class TestFusedNormAssemblyTritonFp16:
    """Tests for Triton norm assembly in fp16."""

    def test_triton_norm_then_pytorch_division_fp16(self):
        """Triton norm-only + PyTorch division in fp16."""
        torch.manual_seed(905)
        device = torch.device("cuda")
        out_features = 256
        w_norm_sq, cross_term, ba_norm_sq = _random_norm_tensors(out_features, torch.float16, device)
        magnitude = torch.rand(out_features, dtype=torch.float16, device=device) + 0.1
        # Use a larger eps suitable for fp16 range (max ~65504) to prevent
        # division-by-near-zero from producing inf in the reference path.
        eps = 1e-3

        (weight_norm,) = fused_norm_assembly(
            w_norm_sq,
            cross_term,
            ba_norm_sq,
            0.5,
        )
        # Division in PyTorch (de-fused path) — happens in fp16 natively.
        mag_scale = magnitude / weight_norm.clamp_min(eps)

        # Compute reference in float32 and cast back, to avoid fp16 overflow
        # in the reference path itself.
        ref_norm = _ref_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, 0.5)
        ref_scale = magnitude.float() / ref_norm.float().clamp_min(eps)

        assert torch.all(torch.isfinite(weight_norm)), "weight_norm has non-finite values"
        assert torch.all(torch.isfinite(mag_scale)), "mag_scale has non-finite values"
        assert _max_diff(weight_norm, ref_norm) <= 5e-2, f"weight_norm diff: {_max_diff(weight_norm, ref_norm)}"
        # mag_scale is computed in fp16 natively (de-fused path), while the
        # reference computes in fp32 then casts.  When weight_norm is near eps,
        # mag/eps can be large (~1e3), and fp16 rounding in the division vs
        # fp32-then-cast produces absolute diffs up to ~0.25.  Use relative
        # tolerance to handle the dynamic range correctly.
        rel_diff = (mag_scale.float() - ref_scale).abs() / ref_scale.abs().clamp_min(1e-6)
        assert rel_diff.max().item() <= 5e-2, f"mag_scale relative diff: {rel_diff.max().item():.4f}"


# ===================================================================
# VRAM leak detection test
# ===================================================================


@requires_cuda
class TestNoVRAMLeak:
    """Tests for VRAM memory stability across iterations."""

    def test_no_vram_leak_across_iterations(self, monkeypatch):
        """VRAM usage should not grow across forward/backward iterations."""
        torch.manual_seed(906)
        device = torch.device("cuda")
        dtype = torch.float32
        in_features = 64
        out_features = 32
        rank = 8
        scaling = 0.5
        warmup_iters = 10
        test_iters = 40

        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        base = nn.Linear(in_features, out_features, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

        def _run_one_cycle():
            x = torch.randn(8, in_features, device=device, dtype=dtype)
            base_result = base(x).detach()
            out = layer(
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=scaling,
                base_layer=base,
                base_result=base_result.clone(),
            )
            loss = out.sum()
            loss.backward()
            # Zero grads for next cycle
            lora_A.weight.grad = None
            lora_B.weight.grad = None
            layer.weight.grad = None

        # Warmup phase
        for _ in range(warmup_iters):
            _run_one_cycle()

        gc.collect()
        torch.cuda.empty_cache()
        baseline_mem = torch.cuda.memory_allocated()

        # Test phase
        for _ in range(test_iters):
            _run_one_cycle()

        gc.collect()
        torch.cuda.empty_cache()
        final_mem = torch.cuda.memory_allocated()

        # Allow 10% growth tolerance
        max_allowed = baseline_mem * 1.1 + 1024  # small absolute buffer for rounding
        assert final_mem <= max_allowed, (
            f"VRAM leak detected: baseline={baseline_mem} bytes, "
            f"final={final_mem} bytes, growth={final_mem - baseline_mem} bytes "
            f"({(final_mem - baseline_mem) / max(baseline_mem, 1) * 100:.1f}%)"
        )


# ===================================================================
# torch.compile smoke test
# ===================================================================


@requires_cuda
class TestTorchCompileSmoke:
    """Smoke test for torch.compile with DoRA fused paths.

    With the custom_op registration (PyTorch 2.4+), the base_result path
    through _compose_with_dispatch is fully compile-friendly — Dynamo treats
    peft::fused_dora_compose as a single opaque node.
    """

    @pytest.fixture(autouse=True)
    def _reset_dynamo(self):
        """Clear Dynamo compilation caches between tests.

        Without this, stale Dynamo guards from an earlier torch.compile call
        can cause spurious failures when a subsequent test recompiles.
        Warming _get_dora_fused() ensures the lazy-import is resolved in
        sys.modules before Dynamo traces through the compose dispatch path.

        NOTE: autouse is scoped to this class (all compile tests).  If
        non-compile tests are added to this class, consider narrowing scope
        — torch._dynamo.reset() is not free.
        """
        torch._dynamo.reset()
        dora_mod._get_dora_fused()
        yield
        torch._dynamo.reset()

    def test_torch_compile_forward_backward_smoke(self, monkeypatch):
        """DoRA forward/backward should work under torch.compile.

        The base_result path (common case) is compile-friendly: no graph breaks
        in _compose_with_dispatch or forward().  Only _compose_with_base_chunks
        (dropout / no precomputed base) retains @dynamo_disable.
        """
        torch.manual_seed(907)
        device = torch.device("cuda")
        dtype = torch.float32
        in_features = 32
        out_features = 16
        rank = 4
        scaling = 0.5

        monkeypatch.setenv("PEFT_DORA_FUSED", "1")
        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        base = nn.Linear(in_features, out_features, bias=True, dtype=dtype).to(device)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        lora_A = nn.Linear(in_features, rank, bias=False, dtype=dtype).to(device)
        lora_B = nn.Linear(rank, out_features, bias=False, dtype=dtype).to(device)
        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

        def dora_forward(x_in):
            base_result = base(x_in).detach()
            return layer(
                x_in,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=scaling,
                base_layer=base,
                base_result=base_result.clone(),
            )

        # fullgraph=False: the norm computation path may still cause graph
        # breaks (e.g. dequantize_module_weight, chunked loops under no_grad).
        # The compose path itself is fully traceable.
        compiled_fn = torch.compile(dora_forward, fullgraph=False)

        x = torch.randn(4, in_features, device=device, dtype=dtype)

        # Eager reference for numerical comparison
        with torch.no_grad():
            eager_out = dora_forward(x.clone())

        out = compiled_fn(x)

        assert torch.all(torch.isfinite(out)), "compiled output has non-finite values"
        assert out.shape == (4, out_features)

        # Compiled output should match eager output
        assert _max_diff(out, eager_out) <= 1e-5, f"Compiled vs eager output mismatch: {_max_diff(out, eager_out)}"

        # Backward
        loss = out.sum()
        loss.backward()

        assert lora_A.weight.grad is not None, "lora_A should have grad after compiled backward"
        assert lora_B.weight.grad is not None, "lora_B should have grad after compiled backward"
        assert layer.weight.grad is not None, "magnitude should have grad after compiled backward"
        assert torch.all(torch.isfinite(lora_A.weight.grad)), "lora_A grad non-finite"
        assert torch.all(torch.isfinite(lora_B.weight.grad)), "lora_B grad non-finite"
        assert torch.all(torch.isfinite(layer.weight.grad)), "magnitude grad non-finite"

    @pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op not available")
    def test_compose_dispatch_no_graph_break(self, monkeypatch):
        """_compose_with_dispatch should not cause graph breaks.

        Verifies the core fix: removing @dynamo_disable from
        _compose_with_dispatch allows Dynamo to trace through
        the composition as a single opaque node.
        """
        torch.manual_seed(908)
        device = torch.device("cuda")
        dtype = torch.float32
        out_features = 16

        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
        # Manually set layer.weight for a standalone compose test
        layer.weight = nn.Parameter(torch.rand(out_features, device=device, dtype=dtype) + 0.5)

        lora_out = torch.randn(4, out_features, device=device, dtype=dtype, requires_grad=True)
        base_result = torch.randn(4, out_features, device=device, dtype=dtype, requires_grad=True)
        mag = (torch.rand(1, out_features, device=device, dtype=dtype) + 0.5).requires_grad_(True)

        def compose_fn(lo, br, mn):
            return layer._compose_with_dispatch(
                lora_out=lo,
                base_result=br,
                mag_norm_scale=mn,
                scale=0.5,
            )

        # fullgraph=True: the compose dispatch should be fully traceable
        compiled_fn = torch.compile(compose_fn, fullgraph=True)

        out = compiled_fn(lora_out, base_result, mag)
        assert torch.all(torch.isfinite(out)), "compiled compose output has non-finite values"
        assert out.shape == (4, out_features)

        loss = out.sum()
        loss.backward()

        assert lora_out.grad is not None
        assert base_result.grad is not None
        assert mag.grad is not None

    @pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op not available")
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_custom_op_cuda_reduced_precision(self, monkeypatch, dtype):
        """Custom op on CUDA in bf16/fp16 must match eager reference within tolerance."""
        torch.manual_seed(909)
        device = torch.device("cuda")
        out_features = 64
        scale = 0.5

        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        lora = torch.randn(8, out_features, device=device, dtype=dtype, requires_grad=True)
        base = torch.randn(8, out_features, device=device, dtype=dtype, requires_grad=True)
        mag = (torch.rand(1, out_features, device=device, dtype=dtype) + 0.5).requires_grad_(True)

        # Eager reference via FusedDoRAComposeFunction (Triton path)
        ref = FusedDoRAComposeFunction.apply(lora, base, mag, scale)

        # Compiled path via custom op
        compiled = torch.compile(
            lambda l, b, m: fused_dora_compose_autograd(l, b, m, scale),
            fullgraph=True,
        )
        result = compiled(lora, base, mag)

        # Forward match — reduced precision has wider tolerance
        assert _max_diff(result, ref) <= 1e-2, (
            f"Custom op CUDA forward diverges from eager: max_diff={_max_diff(result, ref)}"
        )

    @pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op not available")
    def test_compile_backward_numerical_match(self, monkeypatch):
        """Compiled backward grads must numerically match eager backward grads."""
        torch.manual_seed(910)
        device = torch.device("cuda")
        dtype = torch.float32
        out_features = 32
        scale = 0.7

        monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "1")
        _invalidate_fused_cache()

        # Create two independent sets of inputs with same values
        lora_e = torch.randn(4, out_features, device=device, dtype=dtype, requires_grad=True)
        base_e = torch.randn(4, out_features, device=device, dtype=dtype, requires_grad=True)
        mag_e = (torch.rand(1, out_features, device=device, dtype=dtype) + 0.5).requires_grad_(True)
        d_out = torch.randn(4, out_features, device=device, dtype=dtype)

        lora_c = lora_e.detach().clone().requires_grad_(True)
        base_c = base_e.detach().clone().requires_grad_(True)
        mag_c = mag_e.detach().clone().requires_grad_(True)

        # Eager backward (Triton path)
        out_e = FusedDoRAComposeFunction.apply(lora_e, base_e, mag_e, scale)
        out_e.backward(d_out)

        # Compiled backward (custom op path)
        compiled = torch.compile(
            lambda l, b, m: fused_dora_compose_autograd(l, b, m, scale),
            fullgraph=True,
        )
        out_c = compiled(lora_c, base_c, mag_c)
        out_c.backward(d_out)

        tol = 1e-5
        assert _max_diff(lora_e.grad, lora_c.grad) <= tol, f"d_lora mismatch: {_max_diff(lora_e.grad, lora_c.grad)}"
        assert _max_diff(base_e.grad, base_c.grad) <= tol, f"d_base mismatch: {_max_diff(base_e.grad, base_c.grad)}"
        assert _max_diff(mag_e.grad, mag_c.grad) <= tol, f"d_mag mismatch: {_max_diff(mag_e.grad, mag_c.grad)}"


class TestCrossPathParity:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_all_paths_identical(self, dtype):
        """Assert same-dtype DoRA PyTorch composition paths are bitwise identical.

        All paths use the canonical evaluation order ``mag * (scale * lora)``
        (see dora_fused.py docstring).  The in-place path achieves this via
        ``lora.mul_(scale).mul_(mag)`` instead of the old ``lora.mul_(mag*scale)``.
        Triton kernels use the same canonical order but FMA hardware may
        produce different rounding, so Triton is tested within tolerance.
        """
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch, out_features = 32, 256
        scale = 0.3

        lora_base, base_t, mag_norm_scale = _random_compose_tensors(batch, out_features, dtype, device)

        # 1. Reference: (mag-1)*base + mag*(scale*lora)
        ref_out = _ref_compose(lora_base.clone(), base_t, mag_norm_scale, scale)

        # 2. PyTorch Out-of-Place — same eval order as reference, must be bitwise.
        oop_out = dora_fused_mod._fused_dora_compose_torch(
            lora_base.clone(), base_t, mag_norm_scale, scale, inplace=False
        )
        assert _max_diff(oop_out, ref_out) == 0.0, "PyTorch Out-of-Place diverged"

        # 3. PyTorch In-Place — same canonical order via lora.mul_(s).mul_(mag),
        # must be bitwise identical to the out-of-place path.
        ip_out = lora_base.clone()
        dora_fused_mod._fused_dora_compose_torch(ip_out, base_t, mag_norm_scale, scale, inplace=True)
        assert _max_diff(ip_out, ref_out) == 0.0, "PyTorch In-Place diverged"

        # 4. PyTorch Dual Output — same eval order as reference, must be bitwise.
        dual_out, _ = dora_fused_mod._fused_dora_forward_and_inner_torch(
            lora_base.clone(), base_t, mag_norm_scale, scale
        )
        assert _max_diff(dual_out, ref_out) == 0.0, "PyTorch Dual Output diverged"

        # The following tests require CUDA and Triton
        if not device.type == "cuda" or not dora_fused_mod._TRITON_AVAILABLE:
            return

        # 5. Triton Compose (Out-of-Place)
        triton_oop = dora_fused_mod._fused_dora_compose_triton(
            lora_base.clone(), base_t, mag_norm_scale, scale, inplace=False
        )

        # 6. Triton Compose (In-Place) — same Triton eval order as OOP, bitwise.
        triton_ip = lora_base.clone()
        dora_fused_mod._fused_dora_compose_triton(triton_ip, base_t, mag_norm_scale, scale, inplace=True)
        assert _max_diff(triton_ip, triton_oop) == 0.0, "Triton In-Place diverged from Triton OOP"

        # 7. Triton Dual Output — same Triton eval order, bitwise.
        triton_dual, _ = dora_fused_mod._fused_dora_forward_and_inner_triton(
            lora_base.clone(), base_t, mag_norm_scale, scale
        )
        assert _max_diff(triton_dual, triton_oop) == 0.0, "Triton Dual Output diverged from Triton OOP"

        # 8. Triton vs PyTorch — FMA vs separate ops, not bitwise.
        tol = 1e-5 if dtype == torch.float32 else 1e-2
        assert _max_diff(triton_oop, ref_out) <= tol, (
            f"Triton diverged from PyTorch reference beyond tolerance: {_max_diff(triton_oop, ref_out)}"
        )


# ===================================================================
# Step 6A: Empirical precision envelopes for Triton forward paths
# ===================================================================


@requires_cuda_triton
class TestComposePrecisionEnvelope:
    """Empirical forward-error envelopes used by the paper and regression suite."""

    _MAX_ABS_BOUND = {
        torch.float32: 1e-4,
    }
    _REL_L2_BOUND = {
        torch.float32: 1e-6,
        torch.bfloat16: 5e-3,
        torch.float16: 1e-3,
    }
    _COSINE_BOUND = {
        torch.float32: 0.999999,
        torch.bfloat16: 0.99999,
        torch.float16: 0.99999,
    }
    _QUANTIZATION_FLOOR_MULTIPLIER = {
        torch.bfloat16: 4.0,
        torch.float16: 4.0,
    }

    @classmethod
    def _assert_precision_envelope(cls, out, ref, *, dtype, label, exact_ref=None):
        if dtype == torch.float32:
            compare = ref
            max_abs = _max_diff(out, compare)
            abs_bound = cls._MAX_ABS_BOUND[dtype]
        else:
            if exact_ref is None:
                raise AssertionError("exact_ref is required for reduced-precision envelope checks")
            compare = exact_ref
            max_abs = _max_diff_fp64(out, compare)
            quant_floor = _quantization_floor_max_abs(compare, dtype)
            abs_bound = cls._QUANTIZATION_FLOOR_MULTIPLIER[dtype] * quant_floor

        rel_l2 = _relative_l2_error(out, compare)
        cos_sim = _flat_cosine_similarity(out, compare)

        assert max_abs <= abs_bound, f"{label}: max_abs={max_abs:.6g} exceeded {abs_bound:.6g}"
        assert rel_l2 <= cls._REL_L2_BOUND[dtype], (
            f"{label}: rel_l2={rel_l2:.6g} exceeded {cls._REL_L2_BOUND[dtype]:.6g}"
        )
        assert cos_sim >= cls._COSINE_BOUND[dtype], (
            f"{label}: cos_sim={cos_sim:.6f} below {cls._COSINE_BOUND[dtype]:.6f}"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        ("batch", "out_features", "scale", "base_scale"),
        [
            pytest.param(32, 256, 0.3, 1.0, id="compact"),
            pytest.param(16, 2048, 0.7, 1.0, id="mid"),
            pytest.param(32, 3840, 0.3, 62.0, id="gemma3_like"),
        ],
    )
    def test_triton_compose_forward_envelope(self, dtype, batch, out_features, scale, base_scale):
        torch.manual_seed(2022)
        device = torch.device("cuda")
        lora = torch.randn(batch, out_features, device=device, dtype=dtype)
        base = torch.randn(batch, out_features, device=device, dtype=dtype) * base_scale
        mag = torch.rand(1, out_features, device=device, dtype=dtype) + 0.5

        ref = _ref_compose(lora.clone(), base, mag, scale)
        exact_ref = _ref_compose(
            lora.to(torch.float64),
            base.to(torch.float64),
            mag.to(torch.float64),
            scale,
        )
        out = fused_dora_compose(lora.clone(), base, mag, scale, inplace=False)

        self._assert_precision_envelope(
            out,
            ref,
            dtype=dtype,
            label=f"compose dtype={dtype} batch={batch} out={out_features} base_scale={base_scale}",
            exact_ref=exact_ref,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        ("batch", "out_features", "scale", "base_scale"),
        [
            pytest.param(32, 256, 0.3, 1.0, id="compact"),
            pytest.param(16, 2048, 0.7, 1.0, id="mid"),
            pytest.param(32, 3840, 0.3, 62.0, id="gemma3_like"),
        ],
    )
    def test_triton_forward_and_inner_envelope(self, dtype, batch, out_features, scale, base_scale):
        torch.manual_seed(2023)
        device = torch.device("cuda")
        lora = torch.randn(batch, out_features, device=device, dtype=dtype)
        base = torch.randn(batch, out_features, device=device, dtype=dtype) * base_scale
        mag = torch.rand(1, out_features, device=device, dtype=dtype) + 0.5

        ref_out = _ref_compose(lora.clone(), base, mag, scale)
        ref_inner = scale * lora + base
        exact_out = _ref_compose(
            lora.to(torch.float64),
            base.to(torch.float64),
            mag.to(torch.float64),
            scale,
        )
        exact_inner = scale * lora.to(torch.float64) + base.to(torch.float64)
        out, inner = fused_dora_forward_and_inner(lora.clone(), base, mag, scale)

        self._assert_precision_envelope(
            out,
            ref_out,
            dtype=dtype,
            label=f"forward_and_inner/out dtype={dtype} batch={batch} out={out_features} base_scale={base_scale}",
            exact_ref=exact_out,
        )
        self._assert_precision_envelope(
            inner,
            ref_inner,
            dtype=dtype,
            label=f"forward_and_inner/inner dtype={dtype} batch={batch} out={out_features} base_scale={base_scale}",
            exact_ref=exact_inner,
        )


# ===================================================================
# Step 6B: BF16 Absorption Baseline (documents current d_mag behavior)
# ===================================================================


@requires_cuda
class TestBF16AbsorptionBaseline:
    """Documents current d_mag behavior at bf16 scale.

    These tests characterize (not prescribe) the inner-based d_mag
    computation's precision at various base activation scales.
    """

    @pytest.mark.parametrize("base_scale", [1.0, 10.0, 64.0, 128.0])
    def test_d_mag_preserves_lora_signal(self, base_scale):
        """FusedDoRAComposeFunction d_mag cosine sim > 0.99 vs fp64 reference."""
        torch.manual_seed(2000)
        device = torch.device("cuda")
        batch, out_features = 32, 256
        scale = 0.3

        lora = torch.randn(batch, out_features, device=device, dtype=torch.bfloat16, requires_grad=True)
        base = torch.randn(batch, out_features, device=device, dtype=torch.bfloat16) * base_scale
        base.requires_grad_(True)
        mag = (torch.rand(1, out_features, device=device, dtype=torch.bfloat16) + 0.5).requires_grad_(True)

        # Fused path (bf16)
        out = FusedDoRAComposeFunction.apply(lora, base, mag, scale)
        d_out = torch.randn_like(out)
        out.backward(d_out)
        d_mag_fused = mag.grad.clone()

        # fp64 reference
        lora64 = lora.detach().to(torch.float64).requires_grad_(True)
        base64 = base.detach().to(torch.float64).requires_grad_(True)
        mag64 = mag.detach().to(torch.float64).requires_grad_(True)
        d_out64 = d_out.to(torch.float64)
        inner64 = scale * lora64 + base64
        out64 = mag64 * inner64 - base64
        out64.backward(d_out64)
        d_mag_ref = mag64.grad.clone()

        # Cosine similarity — documents current behavior
        cos_sim = F.cosine_similarity(
            d_mag_fused.float().flatten(),
            d_mag_ref.float().flatten(),
            dim=0,
        ).item()
        assert cos_sim > 0.99, f"d_mag cosine sim {cos_sim:.4f} < 0.99 at base_scale={base_scale}"

    def test_d_mag_lora_component_nonzero(self):
        """With zero base, d_mag should be nonzero (pure LoRA signal)."""
        torch.manual_seed(2001)
        device = torch.device("cuda")
        batch, out_features = 16, 128
        scale = 0.5

        lora = torch.randn(batch, out_features, device=device, dtype=torch.bfloat16, requires_grad=True)
        base = torch.zeros(batch, out_features, device=device, dtype=torch.bfloat16, requires_grad=True)
        mag = (torch.rand(1, out_features, device=device, dtype=torch.bfloat16) + 0.5).requires_grad_(True)

        out = FusedDoRAComposeFunction.apply(lora, base, mag, scale)
        d_out = torch.ones_like(out)
        out.backward(d_out)

        assert mag.grad is not None
        assert mag.grad.abs().sum().item() > 0, "d_mag should be nonzero with zero base"


# ===================================================================
# Step 6C: Cross-Path Gradient Baseline (fused vs eager d_mag)
# ===================================================================


@requires_cuda
class TestCrossPathGradientBaseline:
    """Documents current fused-vs-eager d_mag agreement."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("base_scale", [1.0, 64.0])
    def test_d_mag_fused_vs_eager(self, dtype, base_scale):
        """Compare FusedDoRAComposeFunction d_mag vs PyTorch autograd d_mag."""
        torch.manual_seed(2010)
        device = torch.device("cuda")
        batch, out_features = 16, 128
        scale = 0.5

        # Fused path
        lora_f = torch.randn(batch, out_features, device=device, dtype=dtype, requires_grad=True)
        base_f = (torch.randn(batch, out_features, device=device, dtype=dtype) * base_scale).requires_grad_(True)
        mag_f = (torch.rand(1, out_features, device=device, dtype=dtype) + 0.5).requires_grad_(True)
        d_out = torch.randn(batch, out_features, device=device, dtype=dtype)

        out_f = FusedDoRAComposeFunction.apply(lora_f, base_f, mag_f, scale)
        out_f.backward(d_out)

        # Eager path (PyTorch autograd)
        lora_e = lora_f.detach().clone().requires_grad_(True)
        base_e = base_f.detach().clone().requires_grad_(True)
        mag_e = mag_f.detach().clone().requires_grad_(True)

        out_e = _ref_compose(lora_e, base_e, mag_e, scale)
        out_e.backward(d_out)

        if dtype == torch.float32:
            torch.testing.assert_close(mag_f.grad, mag_e.grad, rtol=1e-5, atol=1e-5)
        else:
            cos_sim = F.cosine_similarity(
                mag_f.grad.float().flatten(),
                mag_e.grad.float().flatten(),
                dim=0,
            ).item()
            assert cos_sim > 0.999, (
                f"d_mag fused vs eager cosine sim {cos_sim:.6f} < 0.999 (dtype={dtype}, base_scale={base_scale})"
            )


# ===================================================================
# Step 6D: Inference Fidelity after De-fusion
# ===================================================================


@requires_cuda_triton
class TestInferenceFidelityDefused:
    """Verifies that de-fused norm path matches PyTorch exactly."""

    def test_norm_division_matches_pytorch(self):
        """Triton norm-only + PyTorch div must match pure-PyTorch path.

        After de-fusion, the division is done in identical PyTorch ops,
        so differences should come only from the Triton norm kernel.
        """
        torch.manual_seed(2020)
        device = torch.device("cuda")
        out_features = 3840  # Gemma3-scale
        w, c, b = _random_norm_tensors(out_features, torch.float32, device)
        magnitude = torch.rand(out_features, device=device) + 0.1
        eps = 1e-6

        # Triton norm + PyTorch div
        (wn_triton,) = fused_norm_assembly(w, c, b, 0.5)
        mag_scale_triton = magnitude / wn_triton.clamp_min(eps)

        # Pure PyTorch
        (wn_torch,) = _fused_norm_assembly_torch(w, c, b, 0.5)
        mag_scale_torch = magnitude / wn_torch.clamp_min(eps)

        # The division is identical PyTorch code on identical inputs modulo
        # Triton-vs-PyTorch norm rounding, so mag_scale differences track norm diffs.
        assert _max_diff(wn_triton, wn_torch) <= 1e-6
        assert _max_diff(mag_scale_triton, mag_scale_torch) <= 1e-6

    @pytest.mark.parametrize("embed_scale", [1.0, 62.0])
    def test_compose_output_with_large_base(self, embed_scale):
        """Fused compose matches eager at Gemma3-scale activations."""
        torch.manual_seed(2021)
        device = torch.device("cuda")
        batch, out_features = 32, 3840
        scale = 0.3

        lora = torch.randn(batch, out_features, device=device, dtype=torch.bfloat16)
        base = torch.randn(batch, out_features, device=device, dtype=torch.bfloat16) * embed_scale
        mag = torch.rand(1, out_features, device=device, dtype=torch.bfloat16) + 0.5

        ref = _ref_compose(lora.clone(), base, mag, scale)
        result = fused_dora_compose(lora.clone(), base, mag, scale, inplace=False)

        cos_sim = F.cosine_similarity(
            result.float().flatten(),
            ref.float().flatten(),
            dim=0,
        ).item()
        assert cos_sim > 0.9999, f"cos_sim={cos_sim:.6f} at embed_scale={embed_scale}"


# ===================================================================
# Step 6E: Numerical Edge Cases Extended
# ===================================================================


@requires_cuda
class TestNumericalEdgeCasesExtended:
    """Extended numerical edge cases for d_mag backward."""

    def test_d_mag_with_zero_lora(self):
        """B=0 init case: d_mag should be base-only."""
        torch.manual_seed(2030)
        device = torch.device("cuda")
        batch, out_features = 8, 64
        scale = 0.5

        lora = torch.zeros(batch, out_features, device=device, dtype=torch.float32, requires_grad=True)
        base = torch.randn(batch, out_features, device=device, dtype=torch.float32, requires_grad=True)
        mag = (torch.rand(1, out_features, device=device, dtype=torch.float32) + 0.5).requires_grad_(True)

        out = FusedDoRAComposeFunction.apply(lora, base, mag, scale)
        d_out = torch.ones_like(out)
        out.backward(d_out)

        # inner = scale*0 + base = base → d_mag = (base * d_out).sum(broadcast_dims)
        expected_d_mag = (base.detach() * d_out).sum(dim=0, keepdim=True)
        assert _max_diff(mag.grad, expected_d_mag) <= 1e-5

    def test_d_mag_with_zero_base(self):
        """d_mag = scale * sum(lora * d_out) when base=0."""
        torch.manual_seed(2031)
        device = torch.device("cuda")
        batch, out_features = 8, 64
        scale = 0.5

        lora = torch.randn(batch, out_features, device=device, dtype=torch.float32, requires_grad=True)
        base = torch.zeros(batch, out_features, device=device, dtype=torch.float32, requires_grad=True)
        mag = (torch.rand(1, out_features, device=device, dtype=torch.float32) + 0.5).requires_grad_(True)

        out = FusedDoRAComposeFunction.apply(lora, base, mag, scale)
        d_out = torch.ones_like(out)
        out.backward(d_out)

        # inner = scale*lora + 0 = scale*lora → d_mag = (scale*lora * d_out).sum(broadcast_dims)
        expected_d_mag = (scale * lora.detach() * d_out).sum(dim=0, keepdim=True)
        assert _max_diff(mag.grad, expected_d_mag) <= 1e-5

    def test_d_mag_with_equal_magnitude_components(self):
        """|scale*lora| ~ |base| regime."""
        torch.manual_seed(2032)
        device = torch.device("cuda")
        batch, out_features = 16, 128
        scale = 1.0

        # Make |scale*lora| ~ |base| by using same generator
        base = torch.randn(batch, out_features, device=device, dtype=torch.float32)
        lora = base.clone() + torch.randn_like(base) * 0.1  # nearly equal
        lora.requires_grad_(True)
        base.requires_grad_(True)
        mag = (torch.rand(1, out_features, device=device, dtype=torch.float32) + 0.5).requires_grad_(True)

        out = FusedDoRAComposeFunction.apply(lora, base, mag, scale)
        d_out = torch.randn_like(out)
        out.backward(d_out)

        # Verify gradients are finite and nonzero
        assert torch.all(torch.isfinite(mag.grad)), "d_mag has non-finite values"
        assert mag.grad.abs().sum().item() > 0, "d_mag should be nonzero"
        assert torch.all(torch.isfinite(lora.grad)), "d_lora has non-finite values"
        assert torch.all(torch.isfinite(base.grad)), "d_base has non-finite values"

    @pytest.mark.parametrize("scale", [0.0, 0.001, 0.1, 1.0, 10.0])
    def test_d_mag_across_lora_scales(self, scale):
        """d_mag stays finite and correct across LoRA scaling range."""
        torch.manual_seed(2033)
        device = torch.device("cuda")
        batch, out_features = 8, 64

        lora = torch.randn(batch, out_features, device=device, dtype=torch.float32, requires_grad=True)
        base = torch.randn(batch, out_features, device=device, dtype=torch.float32, requires_grad=True)
        mag = (torch.rand(1, out_features, device=device, dtype=torch.float32) + 0.5).requires_grad_(True)
        d_out = torch.randn(batch, out_features, device=device, dtype=torch.float32)

        out = FusedDoRAComposeFunction.apply(lora, base, mag, scale)
        out.backward(d_out)

        # All grads should be finite
        assert torch.all(torch.isfinite(mag.grad)), f"d_mag non-finite at scale={scale}"
        assert torch.all(torch.isfinite(lora.grad)), f"d_lora non-finite at scale={scale}"
        assert torch.all(torch.isfinite(base.grad)), f"d_base non-finite at scale={scale}"

        # Compare vs fp64 reference
        lora64 = lora.detach().to(torch.float64).requires_grad_(True)
        base64 = base.detach().to(torch.float64).requires_grad_(True)
        mag64 = mag.detach().to(torch.float64).requires_grad_(True)
        inner64 = scale * lora64 + base64
        out64 = mag64 * inner64 - base64
        out64.backward(d_out.to(torch.float64))

        torch.testing.assert_close(mag.grad.double(), mag64.grad, rtol=1e-4, atol=1e-4)
