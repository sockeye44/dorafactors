import os
import sys
from contextlib import contextmanager
from unittest.mock import patch as _mock_patch

# To avoid cross-file class identity mismatches, only clear peft imports when explicitly requested
if os.environ.get("PEFT_TEST_ISOLATED_IMPORTS") == "1":
    for _name in [name for name in list(sys.modules) if name.startswith("peft")]:
        sys.modules.pop(_name)

import pytest
import torch
from torch import nn
import torch.nn.functional as F

import peft.tuners.lora.dora as dora_mod
import peft.tuners.lora.dora_fused as dora_fused_mod
from peft.tuners.lora.dora import (
    DoraConv1dLayer,
    DoraConv2dLayer,
    DoraConv3dLayer,
    DoraEmbeddingLayer,
    DoraLinearLayer,
    _compose_eager_inplace,
    _dtype_element_size,
    _invalidate_fused_cache,
    _invalidate_threshold_cache,
    _mag_broadcasts_last_dim,
    _should_auto_use_fused_backward_shape,
    _DORA_FUSED_MODULE_NAME,
    _get_dora_fused,
    _snapshot_dequantized_weight,
    get_dora_norm_threshold_bytes,
    get_dora_norm_threshold_mb,
    set_dora_norm_threshold_mb,
)
from peft.utils.other import transpose


@pytest.fixture(autouse=True)
def _reset_threshold_cache():
    """Invalidate cached threshold values before and after each test."""
    _invalidate_fused_cache()
    yield
    _invalidate_fused_cache()


@pytest.fixture
def recording_gather(monkeypatch):
    """Fixture that monkey-patches gather_params_ctx to record which params are gathered."""
    gathered = []

    @contextmanager
    def _gather(params):
        if isinstance(params, tuple):
            gathered.extend(params)
        elif hasattr(params, "parameters"):
            gathered.extend(params.parameters())
        yield

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _gather)
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")
    return gathered


class _BoomError(RuntimeError):
    pass


def _device_for_dtype(dtype: torch.dtype) -> torch.device:
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("float16 matmul not supported on CPU")
    return torch.device("cuda" if torch.cuda.is_available() and dtype == torch.float16 else "cpu")


def _random_linear_tensors(out_features, in_features, rank, dtype, fan_in_fan_out):
    device = _device_for_dtype(dtype)
    base_shape = (out_features, in_features)
    if fan_in_fan_out:
        base_shape = base_shape[::-1]
    base_weight = torch.randn(base_shape, device=device, dtype=dtype)
    lora_A = torch.randn(rank, in_features, device=device, dtype=dtype)
    lora_B = torch.randn(out_features, rank, device=device, dtype=dtype)
    return base_weight, lora_A, lora_B


def _random_conv_tensors(out_channels, in_channels, rank, kernel_size, dtype):
    device = _device_for_dtype(dtype)
    base_weight = torch.randn((out_channels, in_channels, kernel_size, kernel_size), device=device, dtype=dtype)
    lora_A = torch.randn((rank, in_channels, kernel_size, kernel_size), device=device, dtype=dtype)
    lora_B = torch.randn((out_channels, rank, 1, 1), device=device, dtype=dtype)
    return base_weight, lora_A, lora_B


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(a - b))


def _make_nonleaf(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    leaf = tensor.detach().clone().requires_grad_(True)
    return leaf, leaf * 1


def test_fsdp_ctx_does_not_mask_exception():
    module = nn.Linear(4, 4)
    with pytest.raises(_BoomError):
        with dora_mod._fsdp_full_param_ctx(module):
            raise _BoomError("boom")


def test_fsdp_ctx_noop_without_fsdp(monkeypatch):
    monkeypatch.setattr(dora_mod, "FSDP", None)
    module = nn.Linear(3, 2)
    with dora_mod._fsdp_full_param_ctx(module):
        pass


@pytest.mark.parametrize("fan_in_fan_out", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [8, 32, 64])
@pytest.mark.parametrize("scaling", [0.0, 0.1, 1.0])
def test_linear_norm_equivalence_vs_dense(fan_in_fan_out, dtype, rank, scaling):
    torch.manual_seed(1)
    out_features, in_features = 128, 256
    base_weight, lora_A, lora_B = _random_linear_tensors(out_features, in_features, rank, dtype, fan_in_fan_out)

    layer = DoraLinearLayer(fan_in_fan_out=fan_in_fan_out)
    with torch.no_grad():
        weight_norm = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=scaling,
        )

    W = transpose(base_weight, fan_in_fan_out).to(torch.float64)
    BA = lora_B.to(torch.float64) @ lora_A.to(torch.float64)
    ref = torch.linalg.vector_norm(W + scaling * BA, dim=1).to(weight_norm.dtype)

    diff = _max_diff(weight_norm, ref)
    tol = 5e-3 if dtype in (torch.bfloat16, torch.float16) else 2e-5
    assert diff <= tol


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [8, 32, 64])
@pytest.mark.parametrize("scaling", [0.0, 0.1, 1.0])
def test_conv_norm_equivalence_vs_dense(dtype, rank, scaling):
    torch.manual_seed(2)
    out_channels, in_channels, kernel = 64, 32, 3
    base_weight, lora_A, lora_B = _random_conv_tensors(out_channels, in_channels, rank, kernel, dtype)

    layer = DoraConv2dLayer(fan_in_fan_out=False)
    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=scaling,
        )

    flat_base = base_weight.reshape(out_channels, -1).to(torch.float64)
    flat_A = lora_A.reshape(rank, -1).to(torch.float64)
    flat_B = lora_B.reshape(out_channels, -1).to(torch.float64)
    BA = (flat_B @ flat_A).reshape_as(base_weight)
    dims = tuple(range(1, base_weight.dim()))
    ref = (flat_base.reshape_as(base_weight) + scaling * BA).norm(p=2, dim=dims, keepdim=True).transpose(1, 0)
    ref = ref.to(weight_norm.dtype)

    diff = _max_diff(weight_norm, ref)
    tol = 5e-3 if dtype in (torch.bfloat16, torch.float16) else 2e-5
    assert diff <= tol


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast test")
def test_autocast_stability_and_grad_flow(monkeypatch):
    torch.manual_seed(3)
    device = torch.device("cuda")

    base = nn.Linear(32, 48, bias=True, dtype=torch.float16).to(device)
    base.weight.requires_grad_(False)
    if base.bias is not None:
        base.bias.requires_grad_(False)

    rank = 16
    lora_A = nn.Linear(32, rank, bias=False, dtype=torch.float16).to(device)
    lora_B = nn.Linear(rank, 48, bias=False, dtype=torch.float16).to(device)
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

    captured = {}
    original = dora_mod.DoraLinearLayer._get_weight_norm_linear

    def record(self, **kwargs):
        result = original(self, **kwargs)
        captured["norm"] = result
        return result

    monkeypatch.setattr(dora_mod.DoraLinearLayer, "_get_weight_norm_linear", record)

    x = torch.randn(4, 32, device=device, dtype=torch.float16)
    with torch.autocast("cuda"):
        base_result = base(x).detach()
        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
            base_layer=base,
            base_result=base_result,
        )
        loss = out.float().sum()

    loss.backward()

    assert captured["norm"].requires_grad is False
    assert lora_A.weight.grad is not None
    assert lora_B.weight.grad is not None
    assert layer.weight.grad is not None
    assert base.weight.grad is None
    assert torch.all(torch.isfinite(out))


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        pytest.param(
            torch.float16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="float16 requires CUDA")
        ),
    ],
)
def test_composition_equivalence(dtype):
    torch.manual_seed(4)
    device = _device_for_dtype(dtype)

    base = nn.Linear(24, 12, bias=True, dtype=dtype).to(device)
    rank = 6
    lora_A = nn.Linear(24, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 12, bias=False, dtype=dtype).to(device)
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.7)

    x = torch.randn(5, 24, device=device, dtype=dtype)
    base_result = base(x).detach()

    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.7,
        base_layer=base,
        base_result=base_result,
    )

    with torch.no_grad():
        base_weight = dora_mod.dequantize_module_weight(base)
        weight_norm = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=0.7,
        )
    mag_norm_scale = (layer.weight / weight_norm).view(1, -1)
    ref = (mag_norm_scale - 1) * base_result + mag_norm_scale * (0.7 * lora_B(lora_A(x)))

    tol = 1e-6 if dtype == torch.float32 else 5e-3
    assert _max_diff(out, ref.to(out.dtype)) <= tol


def test_embedding_formula_old_vs_new_regression():
    """Explicitly document the old-to-new embedding formula change.

    The **old** embedding DoRA composition was::

        result = base + mag * scale * lora          # INCORRECT

    This is wrong because it applies magnitude scaling only to the LoRA delta,
    producing ``base + mag * s * lora`` instead of ``mag * (base + s * lora)``.
    The ``(mag - 1) * base`` term was missing.

    The **new** (correct) formula is::

        result = base + (mag - 1) * base + mag * scale * lora
               = mag * (base + scale * lora) - base + base
               = mag * (base + scale * lora)

    This matches the DoRA paper (2402.09353, Section 4.3).

    Existing checkpoints trained under the old formula will see a numerical
    difference; re-finetuning is recommended (see docs/lora.md).
    """
    torch.manual_seed(42)
    dtype = torch.float32
    device = torch.device("cpu")

    num_embeddings, embedding_dim, rank = 64, 24, 8
    scaling = 0.7
    base = nn.Embedding(num_embeddings, embedding_dim, dtype=dtype).to(device)
    lora_A_param = torch.randn(rank, num_embeddings, dtype=dtype, device=device)
    lora_B_param = torch.randn(embedding_dim, rank, dtype=dtype, device=device)
    lora_A = lora_A_param.T
    lora_B = lora_B_param.T
    layer = DoraEmbeddingLayer(fan_in_fan_out=True).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A_param, lora_B=lora_B_param, scaling=scaling)

    x = torch.randint(0, num_embeddings, (6, 5), device=device)
    base_result = F.embedding(x, base.weight)

    weight_norm = layer.get_weight_norm(base.weight, (lora_A @ lora_B).T.detach(), scaling)
    mag = layer.weight / weight_norm

    # OLD formula (what the code used to produce): base + mag * scale * lora
    lora_result = F.embedding(x, lora_A) @ lora_B
    old_behavior = base_result + mag * (scaling * lora_result)

    # NEW formula (correct DoRA): mag * (base + scale * lora)
    new_behavior = (mag - 1) * base_result + mag * (scaling * lora_result)

    # Verify old != new (they differ by the (mag - 1) * base term)
    diff = _max_diff(old_behavior, new_behavior)
    assert diff > 1e-3, (
        f"Old and new formulas should differ significantly but diff={diff}. "
        "If mag ≈ 1 everywhere the test may need different random seeds."
    )

    # Verify the actual layer produces the NEW formula
    _, delta = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=base_result,
    )
    assert _max_diff(delta, new_behavior) <= 1e-6, "Layer output should match the new (correct) DoRA formula"


def test_embedding_composition_delta_contract():
    torch.manual_seed(41)
    dtype = torch.float32
    device = torch.device("cpu")

    num_embeddings, embedding_dim, rank = 64, 24, 8
    scaling = 0.7
    base = nn.Embedding(num_embeddings, embedding_dim, dtype=dtype).to(device)
    lora_A_param = torch.randn(rank, num_embeddings, dtype=dtype, device=device)
    lora_B_param = torch.randn(embedding_dim, rank, dtype=dtype, device=device)
    lora_A = lora_A_param.T
    lora_B = lora_B_param.T
    layer = DoraEmbeddingLayer(fan_in_fan_out=True).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A_param, lora_B=lora_B_param, scaling=scaling)

    x = torch.randint(0, num_embeddings, (6, 5), device=device)
    base_result = F.embedding(x, base.weight)

    mag_norm_scale, delta = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=base_result,
    )
    _, delta_default_base = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
    )

    lora_result = F.embedding(x, lora_A) @ lora_B
    ref_mag = layer.weight / layer.get_weight_norm(base.weight, (lora_A @ lora_B).T.detach(), scaling)
    ref_delta = (ref_mag - 1) * base_result + ref_mag * (scaling * lora_result)

    assert _max_diff(mag_norm_scale, ref_mag) <= 1e-6
    assert _max_diff(delta, ref_delta) <= 1e-6
    assert _max_diff(delta_default_base, ref_delta) <= 1e-6


def test_embedding_forward_norm_uses_no_grad_and_base_contexts(monkeypatch):
    torch.manual_seed(43)
    dtype = torch.float32
    device = torch.device("cpu")

    num_embeddings, embedding_dim, rank = 32, 16, 4
    scaling = 0.7
    base = nn.Embedding(num_embeddings, embedding_dim, dtype=dtype).to(device)
    lora_A_param = torch.randn(rank, num_embeddings, dtype=dtype, device=device)
    lora_B_param = torch.randn(embedding_dim, rank, dtype=dtype, device=device)
    lora_A = lora_A_param.T
    lora_B = lora_B_param.T

    layer = DoraEmbeddingLayer(fan_in_fan_out=True).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A_param, lora_B=lora_B_param, scaling=scaling)

    flags = {}

    @contextmanager
    def _fake_gather_ctx(base_layer, *extra_modules):
        flags["gather_calls"] = flags.get("gather_calls", 0) + 1
        assert base_layer is base
        yield

    @contextmanager
    def _fake_disable_autocast(device_type):
        flags["autocast_calls"] = flags.get("autocast_calls", 0) + 1
        flags["device_type"] = device_type
        yield

    def _fake_dequantize(module):
        flags["dequant_calls"] = flags.get("dequant_calls", 0) + 1
        assert module is base
        return module.weight

    orig_get_weight_norm = DoraEmbeddingLayer.get_weight_norm

    def _wrapped_get_weight_norm(self, weight, lora_weight, scaling):
        flags["grad_enabled"] = torch.is_grad_enabled()
        flags["lora_requires_grad"] = lora_weight.requires_grad
        flags["lora_has_grad_fn"] = lora_weight.grad_fn is not None
        return orig_get_weight_norm(self, weight, lora_weight, scaling)

    monkeypatch.setattr(dora_mod, "_maybe_gather_base_params_ctx", _fake_gather_ctx)
    monkeypatch.setattr(dora_mod, "_disable_autocast", _fake_disable_autocast)
    monkeypatch.setattr(dora_mod, "dequantize_module_weight", _fake_dequantize)
    monkeypatch.setattr(DoraEmbeddingLayer, "get_weight_norm", _wrapped_get_weight_norm)

    x = torch.randint(0, num_embeddings, (5, 3), device=device)
    base_result = F.embedding(x, base.weight)

    _, delta = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=base_result,
    )

    lora_result = F.embedding(x, lora_A) @ lora_B
    ref_weight_norm = orig_get_weight_norm(layer, base.weight, (lora_A @ lora_B).T.detach(), scaling)
    ref_mag = layer.weight / ref_weight_norm
    ref_delta = (ref_mag - 1) * base_result + ref_mag * (scaling * lora_result)

    assert flags["gather_calls"] == 1
    assert flags["autocast_calls"] == 1
    assert flags["dequant_calls"] == 1
    assert flags["device_type"] == base.weight.device.type
    assert flags["grad_enabled"] is False
    assert flags["lora_requires_grad"] is False
    assert flags["lora_has_grad_fn"] is False
    assert _max_diff(delta, ref_delta) <= 1e-6


def test_dropout_forward_chunk_equivalence(monkeypatch):
    torch.manual_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    base = nn.Linear(64, 48, bias=True, dtype=dtype).to(device)
    base.weight.requires_grad_(False)
    if base.bias is not None:
        base.bias.requires_grad_(False)

    rank = 16
    scaling = 0.3
    lora_A = nn.Linear(64, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 48, bias=False, dtype=dtype).to(device)
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    dropout = nn.Dropout(p=0.4).to(device)
    dropout.train()

    x = torch.randn(128, 64, dtype=dtype, device=device)
    dropout_x = dropout(x)

    # Force the forward path to chunk by shrinking the working-set budget
    monkeypatch.setattr(dora_mod, "_get_forward_chunk_threshold_bytes", lambda: 2_048)

    out = layer(
        dropout_x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        base_result=None,
    )

    base_weight = dora_mod.dequantize_module_weight(base).to(dtype)
    weight_norm = layer._get_weight_norm_linear(
        base_weight=base_weight,
        lora_A_w=lora_A.weight,
        lora_B_w=lora_B.weight,
        scaling=scaling,
    )
    mag_norm_scale = (layer.weight / weight_norm).view(1, -1)
    base_dense = F.linear(dropout_x, transpose(base_weight, False))
    lora_dense = lora_B(lora_A(dropout_x))
    ref = lora_dense.mul(scaling)
    ref.add_(base_dense)
    ref.mul_(mag_norm_scale)
    ref.add_(base_dense, alpha=-1)

    assert layer._last_forward_chunk_size is not None
    assert layer._last_forward_chunk_size < lora_dense.shape[-1]
    assert _max_diff(out, ref.to(out.dtype)) <= 1e-6


def test_dropout_forward_chunk_grad_flow(monkeypatch):
    torch.manual_seed(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    base = nn.Linear(32, 24, bias=True, dtype=dtype).to(device)
    base.weight.requires_grad_(False)
    if base.bias is not None:
        base.bias.requires_grad_(False)

    rank = 8
    scaling = 0.5
    lora_A = nn.Linear(32, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 24, bias=False, dtype=dtype).to(device)
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    dropout = nn.Dropout(p=0.2).to(device)
    dropout.train()

    x = torch.randn(48, 32, dtype=dtype, device=device, requires_grad=True)
    dropout_x = dropout(x)

    monkeypatch.setattr(dora_mod, "_get_forward_chunk_threshold_bytes", lambda: 1_024)

    out = layer(
        dropout_x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        base_result=None,
    )
    loss = out.float().sum()
    loss.backward()

    assert layer._last_forward_chunk_size is not None
    assert layer._last_forward_chunk_size < out.shape[-1]
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert lora_A.weight.grad is not None and torch.all(torch.isfinite(lora_A.weight.grad))
    assert lora_B.weight.grad is not None and torch.all(torch.isfinite(lora_B.weight.grad))
    assert layer.weight.grad is not None and torch.all(torch.isfinite(layer.weight.grad))


def test_gather_ctx_is_entered_once(monkeypatch):
    calls = {"enter": 0, "exit": 0}

    @contextmanager
    def recorder(_module):
        calls["enter"] += 1
        try:
            yield None
        finally:
            calls["exit"] += 1

    monkeypatch.setattr(dora_mod, "gather_params_ctx", recorder)
    # Force gather to be enabled under new gating logic
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")

    base = nn.Linear(8, 8, bias=False)
    lora_A = nn.Linear(8, 4, bias=False)
    lora_B = nn.Linear(4, 8, bias=False)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=1.0)

    calls["enter"] = 0
    calls["exit"] = 0

    x = torch.randn(2, 8)
    layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=1.0,
        base_layer=base,
    )

    assert calls["enter"] == 1
    assert calls["exit"] == 1


def test_forward_runs_without_deepspeed(monkeypatch):
    base = nn.Linear(10, 6, bias=False)
    lora_A = nn.Linear(10, 3, bias=False)
    lora_B = nn.Linear(3, 6, bias=False)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.3)

    monkeypatch.setattr(dora_mod, "gather_params_ctx", None)

    x = torch.randn(4, 10)
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.3,
        base_layer=base,
    )

    assert out.shape == (4, 6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inplace_compose_equivalence(dtype):
    torch.manual_seed(17)
    device = torch.device("cuda" if (torch.cuda.is_available() and dtype != torch.float32) else "cpu")
    base = nn.Linear(16, 12, bias=True, dtype=dtype).to(device)
    rank = 4
    lora_A = nn.Linear(16, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 12, bias=False, dtype=dtype).to(device)
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.8)

    x = torch.randn(3, 16, dtype=dtype, device=device)
    base_result = base(x).detach()
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.8,
        base_layer=base,
        base_result=base_result,
    )

    with torch.no_grad():
        weight = dora_mod.dequantize_module_weight(base)
        weight_norm = layer._get_weight_norm_linear(
            base_weight=weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=0.8,
        )
        eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
        weight_norm = weight_norm.clamp_min(eps)
        mag_norm_scale = (layer.weight / weight_norm).view(1, -1)
        ref = (mag_norm_scale - 1) * base_result + mag_norm_scale * (0.8 * lora_B(lora_A(x)))

    tol = 1e-6 if dtype == torch.float32 else 5e-3
    assert _max_diff(out, ref.to(out.dtype)) <= tol


def test_env_threshold_chunk_selection(monkeypatch):
    monkeypatch.setenv("PEFT_DORA_NORM_CHUNK_MB", "64")
    torch.manual_seed(5)
    base_weight, lora_A, lora_B = _random_linear_tensors(256, 512, 32, torch.float32, False)

    layer = DoraLinearLayer(fan_in_fan_out=False)
    with torch.no_grad():
        layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=0.5,
        )

    assert layer._last_chunk_size == 512


def test_env_threshold_chunk_selection_small(monkeypatch):
    monkeypatch.setenv("PEFT_DORA_NORM_CHUNK_MB", "16")
    torch.manual_seed(6)
    base_weight, lora_A, lora_B = _random_linear_tensors(8192, 8192, 32, torch.float32, False)

    layer = DoraLinearLayer(fan_in_fan_out=False)
    with torch.no_grad():
        layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=0.2,
        )

    assert layer._last_chunk_size < 8192
    assert layer._last_chunk_size > 0


def test_env_threshold_default(monkeypatch):
    monkeypatch.delenv("PEFT_DORA_NORM_CHUNK_MB", raising=False)
    torch.manual_seed(7)
    base_weight, lora_A, lora_B = _random_linear_tensors(1024, 1024, 16, torch.float32, False)

    layer = DoraLinearLayer(fan_in_fan_out=False)
    with torch.no_grad():
        layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=1.0,
        )

    assert layer._last_chunk_size == 1024


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_scaling_returns_base_norm(dtype):
    torch.manual_seed(8)
    base_weight, lora_A, lora_B = _random_linear_tensors(64, 128, 16, dtype, False)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    with torch.no_grad():
        norm = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=0.0,
        )
    base_norm = torch.linalg.vector_norm(base_weight.to(norm.dtype), dim=1)
    assert _max_diff(norm, base_norm) <= (5e-3 if dtype == torch.bfloat16 else 1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_scaling_conv(dtype):
    torch.manual_seed(9)
    base_weight, lora_A, lora_B = _random_conv_tensors(16, 8, 4, 3, dtype)
    layer = DoraConv2dLayer(fan_in_fan_out=False)
    with torch.no_grad():
        norm = layer._get_weight_norm_conv_factored(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=0.0,
        )
    ref = base_weight.to(norm.dtype).norm(p=2, dim=(1, 2, 3), keepdim=True).transpose(1, 0)
    assert _max_diff(norm, ref) <= (5e-3 if dtype == torch.bfloat16 else 1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for large-shape smoke test")
def test_large_shape_smoke(monkeypatch):
    monkeypatch.setenv("PEFT_DORA_NORM_CHUNK_MB", "64")
    torch.manual_seed(10)
    device = torch.device("cuda")
    out_features, in_features, rank = 4096, 4096, 256

    base_weight = torch.randn((out_features, in_features), device=device, dtype=torch.bfloat16)
    lora_A = torch.randn((rank, in_features), device=device, dtype=torch.bfloat16)
    lora_B = torch.randn((out_features, rank), device=device, dtype=torch.bfloat16)

    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    with torch.no_grad():
        norm = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=1.0,
        )

    assert torch.all(torch.isfinite(norm))


@pytest.mark.parametrize("fan_in_fan_out", [False, True])
def test_fan_in_fan_out_toggle(fan_in_fan_out):
    torch.manual_seed(11)
    dtype = torch.float32
    base_weight, lora_A, lora_B = _random_linear_tensors(32, 16, 8, dtype, fan_in_fan_out)

    layer = DoraLinearLayer(fan_in_fan_out=fan_in_fan_out)
    with torch.no_grad():
        norm = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=0.4,
        )

    W = transpose(base_weight, fan_in_fan_out)
    dense = torch.linalg.vector_norm(
        W.to(torch.float64) + 0.4 * (lora_B.to(torch.float64) @ lora_A.to(torch.float64)), dim=1
    ).to(norm.dtype)
    assert _max_diff(norm, dense) <= 1e-6


def test_bias_handling_linear():
    torch.manual_seed(12)
    dtype = torch.float32
    base = nn.Linear(10, 6, bias=True, dtype=dtype)
    rank = 4
    lora_A = nn.Linear(10, rank, bias=False, dtype=dtype)
    lora_B = nn.Linear(rank, 6, bias=False, dtype=dtype)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.9)

    x = torch.randn(3, 10)
    base_result = base(x).detach()
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.9,
        base_layer=base,
        base_result=base_result,
    )

    with torch.no_grad():
        weight = dora_mod.dequantize_module_weight(base)
        weight_norm = layer._get_weight_norm_linear(
            base_weight=weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=0.9,
        )
    mag_norm_scale = (layer.weight / weight_norm).view(1, -1)
    ref = (mag_norm_scale - 1) * base_result + mag_norm_scale * (0.9 * lora_B(lora_A(x)))
    assert _max_diff(out, ref) <= 1e-6


def test_bias_handling_conv():
    torch.manual_seed(13)
    dtype = torch.float32
    base = nn.Conv2d(4, 5, 3, padding=1, bias=True, dtype=dtype)
    rank = 3
    lora_A = nn.Conv2d(4, rank, 3, padding=1, bias=False, dtype=dtype)
    lora_B = nn.Conv2d(rank, 5, 1, bias=False, dtype=dtype)
    layer = DoraConv2dLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.6)

    x = torch.randn(2, 4, 8, 8)
    base_result = base(x).detach()
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.6,
        base_layer=base,
        base_result=base_result,
    )

    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base.weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=0.6,
        )
    mag_norm_scale = layer.weight / weight_norm
    ref = (mag_norm_scale - 1) * (base_result - base.bias.view(1, -1, 1, 1))
    ref = ref + mag_norm_scale * (0.6 * lora_B(lora_A(x)))
    assert _max_diff(out, ref) <= 1e-5


def test_conv_rectangular_kernel():
    torch.manual_seed(14)
    dtype = torch.float32
    base = nn.Conv2d(3, 7, (3, 5), padding=(1, 2), bias=False, dtype=dtype)
    rank = 4
    lora_A = nn.Conv2d(3, rank, (3, 5), padding=(1, 2), bias=False, dtype=dtype)
    lora_B = nn.Conv2d(rank, 7, 1, bias=False, dtype=dtype)
    layer = DoraConv2dLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

    x = torch.randn(2, 3, 8, 6)
    base_result = base(x).detach()
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.5,
        base_layer=base,
        base_result=base_result,
    )

    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base.weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=0.5,
        )
    mag_norm_scale = layer.weight / weight_norm
    ref = (mag_norm_scale - 1) * base_result + mag_norm_scale * (0.5 * lora_B(lora_A(x)))
    assert _max_diff(out, ref) <= 1e-5


def test_conv_groups():
    torch.manual_seed(15)
    dtype = torch.float32
    groups = 2
    in_ch = 6
    out_ch = 8
    base = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups, bias=False, dtype=dtype)
    rank = 4
    lora_A = nn.Conv2d(in_ch, rank, 3, padding=1, groups=groups, bias=False, dtype=dtype)
    lora_B = nn.Conv2d(rank, out_ch, 1, groups=groups, bias=False, dtype=dtype)
    layer = DoraConv2dLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.3)

    x = torch.randn(1, in_ch, 10, 10)
    base_result = base(x).detach()
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.3,
        base_layer=base,
        base_result=base_result,
    )
    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base.weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=0.3,
        )
    mag_norm_scale = layer.weight / weight_norm
    ref = (mag_norm_scale - 1) * base_result + mag_norm_scale * (0.3 * lora_B(lora_A(x)))
    assert _max_diff(out, ref) <= 1e-5


def test_conv2d_nhwc_smoke():
    torch.manual_seed(16)
    base = nn.Conv2d(4, 5, 3, padding=1, bias=False)
    rank = 3
    lora_A = nn.Conv2d(4, rank, 3, padding=1, bias=False)
    lora_B = nn.Conv2d(rank, 5, 1, bias=False)
    layer = DoraConv2dLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=1.0)

    x = torch.randn(2, 10, 10, 4).contiguous(memory_format=torch.channels_last)
    # Convert to NCHW for the actual call; this is a smoke test for shape handling
    x_nchw = x.permute(0, 3, 1, 2).contiguous()
    out = layer(
        x_nchw,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=1.0,
        base_layer=base,
    )
    assert out.shape == (2, 5, 10, 10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, 0.3, 1.0, 2.5])
def test_compose_formula_cross_reference(dtype, scale):
    """Assert _compose_eager_inplace (dora.py) and _fused_dora_compose_torch (dora_fused.py)
    produce identical results for matching inputs.

    These two functions implement the same DoRA composition formula in separate
    modules.  If someone changes one without the other, this test catches the
    divergence.
    """
    torch.manual_seed(99)
    device = torch.device("cpu")
    rows, cols = 32, 64
    lora_a = torch.randn(rows, cols, dtype=dtype, device=device)
    base = torch.randn(rows, cols, dtype=dtype, device=device)
    mag = 1.0 + 0.1 * torch.randn(1, cols, dtype=dtype, device=device)

    # _compose_eager_inplace mutates lora in-place
    lora_eager = lora_a.clone()
    _compose_eager_inplace(lora_eager, base, mag, scale)

    # _fused_dora_compose_torch with inplace=True
    lora_fused_ip = lora_a.clone()
    dora_fused_mod._fused_dora_compose_torch(lora_fused_ip, base, mag, scale, inplace=True)

    # All PyTorch paths now use canonical order mag*(scale*lora) — must be bitwise.
    assert _max_diff(lora_eager, lora_fused_ip) == 0.0, (
        f"eager vs fused-inplace diverged: {_max_diff(lora_eager, lora_fused_ip)}"
    )

    # Out-of-place must also match (same canonical evaluation order).
    lora_fused_oop = dora_fused_mod._fused_dora_compose_torch(lora_a.clone(), base, mag, scale, inplace=False)
    assert _max_diff(lora_eager, lora_fused_oop) == 0.0, (
        f"eager-inplace vs fused-oop diverged: {_max_diff(lora_eager, lora_fused_oop)}"
    )


def test_compose_eager_inplace_rejects_leaf_tensor():
    """_compose_eager_inplace must raise RuntimeError on a leaf tensor requiring grad.

    In-place mutation of a leaf tensor with requires_grad=True is incompatible
    with autograd's saved tensor tracking.  This test confirms the contract
    boundary: callers must pass a non-leaf (e.g. lora_B(lora_A(x))).
    """
    lora = torch.randn(4, 8, requires_grad=True)  # leaf tensor
    base = torch.randn(4, 8)
    mag = torch.rand(1, 8) + 0.5
    with pytest.raises(RuntimeError):
        _compose_eager_inplace(lora, base, mag, 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for mixed-dtype eager parity")
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float16])
def test_mixed_dtype_eager_helper_bitwise_parity(act_dtype):
    """Mixed-dtype eager helpers must match the fp32-mag eager reference exactly."""
    torch.manual_seed(171)
    device = torch.device("cuda")
    rows, cols = 32, 256
    scale = 0.3

    lora = torch.randn(rows, cols, dtype=act_dtype, device=device)
    base = torch.randn(rows, cols, dtype=act_dtype, device=device)
    mag = torch.rand(1, cols, dtype=torch.float32, device=device) + 0.5

    ref = (mag * (scale * lora) + (mag - 1) * base).to(act_dtype)

    eager_ip = _compose_eager_inplace(lora.clone(), base.clone(), mag.clone(), scale)
    fused_ip = dora_fused_mod._fused_dora_compose_torch(lora.clone(), base.clone(), mag.clone(), scale, inplace=True)
    dual_out, _ = dora_fused_mod._fused_dora_forward_and_inner_torch(lora.clone(), base.clone(), mag.clone(), scale)

    assert torch.equal(eager_ip, ref)
    assert torch.equal(fused_ip, ref)
    assert dual_out.dtype == act_dtype, f"Expected {act_dtype}, got {dual_out.dtype}"
    assert torch.equal(dual_out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for mixed-dtype eager parity")
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float16])
def test_mixed_dtype_eager_helper_grad_parity(act_dtype):
    """Mixed-dtype eager in-place helpers must preserve exact eager-reference grads."""
    torch.manual_seed(172)
    device = torch.device("cuda")
    rows, cols = 16, 128
    scale = 0.7

    lora_base = torch.randn(rows, cols, dtype=act_dtype, device=device)
    base_base = torch.randn(rows, cols, dtype=act_dtype, device=device)
    mag_base = torch.rand(1, cols, dtype=torch.float32, device=device) + 0.5

    ref_lora = lora_base.detach().clone().requires_grad_(True)
    ref_base = base_base.detach().clone().requires_grad_(True)
    ref_mag = mag_base.detach().clone().requires_grad_(True)
    ref_out = (ref_mag * (scale * ref_lora) + (ref_mag - 1) * ref_base).to(act_dtype)
    d_out = torch.randn_like(ref_out)
    ref_grads = torch.autograd.grad(ref_out, (ref_lora, ref_base, ref_mag), d_out)

    for compose_fn in (
        lambda lora, base, mag: _compose_eager_inplace(lora, base, mag, scale),
        lambda lora, base, mag: dora_fused_mod._fused_dora_compose_torch(lora, base, mag, scale, inplace=True),
    ):
        lora_leaf, lora = _make_nonleaf(lora_base)
        base_leaf, base = _make_nonleaf(base_base)
        mag_leaf, mag = _make_nonleaf(mag_base)
        out = compose_fn(lora, base, mag)
        grads = torch.autograd.grad(out, (lora_leaf, base_leaf, mag_leaf), d_out)

        assert torch.equal(out.detach(), ref_out.detach())
        for grad, ref_grad in zip(grads, ref_grads):
            assert torch.equal(grad, ref_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP eager parity")
@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_amp_eager_base_result_and_chunk_paths_match_bitwise(monkeypatch, amp_dtype):
    """AMP eager base_result and chunked eager paths must stay bitwise-identical."""
    torch.manual_seed(173)
    device = torch.device("cuda")
    in_features, out_features, rank = 64, 96, 16
    scaling = 0.5

    base = nn.Linear(in_features, out_features, bias=False).to(device)
    base.weight.requires_grad_(False)

    lora_A = nn.Linear(in_features, rank, bias=False).to(device)
    lora_B = nn.Linear(rank, out_features, bias=False).to(device)

    monkeypatch.setenv("PEFT_DORA_FUSED", "0")
    monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
    monkeypatch.setattr(dora_mod, "_get_forward_chunk_threshold_bytes", lambda: 2_048)

    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)
    with torch.no_grad():
        layer.weight.mul_(1.01)

    x = torch.randn(128, in_features, device=device, requires_grad=True)
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        base_result = base(x).detach()
        eager_out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=base,
            base_result=base_result.clone(),
        )
        chunk_out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=base,
            base_result=None,
        )

    assert eager_out.dtype == amp_dtype
    assert chunk_out.dtype == amp_dtype
    assert layer._last_forward_chunk_size is not None
    assert layer._last_forward_chunk_size < out_features
    assert torch.equal(eager_out, chunk_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for mixed-dtype chunk budgeting")
def test_forward_chunk_budget_uses_promoted_mixed_dtype_size(monkeypatch):
    """Chunk sizing should budget against the widened eager mixed-dtype compose temp."""
    torch.manual_seed(174)
    device = torch.device("cuda")
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)

    x = torch.randn(16, 8, dtype=torch.bfloat16, device=device, requires_grad=True)
    lora_input = torch.randn(16, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    lora_result = lora_input * 1
    base_weight_t = torch.randn(32, 8, dtype=torch.bfloat16, device=device)
    mag_norm_scale = (torch.rand(1, 32, dtype=torch.float32, device=device) + 0.5).requires_grad_(True)

    monkeypatch.setenv("PEFT_DORA_FUSED", "0")
    monkeypatch.setenv("PEFT_DORA_FUSED_BACKWARD", "0")
    monkeypatch.setattr(dora_mod, "_get_forward_chunk_threshold_bytes", lambda: 320)

    layer._compose_with_base_chunks(
        x=x,
        lora_result=lora_result,
        base_weight_t=base_weight_t,
        mag_norm_scale=mag_norm_scale,
        scale=0.5,
    )

    assert layer._last_forward_chunk_size == 5


def test_gather_ctx_zero3_env(monkeypatch):
    """Exercise the DS_ZERO_STAGE=3 branch in _maybe_gather_base_params_ctx."""
    calls = {"enter": 0, "exit": 0}

    @contextmanager
    def recorder(_module):
        calls["enter"] += 1
        try:
            yield None
        finally:
            calls["exit"] += 1

    monkeypatch.setattr(dora_mod, "gather_params_ctx", recorder)
    monkeypatch.setenv("DS_ZERO_STAGE", "3")
    monkeypatch.delenv("PEFT_FORCE_GATHER", raising=False)
    # DS_ZERO_STAGE is only checked when distributed is initialized (P2.7),
    # so we must simulate an initialized process group for the env var path.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    base = nn.Linear(8, 8, bias=False)
    lora_A = nn.Linear(8, 4, bias=False)
    lora_B = nn.Linear(4, 8, bias=False)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=1.0)

    calls["enter"] = 0
    calls["exit"] = 0

    x = torch.randn(2, 8)
    layer(x, lora_A=lora_A, lora_B=lora_B, scaling=1.0, base_layer=base)

    assert calls["enter"] >= 1
    assert calls["exit"] >= 1


@pytest.mark.parametrize(("env_value", "expected"), [("1", True), ("0", False)])
def test_is_zero3_active_force_gather_override_is_cached(monkeypatch, env_value, expected):
    """Explicit PEFT_FORCE_GATHER values should short-circuit repeated DS checks."""
    monkeypatch.setenv("PEFT_FORCE_GATHER", env_value)
    monkeypatch.delenv("DS_ZERO_STAGE", raising=False)
    dora_mod._invalidate_fused_cache()

    calls = {"count": 0}

    def _unexpected_probe():
        calls["count"] += 1
        raise AssertionError("Explicit PEFT_FORCE_GATHER override should skip DS probing")

    monkeypatch.setattr(dora_mod, "check_deepspeed_zero3_enabled", _unexpected_probe)

    assert dora_mod._is_zero3_active() is expected
    assert dora_mod._is_zero3_active() is expected
    assert calls["count"] == 0


def test_is_zero3_active_unset_false_rechecks(monkeypatch):
    """Without PEFT_FORCE_GATHER, auto-detected False should not be cached."""
    monkeypatch.delenv("PEFT_FORCE_GATHER", raising=False)
    monkeypatch.delenv("DS_ZERO_STAGE", raising=False)
    dora_mod._invalidate_fused_cache()

    calls = {"count": 0}

    def _false_probe():
        calls["count"] += 1
        return False

    monkeypatch.setattr(dora_mod, "check_deepspeed_zero3_enabled", _false_probe)
    # Simulate distributed being initialized so check_deepspeed_zero3_enabled is called
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    assert dora_mod._is_zero3_active() is False
    assert dora_mod._is_zero3_active() is False
    assert calls["count"] == 2


def test_is_zero3_active_unset_true_is_cached(monkeypatch):
    """Without PEFT_FORCE_GATHER, auto-detected True should still be cached."""
    monkeypatch.delenv("PEFT_FORCE_GATHER", raising=False)
    monkeypatch.delenv("DS_ZERO_STAGE", raising=False)
    dora_mod._invalidate_fused_cache()

    calls = {"count": 0}

    def _true_probe():
        calls["count"] += 1
        return True

    monkeypatch.setattr(dora_mod, "check_deepspeed_zero3_enabled", _true_probe)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    assert dora_mod._is_zero3_active() is True
    assert dora_mod._is_zero3_active() is True
    assert calls["count"] == 1


def test_is_zero3_active_skips_ds_check_without_distributed(monkeypatch):
    """When distributed is not initialized, skip expensive DS check entirely."""
    monkeypatch.delenv("PEFT_FORCE_GATHER", raising=False)
    monkeypatch.delenv("DS_ZERO_STAGE", raising=False)
    dora_mod._invalidate_fused_cache()

    calls = {"count": 0}

    def _should_not_be_called():
        calls["count"] += 1
        return True

    monkeypatch.setattr(dora_mod, "check_deepspeed_zero3_enabled", _should_not_be_called)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    assert dora_mod._is_zero3_active() is False
    assert calls["count"] == 0, "check_deepspeed_zero3_enabled should not be called without distributed"


def test_is_zero3_active_late_init_scenario(monkeypatch):
    """Late-init: distributed initializes AFTER first _is_zero3_active() call.

    HF Trainer initializes DeepSpeed after PEFT wrapping. The asymmetric cache
    (only cache True, re-evaluate False) is designed for this: the first call
    returns False (not cached), then after distributed init the next call must
    return True (and then cache it).
    """
    monkeypatch.delenv("PEFT_FORCE_GATHER", raising=False)
    monkeypatch.delenv("DS_ZERO_STAGE", raising=False)
    dora_mod._invalidate_fused_cache()

    dist_state = {"initialized": False}
    ds_state = {"zero3": False}

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: dist_state["initialized"])
    monkeypatch.setattr(dora_mod, "check_deepspeed_zero3_enabled", lambda: ds_state["zero3"])

    # Phase 1: distributed not initialized yet → False, not cached
    assert dora_mod._is_zero3_active() is False

    # Phase 2: distributed initializes, DeepSpeed ZeRO-3 becomes active
    dist_state["initialized"] = True
    ds_state["zero3"] = True

    # Must detect the change (False was not cached)
    assert dora_mod._is_zero3_active() is True

    # Phase 3: now True IS cached — even if we flip the mock, it stays True
    ds_state["zero3"] = False
    assert dora_mod._is_zero3_active() is True


def test_forward_chunk_threshold_caching(monkeypatch):
    """Verify that _get_forward_chunk_threshold_bytes caches correctly."""
    monkeypatch.setenv("PEFT_DORA_FWD_CHUNK_MB", "32")
    val1 = dora_mod._get_forward_chunk_threshold_bytes()
    assert val1 == 32 * 1024 * 1024

    # Calling again without invalidating should return cached value
    monkeypatch.setenv("PEFT_DORA_FWD_CHUNK_MB", "64")
    val2 = dora_mod._get_forward_chunk_threshold_bytes()
    assert val2 == val1  # still cached

    # After invalidation, should pick up new value
    _invalidate_fused_cache()
    val3 = dora_mod._get_forward_chunk_threshold_bytes()
    assert val3 == 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# Conv1d / Conv3d helpers
# ---------------------------------------------------------------------------


def _random_conv1d_tensors(out_channels, in_channels, rank, kernel_size, dtype):
    """Generate random tensors for Conv1d DoRA testing.

    Weight: (out_channels, in_channels, kernel_size) -- 3D
    lora_A: (rank, in_channels, kernel_size)
    lora_B: (out_channels, rank, 1)
    """
    device = _device_for_dtype(dtype)
    base_weight = torch.randn((out_channels, in_channels, kernel_size), device=device, dtype=dtype)
    lora_A = torch.randn((rank, in_channels, kernel_size), device=device, dtype=dtype)
    lora_B = torch.randn((out_channels, rank, 1), device=device, dtype=dtype)
    return base_weight, lora_A, lora_B


def _random_conv3d_tensors(out_channels, in_channels, rank, kernel_size, dtype):
    """Generate random tensors for Conv3d DoRA testing.

    Weight: (out_channels, in_channels, kD, kH, kW) -- 5D
    lora_A: (rank, in_channels, kD, kH, kW)
    lora_B: (out_channels, rank, 1, 1, 1)
    """
    device = _device_for_dtype(dtype)
    k = kernel_size
    base_weight = torch.randn((out_channels, in_channels, k, k, k), device=device, dtype=dtype)
    lora_A = torch.randn((rank, in_channels, k, k, k), device=device, dtype=dtype)
    lora_B = torch.randn((out_channels, rank, 1, 1, 1), device=device, dtype=dtype)
    return base_weight, lora_A, lora_B


# ---------------------------------------------------------------------------
# Conv1d tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [4, 8])
@pytest.mark.parametrize("scaling", [0.0, 0.5, 1.0])
def test_conv1d_norm_equivalence_vs_dense(dtype, rank, scaling):
    torch.manual_seed(100)
    out_channels, in_channels, kernel = 32, 16, 3
    base_weight, lora_A, lora_B = _random_conv1d_tensors(out_channels, in_channels, rank, kernel, dtype)

    layer = DoraConv1dLayer(fan_in_fan_out=False)
    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=scaling,
        )

    # Dense reference: materialize B @ A, reshape, compute per-output-channel norm
    flat_A = lora_A.reshape(rank, -1).to(torch.float64)
    flat_B = lora_B.reshape(out_channels, -1).to(torch.float64)
    BA = (flat_B @ flat_A).reshape_as(base_weight)
    combined = base_weight.to(torch.float64) + scaling * BA
    dims = tuple(range(1, base_weight.dim()))
    ref = combined.norm(p=2, dim=dims, keepdim=True).transpose(1, 0)
    ref = ref.to(weight_norm.dtype)

    diff = _max_diff(weight_norm, ref)
    tol = 5e-3 if dtype in (torch.bfloat16, torch.float16) else 2e-5
    assert diff <= tol


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_conv1d_forward_with_base_result(dtype):
    torch.manual_seed(101)
    device = _device_for_dtype(dtype)
    out_channels, in_channels, kernel, rank = 16, 8, 3, 4
    scaling = 0.6
    padding = kernel // 2

    base = nn.Conv1d(in_channels, out_channels, kernel, padding=padding, bias=True, dtype=dtype).to(device)
    lora_A = nn.Conv1d(in_channels, rank, kernel, padding=padding, bias=False, dtype=dtype).to(device)
    lora_B = nn.Conv1d(rank, out_channels, 1, bias=False, dtype=dtype).to(device)
    layer = DoraConv1dLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    x = torch.randn(2, in_channels, 16, device=device, dtype=dtype)
    base_result = base(x).detach()
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        base_result=base_result,
    )

    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base.weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=scaling,
        )
    mag_norm_scale = layer.weight / weight_norm
    # When base_result is provided, bias is subtracted inside forward
    ref = (mag_norm_scale - 1) * (base_result - base.bias.view(1, -1, 1))
    ref = ref + mag_norm_scale * (scaling * lora_B(lora_A(x)))

    tol = 1e-5 if dtype == torch.float32 else 5e-3
    assert _max_diff(out, ref.to(out.dtype)) <= tol


def test_conv1d_backward_grad_flow():
    torch.manual_seed(102)
    dtype = torch.float32
    device = torch.device("cpu")
    out_channels, in_channels, kernel, rank = 12, 6, 3, 4
    scaling = 0.5
    padding = kernel // 2

    base = nn.Conv1d(in_channels, out_channels, kernel, padding=padding, bias=False, dtype=dtype).to(device)
    base.weight.requires_grad_(False)
    lora_A = nn.Conv1d(in_channels, rank, kernel, padding=padding, bias=False, dtype=dtype).to(device)
    lora_B = nn.Conv1d(rank, out_channels, 1, bias=False, dtype=dtype).to(device)
    layer = DoraConv1dLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    x = torch.randn(2, in_channels, 10, device=device, dtype=dtype, requires_grad=True)
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
    )
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert lora_A.weight.grad is not None and torch.all(torch.isfinite(lora_A.weight.grad))
    assert lora_B.weight.grad is not None and torch.all(torch.isfinite(lora_B.weight.grad))
    assert layer.weight.grad is not None and torch.all(torch.isfinite(layer.weight.grad))


# ---------------------------------------------------------------------------
# Conv3d tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rank", [4, 8])
@pytest.mark.parametrize("scaling", [0.0, 0.5, 1.0])
def test_conv3d_norm_equivalence_vs_dense(dtype, rank, scaling):
    torch.manual_seed(103)
    out_channels, in_channels, kernel = 16, 8, 2
    base_weight, lora_A, lora_B = _random_conv3d_tensors(out_channels, in_channels, rank, kernel, dtype)

    layer = DoraConv3dLayer(fan_in_fan_out=False)
    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=scaling,
        )

    flat_A = lora_A.reshape(rank, -1).to(torch.float64)
    flat_B = lora_B.reshape(out_channels, -1).to(torch.float64)
    BA = (flat_B @ flat_A).reshape_as(base_weight)
    combined = base_weight.to(torch.float64) + scaling * BA
    dims = tuple(range(1, base_weight.dim()))
    ref = combined.norm(p=2, dim=dims, keepdim=True).transpose(1, 0)
    ref = ref.to(weight_norm.dtype)

    diff = _max_diff(weight_norm, ref)
    tol = 5e-3 if dtype in (torch.bfloat16, torch.float16) else 2e-5
    assert diff <= tol


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_conv3d_forward_with_base_result(dtype):
    torch.manual_seed(104)
    device = _device_for_dtype(dtype)
    out_channels, in_channels, kernel, rank = 8, 4, 2, 4
    scaling = 0.5
    padding = 0

    base = nn.Conv3d(in_channels, out_channels, kernel, padding=padding, bias=True, dtype=dtype).to(device)
    lora_A = nn.Conv3d(in_channels, rank, kernel, padding=padding, bias=False, dtype=dtype).to(device)
    lora_B = nn.Conv3d(rank, out_channels, 1, bias=False, dtype=dtype).to(device)
    layer = DoraConv3dLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    x = torch.randn(1, in_channels, 4, 4, 4, device=device, dtype=dtype)
    base_result = base(x).detach()
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        base_result=base_result,
    )

    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base.weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=scaling,
        )
    mag_norm_scale = layer.weight / weight_norm
    ref = (mag_norm_scale - 1) * (base_result - base.bias.view(1, -1, 1, 1, 1))
    ref = ref + mag_norm_scale * (scaling * lora_B(lora_A(x)))

    tol = 1e-5 if dtype == torch.float32 else 5e-3
    assert _max_diff(out, ref.to(out.dtype)) <= tol


def test_conv3d_forward_without_base_result():
    """Exercises the conv_fn=F.conv3d path (base_result=None)."""
    torch.manual_seed(105)
    dtype = torch.float32
    device = torch.device("cpu")
    out_channels, in_channels, kernel, rank = 8, 4, 2, 4
    scaling = 0.5
    padding = 0

    base = nn.Conv3d(in_channels, out_channels, kernel, padding=padding, bias=False, dtype=dtype).to(device)
    lora_A = nn.Conv3d(in_channels, rank, kernel, padding=padding, bias=False, dtype=dtype).to(device)
    lora_B = nn.Conv3d(rank, out_channels, 1, bias=False, dtype=dtype).to(device)
    layer = DoraConv3dLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    x = torch.randn(1, in_channels, 4, 4, 4, device=device, dtype=dtype)
    # No base_result -- forward should compute it internally via F.conv3d
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
    )

    # Manually compute expected output
    with torch.no_grad():
        weight_norm = layer._get_weight_norm_conv_factored(
            base_weight=base.weight,
            lora_A_w=lora_A.weight,
            lora_B_w=lora_B.weight,
            scaling=scaling,
        )
    mag_norm_scale = layer.weight / weight_norm
    internal_base = F.conv3d(
        x,
        base.weight,
        bias=None,
        stride=base.stride,
        padding=base.padding,
        dilation=base.dilation,
        groups=base.groups,
    )
    ref = (mag_norm_scale - 1) * internal_base + mag_norm_scale * (scaling * lora_B(lora_A(x)))
    assert _max_diff(out, ref) <= 1e-5


def test_conv3d_backward_grad_flow():
    torch.manual_seed(106)
    dtype = torch.float32
    device = torch.device("cpu")
    out_channels, in_channels, kernel, rank = 8, 4, 2, 4
    scaling = 0.5

    base = nn.Conv3d(in_channels, out_channels, kernel, padding=0, bias=False, dtype=dtype).to(device)
    base.weight.requires_grad_(False)
    lora_A = nn.Conv3d(in_channels, rank, kernel, padding=0, bias=False, dtype=dtype).to(device)
    lora_B = nn.Conv3d(rank, out_channels, 1, bias=False, dtype=dtype).to(device)
    layer = DoraConv3dLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    x = torch.randn(1, in_channels, 4, 4, 4, device=device, dtype=dtype, requires_grad=True)
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
    )
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert lora_A.weight.grad is not None and torch.all(torch.isfinite(lora_A.weight.grad))
    assert lora_B.weight.grad is not None and torch.all(torch.isfinite(lora_B.weight.grad))
    assert layer.weight.grad is not None and torch.all(torch.isfinite(layer.weight.grad))


# ---------------------------------------------------------------------------
# Near-zero magnitude / weight-norm edge cases
# ---------------------------------------------------------------------------


def test_near_zero_magnitude_forward_finite():
    """Forward output should remain finite even when magnitude is near zero (~1e-7)."""
    torch.manual_seed(107)
    dtype = torch.float32
    device = torch.device("cpu")

    base = nn.Linear(16, 8, bias=False, dtype=dtype).to(device)
    base.weight.requires_grad_(False)
    rank = 4
    lora_A = nn.Linear(16, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype).to(device)
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

    # Overwrite magnitude to near-zero values
    with torch.no_grad():
        layer.weight.fill_(1e-7)

    x = torch.randn(3, 16, device=device, dtype=dtype)
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.5,
        base_layer=base,
    )
    assert torch.all(torch.isfinite(out)), f"Non-finite values in output: {out}"


def test_near_zero_weight_norm_clamp():
    """When norm_sq is near-negative due to floating point, sqrt should produce 0.0, not NaN.

    Construct inputs where base_weight and lora contribution nearly cancel,
    yielding a tiny (or slightly negative due to rounding) norm_sq. The clamp
    inside _get_weight_norm_linear should prevent NaN.
    """
    torch.manual_seed(108)
    dtype = torch.float32
    device = torch.device("cpu")

    out_features, in_features, rank = 4, 8, 4
    # Make base_weight and B @ A nearly cancel each other
    lora_A = torch.randn(rank, in_features, device=device, dtype=dtype)
    lora_B = torch.randn(out_features, rank, device=device, dtype=dtype)
    BA = lora_B @ lora_A
    scaling = 1.0
    # base_weight = -(scaling * B@A) + tiny perturbation
    base_weight = -scaling * BA + 1e-20 * torch.randn_like(BA)

    layer = DoraLinearLayer(fan_in_fan_out=False)
    with torch.no_grad():
        norm = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=scaling,
        )

    assert torch.all(torch.isfinite(norm)), f"Non-finite norm: {norm}"
    assert torch.all(norm >= 0.0), f"Negative norm values: {norm}"


# ---------------------------------------------------------------------------
# Reference vs optimized forward equivalence
# ---------------------------------------------------------------------------


def test_reference_vs_optimized_forward_equivalence():
    """Import the reference DoRA implementation from docs/ and compare outputs
    with the optimized DoraLinearLayer."""
    import importlib.util

    _ref_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "dora.reference_hf_peft.py")
    spec = importlib.util.spec_from_file_location("dora_reference", _ref_path)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    RefDoraLinearLayer = ref_mod.DoraLinearLayer

    torch.manual_seed(109)
    dtype = torch.float32
    device = torch.device("cpu")

    base = nn.Linear(24, 12, bias=False, dtype=dtype).to(device)
    rank = 6
    lora_A = nn.Linear(24, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 12, bias=False, dtype=dtype).to(device)
    scaling = 0.7

    # Set up the optimized layer
    opt_layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    opt_layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    # Set up the reference layer sharing the same magnitude
    ref_layer = RefDoraLinearLayer(fan_in_fan_out=False).to(device)
    ref_layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    # Ensure same magnitude
    with torch.no_grad():
        ref_layer.weight.copy_(opt_layer.weight)

    x = torch.randn(5, 24, device=device, dtype=dtype)

    opt_out = opt_layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
    )

    ref_out = ref_layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
    )

    assert _max_diff(opt_out, ref_out) <= 1e-5, (
        f"Optimized and reference outputs diverge: max diff = {_max_diff(opt_out, ref_out)}"
    )


# ---------------------------------------------------------------------------
# place_on_cpu
# ---------------------------------------------------------------------------


def test_update_layer_place_on_cpu():
    """Verify that magnitude ends up on CPU when place_on_cpu=True."""
    torch.manual_seed(110)
    dtype = torch.float32

    base = nn.Linear(16, 8, bias=False, dtype=dtype)
    rank = 4
    lora_A = nn.Linear(16, rank, bias=False, dtype=dtype)
    lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A.weight,
        lora_B=lora_B.weight,
        scaling=0.5,
        place_on_cpu=True,
    )

    assert layer.weight.device == torch.device("cpu"), f"Expected magnitude on CPU, got {layer.weight.device}"
    assert layer.weight.requires_grad is True


# ---------------------------------------------------------------------------
# Double backward
# ---------------------------------------------------------------------------


def test_inplace_safety_when_result_requires_grad_is_false():
    """Regression test for the variants.py autograd safety fix.

    Scenario: frozen base model at the first layer where input embeddings
    don't require grad.  base(x) produces ``result`` with requires_grad=False.
    The DoRA compose expression saves ``base_result`` (which aliases ``result``)
    in the autograd graph for the magnitude gradient path.

    **Old (buggy) code** checked ``result.requires_grad`` — which was False —
    and allowed in-place ``result.add_(delta)``, corrupting the tensor saved
    by autograd and producing incorrect magnitude gradients.

    **New (fixed) code** checks ``torch.is_grad_enabled()`` instead, correctly
    recognizing that autograd is active (LoRA params require grad) even when
    ``result`` itself doesn't.
    """
    torch.manual_seed(200)
    dtype = torch.float32
    device = torch.device("cpu")

    # Frozen base (no grad on weight or bias)
    base = nn.Linear(16, 8, bias=False, dtype=dtype).to(device)
    base.weight.requires_grad_(False)

    rank = 4
    scaling = 0.5
    lora_A = nn.Linear(16, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype).to(device)
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=scaling)

    # Input without requires_grad (simulates first layer after embedding lookup)
    x = torch.randn(4, 16, device=device, dtype=dtype, requires_grad=False)

    # base(x) has requires_grad=False since both base.weight and x don't require grad
    result = base(x)
    assert not result.requires_grad, "Precondition: result should not require grad"

    # Pass base_result=result (aliasing, as happens in DoraLinearVariant.forward
    # when bias=None and dropout=Identity)
    delta = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        base_result=result,  # aliased with result
    )

    # Simulate what DoraLinearVariant.forward does:
    # delta_depends_on_result = True (base_result is result, bias is None)
    # The fix ensures we take the out-of-place path here
    assert torch.is_grad_enabled(), "Precondition: grad should be enabled"
    final = result + delta  # out-of-place (what the fixed code does)

    loss = final.sum()
    loss.backward()

    # All LoRA parameters should have finite, non-zero gradients
    assert lora_A.weight.grad is not None, "lora_A should have grad"
    assert lora_B.weight.grad is not None, "lora_B should have grad"
    assert layer.weight.grad is not None, "magnitude should have grad"
    assert torch.all(torch.isfinite(lora_A.weight.grad))
    assert torch.all(torch.isfinite(lora_B.weight.grad))
    assert torch.all(torch.isfinite(layer.weight.grad))

    # Verify magnitude gradients are non-trivial (not all zeros)
    assert layer.weight.grad.abs().max() > 1e-8, "Magnitude gradients should be non-trivial"


def test_double_backward_documents_behavior():
    """Document whether higher-order (double) backward through DoRA works or raises.

    DoRA detaches the weight_norm from the graph (per the paper, Section 4.3),
    so only first-order gradients through lora_A/lora_B/magnitude are expected.
    This test documents the current behavior: double backward may or may not be
    supported depending on the implementation. We simply verify it either
    succeeds cleanly or raises a RuntimeError.
    """
    torch.manual_seed(111)
    dtype = torch.float32
    device = torch.device("cpu")

    base = nn.Linear(12, 8, bias=False, dtype=dtype).to(device)
    base.weight.requires_grad_(False)
    rank = 4
    lora_A = nn.Linear(12, rank, bias=False, dtype=dtype).to(device)
    lora_B = nn.Linear(rank, 8, bias=False, dtype=dtype).to(device)
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

    x = torch.randn(3, 12, device=device, dtype=dtype, requires_grad=True)
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.5,
        base_layer=base,
    )

    # First backward with create_graph=True to attempt double backward
    loss = out.sum()
    try:
        grads = torch.autograd.grad(loss, [lora_A.weight, lora_B.weight, layer.weight], create_graph=True)
        # If first-order with create_graph succeeds, try second-order
        grad_sum = sum(g.sum() for g in grads if g is not None)
        try:
            grad_sum.backward()
            double_backward_works = True
        except RuntimeError:
            double_backward_works = False
    except RuntimeError:
        double_backward_works = False

    # Document the result -- we don't assert a specific outcome, just that
    # the code doesn't produce silent corruption
    if double_backward_works:
        # All second-order grads should be finite if they exist
        for param in [lora_A.weight, lora_B.weight, layer.weight]:
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad))


# ---------------------------------------------------------------------------
# Bug #1: ZeRO-3 gather scope must include LoRA factors
# ---------------------------------------------------------------------------


def test_zero3_gather_includes_lora_factors(recording_gather):
    """Verify that _maybe_gather_base_params_ctx gathers LoRA module params too.

    Under ZeRO-3 adapter weights can also be sharded.  The gather context must
    include lora_A and lora_B parameters so that weight_norm is computed from
    fully gathered tensors.
    """
    base = nn.Linear(8, 8, bias=False)
    lora_A = nn.Linear(8, 4, bias=False)
    lora_B = nn.Linear(4, 8, bias=False)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=1.0)

    recording_gather.clear()

    x = torch.randn(2, 8)
    layer(x, lora_A=lora_A, lora_B=lora_B, scaling=1.0, base_layer=base)

    # The gathered params must include base, lora_A, AND lora_B weights.
    gathered_ids = {id(p) for p in recording_gather}
    assert id(base.weight) in gathered_ids, "base_layer weight was not gathered"
    assert id(lora_A.weight) in gathered_ids, "lora_A weight was not gathered"
    assert id(lora_B.weight) in gathered_ids, "lora_B weight was not gathered"


def test_zero3_gather_lora_factors_sharded_parity(monkeypatch):
    """Simulate sharded LoRA factors and verify forward parity with gathered ref.

    Outside the gather context, lora_A/lora_B parameters are replaced with
    zeros (simulating sharded state).  Inside the gather context, they're
    restored.  If the gather scope is correct, the forward output must match
    the fully-gathered reference.
    """
    torch.manual_seed(42)
    base = nn.Linear(16, 8, bias=False)
    lora_A = nn.Linear(16, 4, bias=False)
    lora_B = nn.Linear(4, 8, bias=False)

    # Save full weights for reference
    full_A = lora_A.weight.data.clone()
    full_B = lora_B.weight.data.clone()
    full_base = base.weight.data.clone()

    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

    x = torch.randn(3, 16)

    # Reference output with fully gathered weights
    monkeypatch.setattr(dora_mod, "gather_params_ctx", None)
    ref_out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base)

    # Now simulate sharded state: zero out weights outside gather, restore inside
    @contextmanager
    def _shard_then_gather(params):
        """Restore full weights inside context (simulates ZeRO-3 gather).

        In a real ZeRO-3 setup, parameters would be re-sharded on context exit.
        Here we only restore them inside the context and leave them restored
        afterwards, because the forward path also uses lora_A/lora_B modules
        directly outside the gather scope (for ``lora_B(lora_A(x))``).  The key
        property being tested is that *norm computation* sees full weights.
        """
        if isinstance(params, tuple):
            for p in params:
                if id(p) == id(lora_A.weight):
                    p.data.copy_(full_A)
                elif id(p) == id(lora_B.weight):
                    p.data.copy_(full_B)
                elif id(p) == id(base.weight):
                    p.data.copy_(full_base)
            yield
        else:
            yield

    # Zero out LoRA weights to simulate sharding
    lora_A.weight.data.zero_()
    lora_B.weight.data.zero_()

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _shard_then_gather)
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")
    dora_mod._invalidate_fused_cache()
    # Re-init magnitude with gathered weights
    lora_A.weight.data.copy_(full_A)
    lora_B.weight.data.copy_(full_B)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)
    # Zero again for the forward test
    lora_A.weight.data.zero_()
    lora_B.weight.data.zero_()

    try:
        out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base)

        assert _max_diff(out, ref_out) <= 1e-5, (
            f"Forward with simulated shard/gather diverged from reference: {_max_diff(out, ref_out)}"
        )
    finally:
        # Restore weights even on failure so zeroed state doesn't leak to
        # other tests if modules were shared (e.g. parametrized expansion).
        lora_A.weight.data.copy_(full_A)
        lora_B.weight.data.copy_(full_B)


def test_zero3_gather_conv_includes_lora_factors(recording_gather):
    """Conv DoRA forward should also gather LoRA factors under ZeRO-3."""
    base = nn.Conv2d(4, 8, 3, padding=1, bias=False)
    lora_A = nn.Conv2d(4, 2, 3, padding=1, bias=False)
    lora_B = nn.Conv2d(2, 8, 1, bias=False)
    layer = DoraConv2dLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

    recording_gather.clear()

    x = torch.randn(1, 4, 6, 6)
    base_result = base(x).detach()
    layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base, base_result=base_result)

    gathered_ids = {id(p) for p in recording_gather}
    assert id(base.weight) in gathered_ids, "base_layer weight was not gathered"
    assert id(lora_A.weight) in gathered_ids, "lora_A weight was not gathered"
    assert id(lora_B.weight) in gathered_ids, "lora_B weight was not gathered"


def test_zero3_gather_conv1d_includes_lora_factors(recording_gather):
    """Conv1d DoRA forward should also gather LoRA factors under ZeRO-3."""
    base = nn.Conv1d(4, 8, 3, padding=1, bias=False)
    lora_A = nn.Conv1d(4, 2, 3, padding=1, bias=False)
    lora_B = nn.Conv1d(2, 8, 1, bias=False)
    layer = DoraConv1dLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

    recording_gather.clear()

    x = torch.randn(1, 4, 16)
    base_result = base(x).detach()
    layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base, base_result=base_result)

    gathered_ids = {id(p) for p in recording_gather}
    assert id(base.weight) in gathered_ids, "base_layer weight was not gathered"
    assert id(lora_A.weight) in gathered_ids, "lora_A weight was not gathered"
    assert id(lora_B.weight) in gathered_ids, "lora_B weight was not gathered"


def test_zero3_gather_embedding_includes_lora_tensors(recording_gather):
    """Embedding DoRA forward should gather raw lora_A/lora_B tensors under ZeRO-3.

    Unlike linear/conv where lora_A/lora_B are nn.Module instances, the
    embedding path receives raw tensors (transposed nn.Parameters from the
    parent module's ParameterDict).  _maybe_gather_base_params_ctx must
    handle these via isinstance(mod, torch.Tensor).
    """
    num_embeddings, embedding_dim, rank = 16, 8, 4
    scaling = 0.5
    base = nn.Embedding(num_embeddings, embedding_dim)

    lora_A_param = torch.randn(rank, num_embeddings)
    lora_B_param = torch.randn(embedding_dim, rank)
    lora_A = lora_A_param.T  # raw tensor, not a module
    lora_B = lora_B_param.T

    layer = DoraEmbeddingLayer(fan_in_fan_out=True)
    layer.update_layer(base_layer=base, lora_A=lora_A_param, lora_B=lora_B_param, scaling=scaling)

    recording_gather.clear()

    x = torch.randint(0, num_embeddings, (3, 5))
    base_result = F.embedding(x, base.weight)
    layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=base_result,
    )

    # base.weight must be gathered (via module.parameters())
    gathered_ids = {id(p) for p in recording_gather}
    assert id(base.weight) in gathered_ids, "base_layer weight was not gathered"
    # lora_A and lora_B are raw tensors — they must appear in the gathered set
    assert id(lora_A) in gathered_ids, "lora_A tensor was not gathered"
    assert id(lora_B) in gathered_ids, "lora_B tensor was not gathered"


# ---------------------------------------------------------------------------
# Bug #2: FSDP2 detection — fail loudly instead of silently no-op
# ---------------------------------------------------------------------------


def test_fsdp2_detection_raises(monkeypatch):
    """_fsdp_full_param_ctx should raise RuntimeError for FSDP2-managed modules."""
    module = nn.Linear(4, 4)

    # Mock _is_fsdp2_managed to return True
    monkeypatch.setattr(dora_mod, "_is_fsdp2_managed", lambda m: True)

    with pytest.raises(RuntimeError, match="FSDP2"):
        with dora_mod._fsdp_full_param_ctx(module):
            pass


def test_fsdp2_detection_allows_fsdp1(monkeypatch):
    """_fsdp_full_param_ctx should not raise for non-FSDP2 modules."""
    module = nn.Linear(4, 4)

    # Mock _is_fsdp2_managed to return False
    monkeypatch.setattr(dora_mod, "_is_fsdp2_managed", lambda m: False)

    # Should not raise
    with dora_mod._fsdp_full_param_ctx(module):
        pass


def test_is_fsdp2_managed_returns_false_no_fsdp():
    """_is_fsdp2_managed should return False for plain modules."""
    module = nn.Linear(4, 4)
    assert dora_mod._is_fsdp2_managed(module) is False


# ---------------------------------------------------------------------------
# Bug #3: CPU-offloaded weights device transfer
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_linear_forward_base_result_none_smoke():
    """Smoke test: linear forward with base_result=None completes on CPU."""
    torch.manual_seed(200)
    base = nn.Linear(8, 4, bias=False)
    lora_A = nn.Linear(8, 2, bias=False)
    lora_B = nn.Linear(2, 4, bias=False)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=1.0)

    x = torch.randn(2, 8)
    out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=1.0, base_layer=base, base_result=None)
    assert torch.all(torch.isfinite(out))
    assert out.shape == (2, 4)


def test_linear_forward_base_result_none_snapshots_live_weight(monkeypatch):
    """base_result=None should not keep a live ZeRO-sharded linear weight reference."""
    torch.manual_seed(2001)
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")
    dora_mod._invalidate_fused_cache()

    base = nn.Linear(8, 4, bias=False)
    full_weight = base.weight.detach().clone()
    lora_A = nn.Linear(8, 2, bias=False)
    lora_B = nn.Linear(2, 4, bias=False)
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.0)
    layer.weight.data.mul_(2.0)

    base.weight.data.zero_()
    gathered_weight = full_weight.clone()

    @contextmanager
    def _fake_gather(params):
        originals = []
        try:
            for param in params:
                if param is base.weight:
                    originals.append((param, param.data))
                    param.data = gathered_weight
            yield
        finally:
            gathered_weight.zero_()
            for param, original in reversed(originals):
                param.data = original

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _fake_gather)
    monkeypatch.setattr(dora_mod, "dequantize_module_weight", lambda module: module.weight)

    x = torch.randn(2, 8)
    out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.0, base_layer=base, base_result=None)
    ref = F.linear(x, full_weight)

    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_embedding_forward_base_result_none_smoke():
    """Smoke test: embedding forward with base_result=None completes on CPU.

    Verifies that:
    - The result dtype is floating-point (not Long from token indices).
    - The base_result=None path matches the base_result-provided path.
    """
    torch.manual_seed(201)
    num_embeddings, embedding_dim, rank = 16, 8, 4
    scaling = 0.5
    base = nn.Embedding(num_embeddings, embedding_dim)
    lora_A_param = torch.randn(rank, num_embeddings)
    lora_B_param = torch.randn(embedding_dim, rank)
    lora_A = lora_A_param.T
    lora_B = lora_B_param.T

    layer = DoraEmbeddingLayer(fan_in_fan_out=True)
    layer.update_layer(base_layer=base, lora_A=lora_A_param, lora_B=lora_B_param, scaling=scaling)

    x = torch.randint(0, num_embeddings, (3, 5))

    # Run with base_result=None (the path under test)
    _mag_norm_scale, result_none = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=None,
    )
    assert torch.all(torch.isfinite(result_none))
    # Result must be floating-point — not Long (token index dtype).
    assert result_none.is_floating_point(), (
        f"Expected floating-point result, got {result_none.dtype}. "
        "This likely means the embedding weight was cast to the index dtype."
    )

    # Run with base_result provided (the common path) and compare.
    base_result = F.embedding(x, base.weight)
    _, result_with_base = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=base_result,
    )
    torch.testing.assert_close(result_none, result_with_base, atol=1e-5, rtol=1e-5)


def test_conv_forward_base_result_none_smoke():
    """Smoke test: conv forward with base_result=None completes on CPU."""
    torch.manual_seed(202)
    base = nn.Conv2d(4, 8, 3, padding=1, bias=False)
    lora_A = nn.Conv2d(4, 2, 3, padding=1, bias=False)
    lora_B = nn.Conv2d(2, 8, 1, bias=False)
    layer = DoraConv2dLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

    x = torch.randn(1, 4, 6, 6)
    out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base, base_result=None)
    assert torch.all(torch.isfinite(out))
    assert out.shape[1] == 8


def test_conv_forward_base_result_none_snapshots_live_weight(monkeypatch):
    """base_result=None should not keep a live ZeRO-sharded conv weight reference."""
    torch.manual_seed(2002)
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")
    dora_mod._invalidate_fused_cache()

    base = nn.Conv2d(4, 8, 3, padding=1, bias=False)
    full_weight = base.weight.detach().clone()
    lora_A = nn.Conv2d(4, 2, 3, padding=1, bias=False)
    lora_B = nn.Conv2d(2, 8, 1, bias=False)
    layer = DoraConv2dLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.0)
    layer.weight.data.mul_(2.0)

    base.weight.data.zero_()
    gathered_weight = full_weight.clone()

    @contextmanager
    def _fake_gather(params):
        originals = []
        try:
            for param in params:
                if param is base.weight:
                    originals.append((param, param.data))
                    param.data = gathered_weight
            yield
        finally:
            gathered_weight.zero_()
            for param, original in reversed(originals):
                param.data = original

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _fake_gather)
    monkeypatch.setattr(dora_mod, "dequantize_module_weight", lambda module: module.weight)

    x = torch.randn(1, 4, 6, 6)
    out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.0, base_layer=base, base_result=None)
    ref = F.conv2d(
        x,
        full_weight,
        bias=None,
        stride=base.stride,
        padding=base.padding,
        dilation=base.dilation,
        groups=base.groups,
    )

    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


@requires_cuda
def test_linear_forward_cpu_offloaded_weight_to_cuda():
    """Actual cross-device test: base_layer on CPU, magnitude/x on CUDA.

    Simulates ZeRO-3 CPU offload where dequantize_module_weight returns CPU
    tensors but activations and magnitude live on GPU.  Without the
    .to(device=...) fix, this raises a device mismatch error.
    """
    torch.manual_seed(203)
    device = torch.device("cuda")

    # Create base on CPU (simulates CPU-offloaded gathered weight)
    base = nn.Linear(16, 8, bias=False)
    lora_A = nn.Linear(16, 4, bias=False).to(device)
    lora_B = nn.Linear(4, 8, bias=False).to(device)

    # Layer magnitude on CUDA
    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    # update_layer with CPU base — move lora weights to CPU for init, then back
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A.weight.data.cpu(),
        lora_B=lora_B.weight.data.cpu(),
        scaling=0.5,
    )
    layer = layer.to(device)

    x = torch.randn(2, 16, device=device)
    # base_result=None: forces the path that uses dequantized base weight
    # base_layer stays on CPU to simulate CPU offload
    out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base, base_result=None)
    assert out.device.type == "cuda"
    assert torch.all(torch.isfinite(out))
    assert out.shape == (2, 8)


@requires_cuda
def test_linear_forward_cpu_weight_norm_gpu_magnitude():
    """Cross-device: weight_norm computed from CPU base, divided by GPU magnitude.

    With base_result provided (common path), weight_norm is computed under the
    gather context from the CPU base_layer weight.  magnitude lives on CUDA.
    Without .to(device=magnitude.device), the division crashes.
    """
    torch.manual_seed(204)
    device = torch.device("cuda")

    base = nn.Linear(16, 8, bias=True)  # CPU
    lora_A = nn.Linear(16, 4, bias=False).to(device)
    lora_B = nn.Linear(4, 8, bias=False).to(device)

    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A.weight.data.cpu(),
        lora_B=lora_B.weight.data.cpu(),
        scaling=0.5,
    )
    layer = layer.to(device)

    x = torch.randn(2, 16, device=device)
    base_result = torch.randn(2, 8, device=device)
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=0.5,
        base_layer=base,
        base_result=base_result,
    )
    assert out.device.type == "cuda"
    assert torch.all(torch.isfinite(out))


@requires_cuda
def test_conv_forward_cpu_offloaded_weight_to_cuda():
    """Cross-device: conv base_layer on CPU, x on CUDA, base_result=None."""
    torch.manual_seed(205)
    device = torch.device("cuda")

    base = nn.Conv2d(4, 8, 3, padding=1, bias=False)  # CPU
    lora_A = nn.Conv2d(4, 2, 3, padding=1, bias=False).to(device)
    lora_B = nn.Conv2d(2, 8, 1, bias=False).to(device)

    layer = DoraConv2dLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A.weight.data.cpu(),
        lora_B=lora_B.weight.data.cpu(),
        scaling=0.5,
    )
    layer = layer.to(device)

    x = torch.randn(1, 4, 6, 6, device=device)
    out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base, base_result=None)
    assert out.device.type == "cuda"
    assert torch.all(torch.isfinite(out))


@requires_cuda
def test_conv1d_forward_cpu_offloaded_weight_to_cuda():
    """Cross-device: Conv1d base_layer on CPU, x on CUDA, base_result=None."""
    torch.manual_seed(206)
    device = torch.device("cuda")

    base = nn.Conv1d(4, 8, 3, padding=1, bias=False)  # CPU
    lora_A = nn.Conv1d(4, 2, 3, padding=1, bias=False).to(device)
    lora_B = nn.Conv1d(2, 8, 1, bias=False).to(device)

    layer = DoraConv1dLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A.weight.data.cpu(),
        lora_B=lora_B.weight.data.cpu(),
        scaling=0.5,
    )
    layer = layer.to(device)

    x = torch.randn(1, 4, 16, device=device)
    out = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base, base_result=None)
    assert out.device.type == "cuda"
    assert torch.all(torch.isfinite(out))


@requires_cuda
def test_conv3d_forward_cpu_offloaded_bias_cross_device():
    """Cross-device: Conv3d with bias, base_layer on CPU, x on CUDA.

    Conv3d has extra spatial dimensions in the bias reshape path
    (1, -1, 1, 1, 1) — exercise that under cross-device conditions.
    Tests both base_result=None (full weight materialization) and
    base_result provided (bias subtraction path).
    """
    torch.manual_seed(207)
    device = torch.device("cuda")

    base = nn.Conv3d(4, 8, 3, padding=1, bias=True)  # CPU, with bias
    lora_A = nn.Conv3d(4, 2, 3, padding=1, bias=False).to(device)
    lora_B = nn.Conv3d(2, 8, 1, bias=False).to(device)

    layer = DoraConv3dLayer(fan_in_fan_out=False).to(device)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A.weight.data.cpu(),
        lora_B=lora_B.weight.data.cpu(),
        scaling=0.5,
    )
    layer = layer.to(device)

    x = torch.randn(1, 4, 4, 4, 4, device=device)

    # Path 1: base_result=None — forces dequantized weight materialization
    out_none = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base, base_result=None)
    assert out_none.device.type == "cuda"
    assert torch.all(torch.isfinite(out_none))
    assert out_none.shape[1] == 8

    # Path 2: base_result provided — exercises bias subtraction with reshape
    base_result = torch.randn_like(out_none)
    out_bias = layer(x, lora_A=lora_A, lora_B=lora_B, scaling=0.5, base_layer=base, base_result=base_result)
    assert out_bias.device.type == "cuda"
    assert torch.all(torch.isfinite(out_bias))


@requires_cuda
def test_embedding_forward_cpu_offloaded_weight_to_cuda():
    """Cross-device: embedding base_layer on CPU, magnitude/x on CUDA.

    Simulates ZeRO-3 CPU offload where dequantize_module_weight returns CPU
    tensors but activations and magnitude live on GPU.  The base_result=None
    path triggers weight.to(device=x.device) which must NOT cast dtype.
    LoRA factors live on CUDA (they're the adapter weights, not offloaded).
    """
    torch.manual_seed(210)
    device = torch.device("cuda")

    num_embeddings, embedding_dim, rank = 32, 16, 4
    scaling = 0.5
    base = nn.Embedding(num_embeddings, embedding_dim)  # CPU (offloaded)

    lora_A_param = torch.randn(rank, num_embeddings, device=device)
    lora_B_param = torch.randn(embedding_dim, rank, device=device)
    lora_A = lora_A_param.T
    lora_B = lora_B_param.T

    layer = DoraEmbeddingLayer(fan_in_fan_out=True).to(device)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A_param.cpu(),
        lora_B=lora_B_param.cpu(),
        scaling=scaling,
    )
    layer = layer.to(device)

    x = torch.randint(0, num_embeddings, (3, 5), device=device)

    # base_result=None path (forces weight.to(device=x.device))
    _mag_scale, result_none = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=None,
    )
    assert result_none.device.type == "cuda"
    assert result_none.is_floating_point(), f"Expected float, got {result_none.dtype}"
    assert torch.all(torch.isfinite(result_none))

    # base_result provided path
    base_result = F.embedding(x, base.weight.to(device))
    _, result_with_base = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=base_result,
    )
    assert result_with_base.device.type == "cuda"
    torch.testing.assert_close(result_none, result_with_base, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# View-to-Parameter resolution for ZeRO-3 gather
# ---------------------------------------------------------------------------


def test_gather_resolves_transposed_views_to_base_params(monkeypatch):
    """_maybe_gather_base_params_ctx resolves .T views to the underlying Parameter.

    DeepSpeed GatheredParameters filters by hasattr(p, 'ds_id'), which only
    exists on original nn.Parameter objects, not on views like .T.  The fix
    resolves tensor._base back to the original Parameter so that gathering
    works correctly for embedding LoRA factors.
    """
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")

    base = nn.Linear(8, 4, bias=False)
    param = nn.Parameter(torch.randn(4, 8))
    transposed = param.T  # view of param, no ds_id

    # Track what params are passed to the gather context
    gathered_params = []

    @contextmanager
    def _tracking_gather(params):
        gathered_params.extend(params if isinstance(params, (list, tuple)) else [params])
        yield

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _tracking_gather)

    # Clear cached state so monkeypatched env var takes effect
    dora_mod._invalidate_fused_cache()

    with dora_mod._maybe_gather_base_params_ctx(base, transposed):
        pass

    # The transposed view should have been resolved to the original Parameter.
    # Use `is` (identity) not `==` (value) to avoid ambiguous tensor comparison.
    assert any(p is param for p in gathered_params), (
        f"Expected original Parameter in gathered params, got types: {[type(p).__name__ for p in gathered_params]}"
    )
    assert not any(p is transposed for p in gathered_params), "Transposed view should not be passed directly"


def test_resolve_tensor_base_breaks_self_referential_base_cycle():
    """_resolve_tensor_base should stop if a malformed ``_base`` chain cycles."""

    class _SelfReferentialTensor:
        def __init__(self):
            self.base_reads = 0

        @property
        def _base(self):
            self.base_reads += 1
            if self.base_reads > 2:
                raise AssertionError("_resolve_tensor_base looped on a cyclic _base chain")
            return self

    tensor = _SelfReferentialTensor()

    assert dora_mod._resolve_tensor_base(tensor) is tensor
    assert tensor.base_reads == 1


def test_embedding_forward_clones_gathered_views_for_out_of_scope_autograd(monkeypatch):
    """Embedding DoRA should clone gathered LoRA factors before leaving gather."""
    torch.manual_seed(211)
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")
    dora_mod._invalidate_fused_cache()

    num_embeddings, embedding_dim, rank = 16, 8, 4
    scaling = 0.5
    base = nn.Embedding(num_embeddings, embedding_dim)

    full_A = torch.randn(rank, num_embeddings)
    full_B = torch.randn(embedding_dim, rank)
    lora_A_param = nn.Parameter(full_A.clone())
    lora_B_param = nn.Parameter(full_B.clone())

    layer = DoraEmbeddingLayer(fan_in_fan_out=True)
    layer.update_layer(base_layer=base, lora_A=full_A, lora_B=full_B, scaling=scaling)

    # Simulate ZeRO-3 shards being visible outside the gather scope.
    lora_A_param.data = torch.zeros_like(full_A)
    lora_B_param.data = torch.zeros_like(full_B)
    lora_A = lora_A_param.T
    lora_B = lora_B_param.T

    gather_state = {"active": False}
    gathered_full = {}
    full_params = {
        id(lora_A_param): full_A.clone(),
        id(lora_B_param): full_B.clone(),
    }

    @contextmanager
    def _fake_gather(params):
        originals = []
        gather_state["active"] = True
        try:
            for param in params:
                full = full_params.get(id(param))
                if full is None:
                    continue
                originals.append((param, param.data))
                param.data = full
                gathered_full[id(param)] = full
            yield
        finally:
            for full in gathered_full.values():
                full.zero_()
            gather_state["active"] = False
            for param, original in reversed(originals):
                param.data = original
            gathered_full.clear()

    def _tracking_embed(input_ids, weight):
        assert not gather_state["active"], "Embedding LoRA matmul should use stable tensors after gather exits"
        expected = full_params[id(lora_A_param)]
        assert weight.data_ptr() != expected.data_ptr(), "Embedding LoRA tensor should be cloned off gathered storage"
        return F.embedding(input_ids, weight)

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _fake_gather)

    x = torch.randint(0, num_embeddings, (3, 5))
    base_result = F.embedding(x, base.weight).detach()
    mag_norm_scale, result_dora = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        embed_fn=_tracking_embed,
        base_result=base_result,
    )
    loss = result_dora.float().sum()
    loss.backward()

    full_A_t = full_A.T
    full_B_t = full_B.T
    with torch.no_grad():
        weight = dora_mod.dequantize_module_weight(base)
        lora_weight = (full_A_t @ full_B_t).T
        weight_norm = layer.get_weight_norm(weight, lora_weight, scaling).detach()
        weight_norm = weight_norm.clamp_min(1e-12)
        ref_mag_norm_scale = (layer.weight / weight_norm).detach()
    ref_A = full_A_t.clone().requires_grad_(True)
    ref_B = full_B_t.clone().requires_grad_(True)
    ref_lora_out = F.embedding(x, ref_A) @ ref_B
    ref_result_dora = ref_mag_norm_scale * (scaling * ref_lora_out) + (ref_mag_norm_scale - 1) * base_result
    ref_loss = ref_result_dora.float().sum()
    ref_loss.backward()

    torch.testing.assert_close(mag_norm_scale, ref_mag_norm_scale, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result_dora, ref_result_dora, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lora_A_param.grad, ref_A.grad.T, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lora_B_param.grad, ref_B.grad.T, atol=1e-5, rtol=1e-5)


def test_module_fallback_extracts_parameters_not_module(monkeypatch):
    """Per-module fallback must pass parameters (not module) to gather_params_ctx.

    DeepSpeed GatheredParameters silently no-ops when given an nn.Module
    directly (it's not Iterable, has no ds_id).  The fallback must extract
    .parameters() and pass those as a tuple.
    """
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")

    base = nn.Linear(8, 4, bias=False)
    lora_A = nn.Linear(8, 2, bias=False)

    gathered_args = []

    @contextmanager
    def _tracking_gather(params):
        gathered_args.append(params)
        if isinstance(params, tuple) and len(params) > 1:
            # Fail combined tuple to force per-module fallback
            raise TypeError("force fallback")
        yield

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _tracking_gather)
    dora_mod._invalidate_fused_cache()

    with dora_mod._maybe_gather_base_params_ctx(base, lora_A):
        pass

    # After the combined tuple fails, the fallback should pass parameter
    # tuples, never raw module objects.
    fallback_calls = gathered_args[1:]  # skip the initial combined tuple
    for arg in fallback_calls:
        assert isinstance(arg, tuple), f"Expected tuple of params, got {type(arg)}"
        for p in arg:
            assert isinstance(p, (torch.Tensor, nn.Parameter)), f"Expected Parameter in tuple, got {type(p).__name__}"


# ---------------------------------------------------------------------------
# Partial-gather warning path
# ---------------------------------------------------------------------------


def test_partial_gather_raises_on_mixed_success(monkeypatch):
    """When some modules gather successfully but others fail, raise RuntimeError.

    This exercises the per-module fallback path in _maybe_gather_base_params_ctx
    where the param-tuple path fails (forced via TypeError) and the fallback
    iterates modules individually — succeeding for some, failing for others.
    Partial gather silently mixes fully gathered and sharded tensors, so it
    must be a hard error (not just a warning).
    """
    base = nn.Linear(8, 4, bias=False)
    lora_A = nn.Linear(8, 2, bias=False)
    lora_B = nn.Linear(2, 4, bias=False)

    # Initialize layer BEFORE installing the selective mock — update_layer
    # also uses _maybe_gather_base_params_ctx and would trigger the error.
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=1.0)

    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")
    base_param_ids = {id(p) for p in base.parameters()}

    @contextmanager
    def _selective_gather(params):
        """Fail for combined tuple (forces per-module fallback), succeed for
        base_layer params, fail for LoRA params — triggers partial-gather."""
        if isinstance(params, tuple) and len(params) > 1:
            # Combined param-tuple path — fail to force per-module fallback
            raise TypeError("tuple path deliberately broken")
        # Per-module fallback now passes tuple(mod.parameters())
        param_ids = {id(p) for p in (params if isinstance(params, tuple) else [params])}
        if param_ids & base_param_ids:
            yield
        else:
            raise TypeError("simulated gather failure for LoRA module")

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _selective_gather)
    dora_mod._invalidate_fused_cache()

    x = torch.randn(2, 8)
    with pytest.raises(RuntimeError, match="partial gather"):
        layer(x, lora_A=lora_A, lora_B=lora_B, scaling=1.0, base_layer=base)


def test_partial_gather_env_var_downgrades_to_warning(monkeypatch):
    """PEFT_DORA_ALLOW_PARTIAL_GATHER=1 downgrades partial-gather to a warning."""
    import logging

    base = nn.Linear(8, 4, bias=False)
    lora_A = nn.Linear(8, 2, bias=False)
    lora_B = nn.Linear(2, 4, bias=False)

    # Initialize layer BEFORE installing the selective mock.
    layer = DoraLinearLayer(fan_in_fan_out=False)
    layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=1.0)

    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")
    monkeypatch.setenv("PEFT_DORA_ALLOW_PARTIAL_GATHER", "1")
    base_param_ids = {id(p) for p in base.parameters()}

    @contextmanager
    def _selective_gather(params):
        if isinstance(params, tuple) and len(params) > 1:
            raise TypeError("tuple path deliberately broken")
        param_ids = {id(p) for p in (params if isinstance(params, tuple) else [params])}
        if param_ids & base_param_ids:
            yield
        else:
            raise TypeError("simulated gather failure for LoRA module")

    monkeypatch.setattr(dora_mod, "gather_params_ctx", _selective_gather)
    dora_mod._invalidate_fused_cache()

    logger = logging.getLogger("peft.tuners.lora.dora")
    logger.setLevel(logging.WARNING)
    with _mock_patch.object(logger, "warning") as mock_warn:
        x = torch.randn(2, 8)
        # Should NOT raise, should warn instead
        layer(x, lora_A=lora_A, lora_B=lora_B, scaling=1.0, base_layer=base)

        warning_calls = [str(c) for c in mock_warn.call_args_list]
        assert any("partial gather" in w.lower() for w in warning_calls), (
            f"Expected partial-gather warning, got: {warning_calls}"
        )


# ---------------------------------------------------------------------------
# Embedding composition equivalence (P0.1 from review round 9)
# ---------------------------------------------------------------------------


def test_embedding_composition_matches_dora_paper_eq5():
    """Verify _compose_with_dispatch gives the correct DoRA formula for embeddings.

    The upstream reference (docs/dora.reference_hf_peft.py:126) computed
    ``m * s * lora_out`` without the ``(m-1)*base`` correction term.
    Our unified compose path computes ``m*s*lora + (m-1)*base``, which when
    added to ``base`` by the variant gives ``m*(base + s*lora)`` — the correct
    DoRA Eq. 5 result.

    This test verifies the algebraic identity explicitly:
        base + compose(lora, base, m, s) == m * (base + s * lora)
    """
    torch.manual_seed(9001)
    batch, seq, dim = 2, 5, 16

    base_result = torch.randn(batch, seq, dim)
    lora_out = torch.randn(batch, seq, dim)
    # mag_norm_scale = m / ||W+sBA||_c, typically close to 1
    mag_norm_scale = 0.8 + 0.4 * torch.rand(1, dim)
    scale = 0.7

    # Compute expected BEFORE compose (which may mutate lora_out in-place)
    expected = mag_norm_scale * (base_result + scale * lora_out)

    layer = DoraLinearLayer(fan_in_fan_out=False)
    # Use eager path (no grad, no fused) via dispatch
    composed = layer._compose_with_dispatch(
        lora_out=lora_out,
        base_result=base_result,
        mag_norm_scale=mag_norm_scale,
        scale=scale,
    )
    # The variant adds composed to base_result
    final = base_result + composed
    # Expected per DoRA Eq. 5: m * (base + s * lora)
    torch.testing.assert_close(final, expected, atol=1e-6, rtol=1e-6)


def test_embedding_forward_composition_matches_dense_formula():
    """End-to-end: embedding DoRA forward matches the dense reference formula.

    Computes the full DoRA embedding output and verifies it equals
    ``m/||W+sBA||_c * (W+sBA)[x]`` per the paper.
    """
    torch.manual_seed(9002)
    num_embeddings, embedding_dim, rank = 32, 16, 4
    scaling = 0.5

    base = nn.Embedding(num_embeddings, embedding_dim)
    lora_A_param = nn.Parameter(torch.randn(rank, num_embeddings))
    lora_B_param = nn.Parameter(torch.randn(embedding_dim, rank))

    layer = DoraEmbeddingLayer(fan_in_fan_out=True)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A_param,
        lora_B=lora_B_param,
        scaling=scaling,
    )

    x = torch.randint(0, num_embeddings, (2, 5))
    base_result = F.embedding(x, base.weight)
    mag_norm_scale, result_dora = layer(
        x,
        lora_A=lora_A_param.T,
        lora_B=lora_B_param.T,
        scaling=scaling,
        base_layer=base,
        embed_fn=F.embedding,
        base_result=base_result,
    )

    # Compute reference: m/||W+sBA||_c * (W+sBA)[x]
    with torch.no_grad():
        W = base.weight.clone()  # [num_embeddings, embedding_dim]
        # lora_A_param: [rank, num_embeddings], lora_B_param: [embedding_dim, rank]
        # In forward: lora_A_fwd = lora_A_param.T [num_embeddings, rank]
        #             lora_B_fwd = lora_B_param.T [rank, embedding_dim]
        # lora_weight = (lora_A_fwd @ lora_B_fwd).T  [embedding_dim, num_embeddings]
        lora_weight = (lora_A_param.T @ lora_B_param.T).T  # [embedding_dim, num_embeddings]
        # get_weight_norm transposes W (fan_in_fan_out=True):
        # W.T [embedding_dim, num_embeddings] + scaling * lora_weight [embedding_dim, num_embeddings]
        W_t = transpose(W, fan_in_fan_out=True)  # [embedding_dim, num_embeddings]
        total = W_t + scaling * lora_weight  # [embedding_dim, num_embeddings]
        weight_norm = torch.linalg.vector_norm(total, dim=1)  # [embedding_dim]
        eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
        weight_norm = weight_norm.clamp_min(eps)
        m = layer.weight.detach()
        mag_norm_ref = m / weight_norm
        # DoRA output = mag_norm * (W + sBA)[x]
        # total.T = W + sBA in [num_embeddings, embedding_dim] space
        W_adapted = total.T  # [num_embeddings, embedding_dim]
        full_output = mag_norm_ref.view(1, -1) * F.embedding(x, W_adapted)

    # Our output: base + composed = base + result_dora
    our_output = base_result + result_dora
    torch.testing.assert_close(our_output, full_output, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Conv base_result=None gradient test (P3.10 from review round 9)
# ---------------------------------------------------------------------------


def test_conv_base_result_none_gradient_flow():
    """Conv base_result=None with nonzero scaling must propagate gradients.

    The existing snapshot test uses scaling=0.0 which zeroes out LoRA.
    This test verifies gradient flow through the conv base_result=None path
    with active LoRA contribution.
    """
    torch.manual_seed(9003)
    base = nn.Conv2d(4, 8, 3, padding=1, bias=False)
    lora_A = nn.Conv2d(4, 2, 3, padding=1, bias=False)
    lora_B = nn.Conv2d(2, 8, 1, bias=False)
    scaling = 0.5

    layer = DoraConv2dLayer(fan_in_fan_out=False)
    layer.update_layer(
        base_layer=base,
        lora_A=lora_A.weight,
        lora_B=lora_B.weight,
        scaling=scaling,
    )
    # Ensure magnitude requires grad (DoRA trains it)
    layer.weight.requires_grad_(True)
    lora_A.weight.requires_grad_(True)
    lora_B.weight.requires_grad_(True)

    x = torch.randn(1, 4, 6, 6, requires_grad=True)
    out = layer(
        x,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=scaling,
        base_layer=base,
        base_result=None,
    )
    loss = out.float().sum()
    loss.backward()

    # All trainable params should have gradients
    assert lora_A.weight.grad is not None, "lora_A should have gradients"
    assert lora_B.weight.grad is not None, "lora_B should have gradients"
    assert layer.weight.grad is not None, "magnitude should have gradients"
    assert x.grad is not None, "input should have gradients"
    # Gradients should be finite
    assert torch.all(torch.isfinite(lora_A.weight.grad)), "lora_A grad not finite"
    assert torch.all(torch.isfinite(lora_B.weight.grad)), "lora_B grad not finite"
    assert torch.all(torch.isfinite(layer.weight.grad)), "magnitude grad not finite"
    assert torch.all(torch.isfinite(x.grad)), "input grad not finite"
    # Gradients should be nonzero (scaling > 0, so LoRA contributes)
    assert lora_A.weight.grad.abs().sum() > 0, "lora_A grad is all zeros"
    assert lora_B.weight.grad.abs().sum() > 0, "lora_B grad is all zeros"


# ---------------------------------------------------------------------------
# FSDP2 positive detection
# ---------------------------------------------------------------------------


def test_is_fsdp2_managed_positive_detection(monkeypatch):
    """_is_fsdp2_managed returns True when a module has FSDPState attached.

    Mocks the internal torch.distributed APIs to simulate a module that was
    wrapped with fully_shard (FSDP2).  Since _is_fsdp2_managed uses local
    imports, injecting mock modules into sys.modules is sufficient.
    """
    import sys
    from unittest.mock import MagicMock

    module = nn.Linear(4, 4)

    # Create a mock FSDPState class and instance
    class _MockFSDPState:
        pass

    mock_state = _MockFSDPState()

    # Mock the primary detection path modules
    mock_fsdp_module = MagicMock()
    mock_fsdp_module.FSDPState = _MockFSDPState
    mock_composable_state = MagicMock()
    mock_composable_state._get_module_state = lambda m: mock_state

    monkeypatch.setitem(sys.modules, "torch.distributed._composable.fsdp", mock_fsdp_module)
    monkeypatch.setitem(sys.modules, "torch.distributed._composable_state", mock_composable_state)

    # Explicitly reset the cached detection functions so that our mocked
    # sys.modules entries are picked up.  Without this, a stale cache from
    # a prior test could cause this test to silently pass or fail depending
    # on execution order (the autouse _reset_threshold_cache fixture also
    # does this, but we should not depend on it).
    dora_mod._invalidate_fused_cache()
    result = dora_mod._is_fsdp2_managed(module)
    assert result is True, f"Expected True for FSDP2-managed module, got {result}"


def test_is_fsdp2_managed_new_import_path(monkeypatch):
    """_is_fsdp2_managed detects FSDP2 when FSDPState is in the 2.10+ location.

    On PyTorch 2.10+, FSDPState moved from torch.distributed._composable.fsdp
    to torch.distributed.fsdp._fully_shard._fsdp_state.  The old import fails,
    so _resolve_fsdp2_detect_fns must try the new path.
    """
    import sys
    from unittest.mock import MagicMock

    module = nn.Linear(4, 4)

    class _MockFSDPState:
        pass

    mock_state = _MockFSDPState()

    # Simulate 2.10+: old path has NO FSDPState attribute
    mock_old_fsdp = MagicMock(spec=[])  # empty spec = no attributes
    del mock_old_fsdp.FSDPState  # ensure AttributeError on access

    # New path has FSDPState
    mock_new_fsdp_state = MagicMock()
    mock_new_fsdp_state.FSDPState = _MockFSDPState

    # _get_module_state returns our mock state
    mock_composable_state = MagicMock()
    mock_composable_state._get_module_state = lambda m: mock_state

    monkeypatch.setitem(sys.modules, "torch.distributed._composable.fsdp", mock_old_fsdp)
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp._fully_shard._fsdp_state", mock_new_fsdp_state)
    monkeypatch.setitem(sys.modules, "torch.distributed._composable_state", mock_composable_state)

    dora_mod._invalidate_fused_cache()
    result = dora_mod._is_fsdp2_managed(module)
    assert result is True, f"Expected True for FSDP2 via new import path, got {result}"


def test_is_fsdp2_managed_rejects_raw_tensors():
    """_is_fsdp2_managed returns False for raw tensors, not just nn.Module."""
    dora_mod._invalidate_fused_cache()
    assert dora_mod._is_fsdp2_managed(torch.randn(4, 4)) is False
    # Reset between assertions to prevent stale cached detection functions
    dora_mod._invalidate_fused_cache()
    assert dora_mod._is_fsdp2_managed(nn.Parameter(torch.randn(4, 4))) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VRAM measurement")
def test_chunked_norm_no_gemm_temporary_spike(monkeypatch):
    """Verify that _get_weight_norm_linear uses in-place GEMM accumulation
    (addmm_) so peak VRAM stays within the memory budget and doesn't spike
    from materializing large matmul temporaries."""
    # Force small chunk size to ensure chunked path is exercised
    monkeypatch.setenv("PEFT_DORA_NORM_CHUNK_MB", "4")
    _invalidate_fused_cache()

    torch.manual_seed(42)
    device = torch.device("cuda")
    out_features, in_features, rank = 4096, 4096, 256
    dtype = torch.bfloat16

    base_weight = torch.randn((out_features, in_features), device=device, dtype=dtype)
    lora_A = torch.randn((rank, in_features), device=device, dtype=dtype)
    lora_B = torch.randn((out_features, rank), device=device, dtype=dtype)

    layer = DoraLinearLayer(fan_in_fan_out=False).to(device)

    # Warmup call: forces lazy CUDA/cuBLAS workspace allocation so the
    # measured call only reflects the function's own tensor allocations.
    with torch.no_grad():
        _ = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=1.0,
        )
    del _
    torch.cuda.synchronize()

    # Baseline: memory after warmup, before the measured call
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        norm = layer._get_weight_norm_linear(
            base_weight=base_weight,
            lora_A_w=lora_A,
            lora_B_w=lora_B,
            scaling=1.0,
        )

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)

    # All fp32 (4 bytes/element) allocations inside _get_weight_norm_linear:
    #   Chunk loop:  w_norm_sq (out), U (out*rank), gram (rank*rank),
    #                W_chunk/A_chunk casts (small with tiny chunk_size)
    #   Post-loop:   B_comp (out*rank), cross_term (out),
    #                B_comp*U temp (out*rank), BA (out*rank),
    #                BA*B_comp temp (out*rank), ba_norm_sq (out),
    #                norm_sq/weight_norm (out)
    # Not all are live simultaneously — peak is ~5 (out*rank) sized buffers.
    # With addmm_ the chunk loop adds NO extra (out*rank) temporary.
    element_size = 4  # fp32 compute dtype
    or_bytes = out_features * rank * element_size  # one (out, rank) buffer
    expected_bytes = (
        out_features * element_size * 4  # w_norm_sq, cross_term, ba_norm_sq, norm_sq
        + or_bytes * 5  # U, B_comp, B_comp*U temp, BA, BA*B_comp temp
        + rank * rank * element_size  # gram
    )
    # 2× headroom for CUDA allocator block rounding and fragmentation
    budget = mem_before + expected_bytes * 2

    assert peak <= budget, (
        f"Peak VRAM {peak / 1e6:.1f} MB exceeded budget {budget / 1e6:.1f} MB — "
        f"likely a GEMM temporary was not eliminated by addmm_"
    )
    assert torch.all(torch.isfinite(norm))


# ---------------------------------------------------------------------------
# Public helper function tests
# ---------------------------------------------------------------------------


class TestSetDoRANormThreshold:
    """Tests for set_dora_norm_threshold_mb bounds checking."""

    def test_rejects_non_int(self):
        with pytest.raises(ValueError, match="must be an integer"):
            set_dora_norm_threshold_mb(256.0)

    def test_rejects_string(self):
        with pytest.raises(ValueError, match="must be an integer"):
            set_dora_norm_threshold_mb("256")  # type: ignore[arg-type]

    def test_rejects_below_minimum(self):
        with pytest.raises(ValueError, match="must be between"):
            set_dora_norm_threshold_mb(8)

    def test_rejects_above_maximum(self):
        with pytest.raises(ValueError, match="must be between"):
            set_dora_norm_threshold_mb(100_000)

    def test_accepts_minimum(self, monkeypatch):
        monkeypatch.delenv("PEFT_DORA_NORM_CHUNK_MB", raising=False)
        set_dora_norm_threshold_mb(16)
        assert os.environ["PEFT_DORA_NORM_CHUNK_MB"] == "16"

    def test_accepts_maximum(self, monkeypatch):
        monkeypatch.delenv("PEFT_DORA_NORM_CHUNK_MB", raising=False)
        set_dora_norm_threshold_mb(65536)
        assert os.environ["PEFT_DORA_NORM_CHUNK_MB"] == "65536"

    def test_accepts_typical_value(self, monkeypatch):
        monkeypatch.delenv("PEFT_DORA_NORM_CHUNK_MB", raising=False)
        set_dora_norm_threshold_mb(512)
        assert os.environ["PEFT_DORA_NORM_CHUNK_MB"] == "512"

    def test_invalidates_cache(self, monkeypatch):
        """Setting threshold must invalidate the cached value so the next
        call to get_dora_norm_threshold_mb reads the new value."""
        monkeypatch.delenv("PEFT_DORA_NORM_CHUNK_MB", raising=False)
        set_dora_norm_threshold_mb(128)
        assert get_dora_norm_threshold_mb() == 128
        set_dora_norm_threshold_mb(64)
        assert get_dora_norm_threshold_mb() == 64


class TestGetDoRANormThreshold:
    """Tests for get_dora_norm_threshold_mb and get_dora_norm_threshold_bytes."""

    def test_returns_int(self):
        assert isinstance(get_dora_norm_threshold_mb(), int)
        assert isinstance(get_dora_norm_threshold_bytes(), int)

    def test_byte_consistency(self):
        """Bytes value must equal MB value * 1024 * 1024."""
        mb = get_dora_norm_threshold_mb()
        assert get_dora_norm_threshold_bytes() == mb * 1024 * 1024

    def test_default_value(self, monkeypatch):
        """Default is 256 MB when env var is unset."""
        monkeypatch.delenv("PEFT_DORA_NORM_CHUNK_MB", raising=False)
        _invalidate_threshold_cache()
        assert get_dora_norm_threshold_mb() == 256


class TestGetDoraFused:
    """Tests for _get_dora_fused lazy-import via sys.modules."""

    def test_returns_module(self):
        mod = _get_dora_fused()
        assert mod is not None
        assert hasattr(mod, "fused_dora_compose")
        assert hasattr(mod, "fused_dora_compose_autograd")

    def test_populates_sys_modules(self):
        mod = _get_dora_fused()
        assert _DORA_FUSED_MODULE_NAME in sys.modules
        assert sys.modules[_DORA_FUSED_MODULE_NAME] is mod

    def test_idempotent(self):
        """Repeated calls must return the same module object."""
        mod1 = _get_dora_fused()
        mod2 = _get_dora_fused()
        assert mod1 is mod2

    def test_concurrent_first_call(self):
        """Verify thread-safety: concurrent calls should all get the same module."""
        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_get_dora_fused) for _ in range(8)]
            results = [f.result() for f in futures]
        assert all(r is results[0] for r in results)


class TestDtypeElementSize:
    """Tests for _dtype_element_size, including the fallback path for non-float dtypes."""

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (torch.float32, 4),
            (torch.float64, 8),
            (torch.float16, 2),
            (torch.bfloat16, 2),
        ],
    )
    def test_cached_dtypes(self, dtype, expected):
        assert _dtype_element_size(dtype) == expected

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (torch.int8, 1),
            (torch.uint8, 1),
            (torch.int16, 2),
            (torch.int32, 4),
            (torch.int64, 8),
        ],
    )
    def test_fallback_dtypes(self, dtype, expected):
        """Non-float dtypes must fall through to the torch.tensor fallback."""
        assert _dtype_element_size(dtype) == expected

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (torch.complex64, 8),
            (torch.complex128, 16),
        ],
    )
    def test_complex_dtypes_via_fallback(self, dtype, expected):
        """Complex dtypes are not in the fast-path dict but work via fallback."""
        assert _dtype_element_size(dtype) == expected


class TestMagBroadcastsLastDim:
    """Tests for _mag_broadcasts_last_dim shape validation."""

    def test_valid_1d_mag(self):
        mag = torch.randn(16)
        out = torch.randn(4, 16)
        assert _mag_broadcasts_last_dim(mag, out) is True

    def test_valid_2d_mag(self):
        mag = torch.randn(1, 16)
        out = torch.randn(4, 16)
        assert _mag_broadcasts_last_dim(mag, out) is True

    def test_scalar_mag_rejected(self):
        mag = torch.tensor(1.0)
        out = torch.randn(4, 16)
        assert _mag_broadcasts_last_dim(mag, out) is False

    def test_scalar_out_rejected(self):
        mag = torch.randn(16)
        out = torch.tensor(1.0)
        assert _mag_broadcasts_last_dim(mag, out) is False

    def test_last_dim_mismatch_rejected(self):
        mag = torch.randn(8)
        out = torch.randn(4, 16)
        assert _mag_broadcasts_last_dim(mag, out) is False

    def test_degenerate_shape_rejected(self):
        """mag=[F] applied to out=[B, F, 1] where numel()==F but last dim is 1.

        This is the degenerate case the function was designed to reject:
        mag has the right number of elements but the last dim doesn't match.
        """
        F = 16
        mag = torch.randn(F)
        out = torch.randn(4, F, 1)
        assert _mag_broadcasts_last_dim(mag, out) is False

    def test_3d_broadcast_valid(self):
        mag = torch.randn(1, 1, 32)
        out = torch.randn(2, 4, 32)
        assert _mag_broadcasts_last_dim(mag, out) is True


class TestShouldAutoUseFusedBackwardShape:
    """Tests for _should_auto_use_fused_backward_shape boundary conditions."""

    def test_below_min_cols_rejected(self):
        # 2047 cols < 2048 threshold
        assert _should_auto_use_fused_backward_shape(8192, 2047) is False

    def test_exactly_min_cols_accepted_if_enough_work(self):
        # 2048 cols, need rows * cols >= 2048 * 6144
        # 6144 rows * 2048 cols = 12_582_912 == threshold
        assert _should_auto_use_fused_backward_shape(6144, 2048) is True

    def test_just_below_work_threshold_rejected(self):
        # 2048 cols, 6143 rows → 12_580_864 < 12_582_912
        assert _should_auto_use_fused_backward_shape(6143, 2048) is False

    def test_zero_rows_rejected(self):
        assert _should_auto_use_fused_backward_shape(0, 4096) is False

    def test_negative_rows_rejected(self):
        assert _should_auto_use_fused_backward_shape(-1, 4096) is False

    def test_large_shape_accepted(self):
        assert _should_auto_use_fused_backward_shape(8192, 4096) is True

    def test_large_cols_small_rows_accepted(self):
        # 1 row * 16384 cols = 16384 < threshold → rejected despite large cols
        assert _should_auto_use_fused_backward_shape(1, 16384) is False

    def test_many_rows_at_min_cols(self):
        # 65536 rows * 2048 cols = 134_217_728 >> threshold
        assert _should_auto_use_fused_backward_shape(65536, 2048) is True


class TestSnapshotDequantizedWeight:
    """Tests for _snapshot_dequantized_weight data_ptr cloning logic."""

    def test_shared_storage_cloned(self):
        """When weight shares data_ptr with module.weight, it must be cloned."""
        module = nn.Linear(8, 4)
        weight = module.weight  # same object → same data_ptr
        result = _snapshot_dequantized_weight(module, weight)
        assert result.data_ptr() != module.weight.data_ptr()
        assert torch.allclose(result, module.weight)

    def test_view_sharing_storage_cloned(self):
        """A view that shares storage (data_ptr match, different object) must be cloned."""
        module = nn.Linear(8, 4)
        # Create a view that shares the underlying storage
        weight_view = module.weight.view(4, 8)
        assert weight_view.data_ptr() == module.weight.data_ptr()
        assert weight_view is not module.weight
        result = _snapshot_dequantized_weight(module, weight_view)
        assert result.data_ptr() != module.weight.data_ptr()

    def test_independent_tensor_not_cloned(self):
        """A new tensor (different data_ptr) should be returned as-is."""
        module = nn.Linear(8, 4)
        weight = module.weight.detach().clone()  # different data_ptr
        result = _snapshot_dequantized_weight(module, weight)
        assert result is weight  # no clone, same object returned

    def test_module_without_weight(self):
        """Module without .weight attribute should return tensor as-is."""
        module = nn.Module()
        weight = torch.randn(4, 8)
        result = _snapshot_dequantized_weight(module, weight)
        assert result is weight

    def test_offset_view_shares_storage_but_different_data_ptr(self):
        """A non-zero-offset view shares storage but has a different data_ptr.

        _snapshot_dequantized_weight uses data_ptr comparison, so an offset view
        will NOT be detected as aliasing module.weight.  This documents the known
        limitation: in practice dequantize_module_weight never returns an offset
        view, so data_ptr comparison is sufficient for real workloads.
        """
        module = nn.Linear(8, 4)
        weight_offset = module.weight.flatten()[4:]
        assert weight_offset.data_ptr() != module.weight.data_ptr()
        assert weight_offset.storage().data_ptr() == module.weight.storage().data_ptr()
        result = _snapshot_dequantized_weight(module, weight_offset)
        # Not cloned — data_ptr differs, so the function treats it as independent.
        # This is acceptable: the only callers pass dequantize_module_weight output,
        # which returns either module.weight itself or a freshly-allocated tensor.
        assert result is weight_offset


class TestConvGroupedNorm:
    """Tests for grouped Conv1d and Conv3d norm computation (parity with Conv2d)."""

    def test_conv1d_groups(self):
        torch.manual_seed(170)
        groups = 2
        in_ch, out_ch, rank = 6, 8, 4
        base = nn.Conv1d(in_ch, out_ch, 3, padding=1, groups=groups, bias=False)
        lora_A = nn.Conv1d(in_ch, rank, 3, padding=1, groups=groups, bias=False)
        lora_B = nn.Conv1d(rank, out_ch, 1, groups=groups, bias=False)
        layer = DoraConv1dLayer(fan_in_fan_out=False)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.3)

        x = torch.randn(1, in_ch, 20)
        base_result = base(x).detach()
        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.3,
            base_layer=base,
            base_result=base_result,
        )
        with torch.no_grad():
            weight_norm = layer._get_weight_norm_conv_factored(
                base_weight=base.weight,
                lora_A_w=lora_A.weight,
                lora_B_w=lora_B.weight,
                scaling=0.3,
            )
        mag_norm_scale = layer.weight / weight_norm
        ref = (mag_norm_scale - 1) * base_result + mag_norm_scale * (0.3 * lora_B(lora_A(x)))
        assert _max_diff(out, ref) <= 1e-5

    def test_conv3d_groups(self):
        torch.manual_seed(171)
        groups = 2
        in_ch, out_ch, rank = 4, 6, 2
        base = nn.Conv3d(in_ch, out_ch, 3, padding=1, groups=groups, bias=False)
        lora_A = nn.Conv3d(in_ch, rank, 3, padding=1, groups=groups, bias=False)
        lora_B = nn.Conv3d(rank, out_ch, 1, groups=groups, bias=False)
        layer = DoraConv3dLayer(fan_in_fan_out=False)
        layer.update_layer(base_layer=base, lora_A=lora_A.weight, lora_B=lora_B.weight, scaling=0.5)

        x = torch.randn(1, in_ch, 4, 4, 4)
        base_result = base(x).detach()
        out = layer(
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
            base_layer=base,
            base_result=base_result,
        )
        with torch.no_grad():
            weight_norm = layer._get_weight_norm_conv_factored(
                base_weight=base.weight,
                lora_A_w=lora_A.weight,
                lora_B_w=lora_B.weight,
                scaling=0.5,
            )
        mag_norm_scale = layer.weight / weight_norm
        ref = (mag_norm_scale - 1) * base_result + mag_norm_scale * (0.5 * lora_B(lora_A(x)))
        assert _max_diff(out, ref) <= 1e-5


class TestComposeWithBaseChunksZeroOutput:
    """Test that _compose_with_base_chunks handles zero-output-features gracefully."""

    def test_zero_output_features_early_return(self):
        """When out_features==0, the chunk path should return immediately."""
        layer = DoraLinearLayer(fan_in_fan_out=False)
        # Create a zero-output linear layer
        base = nn.Linear(4, 0, bias=False)
        rank = 2
        lora_A_w = torch.randn(rank, 4)
        lora_B_w = torch.randn(0, rank)
        layer.update_layer(base_layer=base, lora_A=lora_A_w, lora_B=lora_B_w, scaling=1.0)

        x = torch.randn(2, 4)
        lora_result = torch.empty(2, 0)
        base_weight_t = torch.empty(0, 4)
        mag_norm_scale = torch.empty(0)

        # Should not raise
        layer._compose_with_base_chunks(
            x=x,
            lora_result=lora_result,
            base_weight_t=base_weight_t,
            mag_norm_scale=mag_norm_scale,
            scale=1.0,
        )
        assert layer._last_forward_chunk_size == 0


@pytest.mark.parametrize(
    ("lora_dtype", "base_dtype", "mag_dtype"),
    [
        (torch.float16, torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float16, torch.float32),
        (torch.float32, torch.float32, torch.float16),
    ],
)
def test_promoted_compose_dtype_matches_nested_torch_promotion(lora_dtype, base_dtype, mag_dtype):
    """The helper should mirror PyTorch's pairwise promotion contract exactly."""
    expected = torch.promote_types(torch.promote_types(lora_dtype, base_dtype), mag_dtype)
    assert dora_mod._promoted_compose_dtype(lora_dtype, base_dtype, mag_dtype) == expected


def test_refresh_embedding_lora_view_rebuilds_transposed_parameter_views():
    """Parameter-backed embedding transposes should be rebuilt from the base Parameter."""
    param = nn.Parameter(torch.randn(7, 5))
    view = param.T

    refreshed = dora_mod._refresh_embedding_lora_view(view)

    assert torch.equal(refreshed, view)
    assert refreshed is not view
    assert refreshed._base is param

    # Detached tensors (no ._base chain to a Parameter) should pass through.
    detached = view.clone()
    assert dora_mod._refresh_embedding_lora_view(detached) is detached


def test_gather_total_failure_logs_warning(monkeypatch, caplog):
    """When all gather attempts fail, a warning should be logged."""
    import logging

    monkeypatch.setattr(dora_mod, "_is_zero3_active", lambda: True)
    monkeypatch.setattr(dora_mod, "gather_params_ctx", lambda params: (_ for _ in ()).throw(TypeError("mock fail")))
    monkeypatch.setenv("PEFT_FORCE_GATHER", "1")

    module = nn.Linear(4, 4)
    with caplog.at_level(logging.WARNING, logger="peft.tuners.lora.dora"):
        with dora_mod._maybe_gather_base_params_ctx(module):
            pass

    assert any("gather failed for all" in msg.lower() for msg in caplog.messages)


def test_disable_autocast_fallback_on_unsupported_device(monkeypatch):
    """When autocast raises on construction, _disable_autocast should fall back to bare yield."""
    from peft.tuners.lora.dora import _disable_autocast

    # Mock autocast to raise TypeError (simulating unsupported device_type)
    def _broken_autocast(**kwargs):
        raise TypeError("unsupported device type")

    monkeypatch.setattr(dora_mod, "autocast", _broken_autocast)

    # Should not raise — falls through to bare yield
    with _disable_autocast("xpu"):
        result = torch.tensor(1.0) + torch.tensor(2.0)
    assert result.item() == 3.0


def test_compose_with_base_chunks_empty_batch():
    """Empty batch (prefix_rows==0, out_features>0) should execute without error."""
    layer = DoraLinearLayer(fan_in_fan_out=False)
    base = nn.Linear(4, 8, bias=False)
    rank = 2
    lora_A_w = torch.randn(rank, 4)
    lora_B_w = torch.randn(8, rank)
    layer.update_layer(base_layer=base, lora_A=lora_A_w, lora_B=lora_B_w, scaling=1.0)

    # Empty batch: 0 rows, 4 input features → lora_result has 0 rows, 8 out features
    x = torch.empty(0, 4)
    lora_result = torch.empty(0, 8)
    base_weight_t = base.weight.detach()
    mag_norm_scale = torch.ones(8)

    layer._compose_with_base_chunks(
        x=x,
        lora_result=lora_result,
        base_weight_t=base_weight_t,
        mag_norm_scale=mag_norm_scale,
        scale=1.0,
    )
    # prefix_rows == 0, so chunk_size should be set to out_features
    assert layer._last_forward_chunk_size == 8
