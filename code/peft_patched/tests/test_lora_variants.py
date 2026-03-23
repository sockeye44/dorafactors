# Copyright 2025-present the HuggingFace Inc. team.
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

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Only clear peft imports when explicitly requested, matching test_dora_math.py.
# Unconditional clearing can cause import-order-dependent failures in CI.
if os.environ.get("PEFT_TEST_ISOLATED_IMPORTS") == "1":
    for _name in [name for name in list(sys.modules) if name.startswith("peft")]:
        sys.modules.pop(_name)

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.dora import DoraLinearLayer
from peft.tuners.lora.layer import Conv1d as LoraConv1d
from peft.tuners.lora.layer import Conv2d as LoraConv2d
from peft.tuners.lora.layer import Embedding as LoraEmbedding
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.variants import (
    ALoraLinearVariant,
    DoraConv1dVariant,
    DoraConv2dVariant,
    DoraEmbeddingVariant,
    DoraLinearVariant,
    calculate_alora_offsets,
    get_alora_offsets_for_forward,
    get_alora_offsets_for_generate,
)
from peft.utils.other import transpose


class _DummyLinearMergeMagnitude:
    def __init__(self, weight, fan_in_fan_out):
        self.weight = weight
        self.fan_in_fan_out = fan_in_fan_out

    def get_weight_norm(self, orig_weight, lora_weight, scaling):
        total = transpose(orig_weight, self.fan_in_fan_out) + scaling * lora_weight
        return torch.linalg.vector_norm(total, dim=1)


class _DummyEmbeddingMergeMagnitude:
    def __init__(self, weight):
        self.weight = weight

    def get_weight_norm(self, orig_weight, lora_weight, scaling):
        total = orig_weight.T + scaling * lora_weight
        return torch.linalg.vector_norm(total, dim=1)


class _DummyVariantMergeModule:
    def __init__(self, delta_weight, magnitude_layer, fan_in_fan_out=False):
        self._delta_weight = delta_weight
        self._cache = {}
        self.fan_in_fan_out = fan_in_fan_out
        self.lora_magnitude_vector = {"default": magnitude_layer}

    def get_delta_weight(self, active_adapter):
        assert active_adapter == "default"
        return self._delta_weight

    def _cache_store(self, key, value):
        self._cache[key] = value

    def _cache_pop(self, key):
        return self._cache.pop(key)


class _DummyForwardMagnitude:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, x, **kwargs):
        return self.delta


class _DummyForwardModule:
    def __init__(self, base_layer, delta, *, training=False, dropout=None):
        self.lora_A = {"default": object()}
        self.lora_B = {"default": object()}
        self.lora_dropout = {"default": nn.Identity() if dropout is None else dropout}
        self.scaling = {"default": 1.0}
        self.training = training
        self._base_layer = base_layer
        self.lora_magnitude_vector = {"default": _DummyForwardMagnitude(delta)}

    def get_base_layer(self):
        return self._base_layer


class TestDoraLinearWeightNorm:
    @pytest.mark.parametrize("fan_in_fan_out", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_matches_dense_computation(self, fan_in_fan_out, dtype):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == torch.float16 and device.type == "cpu":
            pytest.skip("float16 matmul not supported on CPU")

        torch.manual_seed(0)
        out_features, in_features, rank = 13, 17, 4
        base_shape = (out_features, in_features)
        if fan_in_fan_out:
            base_shape = base_shape[::-1]

        base_weight = torch.randn(base_shape, device=device, dtype=dtype)
        lora_A = torch.randn(rank, in_features, device=device, dtype=dtype)
        lora_B = torch.randn(out_features, rank, device=device, dtype=dtype)
        scaling = 0.7

        layer = DoraLinearLayer(fan_in_fan_out=fan_in_fan_out)
        with torch.no_grad():
            computed = layer._get_weight_norm_linear(
                base_weight=base_weight,
                lora_A_w=lora_A,
                lora_B_w=lora_B,
                scaling=scaling,
            )

        # Dense reference in higher precision for numerical stability
        W = transpose(base_weight, fan_in_fan_out)
        W64 = W.to(torch.float64)
        dense = lora_B.to(torch.float64) @ lora_A.to(torch.float64)
        expected = torch.linalg.vector_norm(W64 + scaling * dense, dim=1).to(computed.dtype)

        assert computed.dtype == base_weight.dtype
        rtol = 5e-3 if dtype == torch.float16 else 1e-4
        atol = 5e-3 if dtype == torch.float16 else 1e-5
        assert torch.allclose(computed, expected, rtol=rtol, atol=atol)


# Custom model featuring embeddings and a 'visual stack'
class CustomModel(nn.Module):
    """pytorch module that contains common targetable layers (linear, embedding, conv, ...)"""

    def __init__(self, num_embeddings=100, embedding_dim=16, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.dummy_conv1d_output_dim = 32 * 10
        self.dummy_conv2d_output_dim = 16 * 10 * 10
        self.linear1 = nn.Linear(self.dummy_conv1d_output_dim + self.dummy_conv2d_output_dim, 64)
        self.linear2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, dummy_image_input):
        # Path 1: Embedding -> Conv1d
        x1 = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        x1 = x1.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        x1 = self.relu(self.conv1d(x1))  # (batch_size, 32, seq_len)
        x1_flat = self.flatten(x1)
        # Path 2: Conv2d -> Linear
        x2 = self.relu(self.conv2d(dummy_image_input))  # (batch_size, 16, H, W)
        x2_flat = self.flatten(x2)  # (batch_size, 16*H*W)
        # Combine or select paths if making a functional model.
        # For this test, we mainly care about layer types, so forward might not be fully executed.
        # Let's use x2_flat for subsequent linear layers.
        output = self.relu(self.linear1(torch.concat([x1_flat, x2_flat], dim=1)))
        output = self.linear2(output)
        return output


# Used for testing alora_offsets for aLoRA
class DummyLM(nn.Module):
    def __init__(self, vocab_size: int = 10, hidden_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, X=None, embeds=None, num_beams=None, alora_offsets=None):
        if X is not None:
            embeds = self.embed(X)
        return self.linear(embeds)


class MockTransformerWrapper:
    """Mock class to behave like a transformers model.

    This is needed because the tests initialize the model by calling transformers_class.from_pretrained.

    """

    @classmethod
    def from_pretrained(cls):
        # set the seed so that from_pretrained always returns the same model
        torch.manual_seed(0)

        torch_dtype = torch.float32

        return DummyLM().to(torch_dtype)


VARIANT_MAP = {
    "dora": {
        LoraLinear: DoraLinearVariant,
        LoraEmbedding: DoraEmbeddingVariant,
        LoraConv1d: DoraConv1dVariant,
        LoraConv2d: DoraConv2dVariant,
    },
    "alora": {
        LoraLinear: ALoraLinearVariant,
    },
}


TEST_CASES = [
    (
        "dora",
        LoraConfig,
        {"target_modules": ["linear1", "linear2", "conv1d", "conv2d", "embedding"], "use_dora": True},
    ),
    (
        "alora",
        LoraConfig,
        {"target_modules": ["linear1", "linear2"], "alora_invocation_tokens": [1]},
    ),
]


class TestLoraVariants:
    @pytest.mark.parametrize("variant_name, config_cls, config_kwargs", TEST_CASES)
    def test_variant_is_applied_to_layers(self, variant_name, config_cls, config_kwargs):
        # This test assumes that targeting and replacing layers works and that after `get_peft_model` we
        # have a model with LoRA layers. We just make sure that each LoRA layer has its variant set and
        # it is also the correct variant for that layer.
        base_model = CustomModel()
        peft_config = config_cls(**config_kwargs)
        peft_model = get_peft_model(base_model, peft_config)

        layer_type_map = VARIANT_MAP[variant_name]

        for _, module in peft_model.named_modules():
            if not hasattr(module, "lora_variant"):
                continue

            # Note that not every variant supports every layer. If it is not mapped it is deemed unsupported and
            # will not be tested.
            expected_variant_type = layer_type_map.get(type(module), None)
            if not expected_variant_type:
                continue

            assert isinstance(module.lora_variant["default"], expected_variant_type)

    def custom_model_with_loss_backpropagated(self, peft_config):
        """Returns the CustomModel + PEFT model instance with a dummy loss that was backpropagated once."""
        base_model = CustomModel()
        peft_model = get_peft_model(base_model, peft_config)

        x, y = torch.ones(10, 10).long(), torch.ones(10, 1, 10, 10)
        out = peft_model(x, y)
        loss = out.sum()
        loss.backward()

        return base_model, peft_model

    def test_dora_params_have_gradients(self):
        """Ensure that the parameters added by the DoRA variant are participating in the output computation."""
        layer_names = ["linear1", "linear2", "conv1d", "conv2d", "embedding"]
        peft_config = LoraConfig(target_modules=layer_names, use_dora=True)
        base_model, peft_model = self.custom_model_with_loss_backpropagated(peft_config)

        for layer in layer_names:
            assert getattr(peft_model.base_model.model, layer).lora_magnitude_vector["default"].weight.grad is not None




def test_dora_embedding_variant_forward_matches_scaled_sum_formula():
    torch.manual_seed(44)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = nn.Embedding(32, 12).to(device)
    module = LoraEmbedding(
        base,
        adapter_name="default",
        r=4,
        lora_alpha=8,
        use_dora=True,
    ).to(device)
    module.eval()

    x = torch.randint(0, 32, (3, 5), device=device)
    result = module.base_layer(x)
    result_clone = result.clone()

    updated = module.lora_variant["default"].forward(module, "default", x, result)

    embedding_A = module.lora_embedding_A["default"].T
    embedding_B = module.lora_embedding_B["default"].T
    scaling = module.scaling["default"]

    mag_norm_scale, dora_delta = module.lora_magnitude_vector["default"](
        x,
        lora_A=embedding_A,
        lora_B=embedding_B,
        scaling=scaling,
        base_layer=module.get_base_layer(),
        embed_fn=module._embed,
        base_result=result_clone,
    )
    lora_result = F.embedding(x, embedding_A) @ embedding_B

    expected_total = mag_norm_scale * (result_clone + scaling * lora_result)

    assert torch.allclose(updated, expected_total.to(updated.dtype), rtol=1e-5, atol=1e-5)
    assert torch.allclose(updated, (result_clone + dora_delta).to(updated.dtype), rtol=1e-5, atol=1e-5)


def test_dora_variant_forward_accumulates_in_place():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = nn.Linear(16, 12, bias=True).to(device)
    module = LoraLinear(
        base,
        adapter_name="default",
        r=4,
        lora_alpha=8,
        use_dora=True,
    ).to(device)
    module.eval()
    module.lora_dropout["default"] = nn.Identity()

    x = torch.randn(3, 16, device=device, requires_grad=True)
    result = module.base_layer(x)
    result_clone = result.clone()

    delta = module.lora_magnitude_vector["default"](
        x,
        lora_A=module.lora_A["default"],
        lora_B=module.lora_B["default"],
        scaling=module.scaling["default"],
        base_layer=module.get_base_layer(),
        base_result=result_clone.clone(),
    )

    result_ptr = result.data_ptr()
    updated = module.lora_variant["default"].forward(module, "default", x, result)

    assert updated.data_ptr() == result_ptr
    assert torch.allclose(updated, result_clone + delta.to(updated.dtype))

    loss = updated.sum()
    loss.backward()

    assert x.grad is not None and torch.all(torch.isfinite(x.grad))


def test_dora_variant_forward_avoids_inplace_when_delta_depends_on_result():
    torch.manual_seed(43)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bias=False keeps base_result aliased to `result` inside DoRA compose.
    base = nn.Linear(16, 12, bias=False).to(device)
    module = LoraLinear(
        base,
        adapter_name="default",
        r=4,
        lora_alpha=8,
        use_dora=True,
    ).to(device)
    module.train()
    module.lora_dropout["default"] = nn.Identity()

    x = torch.randn(3, 16, device=device, requires_grad=True)
    result = module.base_layer(x)
    result_ptr = result.data_ptr()

    updated = module.lora_variant["default"].forward(module, "default", x, result)

    # Must be out-of-place to avoid autograd versioning errors.
    assert updated.data_ptr() != result_ptr

    loss = updated.sum()
    loss.backward()

    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert module.lora_magnitude_vector["default"].weight.grad is not None


def test_vanilla_lora_linear_accumulates_in_place():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = nn.Linear(16, 12, bias=True).to(device)
    module = LoraLinear(
        base,
        adapter_name="default",
        r=4,
        lora_alpha=8,
        use_dora=False,
    ).to(device)
    module.eval()
    module.lora_dropout["default"] = nn.Identity()

    x = torch.randn(5, 16, device=device, requires_grad=True)

    captured = {}

    def hook(_module, _inputs, output):
        captured["ptr"] = output.data_ptr()
        captured["base"] = output.detach().clone()

    handle = module.base_layer.register_forward_hook(hook)

    out = module(x)

    handle.remove()

    assert "ptr" in captured
    assert out.data_ptr() == captured["ptr"]

    delta = module.lora_B["default"](module.lora_A["default"](x))
    expected = captured["base"].to(out.dtype)
    expected.add_(delta.to(out.dtype), alpha=module.scaling["default"])
    assert torch.allclose(out, expected)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.all(torch.isfinite(x.grad))


class TestActivatedLora:
    @pytest.mark.parametrize(
        "input_ids, alora_invocation_tokens, expected_offsets",
        [
            ([[0, 1, 2, 3], [0, 4, 5, 6]], [1, 2], [3, None]),
            ([[1, 2, 1, 2], [0, 4, 1, 2]], [1, 2], [2, 2]),
            ([[1, 2, 3, 4], [0, 4, 1, 4]], [1, 2], [4, None]),
            ([[1, 2, 3, 4]], None, [None]),
        ],
    )
    # Verify alora_offsets are calculated correctly
    def test_calculate_alora_offsets(self, input_ids, alora_invocation_tokens, expected_offsets):
        config = LoraConfig(alora_invocation_tokens=alora_invocation_tokens)
        peft_config = {"default": config}

        # compute offsets
        offsets = calculate_alora_offsets(peft_config, "default", torch.tensor(input_ids))

        assert offsets == expected_offsets

    @pytest.mark.parametrize(
        "input_ids, alora_invocations, expected_offsets",
        [
            ([[0, 1, 1], [0, 2, 2]], {"a1": [1], "a2": [2]}, [1, 1]),
            ([[0, 1, 1], [0, 2, 2]], {"a1": [1], "a2": None}, [1, None]),
        ],
    )
    # Verify alora_offsets are correct with adapter names
    def test_calculate_alora_offsets_with_adapter_names(self, input_ids, alora_invocations, expected_offsets):
        peft_config = {}
        for alora_name in alora_invocations.keys():
            peft_config[alora_name] = LoraConfig(alora_invocation_tokens=alora_invocations[alora_name])

        adapter_names = list(alora_invocations.keys())
        offsets = calculate_alora_offsets(
            peft_config, adapter_names[0], torch.tensor(input_ids), adapter_names=adapter_names
        )

        assert offsets == expected_offsets

    # Verify that the adapter does not modify outputs prior to invocation point
    def test_alora_activation_matches_base_until_invocation(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        start = 2
        with lora_model.disable_adapter():
            with torch.no_grad():
                base_out = lora_model(X=input_ids)

        kwargs = get_alora_offsets_for_forward(lora_model, input_ids)
        with torch.no_grad():
            lora_out = lora_model(X=input_ids, **kwargs)
        assert torch.allclose(lora_out[:, :start], base_out[:, :start])
        assert not torch.allclose(lora_out[:, start:], base_out[:, start:])

    # Verify that warning is given for alora when providing embeddings only
    def test_input_embeds_warning(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        input_embeds = base_model.embed(input_ids)
        with pytest.warns(
            UserWarning,
            match="Cannot calculate aLoRA offsets when only inputs_embeds are provided. Disabling aLoRA for this forward pass.",
        ):
            kwargs = get_alora_offsets_for_forward(lora_model, inputs_embeds=input_embeds)
        assert kwargs.get("alora_offsets") is None
        with pytest.warns(
            UserWarning,
            match="Cannot calculate aLoRA offsets during generate as input_ids are not available. Disabling aLoRA.",
        ):
            kwargs = get_alora_offsets_for_generate(lora_model, inputs_embeds=input_embeds)
        assert kwargs.get("alora_offsets") is None

    # Verify that error is raised when requesting num_beams > 1 for alora
    def test_num_beams_error(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        with pytest.raises(ValueError) as e:
            with torch.no_grad():
                lora_out = lora_model(X=input_ids, num_beams=2, alora_offsets=[3])
        assert "Beam search not yet supported for aLoRA." in str(e.value)


@pytest.mark.parametrize("fan_in_fan_out", [False, True])
@pytest.mark.parametrize("merge_mode", ["safe", "unsafe"])
def test_dora_linear_variant_merge_roundtrip_preserves_weights(merge_mode, fan_in_fan_out):
    """Linear DoRA merge and unmerge should round-trip the standard weight layout."""
    torch.manual_seed(120)
    orig_weight = torch.randn((5, 4), dtype=torch.float64)
    delta_weight = torch.randn((5, 4), dtype=torch.float64) * 0.05

    weight_norm = torch.linalg.vector_norm(
        transpose(orig_weight, fan_in_fan_out) + transpose(delta_weight, fan_in_fan_out),
        dim=1,
    )
    magnitude = weight_norm * (1.0 + 0.2 * torch.rand_like(weight_norm))
    module = _DummyVariantMergeModule(
        delta_weight=delta_weight,
        magnitude_layer=_DummyLinearMergeMagnitude(magnitude, fan_in_fan_out),
        fan_in_fan_out=fan_in_fan_out,
    )

    if merge_mode == "safe":
        original = orig_weight.clone()
        merged = DoraLinearVariant.merge_safe(module, "default", original)
        assert torch.allclose(original, orig_weight)
    else:
        merged = orig_weight.clone()
        DoraLinearVariant.merge_unsafe(module, "default", merged)

    restored = DoraLinearVariant.unmerge(module, "default", merged.clone())

    assert torch.allclose(restored, orig_weight)
    assert module._cache == {}


@pytest.mark.parametrize("merge_mode", ["safe", "unsafe"])
def test_dora_embedding_variant_merge_roundtrip_preserves_weights(merge_mode):
    """Embedding DoRA merge and unmerge should round-trip the original weight."""
    torch.manual_seed(121)
    orig_weight = torch.randn(6, 4, dtype=torch.float64)
    delta_weight = torch.randn(6, 4, dtype=torch.float64) * 0.05

    weight_norm = torch.linalg.vector_norm(orig_weight.T + delta_weight.T, dim=1)
    magnitude = weight_norm * (1.0 + 0.2 * torch.rand_like(weight_norm))
    module = _DummyVariantMergeModule(
        delta_weight=delta_weight,
        magnitude_layer=_DummyEmbeddingMergeMagnitude(magnitude),
    )

    if merge_mode == "safe":
        original = orig_weight.clone()
        merged = DoraEmbeddingVariant.merge_safe(module, "default", original)
        assert torch.allclose(original, orig_weight)
    else:
        merged = orig_weight.clone()
        DoraEmbeddingVariant.merge_unsafe(module, "default", merged)

    restored = DoraEmbeddingVariant.unmerge(module, "default", merged.clone())

    assert torch.allclose(restored, orig_weight)
    assert module._cache == {}


@pytest.mark.parametrize(
    "message",
    [
        "view was created in no_grad mode",
        "is a view and is being modified inplace",
    ],
)
def test_dora_linear_variant_forward_falls_back_on_protected_view_errors(message, monkeypatch):
    """Protected-view inplace errors should trigger the out-of-place accumulation fallback."""
    torch.manual_seed(122)
    base_layer = nn.Linear(4, 3, bias=True)
    x = torch.randn(2, 4)
    result = torch.randn(2, 3)
    delta = torch.randn_like(result) * 0.1
    module = _DummyForwardModule(base_layer, delta, training=False)

    original_add_ = torch.Tensor.add_

    def _patched_add_(self, other):
        if self.data_ptr() == result.data_ptr():
            raise RuntimeError(message)
        return original_add_(self, other)

    monkeypatch.setattr(torch.Tensor, "add_", _patched_add_, raising=False)

    updated = DoraLinearVariant.forward(module, "default", x, result)

    assert updated.data_ptr() != result.data_ptr()
    assert torch.allclose(updated, result + delta)


def test_dora_linear_variant_forward_reraises_unrelated_inplace_errors(monkeypatch):
    """Unexpected inplace failures should still surface to the caller."""
    torch.manual_seed(123)
    base_layer = nn.Linear(4, 3, bias=True)
    x = torch.randn(2, 4)
    result = torch.randn(2, 3)
    delta = torch.randn_like(result) * 0.1
    module = _DummyForwardModule(base_layer, delta, training=False)

    original_add_ = torch.Tensor.add_

    def _patched_add_(self, other):
        if self.data_ptr() == result.data_ptr():
            raise RuntimeError("unexpected inplace failure")
        return original_add_(self, other)

    monkeypatch.setattr(torch.Tensor, "add_", _patched_add_, raising=False)

    with pytest.raises(RuntimeError, match="unexpected inplace failure"):
        DoraLinearVariant.forward(module, "default", x, result)
