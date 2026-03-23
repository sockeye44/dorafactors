#!/usr/bin/env python3
"""Mechanistic DoRA Inference Audit.

Phase-gated measurement of DoRA composition accuracy across norm computation,
compose paths, backward paths, and end-to-end decode.

Phases:
  A: Static audit — self-consistency (deterministic forward)
  B: Norm triangle — dense vs factored vs base-only norms
  C: Compose audit — offline recomputation on captured activations
  D: Backward audit — gradient comparison across tiers
  E: End-to-end decode — token-level comparison (conditional)
  F: Dispatch/dtype trace — observability

Usage:
    python scripts/dora_inference_audit.py \\
        --base-model /root/Qwen2-VL-7B-Instruct \\
        --adapter not_so_smol_study_participants/manyeyes-v0-59-n01-ckpt2000-lr \\
        --prompts "The capital of France is" "In quantum computing, a qubit" "def fibonacci(n):" \\
        --max-tokens 32 \\
        --phases A,B,C,D,E,F \\
        --capture-modules 4 \\
        --out-dir audit_results/ \\
        --verbose
"""

import argparse
import functools
import hashlib
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("dora_audit")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
class AuditEncoder(json.JSONEncoder):
    """Encode torch dtypes and other non-JSON-native types."""

    def default(self, o):
        if isinstance(o, torch.dtype):
            return str(o)
        if isinstance(o, torch.device):
            return str(o)
        if isinstance(o, torch.Size):
            return list(o)
        return super().default(o)


def _save_json(path, obj, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with open(path, "a") as f:
            f.write(json.dumps(obj, cls=AuditEncoder) + "\n")
    else:
        with open(path, "w") as f:
            json.dump(obj, f, cls=AuditEncoder, indent=2)
    log.info("Wrote %s", path)


def _tensor_stats(t: torch.Tensor, *, prefix: str = "") -> dict:
    """Summary stats for a tensor (computed in fp32)."""
    t = t.detach().float()
    d = {
        f"{prefix}shape": list(t.shape),
        f"{prefix}dtype": str(t.dtype),
        f"{prefix}mean": t.mean().item(),
        f"{prefix}std": t.std().item() if t.numel() > 1 else 0.0,
        f"{prefix}min": t.min().item(),
        f"{prefix}max": t.max().item(),
        f"{prefix}absmax": t.abs().max().item(),
    }
    return d


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model_and_adapter(base_model_path, adapter_path, device="cuda"):
    """Load base model + PeftModel with DoRA adapter."""
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    log.info("Loading base model from %s", base_model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(base_model_path)

    log.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, processor


def _load_merged_model(merged_path, device="cuda"):
    """Load a pre-merged model (no DoRA composition)."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    log.info("Loading merged model from %s", merged_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        merged_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(merged_path)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# DoRA module enumeration
# ---------------------------------------------------------------------------
def _find_dora_modules(model):
    """Find all DoRA-adapted modules in a PeftModel.

    Returns:
        List of (name, lora_layer, dora_layer) where:
          - name: full dotted module path
          - lora_layer: the LoRA Linear module (has lora_A, lora_B, scaling, base_layer)
          - dora_layer: the DoraLinearLayer (has .weight = magnitude vector)
    """
    from peft.tuners.lora.dora import DoraLinearLayer

    results = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_magnitude_vector") and len(module.lora_magnitude_vector) > 0:
            for adapter_name, dora_layer in module.lora_magnitude_vector.items():
                if isinstance(dora_layer, DoraLinearLayer):
                    results.append((f"{name}[{adapter_name}]", module, dora_layer))
    return results


def _get_adapter_name(module):
    """Get the active adapter name for a LoRA module."""
    if hasattr(module, "active_adapter"):
        adapters = module.active_adapter
        if isinstance(adapters, (list, tuple)):
            return adapters[0] if adapters else "default"
        return adapters
    return "default"


def _get_lora_weights(module, adapter_name):
    """Extract lora_A, lora_B weights and scaling for an adapter."""
    lora_A_w = module.lora_A[adapter_name].weight
    lora_B_w = module.lora_B[adapter_name].weight
    scaling = module.scaling[adapter_name]
    return lora_A_w, lora_B_w, scaling


def _get_base_weight(module):
    """Get the base (non-LoRA) weight from a LoRA module."""
    from peft.utils.integrations import dequantize_module_weight

    base_layer = module.get_base_layer()
    return dequantize_module_weight(base_layer)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def _tokenize_prompts(processor, prompts):
    """Tokenize text prompts for Qwen2-VL (text-only, no images)."""
    results = []
    for prompt in prompts:
        inputs = processor(
            text=prompt,
            return_tensors="pt",
            padding=False,
        )
        # Compute hash for reproducibility tracking
        ids = inputs["input_ids"][0]
        h = hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:16]
        results.append({
            "prompt": prompt,
            "input_ids": ids,
            "hash": h,
            "length": len(ids),
        })
    return results


# ---------------------------------------------------------------------------
# Phase A: Static Audit
# ---------------------------------------------------------------------------
def phase_a(model, processor, prompts, out_dir, verbose=False):
    """Self-consistency check: two identical eager forwards must match exactly."""
    log.info("=" * 60)
    log.info("PHASE A: Static Audit (Self-Consistency)")
    log.info("=" * 60)

    dora_modules = _find_dora_modules(model)
    log.info("Found %d DoRA modules", len(dora_modules))

    # Get adapter config
    adapter_name = _get_adapter_name(dora_modules[0][1]) if dora_modules else "default"
    config_info = {}
    if hasattr(model, "peft_config"):
        peft_cfg = model.peft_config.get(adapter_name, None)
        if peft_cfg:
            config_info = {
                "r": peft_cfg.r,
                "lora_alpha": peft_cfg.lora_alpha,
                "use_dora": peft_cfg.use_dora,
                "use_rslora": peft_cfg.use_rslora,
                "target_modules": list(peft_cfg.target_modules) if peft_cfg.target_modules else [],
            }

    tokenized = _tokenize_prompts(processor, prompts)
    prompt_hashes = {t["prompt"]: t["hash"] for t in tokenized}

    # Dtype mix across DoRA modules
    dtype_mix = {}
    for name, lora_mod, dora_layer in dora_modules:
        an = _get_adapter_name(lora_mod)
        mag_dtype = str(dora_layer.weight.dtype)
        base_dtype = str(_get_base_weight(lora_mod).dtype)
        lora_A_w, _, _ = _get_lora_weights(lora_mod, an)
        lora_dtype = str(lora_A_w.dtype)
        key = f"mag={mag_dtype}, base={base_dtype}, lora={lora_dtype}"
        dtype_mix[key] = dtype_mix.get(key, 0) + 1

    # Self-consistency: two eager forwards, force no fused
    prompt = prompts[0]
    tok = tokenized[0]
    input_ids = tok["input_ids"].unsqueeze(0).to(model.device)

    # Force eager paths
    old_fused = os.environ.get("PEFT_DORA_FUSED")
    old_fused_bw = os.environ.get("PEFT_DORA_FUSED_BACKWARD")
    os.environ["PEFT_DORA_FUSED"] = "0"
    os.environ["PEFT_DORA_FUSED_BACKWARD"] = "0"
    # Clear cached flags
    _clear_fused_caches()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out1 = model(input_ids=input_ids)
        logits1 = out1.logits.clone()

        out2 = model(input_ids=input_ids)
        logits2 = out2.logits.clone()

    _restore_env("PEFT_DORA_FUSED", old_fused)
    _restore_env("PEFT_DORA_FUSED_BACKWARD", old_fused_bw)
    _clear_fused_caches()

    max_diff = (logits1 - logits2).abs().max().item()
    self_consistent = max_diff == 0.0

    setup = {
        "module_count": len(dora_modules),
        "adapter_name": adapter_name,
        "adapter_config": config_info,
        "dtype_mix": dtype_mix,
        "prompt_hashes": prompt_hashes,
        "self_consistency": {
            "max_diff": max_diff,
            "passed": self_consistent,
        },
        "module_names": [name for name, _, _ in dora_modules],
    }

    _save_json(out_dir / "setup.json", setup)

    if not self_consistent:
        log.error("PHASE A FAILED: Non-deterministic forward! max_diff=%e", max_diff)
        log.error("Stopping audit — something is non-deterministic.")
        return False, setup

    log.info("PHASE A PASSED: Self-consistent (max_diff=0.0)")
    return True, setup


# ---------------------------------------------------------------------------
# Phase B: Norm Triangle
# ---------------------------------------------------------------------------
def phase_b(model, adapter_paths, base_model_path, out_dir, verbose=False):
    """Compare dense vs factored vs base-only norms for all DoRA modules.

    Runs on all provided adapters to verify norm triangle across ranks.
    """
    log.info("=" * 60)
    log.info("PHASE B: Norm Triangle")
    log.info("=" * 60)

    from peft.utils.other import transpose as peft_transpose

    all_summaries = {}

    for adapter_path in adapter_paths:
        adapter_name_str = Path(adapter_path).name
        log.info("--- Adapter: %s ---", adapter_name_str)

        model_b, _ = _load_model_and_adapter(base_model_path, adapter_path)
        dora_modules = _find_dora_modules(model_b)

        per_module = []
        for mod_name, lora_mod, dora_layer in dora_modules:
            adapter_name = _get_adapter_name(lora_mod)
            lora_A_w, lora_B_w, scaling = _get_lora_weights(lora_mod, adapter_name)
            base_weight = _get_base_weight(lora_mod)
            magnitude = dora_layer.weight.detach()

            fan_in_fan_out = getattr(dora_layer, "fan_in_fan_out", False)

            # All computations in fp32, no autocast
            with torch.no_grad():
                W = peft_transpose(base_weight, fan_in_fan_out).float()
                A = lora_A_w.float().to(W.device)
                B = lora_B_w.float().to(W.device)
                s = float(scaling)
                mag = magnitude.float().to(W.device)

                # 1. Dense: ||W + s*(B@A)||_row
                BA = B @ A
                dense_norm = torch.linalg.norm(W + s * BA, dim=1)

                # 2. Factored: recompute the factored expansion in fp32
                # (same math as _get_weight_norm_linear but without the
                # output-dtype cast, so we measure factorization accuracy
                # independent of quantization).
                U_full = W @ A.T  # [out, r]
                gram = A @ A.T   # [r, r]
                cross_term = (B * U_full).sum(dim=1)
                BA_gram = B @ gram
                ba_norm_sq = (BA_gram * B).sum(dim=1)
                w_norm_sq = (W * W).sum(dim=1)
                norm_sq = w_norm_sq + (2.0 * s) * cross_term + (s * s) * ba_norm_sq
                norm_sq = norm_sq.clamp_min(0)
                factored_norm = torch.sqrt(norm_sq)

                # Also get the actual output from _get_weight_norm_linear
                # to measure the output-dtype quantization error
                factored_norm_native = dora_layer._get_weight_norm_linear(
                    base_weight=base_weight,
                    lora_A_w=lora_A_w,
                    lora_B_w=lora_B_w,
                    scaling=scaling,
                )
                factored_output_dtype = factored_norm_native.dtype
                factored_norm_output = factored_norm_native.float().to(W.device)

                # Dense cast to output dtype for apples-to-apples comparison
                dense_norm_quantized = dense_norm.to(factored_output_dtype).float()

                # 3. Base-only: ||W||_row (what analyze_dora_magnitudes uses WRONGLY)
                base_only_norm = torch.linalg.norm(W, dim=1)

                # Compute mag_norm_scale for each
                eps = 1e-12
                mns_dense = mag / dense_norm.clamp(min=eps)
                mns_factored = mag / factored_norm.clamp(min=eps)
                mns_base_only = mag / base_only_norm.clamp(min=eps)

                # Collapse rates
                EPS_BF16_HALF = 0.00390625
                collapse_dense = ((mns_dense - 1).abs() < EPS_BF16_HALF).float().mean().item()
                collapse_factored = ((mns_factored - 1).abs() < EPS_BF16_HALF).float().mean().item()
                collapse_base_only = ((mns_base_only - 1).abs() < EPS_BF16_HALF).float().mean().item()

                # Dense vs factored: fp32 (before output quantization)
                # This measures the actual factorization accuracy
                rel_err_dense_vs_factored = (
                    (dense_norm - factored_norm).abs() / dense_norm.clamp(min=eps)
                )
                max_rel_err = rel_err_dense_vs_factored.max().item()
                mean_rel_err = rel_err_dense_vs_factored.mean().item()

                # Dense vs base_only
                rel_err_dense_vs_base = (
                    (dense_norm - base_only_norm).abs() / dense_norm.clamp(min=eps)
                )

                entry = {
                    "module": mod_name,
                    "out_features": W.shape[0],
                    "in_features": W.shape[1],
                    "rank": A.shape[0],
                    "scaling": s,
                    "factored_output_dtype": str(factored_output_dtype),
                    "dense_vs_factored": {
                        "max_rel_err": max_rel_err,
                        "mean_rel_err": mean_rel_err,
                        "max_abs_err": (dense_norm - factored_norm).abs().max().item(),
                        "note": "fp32 comparison (measures factorization + output quantization)",
                    },
                    "dense_vs_factored_output": {
                        "max_rel_err": (
                            (dense_norm - factored_norm_output).abs()
                            / dense_norm.clamp(min=eps)
                        ).max().item(),
                        "mean_rel_err": (
                            (dense_norm - factored_norm_output).abs()
                            / dense_norm.clamp(min=eps)
                        ).mean().item(),
                        "note": "fp32 dense vs bf16-output-cast factored (includes quantization)",
                    },
                    "dense_vs_base_only": {
                        "max_rel_err": rel_err_dense_vs_base.max().item(),
                        "mean_rel_err": rel_err_dense_vs_base.mean().item(),
                    },
                    "collapse_rates_bf16": {
                        "dense": collapse_dense,
                        "factored": collapse_factored,
                        "base_only": collapse_base_only,
                    },
                    "norm_stats": {
                        "dense": _tensor_stats(dense_norm, prefix="dense_"),
                        "factored": _tensor_stats(factored_norm, prefix="factored_"),
                        "base_only": _tensor_stats(base_only_norm, prefix="base_only_"),
                    },
                    "mns_stats": {
                        "dense": _tensor_stats(mns_dense, prefix="mns_dense_"),
                        "factored": _tensor_stats(mns_factored, prefix="mns_factored_"),
                        "base_only": _tensor_stats(mns_base_only, prefix="mns_base_"),
                    },
                }
                per_module.append(entry)
                _save_json(out_dir / "norm_audit.jsonl", {
                    "adapter": adapter_name_str, **entry
                }, jsonl=True)

        # Aggregate summary — use fp32 factored comparison for pass/fail
        # (dense_vs_factored compares both in fp32, no output dtype cast)
        max_rel_errs = [e["dense_vs_factored"]["max_rel_err"] for e in per_module]
        output_max_rel_errs = [e["dense_vs_factored_output"]["max_rel_err"] for e in per_module]

        summary = {
            "adapter": adapter_name_str,
            "num_modules": len(per_module),
            "dense_vs_factored": {
                "worst_max_rel_err_fp32": max(max_rel_errs) if max_rel_errs else 0,
                "worst_max_rel_err_with_output_cast": max(output_max_rel_errs) if output_max_rel_errs else 0,
                "all_below_1e6": all(e < 1e-6 for e in max_rel_errs),
                "any_above_1e4": any(e > 1e-4 for e in max_rel_errs),
                "note": "fp32 = both computed in fp32 (factored formula, no output cast)",
            },
            "collapse_rates_bf16_aggregate": {
                "dense_mean": sum(e["collapse_rates_bf16"]["dense"] for e in per_module) / len(per_module) if per_module else 0,
                "factored_mean": sum(e["collapse_rates_bf16"]["factored"] for e in per_module) / len(per_module) if per_module else 0,
                "base_only_mean": sum(e["collapse_rates_bf16"]["base_only"] for e in per_module) / len(per_module) if per_module else 0,
            },
        }
        all_summaries[adapter_name_str] = summary

        log.info(
            "Adapter %s: worst dense-vs-factored max_rel_err: fp32=%e, with_output_cast=%e",
            adapter_name_str,
            summary["dense_vs_factored"]["worst_max_rel_err_fp32"],
            summary["dense_vs_factored"]["worst_max_rel_err_with_output_cast"],
        )

        # Free model
        del model_b
        torch.cuda.empty_cache()

    # Decision rule
    passed = True
    for name, s in all_summaries.items():
        if s["dense_vs_factored"]["any_above_1e4"]:
            log.error("PHASE B FAIL: %s has dense-vs-factored > 1e-4 — factorization bug!", name)
            passed = False
        elif s["dense_vs_factored"]["all_below_1e6"]:
            log.info("PHASE B: %s dense-vs-factored < 1e-6 — factorization is sound", name)
        else:
            log.info(
                "PHASE B: %s dense-vs-factored in [1e-6, 1e-4] — acceptable",
                name,
            )

    norm_summary = {
        "passed": passed,
        "per_adapter": all_summaries,
    }
    _save_json(out_dir / "norm_summary.json", norm_summary)
    return passed, norm_summary


# ---------------------------------------------------------------------------
# Phase C: Compose Audit
# ---------------------------------------------------------------------------
def _select_capture_modules(norm_entries, n=4):
    """Select modules for full tensor capture based on Phase B results.

    Criteria (not smallest/median/largest — informed by Phase B):
    1. Highest bf16 collapse rate (dense mns)
    2. Lowest bf16 collapse rate
    3. Largest out_features
    4. Largest norm mismatch (dense vs factored)
    """
    if not norm_entries:
        return []

    selected = set()
    names = []

    # 1. Highest collapse
    by_collapse = sorted(norm_entries, key=lambda e: e["collapse_rates_bf16"]["dense"], reverse=True)
    if by_collapse:
        selected.add(by_collapse[0]["module"])
        names.append(("highest_collapse", by_collapse[0]["module"]))

    # 2. Lowest collapse
    if by_collapse:
        for e in reversed(by_collapse):
            if e["module"] not in selected:
                selected.add(e["module"])
                names.append(("lowest_collapse", e["module"]))
                break

    # 3. Largest out_features
    by_size = sorted(norm_entries, key=lambda e: e["out_features"], reverse=True)
    for e in by_size:
        if e["module"] not in selected:
            selected.add(e["module"])
            names.append(("largest_out_features", e["module"]))
            break

    # 4. Largest norm mismatch
    by_mismatch = sorted(
        norm_entries,
        key=lambda e: e["dense_vs_factored"]["max_rel_err"],
        reverse=True,
    )
    for e in by_mismatch:
        if e["module"] not in selected:
            selected.add(e["module"])
            names.append(("largest_norm_mismatch", e["module"]))
            break

    # Fill remaining if needed
    while len(names) < min(n, len(norm_entries)):
        for e in norm_entries:
            if e["module"] not in selected:
                selected.add(e["module"])
                names.append(("extra", e["module"]))
                break
        else:
            break

    return names


def phase_c(model, processor, prompts, norm_entries, out_dir, n_capture=4, verbose=False,
            write_results=True):
    """Compose audit: capture activations, recompute with different formulas.

    C1: Dense norm, fp32 (oracle)
    C2: Factored norm, fp32
    C3: Factored norm, bf16 (cast)
    C4: Triton kernel on bf16 inputs
    """
    log.info("=" * 60)
    log.info("PHASE C: Compose Audit (Captured Activations)")
    log.info("=" * 60)

    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.tuners.lora.dora_fused import (
        _fused_dora_compose_torch,
        _fused_dora_compose_triton,
        fused_dora_compose,
    )

    # Select modules for full capture
    capture_selection = _select_capture_modules(norm_entries, n_capture)
    capture_module_names = {name for _, name in capture_selection}
    log.info("Selected %d modules for full tensor capture:", len(capture_selection))
    for reason, name in capture_selection:
        log.info("  [%s] %s", reason, name)

    # Step 1: Hook _compose_with_dispatch to capture tensors
    captured = {}  # module_name -> {lora_out, base_result, mag_norm_scale, scale}
    summary_stats = {}  # module_name -> summary stats for all modules

    dora_modules = _find_dora_modules(model)
    original_methods = {}

    def _make_hook(mod_name, dora_layer, orig_method):
        @functools.wraps(orig_method)
        def hooked(*, lora_out, base_result, mag_norm_scale, scale):
            # Always record summary stats
            summary_stats[mod_name] = {
                "lora_out": _tensor_stats(lora_out, prefix="lora_"),
                "base_result": _tensor_stats(base_result, prefix="base_"),
                "mag_norm_scale": _tensor_stats(mag_norm_scale, prefix="mag_"),
                "scale": scale,
                "lora_dtype": str(lora_out.dtype),
                "base_dtype": str(base_result.dtype),
                "mag_dtype": str(mag_norm_scale.dtype),
            }

            # Full tensor capture for selected modules
            if mod_name in capture_module_names:
                captured[mod_name] = {
                    "lora_out": lora_out.detach().clone(),
                    "base_result": base_result.detach().clone(),
                    "mag_norm_scale": mag_norm_scale.detach().clone(),
                    "scale": scale,
                }

            return orig_method(
                lora_out=lora_out,
                base_result=base_result,
                mag_norm_scale=mag_norm_scale,
                scale=scale,
            )
        return hooked

    # Install hooks
    for mod_name, lora_mod, dora_layer in dora_modules:
        orig = dora_layer._compose_with_dispatch
        original_methods[mod_name] = (dora_layer, orig)
        dora_layer._compose_with_dispatch = _make_hook(mod_name, dora_layer, orig)

    # Run forward with AMP (eager path)
    old_fused = os.environ.get("PEFT_DORA_FUSED")
    old_fused_bw = os.environ.get("PEFT_DORA_FUSED_BACKWARD")
    os.environ["PEFT_DORA_FUSED"] = "0"
    os.environ["PEFT_DORA_FUSED_BACKWARD"] = "0"
    _clear_fused_caches()

    tok = _tokenize_prompts(processor, prompts[:1])[0]
    input_ids = tok["input_ids"].unsqueeze(0).to(model.device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=input_ids)

    _restore_env("PEFT_DORA_FUSED", old_fused)
    _restore_env("PEFT_DORA_FUSED_BACKWARD", old_fused_bw)
    _clear_fused_caches()

    # Restore original methods
    for mod_name, (dora_layer, orig) in original_methods.items():
        dora_layer._compose_with_dispatch = orig

    log.info("Captured summary stats for %d modules, full tensors for %d",
             len(summary_stats), len(captured))

    # Step 2: Offline recomputation on captured tensors
    compose_results = []
    for mod_name, tensors in captured.items():
        lora_out = tensors["lora_out"]
        base_result = tensors["base_result"]
        mag_norm_scale = tensors["mag_norm_scale"]
        scale = tensors["scale"]

        with torch.no_grad():
            # C1: Oracle — dense formula, fp32
            lora_fp32 = lora_out.float()
            base_fp32 = base_result.float()
            mag_fp32 = mag_norm_scale.float()
            c1 = (mag_fp32 - 1) * base_fp32 + mag_fp32 * (scale * lora_fp32)

            # C2: Factored norm (same formula as C1 since we're recomputing
            # compose, not norm — the norm difference is already measured in
            # Phase B). In compose space, C2 = C1 when using fp32.
            c2 = c1.clone()

            # C3: Cast mag to bf16 (as happens in _compose_with_dispatch)
            mag_bf16 = mag_norm_scale.to(torch.bfloat16)
            lora_bf16 = lora_out.to(torch.bfloat16)
            base_bf16 = base_result.to(torch.bfloat16)
            c3 = (mag_bf16.float() - 1) * base_bf16.float() + mag_bf16.float() * (scale * lora_bf16.float())
            # Actually do it in bf16 natively
            c3_native = (mag_bf16 - 1) * base_bf16 + mag_bf16 * (scale * lora_bf16)

            # C4: Triton kernel on bf16 inputs
            lora_c4 = lora_bf16.clone().contiguous()
            try:
                c4 = fused_dora_compose(
                    lora_c4, base_bf16.contiguous(),
                    mag_bf16.contiguous(), scale, inplace=False,
                )
            except Exception as e:
                log.warning("Triton compose failed for %s: %s, using PyTorch fallback", mod_name, e)
                c4 = _fused_dora_compose_torch(
                    lora_bf16.clone(), base_bf16, mag_bf16, scale, inplace=False,
                )

            # Comparisons (all against C1 oracle in fp32)
            def _compare(a, b, name_a, name_b):
                a_f = a.float()
                b_f = b.float()
                diff = (a_f - b_f).abs()
                denom = a_f.abs().clamp(min=1e-12)
                rel = diff / denom
                return {
                    "comparison": f"{name_a} vs {name_b}",
                    "max_abs_err": diff.max().item(),
                    "mean_abs_err": diff.mean().item(),
                    "max_rel_err": rel.max().item(),
                    "mean_rel_err": rel.mean().item(),
                    "cosine_sim": F.cosine_similarity(
                        a_f.reshape(1, -1), b_f.reshape(1, -1)
                    ).item(),
                }

            entry = {
                "module": mod_name,
                "shape": list(lora_out.shape),
                "scale": scale,
                "C1_vs_C2": _compare(c1, c2, "C1_fp32_oracle", "C2_factored_fp32"),
                "C1_vs_C3_native": _compare(c1, c3_native, "C1_fp32_oracle", "C3_bf16_native"),
                "C1_vs_C4": _compare(c1, c4, "C1_fp32_oracle", "C4_kernel_bf16"),
                "C3_vs_C4": _compare(c3_native, c4, "C3_bf16_native", "C4_kernel_bf16"),
                "mag_collapse_bf16": {
                    "frac": ((mag_bf16.float() - 1).abs() < 0.00390625).float().mean().item(),
                    "gm1_absmax": (mag_bf16.float() - 1).abs().max().item(),
                    "gm1_absmin": (mag_bf16.float() - 1).abs().min().item(),
                },
            }
            compose_results.append(entry)
            if write_results:
                _save_json(out_dir / "compose_audit.jsonl", entry, jsonl=True)

            log.info(
                "  %s: C1↔C3 cosine=%.8f, C1↔C4 cosine=%.8f, C3↔C4 cosine=%.8f",
                mod_name,
                entry["C1_vs_C3_native"]["cosine_sim"],
                entry["C1_vs_C4"]["cosine_sim"],
                entry["C3_vs_C4"]["cosine_sim"],
            )

    return True, compose_results, captured, summary_stats


# ---------------------------------------------------------------------------
# Phase D: Backward Audit
# ---------------------------------------------------------------------------
def phase_d(model, processor, prompts, captured, out_dir, verbose=False):
    """Backward audit on captured tensors + real loss.

    D1: Micro-audit with synthetic d_out
    D2: One-step training audit with real CE loss
    """
    log.info("=" * 60)
    log.info("PHASE D: Backward Audit")
    log.info("=" * 60)

    from peft.tuners.lora.dora_fused import (
        _fused_backward_torch,
        _fused_backward_triton,
    )

    backward_results = []

    # D1: Micro-audit on captured tensors
    log.info("--- D1: Micro-audit (synthetic d_out) ---")
    for mod_name, tensors in captured.items():
        lora_out = tensors["lora_out"]
        base_result = tensors["base_result"]
        mag_norm_scale = tensors["mag_norm_scale"]
        scale = tensors["scale"]

        # Synthetic d_out in bf16 (matching activation dtype)
        torch.manual_seed(42)
        d_out_bf16 = torch.randn_like(lora_out, dtype=torch.bfloat16)

        with torch.no_grad():
            # inner = scale * lora + base (what the fused autograd saves)
            inner_fp32 = scale * lora_out.float() + base_result.float()
            inner_bf16 = (scale * lora_out.to(torch.bfloat16) + base_result.to(torch.bfloat16))

            mag_fp32 = mag_norm_scale.float()
            mag_bf16 = mag_norm_scale.to(torch.bfloat16)

            # Path 1: Eager fp32
            d_lora_fp32 = mag_fp32 * scale * d_out_bf16.float()
            d_base_fp32 = (mag_fp32 - 1) * d_out_bf16.float()

            # Path 2: Fused PyTorch bf16
            d_lora_fused_pt, d_base_fused_pt, _ = _fused_backward_torch(
                d_out_bf16, inner_bf16, mag_bf16, scale,
                needs_lora_grad=True, needs_base_grad=True, needs_mag_grad=True,
            )

            # Path 3: Fused Triton bf16
            try:
                d_lora_fused_tr, d_base_fused_tr, _ = _fused_backward_triton(
                    d_out_bf16.contiguous(), inner_bf16.contiguous(),
                    mag_bf16.contiguous(), scale,
                    needs_lora_grad=True, needs_base_grad=True, needs_mag_grad=True,
                )
            except Exception as e:
                log.warning("Triton backward failed for %s: %s", mod_name, e)
                d_lora_fused_tr = d_lora_fused_pt
                d_base_fused_tr = d_base_fused_pt

            # Metrics
            def _grad_metrics(ref, test, name):
                ref_f = ref.float().reshape(-1)
                test_f = test.float().reshape(-1)
                diff = (ref_f - test_f)
                rel_err = diff.abs() / ref_f.abs().clamp(min=1e-12)
                cos = F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()
                sign_agree = ((ref_f.sign() == test_f.sign()) | (ref_f == 0)).float().mean().item()
                return {
                    "comparison": name,
                    "rel_err_max": rel_err.max().item(),
                    "rel_err_mean": rel_err.mean().item(),
                    "cosine_sim": cos,
                    "sign_agreement": sign_agree,
                    "ref_norm": ref_f.norm().item(),
                    "test_norm": test_f.norm().item(),
                }

            # Gradient energy share
            d_base_energy = d_base_fp32.float().norm().item() ** 2
            d_lora_energy = d_lora_fp32.float().norm().item() ** 2
            total_energy = d_base_energy + d_lora_energy
            d_base_share = d_base_energy / max(total_energy, 1e-30)

            entry = {
                "module": mod_name,
                "phase": "D1",
                "d_base_metrics": {
                    "eager_fp32_vs_fused_torch_bf16": _grad_metrics(
                        d_base_fp32, d_base_fused_pt, "d_base: eager_fp32 vs fused_torch_bf16"
                    ),
                    "eager_fp32_vs_fused_triton_bf16": _grad_metrics(
                        d_base_fp32, d_base_fused_tr, "d_base: eager_fp32 vs fused_triton_bf16"
                    ),
                    "fused_torch_vs_triton": _grad_metrics(
                        d_base_fused_pt, d_base_fused_tr, "d_base: fused_torch vs fused_triton"
                    ),
                },
                "d_lora_metrics": {
                    "eager_fp32_vs_fused_torch_bf16": _grad_metrics(
                        d_lora_fp32, d_lora_fused_pt, "d_lora: eager_fp32 vs fused_torch_bf16"
                    ),
                    "eager_fp32_vs_fused_triton_bf16": _grad_metrics(
                        d_lora_fp32, d_lora_fused_tr, "d_lora: eager_fp32 vs fused_triton_bf16"
                    ),
                },
                "gradient_energy": {
                    "d_base_energy": d_base_energy,
                    "d_lora_energy": d_lora_energy,
                    "d_base_share": d_base_share,
                    "d_base_second_order": d_base_share < 0.01,
                },
            }
            backward_results.append(entry)
            _save_json(out_dir / "backward_audit.jsonl", entry, jsonl=True)

            log.info(
                "  %s: d_base cosine(fp32,triton_bf16)=%.6f, energy_share=%.4f%%",
                mod_name,
                entry["d_base_metrics"]["eager_fp32_vs_fused_triton_bf16"]["cosine_sim"],
                d_base_share * 100,
            )

    # D2: One-step training audit
    log.info("--- D2: One-step training audit (real CE loss) ---")
    d2_result = _d2_one_step_audit(model, processor, prompts, out_dir, verbose)
    if d2_result is not None:
        backward_results.append(d2_result)
        _save_json(out_dir / "backward_audit.jsonl", d2_result, jsonl=True)

    return True, backward_results


def _d2_one_step_audit(model, processor, prompts, out_dir, verbose):
    """One-step training audit: compare eager vs fused backward with real loss."""
    from peft.tuners.lora.dora import DoraLinearLayer

    tok = _tokenize_prompts(processor, prompts[:1])[0]
    input_ids = tok["input_ids"].unsqueeze(0).to(model.device)
    # Use input as both input and labels for CE loss
    labels = input_ids.clone()

    dora_modules = _find_dora_modules(model)

    results = {}

    for config_name, fused_val, fused_bw_val in [
        ("eager", "0", "0"),
        ("fused_backward", "1", "1"),
    ]:
        old_fused = os.environ.get("PEFT_DORA_FUSED")
        old_fused_bw = os.environ.get("PEFT_DORA_FUSED_BACKWARD")
        os.environ["PEFT_DORA_FUSED"] = fused_val
        os.environ["PEFT_DORA_FUSED_BACKWARD"] = fused_bw_val
        _clear_fused_caches()

        # Enable training mode — PeftModel.from_pretrained loads in
        # inference mode (all requires_grad=False). Re-enable adapter params.
        model.train()
        for mod_name, lora_mod, dora_layer in dora_modules:
            adapter_name = _get_adapter_name(lora_mod)
            # Enable grad on LoRA weights and magnitude
            lora_mod.lora_A[adapter_name].weight.requires_grad_(True)
            lora_mod.lora_B[adapter_name].weight.requires_grad_(True)
            dora_layer.weight.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss

        if loss is None:
            log.warning("D2: loss is None for config %s, skipping", config_name)
            _restore_env("PEFT_DORA_FUSED", old_fused)
            _restore_env("PEFT_DORA_FUSED_BACKWARD", old_fused_bw)
            _clear_fused_caches()
            continue

        loss.backward()

        # Collect gradients from DoRA modules
        grads = {}
        for mod_name, lora_mod, dora_layer in dora_modules:
            adapter_name = _get_adapter_name(lora_mod)
            mag_grad = dora_layer.weight.grad
            lora_A_grad = lora_mod.lora_A[adapter_name].weight.grad
            lora_B_grad = lora_mod.lora_B[adapter_name].weight.grad

            grads[mod_name] = {
                "mag": mag_grad.detach().clone().float() if mag_grad is not None else None,
                "lora_A": lora_A_grad.detach().clone().float() if lora_A_grad is not None else None,
                "lora_B": lora_B_grad.detach().clone().float() if lora_B_grad is not None else None,
            }

        results[config_name] = {
            "loss": loss.item(),
            "grads": grads,
        }

        _restore_env("PEFT_DORA_FUSED", old_fused)
        _restore_env("PEFT_DORA_FUSED_BACKWARD", old_fused_bw)
        _clear_fused_caches()

    # Restore inference mode
    model.eval()
    for mod_name, lora_mod, dora_layer in dora_modules:
        adapter_name = _get_adapter_name(lora_mod)
        lora_mod.lora_A[adapter_name].weight.requires_grad_(False)
        lora_mod.lora_B[adapter_name].weight.requires_grad_(False)
        dora_layer.weight.requires_grad_(False)

    # Compare eager vs fused
    if "eager" not in results or "fused_backward" not in results:
        log.warning("D2: Could not get both eager and fused results")
        return None

    eager_grads = results["eager"]["grads"]
    fused_grads = results["fused_backward"]["grads"]

    per_module_comparison = {}
    all_eager_flat = []
    all_fused_flat = []

    for mod_name in eager_grads:
        if mod_name not in fused_grads:
            continue
        eg = eager_grads[mod_name]
        fg = fused_grads[mod_name]

        mod_comp = {}
        for grad_name in ["mag", "lora_A", "lora_B"]:
            e_grad = eg[grad_name]
            f_grad = fg[grad_name]
            if e_grad is None or f_grad is None:
                continue

            e_flat = e_grad.reshape(-1)
            f_flat = f_grad.reshape(-1)
            all_eager_flat.append(e_flat)
            all_fused_flat.append(f_flat)

            cos = F.cosine_similarity(e_flat.unsqueeze(0), f_flat.unsqueeze(0)).item()
            rel_err = ((e_flat - f_flat).abs() / e_flat.abs().clamp(min=1e-12))

            mod_comp[grad_name] = {
                "cosine_sim": cos,
                "rel_err_max": rel_err.max().item(),
                "rel_err_mean": rel_err.mean().item(),
            }

        per_module_comparison[mod_name] = mod_comp

    # Full-model parameter-grad cosine
    if all_eager_flat and all_fused_flat:
        all_e = torch.cat(all_eager_flat)
        all_f = torch.cat(all_fused_flat)
        full_cosine = F.cosine_similarity(all_e.unsqueeze(0), all_f.unsqueeze(0)).item()
    else:
        full_cosine = None

    d2_entry = {
        "phase": "D2",
        "eager_loss": results["eager"]["loss"],
        "fused_loss": results["fused_backward"]["loss"],
        "loss_diff": abs(results["eager"]["loss"] - results["fused_backward"]["loss"]),
        "full_model_param_grad_cosine": full_cosine,
        "per_module": per_module_comparison,
    }

    log.info(
        "D2: eager_loss=%.6f, fused_loss=%.6f, param_grad_cosine=%.8f",
        d2_entry["eager_loss"],
        d2_entry["fused_loss"],
        full_cosine if full_cosine is not None else 0.0,
    )

    return d2_entry


# ---------------------------------------------------------------------------
# Phase E: End-to-End Decode
# ---------------------------------------------------------------------------
def phase_e(model, processor, prompts, base_model_path, adapter_path, out_dir,
            max_tokens=32, verbose=False):
    """End-to-end decode comparison: eager vs fused vs merged."""
    log.info("=" * 60)
    log.info("PHASE E: End-to-End Decode")
    log.info("=" * 60)

    tokenized = _tokenize_prompts(processor, prompts)
    configs = []

    # Config 1: Eager
    configs.append(("eager", model, processor, {"PEFT_DORA_FUSED": "0", "PEFT_DORA_FUSED_BACKWARD": "0"}))

    # Config 2: Fused forward (no fused backward — inference)
    configs.append(("fused_forward", model, processor, {"PEFT_DORA_FUSED": "1", "PEFT_DORA_FUSED_BACKWARD": "0"}))

    # Config 3: Merged model (if available)
    merged_path = Path(adapter_path).parent / (Path(adapter_path).name + "_merged_model")
    merged_model = None
    merged_proc = None
    if merged_path.exists():
        log.info("Loading merged model from %s", merged_path)
        merged_model, merged_proc = _load_merged_model(str(merged_path))
        configs.append(("merged", merged_model, merged_proc, {}))
    else:
        log.warning("No merged model at %s, skipping merged comparison", merged_path)

    all_results = []

    for tok_info in tokenized:
        prompt = tok_info["prompt"]
        log.info("Prompt: '%s'", prompt)

        prompt_results = {"prompt": prompt, "prompt_hash": tok_info["hash"], "configs": {}}

        for config_name, m, p, env_overrides in configs:
            old_env = {}
            for k, v in env_overrides.items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = v
            if env_overrides:
                _clear_fused_caches()

            input_ids = tok_info["input_ids"].unsqueeze(0).to(m.device)

            # Greedy decode step by step to capture per-step logits
            generated_ids = input_ids.clone()
            step_data = []

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for step in range(max_tokens):
                    out = m(input_ids=generated_ids)
                    logits = out.logits[:, -1, :]  # [1, vocab_size]
                    next_token = logits.argmax(dim=-1, keepdim=True)

                    # Record step data
                    logits_f = logits.float().squeeze(0)
                    top2 = torch.topk(logits_f, k=2)
                    margin = (top2.values[0] - top2.values[1]).item()

                    step_data.append({
                        "step": step,
                        "token_id": next_token.item(),
                        "top1_logit": top2.values[0].item(),
                        "top2_logit": top2.values[1].item(),
                        "margin": margin,
                        "logits_norm": logits_f.norm().item(),
                    })

                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    # Stop on EOS
                    if hasattr(p, "tokenizer") and p.tokenizer is not None:
                        if next_token.item() == p.tokenizer.eos_token_id:
                            break
                    elif hasattr(p, "eos_token_id"):
                        if next_token.item() == p.eos_token_id:
                            break

            token_ids = generated_ids[0, input_ids.shape[1]:].tolist()

            # Try to decode
            try:
                if hasattr(p, "tokenizer") and p.tokenizer is not None:
                    text = p.tokenizer.decode(token_ids, skip_special_tokens=True)
                else:
                    text = p.decode(token_ids, skip_special_tokens=True)
            except Exception:
                text = str(token_ids)

            prompt_results["configs"][config_name] = {
                "token_ids": token_ids,
                "text": text,
                "steps": step_data,
            }

            # Restore env
            for k, old_v in old_env.items():
                _restore_env(k, old_v)
            if env_overrides:
                _clear_fused_caches()

        # Cross-config comparison
        config_names = list(prompt_results["configs"].keys())
        comparisons = {}
        for i, c1 in enumerate(config_names):
            for c2 in config_names[i + 1:]:
                ids1 = prompt_results["configs"][c1]["token_ids"]
                ids2 = prompt_results["configs"][c2]["token_ids"]
                min_len = min(len(ids1), len(ids2))

                first_diverge = None
                for j in range(min_len):
                    if ids1[j] != ids2[j]:
                        first_diverge = j
                        break

                tokens_match = ids1[:min_len] == ids2[:min_len] and len(ids1) == len(ids2)

                # Pre-divergence margin
                pre_div_margins = []
                if first_diverge is not None and first_diverge > 0:
                    for j in range(first_diverge):
                        s1 = prompt_results["configs"][c1]["steps"][j]
                        pre_div_margins.append(s1["margin"])

                comparisons[f"{c1}_vs_{c2}"] = {
                    "tokens_match": tokens_match,
                    "first_divergence": first_diverge,
                    "pre_divergence_avg_margin": (
                        sum(pre_div_margins) / len(pre_div_margins) if pre_div_margins else None
                    ),
                }

        prompt_results["comparisons"] = comparisons
        all_results.append(prompt_results)

        for comp_name, comp in comparisons.items():
            log.info(
                "  %s: match=%s, first_div=%s",
                comp_name,
                comp["tokens_match"],
                comp["first_divergence"],
            )

    _save_json(out_dir / "token_trace.json", all_results)

    # Cleanup merged model
    if merged_model is not None:
        del merged_model
        torch.cuda.empty_cache()

    return True, all_results


# ---------------------------------------------------------------------------
# Phase F: Dispatch/Dtype Trace
# ---------------------------------------------------------------------------
def phase_f(model, processor, prompts, out_dir, verbose=False):
    """Dispatch and dtype trace across AMP/no-AMP conditions."""
    log.info("=" * 60)
    log.info("PHASE F: Dispatch/Dtype Trace")
    log.info("=" * 60)

    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.tuners.lora.dora_fused import (
        fused_dora_compose as _orig_fused_compose,
        fused_dora_forward_and_inner as _orig_fused_fwd_inner,
    )
    import peft.tuners.lora.dora_fused as dora_fused_module

    dora_modules = _find_dora_modules(model)
    tok = _tokenize_prompts(processor, prompts[:1])[0]
    input_ids = tok["input_ids"].unsqueeze(0).to(model.device)

    dispatch_log = []

    for condition_name, use_autocast in [("no_autocast", False), ("amp_bf16", True)]:
        log.info("--- Condition: %s ---", condition_name)

        # Hook compose_with_dispatch
        compose_traces = {}
        norm_traces = {}
        fused_compose_traces = []
        fused_fwd_inner_traces = []

        original_methods = {}

        def _make_compose_hook(mod_name, dora_layer, orig_method):
            @functools.wraps(orig_method)
            def hooked(*, lora_out, base_result, mag_norm_scale, scale):
                compose_traces[mod_name] = {
                    "lora_dtype": str(lora_out.dtype),
                    "base_dtype": str(base_result.dtype),
                    "mag_dtype": str(mag_norm_scale.dtype),
                    "mag_dtype_after_cast": None,  # filled by dispatch logic
                    "scale": scale,
                    "needs_grad": (
                        lora_out.requires_grad or base_result.requires_grad
                        or mag_norm_scale.requires_grad
                    ),
                }
                result = orig_method(
                    lora_out=lora_out, base_result=base_result,
                    mag_norm_scale=mag_norm_scale, scale=scale,
                )
                compose_traces[mod_name]["output_dtype"] = str(result.dtype)
                return result
            return hooked

        def _make_norm_hook(mod_name, dora_layer, orig_method):
            @functools.wraps(orig_method)
            def hooked(*, base_weight, lora_A_w, lora_B_w, scaling, chunk_size=None):
                input_dtypes = {
                    "base_weight": str(base_weight.dtype),
                    "lora_A_w": str(lora_A_w.dtype),
                    "lora_B_w": str(lora_B_w.dtype),
                }
                result = orig_method(
                    base_weight=base_weight, lora_A_w=lora_A_w,
                    lora_B_w=lora_B_w, scaling=scaling, chunk_size=chunk_size,
                )
                norm_traces[mod_name] = {
                    "input_dtypes": input_dtypes,
                    "output_dtype": str(result.dtype),
                }
                return result
            return hooked

        # Hook fused_dora_compose at module level
        fused_compose_call_log = []

        @functools.wraps(_orig_fused_compose)
        def _hooked_fused_compose(lora, base, mag_norm_scale, scale, inplace=True):
            is_triton = (
                hasattr(dora_fused_module, "_TRITON_AVAILABLE")
                and dora_fused_module._TRITON_AVAILABLE
                and lora.is_cuda
                and lora.is_contiguous()
                and base.is_contiguous()
                and mag_norm_scale.is_contiguous()
                and lora.dtype == base.dtype == mag_norm_scale.dtype
            )
            fused_compose_call_log.append({
                "condition": condition_name,
                "triton": is_triton,
                "lora_dtype": str(lora.dtype),
                "base_dtype": str(base.dtype),
                "mag_dtype": str(mag_norm_scale.dtype),
                "inplace": inplace,
            })
            return _orig_fused_compose(lora, base, mag_norm_scale, scale, inplace)

        @functools.wraps(_orig_fused_fwd_inner)
        def _hooked_fused_fwd_inner(lora, base, mag_norm_scale, scale):
            is_triton = (
                hasattr(dora_fused_module, "_TRITON_AVAILABLE")
                and dora_fused_module._TRITON_AVAILABLE
                and lora.is_cuda
                and lora.is_contiguous()
                and base.is_contiguous()
                and mag_norm_scale.is_contiguous()
                and lora.dtype == base.dtype == mag_norm_scale.dtype
            )
            fused_fwd_inner_traces.append({
                "condition": condition_name,
                "triton": is_triton,
                "lora_dtype": str(lora.dtype),
                "mag_dtype": str(mag_norm_scale.dtype),
            })
            return _orig_fused_fwd_inner(lora, base, mag_norm_scale, scale)

        # Install hooks
        for mod_name, lora_mod, dora_layer in dora_modules:
            orig_compose = dora_layer._compose_with_dispatch
            orig_norm = dora_layer._get_weight_norm_linear
            original_methods[mod_name] = (dora_layer, orig_compose, orig_norm)
            dora_layer._compose_with_dispatch = _make_compose_hook(mod_name, dora_layer, orig_compose)
            dora_layer._get_weight_norm_linear = _make_norm_hook(mod_name, dora_layer, orig_norm)

        dora_fused_module.fused_dora_compose = _hooked_fused_compose
        dora_fused_module.fused_dora_forward_and_inner = _hooked_fused_fwd_inner

        # Run forward
        model.eval()
        with torch.no_grad():
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=input_ids)
            else:
                model(input_ids=input_ids)

        # Restore
        for mod_name, (dora_layer, orig_compose, orig_norm) in original_methods.items():
            dora_layer._compose_with_dispatch = orig_compose
            dora_layer._get_weight_norm_linear = orig_norm

        dora_fused_module.fused_dora_compose = _orig_fused_compose
        dora_fused_module.fused_dora_forward_and_inner = _orig_fused_fwd_inner

        # Record
        dispatch_entry = {
            "condition": condition_name,
            "use_autocast": use_autocast,
            "num_modules": len(compose_traces),
            "compose_traces": compose_traces,
            "norm_traces": norm_traces,
            "fused_compose_calls": fused_compose_call_log,
            "fused_fwd_inner_calls": fused_fwd_inner_traces,
        }
        dispatch_log.append(dispatch_entry)
        _save_json(out_dir / "dispatch_trace.jsonl", dispatch_entry, jsonl=True)

        # Summary
        compose_dtypes = set()
        for name, trace in compose_traces.items():
            key = f"lora={trace['lora_dtype']}, mag={trace['mag_dtype']}"
            compose_dtypes.add(key)

        log.info(
            "  %s: %d modules traced, compose dtypes: %s, fused_compose calls: %d",
            condition_name,
            len(compose_traces),
            compose_dtypes,
            len(fused_compose_call_log),
        )

    return True, dispatch_log


# ---------------------------------------------------------------------------
# Env var helpers
# ---------------------------------------------------------------------------
def _restore_env(key, old_val):
    if old_val is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = old_val


def _clear_fused_caches():
    """Clear cached env-var flags in dora module (module-level globals)."""
    try:
        from peft.tuners.lora.dora import _invalidate_fused_cache
        _invalidate_fused_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Mechanistic DoRA Inference Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model", type=str, required=True,
        help="Path to base model (e.g. /root/Qwen2-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--adapter", type=str, required=True,
        help="Path to primary adapter for phases A,C,D,E,F",
    )
    parser.add_argument(
        "--extra-adapters", type=str, nargs="*", default=[],
        help="Additional adapters for Phase B norm triangle (multi-rank coverage)",
    )
    parser.add_argument(
        "--prompts", type=str, nargs="+",
        default=[
            "The capital of France is",
            "In quantum computing, a qubit",
            "def fibonacci(n):",
        ],
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--phases", type=str, default="A,B,C,D,E,F",
        help="Comma-separated phases to run",
    )
    parser.add_argument("--capture-modules", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="audit_results")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phases = [p.strip().upper() for p in args.phases.split(",")]
    log.info("Phases to run: %s", phases)
    log.info("Output dir: %s", out_dir)

    # Clear jsonl files only for phases being run
    phase_jsonl_map = {
        "B": ["norm_audit.jsonl"],
        "C": ["compose_audit.jsonl"],
        "D": ["backward_audit.jsonl"],
        "F": ["dispatch_trace.jsonl"],
    }
    for phase in phases:
        for fn in phase_jsonl_map.get(phase, []):
            p = out_dir / fn
            if p.exists():
                p.unlink()

    all_adapter_paths = [args.adapter] + args.extra_adapters

    # -----------------------------------------------------------------------
    # Phase A
    # -----------------------------------------------------------------------
    model = None
    processor = None
    setup = None

    if "A" in phases:
        model, processor = _load_model_and_adapter(args.base_model, args.adapter)
        passed, setup = phase_a(model, processor, args.prompts, out_dir, args.verbose)
        if not passed:
            log.error("Phase A failed. Aborting.")
            return 1

    # -----------------------------------------------------------------------
    # Phase B — runs on ALL adapters, loads/unloads each independently
    # -----------------------------------------------------------------------
    norm_entries = []  # Per-module entries from primary adapter for Phase C selection
    if "B" in phases:
        # Free model from Phase A to make room for Phase B loads
        if model is not None:
            del model
            model = None
            torch.cuda.empty_cache()

        passed, norm_summary = phase_b(
            None, all_adapter_paths, args.base_model, out_dir, args.verbose
        )

        # Read back norm_audit.jsonl to get per-module entries for primary adapter
        norm_audit_path = out_dir / "norm_audit.jsonl"
        if norm_audit_path.exists():
            primary_name = Path(args.adapter).name
            with open(norm_audit_path) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("adapter") == primary_name:
                        norm_entries.append(entry)

        if not passed:
            log.error("Phase B failed (factorization bug detected). Investigate before continuing.")
            return 1

        # Decision: if all < 1e-6, we could stop — but continue for mechanistic story
        all_clean = all(
            s["dense_vs_factored"]["all_below_1e6"]
            for s in norm_summary["per_adapter"].values()
        )
        if all_clean:
            log.info("Phase B: All adapters clean (< 1e-6). Factorization is sound.")
            log.info("Fix analysis script denominator + writeup numbers.")

    # -----------------------------------------------------------------------
    # Phases C-F use primary adapter only
    # -----------------------------------------------------------------------
    if any(p in phases for p in ["C", "D", "E", "F"]):
        if model is None:
            model, processor = _load_model_and_adapter(args.base_model, args.adapter)

    # If norm_entries is empty but norm_audit.jsonl exists from a prior run, load it
    if not norm_entries:
        norm_audit_path = out_dir / "norm_audit.jsonl"
        if norm_audit_path.exists():
            primary_name = Path(args.adapter).name
            with open(norm_audit_path) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("adapter") == primary_name:
                        norm_entries.append(entry)
            if norm_entries:
                log.info("Loaded %d norm entries from previous Phase B run", len(norm_entries))

    captured = {}

    # -----------------------------------------------------------------------
    # Phase C
    # -----------------------------------------------------------------------
    if "C" in phases:
        passed, compose_results, captured, summary_stats = phase_c(
            model, processor, args.prompts, norm_entries, out_dir,
            n_capture=args.capture_modules, verbose=args.verbose,
        )

    # -----------------------------------------------------------------------
    # Phase D
    # -----------------------------------------------------------------------
    if "D" in phases:
        if not captured:
            # Need to run Phase C first to get captured tensors
            log.info("Phase D requires captured tensors from Phase C, running capture pass...")
            _, _, captured, _ = phase_c(
                model, processor, args.prompts, norm_entries, out_dir,
                n_capture=args.capture_modules, verbose=args.verbose,
                write_results=False,
            )

        passed, backward_results = phase_d(
            model, processor, args.prompts, captured, out_dir, args.verbose,
        )

        # Stop condition: check if fused backward is fine
        d2_entries = [e for e in backward_results if e.get("phase") == "D2"]
        if d2_entries:
            d2 = d2_entries[0]
            cosine = d2.get("full_model_param_grad_cosine")
            if cosine is not None and cosine > 0.999:
                # Check d_base energy share
                d1_entries = [e for e in backward_results if e.get("phase") == "D1"]
                all_second_order = all(
                    e["gradient_energy"]["d_base_second_order"]
                    for e in d1_entries
                )
                if all_second_order:
                    log.info(
                        "STOP CONDITION MET: param_grad_cosine=%.6f (>0.999), "
                        "all d_base energy < 1%%. Fused backward is fine.",
                        cosine,
                    )

    # -----------------------------------------------------------------------
    # Phase E (conditional)
    # -----------------------------------------------------------------------
    if "E" in phases:
        passed, decode_results = phase_e(
            model, processor, args.prompts, args.base_model, args.adapter,
            out_dir, max_tokens=args.max_tokens, verbose=args.verbose,
        )

    # -----------------------------------------------------------------------
    # Phase F
    # -----------------------------------------------------------------------
    if "F" in phases:
        passed, dispatch_log = phase_f(
            model, processor, args.prompts, out_dir, args.verbose,
        )

    log.info("=" * 60)
    log.info("Audit complete. Results in %s", out_dir)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
