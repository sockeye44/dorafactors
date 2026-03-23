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

"""DoRA (Weight-Decomposed Low-Rank Adaptation) layer implementations.

``torch.compile`` compatibility
-------------------------------
The chunked composition method ``_compose_with_base_chunks`` is decorated with
``@dynamo_disable`` because its Python loop has data-dependent bounds that
change across layers, leading to runaway recompilations under Dynamo.

The main composition dispatch (``_compose_with_dispatch``) and ``forward()``
are **compile-friendly**: they contain only metadata-dependent branches
(``requires_grad``, cached env-var booleans, ``is_cuda``) that Dynamo can
guard on without graph breaks.

When ``PEFT_DORA_FUSED_BACKWARD=1``, the fused compose autograd path is
registered as a ``torch.library.custom_op`` (``peft::fused_dora_compose``,
requires PyTorch 2.4+).  Dynamo treats the custom op as a single opaque node,
allowing Inductor to fuse the LoRA A/B matmuls and subsequent activation
functions around it for a significant global throughput boost.

On PyTorch < 2.4 or when the custom-op registration is unavailable, the
autograd path falls back to plain PyTorch under Dynamo (no graph break, but
no opaque-node fusion benefit either).

The only remaining graph break is on the ``base_result is None`` path
(dropout-active / no precomputed base) which routes through the chunked
``_compose_with_base_chunks``.
"""

import logging
import os
import sys
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from functools import lru_cache
from typing import Optional

import torch

import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import (
    dequantize_module_weight,
    gather_params_ctx,
    check_deepspeed_zero3_enabled,
)
from peft.utils.other import transpose

try:
    from torch.amp import autocast
except ImportError:  # pragma: no cover - older torch versions
    autocast = None

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - FSDP not available in the environment
    FSDP = None

try:  # pragma: no cover - torch._dynamo only exists on recent PyTorch
    from torch._dynamo import graph_break as dynamo_graph_break
    from torch._dynamo import is_compiling as dynamo_is_compiling
    from torch._dynamo import disable as dynamo_disable
except Exception:  # pragma: no cover - torch._dynamo absent/older torch
    dynamo_graph_break = None
    dynamo_is_compiling = None

    def dynamo_disable(fn):  # type: ignore
        return fn



# Mapping from dtype to element size in bytes; defined at module level and
# shared by _get_weight_norm_linear and _dtype_element_size (used in chunked
# compose budget calculation).
_DTYPE_TO_ELEMENT_SIZE = {torch.float32: 4, torch.float64: 8, torch.float16: 2, torch.bfloat16: 2}


# Lazy-import dora_fused to avoid import-time Triton/LLVM overhead (can be
# 1-2 s) for users who ``import peft`` but don't use DoRA.  The module is
# imported on first use via _get_dora_fused(), which checks sys.modules
# (Dynamo-friendly) and falls back to a locked import on the first call.
_dora_fused_lock = threading.Lock()


def _dtype_element_size(dtype: torch.dtype) -> int:
    cached = _DTYPE_TO_ELEMENT_SIZE.get(dtype)
    if cached is not None:
        return cached
    # Fallback: works for any dtype including integer/quantized types
    # (torch.finfo would throw TypeError on non-floating types).
    return torch.tensor([], dtype=dtype).element_size()


def _promoted_compose_dtype(
    lora_dtype: torch.dtype,
    base_dtype: torch.dtype,
    mag_dtype: torch.dtype,
) -> torch.dtype:
    """Return the 3-way promoted dtype for the DoRA composition expression.

    Under mixed-dtype AMP (e.g. fp32 magnitude with bf16 activations), this is
    the dtype that the stable ``(m-1)*base + m*s*lora`` expression evaluates in
    before being cast back to the activation dtype.
    """
    return torch.promote_types(torch.promote_types(lora_dtype, base_dtype), mag_dtype)


_DORA_FUSED_MODULE_NAME = "peft.tuners.lora.dora_fused"


def _get_dora_fused():
    # Return from sys.modules if already imported — this path is what Dynamo
    # traces through.  Using sys.modules avoids the fragile MODULE_MATCH
    # guard that Dynamo places on module-level globals (which breaks across
    # torch._dynamo.reset() calls in test suites).
    _mod = sys.modules.get(_DORA_FUSED_MODULE_NAME)
    if _mod is not None:
        return _mod
    # First call: import under a lock for thread-safety.  This branch is
    # never hit during Dynamo tracing (the module is imported eagerly by
    # the first real forward pass before any torch.compile call).
    with _dora_fused_lock:
        _mod = sys.modules.get(_DORA_FUSED_MODULE_NAME)
        if _mod is None:
            import peft.tuners.lora.dora_fused  # populates sys.modules

            _mod = sys.modules[_DORA_FUSED_MODULE_NAME]
            is_triton_available.cache_clear()
    return _mod


def fused_dora_compose(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
    inplace: bool = True,
) -> torch.Tensor:
    return _get_dora_fused().fused_dora_compose(lora, base, mag_norm_scale, scale, inplace=inplace)


def fused_norm_assembly(
    w_norm_sq: torch.Tensor,
    cross_term: torch.Tensor,
    ba_norm_sq: torch.Tensor,
    scale: float,
) -> tuple:
    return _get_dora_fused().fused_norm_assembly(w_norm_sq, cross_term, ba_norm_sq, scale)


def fused_dora_compose_autograd(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return _get_dora_fused().fused_dora_compose_autograd(lora, base, mag_norm_scale, scale)


@lru_cache(maxsize=1)
def is_triton_available() -> bool:
    """Return True if Triton is importable, without importing dora_fused.

    Keep this lightweight probe separate from ``dora_fused.is_triton_available``
    to preserve lazy import behavior for users that never execute DoRA paths.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


logger = logging.getLogger(__name__)


_SENTINEL = object()
_cached_use_fused_kernels = _SENTINEL
_cached_use_fused_backward = _SENTINEL
_cached_fused_backward_explicit = _SENTINEL  # None/True/False 3-state
_cached_allow_partial_gather = _SENTINEL
_cached_force_gather_override = _SENTINEL  # None/True/False 3-state
_cached_is_zero3 = _SENTINEL
_fused_cache_lock = threading.Lock()


def _resolve_fused_kernels() -> bool:
    """Evaluate ``PEFT_DORA_FUSED`` from the environment (no lock, no cache)."""
    env = os.environ.get("PEFT_DORA_FUSED")
    if env is not None:
        return env.strip().lower() in ("1", "true")
    return is_triton_available()


def _resolve_fused_backward() -> bool:
    """Evaluate ``PEFT_DORA_FUSED_BACKWARD`` from the environment (no lock, no cache)."""
    env = os.environ.get("PEFT_DORA_FUSED_BACKWARD")
    if env is not None:
        return env.strip().lower() in ("1", "true")
    return True


def _resolve_fused_backward_explicit() -> Optional[bool]:
    """Return True/False if user explicitly set ``PEFT_DORA_FUSED_BACKWARD``, None if unset.

    Cached after first call.  Used by ``_should_use_fused_backward_for_tensor``
    to distinguish "default on → apply auto heuristic" from "explicitly forced
    on → skip heuristic."
    """
    global _cached_fused_backward_explicit  # noqa: PLW0603
    val = _cached_fused_backward_explicit
    if val is not _SENTINEL:
        return val
    # Dynamo can't trace through threading.Lock — return uncached.
    if dynamo_is_compiling is not None and dynamo_is_compiling():
        env = os.environ.get("PEFT_DORA_FUSED_BACKWARD")
        return env.strip().lower() in ("1", "true") if env is not None else None
    with _fused_cache_lock:
        if _cached_fused_backward_explicit is not _SENTINEL:
            return _cached_fused_backward_explicit
        env = os.environ.get("PEFT_DORA_FUSED_BACKWARD")
        if env is not None:
            _cached_fused_backward_explicit = env.strip().lower() in ("1", "true")
        else:
            _cached_fused_backward_explicit = None
        return _cached_fused_backward_explicit


def _resolve_force_gather_override() -> Optional[bool]:
    """Return explicit ``PEFT_FORCE_GATHER`` override, or ``None`` if unset.

    Ternary behavior:
      * unset -> ``None`` (auto-detect ZeRO-3 each call until a ``True`` is found)
      * ``"1"`` / ``"true"`` -> ``True`` (force gathering on)
      * ``"0"`` / ``"false"`` -> ``False`` (force gathering off)

    Like the other PEFT env toggles, any explicitly set non-truthy value is
    treated as ``False`` so the override remains forever-cacheable.
    """
    env = os.environ.get("PEFT_FORCE_GATHER")
    if env is None:
        return None
    return env.strip().lower() in ("1", "true")


def _force_gather_override() -> Optional[bool]:
    """Return cached explicit ``PEFT_FORCE_GATHER`` override, if any."""
    global _cached_force_gather_override  # noqa: PLW0603
    val = _cached_force_gather_override
    if val is not _SENTINEL:
        return val
    if dynamo_is_compiling is not None and dynamo_is_compiling():
        return _resolve_force_gather_override()
    with _fused_cache_lock:
        if _cached_force_gather_override is not _SENTINEL:
            return _cached_force_gather_override
        _cached_force_gather_override = _resolve_force_gather_override()
        return _cached_force_gather_override


def _use_fused_kernels() -> bool:
    """Return True if fused Triton kernels should be used (when available).

    Controlled by env var ``PEFT_DORA_FUSED``:
      * ``"1"`` or ``"true"`` (case-insensitive) → enable
      * ``"0"`` or ``"false"`` → disable
      * unset → enable by default when Triton is available

    The result is cached after the first call so that ``os.environ.get()``
    is not invoked on every forward pass.  A ``threading.Lock`` guards the
    first-write to avoid TOCTOU races in threaded launchers.

    Note:
        This flag enables Triton kernels for inference-style forward paths.
        During training, fused compose uses the custom autograd path by
        default (disable with ``PEFT_DORA_FUSED_BACKWARD=0``).
    """
    global _cached_use_fused_kernels  # noqa: PLW0603
    # Double-checked locking: the first read outside the lock is a non-atomic
    # read of a module-global reference.  Under CPython (with or without GIL)
    # this is safe because pointer-width writes are atomic on all supported
    # platforms.  Under free-threaded Python (PEP 703, 3.13t+) the explicit
    # lock below serializes the first-write; subsequent reads of a fully
    # constructed Python object reference are safe without the lock.  This
    # relies on CPython implementation details (pointer-width atomicity), not
    # language-level guarantees.
    val = _cached_use_fused_kernels
    if val is not _SENTINEL:
        return val
    # Dynamo cannot trace through threading.Lock context managers (it raises
    # ``Unsupported: Unsupported context manager``).  During compilation,
    # tracing is single-threaded so the lock is unnecessary — resolve the
    # env var directly and let Dynamo inline the boolean constant.
    if dynamo_is_compiling is not None and dynamo_is_compiling():
        return _resolve_fused_kernels()
    with _fused_cache_lock:
        # Double-check after acquiring the lock
        if _cached_use_fused_kernels is not _SENTINEL:
            return _cached_use_fused_kernels
        _cached_use_fused_kernels = _resolve_fused_kernels()
        return _cached_use_fused_kernels


def _use_fused_backward() -> bool:
    """Return True if the custom autograd backward path should be used.

    Controlled by env var ``PEFT_DORA_FUSED_BACKWARD``:
      * ``"1"`` or ``"true"`` → enable
      * ``"0"`` or ``"false"`` → disable
      * unset → **enabled by default unconditionally** (the fused backward
        uses PyTorch fallbacks when Triton is unavailable, so Triton is not
        required; set ``PEFT_DORA_FUSED_BACKWARD=0`` to opt out)

    Enabled by default because the fused forward-and-inner kernel eliminates
    the VRAM spike from sequential PyTorch ops, and the frozen-mag path skips
    the ``inner`` allocation entirely when ``mag_norm_scale`` doesn't require
    gradients.  Overhead in the normal (unfrozen) case is exactly 1x
    ``lora``-sized activation per layer (the saved ``inner``).

    Set ``PEFT_DORA_FUSED_BACKWARD=0`` to opt out if VRAM is extremely tight.
    """
    global _cached_use_fused_backward  # noqa: PLW0603
    val = _cached_use_fused_backward
    if val is not _SENTINEL:
        return val
    if dynamo_is_compiling is not None and dynamo_is_compiling():
        return _resolve_fused_backward()
    with _fused_cache_lock:
        if _cached_use_fused_backward is not _SENTINEL:
            return _cached_use_fused_backward
        _cached_use_fused_backward = _resolve_fused_backward()
        return _cached_use_fused_backward


def _invalidate_fused_cache():
    """Reset cached env var results (fused flags + thresholds + FSDP2 detection). Useful for testing."""
    global _cached_use_fused_kernels, _cached_use_fused_backward, _cached_fused_backward_explicit  # noqa: PLW0603
    global _cached_norm_threshold, _cached_fwd_threshold  # noqa: PLW0603
    global _fsdp2_detect_fns, _cached_allow_partial_gather, _cached_force_gather_override, _cached_is_zero3  # noqa: PLW0603
    with _fused_cache_lock:
        _cached_use_fused_kernels = _SENTINEL
        _cached_use_fused_backward = _SENTINEL
        _cached_fused_backward_explicit = _SENTINEL
        _cached_norm_threshold = _SENTINEL
        _cached_fwd_threshold = _SENTINEL
        _cached_allow_partial_gather = _SENTINEL
        _cached_force_gather_override = _SENTINEL
        _cached_is_zero3 = _SENTINEL
        _fsdp2_detect_fns = None
    is_triton_available.cache_clear()


# Cached FSDP2 detection helpers — resolved once on first call to avoid
# repeated try/except import overhead on every forward pass.
_fsdp2_detect_fns = None  # sentinel: None = not yet resolved


def _resolve_fsdp2_detect_fns():
    """Resolve FSDP2 detection functions once and cache the result.

    Returns ``(FSDPState_class, get_module_state_fn)``.
    """
    # Resolve FSDPState class.  The import location has moved across PyTorch
    # releases:
    #   - 2.4–2.9: torch.distributed._composable.fsdp.FSDPState
    #   - 2.10+:   torch.distributed.fsdp._fully_shard._fsdp_state.FSDPState
    # Try both so that detection works across versions.
    import importlib

    fsdp_state_cls = None
    for _modpath in (
        "torch.distributed._composable.fsdp",
        "torch.distributed.fsdp._fully_shard._fsdp_state",
    ):
        try:
            fsdp_state_cls = getattr(importlib.import_module(_modpath), "FSDPState")
            break
        except (ImportError, AttributeError, TypeError):
            continue

    # _get_module_state is the reliable detection function (stable since 2.4).
    try:
        from torch.distributed._composable_state import _get_module_state  # type: ignore[import]

        return (fsdp_state_cls, _get_module_state)
    except (ImportError, AttributeError, TypeError):
        pass

    # Legacy fallback (PyTorch < 2.4, only meaningful when FSDP1 is importable)
    if FSDP is not None:
        try:
            from torch.distributed.fsdp._common_utils import _get_module_fsdp_state  # type: ignore[import]

            return (None, _get_module_fsdp_state)
        except (ImportError, AttributeError, TypeError):
            pass

    return (None, None)


def _is_fsdp2_managed(module) -> bool:
    """Detect whether *module* is wrapped by PyTorch FSDP2 (composable API).

    FSDP2 (``torch.distributed._composable.fsdp.fully_shard``, available since
    PyTorch 2.4) attaches ``FSDPState`` to modules but does **not** wrap them
    with ``FullyShardedDataParallel``, so the FSDP1 ``summon_full_params`` API
    silently no-ops.  We detect FSDP2 by checking for the state object that
    ``fully_shard`` attaches.

    Implementation notes (private API dependencies):
      - ``FSDPState``: the composable-FSDP state class.  Import location moved
        from ``torch.distributed._composable.fsdp`` (2.4–2.9) to
        ``torch.distributed.fsdp._fully_shard._fsdp_state`` (2.10+).
        Both paths are tried by ``_resolve_fsdp2_detect_fns``.
      - ``torch.distributed._composable_state._get_module_state``: stable
        since PyTorch 2.4.  Returns the composable state attached by
        ``fully_shard``.
      - ``torch.distributed.fsdp._common_utils._get_module_fsdp_state``:
        legacy fallback for PyTorch < 2.4.

    Detection functions are resolved once on first call and cached to avoid
    import overhead on every forward pass (hundreds of layers × thousands of
    steps).  All imports are guarded by ``try/except`` so breakage in future
    PyTorch releases degrades to returning ``False`` (FSDP1 behavior preserved).
    Last verified against PyTorch 2.4.0, 2.5.0, 2.6.0, and 2.10.0.
    """
    if not isinstance(module, nn.Module):
        return False

    global _fsdp2_detect_fns  # noqa: PLW0603
    if _fsdp2_detect_fns is None:
        with _fused_cache_lock:
            if _fsdp2_detect_fns is None:
                _fsdp2_detect_fns = _resolve_fsdp2_detect_fns()

    fsdp_state_cls, get_state_fn = _fsdp2_detect_fns

    if get_state_fn is None:
        return False

    try:
        state = get_state_fn(module)
    except (TypeError, AttributeError):
        state = None

    if state is not None:
        if fsdp_state_cls is not None:
            if isinstance(state, fsdp_state_cls):
                return True
        elif FSDP is None or not isinstance(module, FSDP):
            # Has composable state but is not FSDP1-wrapped → FSDP2
            return True

    # Note: we intentionally do NOT check for DTensor parameters here.
    # FSDP2 converts child parameters to DTensor when fully_shard() is
    # called on a parent, so _get_module_state returns None for leaf layers
    # even though their params are sharded.  However, DTensor is also used
    # by Tensor Parallelism and Pipeline Parallelism — checking for DTensor
    # would false-positive on TP-only configs and crash DoRA forward.
    # The primary detection via _get_module_state catches directly-wrapped
    # modules.  Parent-only FSDP2 wrapping is a known detection gap, but
    # in that configuration FSDP2's own pre-forward hooks unshard parameters
    # before DoRA's forward runs, so norms are computed from full params.
    return False


@contextmanager
def _fsdp_full_param_ctx(*modules):
    """
    Best-effort context to expose full parameters when modules are wrapped with
    torch.distributed.fsdp.FullyShardedDataParallel (FSDP).
    - Yields exactly once.
    - No-ops outside FSDP or if modules are not FSDP-wrapped.
    - Does not swallow exceptions raised inside the 'with' body.
    This is safe under ZeRO/DP/DDP (it will simply do nothing).
    Debug logs which modules were successfully summoned (best-effort).

    Callers must only pass ``nn.Module`` instances — raw tensors or
    ``nn.Parameter`` objects (e.g. embedding LoRA factors) are not
    individually FSDP-wrapped and should not be passed here.  Use
    ``_maybe_gather_base_params_ctx`` for those.

    Raises ``RuntimeError`` if any module is managed by FSDP2 (composable API),
    which uses a different full-parameter mechanism that this helper does not
    support.  Failing loudly is preferable to silently computing norms from
    sharded parameters.
    """
    if FSDP is None:
        yield
        return

    # Filter to nn.Module instances only — raw tensors and Parameters should
    # use _maybe_gather_base_params_ctx instead.
    modules = tuple(m for m in modules if m is not None and isinstance(m, nn.Module))
    if not modules:
        yield
        return

    # Detect FSDP2 and fail loudly rather than silently returning shards.
    for m in modules:
        if _is_fsdp2_managed(m):
            raise RuntimeError(
                f"DoRA detected FSDP2-wrapped module ({type(m).__name__}). "
                "The current DoRA implementation only supports FSDP1's "
                "`summon_full_params` API. FSDP2 (composable `fully_shard`) "
                "requires a different full-parameter mechanism that is not yet "
                "implemented. Using FSDP2 with DoRA would silently compute "
                "norms from sharded parameters and produce incorrect results."
            )

    with ExitStack() as stack:
        summoned = 0
        for m in modules:
            try:
                cm = FSDP.summon_full_params(m, writeback=False, with_grads=False)
            except (TypeError, AttributeError):
                # Not FSDP-wrapped or incompatible; skip
                continue
            try:
                stack.enter_context(cm)
                summoned += 1
            except RuntimeError:
                # Some FSDP variants may raise at enter time; skip
                continue
        if summoned:
            logger.debug("DoRA: entered FSDP full-param ctx for %d module(s)", summoned)
        yield


_cached_norm_threshold = _SENTINEL
_cached_fwd_threshold = _SENTINEL
# Crossover thresholds for the auto-enabled fused backward shape heuristic.
# See _should_auto_use_fused_backward_shape() for the benchmark rationale.
_FUSED_BACKWARD_AUTO_MIN_COLS = 2048
_FUSED_BACKWARD_AUTO_MIN_WORK_ITEMS = 2048 * 6144


def _get_norm_memory_threshold_bytes():
    """
    Returns the working-set memory threshold in bytes for the norm computation
    (W_chunk, A_chunk, U, Gram) controlled by env var PEFT_DORA_NORM_CHUNK_MB.
    Default: 256 MB if unset or invalid. Minimum enforced: 16 MB.

    The result is cached after the first call.  Use ``_invalidate_fused_cache``
    or ``_invalidate_threshold_cache`` to re-read the environment.
    """
    global _cached_norm_threshold  # noqa: PLW0603
    val = _cached_norm_threshold
    if val is not _SENTINEL:
        return val
    with _fused_cache_lock:
        if _cached_norm_threshold is not _SENTINEL:
            return _cached_norm_threshold
        default_mb = 256
        env = os.environ.get("PEFT_DORA_NORM_CHUNK_MB")
        mb = default_mb
        if env is not None:
            try:
                mb = max(16, int(env))
            except (ValueError, TypeError):
                mb = default_mb
        _cached_norm_threshold = mb * 1024 * 1024
        return _cached_norm_threshold


def _get_forward_chunk_threshold_bytes():
    """Working-set memory threshold (in bytes) for the forward compose path.

    The result is cached after the first call.  Use ``_invalidate_fused_cache``
    or ``_invalidate_threshold_cache`` to re-read the environment.
    """
    global _cached_fwd_threshold  # noqa: PLW0603
    val = _cached_fwd_threshold
    if val is not _SENTINEL:
        return val
    with _fused_cache_lock:
        if _cached_fwd_threshold is not _SENTINEL:
            return _cached_fwd_threshold
        default_mb = 256
        env = os.environ.get("PEFT_DORA_FWD_CHUNK_MB")
        mb = default_mb
        if env is not None:
            try:
                mb = max(16, int(env))
            except (ValueError, TypeError):
                mb = default_mb
        _cached_fwd_threshold = mb * 1024 * 1024
        return _cached_fwd_threshold


def _invalidate_threshold_cache():
    """Reset cached threshold env var results. Useful for testing."""
    global _cached_norm_threshold, _cached_fwd_threshold  # noqa: PLW0603
    with _fused_cache_lock:
        _cached_norm_threshold = _SENTINEL
        _cached_fwd_threshold = _SENTINEL


def set_dora_norm_threshold_mb(mb: int) -> None:
    """
    Set the PEFT_DORA_NORM_CHUNK_MB environment variable to control the working-set memory threshold (in MB).
    Enforces that mb is an integer >= 16 and <= 65536 (64 GB).
    Raises ValueError if mb is out of bounds.
    """
    min_mb = 16
    max_mb = 65536  # 64 GB, arbitrary upper bound to prevent mistakes
    if not isinstance(mb, int):
        raise ValueError(f"mb must be an integer, got {type(mb).__name__}")
    if not (min_mb <= mb <= max_mb):
        raise ValueError(f"mb must be between {min_mb} and {max_mb} (got {mb})")
    os.environ["PEFT_DORA_NORM_CHUNK_MB"] = str(mb)
    _invalidate_threshold_cache()


def _allow_partial_gather() -> bool:
    """Return True if ``PEFT_DORA_ALLOW_PARTIAL_GATHER=1``.

    Cached after first call, consistent with other env-var accessors.
    """
    global _cached_allow_partial_gather  # noqa: PLW0603
    val = _cached_allow_partial_gather
    if val is not _SENTINEL:
        return val
    # Dynamo cannot trace through threading.Lock — resolve directly during
    # compilation (single-threaded, so the lock is unnecessary).
    if dynamo_is_compiling is not None and dynamo_is_compiling():
        return os.environ.get("PEFT_DORA_ALLOW_PARTIAL_GATHER", "0") == "1"
    with _fused_cache_lock:
        if _cached_allow_partial_gather is not _SENTINEL:
            return _cached_allow_partial_gather
        _cached_allow_partial_gather = os.environ.get("PEFT_DORA_ALLOW_PARTIAL_GATHER", "0") == "1"
        return _cached_allow_partial_gather


def _is_zero3_active() -> bool:
    """Return True when DoRA should gather sharded parameters.

    ``PEFT_FORCE_GATHER`` is a cached ternary override:
      * unset -> auto-detect ZeRO-3
      * ``1`` / ``true`` -> force gather on and cache forever
      * ``0`` / ``false`` -> force gather off and cache forever

    When the override is unset, only detected ``True`` is cached — ZeRO-3 may
    initialize after the first DoRA call (the common HF Trainer flow: create
    model -> PEFT wrap -> deepspeed.initialize), so an auto-detected ``False``
    must still be re-evaluated on later forwards.
    """
    force = _force_gather_override()
    if force is not None:
        return force
    global _cached_is_zero3  # noqa: PLW0603
    if _cached_is_zero3 is True:
        return True
    # No explicit override: don't cache False, re-evaluate late DS init.
    # Skip all DeepSpeed checks when distributed isn't initialized — ZeRO-3
    # can't be active without a process group.  This makes the False path very
    # cheap (~100ns for the is_initialized() boolean) on single-GPU setups
    # with 100+ DoRA layers, avoiding the os.environ.get() on every call.
    is_zero3_ds = False
    try:
        if torch.distributed.is_initialized():
            # Fast path: if the env var is set, trust it immediately.
            if os.environ.get("DS_ZERO_STAGE") == "3":
                with _fused_cache_lock:
                    _cached_is_zero3 = True
                return True
            try:
                is_zero3_ds = check_deepspeed_zero3_enabled()
            except (ImportError, RuntimeError, ValueError):
                pass
            except Exception:
                logger.debug("DoRA: check_deepspeed_zero3_enabled() raised unexpected error", exc_info=True)
    except (RuntimeError, AttributeError):
        pass
    result = is_zero3_ds
    if result:
        with _fused_cache_lock:
            _cached_is_zero3 = True
    return result


def _resolve_tensor_base(tensor: torch.Tensor) -> torch.Tensor:
    """Resolve a view chain back to the underlying ``nn.Parameter``.

    DeepSpeed ``GatheredParameters`` filters by ``hasattr(p, "ds_id")``, which
    is only present on original Parameters, not on views.  Walks the
    ``._base`` chain (handles chained views like ``.T.unsqueeze(0)``) and
    returns the first ``nn.Parameter`` found, or *tensor* unchanged if the
    chain doesn't lead to a Parameter.
    """
    t = tensor
    seen = {id(t)}
    while True:
        base = getattr(t, "_base", None)
        if base is None or id(base) in seen:
            break
        t = base
        if isinstance(t, nn.Parameter):
            return t
        seen.add(id(t))
    return tensor


def _refresh_embedding_lora_view(tensor: torch.Tensor) -> torch.Tensor:
    """Rebuild embedding LoRA ``param.T`` views from gathered Parameters."""
    base = _resolve_tensor_base(tensor)
    if base is tensor or not isinstance(base, nn.Parameter) or base.ndim != 2:
        return tensor
    refreshed = base.T
    if tensor.shape != refreshed.shape:
        return tensor
    return refreshed


def _snapshot_dequantized_weight(module: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """Clone live ``module.weight`` tensors before they escape a gather scope.

    ``dequantize_module_weight`` returns a new dense tensor for quantized
    modules, but for ordinary modules it simply returns ``module.weight``.
    That live Parameter may be re-sharded after the gather context exits, so
    callers that use the tensor later (for ``base_result is None`` paths) need a
    detached snapshot.

    Uses ``data_ptr()`` comparison rather than ``is`` to also catch tensors
    that share storage with ``module.weight`` (e.g. Int8Params dequantization
    may return a view sharing the underlying storage).

    Note: ``data_ptr()`` comparison does not detect non-zero-offset views that
    share the same underlying storage but start at a different offset.  This is
    acceptable because ``dequantize_module_weight`` only returns either
    ``module.weight`` itself or a freshly-allocated dense tensor — never an
    offset view.  If a future quantization backend (bitsandbytes, GPTQ, etc.)
    changes that contract, this function would need ``storage().data_ptr()``
    comparison instead.
    """
    mod_weight = getattr(module, "weight", None)
    if mod_weight is not None and weight.data_ptr() == mod_weight.data_ptr():
        return weight.detach().clone()
    return weight


def _maybe_gather_base_params_ctx(base_layer, *extra_modules):
    """
    Only required under DeepSpeed ZeRO-3 where parameters are sharded. For ZeRO-2, params are
    replicated, so gathering is unnecessary. We gate gathering by:
      - explicit ``PEFT_FORCE_GATHER`` override when set, else
      - DS_ZERO_STAGE==3 (env), or
      - check_deepspeed_zero3_enabled().
    We try param-tuple signature first, else module object; logs which one worked.

    *extra_modules* are additional modules or raw tensors/parameters whose
    parameters should also be gathered (e.g. ``lora_A``, ``lora_B``).  Under
    ZeRO-3 the adapter weights can be sharded too, so every tensor consumed by
    the norm path must be inside the gather scope.

    Items that are ``nn.Module`` contribute via ``.parameters()``.  Items that
    are bare ``torch.Tensor`` / ``nn.Parameter`` (e.g. embedding LoRA factors)
    are included directly in the gather tuple.
    """
    if gather_params_ctx is None or not _is_zero3_active():
        return nullcontext()

    # Collect parameters from all modules (base + extras) into a single tuple.
    # Modules contribute via .parameters(); raw tensors are included directly.
    all_modules = [base_layer] + [m for m in extra_modules if m is not None]
    param_iterable = None
    try:
        params = []
        for mod in all_modules:
            if hasattr(mod, "parameters") and callable(mod.parameters):
                params.extend(mod.parameters())
            elif isinstance(mod, torch.Tensor):
                params.append(_resolve_tensor_base(mod))
        if params:
            param_iterable = tuple(params)
    except TypeError:
        param_iterable = None

    @contextmanager
    def _ctx():
        with ExitStack() as stack:
            entered = False
            if param_iterable is not None:
                try:
                    cm = gather_params_ctx(param_iterable)
                    stack.enter_context(cm)
                    logger.debug("DoRA: ZeRO-3 gather using param tuple (%d params)", len(param_iterable))
                    entered = True
                except (TypeError, AttributeError, RuntimeError) as exc:
                    logger.debug("DoRA: param-tuple gather failed (%s: %s), trying module", type(exc).__name__, exc)
                    entered = False

            if not entered:
                # Fall back to per-module gather.  Track successes and
                # failures separately — a *partial* gather (some modules
                # gathered, others not) is worse than no gather at all
                # because it silently mixes full and sharded tensors.
                gathered_mods = []
                failed_mods = []
                for mod in all_modules:
                    try:
                        # GatheredParameters expects an iterable of Parameters
                        # (or a single Parameter).  Passing an nn.Module directly
                        # makes GatheredParameters a silent no-op (the module
                        # isn't iterable and lacks ds_id).  Always extract params.
                        if isinstance(mod, torch.Tensor):
                            target = (_resolve_tensor_base(mod),)
                        elif hasattr(mod, "parameters") and callable(mod.parameters):
                            target = tuple(mod.parameters())
                            if not target:
                                # Module has no parameters — nothing to gather.
                                gathered_mods.append(type(mod).__name__)
                                continue
                        else:
                            failed_mods.append((type(mod).__name__, "TypeError", "not a Module or Tensor"))
                            continue
                        cm = gather_params_ctx(target)
                        stack.enter_context(cm)
                        gathered_mods.append(type(mod).__name__)
                    except (TypeError, AttributeError, RuntimeError) as exc:
                        failed_mods.append((type(mod).__name__, type(exc).__name__, str(exc)))

                if gathered_mods:
                    entered = True
                    logger.debug("DoRA: ZeRO-3 gather using module objects (%s)", ", ".join(gathered_mods))

                if gathered_mods and failed_mods:
                    # Partial gather: some modules gathered, others failed.
                    # This silently mixes fully gathered and sharded tensors
                    # in the norm computation — a correctness violation.
                    failed_desc = "; ".join(f"{n} ({e}: {m})" for n, e, m in failed_mods)
                    msg = (
                        f"DoRA: ZeRO-3 partial gather — gathered [{', '.join(gathered_mods)}] "
                        f"but failed for [{failed_desc}]. "
                        "Norm computation would mix fully gathered and sharded parameters, "
                        "producing incorrect results."
                    )
                    if _allow_partial_gather():
                        logger.warning(msg + " Continuing due to PEFT_DORA_ALLOW_PARTIAL_GATHER=1.")
                    else:
                        raise RuntimeError(
                            msg + " Set PEFT_DORA_ALLOW_PARTIAL_GATHER=1 to override (at your own risk)."
                        )

            if not entered:
                logger.warning(
                    "DoRA: ZeRO-3 gather failed for all modules. "
                    "Proceeding without gathering — outputs may be incorrect if parameters "
                    "are truly sharded.",
                )
            yield

    return _ctx()


@contextmanager
def _disable_autocast(device_type: str):
    """Disable autocast for the scope if the backend supports it."""

    if autocast is None:
        yield
        return

    # Try to construct the autocast context first; if the device type doesn't
    # support autocast, fall through to a bare yield.  The previous version
    # wrapped both construction AND body in a single try/except, which
    # swallowed body exceptions (including OOM) and triggered
    # "generator didn't stop after throw()".
    try:
        ctx = autocast(device_type=device_type, enabled=False)
    except (TypeError, RuntimeError, ValueError):
        # Device type doesn't support autocast (e.g. cpu on older PyTorch).
        yield
        return

    with ctx:
        yield


def _mag_broadcasts_last_dim(mag: torch.Tensor, out: torch.Tensor) -> bool:
    """Return True when *mag* is a last-dim broadcast vector for *out*."""

    if mag.ndim == 0 or out.ndim == 0:
        return False
    return mag.numel() == out.shape[-1] and mag.shape[-1] == out.shape[-1]


def _should_auto_use_fused_backward_shape(num_rows: int, num_cols: int) -> bool:
    """Benchmark-informed crossover for auto-enabled fused backward.

    ``num_cols`` is the activation's last dimension (``d_out`` for linear
    layers, i.e. ``lora_out.shape[-1]``).  ``num_rows`` is the product of
    all other dimensions (``batch * seq`` for linear layers).

    The warmed 6-GPU benchmark bundle (L40S, A100, RTX 6000 PRO, H200,
    B200, B300) shows the crossover entering the win regime around the
    2048x6144 (rows x cols) activation shape on Blackwell and Ampere;
    L40S and H200 may still trail at the threshold and do not consistently
    win until roughly 2x the threshold work-item count.  The threshold is
    therefore conservative on high-bandwidth HBM GPUs and slightly
    aggressive on lower-bandwidth / older architectures.

    We keep explicit env-var enables as a force-on override; this heuristic
    only applies when PEFT_DORA_FUSED_BACKWARD is unset (the default auto
    mode).
    """

    if num_rows <= 0 or num_cols < _FUSED_BACKWARD_AUTO_MIN_COLS:
        return False
    return num_rows * num_cols >= _FUSED_BACKWARD_AUTO_MIN_WORK_ITEMS


def _should_use_fused_backward_for_tensor(
    lora_out: torch.Tensor,
    mag_norm_scale: Optional[torch.Tensor] = None,
) -> bool:
    """Decide whether training-time compose should route through fused backward."""

    if not (_use_fused_backward() and lora_out.is_cuda):
        return False

    # _use_fused_backward() returns True both for "unset (default on)" and
    # "explicitly set to 1".  We need to distinguish: explicit opt-in skips
    # the auto crossover heuristic, while unset defers to shape analysis.
    # _resolve_fused_backward_explicit() returns True/False for explicit
    # setting, None for unset — all cached.
    explicit = _resolve_fused_backward_explicit()
    if explicit is not None:
        return explicit

    if lora_out.ndim == 0:
        return False

    # Apply the auto heuristic only to the linear/embedding-style broadcast
    # pattern that the Triton benchmark suite covers directly.  For other
    # layouts (for example Conv with mag=[1,C,1,1]), preserve the previous
    # custom-autograd behavior.
    if mag_norm_scale is not None and _mag_broadcasts_last_dim(mag_norm_scale, lora_out):
        num_cols = lora_out.shape[-1]
        num_rows = lora_out.numel() // max(num_cols, 1)
        return _should_auto_use_fused_backward_shape(num_rows, num_cols)

    return True


def _compose_eager_inplace(
    lora: torch.Tensor,
    base: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Numerically stable in-place DoRA composition for eager PyTorch paths.

    Computes ``out = (mag - 1) * base + mag * (scale * lora)`` in-place into
    *lora*, avoiding catastrophic cancellation when ``mag ≈ 1`` in bf16/fp16.

    When all operands already match ``lora.dtype``, the in-place path uses
    ``lora *= scale`` then ``lora *= mag`` then ``lora += (mag - 1) * base``.
    The two-step multiply preserves the canonical associativity
    ``mag * (scale * lora)`` (scale first, then mag).

    Under mixed dtypes (for example fp32 magnitude with bf16 activations under
    AMP), eager training defines the reference contract: evaluate the stable
    form in the promoted dtype, then cast back to the activation dtype.  The
    in-place helper mirrors that by materializing the promoted result and
    copying it back into ``lora``.  This restores bitwise parity across eager
    out-of-place, eager in-place, and chunked eager composition, but it does
    not by itself change the separate fused-autograd dtype contract in
    ``_compose_with_dispatch``.

    This is the single source of truth for the eager in-place formula.
    The Triton kernel in ``dora_fused.py`` computes the same expression
    but is maintained separately because kernel code cannot share Python helpers.
    See ``test_compose_formula_cross_reference`` for the consistency assertion.
    """
    if _promoted_compose_dtype(lora.dtype, base.dtype, mag_norm_scale.dtype) != lora.dtype:
        result = mag_norm_scale * (scale * lora) + (mag_norm_scale - 1) * base
        lora.copy_(result)
        return lora

    # Step 1: lora = scale * lora  (in-place, canonical order: scale first)
    lora.mul_(scale)
    # Step 2: lora = mag * lora  (in-place, canonical order: mag second)
    lora.mul_(mag_norm_scale)
    # Step 3: lora += (mag - 1) * base  (in-place, adds the base correction)
    lora.add_(base * (mag_norm_scale - 1))
    return lora


class DoraLinearLayer(nn.Module):
    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out
        self._last_chunk_size: Optional[int] = None
        self._last_forward_chunk_size: Optional[int] = None

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        weight = transpose(weight, self.fan_in_fan_out)
        compute_dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        weight_comp = weight.to(dtype=compute_dtype)
        lora_weight_comp = lora_weight.to(device=weight.device, dtype=compute_dtype)

        total = weight_comp + scaling * lora_weight_comp
        weight_norm = torch.linalg.vector_norm(total, dim=1)
        if weight_norm.dtype != weight.dtype:
            weight_norm = weight_norm.to(dtype=weight.dtype)
        return weight_norm

    @torch.no_grad()
    def _get_weight_norm_linear(
        self,
        *,
        base_weight: torch.Tensor,
        lora_A_w: torch.Tensor,
        lora_B_w: torch.Tensor,
        scaling: float,
        chunk_size: Optional[int] = None,
    ):
        """
        Compute ||W + s·(B A)||_row-wise without materializing B A.

        We use:
          ||W + s·(B A)||^2 = ||W||^2 + 2 s ⟨W, B A⟩ + s^2 ||B A||^2
        Let U := W A^T  (shape: [out, r])  and  G := A A^T  (shape: [r, r]).
        Then:
          ⟨W, B A⟩ per-row equals (B ⊙ U).sum(dim=1), since U_jk = ⟨W_j, A_k⟩ and (B A)_j = Σ_k B_jk A_k.
          ||B A||^2 per-row equals (B G ⊙ B).sum(dim=1).

        This avoids constructing the dense [out, in] product B A, reduces memory,
        and allows chunking along 'in' to cap working set size.

        Returns:
            weight_norm: tensor of shape [out_features]. Magnitude division is
            always done by the caller in PyTorch, ensuring identical precision
            regardless of whether Triton kernels are used.
        """

        W_t = transpose(base_weight, self.fan_in_fan_out)
        device = W_t.device
        dtype = W_t.dtype

        """
        Compute all norms in float32 for numerical stability when weights are bf16/fp16.
        We disable autocast locally to prevent the backend from downcasting matmuls.
        """
        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        if compute_dtype not in _DTYPE_TO_ELEMENT_SIZE:
            raise ValueError(
                f"Unsupported compute_dtype {compute_dtype} for DoRA norm computation. "
                f"Expected one of {set(_DTYPE_TO_ELEMENT_SIZE.keys())}."
            )
        element_size = _DTYPE_TO_ELEMENT_SIZE[compute_dtype]

        out_features, in_features = W_t.shape
        if chunk_size is None:
            memory_threshold = _get_norm_memory_threshold_bytes()
            # Include all live buffers: W_chunk (out*chunk), A_chunk (r*chunk),
            # plus once-per-call U (out*r) and Gram (r*r), all in compute_dtype.
            rank = lora_A_w.shape[0]
            full_bytes = ((out_features + rank) * in_features + (out_features * rank) + (rank * rank)) * element_size
            if full_bytes <= memory_threshold:
                chunk_size = in_features
            else:
                const_bytes = (out_features * rank + rank * rank) * element_size
                denom = (out_features + rank) * element_size
                if memory_threshold > const_bytes:
                    raw = (memory_threshold - const_bytes) // max(denom, 1)
                    chunk_size = int(max(1, min(in_features, raw)))
                else:
                    chunk_size = 1
                # Align for accelerator tensor core kernels when possible
                if W_t.device.type in ("cuda", "xpu") and chunk_size > 64:
                    chunk_size = (chunk_size // 64) * 64
        else:
            chunk_size = max(1, min(in_features, chunk_size))

        self._last_chunk_size = chunk_size
        logger.debug(
            "DoRA: chunk_size=%d (out=%d, in=%d, rank=%d, thresholdMB=%.1f)",
            chunk_size,
            out_features,
            in_features,
            lora_A_w.shape[0],
            _get_norm_memory_threshold_bytes() / (1024 * 1024),
        )

        scale_value = float(scaling)
        scale_is_zero = scale_value == 0.0

        w_norm_sq = torch.zeros(out_features, device=device, dtype=compute_dtype)
        rank = lora_A_w.shape[0]
        U = None if scale_is_zero else torch.zeros(out_features, rank, device=device, dtype=compute_dtype)
        gram = None if scale_is_zero else torch.zeros(rank, rank, device=device, dtype=compute_dtype)

        for start in range(0, in_features, chunk_size):
            end = min(start + chunk_size, in_features)
            W_chunk = W_t[:, start:end]
            W_chunk = W_chunk.to(dtype=compute_dtype)

            w_norm_sq += (W_chunk * W_chunk).sum(dim=1)

            if scale_is_zero:
                continue

            A_chunk = lora_A_w[:, start:end]
            A_chunk = A_chunk.to(device=device, dtype=compute_dtype)

            U.addmm_(W_chunk, A_chunk.transpose(0, 1))
            gram.addmm_(A_chunk, A_chunk.transpose(0, 1))

        if scale_is_zero:
            norm_sq = w_norm_sq
            norm_sq = norm_sq.clamp_min_(0)  # in-place safe: function runs under @torch.no_grad()
            weight_norm = torch.sqrt(norm_sq)
        else:
            B_comp = lora_B_w.to(device=device, dtype=compute_dtype)
            cross_term = (B_comp * U).sum(dim=1)
            BA = B_comp @ gram
            ba_norm_sq = (BA * B_comp).sum(dim=1)

            if _use_fused_kernels() and w_norm_sq.is_cuda:
                (weight_norm,) = fused_norm_assembly(
                    w_norm_sq,
                    cross_term,
                    ba_norm_sq,
                    scale_value,
                )
            else:
                # Use Python float directly — PyTorch handles scalar-tensor
                # promotion natively, avoiding a tiny CUDA alloc per call.
                norm_sq = w_norm_sq + (2.0 * scale_value) * cross_term + (scale_value * scale_value) * ba_norm_sq
                norm_sq = norm_sq.clamp_min_(0)  # in-place safe: function runs under @torch.no_grad()
                weight_norm = torch.sqrt(norm_sq)

        if weight_norm.dtype != dtype:
            weight_norm = weight_norm.to(dtype=dtype)

        return weight_norm

    @dynamo_disable
    def _compose_with_base_chunks(
        self,
        *,
        x: torch.Tensor,
        lora_result: torch.Tensor,
        base_weight_t: torch.Tensor,
        mag_norm_scale: torch.Tensor,
        scale: float,
    ) -> None:
        """Compose DoRA output chunk-wise to cap peak memory.

        Recomputes ``base_result`` from ``x @ base_weight_t`` in chunks,
        applying the stable composition form per chunk so that only one
        chunk's worth of temporaries is live at a time.
        """

        if dynamo_graph_break is not None and dynamo_is_compiling is not None and dynamo_is_compiling():
            # The loop below depends on Python control flow and small-integer guards that
            # frequently change across layers; let Dynamo drop to eager to avoid runaway
            # recompilations when torch.compile is enabled.
            dynamo_graph_break()

        out_features = base_weight_t.shape[0]
        if out_features == 0:
            self._last_forward_chunk_size = 0
            return

        # Number of rows in the linear output (product of non-feature dims)
        prefix_rows = lora_result.numel() // out_features
        needs_grad = lora_result.requires_grad or mag_norm_scale.requires_grad or x.requires_grad
        use_fused = _use_fused_kernels() and lora_result.is_cuda and not needs_grad
        threshold = _get_forward_chunk_threshold_bytes()

        # The eager mixed-dtype path materializes the stable compose result in
        # the promoted dtype before copying it back into ``lora_result``. Budget
        # chunking against that wider temporary so AMP eager chunking does not
        # assume peak memory still scales only with the activation dtype.
        if use_fused:
            working_element_size = lora_result.element_size()
        else:
            compose_dtype = _promoted_compose_dtype(lora_result.dtype, lora_result.dtype, mag_norm_scale.dtype)
            working_element_size = _dtype_element_size(compose_dtype)

        if prefix_rows == 0:
            chunk_size = out_features
        else:
            denom = prefix_rows * max(working_element_size, 1)
            if denom == 0:
                chunk_size = out_features
            else:
                capacity = threshold // denom
                if capacity <= 0:
                    chunk_size = 1
                else:
                    chunk_size = int(min(out_features, capacity))

        if chunk_size <= 0:
            chunk_size = 1

        device_type = lora_result.device.type
        if device_type in ("cuda", "xpu") and chunk_size > 64:
            aligned = (chunk_size // 64) * 64
            chunk_size = max(64, aligned)
            chunk_size = min(chunk_size, out_features)

        self._last_forward_chunk_size = chunk_size
        logger.debug(
            "DoRA: forward chunk_size=%d (rows=%d, out=%d, thresholdMB=%.1f)",
            chunk_size,
            prefix_rows,
            out_features,
            threshold / (1024 * 1024),
        )

        # NOTE: chunked composition mutates lora_result slices in-place, so it
        # cannot use fused_dora_compose_autograd (which saves ``inner`` for
        # backward). Fall back to eager PyTorch compose when grads are needed.
        if _use_fused_backward() and needs_grad:
            logger.debug(
                "DoRA: chunked compose falling back to eager path because "
                "fused backward is incompatible with in-place chunk mutation."
            )

        # Cast mag only for the Triton inference path (use_fused requires
        # not needs_grad).  The eager chunk path keeps fp32 mag so mixed-dtype
        # eager chunks follow the same promoted reference as eager training.
        if use_fused and mag_norm_scale.dtype != lora_result.dtype:
            mag_norm_scale = mag_norm_scale.to(lora_result.dtype)

        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            base_slice = F.linear(x, base_weight_t[start:end, :])
            chunk = lora_result[..., start:end]
            mag_chunk = mag_norm_scale[..., start:end]
            if use_fused:
                fused_dora_compose(chunk, base_slice, mag_chunk, scale, inplace=True)
            else:
                # INVARIANT: In-place mutation of ``chunk`` (a view of ``lora_result``)
                # is safe here because ``lora_result`` is a non-leaf intermediate
                # (produced by ``lora_B(lora_A(x))``).  PyTorch allows in-place ops
                # on non-leaf intermediates whose own data is not needed by any other
                # autograd node.  The only grad-requiring tensor whose value propagates
                # through this in-place op is ``mag_norm_scale`` (via multiplication),
                # and its gradient only needs the *result* of the in-place op, not the
                # original ``lora_result`` value.  If this invariant changes (e.g. if
                # ``lora_result`` becomes a leaf or is reused elsewhere), this in-place
                # path will raise an autograd error at backward time.
                _compose_eager_inplace(chunk, base_slice, mag_chunk, scale)

    def _compose_with_dispatch(
        self,
        *,
        lora_out: torch.Tensor,
        base_result: torch.Tensor,
        mag_norm_scale: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Compose DoRA output via fused-backward, fused-forward, or eager fallback.

        Forward-only fused compose is inference-only (no autograd graph nodes).
        Training-time fused composition uses the custom autograd path by default;
        disable with ``PEFT_DORA_FUSED_BACKWARD=0``.

        This method is compile-friendly: all branches depend on tensor metadata
        and cached env-var booleans, not on tensor data.  Dynamo can guard on
        these without graph breaks.
        """
        # base_result may require gradients even when LoRA/magnitude are frozen.
        needs_grad = lora_out.requires_grad or base_result.requires_grad or mag_norm_scale.requires_grad

        # Under AMP, mag_norm_scale is fp32 (computed under _disable_autocast)
        # while activations are bf16/fp16.  The fused Triton paths still need
        # homogeneous dtypes; the eager paths keep fp32 mag so the stable form
        # computes (g-1) in fp32, preserving small corrections that bf16 would
        # round to zero.  Mixed-dtype eager-vs-fused parity therefore remains a
        # separate open issue until the fused-autograd path adopts the same
        # dtype contract.
        if mag_norm_scale.dtype != lora_out.dtype:
            mag_norm_scale_cast = mag_norm_scale.to(lora_out.dtype)
        else:
            mag_norm_scale_cast = mag_norm_scale

        if needs_grad and _should_use_fused_backward_for_tensor(lora_out, mag_norm_scale_cast):
            # Fused autograd path — keeps the historical homogeneous-dtype
            # contract for now, so mixed-dtype AMP can still differ from eager.
            return fused_dora_compose_autograd(
                lora_out,
                base_result,
                mag_norm_scale_cast,
                scale,
            )

        if _use_fused_kernels() and lora_out.is_cuda and not needs_grad:
            # Forward-only fused path — no autograd nodes, inference only.
            return fused_dora_compose(
                lora_out,
                base_result,
                mag_norm_scale_cast,
                scale,
                inplace=True,
            )

        if needs_grad:
            # Eager training: keep fp32 mag for (g-1) precision, cast result
            # to activation dtype to prevent fp32 activation memory bloat.
            #
            # VRAM note: when mag is fp32 and activations are bf16 (AMP),
            # PyTorch type-promotion creates transient fp32 intermediates
            # of size [batch*seq, out_features].  Peak is ~2× the activation
            # size at fp32 (4× vs bf16).  This is the non-chunked path
            # (base_result precomputed), so the chunk budget does not bound
            # it.  The fused autograd path (default) avoids this by casting
            # mag to activation dtype first; this path trades higher
            # transient VRAM for fp32 (g-1) precision.  Set
            # PEFT_DORA_FUSED_BACKWARD=1 (default) to avoid this path.
            result = mag_norm_scale * (scale * lora_out) + (mag_norm_scale - 1) * base_result
            if result.dtype != lora_out.dtype:
                result = result.to(lora_out.dtype)
            return result

        # Eager inference in-place: fp32 mag is fine — in-place ops (mul_,
        # add_) truncate to lora_out's dtype at each step, and (g-1) is
        # computed in fp32 before the final truncation.
        return _compose_eager_inplace(lora_out, base_result, mag_norm_scale, scale)

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        # Include lora_A/lora_B in the gather scope — under ZeRO-3 the adapter
        # parameters can also be sharded (e.g. mid-training adapter swaps).
        with _maybe_gather_base_params_ctx(base_layer, lora_A, lora_B):
            if base_layer.__class__.__name__ == "Linear4bit":
                # We have to create a copy of the base layer, otherwise, FSDP will throw an error. 8bit does not work
                # yet because Int8Params cannot be correctly deep-copied (attributes vanish)
                base_layer = deepcopy(base_layer)

            weight = dequantize_module_weight(base_layer)
            weight = weight.to(lora_A.device)
            if weight.data.ndim >= 3:  # For handling LoRAs applied to Conv layers.
                weight_norm = self._get_weight_norm_conv_factored(
                    base_weight=weight,
                    lora_A_w=lora_A,
                    lora_B_w=lora_B,
                    scaling=scaling,
                )
            else:
                weight_norm = self._get_weight_norm_linear(
                    base_weight=weight,
                    lora_A_w=lora_A,
                    lora_B_w=lora_B,
                    scaling=scaling,
                )

            if dtype_is_fp16:
                weight_norm = weight_norm.half()

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        Norm path runs under no_grad in fp32 and is detached (DoRA §4.3).

        Compile-friendly: the ``base_result is not None`` path (common case)
        is fully traceable by Dynamo.  The ``base_result is None`` path
        routes through ``_compose_with_base_chunks`` which has
        ``@dynamo_disable`` due to its data-dependent chunk loop.
        """
        # Compute weight_norm in a memory-efficient way without materializing full lora_weight.
        magnitude = self.weight
        device_type = base_layer.weight.device.type
        base_weight = None
        # _fsdp_full_param_ctx receives lora_A/lora_B alongside base_layer
        # because they are nn.Module instances (Linear sub-layers of the LoRA
        # adapter) that FSDP1 can individually wrap — summon_full_params needs
        # the module handle to unshard their parameters.  In contrast, the
        # embedding path passes only base_layer to _fsdp_full_param_ctx because
        # its lora_A/lora_B are raw tensors (transposed nn.Parameters from a
        # ParameterDict), not nn.Module instances, so summon_full_params cannot
        # act on them.  The embedding path gathers those raw tensors via
        # _maybe_gather_base_params_ctx instead.
        with (
            torch.no_grad(),
            _maybe_gather_base_params_ctx(base_layer, lora_A, lora_B),
            _fsdp_full_param_ctx(base_layer, lora_A, lora_B),
            _disable_autocast(device_type),
        ):
            base_weight = dequantize_module_weight(base_layer)
            # Norm reads weight in-place — no clone needed yet.
            weight_norm = self._get_weight_norm_linear(
                base_weight=base_weight,
                lora_A_w=lora_A.weight,
                lora_B_w=lora_B.weight,
                scaling=scaling,
            )
            # Snapshot AFTER norm computation: the base_result=None path needs
            # the weight to survive the gather scope, but deferring the clone
            # avoids a peak-VRAM spike during the norm computation (which already
            # allocates chunked intermediates).  For large MoE layers (32K×128K)
            # this halves the transient allocation inside the gather scope.
            if base_result is None:
                base_weight = _snapshot_dequantized_weight(base_layer, base_weight)
            # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
            # Computed under ``no_grad`` so the norm stays constant during backpropagation.
        # Division always in PyTorch — identical precision regardless of Triton availability.
        # Ensure weight_norm is on the same device as magnitude (CPU-offloaded
        # gathers can leave weight_norm on CPU while magnitude lives on GPU).
        if weight_norm.device != magnitude.device:
            weight_norm = weight_norm.to(device=magnitude.device)
        if weight_norm.is_floating_point():
            # eps depends on weight_norm's dtype: bf16/fp16 can't represent
            # 1e-12, so use 1e-6 to prevent overflow after the m/||W|| cast.
            eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
            weight_norm = weight_norm.clamp_min(eps)
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        # Compute LoRA output and compose result with minimal temporaries
        lora_result = lora_B(lora_A(x))
        scale = scaling

        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                # Move bias to base_result's device (CPU-offloaded base_layer
                # keeps bias on CPU while base_result lives on GPU), and cast
                # to base_result dtype to avoid fp32 type promotion under AMP
                # (where base_result is bf16 but bias is fp32).  Without the
                # dtype cast, base_result would be promoted to fp32, defeating
                # Triton dispatch in _compose_with_dispatch.
                #
                # Assumption: base_result is a standard floating dtype (bf16,
                # fp16, fp32) from F.linear under autocast.  If a quantized
                # base layer produces int8/fp8 base_result, this cast would
                # silently convert the bias to a quantized type.
                if bias.device != base_result.device or bias.dtype != base_result.dtype:
                    bias = bias.to(device=base_result.device, dtype=base_result.dtype)
                base_result = base_result - bias
            # Release dequantized base weight early on the common base_result path.
            base_weight = None
            self._last_forward_chunk_size = None
            return self._compose_with_dispatch(
                lora_out=lora_result,
                base_result=base_result,
                mag_norm_scale=mag_norm_scale,
                scale=scale,
            )

        # Note: this creates a full copy of the base weight on GPU. For large
        # layers this is a non-trivial allocation, but inherently unavoidable
        # when base_result is not precomputed. The common base_result path
        # (above) avoids this allocation entirely.
        if base_weight.device != x.device or base_weight.dtype != x.dtype:
            base_weight = base_weight.to(device=x.device, dtype=x.dtype)
        base_weight_t = transpose(base_weight, self.fan_in_fan_out)
        self._compose_with_base_chunks(
            x=x,
            lora_result=lora_result,
            base_weight_t=base_weight_t,
            mag_norm_scale=mag_norm_scale,
            scale=scale,
        )
        return lora_result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class DoraEmbeddingLayer(DoraLinearLayer):
    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, embed_fn, base_result=None):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        magnitude = self.weight
        device_type = base_layer.weight.device.type
        # lora_A/lora_B are raw tensors (transposed nn.Parameters from the
        # parent Embedding module's ParameterDict), not nn.Module instances.
        # _maybe_gather_base_params_ctx handles raw tensors directly.
        #
        # _fsdp_full_param_ctx only receives base_layer (not lora_A/lora_B)
        # because FSDP1 wraps nn.Module instances — raw tensors stored in a
        # ParameterDict are not individually FSDP-wrapped, so summon_full_params
        # wouldn't apply to them.  The FSDP2 detection check runs on
        # base_layer only via _get_module_state.  If FSDP2 wraps only a
        # parent module (not base_layer directly), detection may miss it —
        # but in that case FSDP2's pre-forward hooks unshard parameters
        # before this forward runs, so norms see full weights.
        with _maybe_gather_base_params_ctx(base_layer, lora_A, lora_B), _fsdp_full_param_ctx(base_layer):
            gathered_lora_A = _refresh_embedding_lora_view(lora_A)
            gathered_lora_B = _refresh_embedding_lora_view(lora_B)
            if _is_zero3_active():
                # Clone while the full tensors are materialized, then use those
                # stable clones after the gather scope exits.  CloneBackward
                # still routes gradients back to the original Parameters.
                # Note: the clones double the transient LoRA allocation inside
                # the gather scope (e.g. ~64 MB per rank-64 256K-vocab adapter
                # in bf16).  This is inherent — the originals must stay alive
                # for GatheredParameters cleanup.
                lora_A_forward = gathered_lora_A.clone()
                lora_B_forward = gathered_lora_B.clone()
            else:
                lora_A_forward = gathered_lora_A
                lora_B_forward = gathered_lora_B
            with torch.no_grad(), _disable_autocast(device_type):
                # Build the dense embedding delta under no_grad so the norm path
                # matches linear/conv detach semantics and does not allocate an
                # unnecessary autograd graph for the temporary product.
                # Note: this materializes the full [num_embeddings, embedding_dim]
                # LoRA delta — O(V×d) allocation.  Unlike linear/conv, there is no
                # factored norm path for embeddings yet.  Fine for typical vocab
                # sizes, but worth being aware of at 256k+ tokens.
                #
                # VRAM budget: peak inside this scope is the sum of:
                #   1. lora_A/B clones (ZeRO-3 path): 2 × rank × dim × elem_size
                #   2. lora_weight below: num_embeddings × embedding_dim × elem_size
                #   3. gathered base weight (dequantize_module_weight): num_embeddings × embedding_dim × elem_size
                # For a 256K-vocab, 4096-dim, rank-64 adapter in bf16 this is
                # ~2 GB transient.  The clones are unavoidable under ZeRO-3
                # (originals must stay alive for GatheredParameters cleanup).
                lora_weight = (lora_A_forward @ lora_B_forward).T
                weight = dequantize_module_weight(base_layer)
                weight_dtype = weight.dtype
                weight_norm = self.get_weight_norm(weight, lora_weight, scaling)
                # Defer snapshot to after norm — see linear forward for rationale.
                if base_result is None:
                    weight = _snapshot_dequantized_weight(base_layer, weight)
        # see section 4.3 of DoRA (https://huggingface.co/papers/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        # weight_norm is already detached: torch.no_grad() above prevents graph
        # construction for the entire norm computation block, matching the linear
        # path which relies on no_grad() alone without a separate .detach().
        # Ensure weight_norm is on the same device as magnitude (CPU-offloaded
        # gathers can leave weight_norm on CPU while magnitude lives on GPU).
        if weight_norm.device != magnitude.device:
            weight_norm = weight_norm.to(device=magnitude.device)
        if weight_norm.is_floating_point():
            eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
            weight_norm = weight_norm.clamp_min(eps)
        mag_norm_scale = magnitude / weight_norm
        if base_result is None:
            # Ensure weight is on the execution device (CPU-offloaded gathers
            # may return weights still on CPU while x lives on GPU).
            # Note: allocates a full copy of the embedding matrix on GPU;
            # the common base_result path avoids this entirely.
            # Only transfer device — x.dtype is Long (token indices), not a
            # floating-point type suitable for the weight matrix.
            if weight.device != x.device:
                weight = weight.to(device=x.device)
            base_result = embed_fn(x, weight)
        # Route through _compose_with_dispatch so embedding layers benefit
        # from fused Triton kernels when available (same path as linear layers).
        lora_out = embed_fn(x, lora_A_forward) @ lora_B_forward
        result_dora = self._compose_with_dispatch(
            lora_out=lora_out,
            base_result=base_result,
            mag_norm_scale=mag_norm_scale,
            scale=scaling,
        )
        return mag_norm_scale, result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class _DoraConvNdLayer(DoraLinearLayer):
    def _get_weight_norm_conv_factored(
        self,
        *,
        base_weight: torch.Tensor,
        lora_A_w: torch.Tensor,
        lora_B_w: torch.Tensor,
        scaling: float,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        out_channels = base_weight.shape[0]
        flat_weight = base_weight.reshape(out_channels, -1)
        rank = lora_A_w.shape[0]
        flat_A = lora_A_w.reshape(rank, -1)
        flat_B = lora_B_w.reshape(out_channels, -1)

        # Handle grouped convolutions by expanding B across group-specific rank bands
        # lora_A_w has shape [rank, in_per_group, kH, kW]
        # lora_B_w has shape [out_channels, rank_per_group, 1, 1] when groups>1
        rank = lora_A_w.shape[0]
        rank_per_group = flat_B.shape[1]
        if rank_per_group > 0 and rank % rank_per_group == 0:
            groups = rank // rank_per_group
            if groups > 1 and (out_channels % groups == 0):
                out_per_group = out_channels // groups
                B_expanded = flat_B.new_zeros((out_channels, rank))
                for g in range(groups):
                    rows = slice(g * out_per_group, (g + 1) * out_per_group)
                    cols = slice(g * rank_per_group, (g + 1) * rank_per_group)
                    B_expanded[rows, cols] = flat_B[rows, :]
                flat_B = B_expanded

        norms = self._get_weight_norm_linear(
            base_weight=flat_weight,
            lora_A_w=flat_A,
            lora_B_w=flat_B,
            scaling=scaling,
            chunk_size=chunk_size,
        )

        view_shape = (1, out_channels) + (1,) * max(0, base_weight.dim() - 2)
        return norms.view(view_shape)

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        compute_dtype = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        weight_comp = weight.to(dtype=compute_dtype)
        lora_weight_comp = lora_weight.to(device=weight.device, dtype=compute_dtype)

        total = weight_comp + scaling * lora_weight_comp
        # the following is needed to have compatibility with the 4/5D weight tensors of Conv2D/3D
        dim = tuple(range(1, weight.dim()))
        weight_norm = total.norm(p=2, dim=dim, keepdim=True).transpose(1, 0)
        if weight_norm.dtype != weight.dtype:
            weight_norm = weight_norm.to(dtype=weight.dtype)
        return weight_norm

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        Norm path runs under no_grad in fp32 and is detached (DoRA §4.3).
        """
        magnitude = self.weight
        device_type = base_layer.weight.device.type
        # See linear forward for FSDP1 no-op note on passing lora_A/lora_B.
        with (
            torch.no_grad(),
            _maybe_gather_base_params_ctx(base_layer, lora_A, lora_B),
            _fsdp_full_param_ctx(base_layer, lora_A, lora_B),
            _disable_autocast(device_type),
        ):
            weight = dequantize_module_weight(base_layer)
            weight_norm = self._get_weight_norm_conv_factored(
                base_weight=weight,
                lora_A_w=lora_A.weight,
                lora_B_w=lora_B.weight,
                scaling=scaling,
            )
            # Defer snapshot to after norm — see linear forward for rationale.
            if base_result is None:
                weight = _snapshot_dequantized_weight(base_layer, weight)
        # weight_norm is a derived tensor (from norm computation), not a view
        # of the (re-shardable) parameter — safe to use outside the gather scope.
        # Only transfer device when needed (CPU-offloaded gathers can leave
        # weight_norm on CPU while magnitude lives on GPU).
        if weight_norm.device != magnitude.device:
            weight_norm = weight_norm.to(device=magnitude.device)
        if weight_norm.is_floating_point():
            eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
            weight_norm = weight_norm.clamp_min(eps)
        mag_norm_scale = magnitude / weight_norm

        if base_result is None:
            # Ensure weight is on the execution device (CPU-offloaded gathers
            # may return weights still on CPU while x lives on GPU).
            # Note: allocates a full copy of the conv weight on GPU;
            # the common base_result path avoids this entirely.
            if weight.device != x.device or weight.dtype != x.dtype:
                weight = weight.to(device=x.device, dtype=x.dtype)
            base_result = self.conv_fn(
                x,
                weight,
                bias=None,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        else:
            bias = base_layer.bias
            if bias is not None:
                # Move bias to base_result's device and dtype (CPU-offloaded
                # base_layer keeps bias on CPU; AMP keeps bias in fp32 while
                # base_result is bf16).
                if bias.device != base_result.device or bias.dtype != base_result.dtype:
                    bias = bias.to(device=base_result.device, dtype=base_result.dtype)
                # reshape bias to (1, -1, 1, ...)
                bias_shape = (1, -1) + (1,) * (base_result.dim() - 2)
                base_result = base_result - bias.view(*bias_shape)

        lora_out = lora_B(lora_A(x))
        return self._compose_with_dispatch(
            lora_out=lora_out,
            base_result=base_result,
            mag_norm_scale=mag_norm_scale,
            scale=scaling,
        )

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class DoraConv1dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv1d


class DoraConv2dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv2d


class DoraConv3dLayer(_DoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
        self.conv_fn = F.conv3d


# Public helpers for ergonomics
def get_dora_norm_threshold_mb() -> int:
    """Return the current DoRA norm chunk threshold in MB."""
    return int(_get_norm_memory_threshold_bytes() // (1024 * 1024))


def get_dora_norm_threshold_bytes() -> int:
    """Return the current DoRA norm chunk threshold in bytes."""
    return int(_get_norm_memory_threshold_bytes())
