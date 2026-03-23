#!/usr/bin/env python3
"""Comprehensive DoRA benchmark suite.

Sections:
  1. Norm      -- Four-axis: PEFTeye vs Reference (dense B@A) vs Factored vs Factored+Fused
  2. Compose   -- Fused Triton vs Eager PyTorch element-wise composition
  3. Backward  -- Fused vs Standard autograd backward
  4. E2E       -- Single DoRA layer, 4 configurations (rank sweep + batch×seq sweep)
  5. Memory    -- Same as E2E but detailed memory profiling
  6. Models    -- Real HuggingFace model evaluation (opt-in)
  7. Stability -- Catastrophic cancellation demo (bf16 vs fp64, mag near 1)
  8. Precision -- d_mag precision sweep (factored vs dense, per-dtype)
  9. VRAM      -- VRAM impact: eager vs fused-inner compose (fwd+bwd)
 10. Micro     -- Targeted Triton config probes for anomaly windows / boundary scans / row buckets

Usage (from the repo root):
  python code/bench_dora_comprehensive.py [OPTIONS]
    --suite {all,norm,compose,backward,e2e,memory,stability,models,precision,vram,micro}
    --dtype {fp32,bf16,fp16}
    --shapes {default,extended}
    --repeats N  --warmup N
    --json-out PATH  --verbose
    --models MODEL_ID [MODEL_ID ...]
    --rank INT   --batch INT   --seqlen INT
    --loss-tokens INT  (GRPO-like loss over last N tokens; default 512)
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import gc
import hashlib
import inspect
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Repo layout & vendor dependency validation
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "vendor")

# --- Expected artifacts ---
# DoRAFactors PEFT fork (git submodule, branch v1).
# Commit 9bb1084 must be reachable — this is the HEAD of v1 at paper time.
_PEFT_FORK_DIR = os.path.join(_VENDOR_DIR, "dorafactors-peft")
_PEFT_FORK_KNOWN_COMMIT = "9bb1084"
_PEFT_FORK_CLONE_URL = "https://github.com/sockeye44/dorafactors-peft"

# Reference upstream HF PEFT dora.py (unmodified, for baseline comparisons).
# SHA-1 file digest prefix guards against silent edits or truncated downloads.
# Searched in order: code/scripts/ (ships with the repo) → vendor/ → wget.
_REF_SEARCH_PATHS = [
    os.path.join(_REPO_ROOT, "code", "scripts", "dora.reference_hf_peft.py"),
    os.path.join(_VENDOR_DIR, "dora.reference_hf_peft.py"),
]
_REF_SHA1_PREFIX = "86def591d41"
_REF_UPSTREAM_COMMIT = "20a9829f76419149f5e447b856bc0abe865c28a7"
_REF_WGET_CMD = (
    f"wget https://raw.githubusercontent.com/huggingface/peft/"
    f"{_REF_UPSTREAM_COMMIT}/src/peft/tuners/lora/dora.py "
    f"-O vendor/dora.reference_hf_peft.py"
)
# Resolved at validation time; points to whichever path passes the digest check.
_REF_FILE: Optional[str] = None


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_exists(repo_dir: str, commit_prefix: str) -> bool:
    try:
        subprocess.check_output(
            ["git", "cat-file", "-t", commit_prefix],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _is_interactive() -> bool:
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _ask_proceed(context: str) -> bool:
    """Prompt the user to acknowledge divergent benchmark conditions.

    Non-interactive sessions (CI, piped stdin) always refuse — benchmarks
    published without validated vendor artifacts are not comparable to the
    paper's reported numbers and must not silently proceed.
    """
    if not _is_interactive():
        print(
            f"\n  Non-interactive session detected.  Cannot proceed without "
            f"validated vendor artifacts.\n"
            f"  {context}\n",
            file=sys.stderr,
        )
        return False
    print(
        f"\n{'=' * 72}\n"
        f"  WARNING — Benchmark conditions diverge from the paper\n"
        f"{'=' * 72}\n"
        f"\n"
        f"  {context}\n"
        f"\n"
        f"  Results produced under non-standard conditions are NOT directly\n"
        f"  comparable to the numbers reported in the paper.  If you intend\n"
        f"  to publish or share these results, please disclose the deviation.\n"
        f"\n"
        f"  Proceed anyway? [y/N] ",
        end="",
        flush=True,
    )
    answer = input().strip().lower()
    if answer not in ("y", "yes"):
        print("  Aborted.", file=sys.stderr)
        return False
    print()
    return True


def _validate_vendor_artifacts() -> None:
    """Validate (or guide the user to set up) the two vendor dependencies.

    Exits with code 1 if a dependency is missing/invalid and the user
    declines to continue under non-standard conditions.
    """
    global _REF_FILE
    problems: List[str] = []

    # ── 1. DoRAFactors PEFT fork (git submodule) ─────────────────────
    peft_src = os.path.join(_PEFT_FORK_DIR, "src")
    if not os.path.isdir(_PEFT_FORK_DIR):
        problems.append("peft_fork_missing")
        print(
            f"\n  Vendor dependency not found: {_PEFT_FORK_DIR}\n"
            f"\n"
            f"  The PEFT fork is registered as a git submodule.  If you cloned\n"
            f"  without --recurse-submodules, initialize it now:\n"
            f"\n"
            f"    git submodule update --init vendor/dorafactors-peft\n"
            f"\n"
            f"  Alternatively, clone it manually:\n"
            f"\n"
            f"    git clone -b v1 {_PEFT_FORK_CLONE_URL} vendor/dorafactors-peft\n",
            file=sys.stderr,
        )
    else:
        # A. Confirm the known commit is reachable
        if not _git_commit_exists(_PEFT_FORK_DIR, _PEFT_FORK_KNOWN_COMMIT):
            problems.append("peft_fork_commit")
            print(
                f"\n  Vendor integrity check failed: commit {_PEFT_FORK_KNOWN_COMMIT}\n"
                f"  is not reachable in {_PEFT_FORK_DIR}.\n"
                f"\n"
                f"  This may indicate a shallow clone, a different fork, or local\n"
                f"  history rewriting.  To re-initialize the submodule:\n"
                f"\n"
                f"    rm -rf vendor/dorafactors-peft\n"
                f"    git submodule update --init vendor/dorafactors-peft\n",
                file=sys.stderr,
            )

        # B. Confirm it looks like a PEFT package
        has_peft_src = os.path.isdir(os.path.join(peft_src, "peft"))
        has_peft_marker = False
        for cfg_name in ("setup.py", "setup.cfg", "pyproject.toml"):
            cfg_path = os.path.join(_PEFT_FORK_DIR, cfg_name)
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path) as f:
                        if "peft" in f.read().lower():
                            has_peft_marker = True
                            break
                except OSError:
                    pass
        if not (has_peft_src and has_peft_marker):
            problems.append("peft_fork_structure")
            print(
                f"\n  Vendor integrity check failed: {_PEFT_FORK_DIR} does not\n"
                f"  appear to be a valid PEFT package (missing src/peft/ or no\n"
                f"  'peft' reference in setup.py / pyproject.toml).\n"
                f"\n"
                f"  To re-initialize the submodule:\n"
                f"\n"
                f"    rm -rf vendor/dorafactors-peft\n"
                f"    git submodule update --init vendor/dorafactors-peft\n",
                file=sys.stderr,
            )

    # ── 2. Reference upstream dora.py (cascading search) ─────────────
    # Try each candidate path; accept the first one whose SHA-1 matches.
    ref_found = False
    ref_digest_mismatch: Optional[Tuple[str, str]] = None  # (path, actual_digest)

    for candidate in _REF_SEARCH_PATHS:
        if not os.path.isfile(candidate):
            continue
        digest = _sha1_file(candidate)
        if digest.startswith(_REF_SHA1_PREFIX):
            _REF_FILE = candidate
            ref_found = True
            break
        # Remember the first mismatch for diagnostics, keep searching.
        if ref_digest_mismatch is None:
            ref_digest_mismatch = (candidate, digest)

    if not ref_found:
        if ref_digest_mismatch is not None:
            path, digest = ref_digest_mismatch
            problems.append("ref_file_digest")
            print(
                f"\n  Reference file found but SHA-1 mismatch: {path}\n"
                f"  Expected prefix {_REF_SHA1_PREFIX}..., got {digest[:20]}...\n"
                f"\n"
                f"  The file may have been edited or fetched from a different\n"
                f"  upstream revision.  To fetch the canonical version:\n"
                f"\n"
                f"    mkdir -p vendor\n"
                f"    {_REF_WGET_CMD}\n",
                file=sys.stderr,
            )
        else:
            problems.append("ref_file_missing")
            searched = "\n    ".join(_REF_SEARCH_PATHS)
            print(
                f"\n  Reference file not found.  Searched:\n"
                f"    {searched}\n"
                f"\n"
                f"  The benchmark uses the unmodified upstream HF PEFT dora.py as a\n"
                f"  baseline reference.  To fetch it, run (from the repo root):\n"
                f"\n"
                f"    mkdir -p vendor\n"
                f"    {_REF_WGET_CMD}\n",
                file=sys.stderr,
            )

    # ── Gate ──────────────────────────────────────────────────────────
    if not problems:
        return  # all clear

    summary = ", ".join(problems)
    if not _ask_proceed(f"Failed checks: {summary}"):
        raise SystemExit(1)


# Run validation before any vendor imports.
_validate_vendor_artifacts()

# ---------------------------------------------------------------------------
# Repo-local imports (from validated vendor tree)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(_PEFT_FORK_DIR, "src"))

from peft.tuners.lora.dora import (
    DoraLinearLayer,
    _compose_eager_inplace,
    _invalidate_fused_cache,
    set_dora_norm_threshold_mb,
)
from peft.tuners.lora import dora_fused as dora_fused_mod
from peft.tuners.tuners_utils import _maybe_include_all_linear_layers
from peft.tuners.lora.dora_fused import (
    FusedDoRAComposeFunction,
    fused_dora_compose,
    fused_dora_compose_autograd,
    fused_norm_assembly,
    is_triton_available,
)
from peft.utils.other import transpose

# ---------------------------------------------------------------------------
# Import reference implementation via importlib (None when user opted to skip)
# ---------------------------------------------------------------------------
dora_ref = None
if _REF_FILE is not None:
    spec = importlib.util.spec_from_file_location("dora_ref", _REF_FILE)
    dora_ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dora_ref)

# ---------------------------------------------------------------------------
# Dataclasses (reused from bench_dora.py patterns)
# ---------------------------------------------------------------------------


@dataclass
class Stat:
    mean_ms: float
    std_ms: float
    ci95_ms: float


@dataclass
class Measurement:
    time: Stat
    peak_memory_mb: Optional[float]
    extra: Dict[str, Any]


@dataclass
class Meta:
    pytorch_version: str
    cuda_runtime_version: Optional[str]
    cuda_available: bool
    device_name: Optional[str]
    total_device_memory_gb: Optional[float]
    nvidia_driver: Optional[str]
    cuda_driver_version: Optional[str]
    gpu_count: int
    torch_git_version: Optional[str]
    python_version: str
    os: str
    commit_sha: Optional[str]
    triton_available: bool


# ---------------------------------------------------------------------------
# Helpers (reuse bench_dora.py patterns)
# ---------------------------------------------------------------------------


def _get_commit_sha() -> Optional[str]:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return None


def _get_nvidia_smi() -> Tuple[Optional[str], Optional[str]]:
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL).decode()
        header = out.splitlines()[2] if len(out.splitlines()) >= 3 else out.splitlines()[0]
        drv = cud = None
        if "Driver Version:" in header:
            try:
                drv = header.split("Driver Version:")[1].split()[0]
            except Exception:
                pass
        if "CUDA Version:" in header:
            try:
                cud = header.split("CUDA Version:")[1].split()[0]
            except Exception:
                pass
        return drv, cud
    except Exception:
        return None, None


def get_meta(device: torch.device) -> Meta:
    torch_cuda_available = torch.cuda.is_available()
    device_name = None
    total_mem_gb = None
    gpu_count = torch.cuda.device_count() if torch_cuda_available else 0
    if torch_cuda_available:
        try:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(idx)
            total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        except Exception:
            pass
    driver, cuda_drv = _get_nvidia_smi()
    return Meta(
        pytorch_version=torch.__version__,
        cuda_runtime_version=torch.version.cuda if hasattr(torch.version, "cuda") else None,
        cuda_available=torch_cuda_available,
        device_name=device_name,
        total_device_memory_gb=total_mem_gb,
        nvidia_driver=driver,
        cuda_driver_version=cuda_drv,
        gpu_count=gpu_count,
        torch_git_version=getattr(torch.version, "git_version", None),
        python_version=sys.version.replace("\n", " "),
        os=f"{platform.system()} {platform.release()} ({platform.platform()})",
        commit_sha=_get_commit_sha(),
        triton_available=is_triton_available(),
    )


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_cuda_events(fn, warmup=5, iters=50, device="cuda"):
    """Sub-ms timing using CUDA events. Returns all samples in milliseconds (unsorted)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        end_ev.synchronize()
        times.append(start_ev.elapsed_time(end_ev))  # milliseconds
    return times  # all samples, unsorted


def _benchmark_cpu_perf_counter(fn, warmup=5, iters=50):
    """CPU timing using time.perf_counter. Returns all samples in milliseconds (unsorted)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _benchmark_fn(fn, warmup=5, iters=50, device="cuda"):
    """Dispatch to CUDA events or CPU perf_counter based on device type."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda":
        return _benchmark_cuda_events(fn, warmup=warmup, iters=iters, device=device)
    return _benchmark_cpu_perf_counter(fn, warmup=warmup, iters=iters)


def _benchmark_interleaved_cuda_events(
    callables: Dict[str, Any],
    warmup: int = 2,
    iters: int = 5,
    device: str = "cuda",
    seed: int = 1234,
) -> Dict[str, List[float]]:
    """Benchmark a set of callables in shuffled round-robin order.

    This reduces drift and ordering artifacts when comparing many Triton
    candidate configs for the same shape.
    """
    labels = list(callables.keys())
    rng = random.Random(seed)
    for label in labels:
        for _ in range(warmup):
            callables[label]()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    samples: Dict[str, List[float]] = {label: [] for label in labels}
    for _ in range(iters):
        order = labels[:]
        rng.shuffle(order)
        for label in order:
            start_ev.record()
            callables[label]()
            end_ev.record()
            end_ev.synchronize()
            samples[label].append(start_ev.elapsed_time(end_ev))
    return samples


def _median_from_samples(vals: list) -> float:
    """Median from a list of float values."""
    return float(statistics.median(vals))


def _percentile_sorted(sv: List[float], q: float) -> float:
    """Linearly interpolated percentile from a pre-sorted list."""
    if len(sv) == 1:
        return float(sv[0])
    pos = (len(sv) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sv[lo])
    frac = pos - lo
    return float(sv[lo] * (1.0 - frac) + sv[hi] * frac)


def _timing_stats_from_samples(vals: list) -> dict:
    """Compute dispersion statistics from raw timing samples.

    Returns dict with: mean_ms, median_ms, std_ms, iqr_ms, ci95_ms,
    p5_ms, p95_ms, samples_ms, n.
    """
    n = len(vals)
    sv = sorted(vals)
    median = float(statistics.median(sv))
    mean = sum(sv) / n
    var = sum((v - mean) ** 2 for v in sv) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    p5 = _percentile_sorted(sv, 0.05)
    p95 = _percentile_sorted(sv, 0.95)
    q1 = _percentile_sorted(sv, 0.25)
    q3 = _percentile_sorted(sv, 0.75)
    iqr = q3 - q1
    return {
        "mean_ms": round(mean, 4),
        "median_ms": round(median, 4),
        "std_ms": round(std, 4),
        "iqr_ms": round(iqr, 4),
        "ci95_ms": round(ci95, 4),
        "p5_ms": round(p5, 4),
        "p95_ms": round(p95, 4),
        "samples_ms": [round(float(v), 4) for v in vals],
        "n": n,
    }


def _stat_from_list(vals: List[float]) -> Stat:
    m = float(sum(vals) / len(vals))
    var = float(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return Stat(mean_ms=m, std_ms=sd, ci95_ms=ci)


def _timing_stats_with_samples_ms(vals: List[float]) -> Dict[str, Any]:
    s = _stat_from_list(vals)
    return {
        "mean_ms": round(s.mean_ms, 4),
        "std_ms": round(s.std_ms, 4),
        "ci95_ms": round(s.ci95_ms, 4),
        "samples_ms": [round(float(v), 4) for v in vals],
        "n": len(vals),
    }


def _triton_config_to_dict(cfg: Any) -> Dict[str, Any]:
    return {
        "BLOCK_SIZE": int(cfg.kwargs["BLOCK_SIZE"]),
        **({"ROWS_PER_PROGRAM": int(cfg.kwargs["ROWS_PER_PROGRAM"])} if "ROWS_PER_PROGRAM" in cfg.kwargs else {}),
        "num_warps": int(cfg.num_warps),
        "num_stages": int(cfg.num_stages),
    }


def _triton_config_label(cfg: Any) -> str:
    data = _triton_config_to_dict(cfg)
    parts = [f"BS={data['BLOCK_SIZE']}"]
    if "ROWS_PER_PROGRAM" in data:
        parts.append(f"RPP={data['ROWS_PER_PROGRAM']}")
    parts.append(f"W={data['num_warps']}")
    parts.append(f"S={data['num_stages']}")
    return ",".join(parts)


def _format_speedup_vs_baseline(baseline_ms: Optional[float], current_ms: Optional[float]) -> str:
    if baseline_ms is None or current_ms is None:
        return ""
    if baseline_ms <= 0 or current_ms <= 0:
        return ""
    spd = baseline_ms / current_ms
    pct = (spd - 1.0) * 100.0
    return f"  speedup={spd:.3f}x ({pct:+.2f}%)"


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {s}")


def _dtype_label(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _device_from_str(s: str) -> torch.device:
    s = s.lower()
    if s == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    if s == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"Unsupported device: {s}")


def _seed_all(seed: int = 1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _set_fused_config(fwd: int, bwd: int):
    """Set the PEFT_DORA_FUSED and PEFT_DORA_FUSED_BACKWARD env vars and invalidate caches."""
    os.environ["PEFT_DORA_FUSED"] = str(fwd)
    os.environ["PEFT_DORA_FUSED_BACKWARD"] = str(bwd)
    _invalidate_fused_cache()


def _geomean(vals: List[float]) -> float:
    """Geometric mean — more robust for speedup ratios than arithmetic mean."""
    if not vals:
        return 0.0
    log_sum = sum(math.log(max(v, 1e-12)) for v in vals)
    return math.exp(log_sum / len(vals))


# Production-relevance filter: shapes with d_in >= 2048 correspond to
# transformer layers in models >= ~1B parameters.  Matches the filter
# in generate_figures.py used for paper geomean claims.
MIN_DIN_PRODUCTION = 2048


def _is_production_shape(entry: dict) -> bool:
    """Return True if the entry's shape meets the production-relevance threshold."""
    shape = entry.get("shape")
    if isinstance(shape, dict):
        # E2E / memory: {"hidden": h, ...}
        return shape.get("hidden", 0) >= MIN_DIN_PRODUCTION
    if isinstance(shape, (list, tuple)):
        if len(shape) == 3:
            # Norm: [out_f, in_f, rank]
            return shape[1] >= MIN_DIN_PRODUCTION
        if len(shape) == 2:
            # Compose / backward: [rows, cols] where cols is d_in
            return shape[1] >= MIN_DIN_PRODUCTION
    return True  # unknown shape format — include by default


# Four standard DoRAFactors configurations: (label, fused_fwd, fused_bwd)
CONFIGS = [
    ("dorafactors_eager", 0, 0),
    ("dorafactors_fused_fwd", 1, 0),
    ("dorafactors_fused_bwd", 0, 1),
    ("dorafactors_fully_fused", 1, 1),
]


# ---------------------------------------------------------------------------
# Console table helpers
# ---------------------------------------------------------------------------


def _print_table(headers: List[str], rows: List[List[str]], title: str = ""):
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * sum(col_widths) + "-" * (2 * (len(col_widths) - 1)))
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        print(fmt.format(*padded))
    print()


# ---------------------------------------------------------------------------
# SimpleLoraDoraBlock (from bench_dora.py)
# ---------------------------------------------------------------------------


class SimpleLoraDoraBlock(nn.Module):
    def __init__(self, hidden_size: int, rank: int, scaling: float, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device)
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.lora_A = nn.Linear(hidden_size, rank, bias=False, dtype=dtype, device=device)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False, dtype=dtype, device=device)
        self.dora = DoraLinearLayer(fan_in_fan_out=False).to(device)
        with torch.no_grad():
            self.dora.update_layer(
                base_layer=self.linear,
                lora_A=self.lora_A.weight,
                lora_B=self.lora_B.weight,
                scaling=scaling,
            )
        self.scaling = scaling

    def forward(self, x):
        base = self.linear(x)
        extra = self.dora(
            x,
            lora_A=self.lora_A,
            lora_B=self.lora_B,
            scaling=self.scaling,
            base_layer=self.linear,
            base_result=base.detach(),
        )
        return base + extra


# ====================================================================
# Section 1: Norm benchmark — Three-axis comparison
# ====================================================================

NORM_SHAPES = [
    # (out, in, rank) — standard ranks
    (4096, 4096, 16),
    (4096, 4096, 64),
    (8192, 8192, 64),
    # production ranks
    (4096, 4096, 128),
    (4096, 4096, 256),
    (4096, 4096, 384),  # dominant production rank
    (8192, 8192, 384),
    # extreme ranks
    (4096, 4096, 512),
    (8192, 8192, 512),
    (8192, 8192, 768),  # max production rank
    # real MLP shapes at extreme rank
    (4096, 11008, 384),  # Llama-7B up/gate proj
    (8192, 28672, 384),  # Llama-70B up/gate proj
]

# MoE/MLA dimensions for extended benchmarks.
NORM_SHAPES_EXTENDED = NORM_SHAPES + [
    (512, 512, 64),  # MoE expert
    (512, 1024, 64),  # MoE expert
    (512, 512, 384),  # MoE expert at production rank
    (1024, 1024, 128),  # MoE expert
    (1024, 1024, 384),  # MoE expert at production rank
    (2048, 2048, 384),  # Mamba-MoE
    (3072, 3072, 384),  # Mamba-MoE
    (6144, 6144, 384),  # DeepSeek-R1
    (6144, 12288, 384),  # DeepSeek-R1
    (16384, 16384, 384),  # MLA
    (28672, 6144, 384),  # MLA
]


def _measure_norm_memory_delta(fn, device: torch.device) -> float:
    """Measure the working-set memory delta: peak_during_call - allocated_before_call.

    This isolates the temporary allocations made *by the function* from
    pre-existing tensors (base_weight, lora_A/B, etc.) that are already
    on the GPU.  Returns delta in MB.
    """
    if device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return (peak - before) / (1024**2)


def bench_norm(
    dtype: torch.dtype, device: torch.device, repeats: int, warmup: int, verbose: bool, shapes=None
) -> List[Dict]:
    if dora_ref is None:
        print(
            "  Skipping norm benchmark: reference implementation unavailable\n"
            "  (the upstream HF PEFT dora.py was not found or failed SHA-1 verification).",
            file=sys.stderr,
        )
        return []
    results = []
    norm_dtypes = [dtype]

    for out_f, in_f, rank in shapes or NORM_SHAPES:
        for dt in norm_dtypes:
            _seed_all()
            scaling = 0.5

            base_weight = torch.randn(out_f, in_f, device=device, dtype=dt)
            lora_A = torch.randn(rank, in_f, device=device, dtype=dt)
            lora_B = torch.randn(out_f, rank, device=device, dtype=dt)

            # --- Axis 0: PEFT identity-matrix approach ---
            # Replicates the actual upstream PEFT code path: constructs a full
            # identity matrix of size [in_features, in_features], passes it
            # through both LoRA layers to materialize dense B@A, then computes
            # the row-wise norm.  This is O(d_in^2) in memory.
            peft_eye_time_ms = None
            peft_eye_delta_mb = None
            peft_eye_oom = False
            peft_eye_samples = None

            try:

                def _peft_eye():
                    with torch.no_grad():
                        x_eye = torch.eye(in_f, device=device, dtype=dt)
                        lora_weight = (x_eye @ lora_A.T @ lora_B.T).T
                        weight = base_weight + scaling * lora_weight
                        weight_norm = torch.linalg.norm(weight, dim=1)
                        del x_eye, lora_weight, weight, weight_norm

                peft_eye_samples = _benchmark_fn(_peft_eye, warmup=warmup, iters=repeats, device=device)
                peft_eye_time_ms = _median_from_samples(peft_eye_samples)
                peft_eye_delta_mb = _measure_norm_memory_delta(_peft_eye, device)
                del _peft_eye
            except torch.cuda.OutOfMemoryError:
                peft_eye_oom = True
                if verbose:
                    print(f"  norm ({out_f}x{in_f}, r={rank}, {_dtype_label(dt)}): PEFT-eye OOM")
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # --- Axis 1: Reference (upstream dense B@A) ---
            # In production, the reference materializes B@A on every forward
            # call.  We include the matmul inside the measurement window so
            # the reported timing reflects the true cost of the dense path.
            ref_layer = dora_ref.DoraLinearLayer(fan_in_fan_out=False).to(device)

            def _ref():
                with torch.no_grad():
                    dense_lora = lora_B @ lora_A  # materialization cost
                    ref_layer.get_weight_norm(base_weight, dense_lora, scaling)

            # Headline timing: full _ref() including B@A materialization
            ref_samples = _benchmark_fn(_ref, warmup=warmup, iters=repeats, device=device)
            ref_time_ms = _median_from_samples(ref_samples)

            # Diagnostic: norm-only timing (excludes B@A matmul)
            dense_lora_for_timing = lora_B @ lora_A

            def _ref_norm_only():
                with torch.no_grad():
                    ref_layer.get_weight_norm(base_weight, dense_lora_for_timing, scaling)

            ref_norm_only_samples = _benchmark_fn(_ref_norm_only, warmup=warmup, iters=repeats, device=device)
            ref_norm_only_ms = _median_from_samples(ref_norm_only_samples)
            del dense_lora_for_timing

            # Memory delta: includes B@A materialization (the thing we avoid)
            ref_delta_mb = _measure_norm_memory_delta(_ref, device)

            # Save ref norm for numerical diff before freeing ref_layer
            with torch.no_grad():
                dense_lora_check = lora_B @ lora_A
                ref_norm = ref_layer.get_weight_norm(base_weight, dense_lora_check, scaling)
                del dense_lora_check

            del ref_layer, _ref, _ref_norm_only
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # --- Axis 2: Ours (factored, fused kernels OFF) ---
            # Disable fused kernels to isolate factorization benefit
            _set_fused_config(0, 0)
            opt_layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
            set_dora_norm_threshold_mb(256)

            def _factored():
                with torch.no_grad():
                    opt_layer._get_weight_norm_linear(
                        base_weight=base_weight,
                        lora_A_w=lora_A,
                        lora_B_w=lora_B,
                        scaling=scaling,
                    )

            factored_samples = _benchmark_fn(_factored, warmup=warmup, iters=repeats, device=device)
            factored_time_ms = _median_from_samples(factored_samples)

            # Memory delta: no B@A, just U [out,r] + gram [r,r] + chunks
            factored_delta_mb = _measure_norm_memory_delta(_factored, device)

            # Numerical diff (opt_layer still alive) — compare in float32
            with torch.no_grad():
                opt_norm = opt_layer._get_weight_norm_linear(
                    base_weight=base_weight, lora_A_w=lora_A, lora_B_w=lora_B, scaling=scaling
                )
                max_abs_diff = float(torch.max(torch.abs(ref_norm.float() - opt_norm.float())).item())
                del opt_norm

            del opt_layer, _factored
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # --- Axis 3: Ours (factored + fused Triton norm assembly) ---
            _set_fused_config(1, 0)
            fused_layer = DoraLinearLayer(fan_in_fan_out=False).to(device)
            set_dora_norm_threshold_mb(256)

            def _fused():
                with torch.no_grad():
                    fused_layer._get_weight_norm_linear(
                        base_weight=base_weight,
                        lora_A_w=lora_A,
                        lora_B_w=lora_B,
                        scaling=scaling,
                    )

            fused_samples = _benchmark_fn(_fused, warmup=warmup, iters=repeats, device=device)
            fused_time_ms = _median_from_samples(fused_samples)

            # Fused-vs-reference numerical diff — compare in float32
            with torch.no_grad():
                fused_norm = fused_layer._get_weight_norm_linear(
                    base_weight=base_weight, lora_A_w=lora_A, lora_B_w=lora_B, scaling=scaling
                )
                fused_vs_ref_diff = float(torch.max(torch.abs(ref_norm.float() - fused_norm.float())).item())
                del ref_norm, fused_norm

            ref_over_factored = ref_time_ms / factored_time_ms if factored_time_ms > 0 else float("inf")
            factored_over_fused = factored_time_ms / fused_time_ms if fused_time_ms > 0 else float("inf")
            mem_reduction = ref_delta_mb / factored_delta_mb if factored_delta_mb > 0 else float("inf")

            # PEFT-eye speedup ratios (None if OOM)
            peft_eye_over_ref = None
            peft_eye_over_factored = None
            if peft_eye_time_ms is not None and not peft_eye_oom:
                if ref_time_ms > 0:
                    peft_eye_over_ref = peft_eye_time_ms / ref_time_ms
                if factored_time_ms > 0:
                    peft_eye_over_factored = peft_eye_time_ms / factored_time_ms

            # Theoretical persistent working set (what the writeup claims):
            #   Reference: dense BA product [out, in] in compute_dtype
            #   Factored:  U [out, r] + gram [r, r] in compute_dtype
            # These are the buffers that would stack across layers in a real model.
            # The measured deltas above also include transient chunk-processing temps.
            compute_elem = 4  # fp32 always (factored upcasts; ref's BA is in input dtype but norm upcasts)
            ref_theory_mb = (out_f * in_f * compute_elem) / (1024**2)
            fact_theory_mb = (out_f * rank + rank * rank) * compute_elem / (1024**2)
            theory_reduction = ref_theory_mb / fact_theory_mb if fact_theory_mb > 0 else float("inf")

            # Build timings dict
            timings_dict = {
                "ref": _timing_stats_from_samples(ref_samples),
                "ref_norm_only": _timing_stats_from_samples(ref_norm_only_samples),
                "factored": _timing_stats_from_samples(factored_samples),
                "fused": _timing_stats_from_samples(fused_samples),
                "peft_eye": _timing_stats_from_samples(peft_eye_samples) if peft_eye_samples else None,
            }

            row = {
                "shape": [out_f, in_f, rank],
                "dtype": _dtype_label(dt),
                "peft_eye_time_ms": round(peft_eye_time_ms, 4) if peft_eye_time_ms is not None else None,
                "peft_eye_delta_mb": round(peft_eye_delta_mb, 1) if peft_eye_delta_mb is not None else None,
                "peft_eye_oom": peft_eye_oom,
                "peft_eye_over_ref": round(peft_eye_over_ref, 2) if peft_eye_over_ref is not None else None,
                "peft_eye_over_factored": round(peft_eye_over_factored, 2)
                if peft_eye_over_factored is not None
                else None,
                "ref_time_ms": round(ref_time_ms, 4),
                "ref_norm_only_ms": round(ref_norm_only_ms, 4),
                "factored_time_ms": round(factored_time_ms, 4),
                "fused_time_ms": round(fused_time_ms, 4),
                "ref_over_factored": round(ref_over_factored, 2),
                "factored_over_fused": round(factored_over_fused, 2),
                "ref_delta_mb": round(ref_delta_mb, 1),
                "factored_delta_mb": round(factored_delta_mb, 1),
                "mem_reduction": round(mem_reduction, 1),
                "ref_theory_mb": round(ref_theory_mb, 1),
                "fact_theory_mb": round(fact_theory_mb, 1),
                "theory_reduction": round(theory_reduction, 1),
                "max_abs_diff": max_abs_diff,
                "fused_vs_ref_diff": fused_vs_ref_diff,
                "timings": timings_dict,
            }
            results.append(row)

            if verbose:
                if peft_eye_oom:
                    eye_str = "peft_eye=OOM  "
                elif peft_eye_time_ms is not None:
                    eye_str = f"peft_eye={peft_eye_time_ms:.4f}ms  "
                else:
                    eye_str = ""
                print(
                    f"  norm ({out_f}x{in_f}, r={rank}, {_dtype_label(dt)}): "
                    f"{eye_str}"
                    f"ref={ref_time_ms:.4f}ms (norm_only={ref_norm_only_ms:.4f}ms)  "
                    f"factored={factored_time_ms:.4f}ms  fused={fused_time_ms:.4f}ms  "
                    f"ref/fact={ref_over_factored:.2f}x  fact/fused={factored_over_fused:.2f}x  "
                    f"measured: ref={ref_delta_mb:.1f}MB fact={factored_delta_mb:.1f}MB ({mem_reduction:.1f}x)  "
                    f"theory: {ref_theory_mb:.1f}MB vs {fact_theory_mb:.1f}MB ({theory_reduction:.1f}x)  "
                    f"diff={max_abs_diff:.2e}  fused_diff={fused_vs_ref_diff:.2e}"
                )

            # Free remaining tensors (ref_layer and opt_layer already deleted above)
            del base_weight, lora_A, lora_B, fused_layer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


# ====================================================================
# Section 2: Compose benchmark
# ====================================================================

COMPOSE_SHAPES = [
    (1, 4096),  # single token inference
    (512, 4096),  # short seq
    (2048, 4096),  # standard training
    (4096, 4096),  # moderate training
    (2048, 8192),  # large hidden standard
    (8192, 8192),  # large batch*seq, large hidden
    (16384, 8192),  # bs=4 × seq=4096
    (32768, 8192),  # bs=8 × seq=4096 (extreme)
]

# MoE/MLA dimensions: expert MLPs (512-1024), Mamba-MoE model dims,
# DeepSeek-R1 MLP dims, MLA kv_b_proj (up to 28672).
COMPOSE_SHAPES_EXTENDED = COMPOSE_SHAPES + [
    (1, 512),  # MoE expert single token
    (512, 512),  # MoE expert short seq
    (2048, 512),  # MoE expert standard training
    (8192, 512),  # MoE expert large batch
    (2048, 1024),  # MoE expert
    (2048, 2048),  # Mamba-MoE model dim
    (2048, 3072),  # Mamba-MoE model dim
    (2048, 6144),  # DeepSeek-R1 MLP
    (2048, 12288),  # DeepSeek-R1 MLP
    (2048, 16384),  # MLA kv_b_proj
    (2048, 28672),  # MLA kv_b_proj
    (8192, 28672),  # MLA kv_b_proj large batch
]


def bench_compose(
    dtype: torch.dtype, device: torch.device, repeats: int, warmup: int, verbose: bool, shapes=None
) -> List[Dict]:
    results = []
    compose_dtypes = [dtype]

    for rows, cols in shapes or COMPOSE_SHAPES:
        for dt in compose_dtypes:
            _seed_all()
            scale = 0.5
            lora = torch.randn(rows, cols, device=device, dtype=dt)
            base = torch.randn(rows, cols, device=device, dtype=dt)
            mag = torch.randn(1, cols, device=device, dtype=dt).abs() + 0.5

            # --- Eager ---
            def _eager():
                _compose_eager_inplace(lora.clone(), base, mag, scale)

            eager_samples = _benchmark_fn(_eager, warmup=warmup, iters=repeats, device=device)
            eager_us = _median_from_samples(eager_samples) * 1000.0

            # --- Fused forward ---
            def _fused_fwd():
                fused_dora_compose(lora.clone(), base, mag, scale, inplace=True)

            fused_fwd_samples = _benchmark_fn(_fused_fwd, warmup=warmup, iters=repeats, device=device)
            fused_fwd_us = _median_from_samples(fused_fwd_samples) * 1000.0

            # --- Fused out-of-place (autotuned _fused_dora_compose_kernel) ---
            # No clone needed: inplace=False allocates fresh output, never mutates lora.
            def _fused_oop():
                fused_dora_compose(lora, base, mag, scale, inplace=False)

            fused_oop_samples = _benchmark_fn(_fused_oop, warmup=warmup, iters=repeats, device=device)
            fused_oop_us = _median_from_samples(fused_oop_samples) * 1000.0

            # --- Eager out-of-place (fair baseline for fused OOP) ---
            # No clone: computes fresh output like fused_oop, making
            # speedup_oop an apples-to-apples kernel comparison.
            # Uses the standard PyTorch expression (3 elementwise kernel
            # launches) rather than _compose_eager_inplace — this measures
            # the real-world benefit of fusing vs. what a user would write.
            def _eager_oop():
                _ = (mag - 1.0) * base + mag * (scale * lora)

            eager_oop_samples = _benchmark_fn(_eager_oop, warmup=warmup, iters=repeats, device=device)
            eager_oop_us = _median_from_samples(eager_oop_samples) * 1000.0

            # --- Fused autograd ---
            lora_ag = lora.clone().requires_grad_(True)
            base_ag = base.clone().requires_grad_(True)
            mag_ag = mag.clone().requires_grad_(True)

            def _fused_ag():
                out = fused_dora_compose_autograd(lora_ag, base_ag, mag_ag, scale)
                del out  # break autograd graph; detach_() disallowed on views in PT ≥2.10

            fused_ag_samples = _benchmark_fn(_fused_ag, warmup=warmup, iters=repeats, device=device)
            fused_ag_us = _median_from_samples(fused_ag_samples) * 1000.0

            # --- Eager out-of-place autograd (fair baseline for fused AG) ---
            # Same tensors as fused_ag (requires_grad=True), same forward-only
            # graph construction — isolates kernel cost from autograd overhead.
            # Note: both this and _fused_ag build autograd graphs each iteration
            # without backward(); graph nodes are freed by GC.  The overhead is
            # symmetric, so the speedup ratio remains fair.
            def _eager_oop_ag():
                out = (mag_ag - 1.0) * base_ag + mag_ag * (scale * lora_ag)
                del out

            eager_oop_ag_samples = _benchmark_fn(_eager_oop_ag, warmup=warmup, iters=repeats, device=device)
            eager_oop_ag_us = _median_from_samples(eager_oop_ag_samples) * 1000.0

            # One-shot correctness check: eager vs fused output
            with torch.no_grad():
                eager_out = _compose_eager_inplace(lora.clone(), base, mag, scale)
                fused_check = fused_dora_compose(lora.clone(), base, mag, scale, inplace=True)
                compose_max_abs_diff = float(torch.max(torch.abs(eager_out.float() - fused_check.float())).item())
                del eager_out, fused_check

            # Effective bandwidth (approximate).  Uses the same nominal total_bytes
            # for all paths: this is accurate for the fused single-pass kernel but
            # understates actual device traffic for unfused paths (eager_ip/eager_oop)
            # which materialize intermediates across 3+ separate kernel launches.
            elem_size = torch.tensor([], dtype=dt).element_size()
            total_bytes = (3 * rows * cols + cols) * elem_size  # 3 reads + 1 write
            eager_bw = total_bytes / (eager_us / 1e6) / 1e9 if eager_us > 0 else 0
            eager_oop_bw = total_bytes / (eager_oop_us / 1e6) / 1e9 if eager_oop_us > 0 else 0
            fused_bw = total_bytes / (fused_fwd_us / 1e6) / 1e9 if fused_fwd_us > 0 else 0
            fused_oop_bw = total_bytes / (fused_oop_us / 1e6) / 1e9 if fused_oop_us > 0 else 0

            # Speedups: each fused path compared against the matching eager
            # baseline (same allocation pattern) for apples-to-apples ratios.
            speedup_fwd = eager_us / fused_fwd_us if fused_fwd_us > 0 else float("inf")
            speedup_oop = eager_oop_us / fused_oop_us if fused_oop_us > 0 else float("inf")
            speedup_ag = eager_oop_ag_us / fused_ag_us if fused_ag_us > 0 else float("inf")

            row = {
                "shape": [rows, cols],
                "dtype": _dtype_label(dt),
                "eager_us": round(eager_us, 2),
                "eager_oop_us": round(eager_oop_us, 2),
                "eager_oop_ag_us": round(eager_oop_ag_us, 2),
                "fused_fwd_us": round(fused_fwd_us, 2),
                "fused_oop_us": round(fused_oop_us, 2),
                "fused_autograd_us": round(fused_ag_us, 2),
                "speedup_fwd": round(speedup_fwd, 2),
                "speedup_oop": round(speedup_oop, 2),
                "speedup_autograd": round(speedup_ag, 2),
                "compose_max_abs_diff": compose_max_abs_diff,
                "approx_eager_bw_gbps": round(eager_bw, 1),
                "approx_eager_oop_bw_gbps": round(eager_oop_bw, 1),
                "approx_fused_bw_gbps": round(fused_bw, 1),
                "approx_fused_oop_bw_gbps": round(fused_oop_bw, 1),
                "timings": {
                    "eager": _timing_stats_from_samples([v * 1000.0 for v in eager_samples]),
                    "eager_oop": _timing_stats_from_samples([v * 1000.0 for v in eager_oop_samples]),
                    "eager_oop_ag": _timing_stats_from_samples([v * 1000.0 for v in eager_oop_ag_samples]),
                    "fused_fwd": _timing_stats_from_samples([v * 1000.0 for v in fused_fwd_samples]),
                    "fused_oop": _timing_stats_from_samples([v * 1000.0 for v in fused_oop_samples]),
                    "fused_autograd": _timing_stats_from_samples([v * 1000.0 for v in fused_ag_samples]),
                },
            }
            results.append(row)

            if verbose:
                print(
                    f"  compose ({rows}x{cols}, {_dtype_label(dt)}): "
                    f"eager_ip={eager_us:.1f}us  eager_oop={eager_oop_us:.1f}us  "
                    f"fused_ip={fused_fwd_us:.1f}us ({speedup_fwd:.2f}x)  "
                    f"fused_oop={fused_oop_us:.1f}us ({speedup_oop:.2f}x)  "
                    f"autograd={fused_ag_us:.1f}us ({speedup_ag:.2f}x)  "
                    f"BW: eoop={eager_oop_bw:.0f} fip={fused_bw:.0f} foop={fused_oop_bw:.0f} GB/s  "
                    f"diff={compose_max_abs_diff:.2e}"
                )

            del lora, base, mag, lora_ag, base_ag, mag_ag
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


# ====================================================================
# Section 3: Backward benchmark
# ====================================================================


def bench_backward(
    dtype: torch.dtype, device: torch.device, repeats: int, warmup: int, verbose: bool, shapes=None
) -> List[Dict]:
    results = []
    bwd_dtypes = [dtype]

    for rows, cols in shapes or COMPOSE_SHAPES:
        for dt in bwd_dtypes:
            _seed_all()
            scale = 0.5

            # Preallocate template tensors outside timed closures to avoid
            # timing RNG + allocator overhead.
            _tmpl_lora = torch.randn(rows, cols, device=device, dtype=dt)
            _tmpl_base = torch.randn(rows, cols, device=device, dtype=dt)
            _tmpl_mag = torch.randn(1, cols, device=device, dtype=dt).abs().add_(0.5)

            # --- Standard autograd ---
            def _std_fwd_bwd():
                lora = _tmpl_lora.clone().requires_grad_(True)
                base = _tmpl_base.clone().requires_grad_(True)
                mag = _tmpl_mag.clone().requires_grad_(True)
                out = (mag - 1) * base + mag * (scale * lora)
                out.sum().backward()

            std_samples = _benchmark_fn(_std_fwd_bwd, warmup=warmup, iters=repeats, device=device)
            std_us = _median_from_samples(std_samples) * 1000.0

            # --- Fused autograd ---
            def _fused_fwd_bwd():
                lora = _tmpl_lora.clone().requires_grad_(True)
                base = _tmpl_base.clone().requires_grad_(True)
                mag = _tmpl_mag.clone().requires_grad_(True)
                out = fused_dora_compose_autograd(lora, base, mag, scale)
                out.sum().backward()

            fused_bwd_samples = _benchmark_fn(_fused_fwd_bwd, warmup=warmup, iters=repeats, device=device)
            fused_us = _median_from_samples(fused_bwd_samples) * 1000.0

            # Gradient agreement
            lora_s = torch.randn(rows, cols, device=device, dtype=dt, requires_grad=True)
            base_s = torch.randn(rows, cols, device=device, dtype=dt, requires_grad=True)
            mag_s = (torch.randn(1, cols, device=device, dtype=dt).abs() + 0.5).requires_grad_(True)
            out_s = (mag_s - 1) * base_s + mag_s * (scale * lora_s)
            out_s.sum().backward()

            lora_f = lora_s.detach().clone().requires_grad_(True)
            base_f = base_s.detach().clone().requires_grad_(True)
            mag_f = mag_s.detach().clone().requires_grad_(True)
            out_f = fused_dora_compose_autograd(lora_f, base_f, mag_f, scale)
            out_f.sum().backward()

            grad_diffs = {
                "d_lora": float(torch.max(torch.abs(lora_s.grad - lora_f.grad)).item()),
                "d_base": float(torch.max(torch.abs(base_s.grad - base_f.grad)).item()),
                "d_mag": float(torch.max(torch.abs(mag_s.grad - mag_f.grad)).item()),
            }

            speedup = std_us / fused_us if fused_us > 0 else float("inf")

            row = {
                "shape": [rows, cols],
                "dtype": _dtype_label(dt),
                "std_total_us": round(std_us, 2),
                "fused_total_us": round(fused_us, 2),
                "speedup": round(speedup, 2),
                "grad_max_abs_diff": grad_diffs,
                "timings": {
                    "std": _timing_stats_from_samples([v * 1000.0 for v in std_samples]),
                    "fused": _timing_stats_from_samples([v * 1000.0 for v in fused_bwd_samples]),
                },
            }
            results.append(row)

            if verbose:
                print(
                    f"  backward ({rows}x{cols}, {_dtype_label(dt)}): "
                    f"std={std_us:.1f}us  fused={fused_us:.1f}us  speedup={speedup:.2f}x  "
                    f"grads: lora={grad_diffs['d_lora']:.2e} base={grad_diffs['d_base']:.2e} mag={grad_diffs['d_mag']:.2e}"
                )

            del _tmpl_lora, _tmpl_base, _tmpl_mag
            del lora_s, base_s, mag_s, out_s, lora_f, base_f, mag_f, out_f
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


# ====================================================================
# Section 10: Targeted Triton microbenchmarks
# ====================================================================

FORWARD_ANOMALY_WINDOWS = [
    {"label": "forward_window_5k", "rows": 2048, "cols": [5120, 5376, 6144]},
    {"label": "forward_window_21k", "rows": 2048, "cols": [17408, 21504, 25600]},
]

BACKWARD_BOUNDARY_COLS = [6144, 7680, 8192, 9216, 10240, 12288, 17408, 21504, 25600, 35840]
BACKWARD_BOUNDARY_ROWS = 2048

ROW_BUCKET_PROBES = [
    {"kernel": "forward", "cols": 4096, "rows": [1, 8, 64, 512, 4096, 32768]},
    {"kernel": "forward", "cols": 8192, "rows": [1, 8, 64, 512, 4096, 32768]},
    {"kernel": "backward", "cols": 8192, "rows": [1, 8, 64, 512, 4096, 32768]},
    {"kernel": "backward", "cols": 16384, "rows": [1, 8, 64, 512, 4096, 32768]},
]


def _microbench_rounds(repeats: int, warmup: int) -> Tuple[int, int]:
    return max(0, warmup), max(1, repeats)


def _full_forward_comprehensive_configs() -> List[Any]:
    return dora_fused_mod._build_triton_configs(
        dora_fused_mod._compose_comprehensive_meta_options(),
        dora_fused_mod._compose_or_backward_warps,
        dora_fused_mod._compose_or_backward_stages,
    )


def _full_backward_comprehensive_configs() -> List[Any]:
    return dora_fused_mod._build_triton_configs(
        dora_fused_mod._backward_comprehensive_meta_options(),
        dora_fused_mod._compose_or_backward_warps,
        dora_fused_mod._compose_or_backward_stages,
    )


def _backward_boundary_configs() -> List[Any]:
    triton = dora_fused_mod.triton
    if triton is None:
        return []

    configs = []
    for cfg in _full_backward_comprehensive_configs():
        bs = int(cfg.kwargs["BLOCK_SIZE"])
        rpp = int(cfg.kwargs["ROWS_PER_PROGRAM"])
        warps = int(cfg.num_warps)
        if bs == 8192 and rpp == 1 and warps in (4, 8):
            configs.append(cfg)
        elif bs == 16384 and rpp in (1, 2) and warps in (8, 16):
            configs.append(cfg)

    for warps in (8, 16):
        for stages in (2, 3, 4):
            configs.append(
                triton.Config(
                    {"BLOCK_SIZE": 32768, "ROWS_PER_PROGRAM": 1},
                    num_warps=warps,
                    num_stages=stages,
                )
            )
    return configs


def _row_probe_configs(kernel_name: str) -> List[Any]:
    if kernel_name == "forward":
        all_cfgs = _full_forward_comprehensive_configs()
    else:
        all_cfgs = _full_backward_comprehensive_configs()

    selected = []
    for cfg in all_cfgs:
        bs = int(cfg.kwargs["BLOCK_SIZE"])
        rpp = int(cfg.kwargs["ROWS_PER_PROGRAM"])
        warps = int(cfg.num_warps)
        stages = int(cfg.num_stages)
        keep = False
        if bs == 64:
            keep = (
                rpp in ((2, 4, 8) if kernel_name == "forward" else (2, 4)) and warps in (1, 2) and stages in (1, 2, 4)
            )
        elif bs == 128:
            keep = rpp == 2 and warps in (1, 2) and stages in (1, 2)
        elif bs == 512:
            keep = rpp == 1 and warps == 1 and stages in (1, 2, 3)
        elif bs == 1024:
            keep = rpp == 1 and warps in (2, 4, 8) and stages in (1, 2, 3, 4)
        elif bs == 2048:
            keep = rpp == 1 and warps in (2, 4, 8) and stages in (2, 3, 4, 5)
        elif bs == 4096:
            keep = rpp == 1 and warps in (4, 8) and stages in (2, 3, 4, 5)
        elif bs == 8192:
            keep = rpp == 1 and warps in (4, 8) and stages in (2, 3, 4, 5)
        elif bs == 16384 and kernel_name == "backward":
            keep = rpp == 1 and warps in (8, 16) and stages in (2, 3, 4)
        if keep:
            selected.append(cfg)
    return selected


def _make_forward_kernel_call(
    kernel: Any,
    cfg: Any,
    lora: torch.Tensor,
    base: torch.Tensor,
    mag: torch.Tensor,
    out: torch.Tensor,
    inner: torch.Tensor,
    scale: float,
    rows: int,
    cols: int,
):
    meta = _triton_config_to_dict(cfg)
    grid = ((rows + meta["ROWS_PER_PROGRAM"] - 1) // meta["ROWS_PER_PROGRAM"],)
    num_rows_bucket = dora_fused_mod._bucket_num_rows(rows)

    def _call():
        kernel[grid](
            lora,
            base,
            mag,
            out,
            inner,
            scale,
            rows,
            cols,
            num_rows_bucket,
            BLOCK_SIZE=meta["BLOCK_SIZE"],
            ROWS_PER_PROGRAM=meta["ROWS_PER_PROGRAM"],
            num_warps=meta["num_warps"],
            num_stages=meta["num_stages"],
        )

    return _call


def _make_backward_kernel_call(
    kernel: Any,
    cfg: Any,
    d_out: torch.Tensor,
    mag: torch.Tensor,
    d_lora: torch.Tensor,
    d_base: torch.Tensor,
    scale: float,
    rows: int,
    cols: int,
):
    meta = _triton_config_to_dict(cfg)
    grid = ((rows + meta["ROWS_PER_PROGRAM"] - 1) // meta["ROWS_PER_PROGRAM"],)
    num_rows_bucket = dora_fused_mod._bucket_num_rows(rows)

    def _call():
        kernel[grid](
            d_out,
            mag,
            d_lora,
            d_base,
            scale,
            rows,
            cols,
            num_rows_bucket,
            BLOCK_SIZE=meta["BLOCK_SIZE"],
            ROWS_PER_PROGRAM=meta["ROWS_PER_PROGRAM"],
            num_warps=meta["num_warps"],
            num_stages=meta["num_stages"],
        )

    return _call


def _rank_micro_configs(
    callables: Dict[str, Any],
    config_lookup: Dict[str, Any],
    warmup: int,
    iters: int,
    seed: int = 1234,
) -> List[Dict[str, Any]]:
    samples = _benchmark_interleaved_cuda_events(callables, warmup=warmup, iters=iters, seed=seed)
    ranked = []
    for label, vals in samples.items():
        stats = _timing_stats_from_samples(vals)
        ranked.append(
            {
                "config": _triton_config_to_dict(config_lookup[label]),
                "config_label": label,
                "timing": stats,
            }
        )
    ranked.sort(key=lambda item: item["timing"]["median_ms"])
    best = ranked[0]["timing"]["median_ms"]
    for item in ranked:
        item["slowdown_vs_best"] = round(item["timing"]["median_ms"] / best, 4) if best > 0 else 1.0
        item["gap_vs_best_pct"] = round((item["timing"]["median_ms"] / best - 1.0) * 100.0, 3) if best > 0 else 0.0
    return ranked


def _bench_forward_micro_case(
    rows: int,
    cols: int,
    dtype: torch.dtype,
    device: torch.device,
    configs: List[Any],
    repeats: int,
    warmup: int,
) -> Dict[str, Any]:
    _seed_all()
    scale = 0.5
    lora = torch.randn(rows, cols, device=device, dtype=dtype)
    base = torch.randn(rows, cols, device=device, dtype=dtype)
    mag = torch.randn(1, cols, device=device, dtype=dtype).abs().add_(0.5)
    out = torch.empty_like(lora)
    inner = torch.empty_like(lora)
    kernel = dora_fused_mod._fused_dora_forward_and_inner_kernel.fn

    config_lookup = {}
    callables = {}
    for cfg in configs:
        label = _triton_config_label(cfg)
        config_lookup[label] = cfg
        callables[label] = _make_forward_kernel_call(kernel, cfg, lora, base, mag, out, inner, scale, rows, cols)

    bench_warmup, bench_iters = _microbench_rounds(repeats, warmup)
    ranked = _rank_micro_configs(callables, config_lookup, bench_warmup, bench_iters)
    top = ranked[:8]

    del lora, base, mag, out, inner
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "kernel": "forward",
        "shape": [rows, cols],
        "dtype": _dtype_label(dtype),
        "config_count": len(configs),
        "warmup": bench_warmup,
        "iters": bench_iters,
        "best": top[0],
        "top_configs": top,
    }


def _bench_backward_micro_case(
    rows: int,
    cols: int,
    dtype: torch.dtype,
    device: torch.device,
    configs: List[Any],
    repeats: int,
    warmup: int,
) -> Dict[str, Any]:
    _seed_all()
    scale = 0.5
    d_out = torch.randn(rows, cols, device=device, dtype=dtype)
    mag = torch.randn(1, cols, device=device, dtype=dtype).abs().add_(0.5)
    d_lora = torch.empty_like(d_out)
    d_base = torch.empty_like(d_out)
    kernel = dora_fused_mod._fused_dora_backward_kernel.fn

    config_lookup = {}
    callables = {}
    for cfg in configs:
        label = _triton_config_label(cfg)
        config_lookup[label] = cfg
        callables[label] = _make_backward_kernel_call(kernel, cfg, d_out, mag, d_lora, d_base, scale, rows, cols)

    bench_warmup, bench_iters = _microbench_rounds(repeats, warmup)
    ranked = _rank_micro_configs(callables, config_lookup, bench_warmup, bench_iters)
    top = ranked[:8]

    del d_out, mag, d_lora, d_base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "kernel": "backward",
        "shape": [rows, cols],
        "dtype": _dtype_label(dtype),
        "config_count": len(configs),
        "warmup": bench_warmup,
        "iters": bench_iters,
        "best": top[0],
        "top_configs": top,
    }


def _bench_forward_anomaly_windows(
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
    verbose: bool,
) -> List[Dict[str, Any]]:
    configs = _full_forward_comprehensive_configs()
    results = []
    for window in FORWARD_ANOMALY_WINDOWS:
        for cols in window["cols"]:
            result = _bench_forward_micro_case(window["rows"], cols, dtype, device, configs, repeats, warmup)
            result["probe"] = window["label"]
            results.append(result)
            if verbose:
                best = result["best"]
                print(
                    f"  micro forward [{window['label']}] {window['rows']}x{cols}: "
                    f"{best['config_label']}  median={best['timing']['median_ms']:.4f}ms"
                )
    return results


def _bench_backward_boundary_extension(
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
    verbose: bool,
) -> List[Dict[str, Any]]:
    configs = _backward_boundary_configs()
    results = []
    for cols in BACKWARD_BOUNDARY_COLS:
        result = _bench_backward_micro_case(BACKWARD_BOUNDARY_ROWS, cols, dtype, device, configs, repeats, warmup)
        result["probe"] = "backward_boundary_extension"
        results.append(result)
        if verbose:
            best = result["best"]
            print(
                f"  micro backward [boundary] {BACKWARD_BOUNDARY_ROWS}x{cols}: "
                f"{best['config_label']}  median={best['timing']['median_ms']:.4f}ms"
            )
    return results


def _bench_row_bucket_sensitivity(
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
    verbose: bool,
) -> List[Dict[str, Any]]:
    results = []
    for probe in ROW_BUCKET_PROBES:
        configs = _row_probe_configs(probe["kernel"])
        for rows in probe["rows"]:
            if probe["kernel"] == "forward":
                result = _bench_forward_micro_case(rows, probe["cols"], dtype, device, configs, repeats, warmup)
            else:
                result = _bench_backward_micro_case(rows, probe["cols"], dtype, device, configs, repeats, warmup)
            result["probe"] = f"{probe['kernel']}_row_bucket"
            results.append(result)
            if verbose:
                best = result["best"]
                print(
                    f"  micro {probe['kernel']} [row-bucket] {rows}x{probe['cols']}: "
                    f"{best['config_label']}  median={best['timing']['median_ms']:.4f}ms"
                )
    return results


def bench_triton_micro(
    dtype: torch.dtype,
    device: torch.device,
    micro_iters: int,
    micro_warmup: int,
    verbose: bool,
) -> Dict[str, Any]:
    if device.type != "cuda" or not is_triton_available():
        print("  Triton microbenchmarks require CUDA + Triton; skipping.")
        return {}

    return {
        "forward_anomaly_windows": _bench_forward_anomaly_windows(dtype, device, micro_iters, micro_warmup, verbose),
        "backward_boundary_extension": _bench_backward_boundary_extension(
            dtype, device, micro_iters, micro_warmup, verbose
        ),
        "row_bucket_sensitivity": _bench_row_bucket_sensitivity(dtype, device, micro_iters, micro_warmup, verbose),
    }


def _format_micro_table(micro_results: Dict[str, Any]):
    if not micro_results:
        return

    rows = []
    for section_name, entries in micro_results.items():
        for entry in entries:
            best = entry["best"]
            shape_str = f"{entry['shape'][0]}x{entry['shape'][1]}"
            rows.append(
                [
                    section_name,
                    entry["probe"],
                    entry["kernel"],
                    shape_str,
                    best["config_label"],
                    f"{best['timing']['median_ms']:.4f}",
                    str(entry["config_count"]),
                ]
            )
    _print_table(
        ["Set", "Probe", "Kernel", "Shape", "Best Config", "Best ms", "Configs"],
        rows,
        "Section 10a: Triton Microbenchmarks -- Best Config per Probe",
    )

    top_rows = []
    for section_name, entries in micro_results.items():
        for entry in entries:
            for rank, candidate in enumerate(entry["top_configs"][:3], start=1):
                top_rows.append(
                    [
                        section_name,
                        f"{entry['shape'][0]}x{entry['shape'][1]}",
                        str(rank),
                        candidate["config_label"],
                        f"{candidate['timing']['median_ms']:.4f}",
                        f"{candidate['gap_vs_best_pct']:.3f}%",
                    ]
                )
    _print_table(
        ["Set", "Shape", "Rank", "Config", "Median ms", "Gap vs Best"],
        top_rows,
        "Section 10b: Triton Microbenchmarks -- Top-3 Candidates",
    )


# Section 4: E2E single DoRA layer — rank sweep + batch×seq sweep
# ====================================================================

E2E_SHAPES = [
    # (hidden, rank, batch, seqlen)
    # Rank sweep at fixed hidden (demonstrates writeup's "extreme ranks" claim)
    (4096, 16, 4, 2048),
    (4096, 64, 4, 2048),
    (4096, 128, 4, 2048),
    (4096, 256, 4, 2048),
    (4096, 384, 4, 2048),  # dominant production rank
    (4096, 512, 4, 2048),
    # Batch×seq sweep at production rank (shows scaling with activation volume)
    (4096, 384, 1, 1),  # single token inference
    (4096, 384, 2, 512),
    (4096, 384, 4, 2048),  # matches production config
    (4096, 384, 4, 4096),  # large seq
    # Large hidden at production rank
    (8192, 384, 4, 2048),
    (8192, 384, 2, 4096),
]

# MoE/MLA dimensions for extended benchmarks.
E2E_SHAPES_EXTENDED = E2E_SHAPES + [
    (512, 64, 4, 2048),  # MoE expert small rank
    (512, 384, 4, 2048),  # MoE expert production rank
    (1024, 384, 4, 2048),  # MoE expert
    (2048, 384, 4, 2048),  # Mamba-MoE
    (3072, 384, 4, 2048),  # Mamba-MoE
    (6144, 384, 4, 2048),  # DeepSeek-R1
    (12288, 384, 2, 2048),  # DeepSeek-R1 (reduced batch for memory)
    (16384, 384, 2, 2048),  # MLA (reduced batch)
    (28672, 384, 1, 2048),  # MLA kv_b_proj (batch=1 for memory)
]


class _DualOptimizer:
    """Muon for 2D params, AdamW for the rest.

    Minimal benchmark interface only (zero_grad, step). No state_dict support.
    """

    def __init__(self, model):
        params_2d = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
        params_other = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]
        self._muon = torch.optim.Muon(params_2d, lr=1e-3, adjust_lr_fn="match_rms_adamw")
        self._adamw = torch.optim.AdamW(params_other, lr=1e-3) if params_other else None

    def zero_grad(self, set_to_none=True):
        self._muon.zero_grad(set_to_none=set_to_none)
        if self._adamw:
            self._adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self._muon.step()
        if self._adamw:
            self._adamw.step()


def _make_optimizer(model):
    """Create optimizer: Muon (2D params) + AdamW (1D params), or AdamW fallback."""
    if hasattr(torch.optim, "Muon"):
        return _DualOptimizer(model)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)


def _bench_one_e2e(
    hidden: int,
    rank: int,
    batch: int,
    seqlen: int,
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
    config_label: str,
    fused_fwd: int,
    fused_bwd: int,
) -> Dict:
    _set_fused_config(fused_fwd, fused_bwd)
    _seed_all()

    scaling = 0.5
    model = SimpleLoraDoraBlock(hidden, rank, scaling, dtype, device).to(device)
    model.train()
    x = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype)
    y = torch.randn_like(x)
    optimizer = _make_optimizer(model)

    # Warmup (extra for Triton autotuning)
    for _ in range(max(warmup, 5)):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        _sync(device)

    # Free warmup gradients before memory measurement baseline
    optimizer.zero_grad(set_to_none=True)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    fwd_times: List[float] = []
    bwd_times: List[float] = []
    step_times: List[float] = []

    for _ in range(max(1, repeats)):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        out = model(x)
        loss = F.mse_loss(out, y)
        _sync(device)
        t1 = time.perf_counter()
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        optimizer.step()
        _sync(device)
        t3 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000.0)
        bwd_times.append((t2 - t1) * 1000.0)
        step_times.append((t3 - t0) * 1000.0)

    peak_mb = None
    if device.type == "cuda":
        try:
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        except Exception:
            pass

    del model, x, y, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "config": config_label,
        "shape": {"hidden": hidden, "rank": rank, "batch": batch, "seqlen": seqlen},
        "fwd_ms": round(_stat_from_list(fwd_times).mean_ms, 4),
        "bwd_ms": round(_stat_from_list(bwd_times).mean_ms, 4),
        "step_ms": round(_stat_from_list(step_times).mean_ms, 4),
        "peak_mem_mb": round(peak_mb, 1) if peak_mb else None,
        "timings": {
            "fwd": _timing_stats_from_samples(fwd_times),
            "bwd": _timing_stats_from_samples(bwd_times),
            "step": _timing_stats_from_samples(step_times),
        },
    }


def bench_e2e(
    dtype: torch.dtype, device: torch.device, repeats: int, warmup: int, verbose: bool, shapes=None
) -> List[Dict]:
    results = []
    # Include baselines alongside our 4 DoRAFactors configs.
    all_e2e_configs = list(CONFIGS) + [
        ("baseline_dense_ba", 0, 0),
        ("baseline_hf_peft", 0, 0),
    ]
    for hidden, rank, batch, seqlen in shapes or E2E_SHAPES:
        for label, fwd, bwd in all_e2e_configs:
            if label == "baseline_hf_peft":
                patch_ctx = _HfPeftNormPatch()
            elif label == "baseline_dense_ba":
                patch_ctx = _DenseBaNormPatch()
            else:
                patch_ctx = None
            with ExitStack() as stack:
                if patch_ctx:
                    stack.enter_context(patch_ctx)
                try:
                    row = _bench_one_e2e(hidden, rank, batch, seqlen, dtype, device, repeats, warmup, label, fwd, bwd)
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    row = {
                        "config": label,
                        "shape": {"hidden": hidden, "rank": rank, "batch": batch, "seqlen": seqlen},
                        "error": "OOM",
                    }
            results.append(row)
            if verbose:
                if "error" in row:
                    print(f"  e2e h={hidden} r={rank} bs={batch} seq={seqlen} [{label}]: OOM")
                else:
                    print(
                        f"  e2e h={hidden} r={rank} bs={batch} seq={seqlen} [{label}]: "
                        f"fwd={row['fwd_ms']:.3f}ms  bwd={row['bwd_ms']:.3f}ms  "
                        f"step={row['step_ms']:.3f}ms  peak={row['peak_mem_mb']}MB"
                    )
    return results


# ====================================================================
# Section 5: Memory profile
# ====================================================================


def _measure_memory_phases(
    hidden: int,
    rank: int,
    batch: int,
    seqlen: int,
    dtype: torch.dtype,
    device: torch.device,
    config_label: str,
    fused_fwd: int,
    fused_bwd: int,
) -> Dict:
    _set_fused_config(fused_fwd, fused_bwd)
    _seed_all()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    scaling = 0.5
    model = SimpleLoraDoraBlock(hidden, rank, scaling, dtype, device).to(device)
    model.train()
    _sync(device)

    static_mem_mb = None
    if device.type == "cuda":
        static_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    x = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype)
    y = torch.randn_like(x)
    optimizer = _make_optimizer(model)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        _sync(device)

    # Measure forward peak
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss = F.mse_loss(out, y)
    _sync(device)

    post_fwd_mb = None
    if device.type == "cuda":
        post_fwd_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    # Measure backward peak
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss.backward()
    _sync(device)

    post_bwd_mb = None
    if device.type == "cuda":
        post_bwd_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    del model, x, y, optimizer, out, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "config": config_label,
        "shape": {"hidden": hidden, "rank": rank, "batch": batch, "seqlen": seqlen},
        "static_mem_mb": round(static_mem_mb, 1) if static_mem_mb else None,
        "post_fwd_peak_mb": round(post_fwd_mb, 1) if post_fwd_mb else None,
        "post_bwd_peak_mb": round(post_bwd_mb, 1) if post_bwd_mb else None,
    }


def bench_memory(dtype: torch.dtype, device: torch.device, verbose: bool, shapes=None) -> List[Dict]:
    results = []
    for hidden, rank, batch, seqlen in shapes or E2E_SHAPES:
        shape_results = []
        for label, fwd, bwd in CONFIGS:
            row = _measure_memory_phases(hidden, rank, batch, seqlen, dtype, device, label, fwd, bwd)
            shape_results.append(row)
            results.append(row)

        # Compute delta between dorafactors_eager and dorafactors_fused_bwd
        baseline = shape_results[0]
        fused_bwd_row = shape_results[2]  # index 2 = dorafactors_fused_bwd config
        if baseline.get("post_bwd_peak_mb") is not None and fused_bwd_row.get("post_bwd_peak_mb") is not None:
            delta = fused_bwd_row["post_bwd_peak_mb"] - baseline["post_bwd_peak_mb"]
            fused_bwd_row["delta_vs_baseline_mb"] = round(delta, 1)

        if verbose:
            for row in shape_results:
                delta_str = ""
                if "delta_vs_baseline_mb" in row:
                    delta_str = f"  delta={row['delta_vs_baseline_mb']:.1f}MB"
                print(
                    f"  memory h={hidden} r={rank} [{row['config']}]: "
                    f"static={row['static_mem_mb']}MB  fwd_peak={row['post_fwd_peak_mb']}MB  "
                    f"bwd_peak={row['post_bwd_peak_mb']}MB{delta_str}"
                )
    return results


# ====================================================================
# Section 7: Stability demo — catastrophic cancellation near mag ≈ 1
# ====================================================================

STABILITY_SHAPES = [(1, 4096), (2048, 8192)]
STABILITY_MAGNITUDES = [0.9, 0.99, 0.999, 0.9999, 1.0, 1.0001, 1.001, 1.01, 1.1]


def bench_stability(device: torch.device, verbose: bool) -> List[Dict]:
    """Demonstrate catastrophic cancellation when mag approaches 1 in bf16.

    Compares five methods against fp64 ground truth:
      1. naive bf16:   m * (scale * lora + base) - base
      2. stable bf16:  (m - 1) * base + m * scale * lora  (our formulation)
      3. fused bf16:   fused_dora_compose(...)
      4. quantization floor:  fp64 -> bf16 -> fp64 round-trip (best possible bf16)
    """
    results = []
    _seed_all()

    for rows, cols in STABILITY_SHAPES:
        # Fixed random data, same across all magnitudes for this shape
        lora_fp64 = torch.randn(rows, cols, device=device, dtype=torch.float64)
        base_fp64 = torch.randn(rows, cols, device=device, dtype=torch.float64)
        scale = 0.5

        for m_val in STABILITY_MAGNITUDES:
            mag_fp64 = torch.full((1, cols), m_val, device=device, dtype=torch.float64)

            # Ground truth in fp64
            ref_fp64 = (mag_fp64 - 1.0) * base_fp64 + mag_fp64 * (scale * lora_fp64)

            # Convert inputs to bf16
            lora_bf16 = lora_fp64.to(torch.bfloat16)
            base_bf16 = base_fp64.to(torch.bfloat16)
            mag_bf16 = mag_fp64.to(torch.bfloat16)

            # Method 1: naive bf16 — catastrophic cancellation path
            #   m * (scale * lora + base) - base
            naive = (mag_bf16 * (scale * lora_bf16 + base_bf16) - base_bf16).to(torch.float64)

            # Method 2: stable bf16 — our formulation
            #   (m - 1) * base + m * (scale * lora)
            stable = ((mag_bf16 - 1.0) * base_bf16 + mag_bf16 * (scale * lora_bf16)).to(torch.float64)

            # Method 3: fused bf16 — Triton kernel (falls back to PyTorch if Triton unavailable)
            triton_actually_ran = is_triton_available() and device.type == "cuda"
            fused_out = fused_dora_compose(lora_bf16.clone(), base_bf16, mag_bf16, scale, inplace=True).to(
                torch.float64
            )

            # Method 4: quantization floor — best possible bf16 result
            quant_floor = ref_fp64.to(torch.bfloat16).to(torch.float64)

            def _error_stats(result_fp64, ref):
                diff = result_fp64 - ref
                abs_diff = diff.abs()
                return {
                    "max_abs_error": round(float(abs_diff.max().item()), 10),
                    "max_rel_error": round(float((abs_diff / ref.abs().clamp_min(1e-12)).max().item()), 10),
                    "mean_abs_error": round(float(abs_diff.mean().item()), 10),
                }

            entry = {
                "shape": [rows, cols],
                "m": m_val,
                "triton_actually_ran": triton_actually_ran,
                "methods": {
                    "naive_bf16": _error_stats(naive, ref_fp64),
                    "stable_bf16": _error_stats(stable, ref_fp64),
                    "fused_bf16": _error_stats(fused_out, ref_fp64),
                    "quantization_floor": {
                        "max_abs_error": round(float((quant_floor - ref_fp64).abs().max().item()), 10),
                    },
                },
            }
            results.append(entry)

            if verbose:
                n = entry["methods"]["naive_bf16"]
                s = entry["methods"]["stable_bf16"]
                f = entry["methods"]["fused_bf16"]
                q = entry["methods"]["quantization_floor"]
                print(
                    f"  stability ({rows}x{cols}, m={m_val}): "
                    f"naive={n['max_abs_error']:.4e}  stable={s['max_abs_error']:.4e}  "
                    f"fused={f['max_abs_error']:.4e}  floor={q['max_abs_error']:.4e}"
                )

            del naive, stable, fused_out, quant_floor
            gc.collect()

        del lora_fp64, base_fp64
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ====================================================================
# Section 8: d_mag Precision Sweep
# ====================================================================

PRECISION_HIDDEN_DIMS = [2048, 3840, 4096, 8192]
PRECISION_BASE_SCALES = [0.1, 1.0, 10.0, 62.0, 128.0]
PRECISION_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def bench_dmag_precision(device: torch.device, verbose: bool) -> List[Dict]:
    """Sweep base_scale × hidden_dim × dtype measuring d_mag fidelity.

    For each config, runs FusedDoRAComposeFunction backward and compares
    d_mag against an fp64 reference.  Reports cosine similarity, relative
    error, and absorption rate.
    """
    if device.type != "cuda":
        print("  d_mag precision sweep requires CUDA; skipping.")
        return []

    from peft.tuners.lora.dora_fused import FusedDoRAComposeFunction

    results = []
    batch, seqlen = 4, 2048
    rank = 64
    scale = 0.3

    for hidden in PRECISION_HIDDEN_DIMS:
        for base_scale in PRECISION_BASE_SCALES:
            for dt in PRECISION_DTYPES:
                if dt in (torch.float16, torch.bfloat16) and device.type != "cuda":
                    continue
                _seed_all()
                total_tokens = batch * seqlen

                lora = torch.randn(total_tokens, hidden, device=device, dtype=dt, requires_grad=True)
                base = torch.randn(total_tokens, hidden, device=device, dtype=dt) * base_scale
                base.requires_grad_(True)
                mag = (torch.rand(1, hidden, device=device, dtype=dt) + 0.5).requires_grad_(True)
                d_out = torch.randn(total_tokens, hidden, device=device, dtype=dt)

                # Fused backward
                out = FusedDoRAComposeFunction.apply(lora, base, mag, scale)
                out.backward(d_out)
                d_mag = mag.grad.clone()

                # fp64 reference
                lora64 = lora.detach().to(torch.float64).requires_grad_(True)
                base64 = base.detach().to(torch.float64).requires_grad_(True)
                mag64 = mag.detach().to(torch.float64).requires_grad_(True)
                inner64 = scale * lora64 + base64
                out64 = mag64 * inner64 - base64
                out64.backward(d_out.to(torch.float64))
                d_mag_ref = mag64.grad.clone()

                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    d_mag.float().flatten(),
                    d_mag_ref.float().flatten(),
                    dim=0,
                ).item()

                # Relative L2 error
                rel_err = (d_mag.double() - d_mag_ref).norm().item() / d_mag_ref.norm().clamp_min(1e-30).item()

                # Absorption rate
                scaled_lora_abs = (scale * lora.detach().to(torch.float64)).abs()
                base_abs = base.detach().to(torch.float64).abs()
                if dt == torch.bfloat16:
                    ulp_factor = 2**-8
                elif dt == torch.float16:
                    ulp_factor = 2**-11
                else:
                    ulp_factor = 2**-24
                absorbed = scaled_lora_abs < (0.5 * ulp_factor * base_abs)
                absorption_rate = absorbed.float().mean().item()

                row = {
                    "hidden": hidden,
                    "base_scale": base_scale,
                    "dtype": str(dt).split(".")[-1],
                    "cos_sim": round(cos_sim, 6),
                    "rel_err": rel_err,
                    "absorption_rate": round(absorption_rate, 4),
                }
                results.append(row)
                if verbose:
                    print(
                        f"  hidden={hidden} base_scale={base_scale} dtype={dt}: "
                        f"cos_sim={cos_sim:.6f} rel_err={rel_err:.4e} absorption={absorption_rate:.4f}"
                    )

                # Cleanup
                del lora, base, mag, d_out, out, d_mag
                del lora64, base64, mag64, inner64, out64, d_mag_ref
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return results


def _format_precision_table(results: List[Dict]):
    """Format d_mag precision sweep results."""
    headers = ["hidden", "base_scale", "dtype", "cos_sim", "rel_err", "absorption"]
    rows = []
    for r in results:
        rows.append(
            [
                str(r["hidden"]),
                str(r["base_scale"]),
                r["dtype"],
                f"{r['cos_sim']:.6f}",
                f"{r['rel_err']:.4e}",
                f"{r['absorption_rate']:.4f}",
            ]
        )
    _print_table(headers, rows, "Section 8: d_mag Precision Sweep")


# ====================================================================
# Section 9: VRAM Impact — eager vs fused-inner compose
# ====================================================================


def bench_vram_impact(dtype: torch.dtype, device: torch.device, verbose: bool) -> List[Dict]:
    """Compare peak VRAM for eager compose vs fused-inner compose.

    Uses SimpleLoraDoraBlock at production dims to measure realistic VRAM.
    """
    if device.type != "cuda":
        print("  VRAM impact benchmark requires CUDA; skipping.")
        return []

    results = []
    rank = 384
    scaling = 0.5
    configs = [
        # (hidden, batch, seqlen)
        (4096, 4, 2048),
        (4096, 4, 4096),
        (8192, 4, 2048),
        (8192, 2, 4096),
    ]

    for hidden, batch, seqlen in configs:
        for label, fwd, bwd in [
            ("eager", 0, 0),
            ("fused_fwd_bwd", 1, 1),
        ]:
            _seed_all()
            _set_fused_config(fwd, bwd)

            try:
                model = SimpleLoraDoraBlock(hidden, rank, scaling, dtype, device).to(device)
                model.train()
                x = torch.randn(batch, seqlen, hidden, device=device, dtype=dtype)

                # Warmup
                for _ in range(2):
                    out = model(x)
                    loss = out.sum()
                    loss.backward()
                    model.zero_grad(set_to_none=True)

                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                before = torch.cuda.memory_allocated(device)

                out = model(x)
                loss = out.sum()
                loss.backward()

                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                delta_mb = (peak - before) / (1024**2)

                row = {
                    "hidden": hidden,
                    "batch": batch,
                    "seqlen": seqlen,
                    "config": label,
                    "peak_delta_mb": round(delta_mb, 1),
                }
                results.append(row)
                if verbose:
                    print(
                        f"  {hidden}x{hidden} r={rank} b={batch} seq={seqlen} [{label}]: peak_delta={delta_mb:.1f} MB"
                    )

                del model, x, out, loss
            except Exception as e:
                row = {
                    "hidden": hidden,
                    "batch": batch,
                    "seqlen": seqlen,
                    "config": label,
                    "error": str(e),
                }
                results.append(row)
                if verbose:
                    print(f"  {hidden}x{hidden} [{label}]: ERROR {e}")

            gc.collect()
            torch.cuda.empty_cache()

    return results


def _format_vram_table(results: List[Dict]):
    """Format VRAM impact results."""
    headers = ["shape", "config", "peak_delta_MB"]
    rows = []
    for r in results:
        shape = f"{r['hidden']}x{r['hidden']} b={r['batch']} seq={r['seqlen']}"
        if "error" in r:
            rows.append([shape, r["config"], f"ERROR: {r['error']}"])
        else:
            rows.append([shape, r["config"], f"{r['peak_delta_mb']:.1f}"])
    _print_table(headers, rows, "Section 9: VRAM Impact — Eager vs Fused Fwd+Bwd")


# ====================================================================
# Section 6: Model benchmark (opt-in) — multi-loader cascade
# ====================================================================

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "unsloth/Mistral-Small-3.2-24B-Instruct-2506",
]


def _extra_vision_module_prefixes(model: Optional[nn.Module] = None) -> Tuple[str, ...]:
    return ()


def _looks_like_vision_module_name(name: str, extra_prefixes: Tuple[str, ...] = ()) -> bool:
    norm = name.lower()
    if any(norm == prefix.rstrip(".") or norm.startswith(prefix) for prefix in extra_prefixes):
        return True
    if "mm_projector" in norm or "multi_modal_projector" in norm:
        return True

    tokens = {tok for tok in norm.replace("_", ".").split(".") if tok}
    return any(token in tokens for token in ("vision", "visual", "vit", "image", "projector"))


def _summarize_adapted_module_names(
    module_names: List[str], extra_vision_prefixes: Tuple[str, ...] = ()
) -> Dict[str, Any]:
    sorted_names = sorted(set(module_names))
    vision_names = [name for name in sorted_names if _looks_like_vision_module_name(name, extra_vision_prefixes)]
    lm_names = [name for name in sorted_names if name not in vision_names]
    suffix_counts = Counter(name.rsplit(".", 1)[-1] for name in sorted_names)
    top_suffix_examples = []
    for suffix, count in suffix_counts.most_common(8):
        examples = [name for name in sorted_names if name.endswith(f".{suffix}") or name == suffix][:3]
        top_suffix_examples.append({"suffix": suffix, "count": count, "examples": examples})

    return {
        "total": len(sorted_names),
        "lm_count": len(lm_names),
        "vision_count": len(vision_names),
        "top_suffixes": suffix_counts.most_common(8),
        "top_suffix_examples": top_suffix_examples,
        "lm_examples": lm_names[:5],
        "vision_examples": vision_names[:5],
    }


def _print_adapted_module_summary(summary: Dict[str, Any]) -> None:
    print(f"  adapted modules: total={summary['total']}  lm={summary['lm_count']}  vision={summary['vision_count']}")
    if summary["top_suffix_examples"]:
        print("    top suffixes:")
        for item in summary["top_suffix_examples"]:
            examples = ", ".join(item["examples"])
            example_str = f" e.g. {examples}" if examples else ""
            print(f"      {item['suffix']}:{item['count']}{example_str}")
    if summary["lm_examples"]:
        print(f"    lm examples: {', '.join(summary['lm_examples'])}")
    if summary["vision_examples"]:
        print(f"    vision examples: {', '.join(summary['vision_examples'])}")


def _build_model_lora_config(model: nn.Module, rank: int, adapt_vision: bool):
    from peft import LoraConfig

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules="all-linear",
        use_dora=True,
        task_type="CAUSAL_LM",
    )
    lora_config = _maybe_include_all_linear_layers(lora_config, model)

    target_modules = getattr(lora_config, "target_modules", None)
    if not isinstance(target_modules, (set, list, tuple)) or not target_modules:
        raise ValueError("Failed to expand target_modules='all-linear' for this model.")

    target_list = sorted(target_modules)
    extra_vision_prefixes = _extra_vision_module_prefixes(model)
    if not adapt_vision:
        filtered = [name for name in target_list if not _looks_like_vision_module_name(name, extra_vision_prefixes)]
        if filtered:
            target_list = filtered

    lora_config.target_modules = target_list
    return lora_config


def _select_attn_implementation(
    model_id: str, *, explicit: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Choose a safe attention backend for benchmark model loading.

    We prefer ``flash_attention_2`` by default because the benchmark is meant
    to reflect the high-performance production path. Qwen3.5 is the exception:
    leave ``attn_implementation`` unset so the model/runtime can pick its own
    backend instead of forcing a benchmark-side override.
    """
    if explicit:
        return explicit, None

    model_type = None
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", None)
    except Exception:
        pass

    model_id_hint = model_id.lower()
    if model_type == "qwen3_5" or "qwen3.5" in model_id_hint:
        reason = "Qwen3.5 defaults to model-selected attention; leaving attn_implementation unset"
        return None, reason

    return "flash_attention_2", None


def _prefers_image_text_to_text_loader(model_id: str) -> bool:
    """Decide whether a multimodal loader should be tried before CausalLM.

    Some recent Qwen checkpoints advertise multimodal conditional-generation
    architectures in config while still loading through AutoModelForCausalLM.
    That silently drops the vision tower from the benchmark, so prefer the
    image-text loader whenever config makes the multimodal intent explicit.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return "qwen3.5" in model_id.lower()

    architectures = getattr(cfg, "architectures", None) or []
    has_vision_config = getattr(cfg, "vision_config", None) is not None
    return has_vision_config and any("ConditionalGeneration" in arch for arch in architectures)


def _load_model_cascade(model_id: str, torch_dtype, device_map="auto", **extra_kwargs):
    """Multi-loader cascade with model-aware attention backend selection.

    Prefers AutoModelForImageTextToText for multimodal conditional-generation
    configs, otherwise tries AutoModelForCausalLM first.
    Most models use flash attention 2 by default. Known-bad stacks are routed
    to a safer backend up front because an FA2 illegal access poisons the CUDA
    context and prevents in-process fallback.
    """
    from transformers import AutoModelForCausalLM

    loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText

        if _prefers_image_text_to_text_loader(model_id):
            loaders = [AutoModelForImageTextToText, AutoModelForCausalLM]
        else:
            loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass

    explicit_attn_impl = extra_kwargs.pop("attn_implementation", None)
    attn_impl, attn_reason = _select_attn_implementation(model_id, explicit=explicit_attn_impl)
    if attn_reason:
        print(f"  note: {attn_reason}")
    common_kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
        **extra_kwargs,
    )
    if attn_impl is not None:
        common_kwargs["attn_implementation"] = attn_impl

    loader_errors = []
    for cls in loaders:
        try:
            model = cls.from_pretrained(
                model_id,
                dtype=torch_dtype,
                **common_kwargs,
            )
            return model
        except TypeError as e:
            # Backward compatibility with older Transformers that only accept torch_dtype.
            if "dtype" in str(e):
                try:
                    model = cls.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        **common_kwargs,
                    )
                    return model
                except Exception as e2:
                    loader_errors.append((cls.__name__, e2))
                    continue
            loader_errors.append((cls.__name__, e))
            continue
        except Exception as e:
            loader_errors.append((cls.__name__, e))
            continue
    if len(loader_errors) == 1:
        raise loader_errors[0][1]

    first_name, first_err = loader_errors[0]
    detail_lines = [f"{name}: {type(err).__name__}: {err}" for name, err in loader_errors]
    raise RuntimeError(
        f"All model loaders failed for {model_id}. "
        f"First failure came from {first_name}: {type(first_err).__name__}: {first_err}\n"
        f"Loader attempts:\n  " + "\n  ".join(detail_lines)
    ) from first_err


def _ref_get_weight_norm_linear(self, *, base_weight, lora_A_w, lora_B_w, scaling, chunk_size=None):
    """Drop-in replacement for DoraLinearLayer._get_weight_norm_linear that
    faithfully reproduces the reference HF PEFT approach:

    1. Construct an identity matrix of size [in_features, in_features]
    2. Pass it through both LoRA layers to materialize dense B@A
    3. Add to base weight and compute row-wise L2 norm

    This replicates the actual upstream code (dora.reference_hf_peft.py:73-74):
        x_eye = torch.eye(lora_A.weight.shape[1], ...)
        lora_weight = lora_B(lora_A(x_eye)).T

    The identity matrix is O(d_in^2) — the dominant memory cost that our
    factored approach eliminates entirely.
    """
    W_t = transpose(base_weight, self.fan_in_fan_out)
    in_features = lora_A_w.shape[1]

    # Faithfully reproduce reference: identity matrix → forward through LoRA layers.
    # The actual PEFT code (dora.reference_hf_peft.py:73) creates the identity in
    # the activation dtype (x.dtype), not the dequantized weight dtype.  Since this
    # patched function is called from our forward (where base_weight is dequantized
    # to fp32), we use lora_A_w.dtype to match the reference behavior.  The weight
    # is cast to the same dtype before addition, mirroring reference line 78:
    #   weight = weight.to(x.dtype)
    compute_dtype = lora_A_w.dtype
    x_eye = torch.eye(in_features, device=lora_A_w.device, dtype=compute_dtype)
    lora_weight = (x_eye @ lora_A_w.T @ lora_B_w.T).T  # equivalent to lora_B(lora_A(x_eye)).T

    W_t = W_t.to(dtype=compute_dtype)
    weight = W_t + scaling * lora_weight
    weight_norm = torch.linalg.norm(weight, dim=1).to(compute_dtype)

    del x_eye, lora_weight  # free explicitly, as reference would at function exit

    return weight_norm


class _HfPeftNormPatch:
    """Context manager that monkey-patches all DoraLinearLayer instances to use
    the reference dense-B@A norm computation.  Used by the baseline_hf_peft
    model config to demonstrate the memory cost of the upstream approach."""

    def __enter__(self):
        self._orig = DoraLinearLayer._get_weight_norm_linear
        DoraLinearLayer._get_weight_norm_linear = _ref_get_weight_norm_linear
        return self

    def __exit__(self, *args):
        DoraLinearLayer._get_weight_norm_linear = self._orig


def _dense_ba_get_weight_norm_linear(self, *, base_weight, lora_A_w, lora_B_w, scaling, chunk_size=None):
    """Direct B@A matmul norm -- the 'obvious fix' without identity matrix.

    Still materializes the full [out, in] dense product via B@A, but skips the
    wasteful identity-matrix detour.  This is the approach a reviewer would
    suggest as the natural alternative to PEFT's identity-matrix path.

    Upcasts to fp32 for bf16/fp16 inputs, matching the factored approach
    (dora.py line 557).  Without upcast, memory comparison is apples-to-oranges
    because factored always computes norms in fp32.

    WHY DENSE B@A STILL OOMS IN PRODUCTION
    =======================================
    Even without the identity matrix, B@A materializes the full [out, in]
    product in fp32.  For production VLM training (32B+ models, r=384,
    all-linear DoRA), the costliest layers are:

      Qwen2.5-VL-32B  down_proj [5120, 27648]: B@A peak ~1620 MB (fp32)
      InternVL3.5-38B  down_proj [5120, 25600]: B@A peak ~1500 MB (fp32)

    In a realistic GRPO training stack:
      - vLLM colocated (23-32% GPU reserved for generation)
      - ZeRO-2 with offloaded optimizer
      - Gradient checkpointing (recomputes forward during backward,
        so the 1.6 GB norm peak occurs DURING checkpoint recompute
        on top of backward activations and gradients)
      - Vision tokens extend effective seqlen to 8K+ (video input)
      - Per-device batch_size=3-4

    This leaves <20 GB headroom on B200 (180 GB).  The dense B@A norm's
    ~1.6 GB contiguous allocation is the straw that breaks the allocator,
    especially after hundreds of steps of fragmentation.

    The factored approach reduces this to ~0.5 GB (chunked, same fp32),
    saving ~1.1 GB per layer -- enough to keep training alive.

    See also: norm microbenchmark (Section 1) for per-shape memory
    reduction ratios (3-70x depending on shape and dtype).
    """
    W_t = transpose(base_weight, self.fan_in_fan_out)
    dtype = W_t.dtype

    # Upcast to fp32 for numerical stability, matching factored approach
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    W_t = W_t.to(compute_dtype)
    lora_A_fp = lora_A_w.to(compute_dtype)
    lora_B_fp = lora_B_w.to(compute_dtype)

    lora_weight = lora_B_fp @ lora_A_fp  # direct matmul, no identity matrix
    weight = W_t + scaling * lora_weight
    weight_norm = torch.linalg.norm(weight, dim=1).to(dtype)
    del lora_weight, W_t, lora_A_fp, lora_B_fp

    return weight_norm


class _DenseBaNormPatch:
    """Context manager that monkey-patches DoraLinearLayer to use direct B@A
    matmul norm.  Used by baseline_dense_ba to show the 'obvious fix'."""

    def __enter__(self):
        self._orig = DoraLinearLayer._get_weight_norm_linear
        DoraLinearLayer._get_weight_norm_linear = _dense_ba_get_weight_norm_linear
        return self

    def __exit__(self, *args):
        DoraLinearLayer._get_weight_norm_linear = self._orig


def _get_vocab_size_from_config(config: Any) -> int:
    """Best-effort vocab size lookup across text-only and multimodal configs."""
    v = getattr(config, "vocab_size", None)
    if isinstance(v, int) and v > 0:
        return v
    text_cfg = getattr(config, "text_config", None)
    v = getattr(text_cfg, "vocab_size", None)
    if isinstance(v, int) and v > 0:
        return v
    return 32000


def _get_model_forward_param_names(model: nn.Module) -> set[str]:
    queue = [model]
    seen = set()
    param_names = set()

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            queue.append(get_base_model())
        except Exception:
            pass

    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        forward = getattr(candidate, "forward", None)
        if callable(forward):
            try:
                signature = inspect.signature(forward)
                param_names.update(
                    name
                    for name, param in signature.parameters.items()
                    if name != "self"
                    and param.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                )
            except (TypeError, ValueError):
                pass

        queue.extend(getattr(candidate, attr, None) for attr in ("base_model", "model"))

    return param_names


def _build_text_model_inputs(model: nn.Module, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    input_kwargs = {"input_ids": input_ids}
    forward_param_names = _get_model_forward_param_names(model)
    if "attention_mask" in forward_param_names:
        input_kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)
    if "token_type_ids" in forward_param_names:
        input_kwargs["token_type_ids"] = torch.zeros_like(input_ids, dtype=torch.long)
    return input_kwargs


def _forward_model(model: nn.Module, model_inputs: Dict[str, torch.Tensor], **extra_kwargs):
    return model(**model_inputs, **extra_kwargs)


_EAGER_COSSIM_CONFIGS = {
    "dorafactors_fused_fwd",
    "dorafactors_fused_bwd",
    "dorafactors_fully_fused",
    "baseline_dense_ba",
    "baseline_hf_peft",
}


def _cosine_similarity_flat(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    """Cosine similarity between flattened tensors, computed in float64.

    Float64 is required because the vectors being compared are near-parallel
    (same model, different kernel dispatch paths), and catastrophic cancellation
    in the float32 dot product can flip the last few significant digits of the
    cosine — exactly the digits that distinguish 'identical output' from
    'numerically drifted'.  Callers are responsible for bounding tensor size;
    this function will allocate two float64 copies on the input device.
    """
    lhs_flat = lhs.reshape(-1).to(torch.float64)
    rhs_flat = rhs.reshape(-1).to(torch.float64)
    denom = torch.linalg.vector_norm(lhs_flat) * torch.linalg.vector_norm(rhs_flat)
    if float(denom.item()) == 0.0:
        return float("nan")
    cosine = float((torch.dot(lhs_flat, rhs_flat) / denom).item())
    if math.isfinite(cosine):
        return max(-1.0, min(1.0, cosine))
    return cosine


def _collect_eager_comparison_metrics(
    label: str, eager_logits: Optional[torch.Tensor], current_logits: torch.Tensor
) -> Dict[str, float]:
    if eager_logits is None or label not in _EAGER_COSSIM_CONFIGS:
        return {}

    cosine_to_eager = _cosine_similarity_flat(eager_logits, current_logits)
    metrics = {"cosine_similarity_to_eager": cosine_to_eager}
    if label == "dorafactors_fully_fused":
        metrics["max_abs_diff"] = float(torch.max(torch.abs(eager_logits - current_logits)).item())
        metrics["cosine_similarity"] = cosine_to_eager
    return metrics


def _display_cosine_similarity_to_eager(row: Dict[str, Any]) -> Optional[float]:
    if row.get("config") == "dorafactors_fully_fused" and row.get("cosine_similarity") is not None:
        return None
    return row.get("cosine_similarity_to_eager")


def _forward_lm_with_loss(
    model: nn.Module,
    model_inputs: Dict[str, torch.Tensor],
    loss_tokens: int = 0,
) -> Tuple[Any, torch.Tensor]:
    """Forward pass returning (output, loss).

    Args:
        loss_tokens: If >0, compute loss only over the last N tokens,
            simulating GRPO/RLHF where loss is on the response only.
            This avoids the massive [bs, seqlen, vocab] fp32 spike that
            masks memory differences between norm methods.
            If 0 (default), compute loss over all tokens (standard SFT).
    """
    input_ids = model_inputs["input_ids"]
    if loss_tokens > 0:
        loss_tokens = min(loss_tokens, max(0, input_ids.shape[1] - 1))
    if loss_tokens <= 0:
        # Full-sequence loss: try model-native labels path first
        try:
            out = _forward_model(model, model_inputs, labels=input_ids)
            loss = getattr(out, "loss", None)
            if loss is not None:
                return out, loss
        except Exception:
            pass

    # Forward without labels (or loss_tokens mode).
    # Pass logits_to_keep to slice hidden states BEFORE the lm_head, avoiding
    # the full [bs, seqlen, vocab] allocation.  PEFT wrappers use (*args, **kwargs)
    # and forward transparently, so we pass unconditionally and catch TypeError
    # from models that don't support the parameter.
    out = None
    if loss_tokens > 0:
        try:
            out = _forward_model(model, model_inputs, logits_to_keep=loss_tokens + 1)
        except TypeError:
            pass  # model doesn't accept logits_to_keep; fall through
    if out is None:
        out = _forward_model(model, model_inputs)
    logits = out.logits

    if loss_tokens > 0:
        # GRPO-like: only compute loss over last loss_tokens positions.
        # If the model respected logits_to_keep, logits is already sliced.
        # Otherwise, slice here.
        if logits.shape[1] > loss_tokens + 1:
            logits = logits[:, -(loss_tokens + 1) :, :]
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, -loss_tokens:].contiguous()
    else:
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    return out, loss


def bench_models(
    model_ids: List[str],
    rank: int,
    batch: int,
    seqlen: int,
    grad_accum: int,
    gradient_checkpointing: bool,
    device: torch.device,
    repeats: int,
    warmup: int,
    verbose: bool,
    loss_tokens: int = 0,
    adapt_vision: bool = False,
    show_adapted_modules: bool = False,
    device_map: str = "single",
) -> List[Dict]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed, skipping model benchmark.", file=sys.stderr)
        return []
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("ERROR: peft not importable from local src, skipping model benchmark.", file=sys.stderr)
        return []

    results = []

    for model_id in model_ids:
        print(f"\n--- Model: {model_id} ---")
        model = None
        peft_model = None
        try:
            if verbose:
                print(
                    "  params: "
                    f"rank={rank}  micro_bs={batch}  seqlen={seqlen}  grad_accum={grad_accum}  "
                    f"eff_bs={batch * grad_accum}  grad_ckpt={gradient_checkpointing}  "
                    f"loss_tokens={loss_tokens or 'all'}  "
                    f"repeats={repeats}  warmup={warmup}  dtype=bf16  device={device}"
                )

            # Load model via cascade
            dm = {"": 0} if device_map == "single" else "auto"
            model = _load_model_cascade(model_id, torch_dtype=torch.bfloat16, device_map=dm)

            # Warn if multi-device dispatch detected
            hf_device_map = getattr(model, "hf_device_map", None)
            if hf_device_map and len(set(hf_device_map.values())) > 1:
                print(
                    f"  WARNING: model dispatched across multiple devices: "
                    f"{set(hf_device_map.values())}. Memory reporting is single-device only.",
                    file=sys.stderr,
                )

            # Apply DoRA
            lora_config = _build_model_lora_config(model, rank=rank, adapt_vision=adapt_vision)
            try:
                peft_model = get_peft_model(model, lora_config)
            except Exception as e:
                print(f"  expanded all-linear failed ({e}), trying decoder-only fallback...", file=sys.stderr)
                # Fallback: decoder-side modules only.
                lora_config_fallback = LoraConfig(
                    r=rank,
                    lora_alpha=rank * 2,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    use_dora=True,
                    task_type="CAUSAL_LM",
                )
                peft_model = get_peft_model(model, lora_config_fallback)

            # Enable gradient checkpointing (production default)
            if gradient_checkpointing:
                peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                # Avoid "use_cache=True is incompatible with gradient checkpointing" warning
                if hasattr(peft_model.config, "use_cache"):
                    peft_model.config.use_cache = False
                text_cfg = getattr(peft_model.config, "text_config", None)
                if text_cfg is not None and hasattr(text_cfg, "use_cache"):
                    text_cfg.use_cache = False

            # Count DoRA modules and trainable params
            dora_count = sum(1 for m in peft_model.modules() if isinstance(m, DoraLinearLayer))
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            adapted_module_names = getattr(getattr(peft_model, "base_model", None), "targeted_module_names", None)
            if not adapted_module_names:
                adapted_module_names = list(getattr(lora_config, "target_modules", []))
            adapted_summary = _summarize_adapted_module_names(
                list(adapted_module_names),
                extra_vision_prefixes=_extra_vision_module_prefixes(model),
            )
            if show_adapted_modules:
                _print_adapted_module_summary(adapted_summary)

            # Dummy input
            vocab_size = _get_vocab_size_from_config(peft_model.config)
            input_ids = torch.randint(0, vocab_size, (batch, seqlen), device=device)
            model_inputs = _build_text_model_inputs(peft_model, input_ids)

            model_results = []
            eager_output = None
            baseline_fwd_ms = None
            baseline_oom = False  # tracks if reference OOM'd

            # All model configs: our 4 DoRAFactors configs first, then baselines.
            # Baselines run LAST because baseline_hf_peft may OOM during
            # gradient-checkpointed backward, corrupting the shared peft_model's
            # internal checkpointing state.  Running baselines last means any
            # OOM corruption doesn't affect the configs we actually care about.
            ALL_MODEL_CONFIGS = list(CONFIGS) + [
                ("baseline_dense_ba", 0, 0),  # direct B@A matmul ("obvious fix")
                ("baseline_hf_peft", 0, 0),  # identity matrix (upstream PEFT) -- last, may OOM
            ]

            # --- Forward-only (inference) pass ---
            print("  [Inference pass]")
            for label, fwd_flag, bwd_flag in ALL_MODEL_CONFIGS:
                try:
                    _set_fused_config(fwd_flag, bwd_flag)
                    peft_model.eval()
                    # baseline_hf_peft: monkey-patch norm to use dense B@A
                    if label == "baseline_hf_peft":
                        patch_ctx = _HfPeftNormPatch()
                    elif label == "baseline_dense_ba":
                        patch_ctx = _DenseBaNormPatch()
                    else:
                        patch_ctx = None

                    _patch_stack = ExitStack()
                    if patch_ctx:
                        _patch_stack.enter_context(patch_ctx)
                    try:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            # Warmup
                            for _ in range(max(warmup, 2)):
                                with torch.no_grad():
                                    _forward_model(peft_model, model_inputs)
                                _sync(device)

                            baseline_vram_mb = None
                            if device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(device)
                                baseline_vram_mb = torch.cuda.memory_allocated(device) / (1024**2)

                            fwd_times = []
                            for _ in range(max(1, repeats)):
                                t0 = time.perf_counter()
                                with torch.no_grad():
                                    _forward_model(peft_model, model_inputs)
                                _sync(device)
                                t1 = time.perf_counter()
                                fwd_times.append((t1 - t0) * 1000.0)

                            peak_vram_mb = None
                            reserved_vram_mb = None
                            working_set_delta_mb = None
                            if device.type == "cuda":
                                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                                reserved_vram_mb = torch.cuda.memory_reserved(device) / (1024**2)
                                if baseline_vram_mb is not None:
                                    working_set_delta_mb = peak_vram_mb - baseline_vram_mb

                            # Capture output for numerical sanity check.
                            # Full logits → CPU float32 → cosine similarity in float64.
                            # Float64 is required: near-parallel vectors (same model,
                            # different kernel paths) suffer catastrophic cancellation
                            # in float32 dot products.  This can require significant
                            # CPU RAM for large-vocab models (B×T×V×20 bytes).
                            # Timing and VRAM measurements (captured above) are
                            # unaffected — the check runs outside both windows.
                            max_abs_diff = None
                            cos_sim = None
                            eager_comparison = {}
                            try:
                                with torch.no_grad():
                                    check_out = _forward_model(peft_model, model_inputs)
                                    logits = check_out.logits.float().cpu()
                                check_out = None  # free GPU logits immediately
                                if label == "dorafactors_eager":
                                    eager_output = logits.clone()
                                eager_comparison = _collect_eager_comparison_metrics(label, eager_output, logits)
                                max_abs_diff = eager_comparison.get("max_abs_diff")
                                cos_sim = eager_comparison.get("cosine_similarity")
                                logits = None  # free [B,T,V] CPU tensor promptly
                            except (MemoryError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
                                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                                    raise
                                print(
                                    f"    [{label}] logits sanity check skipped ({type(e).__name__})", file=sys.stderr
                                )
                                check_out = None
                                logits = None
                                gc.collect()
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()

                            fwd_timing = _timing_stats_with_samples_ms(fwd_times)
                            if label == "baseline_hf_peft":
                                baseline_fwd_ms = float(fwd_timing["mean_ms"])

                            row = {
                                "model": model_id,
                                "pass": "inference",
                                "config": label,
                                "rank": rank,
                                "batch": batch,
                                "seqlen": seqlen,
                                "repeats": repeats,
                                "warmup": warmup,
                                "fwd_ms": round(fwd_timing["mean_ms"], 2),
                                "fwd_std_ms": round(fwd_timing["std_ms"], 2),
                                "bwd_ms": None,
                                "step_ms": round(fwd_timing["mean_ms"], 2),
                                "step_std_ms": round(fwd_timing["std_ms"], 2),
                                "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else None,
                                "baseline_vram_mb": round(baseline_vram_mb, 1)
                                if baseline_vram_mb is not None
                                else None,
                                "working_set_delta_mb": round(working_set_delta_mb, 1)
                                if working_set_delta_mb is not None
                                else None,
                                "reserved_vram_mb": round(reserved_vram_mb, 1) if reserved_vram_mb else None,
                                "dora_modules": dora_count,
                                "adapted_modules_total": adapted_summary["total"],
                                "adapted_modules_lm": adapted_summary["lm_count"],
                                "adapted_modules_vision": adapted_summary["vision_count"],
                                "trainable_params": trainable_params,
                                "timings": {
                                    "fwd": fwd_timing,
                                    "step": fwd_timing,
                                },
                            }
                            row.update(eager_comparison)
                            model_results.append(row)

                            if verbose:
                                ws_str = (
                                    f"  ws_delta={working_set_delta_mb:.0f}MB"
                                    if working_set_delta_mb is not None
                                    else ""
                                )
                                cossim_eager = _display_cosine_similarity_to_eager({"config": label, **row})
                                cossim_eager_str = (
                                    f"  cossim_eager={cossim_eager:.6f}" if cossim_eager is not None else ""
                                )
                                diff_str = (
                                    f"  maxdiff={max_abs_diff:.2e}  cossim={cos_sim:.6f}"
                                    if max_abs_diff is not None
                                    else ""
                                )
                                if label == "baseline_hf_peft":
                                    spd_str = ""
                                elif baseline_oom:
                                    spd_str = "  [ref: OOM]"
                                else:
                                    spd_str = _format_speedup_vs_baseline(
                                        baseline_fwd_ms, float(fwd_timing["mean_ms"])
                                    )
                                print(
                                    f"    [{label}] infer fwd={row['fwd_ms']:.2f}ms  "
                                    f"rsrvd={row.get('reserved_vram_mb', '-')}MB  alloc={row['peak_vram_mb']}MB"
                                    f"{ws_str}{spd_str}{diff_str}{cossim_eager_str}"
                                )
                    finally:
                        # Release local references that pin large VRAM allocations.
                        # On OOM this runs BEFORE the exception propagates, breaking
                        # the reference chain: loss → autograd graph → grad-ckpt saved tensors.
                        check_out = None
                        logits = None
                        _patch_stack.close()

                except torch.cuda.OutOfMemoryError:
                    print(f"    [{label}] OOM!", file=sys.stderr)
                    if label == "baseline_hf_peft":
                        baseline_oom = True
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    model_results.append(
                        {
                            "model": model_id,
                            "pass": "inference",
                            "config": label,
                            "error": "OOM",
                        }
                    )

                except Exception as e:
                    # Treat "generator didn't stop after throw()" as OOM
                    # (identity matrix materialization exhausts memory,
                    # causing cascading generator cleanup failures)
                    is_oom = "generator didn't stop" in str(e) or "out of memory" in str(e).lower()
                    if label == "baseline_hf_peft" and is_oom:
                        baseline_oom = True
                        print(f"    [{label}] OOM (ref cannot run at this scale)", file=sys.stderr)
                    else:
                        print(f"    ERROR in inference config {label}: {e}", file=sys.stderr)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    model_results.append(
                        {
                            "model": model_id,
                            "pass": "inference",
                            "config": label,
                            "error": str(e),
                        }
                    )

            # --- Grad-compute pass (fwd+bwd, no optimizer step) ---
            print(
                f"  [Grad-compute pass (fwd+bwd, no optimizer step) — grad_accum={grad_accum}, micro_bs={batch}, eff_bs={batch * grad_accum}]"
            )
            eager_output = None  # reset for training pass
            baseline_iter_ms = None
            baseline_train_oom = False
            for label, fwd_flag, bwd_flag in ALL_MODEL_CONFIGS:
                try:
                    _set_fused_config(fwd_flag, bwd_flag)
                    peft_model.train()
                    if label == "baseline_hf_peft":
                        patch_ctx = _HfPeftNormPatch()
                    elif label == "baseline_dense_ba":
                        patch_ctx = _DenseBaNormPatch()
                    else:
                        patch_ctx = None

                    _patch_stack = ExitStack()
                    if patch_ctx:
                        _patch_stack.enter_context(patch_ctx)
                    try:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            # Warmup: full accumulation cycles
                            for _ in range(max(warmup, 2)):
                                peft_model.zero_grad(set_to_none=True)
                                for _ga in range(grad_accum):
                                    _, loss = _forward_lm_with_loss(peft_model, model_inputs, loss_tokens=loss_tokens)
                                    (loss / grad_accum).backward()
                                _sync(device)

                            # Free warmup gradients before memory measurement baseline
                            peft_model.zero_grad(set_to_none=True)
                            _sync(device)

                            baseline_vram_mb = None
                            if device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(device)
                                baseline_vram_mb = torch.cuda.memory_allocated(device) / (1024**2)

                            # Measure: each "repeat" is one full accumulation cycle
                            # (grad_accum sequential micro-batch fwd+bwd passes)
                            micro_fwd_times = []
                            micro_bwd_times = []
                            iter_times = []  # full accumulation cycle time

                            for _ in range(max(1, repeats)):
                                peft_model.zero_grad(set_to_none=True)
                                t_iter_start = time.perf_counter()

                                for _ga in range(grad_accum):
                                    t0 = time.perf_counter()
                                    _, loss = _forward_lm_with_loss(peft_model, model_inputs, loss_tokens=loss_tokens)
                                    _sync(device)
                                    t1 = time.perf_counter()
                                    (loss / grad_accum).backward()
                                    _sync(device)
                                    t2 = time.perf_counter()
                                    micro_fwd_times.append((t1 - t0) * 1000.0)
                                    micro_bwd_times.append((t2 - t1) * 1000.0)

                                t_iter_end = time.perf_counter()
                                iter_times.append((t_iter_end - t_iter_start) * 1000.0)

                            peak_vram_mb = None
                            reserved_vram_mb = None
                            working_set_delta_mb = None
                            if device.type == "cuda":
                                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                                reserved_vram_mb = torch.cuda.memory_reserved(device) / (1024**2)
                                if baseline_vram_mb is not None:
                                    working_set_delta_mb = peak_vram_mb - baseline_vram_mb

                            # Capture output for numerical sanity check.
                            # See inference path comment for float64 rationale and
                            # memory implications.  Timing/VRAM already captured above.
                            max_abs_diff = None
                            cos_sim = None
                            eager_comparison = {}
                            was_training = peft_model.training
                            try:
                                peft_model.eval()
                                with torch.inference_mode():
                                    check_out = _forward_model(peft_model, model_inputs)
                                    logits = check_out.logits.float().cpu()
                                check_out = None
                                if label == "dorafactors_eager":
                                    eager_output = logits.clone()
                                eager_comparison = _collect_eager_comparison_metrics(label, eager_output, logits)
                                max_abs_diff = eager_comparison.get("max_abs_diff")
                                cos_sim = eager_comparison.get("cosine_similarity")
                                logits = None  # free [B,T,V] CPU tensor promptly
                            except (MemoryError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
                                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                                    raise
                                print(
                                    f"    [{label}] logits sanity check skipped ({type(e).__name__})", file=sys.stderr
                                )
                                check_out = None
                                logits = None
                                gc.collect()
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                            finally:
                                peft_model.train(was_training)

                            micro_fwd_timing = _timing_stats_with_samples_ms(micro_fwd_times)
                            micro_bwd_timing = _timing_stats_with_samples_ms(micro_bwd_times)
                            iter_timing = _timing_stats_with_samples_ms(iter_times)
                            micro_step_ms = micro_fwd_timing["mean_ms"] + micro_bwd_timing["mean_ms"]
                            if label == "baseline_hf_peft":
                                baseline_iter_ms = float(iter_timing["mean_ms"])

                            row = {
                                "model": model_id,
                                "pass": "grad_compute",
                                "config": label,
                                "rank": rank,
                                "batch": batch,
                                "seqlen": seqlen,
                                "grad_accum": grad_accum,
                                "eff_batch": batch * grad_accum,
                                "repeats": repeats,
                                "warmup": warmup,
                                "fwd_ms": round(micro_fwd_timing["mean_ms"], 2),
                                "fwd_std_ms": round(micro_fwd_timing["std_ms"], 2),
                                "bwd_ms": round(micro_bwd_timing["mean_ms"], 2),
                                "bwd_std_ms": round(micro_bwd_timing["std_ms"], 2),
                                "micro_step_ms": round(micro_step_ms, 2),
                                "iter_ms": round(iter_timing["mean_ms"], 2),
                                "iter_std_ms": round(iter_timing["std_ms"], 2),
                                "step_ms": round(iter_timing["mean_ms"], 2),
                                "step_std_ms": round(iter_timing["std_ms"], 2),
                                "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else None,
                                "baseline_vram_mb": round(baseline_vram_mb, 1)
                                if baseline_vram_mb is not None
                                else None,
                                "working_set_delta_mb": round(working_set_delta_mb, 1)
                                if working_set_delta_mb is not None
                                else None,
                                "reserved_vram_mb": round(reserved_vram_mb, 1) if reserved_vram_mb else None,
                                "dora_modules": dora_count,
                                "adapted_modules_total": adapted_summary["total"],
                                "adapted_modules_lm": adapted_summary["lm_count"],
                                "adapted_modules_vision": adapted_summary["vision_count"],
                                "trainable_params": trainable_params,
                                "timings": {
                                    "micro_fwd": micro_fwd_timing,
                                    "micro_bwd": micro_bwd_timing,
                                    "iter": iter_timing,
                                },
                            }
                            row.update(eager_comparison)
                            model_results.append(row)

                            if verbose:
                                ws_str = (
                                    f"  ws_delta={working_set_delta_mb:.0f}MB"
                                    if working_set_delta_mb is not None
                                    else ""
                                )
                                cossim_eager = _display_cosine_similarity_to_eager({"config": label, **row})
                                cossim_eager_str = (
                                    f"  cossim_eager={cossim_eager:.6f}" if cossim_eager is not None else ""
                                )
                                diff_str = (
                                    f"  maxdiff={max_abs_diff:.2e}  cossim={cos_sim:.6f}"
                                    if max_abs_diff is not None
                                    else ""
                                )
                                if label == "baseline_hf_peft":
                                    spd_str = ""
                                elif baseline_train_oom:
                                    spd_str = "  [ref: OOM]"
                                else:
                                    spd_str = _format_speedup_vs_baseline(
                                        baseline_iter_ms, float(iter_timing["mean_ms"])
                                    )
                                print(
                                    f"    [{label}] micro: fwd={row['fwd_ms']:.1f}ms bwd={row['bwd_ms']:.1f}ms  "
                                    f"iter({grad_accum}x)={row['iter_ms']:.0f}ms  "
                                    f"rsrvd={row.get('reserved_vram_mb', '-')}MB  alloc={row['peak_vram_mb']}MB"
                                    f"{ws_str}{spd_str}{diff_str}{cossim_eager_str}"
                                )
                    finally:
                        # Release local references that pin large VRAM allocations.
                        # On OOM during backward, `loss` holds the entire autograd graph
                        # (all grad-checkpointing saved tensors); `_` holds the model output.
                        # Nulling these BEFORE the exception propagates is critical.
                        _ = None
                        loss = None
                        check_out = None
                        logits = None
                        try:
                            peft_model.zero_grad(set_to_none=True)
                        except Exception:
                            pass
                        _patch_stack.close()

                except torch.cuda.OutOfMemoryError:
                    if label == "baseline_hf_peft":
                        baseline_train_oom = True
                        print(f"    [{label}] OOM (ref cannot train at this scale)", file=sys.stderr)
                    else:
                        print(f"    [{label}] OOM!", file=sys.stderr)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    model_results.append(
                        {
                            "model": model_id,
                            "pass": "grad_compute",
                            "config": label,
                            "error": "OOM",
                        }
                    )

                except Exception as e:
                    is_oom = "generator didn't stop" in str(e) or "out of memory" in str(e).lower()
                    if label == "baseline_hf_peft" and is_oom:
                        baseline_train_oom = True
                        print(f"    [{label}] OOM (ref cannot train at this scale)", file=sys.stderr)
                    else:
                        print(f"    ERROR in training config {label}: {e}", file=sys.stderr)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    model_results.append(
                        {
                            "model": model_id,
                            "pass": "grad_compute",
                            "config": label,
                            "error": "OOM" if (label == "baseline_hf_peft" and is_oom) else str(e),
                        }
                    )

            results.extend(model_results)

        except Exception as e:
            print(f"  ERROR loading model {model_id}: {e}", file=sys.stderr)
            results.append({"model": model_id, "error": str(e)})

        finally:
            # Aggressively release ALL references that pin GPU memory.
            # Python locals survive block exits — every tensor-holding variable
            # from the inner loops must be explicitly nulled or deleted.
            input_ids = None
            baseline_output = None
            lora_config = None
            lora_config_fallback = None
            check_out = None
            logits = None
            loss = None
            _ = None
            try:
                if peft_model is not None:
                    peft_model.zero_grad(set_to_none=True)
                del peft_model
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            peft_model = None
            model = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gc.collect()
            gc.collect()  # second pass catches ref-cycles freed by first pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# ====================================================================
# Output formatting
# ====================================================================


def _fmt_timing_cell(r: Dict, timings_key: str, flat_key: str, fmt: str = ".4f") -> str:
    """Format a timing cell as 'median ± IQR' from timings dict, or flat value fallback."""
    t = r.get("timings", {}).get(timings_key)
    if t and "median_ms" in t and "iqr_ms" in t:
        return f"{t['median_ms']:{fmt}} ± {t['iqr_ms']:{fmt}}"
    val = r.get(flat_key, 0)
    return f"{val:{fmt}}"


def _format_norm_table(norm_results: List[Dict]):
    # Print two sub-tables: timing and memory
    has_timings = any(r.get("timings") for r in norm_results)
    has_peft_eye = any(r.get("peft_eye_time_ms") is not None or r.get("peft_eye_oom") for r in norm_results)

    # --- Timing table ---
    base_headers = ["Shape", "Rank", "Dtype"]
    if has_peft_eye:
        base_headers.append("PEFTeye(ms)")
    if has_timings:
        base_headers += ["Ref (median±IQR ms)", "Factored (median±IQR ms)", "Fused (median±IQR ms)"]
    else:
        base_headers += ["Ref(ms)", "Factored(ms)", "Fused(ms)"]
    if has_peft_eye:
        base_headers += ["Eye/Ref", "Eye/Fact"]
    base_headers += ["Ref/Factored", "Fact/Fused", "MaxDiff"]
    headers_time = base_headers

    rows_time = []
    for r in norm_results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        rank_str = str(r["shape"][2])
        row_cells = [shape_str, rank_str, r["dtype"]]
        if has_peft_eye:
            if r.get("peft_eye_oom"):
                row_cells.append("OOM")
            elif r.get("peft_eye_time_ms") is not None:
                row_cells.append(_fmt_timing_cell(r, "peft_eye", "peft_eye_time_ms"))
            else:
                row_cells.append("-")
        row_cells += [
            _fmt_timing_cell(r, "ref", "ref_time_ms"),
            _fmt_timing_cell(r, "factored", "factored_time_ms"),
            _fmt_timing_cell(r, "fused", "fused_time_ms"),
        ]
        if has_peft_eye:
            eor = r.get("peft_eye_over_ref")
            eof = r.get("peft_eye_over_factored")
            row_cells.append(f"{eor:.2f}x" if eor is not None else "OOM")
            row_cells.append(f"{eof:.2f}x" if eof is not None else "OOM")
        row_cells += [
            f"{r['ref_over_factored']:.2f}x",
            f"{r['factored_over_fused']:.2f}x",
            f"{r['max_abs_diff']:.2e}",
        ]
        rows_time.append(row_cells)
    _print_table(headers_time, rows_time, "Section 1a: Norm Timing -- PEFTeye vs Reference vs Factored vs Fused")

    # --- Memory table ---
    mem_headers = ["Shape", "Rank", "Dtype"]
    if has_peft_eye:
        mem_headers.append("PEFTeye Delta")
    mem_headers += [
        "Measured: RefDelta",
        "Measured: FactDelta",
        "Measured Redux",
        "Theory(fp32): Ref(BA)",
        "Theory(fp32): Fact(U+G)",
        "Theory Redux",
    ]
    rows_mem = []
    for r in norm_results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        rank_str = str(r["shape"][2])
        mred = r.get("mem_reduction")
        mred_str = f"{mred:.1f}x" if mred is not None and mred < float("inf") else "-"
        tred = r.get("theory_reduction")
        tred_str = f"{tred:.1f}x" if tred is not None and tred < float("inf") else "-"
        row_cells = [shape_str, rank_str, r["dtype"]]
        if has_peft_eye:
            eye_mb = r.get("peft_eye_delta_mb")
            row_cells.append(f"{eye_mb:.1f} MB" if eye_mb is not None else "OOM")
        row_cells += [
            f"{r.get('ref_delta_mb', 0):.1f} MB",
            f"{r.get('factored_delta_mb', 0):.1f} MB",
            mred_str,
            f"{r.get('ref_theory_mb', 0):.1f} MB",
            f"{r.get('fact_theory_mb', 0):.1f} MB",
            tred_str,
        ]
        rows_mem.append(row_cells)
    _print_table(
        mem_headers, rows_mem, "Section 1b: Norm Memory -- Measured Delta vs Theoretical Persistent Working Set"
    )


def _format_compose_table(compose_results: List[Dict]):
    has_timings = any(r.get("timings") for r in compose_results)
    if has_timings:
        headers = [
            "Shape",
            "Dtype",
            "Eager IP (med±IQR us)",
            "Eager OOP (med±IQR us)",
            "Fused IP (med±IQR us)",
            "Fused OOP (med±IQR us)",
            "Fused AG (med±IQR us)",
            "Spd IP",
            "Spd OOP",
            "Spd AG",
            "~BW E.OOP",
            "~BW F.IP",
            "~BW F.OOP",
            "MaxDiff",
        ]
    else:
        headers = [
            "Shape",
            "Dtype",
            "Eager IP (us)",
            "Eager OOP (us)",
            "Fused IP (us)",
            "Fused OOP (us)",
            "Fused AG (us)",
            "Spd IP",
            "Spd OOP",
            "Spd AG",
            "~BW E.OOP",
            "~BW F.IP",
            "~BW F.OOP",
            "MaxDiff",
        ]
    rows = []
    for r in compose_results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        rows.append(
            [
                shape_str,
                r["dtype"],
                _fmt_timing_cell(r, "eager", "eager_us", ".1f"),
                _fmt_timing_cell(r, "eager_oop", "eager_oop_us", ".1f"),
                _fmt_timing_cell(r, "fused_fwd", "fused_fwd_us", ".1f"),
                _fmt_timing_cell(r, "fused_oop", "fused_oop_us", ".1f"),
                _fmt_timing_cell(r, "fused_autograd", "fused_autograd_us", ".1f"),
                f"{r['speedup_fwd']:.2f}x",
                f"{r['speedup_oop']:.2f}x",
                f"{r['speedup_autograd']:.2f}x",
                f"{r.get('approx_eager_oop_bw_gbps', 0):.0f}",
                f"{r['approx_fused_bw_gbps']:.0f}",
                f"{r['approx_fused_oop_bw_gbps']:.0f}",
                f"{r['compose_max_abs_diff']:.2e}",
            ]
        )
    _print_table(
        headers,
        rows,
        "Section 2: Compose -- Fused vs Eager  (Spd IP: vs eager IP | Spd OOP: vs eager OOP | Spd AG: vs eager OOP+grad)",
    )


def _format_backward_table(bwd_results: List[Dict]):
    has_timings = any(r.get("timings") for r in bwd_results)
    if has_timings:
        headers = [
            "Shape",
            "Dtype",
            "Std (med±IQR us)",
            "Fused (med±IQR us)",
            "Speedup",
            "d_lora diff",
            "d_base diff",
            "d_mag diff",
        ]
    else:
        headers = ["Shape", "Dtype", "Std (us)", "Fused (us)", "Speedup", "d_lora diff", "d_base diff", "d_mag diff"]
    rows = []
    for r in bwd_results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        gd = r["grad_max_abs_diff"]
        rows.append(
            [
                shape_str,
                r["dtype"],
                _fmt_timing_cell(r, "std", "std_total_us", ".1f"),
                _fmt_timing_cell(r, "fused", "fused_total_us", ".1f"),
                f"{r['speedup']:.2f}x",
                f"{gd['d_lora']:.2e}",
                f"{gd['d_base']:.2e}",
                f"{gd['d_mag']:.2e}",
            ]
        )
    _print_table(headers, rows, "Section 3: Backward -- Fused vs Standard Autograd")


def _format_e2e_table(e2e_results: List[Dict]):
    headers = ["Shape", "Config", "Fwd (ms)", "Bwd (ms)", "Step (ms)", "Peak MB"]
    rows = []
    for r in e2e_results:
        if "error" in r:
            s = r.get("shape", {})
            shape_str = (
                f"h={s.get('hidden', '?')},r={s.get('rank', '?')},bs={s.get('batch', '?')},seq={s.get('seqlen', '?')}"
            )
            rows.append([shape_str, r.get("config", "?"), r.get("error", "-")] + ["-"] * 3)
            continue
        s = r["shape"]
        shape_str = f"h={s['hidden']},r={s['rank']},bs={s['batch']},seq={s['seqlen']}"
        rows.append(
            [
                shape_str,
                r["config"],
                f"{r['fwd_ms']:.3f}",
                f"{r['bwd_ms']:.3f}",
                f"{r['step_ms']:.3f}",
                str(r.get("peak_mem_mb", "-")),
            ]
        )
    _print_table(headers, rows, "Section 4: E2E Single DoRA Layer")


def _format_memory_table(mem_results: List[Dict]):
    headers = ["Shape", "Config", "Static MB", "Fwd Peak MB", "Bwd Peak MB", "Delta MB"]
    rows = []
    for r in mem_results:
        s = r["shape"]
        shape_str = f"h={s['hidden']},r={s['rank']},bs={s['batch']},seq={s['seqlen']}"
        delta = str(r.get("delta_vs_baseline_mb", "-"))
        rows.append(
            [
                shape_str,
                r["config"],
                str(r.get("static_mem_mb", "-")),
                str(r.get("post_fwd_peak_mb", "-")),
                str(r.get("post_bwd_peak_mb", "-")),
                delta,
            ]
        )
    _print_table(headers, rows, "Section 5: Memory Profile")


def _format_models_table(model_results: List[Dict]):
    headers = [
        "Model",
        "Pass",
        "Config",
        "Fwd (ms)",
        "Bwd (ms)",
        "Iter (ms)",
        "Rsrvd MB",
        "Alloc MB",
        "WS Delta",
        "DoRA Mods",
        "MaxAbsDiff",
        "CosSim",
        "Cos/Eager",
    ]
    rows = []
    for r in model_results:
        if "error" in r and "config" not in r:
            rows.append([r.get("model", "?"), "-", "ERROR", r.get("error", "")] + ["-"] * 9)
            continue
        mad = r.get("max_abs_diff")
        mad_str = f"{mad:.2e}" if mad is not None else "-"
        cs = r.get("cosine_similarity")
        cs_str = f"{cs:.6f}" if cs is not None else "-"
        cse = _display_cosine_similarity_to_eager(r)
        cse_str = f"{cse:.6f}" if cse is not None else "-"
        bwd_str = f"{r['bwd_ms']:.2f}" if r.get("bwd_ms") is not None else "-"
        ws_delta = r.get("working_set_delta_mb")
        ws_str = f"{ws_delta:.0f}" if ws_delta is not None else "-"
        # For training rows with grad_accum, show iter_ms; for inference, show step_ms
        iter_ms = r.get("iter_ms", r.get("step_ms", 0))
        rows.append(
            [
                r.get("model", "?"),
                r.get("pass", "?"),
                r.get("config", "?"),
                f"{r.get('fwd_ms', 0):.2f}" if "fwd_ms" in r else r.get("error", "-"),
                bwd_str,
                f"{iter_ms:.0f}" if iter_ms else "-",
                str(r.get("reserved_vram_mb", "-")),
                str(r.get("peak_vram_mb", "-")),
                ws_str,
                str(r.get("dora_modules", "-")),
                mad_str,
                cs_str,
                cse_str,
            ]
        )
    _print_table(headers, rows, "Section 6: Real Model Evaluation")


def _format_stability_table(stability_results: List[Dict]):
    headers = [
        "Shape",
        "m",
        "Naive MaxAbs",
        "Stable MaxAbs",
        "Fused MaxAbs",
        "Quant Floor",
        "Naive/Stable",
    ]
    rows = []
    for r in stability_results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        n = r["methods"]["naive_bf16"]
        s = r["methods"]["stable_bf16"]
        f = r["methods"]["fused_bf16"]
        q = r["methods"]["quantization_floor"]
        ratio = n["max_abs_error"] / s["max_abs_error"] if s["max_abs_error"] > 0 else float("inf")
        rows.append(
            [
                shape_str,
                f"{r['m']:.4f}",
                f"{n['max_abs_error']:.4e}",
                f"{s['max_abs_error']:.4e}",
                f"{f['max_abs_error']:.4e}",
                f"{q['max_abs_error']:.4e}",
                f"{ratio:.1f}x",
            ]
        )
    _print_table(headers, rows, "Section 7: Stability -- Catastrophic Cancellation near mag=1 (bf16 vs fp64)")


# ====================================================================
# Main CLI
# ====================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Comprehensive DoRA benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--suite",
        choices=[
            "all",
            "norm",
            "compose",
            "backward",
            "e2e",
            "memory",
            "stability",
            "models",
            "precision",
            "vram",
            "micro",
        ],
        default="all",
        help="Which benchmark section to run (default: all, which runs 1-9 but not models or micro)",
    )
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--micro-iters",
        type=int,
        default=None,
        help="Iteration count for --suite micro. Defaults to --repeats when unset.",
    )
    p.add_argument(
        "--micro-warmup",
        type=int,
        default=None,
        help="Warmup count for --suite micro. Defaults to --warmup when unset.",
    )
    p.add_argument("--json-out", type=str, default=None, help="Path to write JSON results")
    p.add_argument(
        "--shapes",
        choices=["default", "extended"],
        default="default",
        help="Shape set: 'default' for regression, 'extended' adds MoE/MLA dims",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable torch autograd anomaly detection (VERY slow; distorts benchmark timings).",
    )
    p.add_argument("--models", nargs="+", default=None, help="Model IDs for --suite models")
    p.add_argument("--rank", type=int, default=384, help="LoRA rank for model benchmark (production: 384-768)")
    p.add_argument("--batch", type=int, default=1, help="Micro-batch size per device (production: 1-6)")
    p.add_argument(
        "--seqlen", type=int, default=4096, help="Sequence length for model benchmark (production: 2048-8192)"
    )
    p.add_argument(
        "--adapt-vision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include vision/ViT-side linear modules in all-linear DoRA targeting (default: off). "
        "Use --adapt-vision to include vision modules.",
    )
    p.add_argument(
        "--show-adapted-modules",
        action="store_true",
        help="Print a compact summary of the modules adapted for each model.",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (production: 4-24). "
        "Each iteration does grad_accum sequential micro-batch fwd+bwd passes. "
        "No optimizer step is included — this isolates DoRA overhead.",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing (default: on, matching production). "
        "Recomputes forward activations during backward — doubles DoRA overhead. "
        "Use --no-gradient-checkpointing to disable.",
    )
    p.add_argument(
        "--device-map",
        choices=["single", "auto"],
        default="single",
        help="Device map for model loading. 'single' (default) puts all params on device 0; "
        "'auto' uses accelerate dispatch (inference-only path, may split across devices).",
    )
    p.add_argument(
        "--loss-tokens",
        type=int,
        default=512,
        help="Compute loss over only the last N tokens (simulates GRPO/RLHF). "
        "Default 512 matches production GRPO where loss is on response tokens only. "
        "Set to 0 for full-sequence loss (standard SFT), which creates a massive "
        "[bs,seqlen,vocab] fp32 spike that masks memory differences between norm methods.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        if args.repeats <= 0 or args.warmup < 0:
            print("--repeats must be > 0 and --warmup >= 0", file=sys.stderr)
            return 2
        if args.micro_iters is not None and args.micro_iters <= 0:
            print("--micro-iters must be > 0", file=sys.stderr)
            return 2
        if args.micro_warmup is not None and args.micro_warmup < 0:
            print("--micro-warmup must be >= 0", file=sys.stderr)
            return 2

        device = _device_from_str("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _dtype_from_str(args.dtype)

        if args.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        if dtype == torch.float16 and device.type != "cuda":
            print("fp16 requires CUDA.", file=sys.stderr)
            return 2

        meta = get_meta(device)
        micro_iters = args.micro_iters if args.micro_iters is not None else args.repeats
        micro_warmup = args.micro_warmup if args.micro_warmup is not None else args.warmup
        print(f"Device: {meta.device_name}")
        print(f"PyTorch: {meta.pytorch_version}")
        print(f"Triton available: {meta.triton_available}")
        print(
            f"Suite: {args.suite}  dtype: {args.dtype}  repeats: {args.repeats}  warmup: {args.warmup}  shapes: {args.shapes}"
        )
        if args.suite == "micro":
            print(f"Microbench rounds: iters={micro_iters}  warmup={micro_warmup}")

        # Select shape sets based on --shapes flag
        if args.shapes == "extended":
            norm_shapes = NORM_SHAPES_EXTENDED
            compose_shapes = COMPOSE_SHAPES_EXTENDED
            e2e_shapes = E2E_SHAPES_EXTENDED
        else:
            norm_shapes = None  # None = use module default
            compose_shapes = None
            e2e_shapes = None

        benchmarks: Dict[str, Any] = {}
        suite = args.suite

        # Determine which sections to run
        run_norm = suite in ("all", "norm")
        run_compose = suite in ("all", "compose")
        run_backward = suite in ("all", "backward")
        run_e2e = suite in ("all", "e2e")
        run_memory = suite in ("all", "memory")
        run_stability = suite in ("all", "stability")
        run_precision = suite in ("all", "precision")
        run_vram = suite in ("all", "vram")
        run_micro = suite == "micro"
        run_models = suite == "models"

        # Section 1: Norm
        if run_norm:
            print("\n[Section 1: Norm — Reference vs Factored vs Fused]")
            norm_results = bench_norm(dtype, device, args.repeats, args.warmup, args.verbose, shapes=norm_shapes)
            benchmarks["norm"] = norm_results
            _format_norm_table(norm_results)

        # Section 2: Compose
        if run_compose:
            print("\n[Section 2: Compose]")
            compose_results = bench_compose(
                dtype, device, args.repeats, args.warmup, args.verbose, shapes=compose_shapes
            )
            benchmarks["compose"] = compose_results
            _format_compose_table(compose_results)

        # Section 3: Backward
        if run_backward:
            print("\n[Section 3: Backward]")
            bwd_results = bench_backward(dtype, device, args.repeats, args.warmup, args.verbose, shapes=compose_shapes)
            benchmarks["backward"] = bwd_results
            _format_backward_table(bwd_results)

        # Section 4: E2E
        if run_e2e:
            print("\n[Section 4: E2E]")
            e2e_results = bench_e2e(dtype, device, args.repeats, args.warmup, args.verbose, shapes=e2e_shapes)
            benchmarks["e2e"] = e2e_results
            _format_e2e_table(e2e_results)

        # Section 5: Memory
        if run_memory:
            print("\n[Section 5: Memory]")
            mem_results = bench_memory(dtype, device, args.verbose, shapes=e2e_shapes)
            benchmarks["memory"] = mem_results
            _format_memory_table(mem_results)

        # Section 7: Stability
        if run_stability:
            if args.dtype != "bf16":
                print("\n  NOTE: Stability demo always runs bf16 vs fp64 (--dtype flag ignored for this section)")
            print("\n[Section 7: Stability — Catastrophic Cancellation Demo]")
            stability_results = bench_stability(device, args.verbose)
            benchmarks["stability"] = stability_results
            _format_stability_table(stability_results)

        # Section 8: d_mag Precision Sweep
        if run_precision:
            print("\n[Section 8: d_mag Precision Sweep]")
            precision_results = bench_dmag_precision(device, args.verbose)
            benchmarks["precision"] = precision_results
            _format_precision_table(precision_results)

        # Section 9: VRAM Impact
        if run_vram:
            print("\n[Section 9: VRAM Impact — Eager vs Fused-Inner]")
            vram_results = bench_vram_impact(dtype, device, args.verbose)
            benchmarks["vram"] = vram_results
            _format_vram_table(vram_results)

        # Section 10: Triton microbenchmarks
        if run_micro:
            print("\n[Section 10: Triton Microbenchmarks]")
            micro_results = bench_triton_micro(dtype, device, micro_iters, micro_warmup, args.verbose)
            benchmarks["micro"] = micro_results
            _format_micro_table(micro_results)

        # Section 6: Models
        if run_models:
            print("\n[Section 6: Models]")
            model_ids = args.models if args.models else DEFAULT_MODELS
            model_results = bench_models(
                model_ids,
                args.rank,
                args.batch,
                args.seqlen,
                args.grad_accum,
                args.gradient_checkpointing,
                device,
                args.repeats,
                args.warmup,
                args.verbose,
                loss_tokens=args.loss_tokens,
                adapt_vision=args.adapt_vision,
                show_adapted_modules=args.show_adapted_modules,
                device_map=args.device_map,
            )
            benchmarks["models"] = model_results
            _format_models_table(model_results)

        # Build summary with geometric means (all shapes + production-filtered)
        summary: Dict[str, Any] = {}
        if "norm" in benchmarks:
            all_norm = benchmarks["norm"]
            prod_norm = [r for r in all_norm if _is_production_shape(r)]
            ref_over_fact = [r["ref_over_factored"] for r in all_norm]
            fact_over_fused = [r["factored_over_fused"] for r in all_norm]
            mem_reductions = [
                r["mem_reduction"] for r in all_norm if r.get("mem_reduction") and r["mem_reduction"] < float("inf")
            ]
            summary["norm_geomean_ref_over_factored"] = round(_geomean(ref_over_fact), 2)
            summary["norm_geomean_factored_over_fused"] = round(_geomean(fact_over_fused), 2)
            # Diagnostic: ref timing using norm-only (excludes B@A matmul) for transparency
            ref_norm_only_over_fact = [
                r["ref_norm_only_ms"] / r["factored_time_ms"]
                for r in all_norm
                if r.get("ref_norm_only_ms") and r["factored_time_ms"] > 0
            ]
            if ref_norm_only_over_fact:
                summary["norm_geomean_ref_norm_only_over_factored"] = round(_geomean(ref_norm_only_over_fact), 2)
            if mem_reductions:
                summary["norm_geomean_mem_reduction"] = round(_geomean(mem_reductions), 1)
            # PEFT-eye geomean (only over shapes that didn't OOM)
            peft_eye_over_ref = [r["peft_eye_over_ref"] for r in all_norm if r.get("peft_eye_over_ref") is not None]
            if peft_eye_over_ref:
                summary["norm_geomean_peft_eye_over_ref"] = round(_geomean(peft_eye_over_ref), 2)
            # Production-filtered norm (only if filter changes the set)
            if len(prod_norm) < len(all_norm):
                prod_mem = [
                    r["mem_reduction"]
                    for r in prod_norm
                    if r.get("mem_reduction") and r["mem_reduction"] < float("inf")
                ]
                if prod_mem:
                    summary["norm_geomean_mem_reduction_prod"] = round(_geomean(prod_mem), 1)
        if "compose" in benchmarks:
            all_compose = benchmarks["compose"]
            prod_compose = [r for r in all_compose if _is_production_shape(r)]
            speedups = [r["speedup_fwd"] for r in all_compose]
            summary["compose_geomean_fwd_speedup"] = round(_geomean(speedups), 2)
            oop_speedups = [r["speedup_oop"] for r in all_compose]
            summary["compose_geomean_oop_speedup"] = round(_geomean(oop_speedups), 2)
            ag_speedups = [r["speedup_autograd"] for r in all_compose]
            summary["compose_geomean_ag_speedup"] = round(_geomean(ag_speedups), 2)
            if len(prod_compose) < len(all_compose):
                prod_speedups = [r["speedup_fwd"] for r in prod_compose]
                summary["compose_geomean_fwd_speedup_prod"] = round(_geomean(prod_speedups), 2)
                prod_oop = [r["speedup_oop"] for r in prod_compose]
                summary["compose_geomean_oop_speedup_prod"] = round(_geomean(prod_oop), 2)
                prod_ag = [r["speedup_autograd"] for r in prod_compose]
                summary["compose_geomean_ag_speedup_prod"] = round(_geomean(prod_ag), 2)
        if "backward" in benchmarks:
            all_bwd = benchmarks["backward"]
            prod_bwd = [r for r in all_bwd if _is_production_shape(r)]
            speedups = [r["speedup"] for r in all_bwd]
            summary["backward_geomean_speedup"] = round(_geomean(speedups), 2)
            if len(prod_bwd) < len(all_bwd):
                prod_speedups = [r["speedup"] for r in prod_bwd]
                summary["backward_geomean_speedup_prod"] = round(_geomean(prod_speedups), 2)
        if "e2e" in benchmarks:
            # Group by shape — collect step_ms for each config
            shape_groups: Dict[str, Dict[str, float]] = {}
            prod_shape_groups: Dict[str, Dict[str, float]] = {}
            for r in benchmarks["e2e"]:
                if "error" in r or "step_ms" not in r:
                    continue
                key = str(r["shape"])
                if key not in shape_groups:
                    shape_groups[key] = {}
                shape_groups[key][r["config"]] = r["step_ms"]
                if _is_production_shape(r):
                    if key not in prod_shape_groups:
                        prod_shape_groups[key] = {}
                    prod_shape_groups[key][r["config"]] = r["step_ms"]

            # Primary: baseline_hf_peft step_ms / dorafactors_fully_fused step_ms
            hf_peft_over_fused = []
            for key, configs in shape_groups.items():
                peft = configs.get("baseline_hf_peft")
                fused = configs.get("dorafactors_fully_fused")
                if peft and fused and fused > 0:
                    hf_peft_over_fused.append(peft / fused)
            if hf_peft_over_fused:
                summary["e2e_speedup_hf_peft_over_fully_fused"] = round(_geomean(hf_peft_over_fused), 2)

            # Internal kernel speedup: dorafactors_eager step_ms / dorafactors_fully_fused step_ms
            eager_over_fused = []
            for key, configs in shape_groups.items():
                eager = configs.get("dorafactors_eager")
                fused = configs.get("dorafactors_fully_fused")
                if eager and fused and fused > 0:
                    eager_over_fused.append(eager / fused)
            if eager_over_fused:
                summary["e2e_speedup_eager_over_fully_fused"] = round(_geomean(eager_over_fused), 2)

            # vs "obvious fix": baseline_dense_ba step_ms / dorafactors_fully_fused step_ms
            dense_over_fused = []
            for key, configs in shape_groups.items():
                dense = configs.get("baseline_dense_ba")
                fused = configs.get("dorafactors_fully_fused")
                if dense and fused and fused > 0:
                    dense_over_fused.append(dense / fused)
            if dense_over_fused:
                summary["e2e_speedup_dense_ba_over_fully_fused"] = round(_geomean(dense_over_fused), 2)

            # Production-filtered (only if filter changes the set)
            if prod_shape_groups and len(prod_shape_groups) < len(shape_groups):
                prod_speedups = []
                for key, configs in prod_shape_groups.items():
                    eager = configs.get("dorafactors_eager")
                    fused = configs.get("dorafactors_fully_fused")
                    if eager and fused and fused > 0:
                        prod_speedups.append(eager / fused)
                if prod_speedups:
                    summary["e2e_speedup_eager_over_fully_fused_prod"] = round(_geomean(prod_speedups), 2)

        if "models" in benchmarks:
            # Group per-model training step_ms by config
            model_groups: Dict[str, Dict[str, float]] = {}
            for r in benchmarks["models"]:
                if r.get("pass") == "grad_compute" and "step_ms" in r and "error" not in r:
                    mid = r["model"]
                    if mid not in model_groups:
                        model_groups[mid] = {}
                    model_groups[mid][r["config"]] = r["step_ms"]

            # Primary: baseline_hf_peft iter_ms / dorafactors_fully_fused iter_ms
            hf_peft_over_fused = []
            for mid, configs in model_groups.items():
                peft = configs.get("baseline_hf_peft")
                fused = configs.get("dorafactors_fully_fused")
                if peft and fused and fused > 0:
                    hf_peft_over_fused.append(peft / fused)
            if hf_peft_over_fused:
                summary["models_speedup_hf_peft_over_fully_fused"] = round(_geomean(hf_peft_over_fused), 2)

            # Internal kernel speedup: dorafactors_eager iter_ms / dorafactors_fully_fused iter_ms
            model_speedups = []
            for mid, configs in model_groups.items():
                eager = configs.get("dorafactors_eager")
                fused = configs.get("dorafactors_fully_fused")
                if eager and fused and fused > 0:
                    model_speedups.append(eager / fused)
            if model_speedups:
                summary["models_speedup_eager_over_fully_fused"] = round(_geomean(model_speedups), 2)

            # vs "obvious fix": baseline_dense_ba iter_ms / dorafactors_fully_fused iter_ms
            dense_over_fused = []
            for mid, configs in model_groups.items():
                dense = configs.get("baseline_dense_ba")
                fused = configs.get("dorafactors_fully_fused")
                if dense and fused and fused > 0:
                    dense_over_fused.append(dense / fused)
            if dense_over_fused:
                summary["models_speedup_dense_ba_over_fully_fused"] = round(_geomean(dense_over_fused), 2)
        if "stability" in benchmarks:
            # Extract m=1.0 results for summary (the worst case for cancellation)
            m1_results = [r for r in benchmarks["stability"] if r["m"] == 1.0]
            if m1_results:
                # Use the largest shape for the headline number
                r = m1_results[-1]
                naive_err = r["methods"]["naive_bf16"]["max_abs_error"]
                stable_err = r["methods"]["stable_bf16"]["max_abs_error"]
                summary["stability_m1_naive_max_err"] = naive_err
                summary["stability_m1_stable_max_err"] = stable_err
                if stable_err > 0:
                    summary["stability_m1_naive_over_stable"] = round(naive_err / stable_err, 1)
        if "micro" in benchmarks and benchmarks["micro"]:
            boundary = benchmarks["micro"].get("backward_boundary_extension", [])
            boundary_32768_wins = 0
            row_bucket = benchmarks["micro"].get("row_bucket_sensitivity", [])
            row_bucket_flips = 0
            last_best_by_probe: Dict[Tuple[str, str], str] = {}
            for r in boundary:
                if r["best"]["config"]["BLOCK_SIZE"] == 32768:
                    boundary_32768_wins += 1
            for r in row_bucket:
                probe_key = (r["probe"], f"{r['kernel']}:{r['shape'][1]}")
                best = r["best"]["config_label"]
                if probe_key in last_best_by_probe and last_best_by_probe[probe_key] != best:
                    row_bucket_flips += 1
                last_best_by_probe[probe_key] = best
            summary["micro_backward_32768_wins"] = boundary_32768_wins
            summary["micro_row_bucket_best_config_flips"] = row_bucket_flips

        # Collect dtypes actually tested across all sections
        dtypes_tested = set()
        for section_results in benchmarks.values():
            if isinstance(section_results, list):
                for r in section_results:
                    dt_val = r.get("dtype")
                    if dt_val:
                        dtypes_tested.add(dt_val)
            elif isinstance(section_results, dict):
                for sub_results in section_results.values():
                    if isinstance(sub_results, list):
                        for r in sub_results:
                            dt_val = r.get("dtype")
                            if dt_val:
                                dtypes_tested.add(dt_val)
        dtypes_tested = sorted(dtypes_tested) if dtypes_tested else [args.dtype]
        summary["dtypes_tested"] = dtypes_tested

        # JSON output
        payload = {
            "meta": asdict(meta),
            "args": vars(args),
            "benchmarks": benchmarks,
            "summary": summary,
        }

        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nWrote JSON results to {args.json_out}")

        # Print summary
        # Extract short GPU name: "NVIDIA H100 80GB HBM3" → "h100"
        _parts = (meta.device_name or "unknown").split()
        gpu_short = (_parts[1] if len(_parts) > 1 and _parts[0].upper() == "NVIDIA" else _parts[0]).lower()
        has_prod = any(k.endswith("_prod") for k in summary)
        if summary:
            dtype_label = "+".join(dtypes_tested) if len(dtypes_tested) > 1 else dtypes_tested[0]
            print("\n" + "=" * 60)
            print(f"  SUMMARY ({dtype_label}, {gpu_short}, geometric means)")
            print("=" * 60)
            for k, v in summary.items():
                suffix = "  [d_in >= 2048]" if k.endswith("_prod") else ""
                print(f"  {k}: {v}{suffix}")
            if has_prod:
                print(f"\n  production filter: d_in >= {MIN_DIN_PRODUCTION}")
            print()

        print("Benchmark complete.")
        return 0

    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except SystemExit as e:
        return int(e.code)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
