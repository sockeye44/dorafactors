#!/usr/bin/env python3
"""Generate publication-quality figures for the DoRA paper from bench_it6 JSON data.

Data sources (bench_it6 exclusively):
  Microbenchmarks (6 GPUs, 200 repeats, extended shapes):
    - sm89_l40s_comprehensive_extended_{bf16,fp16,fp32}.json   — NVIDIA L40S 48GB GDDR6 (SM89, Ada)
    - sm80_a100_comprehensive_extended_{bf16,fp16,fp32}.json   — NVIDIA A100-SXM4 80GB HBM2e (SM80, Ampere)
    - sm120_rtx6000_comprehensive_extended_{bf16,fp16,fp32}.json — NVIDIA RTX PRO 6000 96GB GDDR7 (SM120, Blackwell)
    - sm90_h200_comprehensive_extended_{bf16,fp16,fp32}.json   — NVIDIA H200 141GB HBM3e (SM90, Hopper)
    - sm100_b200_comprehensive_extended_{bf16,fp16,fp32}.json  — NVIDIA B200 192GB HBM3e (SM100, Blackwell)
    - sm103_b300_comprehensive_extended_{bf16,fp16,fp32}.json  — NVIDIA B300 268GB HBM3e (SM103, Blackwell)

  Model-level (3 GPUs, bs=1, seq=4096, ga=8, r=384, loss_tokens=1024, 20 repeats):
    - rtx6000_seq4096_bs1_gas8_seq4k_loss1k_n20w2_*.json
    - h200_seq4096_bs1_gas8_seq4k_loss1k_n20w2_*.json
    - b200_seq4096_bs1_gas8_seq4k_loss1k_n20w2_*.json
"""

import json
import sys
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.grid': False,
    'axes.axisbelow': True,
    'axes.unicode_minus': False,
    'legend.framealpha': 0.95,
    'legend.fancybox': False,
    'grid.alpha': 0.24,
    'grid.linewidth': 0.45,
    'grid.color': '#b8b8b8',
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODEDIR = Path(__file__).parent.parent / 'code'
IT6DIR = CODEDIR / 'bench_it6'
FIGDIR = Path(__file__).parent / 'figures'
FIGDIR.mkdir(exist_ok=True)

OUTPUT_FORMATS = ['pdf', 'png'] if '--png' in sys.argv else ['pdf']

def save_figure(fig, name):
    for fmt in OUTPUT_FORMATS:
        kwargs = {'dpi': 300} if fmt == 'png' else {}
        fig.savefig(FIGDIR / f'{name}.{fmt}', **kwargs)
    exts = '/'.join(OUTPUT_FORMATS)
    print(f'  {name}.{exts}')


def finalize_figure(fig, name, *, rect=None, h_pad=0.8, w_pad=0.8):
    """Apply final layout adjustments, then save and close."""
    fig.align_ylabels()
    fig.tight_layout(rect=rect, h_pad=h_pad, w_pad=w_pad)
    save_figure(fig, name)
    plt.close(fig)


def style_axis(ax, *, grid='y'):
    """Apply a lighter, less cluttered axis style for publication figures."""
    if grid and grid == 'y':
        ax.grid(axis=grid, alpha=0.33, linewidth=0.45, color='#b8b8b8')
        ax.grid(axis='x', alpha=0.10, linewidth=0.45, color='#b8b8b8') # leave it, but much lighter
    elif grid and grid == 'x':
        ax.grid(axis=grid, alpha=0.33, linewidth=0.45, color='#b8b8b8')
        ax.grid(axis='y', alpha=0.10, linewidth=0.45, color='#b8b8b8') # leave it, but much lighter
    elif grid and grid == 'both':
        ax.grid(axis=grid, alpha=0.33, linewidth=0.45, color='#b8b8b8')
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_alpha(0.85)


def _gpu_marker(gpu: str) -> str:
    return next((ch for ch in GPU_MARKERS[gpu] if ch.isalpha() or ch in '<>^vsoDpxX*+'), 'o')


def _gpu_linestyle(gpu: str) -> str:
    style = GPU_MARKERS[gpu]
    if '--' in style:
        return '--'
    if ':' in style:
        return ':'
    return '-'


def gpu_legend_handles(gpus):
    handles = []
    for gpu in gpus:
        info = GPU_REGISTRY[gpu]
        handles.append(Line2D(
            [0], [0],
            color=info['color'],
            linestyle=_gpu_linestyle(gpu),
            marker=_gpu_marker(gpu),
            markersize=4.6,
            linewidth=1.4,
            label=info['label'],
        ))
    return handles


def gpu_color_legend_handles(gpus):
    handles = []
    for gpu in gpus:
        info = GPU_REGISTRY[gpu]
        handles.append(Line2D(
            [0], [0],
            color=info['color'],
            linestyle='-',
            marker=None,
            linewidth=2.0,
            label=info['label'],
        ))
    return handles


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_REF = '#d62728'       # red — reference / eager
C_FACT = '#ff7f0e'      # orange — factored
C_FUSED = '#2ca02c'     # green — fused
C_EAGER = '#d62728'     # red — eager (alias)
C_FUSED_FWD = '#2ca02c' # green — fused fwd
C_FUSED_AG = '#1f77b4'  # blue — fused autograd
C_DENSE_BA = '#8c564b'  # brown — dense_ba baseline
C_PEFT = '#d62728'      # red — HF PEFT baseline (unused; PEFT bars use GPU colors)

# ---------------------------------------------------------------------------
# GPU registry — 6 GPUs spanning 4 architecture generations
# ---------------------------------------------------------------------------
GPU_REGISTRY = {
    'l40s': {
        'label': 'L40S',
        'color': "#84bd22",
        'prefix': 'sm89_l40s',
        'peak_bw_gbps': 864,
        'vram_gb': 48,
        'arch': 'Ada Lovelace',
    },
    'a100': {
        'label': 'A100',
        'color': '#7f7f7f',
        'prefix': 'sm80_a100',
        'peak_bw_gbps': 2039,
        'vram_gb': 80,
        'arch': 'Ampere',
    },
    'rtx6000': {
        'label': 'RTX PRO 6000',
        'color': "#5246be",
        'prefix': 'sm120_rtx6000',
        'peak_bw_gbps': 1600,
        'vram_gb': 96,
        'arch': 'Blackwell',
    },
    'h200': {
        'label': 'H200',
        'color': "#26cadd",
        'prefix': 'sm90_h200',
        'peak_bw_gbps': 4800,
        'vram_gb': 141,
        'arch': 'Hopper',
    },
    'b200': {
        'label': 'B200',
        'color': "#ff7700",
        'prefix': 'sm100_b200',
        'peak_bw_gbps': 7700,
        'vram_gb': 192,
        'arch': 'Blackwell',
    },
    'b300': {
        'label': 'B300',
        'color': "#c22498",
        'prefix': 'sm103_b300',
        'peak_bw_gbps': 7700,
        'vram_gb': 268,
        'arch': 'Blackwell',
    },
}

MICRO_GPUS = ['l40s', 'a100', 'rtx6000', 'h200', 'b200', 'b300']  # ascending BW
MODEL_GPUS = ['rtx6000', 'h200', 'b200']  # only GPUs with model data

GPU_MARKERS = {
    'l40s': 'p-',
    'a100': 'v-',
    'rtx6000': '^:',
    'h200': 'D-',
    'b200': 's--',
    'b300': 'X-',
}

# Short model names for figures
MODEL_SHORT = {
    'Qwen/Qwen2.5-VL-32B-Instruct': 'Qwen2.5-VL-32B',
    'Qwen/Qwen3-VL-32B-Instruct': 'Qwen3-VL-32B',
    'Qwen/Qwen3.5-27B': 'Qwen3.5-27B',
    'google/gemma-3-27b-it': 'Gemma3-27B',
    'unsloth/Mistral-Small-3.2-24B-Instruct-2506': 'Mistral-Sm-24B',
    'Qwen/Qwen3-VL-8B-Instruct': 'Qwen3-VL-8B',
}
MODEL_ORDER = [
    'Qwen/Qwen2.5-VL-32B-Instruct',
    'Qwen/Qwen3-VL-32B-Instruct',
    'Qwen/Qwen3.5-27B',
    'google/gemma-3-27b-it',
    'unsloth/Mistral-Small-3.2-24B-Instruct-2506',
    'Qwen/Qwen3-VL-8B-Instruct',
]

# ---------------------------------------------------------------------------
# Pinned file manifest — every figure traces to these exact files
# ---------------------------------------------------------------------------
_COMPREHENSIVE_MANIFEST = {
    ('l40s',    'bf16'): 'sm89_l40s_comprehensive_extended_bf16_20260316_233031.json',
    ('l40s',    'fp16'): 'sm89_l40s_comprehensive_extended_fp16_20260316_233941.json',
    ('l40s',    'fp32'): 'sm89_l40s_comprehensive_extended_fp32_20260316_231118.json',
    ('a100',    'bf16'): 'sm80_a100_comprehensive_extended_bf16_20260316_232601.json',
    ('a100',    'fp16'): 'sm80_a100_comprehensive_extended_fp16_20260316_233329.json',
    ('a100',    'fp32'): 'sm80_a100_comprehensive_extended_fp32_20260316_230055.json',
    ('rtx6000', 'bf16'): 'sm120_rtx6000_comprehensive_extended_bf16_20260317_000100.json',
    ('rtx6000', 'fp16'): 'sm120_rtx6000_comprehensive_extended_fp16_20260317_000539.json',
    ('rtx6000', 'fp32'): 'sm120_rtx6000_comprehensive_extended_fp32_20260316_235029.json',
    ('h200',    'bf16'): 'sm90_h200_comprehensive_extended_bf16_20260316_223024.json',
    ('h200',    'fp16'): 'sm90_h200_comprehensive_extended_fp16_20260316_223415.json',
    ('h200',    'fp32'): 'sm90_h200_comprehensive_extended_fp32_20260316_222005.json',
    ('b200',    'bf16'): 'sm100_b200_comprehensive_extended_bf16_20260317_030339.json',
    ('b200',    'fp16'): 'sm100_b200_comprehensive_extended_fp16_20260317_030642.json',
    ('b200',    'fp32'): 'sm100_b200_comprehensive_extended_fp32_20260317_025509.json',
    ('b300',    'bf16'): 'sm103_b300_comprehensive_extended_bf16_20260316_234337.json',
    ('b300',    'fp16'): 'sm103_b300_comprehensive_extended_fp16_20260316_234643.json',
    ('b300',    'fp32'): 'sm103_b300_comprehensive_extended_fp32_20260316_233507.json',
}

_MODEL_MANIFEST = {
    'rtx6000': 'rtx6000_seq4096_bs1_gas8_seq4k_loss1k_n20w2_20260315_010523.json',
    'h200':    'h200_seq4096_bs1_gas8_seq4k_loss1k_n20w2_20260314_234842.json',
    'b200':    'b200_seq4096_bs1_gas8_seq4k_loss1k_n20w2_20260314_232248.json',
}


def _sha256(path):
    """Compute SHA-256 hex digest of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest():
    """Validate all manifest files exist and print provenance with SHA-256."""
    print('=== bench_it6 Data Provenance ===')
    errors = []

    print('\nComprehensive microbenchmark files (18):')
    for (gpu, dtype), fname in sorted(_COMPREHENSIVE_MANIFEST.items()):
        path = IT6DIR / fname
        if path.exists():
            size = path.stat().st_size
            digest = _sha256(path)
            print(f'  [{gpu:7s} {dtype:4s}] {fname} ({size:,} bytes) sha256:{digest[:16]}')
        else:
            errors.append(f'MISSING: {path}')
            print(f'  [{gpu:7s} {dtype:4s}] {fname} — MISSING!')

    print('\nModel-level benchmark files (3):')
    for gpu, fname in sorted(_MODEL_MANIFEST.items()):
        path = IT6DIR / fname
        if path.exists():
            size = path.stat().st_size
            digest = _sha256(path)
            print(f'  [{gpu:7s}] {fname} ({size:,} bytes) sha256:{digest[:16]}')
        else:
            errors.append(f'MISSING: {path}')
            print(f'  [{gpu:7s}] {fname} — MISSING!')

    if errors:
        print(f'\nERROR: {len(errors)} manifest file(s) missing!')
        for e in errors:
            print(f'  {e}')
        sys.exit(1)
    print(f'\nAll 21 manifest files validated.\n')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_cache = {}


_EXPECTED_COMPREHENSIVE_SECTIONS = [
    'norm', 'compose', 'backward', 'e2e', 'memory', 'stability', 'precision', 'vram',
]


def load_comprehensive(gpu_key: str, dtype: str) -> dict:
    """Load and cache a comprehensive JSON benchmark file from pinned manifest."""
    key = (gpu_key, dtype)
    if key not in _cache:
        fname = _COMPREHENSIVE_MANIFEST[key]
        path = IT6DIR / fname
        with open(path) as f:
            d = json.load(f)
        # Validate args
        args = d['args']
        assert args['repeats'] == 200, f'{fname}: repeats={args["repeats"]}, expected 200'
        assert args['shapes'] == 'extended', f'{fname}: shapes={args["shapes"]}, expected extended'
        assert args['grad_accum'] == 8, f'{fname}: grad_accum={args["grad_accum"]}, expected 8'
        # Validate all expected benchmark sections exist
        bm = d['benchmarks']
        for section in _EXPECTED_COMPREHENSIVE_SECTIONS:
            assert section in bm, f'{fname}: missing benchmarks.{section}'
            assert len(bm[section]) > 0, f'{fname}: benchmarks.{section} is empty'
        # Validate summary section exists
        assert 'summary' in d, f'{fname}: missing summary section'
        _cache[key] = d
    return _cache[key]


def load_models(gpu_key: str) -> list[dict] | None:
    """Load model-level benchmark data from pinned manifest. Returns None if GPU not in manifest."""
    cache_key = ('models', gpu_key)
    if cache_key not in _cache:
        if gpu_key not in _MODEL_MANIFEST:
            _cache[cache_key] = None
            return None
        fname = _MODEL_MANIFEST[gpu_key]
        path = IT6DIR / fname
        with open(path) as f:
            d = json.load(f)
        # Validate args
        args = d['args']
        assert args['grad_accum'] == 8, f'{fname}: grad_accum={args["grad_accum"]}, expected 8'
        assert args['seqlen'] == 4096, f'{fname}: seqlen={args["seqlen"]}, expected 4096'
        assert args['loss_tokens'] == 1024, f'{fname}: loss_tokens={args["loss_tokens"]}, expected 1024'
        assert args['adapt_vision'] == False, f'{fname}: adapt_vision={args["adapt_vision"]}, expected False'
        # Validate pass names (should be 'grad_compute' and 'inference', never 'training')
        passes = set(e['pass'] for e in d['benchmarks']['models'])
        assert 'training' not in passes, f'{fname}: found stale "training" pass, expected "grad_compute"'
        assert 'grad_compute' in passes, f'{fname}: missing "grad_compute" pass'
        # Validate expected configs exist
        configs = set(e['config'] for e in d['benchmarks']['models'])
        for expected in ['dorafactors_eager', 'dorafactors_fully_fused', 'baseline_hf_peft', 'baseline_dense_ba']:
            assert expected in configs, f'{fname}: missing config "{expected}"'
        # Validate key fields on first grad_compute fused entry
        sample = next(
            (e for e in d['benchmarks']['models']
             if e['pass'] == 'grad_compute' and e['config'] == 'dorafactors_fully_fused' and not e.get('error')),
            None
        )
        assert sample is not None, f'{fname}: no valid dorafactors_fully_fused grad_compute entry'
        for field in ['iter_ms', 'peak_vram_mb', 'dora_modules']:
            assert field in sample, f'{fname}: grad_compute entry missing field "{field}"'
        _cache[cache_key] = d['benchmarks']['models']
    return _cache[cache_key]


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def filter_norm(data: dict, dtype: str = 'float32') -> list[dict]:
    return [e for e in data['benchmarks']['norm'] if e['dtype'] == dtype]


def filter_compose(data: dict, dtype: str = 'bfloat16') -> list[dict]:
    return [e for e in data['benchmarks']['compose'] if e['dtype'] == dtype]


def filter_backward(data: dict, dtype: str = 'bfloat16') -> list[dict]:
    return [e for e in data['benchmarks']['backward'] if e['dtype'] == dtype]


def filter_e2e(data: dict, *, hidden: int, config: str,
               rank: int | None = None, batch: int | None = None,
               seqlen: int | None = None) -> list[dict]:
    seen = set()
    out = []
    for e in data['benchmarks']['e2e']:
        s = e['shape']
        if s['hidden'] != hidden or e['config'] != config:
            continue
        if rank is not None and s['rank'] != rank:
            continue
        if batch is not None and s['batch'] != batch:
            continue
        if seqlen is not None and s['seqlen'] != seqlen:
            continue
        key = (s['hidden'], s['rank'], s['batch'], s['seqlen'])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def filter_memory(data: dict, *, hidden: int, config: str,
                  rank: int | None = None, batch: int | None = None,
                  seqlen: int | None = None) -> list[dict]:
    seen = set()
    out = []
    for e in data['benchmarks']['memory']:
        s = e['shape']
        if s['hidden'] != hidden or e['config'] != config:
            continue
        if rank is not None and s['rank'] != rank:
            continue
        if batch is not None and s['batch'] != batch:
            continue
        if seqlen is not None and s['seqlen'] != seqlen:
            continue
        key = (s['hidden'], s['rank'], s['batch'], s['seqlen'])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def shape_label(shape: list | tuple) -> str:
    """Compact label like '4kx8k' for 2-element shapes, '4k, r384' for 3-element."""
    def _fmt(n):
        if n >= 1024 and n % 1024 == 0:
            return f'{n // 1024}k'
        elif n >= 1000 and n % 1000 == 0:
            return f'{n // 1000}k'
        elif n >= 10000 and n % 1000 > 0:
            return f'{n // 1000}k'
        return str(n)
    if len(shape) == 2:
        if shape[0] == shape[1]:
            return f'{_fmt(shape[0])}'
        return f'{_fmt(shape[0])}×{_fmt(shape[1])}'
    elif len(shape) == 3:
        if shape[0] == shape[1]:
            return f'{_fmt(shape[0])}, r{shape[2]}'
        return f'{_fmt(shape[0])}×{_fmt(shape[1])}, r{shape[2]}'
    return 'x'.join(_fmt(s) for s in shape)


def geomean(vals):
    """Geometric mean of positive values."""
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# =========================================================================
# Figure 1: Norm Memory Reduction (H200, fp32)
# =========================================================================
def fig_norm_memory():
    """Norm memory: theoretical persistent working set vs measured allocation delta."""
    data = load_comprehensive('h200', 'fp32')
    entries = filter_norm(data, dtype='float32')

    labels = []
    ref_theory, fact_theory = [], []
    ref_meas, fact_meas = [], []
    for e in entries:
        labels.append(shape_label(e['shape']))
        ref_theory.append(e['ref_theory_mb'])
        fact_theory.append(e['fact_theory_mb'])
        ref_meas.append(e['ref_delta_mb'])
        fact_meas.append(e['factored_delta_mb'])

    theory_redux = [r / f for r, f in zip(ref_theory, fact_theory)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    x = np.arange(len(entries))
    w = 0.35
    ax1.bar(x - w/2, ref_theory, w, label='Reference (dense BA)', color=C_REF, alpha=0.8)
    ax1.bar(x + w/2, fact_theory, w, label='Factored (U + G)', color=C_FUSED, alpha=0.8)
    ax1.set_ylabel('Working Set (MB)')
    ax1.set_title('(a) Theoretical Persistent Working Set', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.2, rotation=45, ha='right')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=8)
    for i, e in enumerate(entries):
        s = tuple(e['shape'])
        if len(s) == 3 and s[2] >= 384 and s[0] >= 4096:
            y_pos = max(ref_theory[i], fact_theory[i])
            ax1.annotate(f'{theory_redux[i]:.0f}x', xy=(x[i], y_pos),
                        xytext=(0, 6), textcoords='offset points',
                        fontsize=7, ha='center', color='black', fontweight='bold')

    ax2.bar(x - w/2, ref_meas, w, label='Reference (measured)', color=C_REF, alpha=0.8)
    ax2.bar(x + w/2, fact_meas, w, label='Factored (measured)', color=C_FUSED, alpha=0.8)
    ax2.set_ylabel('Allocation Delta (MB)')
    ax2.set_title('(b) Measured Allocation Delta', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.2, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=8)
    meas_redux = [r / f if f > 0 else 0 for r, f in zip(ref_meas, fact_meas)]
    for i, e in enumerate(entries):
        s = tuple(e['shape'])
        if len(s) == 3 and s[2] >= 384 and s[0] >= 4096 and meas_redux[i] > 1:
            y_pos = max(ref_meas[i], fact_meas[i])
            ax2.annotate(f'{meas_redux[i]:.1f}x', xy=(x[i], y_pos),
                        xytext=(0, 6), textcoords='offset points',
                        fontsize=7, ha='center', color='black', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'norm_memory')
    plt.close()


# =========================================================================
# Figure 2: Compose Speedup (6 GPUs, bf16)
# =========================================================================
def fig_compose_speedup():
    """Compose kernel speedup: forward and autograd, 6 GPUs, bf16.
    Shows full shape range, sorted by d_in then total elements."""
    gpu_data = {}
    for gpu in MICRO_GPUS:
        d = load_comprehensive(gpu, 'bf16')
        gpu_data[gpu] = filter_compose(d, dtype='bfloat16')

    # Sort entries by d_in (primary), then total elements (secondary)
    ref_gpu = 'h200'
    sort_order = sorted(range(len(gpu_data[ref_gpu])),
                        key=lambda i: (gpu_data[ref_gpu][i]['shape'][1],
                                       gpu_data[ref_gpu][i]['shape'][0] * gpu_data[ref_gpu][i]['shape'][1]))
    for gpu in MICRO_GPUS:
        gpu_data[gpu] = [gpu_data[gpu][i] for i in sort_order]

    shapes = [shape_label(e['shape']) for e in gpu_data[ref_gpu]]
    n = len(shapes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)

    for gpu in MICRO_GPUS:
        info = GPU_REGISTRY[gpu]
        entries = gpu_data[gpu]
        fwd = [e['speedup_fwd'] for e in entries]
        ag = [e['speedup_autograd'] for e in entries]
        ax1.plot(range(n), fwd, GPU_MARKERS[gpu], color=info['color'],
                label=info['label'], markersize=3.5, linewidth=1.3)
        ax2.plot(range(n), ag, GPU_MARKERS[gpu], color=info['color'],
                label=info['label'], markersize=3.5, linewidth=1.3)

    for ax, title in [(ax1, '(a) Forward (Inference)'), (ax2, '(b) Autograd (Training)')]:
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Speedup vs. Eager')
        ax.set_title(title)
        ax.set_xticks(range(n))
        ax.set_xticklabels(shapes, fontsize=6.8, rotation=48, ha='right')
        ax.set_xlabel('Activation Shape')
        style_axis(ax, grid='y')

    all_fwd = [e['speedup_fwd'] for gpu in MICRO_GPUS for e in gpu_data[gpu]]
    all_ag = [e['speedup_autograd'] for gpu in MICRO_GPUS for e in gpu_data[gpu]]
    ax1.set_ylim(min(all_fwd) - 0.15, max(all_fwd) + 0.3)
    ax2.set_ylim(min(all_ag) - 0.15, max(all_ag) + 0.3)

    fig.suptitle('Compose Kernel Speedup (bf16)', fontsize=10, y=1.02)
    fig.legend(handles=gpu_legend_handles(MICRO_GPUS), loc='upper center',
               bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=7.5,
               columnspacing=1.2, handlelength=1.9)
    finalize_figure(fig, 'compose_speedup', rect=[0, 0, 1, 0.93], h_pad=0.9, w_pad=0.8)


# =========================================================================
# Figure 3: Backward Speedup (6 GPUs, bf16)
# =========================================================================
def fig_backward_speedup():
    """Fused backward kernel speedup, 6 GPUs, bf16.
    Shapes sorted by d_in then total elements."""
    gpu_data = {}
    for gpu in MICRO_GPUS:
        d = load_comprehensive(gpu, 'bf16')
        gpu_data[gpu] = filter_backward(d, dtype='bfloat16')

    ref_gpu = 'h200'
    all_shapes = sorted(
        {tuple(e['shape']) for e in gpu_data[ref_gpu]},
        key=lambda s: (s[1], s[0] * s[1])
    )

    fig, ax = plt.subplots(figsize=(8.1, 3.6))

    gpu_by_shape = {}
    for gpu in MICRO_GPUS:
        lookup = {}
        for e in gpu_data[gpu]:
            s = tuple(e['shape'])
            if s not in lookup:
                lookup[s] = e['speedup']
        gpu_by_shape[gpu] = lookup

    shape_labels = [shape_label(list(s)) for s in all_shapes]
    n = len(all_shapes)

    for gpu in MICRO_GPUS:
        info = GPU_REGISTRY[gpu]
        spd = [gpu_by_shape[gpu].get(s, None) for s in all_shapes]
        xs = [i for i, v in enumerate(spd) if v is not None]
        ys = [v for v in spd if v is not None]
        ax.plot(xs, ys, GPU_MARKERS[gpu], color=info['color'],
               label=info['label'], markersize=4, linewidth=1.3)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhspan(0, 1.0, alpha=0.03, color=C_REF)

    ax.set_ylabel('Speedup vs. Std. Autograd')
    ax.set_xlabel('Activation Shape')
    ax.set_xticks(range(n))
    ax.set_xticklabels(shape_labels, fontsize=6.8, rotation=45, ha='right')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=7.5, ncol=2)
    ax.set_title('Fused Backward Kernel Speedup (bf16)', fontsize=10)
    style_axis(ax, grid='y')

    all_spd = [v for gpu in MICRO_GPUS for v in gpu_by_shape[gpu].values()]
    ax.set_ylim(min(all_spd) - 0.1, max(all_spd) + 0.15)

    finalize_figure(fig, 'backward_speedup')


# =========================================================================
# Figure 4 (appendix): E2E Step Time (B200, bf16, h=4096 rank sweep)
# =========================================================================
def fig_e2e_step_time():
    """E2E single DoRA layer step time across ranks (bf16, B200).
    APPENDIX figure: single-layer E2E does not predict model-level speedup."""
    data = load_comprehensive('b200', 'bf16')
    ranks = [16, 64, 128, 256, 384, 512]

    eager_entries = filter_e2e(data, hidden=4096, config='dorafactors_eager',
                               batch=4, seqlen=2048)
    fused_entries = filter_e2e(data, hidden=4096, config='dorafactors_fully_fused',
                               batch=4, seqlen=2048)

    eager_by_rank = {e['shape']['rank']: e['step_ms'] for e in eager_entries}
    fused_by_rank = {e['shape']['rank']: e['step_ms'] for e in fused_entries}

    eager_step = [eager_by_rank[r] for r in ranks]
    fused_step = [fused_by_rank[r] for r in ranks]
    speedup = [e / f for e, f in zip(eager_step, fused_step)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.3, 3.3))

    ax1.plot(ranks, eager_step, 'o-', color=C_EAGER, label='Eager', markersize=5, linewidth=1.5)
    ax1.plot(ranks, fused_step, 's-', color=C_FUSED, label='Fully Fused', markersize=5, linewidth=1.5)
    ax1.set_xlabel('LoRA Rank')
    ax1.set_ylabel('Step Time (ms)')
    ax1.set_title('(a) Step Time vs. Rank')
    ax1.legend(framealpha=0.9)
    ax1.set_xticks(ranks)
    style_axis(ax1, grid='y')

    bottom_val = min(speedup) - 0.05
    ax2.bar(range(len(ranks)), [s - bottom_val for s in speedup], bottom=bottom_val, color=C_FUSED, alpha=0.8)
    ax2.set_xlabel('LoRA Rank')
    ax2.set_ylabel('Speedup (Eager / Fused)')
    ax2.set_title('(b) Speedup vs. Rank')
    ax2.set_xticks(range(len(ranks)))
    ax2.set_xticklabels([str(r) for r in ranks])
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8)
    for i, v in enumerate(speedup):
        ax2.text(i, v + 0.005, f'{v:.2f}x', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='black')
    ax2.set_ylim(bottom_val, max(speedup) + 0.1)
    style_axis(ax2, grid='y')

    fig.suptitle('Single-Layer E2E: B200, bf16, h=4096, bs=4, seq=2048', fontsize=10, y=1.01)
    finalize_figure(fig, 'e2e_step_time')


# =========================================================================
# Figure 5: Memory Profile (H200, bf16)
# =========================================================================
def fig_memory_profile():
    """Peak memory across ranks and sequence lengths (H200, bf16)."""
    data = load_comprehensive('h200', 'bf16')
    ranks = [16, 64, 128, 256, 384, 512]

    eager_fwd, fused_fwd, eager_bwd = [], [], []
    for r in ranks:
        em = filter_memory(data, hidden=4096, config='dorafactors_eager',
                          rank=r, batch=4, seqlen=2048)
        fm = filter_memory(data, hidden=4096, config='dorafactors_fully_fused',
                          rank=r, batch=4, seqlen=2048)
        eager_fwd.append(em[0]['post_fwd_peak_mb'])
        fused_fwd.append(fm[0]['post_fwd_peak_mb'])
        eager_bwd.append(em[0]['post_bwd_peak_mb'])

    seq_configs = [
        (1, 1, 'bs=1\nseq=1'),
        (2, 512, 'bs=2\nseq=512'),
        (4, 2048, 'bs=4\nseq=2048'),
        (4, 4096, 'bs=4\nseq=4096'),
    ]
    eager_fwd_seq, fused_fwd_seq, eager_bwd_seq = [], [], []
    for batch, seqlen, _ in seq_configs:
        em = filter_memory(data, hidden=4096, config='dorafactors_eager',
                          rank=384, batch=batch, seqlen=seqlen)
        fm = filter_memory(data, hidden=4096, config='dorafactors_fully_fused',
                          rank=384, batch=batch, seqlen=seqlen)
        eager_fwd_seq.append(em[0]['post_fwd_peak_mb'])
        fused_fwd_seq.append(fm[0]['post_fwd_peak_mb'])
        eager_bwd_seq.append(em[0]['post_bwd_peak_mb'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    x = np.arange(len(ranks))
    w = 0.27

    ax1.bar(x - w, eager_fwd, w, label='Eager Fwd Peak', color=C_EAGER, alpha=0.7)
    ax1.bar(x, fused_fwd, w, label='Fused Fwd Peak', color=C_FUSED, alpha=0.7)
    ax1.bar(x + w, eager_bwd, w, label='Bwd Peak (both)', color=C_FUSED_AG, alpha=0.5)
    ax1.set_xlabel('LoRA Rank')
    ax1.set_ylabel('Peak VRAM (MB)')
    ax1.set_title('(a) Peak Memory vs. Rank\n(H200, bf16, h=4096, bs=4, seq=2048)', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(r) for r in ranks])
    ax1.legend(fontsize=7.5, framealpha=0.9, loc='upper left')
    style_axis(ax1, grid='y')

    x2 = np.arange(len(seq_configs))
    seq_labels = [sc[2] for sc in seq_configs]
    ax2.bar(x2 - w, eager_fwd_seq, w, label='Eager Fwd Peak', color=C_EAGER, alpha=0.7)
    ax2.bar(x2, fused_fwd_seq, w, label='Fused Fwd Peak', color=C_FUSED, alpha=0.7)
    ax2.bar(x2 + w, eager_bwd_seq, w, label='Bwd Peak (both)', color=C_FUSED_AG, alpha=0.5)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Peak VRAM (MB)')
    ax2.set_title('(b) Peak Memory vs. Batch/Seq\n(H200, bf16, h=4096, r=384)', fontsize=9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(seq_labels, fontsize=8)
    for i in [2, 3]:
        saving = eager_fwd_seq[i] - fused_fwd_seq[i]
        if saving > 0:
            ax2.annotate(f'-{saving:.0f} MB', xy=(x2[i], fused_fwd_seq[i]),
                        xytext=(0, -12), textcoords='offset points',
                        fontsize=7.5, ha='center', color='black', fontweight='bold')
    style_axis(ax2, grid='y')

    finalize_figure(fig, 'memory_profile')


# =========================================================================
# Figure 6: Norm Time vs Rank (RTX PRO 6000, fp32)
# =========================================================================
def fig_norm_time_vs_rank():
    """Norm computation latency vs rank: reference is constant, factored scales."""
    data = load_comprehensive('rtx6000', 'fp32')
    norms = filter_norm(data, dtype='float32')

    by_base = {}
    for e in norms:
        base = (e['shape'][0], e['shape'][1])
        by_base.setdefault(base, []).append(e)

    panels = [
        ((4096, 4096), '(a) 4096 x 4096, fp32'),
        ((8192, 8192), '(b) 8192 x 8192, fp32'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    for ax, (base, title) in zip(axes, panels):
        entries = sorted(by_base.get(base, []), key=lambda e: e['shape'][2])
        ranks = [e['shape'][2] for e in entries]
        ref = [e['ref_time_ms'] for e in entries]
        fact = [e['factored_time_ms'] for e in entries]
        fused = [e['fused_time_ms'] for e in entries]

        ax.plot(ranks, ref, 'o-', color=C_REF, label='Reference', markersize=4, linewidth=1.5)
        ax.plot(ranks, fact, 's-', color=C_FACT, label='Factored', markersize=4, linewidth=1.5)
        ax.plot(ranks, fused, '^-', color=C_FUSED, label='Fused', markersize=4, linewidth=1.5)
        ax.set_xlabel('LoRA Rank')
        ax.set_ylabel('Norm Time (ms)')
        ax.set_title(title)
        ax.legend(framealpha=0.9, fontsize=8)
        ax.set_xticks(ranks)
        style_axis(ax, grid='y')

    fig.suptitle('RTX PRO 6000: Norm Computation Latency vs. Rank', fontsize=10, y=1.01)
    finalize_figure(fig, 'norm_time_vs_rank')


# =========================================================================
# Figure 7: Bandwidth Utilization (6 GPUs, fp32)
# =========================================================================
def fig_bandwidth_utilization():
    """Memory bandwidth utilization: eager vs fused compose, 6 GPUs, fp32.
    Note: eager bandwidth values are approximate lower bounds (the benchmark uses
    a nominal total_bytes model that understates actual device traffic for unfused
    paths with 3+ kernel launches)."""
    gpu_data = {}
    for gpu in MICRO_GPUS:
        d = load_comprehensive(gpu, 'fp32')
        gpu_data[gpu] = filter_compose(d, dtype='float32')

    ref_gpu = 'h200'
    sort_order = sorted(range(len(gpu_data[ref_gpu])),
                        key=lambda i: (gpu_data[ref_gpu][i]['shape'][1],
                                       gpu_data[ref_gpu][i]['shape'][0] * gpu_data[ref_gpu][i]['shape'][1]))
    for gpu in MICRO_GPUS:
        gpu_data[gpu] = [gpu_data[gpu][i] for i in sort_order]

    shapes = [shape_label(e['shape']) for e in gpu_data[ref_gpu]]
    n = len(shapes)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    for gpu in MICRO_GPUS:
        info = GPU_REGISTRY[gpu]
        entries = gpu_data[gpu]
        eager_bw = [e['approx_eager_bw_gbps'] for e in entries]
        fused_bw = [e['approx_fused_bw_gbps'] for e in entries]
        ax.plot(range(n), eager_bw, '--', color=info['color'],
                marker='o', markersize=4.1, linewidth=0.9, alpha=0.45,
                markerfacecolor='white', markeredgewidth=1.0)
        ax.plot(range(n), fused_bw, '-', color=info['color'],
                marker='D', markersize=4.6, linewidth=1.1)

    # Peak bandwidth reference lines
    seen_peaks = set()
    for gpu in MICRO_GPUS:
        info = GPU_REGISTRY[gpu]
        peak = info['peak_bw_gbps']
        ax.axhline(y=peak, color=info['color'], linestyle=':', linewidth=1.2, alpha=0.8)
        # B200 and B300 share 7700 GB/s — combined label
        if peak in seen_peaks:
            continue
        seen_peaks.add(peak)
        if peak == 7700:
            label_text = 'B200/B300 peak 7.7 TB/s'
        else:
            label_text = f'{info["label"]} peak {peak} GB/s'
        ax.text(n - 0.5, peak * 1.02, label_text,
               fontsize=7, color=info['color'], va='bottom', ha='right',
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))

    ax.set_xlabel('Activation Shape')
    ax.set_ylabel('Approx. Bandwidth (GB/s)')
    ax.set_title('Compose Kernel Bandwidth Utilization (fp32)')
    ax.set_xticks(range(n))
    ax.set_xticklabels(shapes, fontsize=6.8, rotation=48, ha='right')
    style_axis(ax, grid='y')

    gpu_legend = ax.legend(handles=gpu_color_legend_handles(MICRO_GPUS),
                           title='GPU (color)',
                           loc='upper left', bbox_to_anchor=(0.0, 1.0),
                           fontsize=7.5, title_fontsize=7.5, ncol=3,
                           columnspacing=1.0, handlelength=2.0,
                           borderpad=0.4, labelspacing=0.4)
    mode_legend = ax.legend(handles=[
         Line2D([0], [0], color='black', linestyle='--', marker='o', markersize=4.1,
             linewidth=1.0, alpha=0.45, markerfacecolor='white',
             markeredgewidth=1.0, label='Eager (approx.)'),
         Line2D([0], [0], color='black', linestyle='-', marker='D', markersize=4.6,
               linewidth=1.4, label='Fused'),
     ], title='Kernel path (shape)',
         loc='upper left', bbox_to_anchor=(0.0, 0.83), fontsize=7.5,
         title_fontsize=7.5, ncol=2, columnspacing=1.2, handlelength=1.8,
         borderpad=0.4, labelspacing=0.4)
    ax.add_artist(gpu_legend)

    max_peak = max(GPU_REGISTRY[g]['peak_bw_gbps'] for g in MICRO_GPUS)
    ax.set_ylim(0, max_peak * 1.15)

    finalize_figure(fig, 'bandwidth')


# =========================================================================
# Figure 8 (appendix): E2E Cross-GPU Speedup (6 GPUs, bf16)
# =========================================================================
def fig_e2e_cross_gpu():
    """E2E step time speedup: fully fused vs eager, 6 GPUs, bf16.
    APPENDIX figure: single-layer decomposition across GPUs."""
    ranks = [16, 64, 128, 256, 384, 512]

    gpu_speedups = {}
    for gpu in MICRO_GPUS:
        data = load_comprehensive(gpu, 'bf16')
        eager_entries = filter_e2e(data, hidden=4096, config='dorafactors_eager',
                                   batch=4, seqlen=2048)
        fused_entries = filter_e2e(data, hidden=4096, config='dorafactors_fully_fused',
                                   batch=4, seqlen=2048)
        eager_by_rank = {e['shape']['rank']: e['step_ms'] for e in eager_entries}
        fused_by_rank = {e['shape']['rank']: e['step_ms'] for e in fused_entries}
        gpu_speedups[gpu] = [eager_by_rank[r] / fused_by_rank[r] for r in ranks]

    all_spd = [v for gpu in MICRO_GPUS for v in gpu_speedups[gpu]]
    bottom_val = min(0.93, min(all_spd) - 0.02)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(ranks))
    n_gpus = len(MICRO_GPUS)
    w = 0.13
    offsets = np.linspace(-(n_gpus - 1) * w / 2, (n_gpus - 1) * w / 2, n_gpus)

    for gpu, offset in zip(MICRO_GPUS, offsets):
        info = GPU_REGISTRY[gpu]
        spd = gpu_speedups[gpu]
        ax.bar(x + offset, [s - bottom_val for s in spd], w, bottom=bottom_val, label=info['label'],
               color=info['color'], alpha=0.8)
        for i, v in enumerate(spd):
            ax.text(x[i] + offset, v + 0.005, f'{v:.2f}x',
                   ha='center', va='bottom', fontsize=6, fontweight='bold',
                   color='black', rotation=90)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('Single-Layer Speedup (Eager / Fused)')
    ax.set_title('Single-Layer E2E Speedup Across GPUs (bf16, h=4096)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend(framealpha=0.9, fontsize=7.5, ncol=2)
    style_axis(ax, grid='y')

    ax.set_ylim(bottom_val, max(all_spd) + 0.15)

    finalize_figure(fig, 'e2e_cross_gpu')


# =========================================================================
# Figure 9: Dispatch Diagram (no data, hand-drawn)
# =========================================================================
def fig_dispatch_diagram():
    """Dispatch architecture (simplified block diagram). No data dependency."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    box_kw = dict(boxstyle='round,pad=0.35', facecolor='#f0f0f0', edgecolor='#333333', linewidth=1.2)
    fused_kw = dict(boxstyle='round,pad=0.35', facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=1.2)
    eager_kw = dict(boxstyle='round,pad=0.35', facecolor='#fff3e0', edgecolor='#e65100', linewidth=1.2)
    train_kw = dict(boxstyle='round,pad=0.35', facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=1.2)
    decision_kw = dict(boxstyle='round,pad=0.3', facecolor='#fff9c4', edgecolor='#f57f17', linewidth=1.0)

    ax.text(5, 7.5, '_compose_with_dispatch', fontsize=10, fontweight='bold',
           ha='center', va='center', bbox=box_kw, family='monospace')
    ax.annotate('', xy=(5, 6.55), xytext=(5, 7.05),
               arrowprops=dict(arrowstyle='->', lw=1.2, color='#333'))
    ax.text(5, 6.2, 'requires_grad?', fontsize=10, ha='center', va='center',
           bbox=decision_kw, fontweight='bold')

    ax.annotate('', xy=(2.5, 4.95), xytext=(4.1, 5.85),
               arrowprops=dict(arrowstyle='->', lw=1.2, color='#1565c0'))
    ax.text(3.1, 5.55, 'Yes', fontsize=9, color='#1565c0', fontweight='bold')
    ax.text(2.5, 4.5, 'Fused Backward\n(Triton fwd+bwd)', fontsize=9,
           ha='center', va='center', bbox=train_kw)
    ax.text(2.5, 3.4, 'DEFAULT for training\nSaves inner for bwd;\nskips if mag frozen',
           fontsize=7.5, ha='center', va='center', color='#1565c0', linespacing=1.4)

    ax.annotate('', xy=(7.5, 4.95), xytext=(5.9, 5.85),
               arrowprops=dict(arrowstyle='->', lw=1.2, color='#2e7d32'))
    ax.text(6.9, 5.55, 'No', fontsize=9, color='#2e7d32', fontweight='bold')
    ax.text(7.5, 4.5, 'Fused Forward\n(Triton fwd only)', fontsize=9,
           ha='center', va='center', bbox=fused_kw)
    ax.text(7.5, 3.4, 'DEFAULT for inference\nNo autograd nodes;\nno saved tensors',
           fontsize=7.5, ha='center', va='center', color='#2e7d32', linespacing=1.4)

    ax.text(5, 1.5, 'Eager Fallback (PyTorch ops)', fontsize=9,
           ha='center', va='center', bbox=eager_kw)
    ax.text(5, 0.6, 'CPU / non-contiguous / Triton unavailable / env disabled',
           fontsize=7.5, ha='center', va='center', color='#e65100', style='italic')

    ax.annotate('', xy=(4.0, 1.75), xytext=(2.5, 2.85),
               arrowprops=dict(arrowstyle='->', lw=0.8, linestyle='--', color='#555'))
    ax.text(2.8, 2.4, 'fallback', fontsize=7, color='#555', ha='center')
    ax.annotate('', xy=(6.0, 1.75), xytext=(7.5, 2.85),
               arrowprops=dict(arrowstyle='->', lw=0.8, linestyle='--', color='#555'))
    ax.text(7.2, 2.4, 'fallback', fontsize=7, color='#555', ha='center')

    finalize_figure(fig, 'dispatch')


# =========================================================================
# Figure 10: Model-Level Grad Compute Speedup (3 GPUs x 6 models, two-panel)
# =========================================================================
def fig_model_training_speedup():
    """Gradient computation speedup across 3 GPUs and 6 models.
    Two-panel figure:
      (a) PEFT/fused speedup (headline comparison: what practitioners use)
      (b) eager/fused speedup (internal baseline)
    Shows OOM indicators for models that don't fit."""
    gpu_model_data = {}
    for gpu in MODEL_GPUS:
        gpu_model_data[gpu] = load_models(gpu)

    # Build lookup: (model, gpu, config) -> iter_ms
    def _build_lookup(gpu):
        entries = gpu_model_data[gpu]
        lookup = {}
        for e in entries:
            if e['pass'] != 'grad_compute':
                continue
            if e.get('error'):
                continue
            lookup[(e['model'], e['config'])] = e['iter_ms']
        return lookup

    lookups = {gpu: _build_lookup(gpu) for gpu in MODEL_GPUS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.1, 4.0))

    n_models = len(MODEL_ORDER)
    n_gpus = len(MODEL_GPUS)
    w = 0.22
    x = np.arange(n_models)
    offsets = np.linspace(-(n_gpus - 1) * w / 2, (n_gpus - 1) * w / 2, n_gpus)

    # Panel (a): PEFT / fused
    for gpu, offset in zip(MODEL_GPUS, offsets):
        info = GPU_REGISTRY[gpu]
        vals = []
        oom_indices = []
        for i, model in enumerate(MODEL_ORDER):
            peft_ms = lookups[gpu].get((model, 'baseline_hf_peft'))
            fused_ms = lookups[gpu].get((model, 'dorafactors_fully_fused'))
            if peft_ms is not None and fused_ms is not None:
                vals.append(peft_ms / fused_ms)
            else:
                vals.append(0)
                oom_indices.append(i)

        ax1.bar(x + offset, vals, w, label=info['label'],
               color=info['color'], alpha=0.85)
        for i, v in enumerate(vals):
            if v > 0:
                ax1.text(x[i] + offset, v - 0.04, f'{v:.2f}x',
                        ha='center', va='top', fontsize=6, fontweight='bold',
                        color='black', rotation=90)
        for i in oom_indices:
            ax1.text(x[i] + offset, 0.05, 'OOM',
                    ha='center', va='bottom', fontsize=6, fontweight='bold',
                    color='black', rotation=90)

    ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Speedup (PEFT / Fused)')
    ax1.set_title('(a) vs. HF PEFT Baseline', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=7.5, rotation=20, ha='right')
    style_axis(ax1, grid='y')

    # Panel (b): eager / fused
    for gpu, offset in zip(MODEL_GPUS, offsets):
        info = GPU_REGISTRY[gpu]
        vals = []
        for i, model in enumerate(MODEL_ORDER):
            eager_ms = lookups[gpu].get((model, 'dorafactors_eager'))
            fused_ms = lookups[gpu].get((model, 'dorafactors_fully_fused'))
            if eager_ms is not None and fused_ms is not None:
                vals.append(eager_ms / fused_ms)
            else:
                vals.append(0)

        ax2.bar(x + offset, vals, w, label=info['label'],
               color=info['color'], alpha=0.85)
        for i, v in enumerate(vals):
            if v > 0:
                ax2.text(x[i] + offset, v - 0.02, f'{v:.2f}x',
                        ha='center', va='top', fontsize=6, fontweight='bold',
                        color='black', rotation=90)

    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Speedup (Eager / Fused)')
    ax2.set_title('(b) vs. Eager Baseline', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=7.5, rotation=20, ha='right')
    style_axis(ax2, grid='y')

    # Set y-limits
    all_peft_vals = []
    all_eager_vals = []
    for gpu in MODEL_GPUS:
        for model in MODEL_ORDER:
            peft_ms = lookups[gpu].get((model, 'baseline_hf_peft'))
            fused_ms = lookups[gpu].get((model, 'dorafactors_fully_fused'))
            eager_ms = lookups[gpu].get((model, 'dorafactors_eager'))
            if peft_ms and fused_ms:
                all_peft_vals.append(peft_ms / fused_ms)
            if eager_ms and fused_ms:
                all_eager_vals.append(eager_ms / fused_ms)

    if all_peft_vals:
        ax1.set_ylim(0, max(all_peft_vals) + 0.25)
    if all_eager_vals:
        eager_bottom = 0.9
        ax2.set_ylim(eager_bottom, max(all_eager_vals) + 0.1)
        # Re-place OOM markers for panel (b) within visible area
        for gpu, offset in zip(MODEL_GPUS, offsets):
            for i, model in enumerate(MODEL_ORDER):
                eager_ms = lookups[gpu].get((model, 'dorafactors_eager'))
                fused_ms = lookups[gpu].get((model, 'dorafactors_fully_fused'))
                if eager_ms is None or fused_ms is None:
                    ax2.text(x[i] + offset, eager_bottom + 0.005, 'OOM',
                            ha='center', va='bottom', fontsize=6, fontweight='bold',
                            color='black', rotation=90)

    fig.suptitle('Gradient Computation Speedup: Fully Fused (bf16, r=384, seq=4096, 1024 loss tokens)',
                fontsize=10, y=1.02)
    fig.legend(handles=gpu_legend_handles(MODEL_GPUS), loc='upper center',
               bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=7.8,
               columnspacing=1.5, handlelength=1.8)
    finalize_figure(fig, 'model_training_speedup', rect=[0, 0, 1, 0.99], h_pad=1.0, w_pad=0.9)


# =========================================================================
# Figure 11: Model-Level Inference Speedup (3 GPUs x 6 models)
# =========================================================================
def fig_model_inference_speedup():
    """Inference speedup (fully_fused vs PEFT) across 3 GPUs and 6 models.
    Forward-pass only. No OOMs expected (inference uses less VRAM than training)."""
    gpu_model_data = {}
    for gpu in MODEL_GPUS:
        gpu_model_data[gpu] = load_models(gpu)

    def _build_lookup(gpu):
        entries = gpu_model_data[gpu]
        lookup = {}
        for e in entries:
            if e['pass'] != 'inference':
                continue
            if e.get('error'):
                continue
            lookup[(e['model'], e['config'])] = e['step_ms']
        return lookup

    lookups = {gpu: _build_lookup(gpu) for gpu in MODEL_GPUS}

    fig, ax = plt.subplots(figsize=(7, 3.5))

    n_models = len(MODEL_ORDER)
    n_gpus = len(MODEL_GPUS)
    w = 0.22
    x = np.arange(n_models)
    offsets = np.linspace(-(n_gpus - 1) * w / 2, (n_gpus - 1) * w / 2, n_gpus)

    for gpu, offset in zip(MODEL_GPUS, offsets):
        info = GPU_REGISTRY[gpu]
        vals = []
        for i, model in enumerate(MODEL_ORDER):
            peft_ms = lookups[gpu].get((model, 'baseline_hf_peft'))
            fused_ms = lookups[gpu].get((model, 'dorafactors_fully_fused'))
            if peft_ms is not None and fused_ms is not None:
                vals.append(peft_ms / fused_ms)
            else:
                vals.append(0)

        ax.bar(x + offset, vals, w, label=info['label'],
               color=info['color'], alpha=0.85)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(x[i] + offset, v - 0.04, f'{v:.2f}x',
                        ha='center', va='top', fontsize=6.5, fontweight='bold',
                        color='black', rotation=90)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Inference Speedup (PEFT / Fused)')
    ax.set_title('Inference Speedup: Fully Fused vs. HF PEFT (bf16, r=384, seq=4096)')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=7.5, rotation=20, ha='right')
    ax.legend(framealpha=0.9, fontsize=7.5, loc='upper right')
    style_axis(ax, grid='y')

    all_vals = []
    for gpu in MODEL_GPUS:
        for model in MODEL_ORDER:
            peft_ms = lookups[gpu].get((model, 'baseline_hf_peft'))
            fused_ms = lookups[gpu].get((model, 'dorafactors_fully_fused'))
            if peft_ms and fused_ms:
                all_vals.append(peft_ms / fused_ms)
    if all_vals:
        ax.set_ylim(0, max(all_vals) + 0.25)

    finalize_figure(fig, 'model_inference_speedup')


# =========================================================================
# Figure 12: Dense-BA Comparison (3 GPUs, grad_compute)
# =========================================================================
def fig_dense_ba_comparison():
    """Dense-BA vs Fused vs Eager: shows where dense_ba falls in the gap.
    Gradient computation time, 3 GPUs, models that fit on all GPUs.
    Negative gap values mean dense-BA is WORSE than eager for that model/GPU."""
    # Only use models that fit on all 3 GPUs for grad_compute
    common_models = [
        'Qwen/Qwen3.5-27B',
        'google/gemma-3-27b-it',
        'unsloth/Mistral-Small-3.2-24B-Instruct-2506',
        'Qwen/Qwen3-VL-8B-Instruct',
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    n_models = len(common_models)
    n_gpus = len(MODEL_GPUS)
    w = 0.22
    x = np.arange(n_models)
    offsets = np.linspace(-(n_gpus - 1) * w / 2, (n_gpus - 1) * w / 2, n_gpus)

    for gpu, offset in zip(MODEL_GPUS, offsets):
        entries = load_models(gpu) or []
        eager_t = {}
        fused_t = {}
        dense_t = {}
        for e in entries:
            if e['pass'] != 'grad_compute' or e.get('error'):
                continue
            if e['config'] == 'dorafactors_eager':
                eager_t[e['model']] = e['iter_ms']
            elif e['config'] == 'dorafactors_fully_fused':
                fused_t[e['model']] = e['iter_ms']
            elif e['config'] == 'baseline_dense_ba':
                dense_t[e['model']] = e['iter_ms']

        info = GPU_REGISTRY[gpu]
        gap_pcts = []
        for model in common_models:
            if model in eager_t and model in fused_t and model in dense_t:
                eager_v = eager_t[model]
                fused_v = fused_t[model]
                dense_v = dense_t[model]
                gap = eager_v - fused_v
                if abs(gap) > 0.01:
                    pct = (eager_v - dense_v) / gap * 100
                else:
                    pct = 0
                gap_pcts.append(pct)
            else:
                gap_pcts.append(0)

        ax.bar(x + offset, gap_pcts, w, label=info['label'],
               color=info['color'], alpha=0.85)
        for i, v in enumerate(gap_pcts):
            if abs(v) > 0.5:
                y_pos = v + (1.5 if v >= 0 else -1.5)
                va = 'bottom' if v >= 0 else 'top'
                ax.text(x[i] + offset, y_pos, f'{v:.0f}%',
                       ha='center', va=va, fontsize=7, fontweight='bold',
                       color='black', rotation=90)

    ax.axhline(y=100, color=C_FUSED, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0, color=C_EAGER, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(n_models - 0.5, 102, 'Fully Fused', fontsize=7.5, color=C_FUSED, va='bottom', ha='right')
    ax.text(n_models - 0.5, -5, 'Eager', fontsize=7.5, color=C_EAGER, va='top', ha='right')
    ax.set_xlabel('Model')
    ax.set_ylabel('Gap Closed by Dense-BA (%)')
    ax.set_title('Dense-BA Position: 0% = Eager, 100% = Fully Fused (Grad Compute)')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in common_models], fontsize=8)
    ax.legend(framealpha=0.9, fontsize=7.5)
    style_axis(ax, grid='y')

    ax.set_ylim(-15, 115)

    finalize_figure(fig, 'dense_ba_comparison')


# =========================================================================
# Figure 13: Numerical Stability (H200, bf16)
# =========================================================================
def fig_stability():
    """Numerical stability: naive vs stable vs fused at near-unity m values.
    Results should be identical across GPUs (numerical property, not hardware-dependent)."""
    data = load_comprehensive('h200', 'bf16')
    stab = data['benchmarks']['stability']

    entries = [e for e in stab if tuple(e['shape']) == (2048, 8192)]
    entries.sort(key=lambda e: e['m'])

    m_vals = [e['m'] for e in entries]
    naive_err = [e['methods']['naive_bf16']['max_abs_error'] for e in entries]
    stable_err = [e['methods']['stable_bf16']['max_abs_error'] for e in entries]
    fused_err = [e['methods']['fused_bf16']['max_abs_error'] for e in entries]
    qfloor = [e['methods']['quantization_floor']['max_abs_error'] for e in entries]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    m_labels = [str(m) for m in m_vals]
    x = np.arange(len(m_vals))

    ax.plot(x, naive_err, 'o-', color=C_REF, label='Naive bf16', markersize=5, linewidth=1.5)
    ax.plot(x, stable_err, 's-', color=C_FACT, label='Stable bf16', markersize=5, linewidth=1.5)
    ax.plot(x, fused_err, '^-', color=C_FUSED, label='Fused bf16', markersize=5, linewidth=1.5)
    ax.plot(x, qfloor, 'x--', color='gray', label='BF16 quant. floor',
           markersize=5, linewidth=1.0, alpha=0.6)

    ax.set_yscale('log')
    ax.set_xlabel('Magnitude parameter $m$\n($g = m\\,/\\,\\|W\\|_\\mathrm{row}$; benchmark uses unit-norm rows so $g \\approx m$)')
    ax.set_ylabel('Max Absolute Error')
    ax.set_title('Numerical Stability: DoRA Compose at Near-Unity $g$\n(shape 2048×8192, bf16)')
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=8)
    ax.legend(framealpha=0.9, fontsize=8)
    style_axis(ax, grid='y')

    worst_naive_idx = max(range(len(naive_err)), key=lambda i: naive_err[i] / fused_err[i])
    ratio = naive_err[worst_naive_idx] / fused_err[worst_naive_idx]
    ax.annotate(f'{ratio:.1f}x improvement',
               xy=(worst_naive_idx, fused_err[worst_naive_idx]),
               xytext=(worst_naive_idx + 1.5, fused_err[worst_naive_idx] * 3),
               fontsize=7.5, color='black', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    finalize_figure(fig, 'stability')


# =========================================================================
# Figure 13 (appendix): E2E Speedup vs Hidden Dimension (6 GPUs, bf16)
# =========================================================================
def fig_e2e_hidden_sweep():
    """E2E speedup vs hidden dimension at r=384, showing the fusion sweet spot.
    APPENDIX figure: single-layer speedup vs hidden dimension."""
    hidden_dims = [512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 28672]

    gpu_speedups = {}
    for gpu in MICRO_GPUS:
        data = load_comprehensive(gpu, 'bf16')
        eager_lookup = {}
        fused_lookup = {}
        for e in data['benchmarks']['e2e']:
            s = e['shape']
            if s['rank'] != 384:
                continue
            key = s['hidden']
            if e['config'] == 'dorafactors_eager':
                eager_lookup[key] = e['step_ms']
            elif e['config'] == 'dorafactors_fully_fused':
                fused_lookup[key] = e['step_ms']
        speedups = []
        for h in hidden_dims:
            if h in eager_lookup and h in fused_lookup:
                speedups.append(eager_lookup[h] / fused_lookup[h])
            else:
                speedups.append(None)
        gpu_speedups[gpu] = speedups

    fig, ax = plt.subplots(figsize=(6.3, 3.8))

    for gpu in MICRO_GPUS:
        info = GPU_REGISTRY[gpu]
        spd = gpu_speedups[gpu]
        xs = [i for i, v in enumerate(spd) if v is not None]
        ys = [v for v in spd if v is not None]
        ax.plot(xs, ys, GPU_MARKERS[gpu], color=info['color'],
               label=info['label'], markersize=5, linewidth=1.5)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhspan(0, 1.0, alpha=0.03, color=C_REF)

    all_ys = [v for spd in gpu_speedups.values() for v in spd if v is not None]
    max_y = max(all_ys) if all_ys else 1.22

    ax.set_ylim(min(all_ys) - 0.05, max_y + 0.15)

    def _fmt_h(h):
        if h >= 1024:
            return f'{h // 1024}k'
        return str(h)
    ax.set_xticks(range(len(hidden_dims)))
    ax.set_xticklabels([_fmt_h(h) for h in hidden_dims], fontsize=8)
    ax.set_xlabel('Hidden Dimension')
    ax.set_ylabel('Single-Layer E2E Speedup (Eager / Fused)')
    ax.set_title('Single-Layer E2E Speedup vs. Hidden Dimension (bf16, r=384)')
    ax.legend(framealpha=0.9, fontsize=7.5, ncol=2)
    style_axis(ax, grid='y')

    finalize_figure(fig, 'e2e_hidden_sweep')


# =========================================================================
# Startup validation: print geomean speedups from summary sections
# =========================================================================
def print_summary_speedups():
    """Print geomean speedups from each GPU's summary section for cross-reference."""
    print('=== Geomean Speedups (from bench_it6 JSON summary sections, bf16) ===')
    print(f'{"GPU":>8s}  {"Compose Fwd":>12s}  {"Compose OOP":>12s}  {"Compose AG":>11s}  '
          f'{"Backward":>9s}  {"E2E eag/fus":>12s}  {"E2E peft/fus":>13s}')
    for gpu in MICRO_GPUS:
        d = load_comprehensive(gpu, 'bf16')
        s = d['summary']
        print(f'{GPU_REGISTRY[gpu]["label"]:>8s}  '
              f'{s["compose_geomean_fwd_speedup"]:>12.2f}  '
              f'{s["compose_geomean_oop_speedup"]:>12.2f}  '
              f'{s["compose_geomean_ag_speedup"]:>11.2f}  '
              f'{s["backward_geomean_speedup"]:>9.2f}  '
              f'{s["e2e_speedup_eager_over_fully_fused"]:>12.2f}  '
              f'{s["e2e_speedup_hf_peft_over_fully_fused"]:>13.2f}')

    print()
    print('=== Model-Level Grad Compute Speedups (from raw data) ===')
    for gpu in MODEL_GPUS:
        entries = load_models(gpu) or []
        print(f'\n  {GPU_REGISTRY[gpu]["label"]}:')
        eager_t = {}
        fused_t = {}
        peft_t = {}
        peak_vram = {}
        for e in entries:
            if e['pass'] != 'grad_compute' or e.get('error'):
                continue
            if e['config'] == 'dorafactors_eager':
                eager_t[e['model']] = e['iter_ms']
                peak_vram[(e['model'], 'eager')] = e['peak_vram_mb']
            elif e['config'] == 'dorafactors_fully_fused':
                fused_t[e['model']] = e['iter_ms']
                peak_vram[(e['model'], 'fused')] = e['peak_vram_mb']
            elif e['config'] == 'baseline_hf_peft':
                peft_t[e['model']] = e['iter_ms']
                peak_vram[(e['model'], 'peft')] = e['peak_vram_mb']

        for model in MODEL_ORDER:
            short = MODEL_SHORT[model]
            if model in fused_t:
                parts = [f'{short:>18s}']
                if model in eager_t:
                    parts.append(f'eag/fus={eager_t[model]/fused_t[model]:.3f}x')
                if model in peft_t:
                    parts.append(f'peft/fus={peft_t[model]/fused_t[model]:.3f}x')
                fused_vram = peak_vram.get((model, 'fused'), 0)
                peft_vram = peak_vram.get((model, 'peft'), 0)
                if fused_vram and peft_vram:
                    parts.append(f'VRAM delta={(peft_vram-fused_vram)/1024:.2f} GB')
                print(f'    {" | ".join(parts)}')
            else:
                print(f'    {short:>18s} | OOM')

    print()
    print('=== Model-Level Inference Speedups (from raw data) ===')
    for gpu in MODEL_GPUS:
        entries = load_models(gpu) or []
        print(f'\n  {GPU_REGISTRY[gpu]["label"]}:')
        peft_t = {}
        fused_t = {}
        eager_t = {}
        for e in entries:
            if e['pass'] != 'inference' or e.get('error'):
                continue
            if e['config'] == 'baseline_hf_peft':
                peft_t[e['model']] = e['step_ms']
            elif e['config'] == 'dorafactors_fully_fused':
                fused_t[e['model']] = e['step_ms']
            elif e['config'] == 'dorafactors_eager':
                eager_t[e['model']] = e['step_ms']
        for model in MODEL_ORDER:
            short = MODEL_SHORT[model]
            if model in fused_t:
                parts = [f'{short:>18s}']
                if model in peft_t:
                    parts.append(f'peft/fus={peft_t[model]/fused_t[model]:.3f}x')
                if model in eager_t:
                    parts.append(f'eag/fus={eager_t[model]/fused_t[model]:.3f}x')
                print(f'    {" | ".join(parts)}')
            else:
                print(f'    {short:>18s} | OOM')
    print()


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    print('Generating figures from 6-GPU JSON data (bench_it6)...')
    print()

    validate_manifest()
    print_summary_speedups()

    print('--- Main-body figures ---')
    fig_norm_memory()
    fig_compose_speedup()
    fig_backward_speedup()
    fig_memory_profile()
    fig_norm_time_vs_rank()
    fig_bandwidth_utilization()
    fig_dispatch_diagram()
    fig_model_training_speedup()
    fig_model_inference_speedup()
    fig_dense_ba_comparison()
    fig_stability()
    print()
    print('--- Appendix figures (single-layer E2E decomposition) ---')
    fig_e2e_step_time()
    fig_e2e_cross_gpu()
    fig_e2e_hidden_sweep()
    print()
    exts = '/'.join([f.upper() for f in OUTPUT_FORMATS])
    print(f'Done. All {exts}s in paper/figures/')
