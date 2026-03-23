#!/usr/bin/env python3
"""Generate convergence equivalence figure from TensorBoard event files.

Reads TensorBoard events from code/convergence_runs/ and produces a 3-panel
figure showing that eager and fused DoRA training are indistinguishable.

Usage:
    cd paper && python3 generate_training_figure.py [--png]
"""

from __future__ import annotations

import glob
import struct
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Optional tensorboard path
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
    HAS_TB = True
except Exception:
    HAS_TB = False

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

BASE_DIR = Path(__file__).resolve().parent
CONVDIR = BASE_DIR.parent / 'code' / 'convergence_runs'
FIGDIR = BASE_DIR / 'figures'
FIGDIR.mkdir(exist_ok=True)

# Accessible method colors
C_EAGER = '#D55E00'  # vermillion
C_FUSED = '#0072B2'  # blue

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linewidth': 0.55,
    'grid.color': '#999999',
    'axes.axisbelow': True,
})

# ---------------------------------------------------------------------------
# Minimal TF Event parser fallback
# ---------------------------------------------------------------------------
def _build_event_message():
    fd = descriptor_pb2.FileDescriptorProto()
    fd.name = 'event_minimal.proto'
    fd.package = 'minimal'
    fd.syntax = 'proto3'

    def add_field(msg, name, num, typ, label=1, type_name=None):
        f = msg.field.add()
        f.name = name
        f.number = num
        f.type = typ
        f.label = label
        if type_name is not None:
            f.type_name = type_name

    m = fd.message_type.add()
    m.name = 'TensorProto'
    add_field(m, 'dtype', 1, 5)
    add_field(m, 'tensor_content', 4, 12)
    add_field(m, 'float_val', 5, 2, label=3)
    add_field(m, 'double_val', 6, 1, label=3)
    add_field(m, 'int_val', 7, 5, label=3)
    add_field(m, 'int64_val', 10, 3, label=3)
    add_field(m, 'bool_val', 11, 8, label=3)
    add_field(m, 'half_val', 13, 5, label=3)

    m = fd.message_type.add()
    m.name = 'Summary'
    n = m.nested_type.add()
    n.name = 'Value'
    add_field(n, 'tag', 1, 9)
    add_field(n, 'simple_value', 2, 2)
    add_field(n, 'tensor', 8, 11, type_name='.minimal.TensorProto')
    add_field(m, 'value', 1, 11, label=3, type_name='.minimal.Summary.Value')

    m = fd.message_type.add()
    m.name = 'Event'
    add_field(m, 'wall_time', 1, 1)
    add_field(m, 'step', 2, 3)
    add_field(m, 'file_version', 3, 9)
    add_field(m, 'summary', 5, 11, type_name='.minimal.Summary')

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('minimal.Event'))

EventMessage = _build_event_message()

def _tfrecord_records(path: Path):
    """Yield raw TFRecord payloads from a TensorFlow TFRecord file.

    Each TFRecord file is a sequence of records, where each record has the
    following binary layout (see TensorFlow TFRecord format):

        uint64  length        # little-endian, number of bytes in payload
        uint32  length_crc    # masked CRC32C of the length field
        bytes   data[length]  # payload bytes (e.g., serialized Event proto)
        uint32  data_crc      # masked CRC32C of the data payload

    This helper implements a minimal reader suitable for trusted
    TensorBoard event files. It parses the 8-byte length prefix, then skips
    over the 4-byte CRC fields (``length_crc`` and ``data_crc``) instead of
    verifying them, and yields only the raw ``data`` payload for each record.

    Args:
        path: Path to a TFRecord file containing serialized Event messages.

    Yields:
        Bytes objects containing the raw payload of each TFRecord record.

    Raises:
        EOFError: If the file ends in the middle of a record header or payload.
    """
    with open(path, 'rb') as f:
        while True:
            len_bytes = f.read(8)
            if not len_bytes:
                break
            if len(len_bytes) < 8:
                raise EOFError(f'Partial record length in {path}')
            length = struct.unpack('<Q', len_bytes)[0]
            f.read(4)  # len CRC
            data = f.read(length)
            f.read(4)  # data CRC
            if len(data) < length:
                raise EOFError(f'Partial record payload in {path}')
            yield data

def _decode_tensor_scalar(t) -> float | None:
    if t.double_val:
        return float(t.double_val[0])
    if t.float_val:
        return float(t.float_val[0])
    if t.int64_val:
        return float(t.int64_val[0])
    if t.int_val:
        return float(t.int_val[0])
    if t.bool_val:
        return float(bool(t.bool_val[0]))
    if t.half_val:
        hv = int(t.half_val[0]) & 0xFFFF
        if t.dtype == 19:  # DT_HALF
            return np.array([hv], dtype=np.uint16).view(np.float16)[0].item()
        if t.dtype == 14:  # DT_BFLOAT16
            return np.array([hv << 16], dtype=np.uint32).view(np.float32)[0].item()
        return float(hv)
    if t.tensor_content:
        b = t.tensor_content
        if t.dtype == 1 and len(b) >= 4:   # DT_FLOAT
            return struct.unpack('<f', b[:4])[0]
        if t.dtype == 2 and len(b) >= 8:   # DT_DOUBLE
            return struct.unpack('<d', b[:8])[0]
        if t.dtype == 19 and len(b) >= 2:  # DT_HALF
            return np.frombuffer(b[:2], dtype=np.float16)[0].item()
        if t.dtype == 14 and len(b) >= 2:  # DT_BFLOAT16
            u16 = np.frombuffer(b[:2], dtype=np.uint16)[0]
            return np.array([int(u16) << 16], dtype=np.uint32).view(np.float32)[0].item()
    return None

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def find_event_files() -> list[dict]:
    files = sorted(glob.glob(str(CONVDIR / 'events.out.tfevents.*')))
    seen = set()
    runs = []
    for f in files:
        p = Path(f)
        if str(p.resolve()) in seen:
            continue
        seen.add(str(p.resolve()))
        name = p.name.lower()
        is_eager = 'eager' in name
        is_fused = 'fused' in name
        if not (is_eager or is_fused):
            continue
        seed = None
        for part in name.replace('.', '_').split('_'):
            if part.startswith('seed'):
                seed = part.replace('seed', '').replace('sft', '').split('.')[0]
                break
        runs.append({'path': p, 'name': p.name, 'mode': 'fused' if is_fused else 'eager', 'seed': seed})
    return runs

def load_with_tensorboard(path: Path) -> dict:
    ea = EventAccumulator(str(path))
    ea.Reload()
    result = {'train_loss': {}, 'eval_loss': {}, 'grad_norm': {}, 'wall_times': {}}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        if tag == 'train/loss':
            for e in events:
                result['train_loss'][e.step] = e.value
                result['wall_times'][e.step] = e.wall_time
        elif tag == 'eval/loss':
            for e in events:
                result['eval_loss'][e.step] = e.value
        elif tag == 'train/grad_norm':
            for e in events:
                result['grad_norm'][e.step] = e.value
    return result

def load_with_proto(path: Path) -> dict:
    result = {'train_loss': {}, 'eval_loss': {}, 'grad_norm': {}, 'wall_times': {}}
    for data in _tfrecord_records(path):
        ev: Any = EventMessage()
        try:
            ev.ParseFromString(data)
        except Exception:
            continue
        if not ev.summary.value:
            continue
        for v in ev.summary.value:
            tag = v.tag
            value = _decode_tensor_scalar(v.tensor)
            if value is None:
                value = float(v.simple_value)
            if tag == 'train/loss':
                result['train_loss'][int(ev.step)] = float(value)
                result['wall_times'][int(ev.step)] = float(ev.wall_time)
            elif tag == 'eval/loss':
                result['eval_loss'][int(ev.step)] = float(value)
            elif tag == 'train/grad_norm':
                result['grad_norm'][int(ev.step)] = float(value)
    return result

def load_events(path: Path) -> dict:
    if HAS_TB:
        return load_with_tensorboard(path)
    return load_with_proto(path)

def smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode='valid')

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def generate(write_png: bool) -> None:
    runs = find_event_files()
    if not runs:
        raise FileNotFoundError(f'No TensorBoard event files found in {CONVDIR}.')

    eager_runs = {r['seed']: r for r in runs if r['mode'] == 'eager'}
    fused_runs = {r['seed']: r for r in runs if r['mode'] == 'fused'}
    common = sorted(set(eager_runs) & set(fused_runs))
    if not common:
        raise RuntimeError('No matching eager/fused seed pair found.')
    seed = '3' if '3' in common else common[0]

    eager = load_events(eager_runs[seed]['path'])
    fused = load_events(fused_runs[seed]['path'])

    e_steps_all = np.array(sorted(eager['train_loss']))
    f_steps_all = np.array(sorted(fused['train_loss']))
    max_common = int(min(e_steps_all[-1], f_steps_all[-1]))

    e_steps = np.array([s for s in e_steps_all if s <= max_common])
    f_steps = np.array([s for s in f_steps_all if s <= max_common])
    e_loss = np.array([eager['train_loss'][int(s)] for s in e_steps])
    f_loss = np.array([fused['train_loss'][int(s)] for s in f_steps])

    e_grad_steps = np.array([s for s in e_steps if int(s) in eager['grad_norm']])
    f_grad_steps = np.array([s for s in f_steps if int(s) in fused['grad_norm']])
    e_grad = np.array([eager['grad_norm'][int(s)] for s in e_grad_steps])
    f_grad = np.array([fused['grad_norm'][int(s)] for s in f_grad_steps])

    e_eval_steps = np.array(sorted(s for s in eager['eval_loss'] if s <= max_common))
    f_eval_steps = np.array(sorted(s for s in fused['eval_loss'] if s <= max_common))
    e_eval_loss = np.array([eager['eval_loss'][int(s)] for s in e_eval_steps])
    f_eval_loss = np.array([fused['eval_loss'][int(s)] for s in f_eval_steps])

    common_steps = sorted(set(e_steps) & set(f_steps))
    delta_vals = np.array([abs(eager['train_loss'][int(s)] - fused['train_loss'][int(s)]) for s in common_steps])

    e_loss_s = smooth(e_loss, 25)
    f_loss_s = smooth(f_loss, 25)
    e_grad_s = smooth(e_grad, 25)
    f_grad_s = smooth(f_grad, 25)

    e_wall = eager['wall_times']
    f_wall = fused['wall_times']
    eager_min = (e_wall[int(e_steps[-1])] - e_wall[int(e_steps[0])]) / 60.0
    fused_min = (f_wall[int(f_steps[-1])] - f_wall[int(f_steps[0])]) / 60.0
    wall_reduction = 100.0 * (eager_min - fused_min) / eager_min

    fig, axes = plt.subplots(1, 3, figsize=(10.35, 3.7), constrained_layout=False)

    # (a) training loss
    ax = axes[0]
    ax.plot(e_steps, e_loss, color=C_EAGER, alpha=0.10, linewidth=0.45)
    ax.plot(f_steps, f_loss, color=C_FUSED, alpha=0.10, linewidth=0.45)
    ax.plot(e_steps, e_loss_s, color=C_EAGER, linewidth=1.55, label='Eager')
    ax.plot(f_steps, f_loss_s, color=C_FUSED, linewidth=1.55, linestyle='--', label='Fused')
    ax.set_title('(a) Training loss')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.margins(x=0.02)

    # (b) eval loss
    ax = axes[1]
    ax.plot(e_eval_steps, e_eval_loss, marker='o', color=C_EAGER, linewidth=1.45, markersize=4.2, label='Eager')
    ax.plot(f_eval_steps, f_eval_loss, marker='s', color=C_FUSED, linewidth=1.45, linestyle='--', markersize=4.2, label='Fused')
    ax.set_title('(b) Eval loss')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.margins(x=0.03)

    # (c) grad norm
    ax = axes[2]
    ax.plot(e_grad_steps, e_grad, color=C_EAGER, alpha=0.10, linewidth=0.45)
    ax.plot(f_grad_steps, f_grad, color=C_FUSED, alpha=0.10, linewidth=0.45)
    ax.plot(e_grad_steps, e_grad_s, color=C_EAGER, linewidth=1.55, label='Eager')
    ax.plot(f_grad_steps, f_grad_s, color=C_FUSED, linewidth=1.55, linestyle='--', label='Fused')
    ax.set_title('(c) Gradient norm')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Norm')
    ax.margins(x=0.02)

    # Shared legend + compact header / stats
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.80, wspace=0.32)
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.925))
    fig.text(0.015, 0.975, 'Qwen3.5-9B · DoRA r=384 · bf16 · 1×RTX 6000 PRO · seed 3',
             ha='left', va='top', fontsize=8.8)
    fig.text(0.985, 0.975,
             f'max |Δloss|={delta_vals.max():.4f} · mean |Δloss|={delta_vals.mean():.4g} · wall-clock −{wall_reduction:.1f}%',
             ha='right', va='top', fontsize=8.6)

    fig.savefig(FIGDIR / 'training_convergence.pdf')
    if write_png:
        fig.savefig(FIGDIR / 'training_convergence.png', dpi=300)
    plt.close(fig)
    fmts = 'pdf/png' if write_png else 'pdf'
    print(f'  wrote training_convergence.{fmts}')

# ---------------------------------------------------------------------------
def main() -> int:
    write_png = '--png' in sys.argv[1:]
    generate(write_png)
    print(f'Output directory: {FIGDIR}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
