#!/usr/bin/env python3
"""
GPU-agnostic analysis of Triton autotune results for fused DoRA kernels.
Reads all JSON files produced by extract_autotune.py from the local autotune_results/
directory and provides analysis on optimal configurations, speedups, and hardware patterns.

Produces formatted tables covering:
  - Data overview and winning config distributions
  - RPP, warps, stages analysis
  - Autotune impact (best vs worst)
  - Latency bounds and performance cliffs
  - Validation of generalized paper claims
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from numbers import Integral
from pathlib import Path

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument('--json', '-j', action='store_true', help='Token-dense JSON output (suppresses tables)')
ARGS = _parser.parse_args()
OUTPUT = {}
_orig_stdout = sys.stdout
if ARGS.json:
    sys.stdout = io.StringIO()

# ── Configuration ──────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "autotune_results"

KERNEL_SHORT = {
    "_fused_dora_backward_kernel": "backward",
    "_fused_dora_backward_kernel_impl": "backward",
    "_fused_dora_forward_and_inner_kernel": "forward",
    "_fused_dora_forward_and_inner_kernel_impl": "forward",
    "_fused_norm_assembly_kernel_impl": "norm_assembly",
    "_fused_norm_assembly_with_div_kernel_impl": "norm_assembly_div",
}

KERNEL_NAMES = [
    "_fused_dora_backward_kernel",
    "_fused_dora_backward_kernel_impl",
    "_fused_dora_forward_and_inner_kernel",
    "_fused_dora_forward_and_inner_kernel_impl",
    "_fused_norm_assembly_kernel_impl",
    "_fused_norm_assembly_with_div_kernel_impl",
]


def load_all():
    """Load all GPU autotune files from the results directory."""
    data = {}
    if not RESULTS_DIR.exists():
        sys.stderr.write(f"Error: {RESULTS_DIR} not found. Run extract_autotune.py first.\n")
        sys.exit(1)

    for path in RESULTS_DIR.glob("*.json"):
        gpu_name = path.stem
        try:
            with open(path) as f:
                data[gpu_name] = json.load(f)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Warning: Failed to parse {path.name}: {e}\n")
    return data


def config_key(cfg):
    """Create a hashable config tuple for comparison (excluding timing)."""
    bs = cfg.get("BLOCK_SIZE", "?")
    rpp = cfg.get("ROWS_PER_PROGRAM", None)
    warps = cfg.get("num_warps", "?")
    stages = cfg.get("num_stages", "?")
    if rpp is not None:
        return (bs, rpp, warps, stages)
    return (bs, warps, stages)


def config_str(cfg):
    """Human-readable config string."""
    bs = cfg.get("BLOCK_SIZE", "?")
    rpp = cfg.get("ROWS_PER_PROGRAM", None)
    warps = cfg.get("num_warps", "?")
    stages = cfg.get("num_stages", "?")
    if rpp is not None:
        return f"BS={bs},RPP={rpp},W={warps},S={stages}"
    return f"BS={bs},W={warps},S={stages}"


def _input_key(entry):
    return entry.get("input_key", [])


def extract_numeric_prefix(entry):
    """Extract leading integer dimensions from Triton's autotune key."""
    dims = []
    for item in _input_key(entry):
        if isinstance(item, bool):
            break
        if isinstance(item, Integral):
            dims.append(int(item))
        else:
            break
    return tuple(dims)


def extract_dim(entry):
    """Extract the primary width dimension (first numeric autotune key field)."""
    dims = extract_numeric_prefix(entry)
    return dims[0] if dims else "unknown"


def extract_num_rows_bucket(entry):
    """Extract the row bucket when present in the autotune key."""
    dims = extract_numeric_prefix(entry)
    return dims[1] if len(dims) > 1 else None


def extract_dtype_signature(entry):
    """Extract the full dtype signature from the autotune key.

    Triton autotune keys can include multiple tensor dtypes. The old analyzer
    only reported the first ``torch.*`` token, which collapsed mixed-dtype
    forward keys such as ``(float32, bfloat16, float32, ...)`` into a single
    misleading ``float32`` label.
    """
    decoded = entry.get("input_key_decoded", {})
    dtypes = decoded.get("dtypes")
    if dtypes:
        return tuple(str(dt) for dt in dtypes)

    legacy = []
    for item in _input_key(entry):
        if isinstance(item, str) and item.startswith("torch."):
            legacy.append(item.replace("torch.", ""))
    return tuple(legacy) if legacy else ("unknown",)


_DTYPE_ABBREV = {
    "float64": "f64",
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "int64": "i64",
    "int32": "i32",
    "int16": "i16",
    "int8": "i8",
    "bool": "bool",
}


def dtype_signature_str(dtype_sig):
    """Compact human-readable dtype signature label."""
    if not dtype_sig:
        return "unknown"
    if len(set(dtype_sig)) == 1:
        return dtype_sig[0]
    return "[" + ",".join(_DTYPE_ABBREV.get(dt, dt) for dt in dtype_sig) + "]"


def shape_label(entry):
    dim = extract_dim(entry)
    rows_bucket = extract_num_rows_bucket(entry)
    if rows_bucket is None:
        return f"dim={dim}"
    return f"cols={dim},rows_bkt={rows_bucket}"


def build_best_table(data, gpus):
    """Build a table of best configs: (kernel, shape_key, dtype_signature) -> {gpu: best_config}."""
    table = defaultdict(dict)
    for gpu in gpus:
        for kernel_name, entries in data[gpu].items():
            short = KERNEL_SHORT.get(kernel_name, kernel_name)
            for entry in entries:
                dims = extract_numeric_prefix(entry)
                dtype_sig = extract_dtype_signature(entry)
                table[(short, dims, dtype_sig)][gpu] = entry["best"]
    return table


def section_header(num, title):
    print(f"\n{'=' * 100}")
    print(f"  {num}. {title}")
    print(f"{'=' * 100}\n")


# ── Analysis Functions ─────────────────────────────────────────────────────

def analyze_overview(data, gpus):
    """High-level overview of entries per GPU per kernel."""
    section_header(1, "DATA OVERVIEW")

    for gpu in gpus:
        total_entries = sum(len(entries) for entries in data[gpu].values())
        total_kernels = len(data[gpu])
        print(f"  {gpu}: {total_entries} entries across {total_kernels} kernels")

    print(f"\n{'GPU':<15} {'Kernel':<35} {'Entries':>8} {'Candidates/Entry':>18} {'Rows Bucketed':>15}")
    print("-" * 98)
    for gpu in gpus:
        for kernel_name, entries in data[gpu].items():
            short = KERNEL_SHORT.get(kernel_name, kernel_name)
            n_cands = len(entries[0]["candidates"]) if entries else 0
            has_rows_bucket = any(extract_num_rows_bucket(e) is not None for e in entries)
            print(f"{gpu:<15} {short:<35} {len(entries):>8} {n_cands:>18} {str(has_rows_bucket):>15}")
        print()


def analyze_best_configs(data, gpus):
    """For each kernel, show the distribution of winning BLOCK_SIZE values."""
    section_header(2, "WINNING BLOCK_SIZE DISTRIBUTION")

    found_kernels = {k for gpu in gpus for k in data[gpu].keys()}

    for kernel_name in sorted(found_kernels):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                print(f"  {gpu}: (no data)")
                continue
            entries = data[gpu][kernel_name]
            bs_counts = defaultdict(int)
            for e in entries:
                bs_counts[e["best"]["BLOCK_SIZE"]] += 1
            sorted_bs = sorted(bs_counts.items(), key=lambda x: -x[1])
            dist_str = ", ".join(f"BS={bs}: {c}" for bs, c in sorted_bs)
            avg_bs = sum(e["best"]["BLOCK_SIZE"] for e in entries) / len(entries) if entries else 0
            print(f"  {gpu:<15} n={len(entries):>3}  avg_BS={avg_bs:.0f}  {dist_str}")


def analyze_rpp(data, gpus):
    """Analyze ROWS_PER_PROGRAM selection."""
    section_header(3, "ROWS_PER_PROGRAM ANALYSIS")

    rpp_kernels = [k for k in data.get(gpus[0], {}).keys() if "dora" in k]
    if not rpp_kernels:
        print("  No DoRA kernels found with RPP to analyze.")
        return

    for kernel_name in rpp_kernels:
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]
            rpp_counts = defaultdict(int)
            for e in entries:
                rpp = e["best"].get("ROWS_PER_PROGRAM", "N/A")
                rpp_counts[rpp] += 1
            dist_str = ", ".join(f"RPP={r}: {c}" for r, c in sorted(rpp_counts.items(), key=lambda x: str(x[0])))
            unanimous = len(rpp_counts) == 1
            tag = " [UNANIMOUS]" if unanimous else ""
            print(f"  {gpu:<15} {dist_str}{tag}")


def analyze_warps_stages(data, gpus):
    """Analyze num_warps and num_stages selection patterns."""
    section_header(4, "NUM_WARPS & NUM_STAGES ANALYSIS")

    found_kernels = {k for gpu in gpus for k in data[gpu].keys()}

    for kernel_name in sorted(found_kernels):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]
            total = len(entries)

            warp_counts = defaultdict(int)
            stage_counts = defaultdict(int)
            for e in entries:
                warp_counts[e["best"]["num_warps"]] += 1
                stage_counts[e["best"]["num_stages"]] += 1

            w_str = ", ".join(f"W={w}: {c}" for w, c in sorted(warp_counts.items()))
            s_str = ", ".join(f"S={s}: {c} ({100*c/total:.0f}%)" for s, c in sorted(stage_counts.items()))

            print(f"  {gpu:<15} n={total:>3}")
            print(f"    Warps:  {w_str}")
            print(f"    Stages: {s_str}")


def analyze_cross_device_agreement(data, gpus):
    """How often do devices agree on the best config for the same (kernel, dim, dtype)?"""
    if len(gpus) < 2:
        return

    section_header(5, "CROSS-DEVICE CONFIG AGREEMENT")

    best_table = build_best_table(data, gpus)
    kernel_groups = defaultdict(list)
    for (kernel, dims, dtype_sig), gpu_configs in best_table.items():
        kernel_groups[kernel].append((dims, dtype_sig, gpu_configs))

    overall_agree = 0
    overall_total = 0

    for kernel, entries in kernel_groups.items():
        print(f"\n--- {kernel} ---")
        total_pairs = 0
        agree_pairs = 0
        pair_agree = defaultdict(lambda: [0, 0])

        for dims, dtype_sig, gpu_configs in entries:
            avail_gpus = sorted(gpu_configs.keys())
            for i in range(len(avail_gpus)):
                for j in range(i + 1, len(avail_gpus)):
                    g1, g2 = avail_gpus[i], avail_gpus[j]
                    ck1 = config_key(gpu_configs[g1])
                    ck2 = config_key(gpu_configs[g2])
                    total_pairs += 1
                    pair_agree[(g1, g2)][1] += 1
                    if ck1 == ck2:
                        agree_pairs += 1
                        pair_agree[(g1, g2)][0] += 1

        if total_pairs > 0:
            pct = 100 * agree_pairs / total_pairs
            print(f"  Overall agreement: {agree_pairs}/{total_pairs} = {pct:.1f}%")
            overall_agree += agree_pairs
            overall_total += total_pairs
        for (g1, g2), (a, t) in sorted(pair_agree.items()):
            print(f"    {g1:>15} vs {g2:<15}: {a}/{t} = {100*a/t:.1f}%")

    if overall_total > 0:
        print(f"\n  GLOBAL agreement: {overall_agree}/{overall_total} = {100*overall_agree/overall_total:.1f}%")


def analyze_autotune_impact(data, gpus):
    """How much does autotuning matter? Best vs worst config speedup."""
    section_header(6, "AUTOTUNE IMPACT: BEST vs WORST CONFIG SPEEDUP")

    found_kernels = {k for gpu in gpus for k in data[gpu].keys()}

    for kernel_name in sorted(found_kernels):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        print(f"  {'GPU':<15} {'Shape':>18} {'Dtype Signature':<28} {'Best ms':>10} {'Worst ms':>10} {'Slowdown':>10} {'2nd-best ms':>12} {'Top gap':>10}")
        print(f"  {'-'*102}")

        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]
            for entry in entries:
                dtype = dtype_signature_str(extract_dtype_signature(entry))
                best_ms = entry["best"]["median_ms"]
                candidates = entry["candidates"]
                worst_ms = max(c["median_ms"] for c in candidates)
                sorted_cands = sorted(candidates, key=lambda c: c["median_ms"])
                second_ms = sorted_cands[1]["median_ms"] if len(sorted_cands) > 1 else best_ms
                slowdown = worst_ms / best_ms if best_ms > 0 else 0
                top_gap = (second_ms - best_ms) / best_ms * 100 if best_ms > 0 else 0
                print(f"  {gpu:<15} {shape_label(entry):>18} {dtype:<28} {best_ms:>10.6f} {worst_ms:>10.6f} {slowdown:>9.1f}x {second_ms:>12.6f} {top_gap:>9.1f}%")


def analyze_norm_latency_floor(data, gpus):
    """Identify launch-latency-bound entries (all configs same timing)."""
    section_header(7, "LAUNCH LATENCY FLOOR (NORM KERNELS)")

    norm_kernels = [k for k in KERNEL_NAMES if "norm" in k]
    for kernel_name in norm_kernels:
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]
            floor_count = 0
            total = len(entries)
            for entry in entries:
                candidates = entry["candidates"]
                medians = [c["median_ms"] for c in candidates]
                spread = (max(medians) - min(medians)) / min(medians) * 100 if min(medians) > 0 else 0
                if spread < 5.0:
                    floor_count += 1
            print(f"  {gpu:<15} {floor_count}/{total} entries at launch-latency floor (<5% spread across all configs)")


def analyze_bs16k_cliff(data, gpus):
    """Analyze performance cliffs between BLOCK_SIZE 8192 and 16384."""
    section_header(8, "BS=16384 CLIFF ANALYSIS (BACKWARD KERNEL)")

    bwd_kernels = [k for k in data.get(gpus[0], {}).keys() if "backward" in k]
    if not bwd_kernels:
        print("  No backward kernels found to analyze.")
        return

    for kernel_name in bwd_kernels:
        print(f"\n--- {KERNEL_SHORT.get(kernel_name, kernel_name)} ---")
        print(f"  {'GPU':<15} {'Shape':>18} {'Dtype Signature':<28} {'BS=8192 ms':>12} {'BS=16384 ms':>13} {'Slowdown':>10}")
        print(f"  {'-'*89}")

        cliff_count = 0
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]
            for entry in entries:
                dtype = dtype_signature_str(extract_dtype_signature(entry))
                candidates = entry["candidates"]

                bs8k = [c for c in candidates if c["BLOCK_SIZE"] == 8192]
                bs16k = [c for c in candidates if c["BLOCK_SIZE"] == 16384]

                if not bs8k or not bs16k:
                    continue

                best_8k = min(c["median_ms"] for c in bs8k)
                best_16k = min(c["median_ms"] for c in bs16k)

                if best_8k > 0:
                    slowdown = best_16k / best_8k
                    if slowdown > 2.0:
                        cliff_count += 1
                        print(f"  {gpu:<15} {shape_label(entry):>18} {dtype:<28} {best_8k:>12.6f} {best_16k:>13.6f} {slowdown:>9.1f}x")

        if cliff_count == 0:
            print("  No significant cliffs detected (all < 2.0x).")


def analyze_dtype_sensitivity(data, gpus):
    """How does the full dtype signature affect optimal config selection?"""
    section_header(9, "DTYPE SENSITIVITY ANALYSIS")

    best_table = build_best_table(data, gpus)

    found_kernels = {k[0] for k in best_table.keys() if "dora" in k[0] or "forward" in k[0] or "backward" in k[0]}

    for kernel in sorted(found_kernels):
        print(f"\n--- {kernel} ---")
        dim_gpu_groups = defaultdict(dict)
        for (k, dims, dtype_sig), gpu_configs in best_table.items():
            if k != kernel:
                continue
            for gpu, cfg in gpu_configs.items():
                dim_gpu_groups[(dims, gpu)][dtype_sig] = cfg

        total = 0
        dtype_signature_changes = 0
        for (dims, gpu), dtype_sig_cfgs in sorted(dim_gpu_groups.items()):
            if len(dtype_sig_cfgs) < 2:
                continue
            configs = [config_key(c) for c in dtype_sig_cfgs.values()]
            total += 1
            if len(set(configs)) > 1:
                dtype_signature_changes += 1
                dtypes_str = " | ".join(
                    f"{dtype_signature_str(dt_sig)}: {config_str(c)}"
                    for dt_sig, c in sorted(dtype_sig_cfgs.items(), key=lambda item: item[0])
                )
                if len(dims) > 1:
                    shape = f"cols={dims[0]},rows_bkt={dims[1]}"
                elif dims:
                    shape = f"dim={dims[0]}"
                else:
                    shape = "shape=unknown"
                print(f"  {gpu:<15} {shape}: {dtypes_str}")

        if total > 0:
            print(
                f"\n  Dtype signature affects config: "
                f"{dtype_signature_changes}/{total} = {100*dtype_signature_changes/total:.1f}% "
                f"of (shape, gpu) pairs"
            )
        else:
            print("  No comparable data across multiple dtype signatures.")


def analyze_dim_scaling(data, gpus):
    """How does best config change with increasing dimension?"""
    section_header(10, "DIMENSION SCALING: BEST BLOCK_SIZE vs INPUT DIM")

    dora_kernels = [k for gpu in gpus for k in data[gpu].keys() if "dora" in k]
    for kernel_name in set(dora_kernels):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]
            dtype_groups = defaultdict(list)
            for e in entries:
                dtype_groups[(extract_dtype_signature(e), extract_num_rows_bucket(e))].append(e)

            for (dtype_sig, rows_bucket), ents in sorted(dtype_groups.items(), key=lambda item: (item[0][0], item[0][1] if item[0][1] is not None else -1)):
                ents_sorted = sorted(ents, key=lambda e: extract_dim(e))
                pairs = [(extract_dim(e), e["best"]["BLOCK_SIZE"]) for e in ents_sorted]
                pair_str = ", ".join(f"{d}->BS{bs}" for d, bs in pairs)
                suffix = f" rows_bkt={rows_bucket}" if rows_bucket is not None else ""
                print(f"  {gpu:<15} {dtype_signature_str(dtype_sig):<28}{suffix}: {pair_str}")


def analyze_invariant_claims(data, gpus):
    """Verify general claims made in the paper about autotune behavior."""
    section_header(11, "PAPER CLAIMS VERIFICATION (INVARIANTS)")

    # Claim: RPP invariance
    print("Claim: RPP=2 for backward, RPP=4 for forward (100% invariant across all GPUs)")
    print()

    bwd_rpp_total = 0
    bwd_rpp_2 = 0
    fwd_rpp_total = 0
    fwd_rpp_4 = 0

    for gpu in gpus:
        for kernel_name, entries in data[gpu].items():
            if "backward" in kernel_name:
                for e in entries:
                    bwd_rpp_total += 1
                    if e["best"].get("ROWS_PER_PROGRAM") == 2:
                        bwd_rpp_2 += 1
            elif "forward" in kernel_name:
                for e in entries:
                    fwd_rpp_total += 1
                    if e["best"].get("ROWS_PER_PROGRAM") == 4:
                        fwd_rpp_4 += 1

    bwd_status = "PASS" if bwd_rpp_2 == bwd_rpp_total and bwd_rpp_total > 0 else "FAIL" if bwd_rpp_total > 0 else "N/A"
    fwd_status = "PASS" if fwd_rpp_4 == fwd_rpp_total and fwd_rpp_total > 0 else "FAIL" if fwd_rpp_total > 0 else "N/A"

    if bwd_rpp_total > 0:
        print(f"  Backward RPP=2: {bwd_rpp_2}/{bwd_rpp_total} [{bwd_status}]")
    if fwd_rpp_total > 0:
        print(f"  Forward  RPP=4: {fwd_rpp_4}/{fwd_rpp_total} [{fwd_status}]")


def analyze_near_optimal_configs(data, gpus):
    """Analyze how many configurations are within 5% of the best."""
    section_header(12, "NEAR-OPTIMAL CONFIGURATION LANDSCAPE (WITHIN 5% OF BEST)")

    found_kernels = {k for gpu in gpus for k in data[gpu].keys()}

    for kernel_name in sorted(found_kernels):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]
            avg_near_optimal = 0
            for entry in entries:
                best_ms = entry["best"]["median_ms"]
                if best_ms == 0:
                    continue
                near_opt_count = sum(1 for c in entry["candidates"] if c["median_ms"] <= best_ms * 1.05)
                avg_near_optimal += near_opt_count

            avg = avg_near_optimal / len(entries) if entries else 0
            n_cands = len(entries[0]["candidates"]) if entries else 0
            print(f"  {gpu:<15} avg {avg:.1f} / {n_cands} configs within 5% of best")


def analyze_non_monotonic_scaling(data, gpus):
    """Find anomalies where larger dimensions process significantly faster than smaller ones."""
    section_header(13, "NON-MONOTONIC SCALING & ANOMALIES (LATENCY DIPS)")

    found_kernels = {k for gpu in gpus for k in data[gpu].keys()}

    for kernel_name in sorted(found_kernels):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]

            # Group by full dtype signature + row bucket
            dtype_groups = defaultdict(list)
            for e in entries:
                dtype_groups[(extract_dtype_signature(e), extract_num_rows_bucket(e))].append(e)

            anomaly_count = 0
            for (dtype_sig, rows_bucket), ents in sorted(dtype_groups.items(), key=lambda item: (item[0][0], item[0][1] if item[0][1] is not None else -1)):
                ents_sorted = sorted(ents, key=lambda e: extract_dim(e))
                for i in range(1, len(ents_sorted)):
                    prev = ents_sorted[i-1]
                    curr = ents_sorted[i]

                    prev_dim, prev_ms = extract_dim(prev), prev["best"]["median_ms"]
                    curr_dim, curr_ms = extract_dim(curr), curr["best"]["median_ms"]

                    # If current dimension is larger but takes < 60% of previous dimension's time
                    if curr_ms < prev_ms * 0.6 and prev_ms > 0:
                        anomaly_count += 1
                        rows_suffix = f" rows_bkt={rows_bucket}" if rows_bucket is not None else ""
                        print(
                            f"  {gpu:<15} {dtype_signature_str(dtype_sig):<28}{rows_suffix}  "
                            f"Anomaly! dim={prev_dim} ({prev_ms:.6f}ms) -> dim={curr_dim} "
                            f"({curr_ms:.6f}ms) [{(curr_ms/prev_ms):.2f}x]"
                        )

            if anomaly_count == 0:
                print(f"  {gpu:<15} No significant non-monotonic scaling anomalies detected.")


def analyze_parameter_interactions(data, gpus):
    """Analyze the relationship between BLOCK_SIZE and ROWS_PER_PROGRAM / num_warps."""
    section_header(14, "PARAMETER INTERACTIONS (BLOCK_SIZE vs RPP & WARPS)")

    for kernel_name in sorted({k for gpu in gpus for k in data[gpu].keys()}):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            entries = data[gpu][kernel_name]

            bs_to_rpp = defaultdict(list)
            bs_to_warps = defaultdict(list)

            for e in entries:
                bs = e["best"]["BLOCK_SIZE"]
                rpp = e["best"].get("ROWS_PER_PROGRAM", None)
                warps = e["best"]["num_warps"]
                if rpp is not None:
                    bs_to_rpp[bs].append(rpp)
                bs_to_warps[bs].append(warps)

            print(f"  {gpu}:")
            for bs in sorted(bs_to_warps.keys()):
                warp_counts = defaultdict(int)
                for w in bs_to_warps[bs]: warp_counts[w] += 1
                warp_str = ", ".join(f"W{w}:{c}" for w, c in sorted(warp_counts.items()))

                rpp_str = ""
                if bs in bs_to_rpp:
                    rpp_counts = defaultdict(int)
                    for r in bs_to_rpp[bs]: rpp_counts[r] += 1
                    rpp_str = " | " + ", ".join(f"RPP{r}:{c}" for r, c in sorted(rpp_counts.items()))

                print(f"    BS={bs:<5} -> {warp_str}{rpp_str}")


def analyze_pruning_opportunities(data, gpus):
    """Identify configs that never win and can be safely pruned from the search space."""
    section_header(15, "SEARCH SPACE PRUNING OPPORTUNITIES")

    found_kernels = {k for gpu in gpus for k in data[gpu].keys()}

    for kernel_name in sorted(found_kernels):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        print(f"\n--- {short} ---")

        all_candidates = set()
        winning_candidates = set()

        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            for entry in data[gpu][kernel_name]:
                for c in entry["candidates"]:
                    all_candidates.add(config_key(c))
                winning_candidates.add(config_key(entry["best"]))

        if not all_candidates:
            continue

        never_won = all_candidates - winning_candidates
        prune_ratio = len(never_won) / len(all_candidates) * 100
        print(f"  Total unique configs evaluated: {len(all_candidates)}")
        print(f"  Configs that never won ANYWHERE: {len(never_won)} ({prune_ratio:.1f}% space reduction possible)")

        if never_won:
            bs_counts = defaultdict(int)
            for c in never_won:
                bs_counts[c[0]] += 1

            sorted_bs = sorted(bs_counts.items(), key=lambda x: -x[1])
            dist_str = ", ".join(f"BS={bs}: {c}" for bs, c in sorted_bs[:8])
            print(f"  Never-winning BS distribution: {dist_str}")


def analyze_summary(data, gpus):
    """Final summary with key takeaways."""
    section_header(16, "SUMMARY & KEY TAKEAWAYS")

    print("Key findings:\n")

    # RPP summary
    bwd_entries = sum(1 for gpu in gpus for k in data[gpu].keys() if "backward" in k for _ in data[gpu][k])
    fwd_entries = sum(1 for gpu in gpus for k in data[gpu].keys() if "forward" in k for _ in data[gpu][k])
    total_compose = bwd_entries + fwd_entries
    if total_compose > 0:
        print(f"  1. RPP INVARIANTS: Total {total_compose} DoRA entries analyzed for ROWS_PER_PROGRAM constraints.")
        print()

    bucketed_entries = sum(
        1
        for gpu in gpus
        for kernel_name in data[gpu]
        if "dora" in kernel_name
        for entry in data[gpu][kernel_name]
        if extract_num_rows_bucket(entry) is not None
    )
    if bucketed_entries > 0:
        print(f"  1b. ROW BUCKETS: {bucketed_entries} DoRA entries include num_rows_bucket in the autotune key.")
        print()

    dtype_signature_count = len(
        {
            extract_dtype_signature(entry)
            for gpu in gpus
            for kernel_name in data[gpu]
            for entry in data[gpu][kernel_name]
        }
    )
    if dtype_signature_count > 0:
        print(f"  1c. DTYPE SIGNATURES: observed {dtype_signature_count} distinct autotune dtype signatures.")
        print()

    # Autotune impact
    print(f"  2. AUTOTUNE IMPACT: choosing wrong config costs up to:")
    found_kernels = {k for gpu in gpus for k in data[gpu].keys()}
    for kernel_name in sorted(found_kernels):
        short = KERNEL_SHORT.get(kernel_name, kernel_name)
        max_slowdown = 0
        max_info = ""
        for gpu in gpus:
            if kernel_name not in data[gpu]:
                continue
            for entry in data[gpu][kernel_name]:
                best_ms = entry["best"]["median_ms"]
                worst_ms = max(c["median_ms"] for c in entry["candidates"])
                slowdown = worst_ms / best_ms if best_ms > 0 else 0
                if slowdown > max_slowdown:
                    max_slowdown = slowdown
                    max_info = f"{gpu} {shape_label(entry)}"
        print(f"     {short}: {max_slowdown:.1f}x ({max_info})")


def main():
    data = load_all()
    if not data:
        sys.stderr.write("No autotune data loaded. Exiting.\n")
        sys.exit(1)

    gpus = sorted(data.keys())

    print("=" * 100)
    print(f"LOCAL AUTOTUNE ANALYSIS: {len(gpus)} GPU(s) Loaded")
    print(f"Found GPUs: {', '.join(gpus)}")
    print("=" * 100)

    analyze_overview(data, gpus)
    analyze_best_configs(data, gpus)
    analyze_rpp(data, gpus)
    analyze_warps_stages(data, gpus)
    if len(gpus) > 1:
        analyze_cross_device_agreement(data, gpus)
    analyze_autotune_impact(data, gpus)
    analyze_norm_latency_floor(data, gpus)
    analyze_bs16k_cliff(data, gpus)
    analyze_dtype_sensitivity(data, gpus)
    analyze_dim_scaling(data, gpus)
    analyze_invariant_claims(data, gpus)
    analyze_near_optimal_configs(data, gpus)
    analyze_non_monotonic_scaling(data, gpus)
    analyze_parameter_interactions(data, gpus)
    analyze_pruning_opportunities(data, gpus)
    analyze_summary(data, gpus)

    # ── JSON output ───────────────────────────────────────────────────
    if ARGS.json:
        best_table = build_best_table(data, gpus)
        found_kernels = {k for gpu in gpus for k in data[gpu].keys()}

        # RPP invariance
        rpp_results = {}
        for kernel_name in found_kernels:
            if "dora" not in kernel_name:
                continue
            short = KERNEL_SHORT.get(kernel_name, kernel_name)
            per_gpu = {}
            for gpu in gpus:
                if kernel_name not in data[gpu]:
                    continue
                entries = data[gpu][kernel_name]
                counts = defaultdict(int)
                for e in entries:
                    counts[e["best"].get("ROWS_PER_PROGRAM", "N/A")] += 1
                per_gpu[gpu] = dict(counts)
            rpp_results[short] = per_gpu
        OUTPUT["rpp"] = rpp_results

        row_bucket_summary = {}
        for kernel_name in found_kernels:
            short = KERNEL_SHORT.get(kernel_name, kernel_name)
            per_gpu = {}
            for gpu in gpus:
                if kernel_name not in data[gpu]:
                    continue
                entries = data[gpu][kernel_name]
                counts = defaultdict(int)
                for e in entries:
                    rows_bucket = extract_num_rows_bucket(e)
                    if rows_bucket is not None:
                        counts[str(rows_bucket)] += 1
                if counts:
                    per_gpu[gpu] = dict(counts)
            if per_gpu:
                row_bucket_summary[short] = per_gpu
        OUTPUT["row_buckets"] = row_bucket_summary

        dtype_signature_summary = {}
        for kernel_name in found_kernels:
            short = KERNEL_SHORT.get(kernel_name, kernel_name)
            per_gpu = {}
            for gpu in gpus:
                if kernel_name not in data[gpu]:
                    continue
                entries = data[gpu][kernel_name]
                counts = defaultdict(int)
                for e in entries:
                    counts[dtype_signature_str(extract_dtype_signature(e))] += 1
                per_gpu[gpu] = dict(sorted(counts.items()))
            dtype_signature_summary[short] = per_gpu
        OUTPUT["dtype_signatures"] = dtype_signature_summary

        # Cross-architecture agreement
        if len(gpus) > 1:
            total_pairs = 0
            agree_pairs = 0
            pair_detail = defaultdict(lambda: [0, 0])
            for (kernel, dim, dtype_sig), gpu_configs in best_table.items():
                avail_gpus = sorted(gpu_configs.keys())
                for i in range(len(avail_gpus)):
                    for j in range(i + 1, len(avail_gpus)):
                        g1, g2 = avail_gpus[i], avail_gpus[j]
                        total_pairs += 1
                        pair_detail[f"{g1}_vs_{g2}"][1] += 1
                        if config_key(gpu_configs[g1]) == config_key(gpu_configs[g2]):
                            agree_pairs += 1
                            pair_detail[f"{g1}_vs_{g2}"][0] += 1
            OUTPUT["cross_arch_agreement"] = {
                "global_pct": round(100 * agree_pairs / total_pairs, 2) if total_pairs else 0,
                "pairs": {k: {"agree": v[0], "total": v[1], "pct": round(100 * v[0] / v[1], 2)} for k, v in pair_detail.items()},
            }

        # Block size distributions
        bs_dist = {}
        for kernel_name in found_kernels:
            short = KERNEL_SHORT.get(kernel_name, kernel_name)
            per_gpu = {}
            for gpu in gpus:
                if kernel_name not in data[gpu]:
                    continue
                entries = data[gpu][kernel_name]
                counts = defaultdict(int)
                for e in entries:
                    counts[str(e["best"]["BLOCK_SIZE"])] += 1
                avg = sum(e["best"]["BLOCK_SIZE"] for e in entries) / len(entries) if entries else 0
                per_gpu[gpu] = {"avg": round(avg), "dist": dict(counts)}
            bs_dist[short] = per_gpu
        OUTPUT["block_size"] = bs_dist

        # Stages distribution
        stages_dist = {}
        for kernel_name in found_kernels:
            short = KERNEL_SHORT.get(kernel_name, kernel_name)
            per_gpu = {}
            for gpu in gpus:
                if kernel_name not in data[gpu]:
                    continue
                entries = data[gpu][kernel_name]
                counts = defaultdict(int)
                for e in entries:
                    counts[str(e["best"]["num_stages"])] += 1
                per_gpu[gpu] = {f"S{k}": v for k, v in counts.items()}
            stages_dist[short] = per_gpu
        OUTPUT["stages"] = stages_dist

        # Autotune impact (worst-case per kernel per GPU)
        impact = {}
        for kernel_name in found_kernels:
            short = KERNEL_SHORT.get(kernel_name, kernel_name)
            per_gpu = {}
            for gpu in gpus:
                if kernel_name not in data[gpu]:
                    continue
                worst = 0
                for entry in data[gpu][kernel_name]:
                    best_ms = entry["best"]["median_ms"]
                    worst_ms = max(c["median_ms"] for c in entry["candidates"])
                    s = worst_ms / best_ms if best_ms > 0 else 0
                    worst = max(worst, s)
                per_gpu[gpu] = round(worst, 2)
            impact[short] = per_gpu
        OUTPUT["autotune_max_slowdown"] = impact

        sys.stdout = _orig_stdout
        print(json.dumps(OUTPUT, separators=(',', ':')))


if __name__ == "__main__":
    main()

