# Data Provenance & Figure Mapping

This document maps every benchmark data artifact to its origin, purpose, and usage in the paper.

## Software Stack

**Benchmarks + convergence (single pinned stack):**
- **PyTorch**: 2.10.0+cu130 (built against CUDA 13.0)
- **Triton**: 3.6.0
- **Transformers**: 5.2.0
- **CUDA toolkit**: 13.1 (ptxas V13.1.115)
- **NVIDIA Driver**: 580.126.09
- **Python**: 3.12.12
- **OS**: Linux 6.8.0 (Ubuntu 22.04, glibc 2.35)

**Convergence runs (additional dependencies):**
- **ms-swift**: commit `a807cb9`
- **DeepSpeed**: 0.18.6
- **Flash-Attention**: 2.8.3

---

## Source Code Files

| File | Description |
|------|-------------|
| `dora.py` | Our DoRA layer implementation: factored norm, fused dispatch, `DoraLinearLayer` / `DoraEmbeddingLayer` / `_DoraConvNdLayer`. |
| `dora_fused.py` | Fused Triton kernels: compose forward/backward, norm assembly, `FusedDoRAComposeFunction` autograd wrapper. |
| `dora_diagnostics.py` | Diagnostic instrumentation gated by `PEFT_DORA_DIAGNOSE=1`; zero-cost no-op when disabled. |
| `dora_ci.py` | Modal CI entrypoint: orchestrates pytest + benchmarks on remote GPUs. |
| `bench_dora_comprehensive.py` | Comprehensive benchmark suite: norm, compose, backward, e2e, memory, stability, models. Produces JSON consumed by `generate_figures.py`. |
| `test_dora_fused.py` | Regression and performance tests for fused kernels (749 tests total with `test_dora_math.py`). |
| `test_dora_math.py` | Mathematical correctness tests: factored norm equivalence, numerical stability, edge cases. |
| `hf.patch` | Git diff against upstream PEFT `20a9829` (`v0.18.0.rc0`). Symlinked at repo root. |
| `peft_patched/` | Patched PEFT source tree (apply `hf.patch` to upstream for equivalent). |
| `scripts/check_compose_parity.py` | Quick compose parity check at production-scale dimensions. |
| `scripts/dora.reference_hf_peft.py` | Unmodified HF PEFT `DoraLinearLayer` snapshot for baseline comparisons. |
| `scripts/dora_inference_audit.py` | Mechanistic 6-phase DoRA inference audit (norm, compose, backward, decode, dispatch). |
| `scripts/repack_mmfinereason_qr.py` | Dataset preprocessing: field renames + tok_len filtering for MMFineReason. |
| `scripts/run_revision_benchmarks.sh` | High-rank + loss_tokens sensitivity benchmark runner. |
| `kernelagent_sols/` | KernelAgent (Meta) optimized kernel artifacts for compose and backward. |

---

## Benchmark Data (bench_it6)

**Directory**: `code/bench_it6/`

All paper figures, tables, and claims derive from this single data collection.

### Microbenchmarks (6 GPUs, 200 repeats, extended shapes)

| GPU | SM | Memory | Files |
|-----|----|--------|-------|
| L40S | SM89 (Ada) | 48 GB GDDR6 | `sm89_l40s_comprehensive_extended_{bf16,fp16,fp32}.json` |
| A100 | SM80 (Ampere) | 80 GB HBM2e | `sm80_a100_comprehensive_extended_{bf16,fp16,fp32}.json` |
| RTX 6000 PRO | SM120 (Blackwell) | 96 GB GDDR7 | `sm120_rtx6000_comprehensive_extended_{bf16,fp16,fp32}.json` |
| H200 | SM90 (Hopper) | 141 GB HBM3e | `sm90_h200_comprehensive_extended_{bf16,fp16,fp32}.json` |
| B200 | SM100 (Blackwell) | 192 GB HBM3e | `sm100_b200_comprehensive_extended_{bf16,fp16,fp32}.json` |
| B300 | SM103 (Blackwell) | 268 GB HBM3e | `sm103_b300_comprehensive_extended_{bf16,fp16,fp32}.json` |

### Model-level (3 GPUs, r=384, bs=1, seq=4096, ga=8, loss_tokens=1024, 20 repeats)

| GPU | File |
|-----|------|
| RTX 6000 PRO | `rtx6000_seq4096_bs1_gas8_seq4k_loss1k_n20w2_*.json` |
| H200 | `h200_seq4096_bs1_gas8_seq4k_loss1k_n20w2_*.json` |
| B200 | `b200_seq4096_bs1_gas8_seq4k_loss1k_n20w2_*.json` |

### High-rank (H200, r=512, loss_tokens=1024, 20 repeats)

| File | Models |
|------|--------|
| `h200_r512_loss1k_n20w2_*.json` | Qwen3.5-27B, Qwen3-VL-32B |

6 models total: Qwen2.5-VL-32B, Qwen3-VL-32B, Qwen3.5-27B, Gemma3-27B, Mistral-Small-24B, Qwen3-VL-8B.

---

## Convergence Data

**Directory**: `code/convergence_runs/`

### Primary (Qwen3.5-9B-Base, r=384, AdamW, 3 seeds × eager/fused = 6 runs)

| Seed | Mode | File |
|------|------|------|
| 1 | eager | `events.out.tfevents.*eager_seed1sft*` |
| 1 | fused | `events.out.tfevents.*fused_seed1sft*` |
| 2 | eager | `events.out.tfevents.*eager_seed2sft*` |
| 2 | fused | `events.out.tfevents.*fused_seed2sft*` |
| 3 | eager | `events.out.tfevents.*eager_seed3sft*` |
| 3 | fused | `events.out.tfevents.*fused_seed3sft*` |

- Dataset: eyes-ml/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking-QR-max4096
- Hardware: 1×RTX 6000 PRO (96 GB GDDR7)
- Grand mean |Δloss|: 7.1e-4, worst max: 1.1e-2 (seed 1), mean final eval |Δ|: 8.9e-5
- Wall-clock: 330 min fused vs 360 min eager (8.3% reduction)

### Cross-model + cross-optimizer (Qwen3-VL-8B, r=256, Muon+AdamW, seed 4)

| Mode | File |
|------|------|
| eager | `events.out.tfevents.*q3vl_8b_muon_eager_seed4sft*` |
| fused | `events.out.tfevents.*q3vl_8b_muon_fused_seed4sft*` |

- Mean |Δloss|: 7.7e-4, max: 5.7e-3, final eval |Δ|: 3.9e-5
- Wall-clock: 325 min fused vs 354 min eager (8.2% reduction)

---

## Figure Generation

| Script | Generates | Data Source |
|--------|-----------|-------------|
| `paper/generate_figures.py` | 13 benchmark figures | `code/bench_it6/` JSON files |
| `paper/generate_training_figure.py` | 1 convergence figure | `code/convergence_runs/` TensorBoard events |

---

## Historical Data

Benchmark iterations 1–5 (`bench_it1/` through `bench_it5/`), the old convergence
JSONL data (`tboard/`), and the old proprietary-dataset SFT runs (`sft_runs/`) have
been removed from the repository. They are preserved in the git history for
provenance. All paper claims derive exclusively from `bench_it6/` and
`convergence_runs/`.
