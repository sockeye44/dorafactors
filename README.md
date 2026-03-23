# Official code for "Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels"

Memory-efficient DoRA (Weight-Decomposed Low-Rank Adaptation) for PEFT, featuring factored
column norms, fused Triton kernels with custom autograd, and automatic dispatch across eager
PyTorch and Triton backends.

From the paper: *Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels*
(arXiv preprint arXiv:XXXX.XXXXX, 2026).

## Reproducing benchmark results

**Clone with submodules** (recommended — one command, zero setup):

```bash
git clone --recurse-submodules https://github.com/sockeye44/dorafactors
cd dorafactors
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

Install dependencies:

```bash
pip install -r code/requirements.txt
```

**Run the benchmark** (from the repo root):

```bash
python code/bench_dora_comprehensive.py --suite all --verbose
```

The script validates its vendor dependencies at startup and will guide you
through any missing pieces.  All paths are resolved relative to the script's
location, so `python code/bench_dora_comprehensive.py` works from any working
directory — but **always invoke it via the repo-root-relative path** shown above
to keep commands unambiguous and copy-pasteable.

<details>
<summary>What the script expects at startup</summary>

| Artifact | Source | How it gets there |
|---|---|---|
| `vendor/dorafactors-peft/` | Git submodule → [`sockeye44/dorafactors-peft`](https://github.com/sockeye44/dorafactors-peft) branch `v1` | `--recurse-submodules` or `git submodule update --init` |
| Reference `dora.py` | Upstream HF PEFT @ [`20a9829`](https://github.com/huggingface/peft/blob/20a9829f76419149f5e447b856bc0abe865c28a7/src/peft/tuners/lora/dora.py) | Searched at `code/scripts/dora.reference_hf_peft.py` then `vendor/` (SHA-1 verified); fetch with `wget` if absent |

If either is missing or fails integrity checks, the script prints exact
remediation commands.  In interactive sessions it offers a `[y/N]` prompt to
continue under non-standard conditions; in non-interactive sessions (CI, piped
stdin) it exits with an error.

</details>

### Alternative: reconstruct the PEFT fork from patch

Instead of using the submodule, you can reconstruct the patched PEFT module
from `hf.patch` against upstream PEFT commit
[`20a9829`](https://github.com/huggingface/peft/commit/20a9829) (`v0.18.0.rc0`):

```bash
git clone https://github.com/huggingface/peft /tmp/peft-fork
cd /tmp/peft-fork && git checkout 20a9829
git apply /path/to/dorafactors/hf.patch
```

### Generating paper figures

The figure-generation scripts read pre-collected benchmark JSON from
`code/bench_it6/` and TensorBoard events from `code/convergence_runs/`.
Both resolve data paths relative to their own location, so they work from
any working directory.

```bash
# Microbenchmark + model-level figures (requires matplotlib, numpy)
python paper/generate_figures.py          # PDF only
python paper/generate_figures.py --png    # PDF + PNG

# Training convergence figure (requires matplotlib, numpy, tensorboard)
python paper/generate_training_figure.py
```

Outputs go to `paper/figures/`.

### Running the test suite

Unit tests for the factored norm, fused kernels, and DoRA math live in the
PEFT fork submodule.  One test (`test_reference_vs_optimized_forward_equivalence`)
loads the upstream HF PEFT baseline from `docs/dora.reference_hf_peft.py`
inside the fork tree — copy it from the parent repo before running:

```bash
cp code/scripts/dora.reference_hf_peft.py vendor/dorafactors-peft/docs/
cd vendor/dorafactors-peft
pytest tests/test_lora_variants.py \
       tests/tuners/lora/test_dora_fused.py \
       tests/tuners/lora/test_dora_math.py -v
```

Triton kernel tests require an SM 80+ GPU (Ampere or newer); validated on
SM 80 (A100) through SM 120 (RTX 6000 PRO).

## Documentation

Full documentation, how-to guides, and API reference are available at:

- [**Home**](https://sockeye44.github.io/dorafactors-docs/) — overview and quick-start
- [**Getting Started**](https://sockeye44.github.io/dorafactors-docs/getting-started/) — installation, setup, and first steps
- [**Configuration**](https://sockeye44.github.io/dorafactors-docs/config/) — all configuration options and environment variables

## Modules

| Module | Description | Reference |
|--------|-------------|-----------|
| `peft.tuners.lora.dora` | Layer classes (`DoraLinearLayer`, `DoraEmbeddingLayer`, conv variants), configuration functions, FSDP/ZeRO-3 integration, and eager composition helpers | [Layer Classes](layers.md), [Configuration](config.md) |
| `peft.tuners.lora.dora_fused` | Fused Triton kernels for DoRA compose, norm assembly, and forward+inner products; custom autograd function; PyTorch fallbacks; autotune configs | [Fused Kernels](fused.md) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PEFT_DORA_FUSED` | unset (auto) | Enable fused Triton kernels: `"1"`, `"0"`, or unset (auto: use if Triton available) |
| `PEFT_DORA_FUSED_BACKWARD` | unset (on) | Fused backward pass: `"1"` (force on, bypass shape heuristic), `"0"` (off), or unset (on, with shape-based filtering for linear layers) |
| `PEFT_DORA_NORM_CHUNK_MB` | `256` | Column-norm chunking threshold in MB; matrices exceeding this are chunked (min 16) |
| `PEFT_DORA_FWD_CHUNK_MB` | `256` | Forward-pass chunking threshold in MB (min 16) |
| `DORA_AUTOTUNE_COMPREHENSIVE` | `"0"` | Enable comprehensive Triton autotuning (`"1"` for full search) |
| `PEFT_DORA_ALLOW_PARTIAL_GATHER` | `"0"` | Allow partial parameter gathering under ZeRO-3 (`"1"` to enable) |
| `PEFT_FORCE_GATHER` | unset (auto) | Force full parameter gathering: `"1"`, `"0"`, or unset (auto-detect) |

## Module Relationships

`dora.py` is the primary module: it defines all layer classes and configuration functions.
It lazy-imports `dora_fused.py` on first use (guarded by `_get_dora_fused()`) so that Triton
is not required at import time.

## Citation

```bibtex
@article{zelenin2026dorafactors,
  title         = {Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels},
  author        = {Zelenin, Alexandra and Zhuravlyova, Alexandra},
  journal       = {arXiv preprint arXiv:XXXX.XXXXX},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  year          = {2026}
}
```
