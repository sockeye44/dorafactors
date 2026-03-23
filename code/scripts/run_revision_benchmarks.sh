#!/usr/bin/env bash
# Revision benchmarks: high-rank + loss_tokens sensitivity
# Run on B200 (model benchmarks) while OpenMMReasoner convergence runs on H200
#
# Produces 4 JSON files matching bench_it6 naming convention.
# Prerequisites: GPU env var set, e.g. export GPU=b200

set -euo pipefail

GPU="${GPU:?Set GPU env var, e.g. export GPU=b200}"
OUTDIR="../runs"
mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)

MODELS="Qwen/Qwen3.5-27B Qwen/Qwen3-VL-32B-Instruct"
COMMON="--suite models --verbose --repeats 20 --warmup 2 --batch 1 --seqlen 4096 --grad-accum 8 --show-adapted-modules --models $MODELS"

# 1. r=384, loss_tokens=1024 (baseline — validates against existing bench_it6)
python3.12 scripts/bench_dora_comprehensive.py $COMMON \
    --rank 384 --loss-tokens 1024 \
    --json-out "${OUTDIR}/${GPU}_r384_loss1k_n20w2_${TS}.json"

# 2. r=512, loss_tokens=1024 (addresses 1.1: high-rank framing)
python3.12 scripts/bench_dora_comprehensive.py $COMMON \
    --rank 512 --loss-tokens 1024 \
    --json-out "${OUTDIR}/${GPU}_r512_loss1k_n20w2_${TS}.json"

# 3. r=768, loss_tokens=1024 (addresses 1.1: high-rank framing, strongest point)
python3.12 scripts/bench_dora_comprehensive.py $COMMON \
    --rank 768 --loss-tokens 1024 \
    --json-out "${OUTDIR}/${GPU}_r768_loss1k_n20w2_${TS}.json"

# 4. r=384, loss_tokens=4096 (addresses 4.1: loss_tokens sensitivity)
python3.12 scripts/bench_dora_comprehensive.py $COMMON \
    --rank 384 --loss-tokens 4096 \
    --json-out "${OUTDIR}/${GPU}_r384_floss_n20w2_${TS}.json"

echo "Done. Results in ${OUTDIR}/"
