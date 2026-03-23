#!/usr/bin/env python3
"""Quick compose parity check at Gemma3-scale dimensions.

Measures Triton-vs-PyTorch and cross-path agreement after the
canonical associativity fix (mag * (scale * lora) everywhere).

Usage:
    python scripts/check_compose_parity.py          # needs CUDA + Triton
    modal run scripts/check_compose_parity.py       # or via Modal
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

os.environ["PEFT_DORA_FUSED"] = "1"

import torch
from peft.tuners.lora.dora_fused import (
    _fused_dora_compose_torch,
    _fused_dora_compose_triton,
    _fused_dora_forward_and_inner_torch,
    _fused_dora_forward_and_inner_triton,
    _TRITON_AVAILABLE,
)

def main():
    if not torch.cuda.is_available() or not _TRITON_AVAILABLE:
        print("Need CUDA + Triton")
        return

    device = torch.device("cuda")
    F = torch.nn.functional

    print(f"{'hidden':>7} {'scale':>6} | {'PT OOP-vs-IP':>20} {'diff':>10} | "
          f"{'Triton-vs-PT OOP':>20} {'diff':>10} | "
          f"{'PT FAI-vs-OOP':>20}")
    print("-" * 120)

    for hidden in [3840, 4096, 8192]:
        for embed_scale in [1.0, 62.0, 128.0]:
            torch.manual_seed(42)
            total = 4 * 2048
            dtype = torch.bfloat16

            lora = torch.randn(total, hidden, device=device, dtype=dtype)
            base = torch.randn(total, hidden, device=device, dtype=dtype) * embed_scale
            mag = torch.rand(1, hidden, device=device, dtype=dtype) + 0.5
            scale = 0.3

            # OOP paths
            pt_oop = _fused_dora_compose_torch(lora.clone(), base, mag, scale, inplace=False)
            tr_oop = _fused_dora_compose_triton(lora.clone(), base, mag, scale, inplace=False)

            cos_tr = F.cosine_similarity(pt_oop.float().flatten(), tr_oop.float().flatten(), dim=0).item()
            diff_tr = (pt_oop.float() - tr_oop.float()).abs().max().item()

            # In-place
            pt_ip = lora.clone()
            _fused_dora_compose_torch(pt_ip, base, mag, scale, inplace=True)

            cos_ip = F.cosine_similarity(pt_oop.float().flatten(), pt_ip.float().flatten(), dim=0).item()
            diff_ip = (pt_oop.float() - pt_ip.float()).abs().max().item()

            # Forward-and-inner
            pt_fai, _ = _fused_dora_forward_and_inner_torch(lora.clone(), base, mag, scale)
            cos_fai = F.cosine_similarity(pt_fai.float().flatten(), pt_oop.float().flatten(), dim=0).item()

            print(f"{hidden:7d} {embed_scale:6.1f} | "
                  f"cos={cos_ip:.10f} {diff_ip:10.1e} | "
                  f"cos={cos_tr:.10f} {diff_tr:10.1e} | "
                  f"cos={cos_fai:.10f}")

    print()
    print("PT OOP-vs-IP should be 1.0000000000 (bitwise, canonical order)")
    print("PT FAI-vs-OOP should be 1.0000000000 (bitwise, same eval order)")
    print("Triton-vs-PT: O(eps) due to FMA, typically cos > 0.9999999")

if __name__ == "__main__":
    main()
