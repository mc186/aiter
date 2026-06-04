"""
Correctness tests for the optional GQA-aware spatial swizzle (SWIZZLE=1).

Verifies bit-identical output to the default path (SWIZZLE=0) across
multiple (HQ, HK) regimes for both causal and non-causal attention.
"""
import os
import math
import sys

import torch


def _run(swizzle, B, Hq, Hk, S, D, causal):
    """Run flash_attn_func with a given AITER_SWIZZLE value. Reloads the module
    so the env-var change takes effect inside the kernel call."""
    os.environ["AITER_SWIZZLE"] = str(swizzle)
    import importlib
    import aiter.ops.triton.attention.mha as m
    importlib.reload(m)
    q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, Hk, D, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)
    out = m.flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=causal,
        window_size=(-1, -1),
        return_lse=False,
        return_attn_probs=False,
        sink=None,
    )
    return out[0] if isinstance(out, tuple) else out


def check(name, Hq, Hk, S=8192, D=128, causal=True, atol=1e-3):
    """Compare SWIZZLE=1 vs SWIZZLE=0 outputs bit-for-bit."""
    torch.manual_seed(42)
    out0 = _run(0, 1, Hq, Hk, S, D, causal).float()
    torch.manual_seed(42)
    out1 = _run(1, 1, Hq, Hk, S, D, causal).float()
    diff = (out0 - out1).abs()
    rel_max = (diff.max() / out0.abs().max()).item() if out0.abs().max() > 0 else 0.0
    ok = rel_max < atol
    status = "PASS" if ok else f"FAIL (rel_max={rel_max:.3e})"
    print(f"  {name:48s}  rel_max={rel_max:.3e}  {status}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("No CUDA device — skipping")
        sys.exit(0)

    print("== Bit-identity tests: SWIZZLE=1 vs SWIZZLE=0 default ==")
    all_pass = True
    # Aligned regime (HK == NUM_XCDS = 8)
    all_pass &= check("HQ=128 HK=8   causal     (aligned)",      128,  8, causal=True)
    all_pass &= check("HQ=128 HK=8   non-causal (aligned)",      128,  8, causal=False)
    # HK > NUM_XCDS (each XCD owns multiple KV heads)
    all_pass &= check("HQ=128 HK=16  causal     (HK>NXCD)",      128, 16, causal=True)
    all_pass &= check("HQ=128 HK=16  non-causal (HK>NXCD)",      128, 16, causal=False)
    all_pass &= check("HQ=128 HK=32  causal     (HK=4*NXCD)",    128, 32, causal=True)
    # HK < NUM_XCDS (XCDs share KV heads)
    all_pass &= check("HQ=128 HK=4   causal     (HK<NXCD)",      128,  4, causal=True)
    all_pass &= check("HQ=128 HK=2   causal     (HK<<NXCD)",     128,  2, causal=True)
    # MHA (NUM_QUERIES_PER_KV=1) — uses MHA fallback path
    all_pass &= check("HQ=HK=40      causal     (MHA fallback)",  40, 40, causal=True)

    print()
    if all_pass:
        print("ALL TESTS PASS")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
