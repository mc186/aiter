# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import logging
import numpy as np
from aiter.ops.triton.mha import (
    flash_attn_func,
    flash_attn_fp8_func,
)
from aiter.test_mha_common import (
    attention_ref,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False
ATOL_fp8 = 2.5e-1
RTOL_fp8 = 2.5e-1


def fp8_assert_close(
    tensor_a, tensor_b, atol=ATOL_fp8, rtol=RTOL_fp8, max_diff_percentage=0.5
):
    """Assert tensors are close with tolerance for small percentage of elements"""
    # standard comparison
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / torch.abs(tensor_b.clamp(min=1e-6))

    # calculate elements that exceed tolerance
    abs_check = abs_diff > atol
    rel_check = rel_diff > rtol
    failed_check = torch.logical_and(abs_check, rel_check)

    # calculate percentage of failed elements
    failed_percentage = failed_check.sum().item() / failed_check.numel() * 100

    # if percentage is small enough, test passes
    if failed_percentage <= max_diff_percentage:
        return True

    # Otherwise, provide diagnostic information
    max_abs_idx = torch.argmax(abs_diff).item()
    max_rel_idx = torch.argmax(rel_diff).item()

    flat_to_idx = lambda flat_idx, shape: np.unravel_index(flat_idx, shape)

    max_abs_pos = flat_to_idx(max_abs_idx, tensor_a.shape)
    max_rel_pos = flat_to_idx(max_rel_idx, tensor_a.shape)

    max_abs_diff = abs_diff.flatten()[max_abs_idx].item()
    max_rel_diff = rel_diff.flatten()[max_rel_idx].item()

    raise AssertionError(
        f"Tensors not close enough! {failed_percentage:.6f}% elements exceed tolerance.\n"
        f"Greatest absolute difference: {max_abs_diff} at index {max_abs_pos} (up to {atol} allowed)\n"
        f"Greatest relative difference: {max_rel_diff} at index {max_rel_pos} (up to {rtol} allowed)"
    )


def test_mha_single_config(dtype=torch.float16):
    """
    Single configuration test for MHA
    Configuration: BATCH=4, SEQLEN_Q=128, SEQLEN_K=128, NUM_Q_HEADS=16, NUM_K_HEADS=16, 
                  HEAD_SZ=32, DROPOUT=0.0, RETURN_LSE=False, RETURN_SOFTMAX=False, 
                  CAUSAL=False, FP8=False
    """
    # Test configuration
    BATCH = 4
    SEQLEN_Q = 1024
    SEQLEN_K = 1024
    NUM_Q_HEADS = 16
    NUM_K_HEADS = 16
    HEAD_SZ = 32
    DROPOUT = 0.0
    RETURN_LSE = False
    RETURN_SOFTMAX = False
    CAUSAL = False
    FP8 = False

    print(f"Running test with configuration:")
    print(f"  BATCH={BATCH}, SEQLEN_Q={SEQLEN_Q}, SEQLEN_K={SEQLEN_K}")
    print(f"  NUM_Q_HEADS={NUM_Q_HEADS}, NUM_K_HEADS={NUM_K_HEADS}, HEAD_SZ={HEAD_SZ}")
    print(f"  DROPOUT={DROPOUT}, RETURN_LSE={RETURN_LSE}, RETURN_SOFTMAX={RETURN_SOFTMAX}")
    print(f"  CAUSAL={CAUSAL}, FP8={FP8}, dtype={dtype}")

    # Generate input tensors
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    dropout_mask = None
    
    # Run Triton implementation
    if FP8:
        triton_out = flash_attn_fp8_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=RETURN_LSE,
            return_attn_probs=RETURN_SOFTMAX,
        )
    else:
        triton_out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=RETURN_LSE,
            return_attn_probs=RETURN_SOFTMAX,
        )

    # Handle return values
    if RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            assert len(triton_out) == 3
            sd_mask = triton_out[2]
        else:
            assert len(triton_out) == 2
            sd_mask = triton_out[1]
        dropout_mask = sd_mask >= 0
        if DEBUG_MODE:
            print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}")

    if RETURN_SOFTMAX or RETURN_LSE:
        triton_out = triton_out[0]
    
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    # Run reference implementation
    torch_out = attention_ref(
        q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL
    )
    torch_out, attention_scores = torch_out
    
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}")

    # Compare results
    if FP8:
        fp8_assert_close(
            triton_out, torch_out.to(triton_out.dtype), atol=ATOL_fp8, rtol=RTOL_fp8
        )
        print("FP8 test passed!")
    else:
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)
        print("Standard test passed!")

    print(f"Test completed successfully!")
    print(f"Output shape: {triton_out.shape}")
    print(f"Max absolute difference: {torch.abs(triton_out - torch_out).max().item():.6f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the single configuration test
    test_mha_single_config()
