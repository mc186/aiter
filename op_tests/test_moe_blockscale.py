# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, run_perftest
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe, moe_sorting_ck
from aiter.fused_moe_gelu import fused_topk
from aiter.ops.shuffle import shuffle_weight
from aiter import pertoken_quant, ck_moe
from einops import rearrange

BLOCK_SIZE_M = 32


def torch_moe_blockscale(hidden_states,
                         w1,  # [expert, inter_dim*2, model_dim]
                         w2,  # [expert, model_dim, inter_dim]
                         topk_weight, topk_ids,
                         dtype,
                         # following for quant
                         scale_blks=(128, 128),
                         a_scale=None,
                         # [expert, inter_dim/blk_m, model_dim/blk_k]
                         fc1_scale=None,
                         # [expert, model_dim/blk_m, inter_dim/blk_k]
                         fc2_scale=None,
                         expert_mask=None):
    computeType = torch.float
    token_num, topk = topk_ids.shape
    num_experts, model_dim, inter_dim = w2.shape
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = (expert_mask.cumsum(0, dtype=torch.int32) - 1)
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    hidden_states = hidden_states.view(
        token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    if w2.shape[2]*2 == w1.shape[1]:
        moeType = "g1u1"
    else:
        moeType = "g1u0"

    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        blk_n, blk_k = scale_blks
        expert, nblk_n, nblk_k = fc1_scale.shape
        fc1_scale = rearrange(fc1_scale.view(-1, 1).repeat(1, blk_n*blk_k).view(expert, nblk_n, nblk_k, blk_n, blk_k),
                              'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)')
        expert, nblk_n, nblk_k = fc1_scale.shape
        fc2_scale = rearrange(fc2_scale.view(-1, 1).repeat(1, blk_n*blk_k).view(expert, nblk_n, nblk_k, blk_n, blk_k),
                              'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)')
        w1 = w1.to(computeType) * fc1_scale.to(computeType)
        w2 = w2.to(computeType) * fc2_scale.to(computeType)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                act_out = F.silu(gate) * up
            else:
                act_out = F.gelu(act_input)
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


# @perftest(num_warmup=1, num_iters=2)
def torch_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                   # following for int8 quant
                   fc1_scale=None,  # [expert, inter_dim, 1]
                   fc2_scale=None,  # [expert, model_dim, 1]
                   fc1_smooth_scale=None,  # [expert, 1, model_dim]
                   fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                   ):
    return torch_moe(hidden_states,
                     w1,
                     w2,
                     topk_weight,
                     topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale)


def asm_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                 # following for int8 quant
                 fc1_scale=None,  # [expert, inter_dim, 1]
                 fc2_scale=None,  # [expert, model_dim, 1]
                 fc1_smooth_scale=None,  # [expert, 1, model_dim]
                 fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                 a16=False,
                 ):

    return asm_moe(hidden_states,
                   w1,
                   w2,
                   topk_weight,
                   topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, a16)


torch.set_default_device("cuda")


def test_fmoe(dtype, token, model_dim, inter_dim, scale_blks, E, topk, quant='No', use_g1u1=False, shared_E=0):
    input = torch.randn((token, model_dim), dtype=dtype)
    if use_g1u1:
        w1 = torch.randn((E+shared_E, inter_dim*2, model_dim), dtype=dtype)
    else:
        w1 = torch.randn((E+shared_E, inter_dim, model_dim), dtype=dtype)
    w2 = torch.randn((E+shared_E, model_dim, inter_dim), dtype=dtype)
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    scale_blk_n, scale_blk_k = scale_blks
    quant_dtype = torch.float8_e4m3fnuz

    # block quant w1
    tmp = rearrange(w1.view(-1,
                            w1.shape[1]//scale_blk_n, scale_blk_n,
                            w1.shape[2]//scale_blk_k, scale_blk_k),
                    'e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)')
    w1_q, w1_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
    w1_q = rearrange(w1_q.view(-1,
                               w1.shape[1]//scale_blk_n, w1.shape[2]//scale_blk_k,
                               scale_blk_n, scale_blk_k),
                     'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)')

    # block quant w2
    tmp = rearrange(w2.view(-1,
                            model_dim//scale_blk_n, scale_blk_n,
                            inter_dim//scale_blk_k, scale_blk_k),
                    'e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)')
    w2_q, w2_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
    w2_q = rearrange(w2_q.view(-1,
                               w2.shape[1]//scale_blk_n, w2.shape[2]//scale_blk_k,
                               scale_blk_n, scale_blk_k),
                     'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)')

    # block quant input
    input
    print(w1_q.shape, w1_scale.shape)
    # w2, fc2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)

    out_ref, us_ref = run_perftest(torch_moe_blockscale,
                                   input,
                                   w1,
                                   w2,
                                   topk_weights,
                                   topk_ids,
                                   dtype,
                                   num_warmup=1, num_iters=2)
    out_ref2, us_ref = run_perftest(torch_moe,
                                    input,
                                    w1,
                                    w2,
                                    topk_weights,
                                    topk_ids,
                                    num_warmup=1, num_iters=2)
    msg = '111'
    checkAllclose(out_ref, out_ref2, rtol=0.01, atol=100, msg=msg)


for dtype in [torch.float16, torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for idim in [1024]:
                scale_blks = (128, 128)
                test_fmoe(dtype, m, dim, idim, scale_blks, 32, 5, quant='No')
