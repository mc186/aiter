# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# Copyright (C) 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

from typing import Optional
import functools
import json
import os
import triton
import triton.language as tl
import torch
from aiter.ops.triton.activation import _tanh
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH


@triton.jit
def _fwd_grouped_kernel_stage1_rope(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    cos_sin_cache,  # max_seq_len x (rotary_dim * 2)
    positions,  # sequence positions
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank + 1)
    k_pe_t_out,
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_kpe_tokens_out_b,
    stride_cos_sin_cache_s,
    stride_positions_b,
    rotary_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    USE_ROPE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_c = tl.arange(0, BLOCK_C)
    offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe

    off_q_pe = (
        cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :]
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    mask_c = offs_c < kv_lora_rank
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)
    q_pe = tl.load(
        Q + off_q_pe, mask=(mask_h[:, None]) & (mask_qk_r[None, :]), other=0.0
    )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # apply rotary embedding for q_pe, and k_pe (last token per batch of K_PE)
    LAST_SPLIT = split_kv_end == cur_batch_seq_len
    k_pe_last_token = tl.zeros([BLOCK_R], dtype=q.dtype)

    if USE_ROPE:
        if IS_NEOX_STYLE:
            # [BLOCK_ROTARY // 2, BLOCK_ROTARY // 2 + 1, BLOCK_ROTARY // 2 + 2, ..., 0, 1, 2, ..., BLOCK_ROTARY // 2 - 1, pass:]
            offs_qk_rot_r = kv_lora_rank + (
                (tl.arange(0, BLOCK_R) + (rotary_dim // 2)) % rotary_dim
            )
            # Which elements to flip
            mask_rotate = tl.arange(0, BLOCK_R) < (rotary_dim // 2)
            # [0 , 1, 2, ..., rotary_dim // 2 - 1, 0 , 1, 2, ..., rotary_dim // 2 - 1]
            offs_rotary = tl.arange(0, BLOCK_R) % (rotary_dim // 2)
        else:
            # [1, 0, 3, 2, 5, 4, ..., BLOCK_R, BLOCK_R - 1]
            offs_qk_rot_r = (
                kv_lora_rank
                + (((tl.arange(0, BLOCK_R) + 1) % 2) * 2)
                - 1
                + tl.arange(0, BLOCK_R)
            )
            mask_rotate = tl.arange(0, BLOCK_R) % 2 < 1
            # [0, 0, 1, 1, ..., rotary_dim // 2 - 1, rotary_dim // 2 - 1]
            offs_rotary = tl.arange(0, BLOCK_R) // 2

        if qk_rope_head_dim > rotary_dim:
            offs_qk_rot_r = tl.where(
                tl.arange(0, BLOCK_R) < rotary_dim, offs_qk_rot_r, tl.arange(0, BLOCK_R)
            )
            offs_rotary = tl.where(
                tl.arange(0, BLOCK_R) < rotary_dim, offs_rotary, tl.arange(0, BLOCK_R)
            )

        mask_rotary = tl.arange(0, BLOCK_R) < rotary_dim

        pos = tl.load(positions + cur_batch * stride_positions_b)
        cos = tl.load(
            cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary,
            mask=mask_rotary,
            other=1.0,
        )
        sin = tl.load(
            cos_sin_cache
            + pos * stride_cos_sin_cache_s
            + offs_rotary
            + rotary_dim // 2,
            mask_rotary,
            other=0.0,
        )

        off_q_pe_rot = (
            cur_batch * stride_qb
            + cur_head[:, None] * stride_qh
            + offs_qk_rot_r[None, :]
        )
        mask_qk_rot_r = offs_qk_rot_r < (kv_lora_rank + qk_rope_head_dim)

        # 0, 2, 4,.... 1, 3, 5...
        q_pe_rot = tl.load(
            Q + off_q_pe_rot,
            mask=(mask_h[:, None]) & (mask_qk_rot_r[None, :]),
            other=0.0,
        )
        q_pe_rot = tl.where(mask_rotate[None, :], -q_pe_rot, q_pe_rot)

        q_pe = q_pe * cos + q_pe_rot * sin

        # we only apply to the last token in the K_PE
        if LAST_SPLIT:
            # debug assert
            if (
                cur_batch == 0 and cur_head_id == 0
            ) and split_kv_id < NUM_KV_SPLITS - 1:
                tl.device_assert(False, "Only last split should compute k_pe")

            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + cur_batch_seq_len - 1
            )
            offs_buf_k_pe_last_token = kv_loc * stride_buf_kbs + offs_qk_r
            offs_buf_k_pe_rot_last_token = kv_loc * stride_buf_kbs + offs_qk_rot_r
            k_pe_last_token = tl.load(K_Buffer + offs_buf_k_pe_last_token)

            k_pe_rot_last_token = tl.load(K_Buffer + offs_buf_k_pe_rot_last_token)
            k_pe_rot_last_token = tl.where(
                mask_rotate, -k_pe_rot_last_token, k_pe_rot_last_token
            )

            k_pe_last_token = k_pe_last_token * cos + k_pe_rot_last_token * sin

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            offs_buf_kv = kv_loc[None, :] * stride_buf_kbs + offs_c[:, None]
            offs_buf_k_pe = kv_loc[None, :] * stride_buf_kbs + offs_qk_r[:, None]

            k_pe = tl.load(
                K_Buffer + offs_buf_k_pe,
                mask=(offs_n[None, :] < split_kv_end) & (mask_qk_r[:, None]),
                other=0.0,
            )  # positional embedding part of keys

            if (USE_ROPE and LAST_SPLIT) and start_n >= cur_batch_seq_len - BLOCK_N:
                k_pe = tl.where(
                    offs_n[None, :] != (split_kv_end - 1),
                    k_pe,
                    k_pe_last_token[:, None],
                )

            # (16, 64) x (64, 32)
            # dot product of rope parts
            qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))

            kv = tl.load(
                K_Buffer + offs_buf_kv,
                mask=(offs_n[None, :] < split_kv_end) & (mask_c[:, None]),
                other=0.0,
            )  # the shared latent tensor for keys and values

            # (16, 512) x (512, 32)
            # dot product of nope parts
            qk += tl.dot(q, kv)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + offs_c[None, :]
            v = tl.load(
                V_buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_c[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            # (16, 32) x (32, 512)
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_c[None, :]
        )

        if USE_ROPE:
            if LAST_SPLIT:
                k_pe_last_token_ptrs = (
                    k_pe_t_out
                    + cur_batch * stride_kpe_tokens_out_b
                    + tl.arange(0, BLOCK_R)
                )
                tl.store(k_pe_last_token_ptrs, k_pe_last_token, mask=mask_qk_r)

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_c[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + kv_lora_rank
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# TODO rope offset
def _decode_grouped_att_m_fwd_rope(
    q,
    k_buffer,
    v_buffer,
    att_out,
    k_pe_tokens_out,
    kv_lora_rank,  # c
    cos_sin_cache,
    positions,
    rotary_dim,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    sm_scale,
    logit_cap,
    use_rope,
    is_neox_style,
    config
):
    if use_rope:
        assert (
            k_pe_tokens_out is not None
        ), "We must output the k_pe tokens with rope applied if rope fusion enabled."

    qk_rope_head_dim = k_buffer.shape[-1] - kv_lora_rank
    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    config["BLOCK_C"] = triton.next_power_of_2(kv_lora_rank)
    config["BLOCK_R"] = triton.next_power_of_2(qk_rope_head_dim)

    config["NUM_KV_SPLITS"] = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(config["BLOCK_H"], kv_group_num)),
        config["NUM_KV_SPLITS"],
    )


    _fwd_grouped_kernel_stage1_rope[grid](
        q,
        k_buffer,
        v_buffer,
        cos_sin_cache,
        positions,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        k_pe_tokens_out,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        v_buffer.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_pe_tokens_out.stride(0) if use_rope else 0,
        cos_sin_cache.stride(0) if use_rope else 0,
        positions.stride(0) if use_rope else 0,
        rotary_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        logit_cap=logit_cap,
        USE_ROPE=use_rope,
        IS_NEOX_STYLE=is_neox_style,
        **config,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    O,
    kv_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    config,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    config["BLOCK_DV"] = triton.next_power_of_2(Lv)

    config["NUM_KV_SPLITS"] = num_kv_splits

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        kv_indptr,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        Lv=Lv,
        **config,
    )


@functools.lru_cache(maxsize=1024)
def _get_config():
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MLA_DECODE_ROPE-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    return _get_config._config_dict

def decode_attention_fwd_grouped_rope(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    k_pe_tokens: torch.Tensor,
    kv_lora_rank: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    attn_logits: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    logit_cap: Optional[float] = 0.0,
    use_rope: Optional[bool] = False,
    is_neox_style: Optional[bool] = False,
    config: Optional[dict[str, any]] = None,
):
    """
    Implements deepseek decode attention with grouped query attention and rotary positional encoding

    parameters:
    q: Query Tensor
    k_buffer: Key Cache Tensor
    v_buffer: Value Cache Tensor
    o: Output tensor containing the result of decode. Allocated by the caller
    kv_indptr:
    kv_indices:
    k_pe_tokens:
    kv_lora_rank:
    rotary_dim
    cos_sin_cache:
    positions:
    attn_logits:
    num_kv_splits:
    sm_scale
    logit_cap:
    use_rope
    is_neox_style

    Returns:
    o: output Tensor

    """
    if config is None:
        config = _get_config()

    _decode_grouped_att_m_fwd_rope(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        k_pe_tokens,
        kv_lora_rank,
        cos_sin_cache,
        positions,
        rotary_dim,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
        use_rope,
        is_neox_style,
        config["fwd_grouped_kernel_stage1_rope"]
    )
    _decode_softmax_reducev_fwd(attn_logits, 
        q, 
        o, 
        v_buffer, 
        kv_indptr, 
        num_kv_splits,
        config["fwd_kernel_stage2"]
        )
