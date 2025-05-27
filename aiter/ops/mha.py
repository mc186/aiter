# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor, Generator
from typing import Optional, Tuple
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_ROOT_DIR
import torch

@compile_ops("module_mha_fwd", fc_name="mha_fwd")
def mha_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_fwd", fc_name="mha_varlen_fwd")
def mha_varlen_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_bwd", fc_name="mha_bwd")
def mha_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_fmha_v3_bwd", fc_name="fmha_v3_bwd")
def fmha_v3_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_bwd", fc_name="mha_varlen_bwd")
def mha_varlen_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    custom_build_args:Optional[dict]=None,
): ...


@compile_ops("module_mha_batch_prefill", fc_name="mha_batch_prefill")
def mha_batch_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    kv_indptr: Tensor,
    kv_page_indices: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_batch_decode", fc_name="mha_batch_decode")
def mha_batch_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_indptr: Tensor,
    kv_page_indices: Tensor,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    return_softmax_lse: bool,
    out: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
): ...


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (_, seqlen_q, _, _) = q.shape
    # causal=true is the same as causal=false in this case
    if seqlen_q == 1 and alibi_slopes is None:
        causal = False

    md_name = 'mha_fwd'
    filter = '*'
    if q.dtype == torch.float16:
        md_name += '_fp16'
        filter += 'fp16*'
    elif q.dtype == torch.bfloat16:
        md_name += '_bf16'
        filter += 'bf16*'
    if alibi_slopes is None:
        md_name += '_nbias'
        filter += '_nbias*'
    else:
        md_name += '_alibi'
        filter += '_alibi*'
    if not causal and window_size_left == -1 and window_size_right == -1:
        md_name += '_nmask'
        filter += '_nmask*'
    else:
        md_name += '_mask'
        filter += '_mask*'
    if return_lse:
        md_name += '_lse'
        filter += '_lse*'
    else:
        md_name += '_nlse'
        filter+= '_nlse*'
    if dropout_p == 0:
        md_name += '_ndropout'
        filter += '_ndropout*'
    else:
        md_name += '_dropout'
        filter += '_dropout*'

    blob_gen_cmd = f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd ' \
        '--receipt 100 --filter {} --output_dir {{}}'.format(filter)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = mha_fwd(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        return_lse,
        return_softmax,
        None,
        alibi_slopes,
        None,
        custom_build_args={'md_name': md_name, 'blob_gen_cmd': blob_gen_cmd}
    )
    return out, softmax_lse, S_dmask, rng_state


def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1
) -> torch.Tensor:
    md_name = 'mha_bwd'
    filter1 = '*'   # get_bwd_dot_do_o_blobs()
    filter2 = '*'   # get_bwd_convert_dq_blobs()
    filter3 = '*'   # get_bwd_dq_dk_dv_blobs()
    if q.dtype == torch.float16:
        md_name += '_fp16'
        filter1+= 'fp16*'
        filter2+= 'fp16*'
        filter3+= 'fp16*'
    elif q.dtype == torch.bfloat16:
        md_name += '_bf16'
        filter1 += 'bf16*'
        filter2 += 'bf16*'
        filter3 += 'bf16*'
    if alibi_slopes is None:
        md_name += '_nbias'
        filter3 += '_nbias*'
    else:
        md_name += '_alibi'
        filter3 += '_alibi*'
    if not causal and window_size_left == -1 and window_size_right == -1:
        md_name += '_nmask'
        filter3 += '_nmask*'
    else:
        md_name += '_mask'
        filter3 += '_mask*'
    if dropout_p == 0:
        md_name += '_ndropout'
        filter3 += '_ndropout*'
    else:
        md_name += '_dropout'
        filter3 += '_dropout*'
    if deterministic:
        md_name += '_deterministic'
        filter2 += '_deterministic*'
        filter3 += '_deterministic*'
    else:
        md_name += '_ndeterministic'
        filter2 += '_ndeterministic*'
        filter3 += '_ndeterministic*'

    filter = f'{filter1}@{filter2}@{filter3}'

    blob_gen_cmd = f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d bwd ' \
        '--receipt 300 --filter {} --output_dir {{}}'.format(filter)

    (_, seqlen_q, nhead_q, hdim_q) = q.shape
    (_, seqlen_k, nhead_k, hdim_v) = v.shape

    batch_stride_q = q.stride(0)
    stride_q = q.stride(1)
    nhead_stride_q = q.stride(2)

    batch_stride_k = k.stride(0)
    stride_k = k.stride(1)
    nhead_stride_k = k.stride(2)
    
    batch_stride_v = v.stride(0)
    stride_v = v.stride(1)
    nhead_stride_v = v.stride(2)

    batch_stride_do = dout.stride(0)
    stride_do = dout.stride(1)
    nhead_stride_do = dout.stride(2)

    batch_stride_dk = dk.stride(0)
    nhead_stride_dk = dk.stride(2)

    batch_stride_dv = dv.stride(0)
    nhead_stride_dv = dv.stride(2)

    # mask
    window_size_left = -1 if window_size_left >= seqlen_k else window_size_left
    window_size_right = -1 if window_size_right >= seqlen_k else window_size_right
    mask = (causal == True and window_size_left == -1) # causal mask
    nmask = (causal == False and window_size_left == -1 and window_size_right == -1) # no mask
    
    def np():
        # bwd_v3_bf16_a16_rtne
        # bwd_v3_bf16_a16_rtna
        # bwd_v3_bf16_a16_rtz
        # bwd_v3_bf16_a32_rtne
        # bwd_v3_bf16_a32_rtna
        # bwd_v3_bf16_a32_rtz
        # bwd_v3_bf16_causal_a16_rtne
        # bwd_v3_bf16_causal_a16_rtna
        # bwd_v3_bf16_causal_a16_rtz
        # bwd_v3_bf16_causal_a32_rtne
        # bwd_v3_bf16_causal_a32_rtna
        # bwd_v3_bf16_causal_a32_rtz
        # bwd_v3_hd64_bf16_a16_rtne
        # bwd_v3_hd64_bf16_a16_rtna
        # bwd_v3_hd64_bf16_a16_rtz
        # bwd_v3_hd64_bf16_causal_a16_rtne
        # bwd_v3_hd64_bf16_causal_a16_rtna
        # bwd_v3_hd64_bf16_causal_a16_rtz
        # bwd_v3_hd64_fp16_a16
        # bwd_v3_fp16_a16
        # bwd_v3_fp16_a32
        # bwd_v3_hd64_fp16_causal_a16
        # bwd_v3_fp16_causal_a16
        # bwd_v3_fp16_causal_a32
        npssk = seqlen_q == seqlen_k
        npssk &= seqlen_k % 64 == 0
        npssk &= stride_q == stride_do
        npssk &= nhead_stride_q == nhead_stride_do
        npssk &= batch_stride_q == batch_stride_do
        npssk &= stride_k == stride_v
        npssk &= nhead_stride_k == nhead_stride_v
        npssk &= batch_stride_k == batch_stride_v
        npssk &= nhead_stride_k == nhead_stride_dk
        npssk &= nhead_stride_v == nhead_stride_dv
        npssk &= (batch_stride_dk / batch_stride_k) == (nhead_q / nhead_k)
        npssk &= (batch_stride_dv / batch_stride_v) == (nhead_q / nhead_k)

        hd128_case = (hdim_q == 128) and npssk

        hd64_case = (hdim_q == 64 and is_v3_atomic_fp32 == False) and npssk

        ret = hd128_case or hd64_case

        return ret

    def pssk():
        # only for hd64 a32 causal/no causal, fp16/bf16-rtne/rtna/rtz cases
        # nhead_stride_dq_acc >= stride_dq_acc must be guaranteed
        # bwd_v3_hd64_bf16_a32_rtne_pssk
        # bwd_v3_hd64_bf16_a32_rtna_pssk
        # bwd_v3_hd64_bf16_a32_rtz_pssk
        # bwd_v3_hd64_bf16_causal_a32_rtne_pssk
        # bwd_v3_hd64_bf16_causal_a32_rtna_pssk
        # bwd_v3_hd64_bf16_causal_a32_rtz_pssk
        # bwd_v3_hd64_fp16_a32_pssk
        # bwd_v3_hd64_fp16_causal_a32_pssk
        ret = is_v3_atomic_fp32 == True
        ret &= hdim_q == 64
        ret &= nmask or (mask and seqlen_q == seqlen_k) # TODO: or (seqlen_q != seqlen_k and mask_type == top_left)

        return ret

    def pddv():
        # only for a16 causal/no causal, fp16/bf16-rtne/rtna/rtz cases
        # bwd_v3_bf16_a16_rtne_pddv
        # bwd_v3_bf16_a16_rtna_pddv
        # bwd_v3_bf16_a16_rtz_pddv
        # bwd_v3_bf16_causal_a16_rtne_pddv
        # bwd_v3_bf16_causal_a16_rtna_pddv
        # bwd_v3_bf16_causal_a16_rtz_pddv
        # bwd_v3_fp16_a16_pddv
        # bwd_v3_fp16_causal_a16_pddv
        ret = is_v3_atomic_fp32 == False
        ret &= hdim_q > 64 and hdim_q < 128
        ret &= seqlen_q == seqlen_k 
        ret &= seqlen_k % 64 == 0
        ret &= stride_q == stride_do
        ret &= nhead_stride_q == nhead_stride_do
        ret &= batch_stride_q == batch_stride_do
        ret &= stride_k == stride_v
        ret &= nhead_stride_k == nhead_stride_v
        ret &= batch_stride_k == batch_stride_v
        ret &= nhead_stride_k == nhead_stride_dk
        ret &= nhead_stride_v == nhead_stride_dv
        ret &= (batch_stride_dk / batch_stride_k) == (nhead_q / nhead_k)
        ret &= (batch_stride_dv / batch_stride_v) == (nhead_q / nhead_k)

        return ret
    
    def psskddv():
        # only for a32 causal/no causal, fp16/bf16-rtne/rtna/rtz cases
        # bwd_v3_bf16_a32_rtne_psskddv
        # bwd_v3_bf16_a32_rtna_psskddv
        # bwd_v3_bf16_a32_rtz_psskddv
        # bwd_v3_bf16_causal_a32_rtne_psskddv
        # bwd_v3_bf16_causal_a32_rtna_psskddv
        # bwd_v3_bf16_causal_a32_rtz_psskddv
        # bwd_v3_fp16_a32_psskddv
        # bwd_v3_fp16_causal_a32_psskddv
        ret = is_v3_atomic_fp32 == True
        ret &= hdim_q > 64 and hdim_q < 128
        ret &= nmask or (mask and seqlen_q == seqlen_k) # TODO: or (seqlen_q != seqlen_k and mask_type == top_left)

        return ret

    def can_impl_fmha_v3_bwd():
        # basic
        ret = alibi_slopes is None
        ret &= dropout_p == 0.0
        ret &= deterministic == False
        ret &= hdim_q == hdim_v
        ret &= nhead_q % nhead_k == 0
        ret &= hdim_q >= 64 and hdim_q <= 128 and hdim_q % 8 == 0
        ret &= mask or nmask
        ret &= np() or pssk() or pddv() or psskddv()
        # TODO: enable this when GPU_ARCH distinguishable
        # fmha v3 backward ASM kernel only support gfx942
        # archs = validate_and_update_archs()
        # ret &= archs == ["gfx942"]
        return ret

    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if can_impl_fmha_v3_bwd():
        (
            dq,
            dk,
            dv,
            softmax_d,
        ) = fmha_v3_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            deterministic,
            is_v3_atomic_fp32,
            how_v3_bf16_cvt,
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            None
        )
    else:
        (
            dq,
            dk,
            dv,
            softmax_d,
        ) = mha_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            deterministic,
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            None,
            custom_build_args={'md_name': md_name, 'blob_gen_cmd': blob_gen_cmd}
        )
    return softmax_d


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_q_og = q.size(3)
        head_size_v_og = v.size(3)
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return tuple(result)


    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_v_og = dout.size(3)
        dout_padded = dout
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])
        _flash_attn_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state,
            ctx.is_v3_atomic_fp32,
            ctx.how_v3_bf16_cvt
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim_q)
        k: (batch_size, seqlen, nheads_k, headdim_q)
        v: (batch_size, seqlen, nheads_k, headdim_v)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # causal=true is the same as causal=false in this case
    if max_seqlen_q == 1 and alibi_slopes is None:
        causal = False

    md_name = 'mha_varlen_fwd'
    if block_table is None:
        filter_fwd = '*'            # get_fwd_blobs()
        if q.dtype == torch.float16:
            md_name += '_fp16'
            filter_fwd += 'fp16*'
        elif q.dtype == torch.bfloat16:
            md_name += '_bf16'
            filter_fwd += 'bf16*'
        if 0.0 < logits_soft_cap:
            md_name += '_logits'
            filter_fwd += '_logits*'
        else:
            md_name += '_nlogits'
            filter_fwd += '_nlogits*'
        if alibi_slopes is None:
            md_name += '_nbias'
            filter_fwd += '_nbias*'
        else:
            md_name += '_alibi'
            filter_fwd+= '_alibi*'
        if not causal and window_size_left == -1 and window_size_right == -1:
            md_name += '_nmask'
            filter_fwd += '_nmask*'
        else:
            md_name += '_mask'
            filter_fwd += '_mask*'
        if return_lse:
            md_name += '_lse'
            filter_fwd += '_lse*'
        else:
            md_name += '_nlse'
            filter_fwd += '_nlse*'
        if dropout_p == 0:
            md_name += '_ndropout'
            filter_fwd += '_ndropout*'
        else:
            md_name += '_dropout'
            filter_fwd += '_dropout*'
        blob_gen_cmd = [f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd ' \
            '--receipt 200 --filter {} --output_dir {{}}'.format(filter_fwd)]
        blob_gen_cmd.append(f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv ' \
            '--receipt 200 --filter {} --output_dir {{}}'.format('" @ "'))
    else:
        filter_fwd_splitkv1 = '*'   # get_fwd_splitkv_combine_blobs()
        filter_fwd_splitkv2 = '*'   # get_fwd_splitkv_blobs()
        if q.dtype == torch.float16:
            md_name += '_fp16'
            filter_fwd_splitkv1+= 'fp16*'
            filter_fwd_splitkv2+= 'fp16*'
        elif q.dtype == torch.bfloat16:
            md_name += '_bf16'
            filter_fwd_splitkv1+= 'bf16*'
            filter_fwd_splitkv2+= 'bf16*'
        if 0.0 < logits_soft_cap:
            md_name += '_logits'
            filter_fwd_splitkv2 += '_logits*'
        else:
            md_name += '_nlogits'
            filter_fwd_splitkv2 += '_nlogits*'
        if alibi_slopes is None:
            md_name += '_nbias'
            filter_fwd_splitkv2+= '_nbias*'
        else:
            md_name += '_alibi'
            filter_fwd_splitkv2+= '_alibi*'
        if not causal and window_size_left == -1 and window_size_right == -1:
            md_name += '_nmask'
            filter_fwd_splitkv2 += '_nmask*'
        else:
            md_name += '_mask'
            filter_fwd_splitkv2 += '_mask*'
        if return_lse:
            md_name += '_lse'
            # filter_fwd_splitkv1+= '_lse*'
            filter_fwd_splitkv2+= '_lse*'
        else:
            md_name += '_nlse'
            # filter_fwd_splitkv1+= '_nlse*'
            filter_fwd_splitkv2+= '_nlse*'
        md_name += '_pagedkv'
        filter_fwd_splitkv2 += '_pagedkv*'
        filter_fwd_splitkv = f'{filter_fwd_splitkv1}@{filter_fwd_splitkv2}'
        blob_gen_cmd = [f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd ' \
            '--receipt 200 --filter {} --output_dir {{}}'.format('" "')]
        blob_gen_cmd.append(f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv ' \
            '--receipt 200 --filter {} --output_dir {{}}'.format(filter_fwd_splitkv))

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = mha_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        return_lse,
        return_softmax,
        None,
        block_table,
        alibi_slopes,
        None,
        custom_build_args={'md_name': md_name, 'blob_gen_cmd': blob_gen_cmd}
    )
    return out, softmax_lse, S_dmask, rng_state


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    md_name = 'mha_varlen_bwd'
    filter1 = '*'   # get_bwd_dot_do_o_blobs()
    filter2 = '*'   # get_bwd_convert_dq_blobs()
    filter3 = '*'   # get_bwd_dq_dk_dv_blobs()
    if q.dtype == torch.float16:
        md_name += '_fp16'
        filter1 += 'fp16*'
        filter2 += 'fp16*'
        filter3 += 'fp16*'
    elif q.dtype == torch.bfloat16:
        md_name += '_bf16'
        filter1 += 'bf16*'
        filter2 += 'bf16*'
        filter3 += 'bf16*'
    if alibi_slopes is None:
        md_name += '_nbias'
        filter3 += '_nbias*'
    else:
        md_name += '_alibi'
        filter3 += '_alibi*'
    if not causal and window_size_left == -1 and window_size_right == -1:
        md_name += '_nmask'
        filter3 += '_nmask*'
    else:
        md_name += '_mask'
        filter3 += '_mask*'
    if dropout_p == 0:
        md_name += '_ndropout'
        filter3 += '_ndropout*'
    else:
        md_name += '_dropout'
        filter3 += '_dropout*'
    if deterministic:
        md_name += '_deterministic'
        filter2 += '_deterministic*'
        filter3 += '_deterministic*'
    else:
        md_name += '_ndeterministic'
        filter2 += '_ndeterministic*'
        filter3 += '_ndeterministic*'
    filter = f'{filter1}@{filter2}@{filter3}'

    blob_gen_cmd = f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d bwd ' \
        '--receipt 400 --filter {} --output_dir {{}}'.format(filter)

    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = mha_varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        deterministic,
        dq,
        dk,
        dv,
        alibi_slopes,
        rng_state,
        None,
        custom_build_args={'md_name': md_name, 'blob_gen_cmd': blob_gen_cmd}
    )
    return softmax_d

def _flashinfer_batch_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # causal=true is the same as causal=false in this case
    if max_seqlen_q == 1 and alibi_slopes is None:
        causal = False

    md_name = 'mha_batch_prefill'
    filter_fwd = '*'            # get_fwd_blobs()
    if q.dtype == torch.float16:
        md_name += '_fp16'
        filter_fwd += 'fp16*'
    elif q.dtype == torch.bfloat16:
        md_name += '_bf16'
        filter_fwd += 'bf16*'
    if 0.0 < logits_soft_cap:
        md_name += '_logits'
        filter_fwd += '_logits*'
    else:
        md_name += '_nlogits'
        filter_fwd += '_nlogits*'
    if alibi_slopes is None:
        md_name += '_nbias'
        filter_fwd += '_nbias*'
    else:
        md_name += '_alibi'
        filter_fwd+= '_alibi*'
    if not causal and window_size_left == -1 and window_size_right == -1:
        md_name += '_nmask'
        filter_fwd += '_nmask*'
    else:
        md_name += '_mask'
        filter_fwd += '_mask*'
    if return_lse:
        md_name += '_lse'
        filter_fwd += '_lse*'
    else:
        md_name += '_nlse'
        filter_fwd += '_nlse*'
    if dropout_p == 0:
        md_name += '_ndropout'
        filter_fwd += '_ndropout*'
    else:
        md_name += '_dropout'
        filter_fwd += '_dropout*'
    blob_gen_cmd = [f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d batch_prefill ' \
        '--receipt 200 --filter {} --output_dir {{}}'.format(filter_fwd)]

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = mha_batch_prefill(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        return_lse,
        return_softmax,
        None,
        alibi_slopes,
        None,
        custom_build_args={'md_name': md_name, 'blob_gen_cmd': blob_gen_cmd}
    )
    return out, softmax_lse, S_dmask, rng_state

def _flashinfer_batch_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    softmax_scale: float,
    logits_soft_cap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    md_name = 'mha_batch_decode'
    filter_fwd_splitkv1 = '*'   # get_fwd_splitkv_combine_blobs()
    filter_fwd_splitkv2 = '*'   # get_fwd_splitkv_blobs()
    if q.dtype == torch.float16:
        md_name += '_fp16'
        filter_fwd_splitkv1+= 'fp16*'
        filter_fwd_splitkv2+= 'fp16*'
    elif q.dtype == torch.bfloat16:
        md_name += '_bf16'
        filter_fwd_splitkv1+= 'bf16*'
        filter_fwd_splitkv2+= 'bf16*'
    if 0.0 < logits_soft_cap:
        md_name += '_logits'
        filter_fwd_splitkv2 += '_logits*'
    else:
        md_name += '_nlogits'
        filter_fwd_splitkv2 += '_nlogits*'
    if alibi_slopes is None:
        md_name += '_nbias'
        filter_fwd_splitkv2+= '_nbias*'
    else:
        md_name += '_alibi'
        filter_fwd_splitkv2+= '_alibi*'
    md_name += '_nmask'
    filter_fwd_splitkv2 += '_nmask*'
    filter_fwd_splitkv2 += '_pagedkv*'

    filter_fwd_splitkv = f'{filter_fwd_splitkv1}@{filter_fwd_splitkv2}'
    blob_gen_cmd = [f'{CK_DIR}/example/ck_tile/01_fmha/generate.py -d batch_decode ' \
        '--receipt 200 --filter {} --output_dir {{}}'.format(filter_fwd_splitkv)]

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = mha_batch_decode(
        q,
        k,
        v,
        kv_indptr,
        kv_page_indices,
        softmax_scale,
        logits_soft_cap,
        zero_tensors,
        return_lse,
        None,
        alibi_slopes,
        custom_build_args={'md_name': md_name, 'blob_gen_cmd': blob_gen_cmd}
    )
    return out, softmax_lse


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_q_og = q.size(2)
        head_size_v_og = v.size(2)
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_v_og = dout.size(2)
        dout_padded = dout
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])
        _flash_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    logits_soft_cap=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim_q), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim_q), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim_v), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled(),
    )


class FlashInferBatchPrefillFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_q_og = q.size(2)
        head_size_v_og = v.size(2)
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flashinfer_batch_prefill(
            q,
            k,
            v,
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, kv_indptr, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return tuple(result)

def flashinfer_batch_prefill_func(
    q,
    k,
    v,
    cu_seqlens_q,
    kv_indptr,
    kv_page_indices,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    logits_soft_cap=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim_q), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim_q), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim_v), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashInferBatchPrefillFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flashinfer_batch_decode_func(
    q,
    k,
    v,
    kv_indptr,
    kv_page_indices,
    softmax_scale=None,
    logits_soft_cap=0.0,
    alibi_slopes=None,
    return_lse=False,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    head_size_q_og = q.size(2)
    head_size_v_og = v.size(2)
    if head_size_q_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
    if head_size_v_og % 8 != 0:
        v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])
    out_padded, softmax_lse = _flashinfer_batch_decode(
        q,
        k,
        v,
        kv_indptr,
        kv_page_indices,
        softmax_scale,
        logits_soft_cap=logits_soft_cap,
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
    )

    out = out_padded[..., :head_size_v_og]

    result = [out]
    if return_lse:
        result.append(softmax_lse)

    return result[0] if len(result) == 1 else tuple(result)