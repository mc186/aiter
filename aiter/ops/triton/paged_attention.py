import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

def paged_attention_prefill(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        b_loc: torch.Tensor,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_ctx_len: torch.Tensor,
        max_input_len: int,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        alibi_slopes: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        sm_scale: Optional[float] = None
):

def paged_attention_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor] = None,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
):