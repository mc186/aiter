import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def static_per_tensor_scaled_fp8_quant(
    out: Tensor, 
    input: Tensor, 
    scale: Tensor
): 


def dynamic_per_tensor_scaled_fp8_quant(
    out: Tensor, 
    input: Tensor, 
    scale: Tensor
):

def dynamic_per_token_scaled_fp8_quant(
    out: Tensor, 
    input: Tensor, 
    scales: Tensor, 
    scale_ub: Optional[Tensor] = None
):
