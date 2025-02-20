import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _gemm_a8w8_kernel():
    #TBD

def gemm_a8w8( 
        xq: Tensor,
        wq: Tensor,
        x_scale: Tensor,
        w_scale: Tensor,
        bias: Optional[Tensor] = None,
        out_dtype=torch.bfloat16,
        splitK: Optional[int] = None
    ):
    """
    Multiply quantized matrix xq and wq and add an optional bias parameter.

    Parameters:
    xq (Tensor): Matrix of shape (m,k) and fp8 type 
    wq (Tensor): Matrix of shape (n,k) and fp8 type
    x_scale: (Tensor): scalar value that each element in xq is multiplied with
    w_scale: (Tensor): scalar value that each element in wq is multiplied with
    bias (Tensor):  Optional tensor of shape(n)
    out_dtype: Dtype of output
    splitK: If set to true then do splitK GEMM

    Returns:
    Tensor: Result of multiplying ((xq*x_scale) * (wq*y_scale)) + bias(if not None) 
    """

    #Check output_dtype
    #Allocate output
    #Set up grid
    #Call Triton Kernel
    #Return output
