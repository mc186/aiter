import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

def rmsnorm_fwd(
    input: Tensor, 
    weight: Tensor, 
    eps: float = 1e-5, 
    residual_in: Optional[torch.Tensor] = None,
) -> Tensor:

def rmsnorm2d_fwd_smoothquant(
    input: Tensor, 
    weight: Tensor, 
    eps: float = 1e-5, 
    xscale: Tensor,
    yscale: Tensor,
    residual_in: Optional[torch.Tensor] = None,
) -> Tensor | Tuple[Tensor, Tensor]:

def rmsnorm2d_fwd_dynamicquant(
    input: Tensor, 
    weight: Optional[Tensor] = None, 
    eps: float = 1e-5, 
    yscale: Tensor,
    residual_in: Optional[torch.Tensor] = None,
) -> Tensor | Tuple[Tensor, Tensor]: