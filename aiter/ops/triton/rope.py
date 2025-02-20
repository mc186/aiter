
def rope_fwd(
    input: Tensor,
    freqs: Tensor,
    transpose_output: bool = False
) -> Tensor :

def rope_bwd(
    output_grads: Tensor,
    freqs: Tensor,
    transpose_output: bool = False
) -> Tensor :

def rope_cached_fwd(
    input: Tensor,
    cos: Tensor,
    sin: Tensor,
    transpose_output: bool = False
) -> Tensor :

def rope_cached_bwd(
    output_grads: Tensor,
    cos: Tensor,
    sin: Tensor,
    transpose_output: bool = False
) -> Tensor :

def rope_thd_fwd(
    input: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
) -> Tensor :

def rope_thd_bwd(
    output_grads: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor
) -> Tensor :

def rope_2d_fwd(
    input: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    img_height: int,
    img_width: int
) -> Tensor :

def rope_2d_bwd(
    output_grads: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    cos_w: Tensor,
    sin_w: Tensor,
    img_height: int,
    img_width: int
) -> Tensor :


def rotary_embedding_fwd(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
):

def batched_rotary_embedding(
    positions: Tensor,
    query: Tensor,
    key: Tensor,
    head_size: int,
    cos_sin_cache: Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: Tensor,
):