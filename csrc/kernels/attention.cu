/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_runtime.h>

#include "ck_tile/host/hip_check_error.hpp"

#include "paged_attention.hpp"


/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/torch.h>

#include <hip/hip_runtime.h>

#include "paged_attention.hpp"
#include "paged_attention_kernel.hpp"

#define LAUNCH_CUSTOM_ATTENTION(GQA_RATIO)                        \
    paged_attention_ll4mi_QKV_kernel<T,                           \
                                     KVT,                         \
                                     KV_DTYPE,                    \
                                     OUTT,                        \
                                     BLOCK_SIZE,                  \
                                     HEAD_SIZE,                   \
                                     NTHR,                        \
                                     GQA_RATIO>                   \
        <<<grid, block, 0, stream>>>(query_ptr,                   \
                                     key_cache_ptr,               \
                                     value_cache_ptr,             \
                                     args.num_kv_heads,           \
                                     args.scale,                  \
                                     args.block_tables_ptr,       \
                                     args.context_lens_ptr,       \
                                     args.max_num_blocks_per_seq, \
                                     args.alibi_slopes_ptr,       \
                                     args.q_stride,               \
                                     args.kv_block_stride,        \
                                     args.kv_head_stride,         \
                                     args.exp_sums_ptr,           \
                                     args.max_logits_ptr,         \
                                     tmp_out_ptr,                 \
                                     out_ptr,                     \
                                     max_ctx_blocks,              \
                                     args.k_scale,                \
                                     args.v_scale,                \
                                     args.fp8_out_scale_ptr);

#define LAUNCH_CUSTOM_REDUCTION(NPAR_LOOPS)                                                        \
    paged_attention_ll4mi_reduce_kernel<T, OUTT, HEAD_SIZE, HEAD_SIZE, PARTITION_SIZE, NPAR_LOOPS> \
        <<<reduce_grid, reduce_block, 0, stream>>>(out_ptr,                                        \
                                                   args.exp_sums_ptr,                              \
                                                   args.max_logits_ptr,                            \
                                                   tmp_out_ptr,                                    \
                                                   args.context_lens_ptr,                          \
                                                   max_num_partitions,                             \
                                                   args.fp8_out_scale_ptr);

namespace {
template <typename T,
          typename KVT,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE,
          int HEAD_SIZE,
          typename OUTT,
          int PARTITION_SIZE>
void paged_attention_custom_launcher(const native::paged_attention_args& args, hipStream_t stream)
{

    T* tmp_out_ptr       = reinterpret_cast<T*>(args.tmp_out_ptr);
    T* query_ptr         = reinterpret_cast<T*>(args.query_ptr);
    KVT* key_cache_ptr   = reinterpret_cast<KVT*>(args.key_cache_ptr);
    KVT* value_cache_ptr = reinterpret_cast<KVT*>(args.value_cache_ptr);
    OUTT* out_ptr        = reinterpret_cast<OUTT*>(args.out_ptr);

    const int max_ctx_blocks     = DIVIDE_ROUND_UP(args.max_context_len, BLOCK_SIZE);
    const int max_num_partitions = DIVIDE_ROUND_UP(args.max_context_len, PARTITION_SIZE);
    const int gqa_ratio          = args.num_heads / args.num_kv_heads;
    assert(args.num_heads % args.num_kv_heads == 0);
    assert(args.head_size == HEAD_SIZE);

    constexpr int NTHR = PARTITION_SIZE;
    dim3 grid(args.num_seqs, max_num_partitions, args.num_kv_heads);
    dim3 block(NTHR);

    switch(gqa_ratio)
    {
    case 1: LAUNCH_CUSTOM_ATTENTION(1); break;
    case 2: LAUNCH_CUSTOM_ATTENTION(2); break;
    case 3: LAUNCH_CUSTOM_ATTENTION(3); break;
    case 4: LAUNCH_CUSTOM_ATTENTION(4); break;
    case 5: LAUNCH_CUSTOM_ATTENTION(5); break;
    case 6: LAUNCH_CUSTOM_ATTENTION(6); break;
    case 7: LAUNCH_CUSTOM_ATTENTION(7); break;
    case 8: LAUNCH_CUSTOM_ATTENTION(8); break;
    case 9: LAUNCH_CUSTOM_ATTENTION(9); break;
    case 10: LAUNCH_CUSTOM_ATTENTION(10); break;
    case 11: LAUNCH_CUSTOM_ATTENTION(11); break;
    case 12: LAUNCH_CUSTOM_ATTENTION(12); break;
    case 13: LAUNCH_CUSTOM_ATTENTION(13); break;
    case 14: LAUNCH_CUSTOM_ATTENTION(14); break;
    case 15: LAUNCH_CUSTOM_ATTENTION(15); break;
    case 16: LAUNCH_CUSTOM_ATTENTION(16); break;
    default: TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio); break;
    }

    // reduction kernel is only required if max_context_len > partition size,
    // otherwise main kernel writes directly to final output
    //  note there are cases with graphing where max_context_len is the max
    //  supported by graphing, not the actual max among all the sequences: in that
    //  case reduction kernel will still run but return immediately
    if(args.max_context_len > PARTITION_SIZE)
    {
        dim3 reduce_grid(args.num_heads, args.num_seqs);
        dim3 reduce_block(args.head_size);
        const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
        // support upto 8*64*256=128K context length
        switch(npar_loops)
        {
        case 1: LAUNCH_CUSTOM_REDUCTION(1); break;
        case 2: LAUNCH_CUSTOM_REDUCTION(2); break;
        case 3: LAUNCH_CUSTOM_REDUCTION(3); break;
        case 4: LAUNCH_CUSTOM_REDUCTION(4); break;
        case 5: LAUNCH_CUSTOM_REDUCTION(5); break;
        case 6: LAUNCH_CUSTOM_REDUCTION(6); break;
        case 7: LAUNCH_CUSTOM_REDUCTION(7); break;
        case 8: LAUNCH_CUSTOM_REDUCTION(8); break;
        default: TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops); break;
        }
    }
}
} // namespace

#define CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE)              \
    paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE>(args, \
                                                                                        stream);

#define CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT)              \
    switch(args.partition_size)                                                              \
    {                                                                                        \
    case 256: CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, 256); break; \
    case 512: CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, 512); break; \
    default: TORCH_CHECK(false, "Unsupported partition size: ", args.partition_size); break; \
    }

#if defined(__HIPCC__) && defined(__gfx90a__)
#define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)       \
    if(args.fp8_out_scale_ptr)                                                \
    {                                                                         \
        TORCH_CHECK(false, "fp8 out scale unsupported for gfx90a");           \
    }                                                                         \
    else                                                                      \
    {                                                                         \
        CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T); \
    }
#else
#define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)             \
    if(args.fp8_out_scale_ptr)                                                      \
    {                                                                               \
        CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, uint8_t); \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T);       \
    }
#endif
#define CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, HEAD_SIZE)                        \
    switch(args.block_size)                                                          \
    {                                                                                \
    case 16: CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 16, HEAD_SIZE); break;       \
    case 32: CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 32, HEAD_SIZE); break;       \
    default: TORCH_CHECK(false, "Unsupported block size: ", args.block_size); break; \
    }

#define CALL_CUSTOM_LAUNCHER_BLK_HEAD(T, KVT, KV_DTYPE)                            \
    switch(args.head_size)                                                         \
    {                                                                              \
    case 64: CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 64); break;                \
    case 128: CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 128); break;              \
    default: TORCH_CHECK(false, "Unsupported head size: ", args.head_size); break; \
    }

namespace native {
void paged_attention(const paged_attention_traits& traits,
                     const paged_attention_args& args,
                     hipStream_t stream)
{
    if(traits.kv_cache_dtype == "auto")
    {
        if(traits.q_type == ScalarType::Half)
        {
            CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, _Float16, vllm::Fp8KVCacheDataType::kAuto);
        }
        else if(traits.q_type == ScalarType::BFloat16)
        {
            CALL_CUSTOM_LAUNCHER_BLK_HEAD(
                __hip_bfloat16, __hip_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported data type: ", traits.q_type);
        }
    }
    else if(traits.kv_cache_dtype == "fp8" || traits.kv_cache_dtype == "fp8_e4m3")
    {
        if(traits.q_type == ScalarType::Half)
        {
            CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
        }
        else if(traits.q_type == ScalarType::BFloat16)
        {
            CALL_CUSTOM_LAUNCHER_BLK_HEAD(
                __hip_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
        }
        else
        {
            TORCH_CHECK(false, "Unsupported data type: ", traits.q_type);
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported KV cache dtype: ", traits.kv_cache_dtype);
    }
}
} // namespace native


void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& tmp_out,     // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache, // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& context_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    double k_scale,
    double v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale,
    int64_t partition_size)
{
    native::paged_attention_traits traits;

    traits.q_type         = (query.dtype() == at::ScalarType::Half ? native::ScalarType::Half
                                                                   : native::ScalarType::BFloat16);
    traits.kv_cache_dtype = kv_cache_dtype;

    native::paged_attention_args args;

    args.num_seqs               = query.size(0);
    args.num_heads              = query.size(1);
    args.head_size              = query.size(2);
    args.max_num_blocks_per_seq = block_tables.size(1);
    args.q_stride               = query.stride(0);
    args.kv_block_stride        = key_cache.stride(0);
    args.kv_head_stride         = key_cache.stride(1);

    // NOTE: alibi_slopes is optional.
    args.alibi_slopes_ptr =
        alibi_slopes ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr()) : nullptr;

    args.exp_sums_ptr     = reinterpret_cast<float*>(exp_sums.data_ptr());
    args.max_logits_ptr   = reinterpret_cast<float*>(max_logits.data_ptr());
    args.tmp_out_ptr      = tmp_out.data_ptr();
    args.query_ptr        = query.data_ptr();
    args.key_cache_ptr    = key_cache.data_ptr();
    args.value_cache_ptr  = value_cache.data_ptr();
    args.block_tables_ptr = block_tables.data_ptr<int>();
    args.context_lens_ptr = context_lens.data_ptr<int>();

    // NOTE: fp8_out_scale is optional.
    args.fp8_out_scale_ptr =
        fp8_out_scale ? reinterpret_cast<const float*>(fp8_out_scale.value().data_ptr()) : nullptr;
    args.out_ptr = out.data_ptr();

    args.block_size = block_size;

    args.max_context_len = max_context_len;
    args.num_kv_heads    = num_kv_heads;
    args.partition_size  = partition_size;
    args.scale           = scale;
    args.k_scale         = k_scale;
    args.v_scale         = v_scale;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    native::paged_attention(traits, args, stream);
}
