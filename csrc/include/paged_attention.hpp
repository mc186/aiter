#pragma once

#include <iostream>

#include <hip/hip_runtime.h>

namespace native {
enum class ScalarType
{
    Half,
    BFloat16,
};

inline std::ostream& operator<<(std::ostream& stream, ScalarType scalar_type)
{
    switch(scalar_type)
    {
    case ScalarType::Half: stream << "Half"; break;
    case ScalarType::BFloat16: stream << "BFloat16"; break;
    }
    return stream;
}

struct paged_attention_traits
{
    ScalarType q_type;
    std::string kv_cache_dtype;
};

struct paged_attention_args
{
    int head_size;

    int num_seqs;
    int num_heads;
    int num_kv_heads;

    int max_num_blocks_per_seq;
    int q_stride;
    int kv_block_stride;
    int kv_head_stride;

    // NOTE: alibi_slopes is optional.
    const float* alibi_slopes_ptr;

    float* exp_sums_ptr;
    float* max_logits_ptr;
    void* tmp_out_ptr;
    void* query_ptr;
    void* key_cache_ptr;
    void* value_cache_ptr;
    int* block_tables_ptr;
    int* context_lens_ptr;
    const float* fp8_out_scale_ptr;
    void* out_ptr;

    int64_t block_size;
    int64_t max_context_len;
    double scale;
    double k_scale;
    double v_scale;
    int64_t partition_size;
};

void paged_attention(const paged_attention_traits& traits,
                     const paged_attention_args& args,
                     hipStream_t stream);
} // namespace native