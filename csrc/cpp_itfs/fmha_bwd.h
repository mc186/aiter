#include "fmha_bwd.hpp"
#include "mask.hpp"

struct fmha_bwd_traits_all: public fmha_bwd_traits
{
    fmha_bwd_traits_all(const mask_info &mask,
        std::string dtype,
        int head_size,
        bool has_dropout,
        bool enable_alibi,
        bool deterministic,
        bool use_ext_asm,
        bool is_v3_atomic_fp32,
        int how_v3_bf16_cvt): fmha_bwd_traits{head_size,
            head_size,
            dtype,
            false, // is_group_mode
            mask.type,
            enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
            false,    // has_dbias
            has_dropout,
            false, // s_randval
            deterministic}, 
            use_ext_asm(use_ext_asm),
            is_v3_atomic_fp32(is_v3_atomic_fp32),
            how_v3_bf16_cvt(how_v3_bf16_cvt) {}
    bool use_ext_asm;
    bool is_v3_atomic_fp32;
    int how_v3_bf16_cvt;
};

fmha_bwd_traits_all get_ck_fmha_bwd_traits_all(const mask_info &mask,
    std::string dtype,
    int head_size,
    bool has_dropout,
    bool enable_alibi,
    bool deterministic,
    bool use_ext_asm,
    bool is_v3_atomic_fp32,
    int how_v3_bf16_cvt)
{
return fmha_bwd_traits_all(mask,
        dtype,
        head_size,
        has_dropout,
        enable_alibi,
        deterministic,
        use_ext_asm,
        is_v3_atomic_fp32,
        how_v3_bf16_cvt);
}

float fmha_bwd_aiter(fmha_bwd_args args,
    mask_info mask,
    std::string q_dtype_str,
    bool enable_alibi,
    bool deterministic,
    bool use_ext_asm,
    bool is_v3_atomic_fp32,
    int how_v3_bf16_cvt)