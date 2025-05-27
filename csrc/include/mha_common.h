#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace aiter {
__global__ void ParsePhiloxCudaState(at::PhiloxCudaState arg, uint64_t* rng_state)
{
    // Imitate from PyTorch
    // https://github.com/pytorch/pytorch/blob/8b61daaf7349e9102117e1aeefaa51666d887547/aten/src/ATen/cuda/detail/UnpackRaw.cuh#L17
    if (arg.captured_) {
        rng_state[0] = static_cast<uint64_t>(*arg.seed_.ptr);
        rng_state[1] = static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_);
    } else {
        rng_state[0] = arg.seed_.val;
        rng_state[1] = arg.offset_.val;
    }
}

inline int num_splits_heuristic_ck(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    for (int num_splits = 1; num_splits <= max_splits; num_splits *= 2) {
        if (num_SMs < batch_nheads_mblocks * (num_splits * 2)) {
            return num_splits;
        }
    }
 
    return max_splits;
}

inline int override_num_splits_if_necessary(int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits)
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        return num_splits;

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
        return num_splits;

    // TODO - tile size should match the TileFmhaShape, hardcode for now
    const int kM0 = 16; // 16 for fmha_batch_prefill(); 64 for fmha_fwd_splitkv()
    const int kN1 = hdim_v;

    const int num_m_blocks = (max_seqlen_q + kM0 - 1) / kM0;
    const int num_n_blocks = (hdim_v + kN1 - 1) / kN1;

    if(num_splits < 1 && p_drop == 0.0f)
        return num_splits_heuristic_ck(
            batch * nhead * num_m_blocks, props.multiProcessorCount, num_n_blocks, 8);

    return num_splits;
}

} // namespace flash