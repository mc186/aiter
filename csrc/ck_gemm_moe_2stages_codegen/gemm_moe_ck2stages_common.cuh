// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "gemm_moe_ck2stages.h"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include <iostream>

template <
    typename A0DataType,
    typename B0DataType,
    typename AccDataType,
    typename EDataType,
    typename CDEElementOp,
    PipelineVersion PipelineVer,
    int BLOCKSIZE,
    int MPerBlock,
    int NPerBlock,
    int KPerBlock,
    int MWaves,
    int NWaves,
    bool Nswizzle,
    bool PerTensorQuant,
    bool MulRoutedWeight,
    int ActOP>
void ck_moe_stage1_gemm(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
                        int topk,
                        void *&hidden_states,     // [m, k], input token
                        void *&w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                        void *&w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                        void *&sorted_token_ids,  // [max_num_tokens_padded]
                        void *&sorted_expert_ids, // [max_num_m_blocks]
                        void *&sorted_weights,
                        void *&num_valid_ids,           // [1]
                        void *&out,                     // [max_num_tokens_padded, inter_dim]
                        std::optional<void *> w1_scale, // [e, 1, n], gate(up) scale
                        std::optional<void *> a1_scale  // [m, 1], token scale
)
{
    // ~~~~~~~~~~~~~~~~~~~~~~~~following start with ck things
    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = N;
    ck::index_t KBatch = 1;
    // using AccDataType = F32;
    using CShuffleDataType = F32;
    using DsDataType = ck::Tuple<F32, F32, F32>;

    using A0Layout = Row;
    using B0Layout = Col;
    using D0Layout = Row;
    using D1Layout = Col;
    using ELayout = Row;
    using D2Layout = ELayout;
    using DsLayout = ck::Tuple<D0Layout, D1Layout, D2Layout>;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
    static constexpr ck::index_t MNPerXDL = 16;
    static constexpr ck::index_t WAVES = BLOCKSIZE / 64;
    static constexpr ck::index_t MXDLPerWave = MPerBlock / (MNPerXDL * MWaves);
    static constexpr ck::index_t NXDLPerWave = NPerBlock / (MNPerXDL * NWaves);
    // static constexpr ck::index_t NPerBlock = PipelineVer == ck::BlockGemmPipelineVersion::v1 ? 64 : 128;
    static constexpr ck::index_t CShuffleMXDLPerWave = ck::is_same_v<B0DataType, I4> ? 2 : MXDLPerWave;
    static constexpr ck::index_t CShuffleNXDLPerWave = ck::is_same_v<B0DataType, I4> ? 1 : NXDLPerWave;
    static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);
    static constexpr ck::index_t BK1 = ck::is_same_v<B0DataType, I4> ? 32 : 16 / sizeof(B0DataType);
    static constexpr ck::index_t EVec = 16 / sizeof(EDataType);
    static constexpr ck::index_t K0_A = KPerBlock / AK1;
    static constexpr ck::index_t K0_B = KPerBlock / BK1;
    static constexpr ck::index_t K0_M_A = BLOCKSIZE / K0_A;
    static constexpr ck::index_t K0_N_B = BLOCKSIZE / K0_B;
    static constexpr ck::index_t D0Vec = 1;
    static constexpr ck::index_t D1Vec = PerTensorQuant ? 1 : EVec;
    static constexpr ck::index_t D2Vec = 1;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemm
        // clang-format off
///######|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     DsData|     EData|     AccData|         CShuffle|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
///######|         |         |         |        |       Type|       Type|       Type|      Type|        Type|         DataType| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
///######|         |         |         |        |           |           |           |          |            |                 |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
///######|         |         |         |        |           |           |           |          |            |                 |            |            |             |               |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |    S<C, D0, D1>|
///###### RCR
          <      Row,      Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
               AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
               BLOCKSIZE,   MPerBlock,   NPerBlock,    KPerBlock,
               AK1,   BK1,
               MNPerXDL,   MNPerXDL,
               MXDLPerWave,    NXDLPerWave,
               S<K0_A, K0_M_A, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
               S<K0_B, K0_N_B, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, BK1, BK1, 0,
               2,    CShuffleNXDLPerWave,   S<1, 32, 1, 8>, S<EVec, D0Vec, D1Vec>,
               ck::BlockGemmPipelineScheduler::Intrawave, PipelineVer, ActOP, Nswizzle, true, MulRoutedWeight, !PerTensorQuant, ck::long_index_t, A0DataType>;
    // clang-format on

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    constexpr auto I0 = ck::Number<0>{};
    constexpr auto I1 = ck::Number<1>{};
    static constexpr auto DStride = PerTensorQuant ? I0 : I1;

    // do GEMM
    auto device_op = DeviceOpInstance{};

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids,
                               sorted_expert_ids,
                               num_valid_ids,
                               hidden_states,
                               w1,
                               std::array<const void *, NumDTensor>{a1_scale.has_value() ? a1_scale.value() : nullptr,
                                                                    w1_scale.has_value() ? w1_scale.value() : nullptr,
                                                                    MulRoutedWeight ? sorted_weights : nullptr},
                               out,
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, NumDTensor>{DStride, DStride, I0},
                               StrideE,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if (!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    invoker.Run(argument, StreamConfig{stream});
}

#define CK_MOE_STAGE1_GEMM_DEFINE(BLOCKSIZE, MPerfBlock, NPerBlock, KPerBlock, MWaves, NWaves, PipelineVer)                                                                                                                     \
    template void ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, BLOCKSIZE, MPerfBlock, NPerBlock, KPerBlock, MWaves, NWaves, Nswizzle, PerTensorQuant, MulRoutedWeight, ActOP>( \
        const hipStream_t &stream,                                                                                                                                                                                              \
        int tokens, int sorted_size, int N, int K,                                                                                                                                                                              \
        int topk,                                                                                                                                                                                                               \
        void *&hidden_states,                                                                                                                                                                                                   \
        void *&w1,                                                                                                                                                                                                              \
        void *&w2,                                                                                                                                                                                                              \
        void *&sorted_token_ids,                                                                                                                                                                                                \
        void *&sorted_expert_ids,                                                                                                                                                                                               \
        void *&sorted_weights,                                                                                                                                                                                                  \
        void *&num_valid_ids,                                                                                                                                                                                                   \
        void *&out,                                                                                                                                                                                                             \
        std::optional<void *> w1_scale,                                                                                                                                                                                         \
        std::optional<void *> a1_scale);

template <
    typename A0DataType,
    typename B0DataType,
    typename AccDataType,
    typename EDataType,
    typename CDEElementOp,
    PipelineVersion PipelineVer,
    int BLOCKSIZE,
    int MPerBlock,
    int NPerBlock,
    int KPerBlock,
    int MWaves,
    int NWaves,
    bool Nswizzle,
    bool PerTensorQuant,
    bool MulRoutedWeight,
    int ActOP = 0>
void ck_moe_stage2_gemm(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
                        int topk,
                        void *&inter_states,            // [max_num_tokens_padded, k], input token
                        void *&w1,                      // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                        void *&w2,                      // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                        void *&sorted_token_ids,        // [max_num_tokens_padded]
                        void *&sorted_expert_ids,       // [max_num_m_blocks]
                        void *&sorted_weights,          // [max_num_tokens_padded]
                        void *&num_valid_ids,           //[1]
                        void *&out,                     // [m, out_dim]
                        std::optional<void *> w2_scale, // [e, 1, n], gate(up) scale
                        std::optional<void *> a2_scale  // [max_num_tokens_padded, 1], token scale
)
{
    // ~~~~~~~~~~~~~~~~~~~~~~~~following start with ck things
    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = N;
    ck::index_t KBatch = 1;

    // using AccDataType = F32;
    using CShuffleDataType = F32;
    using DsDataType = ck::Tuple<F32, F32, F32>;

    using A0Layout = Row;
    using B0Layout = Col;
    using ELayout = Row;
    using D0Layout = Row;
    using D1Layout = Col;
    using DsLayout = ck::Tuple<D0Layout, D1Layout, ELayout>;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
    // static constexpr ck::index_t BLOCKSIZE = 256;
    static constexpr ck::index_t WAVES = BLOCKSIZE / 64;
    static constexpr ck::index_t MNPerXDL = 16;
    static constexpr ck::index_t MXDLPerWave = MPerBlock / (MNPerXDL * MWaves);
    static constexpr ck::index_t NXDLPerWave = NPerBlock / (MNPerXDL * NWaves);
    static constexpr ck::index_t CShuffleMXDLPerWave = ck::is_same_v<B0DataType, I4> ? 2 : MXDLPerWave;
    static constexpr ck::index_t CShuffleNXDLPerWave = ck::is_same_v<B0DataType, I4> ? 2 : NXDLPerWave;
    static constexpr ck::index_t CShuffleNLane = ck::is_same_v<B0DataType, I4> ? 32 : NPerBlock / 2 / NXDLPerWave; // 64
    static constexpr ck::index_t CShuffleMLane = BLOCKSIZE / CShuffleNLane;
    static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);
    static constexpr ck::index_t BK1 = ck::is_same_v<B0DataType, I4> ? 32 / sizeof(B0DataType) : 16 / sizeof(B0DataType);
    static constexpr ck::index_t EVec = 2;
    static constexpr ck::index_t D0Vec = 1;
    static constexpr ck::index_t D1Vec = PerTensorQuant ? 1 : EVec;
    static constexpr ck::index_t D2Vec = 1;
    static constexpr ck::index_t K0_A = KPerBlock / AK1;
    static constexpr ck::index_t K0_B = KPerBlock / BK1;
    static constexpr ck::index_t K0_M = BLOCKSIZE / K0_A;
    static constexpr ck::index_t K0_N = BLOCKSIZE / K0_B;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemm
        // clang-format off
///#####|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     DsData|     EData|     AccData|         CShuffle|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
///#####|         |         |         |        |       Type|       Type|       Type|      Type|        Type|         DataType| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
///#####|         |         |         |        |           |           |           |          |            |                 |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
///#####|         |         |         |        |           |           |           |          |            |                 |            |            |             |               |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |    S<C, D0, D1>|
///##### RCR
       <      Row,      Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
              AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
              BLOCKSIZE,   MPerBlock,   NPerBlock,    KPerBlock,
              AK1,   BK1,
              MNPerXDL,   MNPerXDL,
              MXDLPerWave, NXDLPerWave,
              S<K0_A, K0_M, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
              S<K0_B, K0_N, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, BK1, BK1, 0,
              CShuffleMXDLPerWave,    1,   S<1, CShuffleMLane, 1, CShuffleNLane>, S<EVec, D0Vec, D1Vec, D2Vec>,
              ck::BlockGemmPipelineScheduler::Intrawave, PipelineVer, 0, Nswizzle, false, MulRoutedWeight, !PerTensorQuant, ck::long_index_t, A0DataType>;


    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    constexpr auto I0 = ck::Number<0>{};
    constexpr auto I1 = ck::Number<1>{};
    static constexpr auto DStride = PerTensorQuant ? I0 : I1;

    // do GEMM
    auto device_op = DeviceOpInstance{};

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids,
                               sorted_expert_ids,
                               num_valid_ids,
                               inter_states,
                               w2,
                               std::array<const void *, NumDTensor>{a2_scale.has_value() ? a2_scale.value() : nullptr,
                                                                    w2_scale.has_value() ? w2_scale.value() : nullptr,
                                                                    MulRoutedWeight ? sorted_weights : nullptr},
                               out,
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, NumDTensor>{DStride, DStride, I0},
                               StrideE,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if (!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }
    invoker.Run(argument, StreamConfig{stream});
}

#define CK_MOE_STAGE2_GEMM_DEFINE(BLOCKSIZE, MPerfBlock, NPerfBlock, KPerBlock, MWaves, NWaves, PipelineVer)                                                                                    \
    template void ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, BLOCKSIZE, MPerfBlock, NPerfBlock, KPerBlock, MWaves, NWaves, Nswizzle, PerTensorQuant, MulRoutedWeight, ActOP>( \
        const hipStream_t &stream,                                                                                                                                   \
        int tokens, int sorted_size, int N, int K,                                                                                                                   \
        int topk,                                                                                                                                                    \
        void *&inter_states,                                                                                                                                         \
        void *&w1,                                                                                                                                                   \
        void *&w2,                                                                                                                                                   \
        void *&sorted_token_ids,                                                                                                                                     \
        void *&sorted_expert_ids,                                                                                                                                    \
        void *&sorted_weights,                                                                                                                                       \
        void *&num_valid_ids,                                                                                                                                        \
        void *&out,                                                                                                                                                  \
        std::optional<void *> w2_scale,                                                                                                                              \
        std::optional<void *> a2_scale);