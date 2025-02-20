#include "moe_ck_gemm.hpp"
template<>
void ck_moe_stage2_gemm<F8, I4, F32, F16, MulABScaleExpertWeight, 128>(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
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
    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = N;
    ck::index_t KBatch = 1;
    using A0DataType       = F8;
    using B0DataType       = I4;
    using EDataType        = F16;
    using AccDataType      = F32;
    using CShuffleDataType = F32;
    using D0DataType       = F32;
    using D1DataType       = F32;
    using D2DataType       = F32;
    using DsDataType       = ck::Tuple<D0DataType, D1DataType, D2DataType>;

    using A0Layout = Row;
    using B0Layout = Col;
    using ELayout  = Row;
    using D0Layout = Row;
    using D1Layout = Col;
    using D2Layout = ELayout;
    using DsLayout = ck::Tuple<D0Layout, D1Layout, D2Layout>;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CDEElementOp = MulABScaleExpertWeight;
    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

    static constexpr ck::index_t MPerBlock = 128;
    static constexpr ck::index_t BLOCKSIZE = 256;
    static constexpr ck::index_t MXDLPerWave = 2; 
    static constexpr ck::index_t NXDLPerWave = 2; 
    static constexpr ck::index_t NPerBlock = 128;
    static constexpr ck::index_t MNPerXDL = 32;
    static constexpr ck::index_t KPerBlock = 128 / sizeof(A0DataType);
    // static constexpr ck::index_t MXDLPerWave = MPerBlock / 32; //todo fix this constraint
    // static constexpr ck::index_t CShuffleMXDLPerWave = MPerBlock / 32;
    static constexpr ck::index_t CShuffleNLane = 32;
    static constexpr ck::index_t CShuffleMLane = BLOCKSIZE / CShuffleNLane;
    static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);
    static constexpr ck::index_t BK1 = 16 / sizeof(B0DataType);
    static constexpr ck::index_t EVec = 2;
    static constexpr ck::index_t D0Vec = 1;
    static constexpr ck::index_t D1Vec = 1;
    static constexpr ck::index_t D2Vec = 1;
    using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemm
    // clang-format off
        <      Row, Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
               AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
               BLOCKSIZE,   MPerBlock,   NPerBlock,    KPerBlock,
               AK1,   BK1,
               MNPerXDL,   MNPerXDL,
               MXDLPerWave,    NXDLPerWave,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
               S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
               MXDLPerWave,    1,   S<1, CShuffleMLane, 1, CShuffleNLane>, S<EVec, D0Vec, D1Vec, D2Vec>,
               ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, false, false, A0DataType>;
    
    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker = device_op.MakeInvoker();

    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr ck::index_t I0 = 0;
    auto argument =
        device_op.MakeArgument(sorted_token_ids,
                               sorted_expert_ids,
                               num_valid_ids,
                               inter_states,
                               w2,
                               std::array<const void *, NumDTensor>{a2_scale.has_value() ? a2_scale.value() : nullptr,
                                                                    w2_scale.has_value() ? w2_scale.value() : nullptr,
                                                                    sorted_weights},
                               out,
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, NumDTensor>{I0, I0, I0},
                               StrideE,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if (!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_op with the specified compilation parameters does "
            "not support this MOE stage2 problem");
    }

    invoker.Run(argument, StreamConfig{stream});
}