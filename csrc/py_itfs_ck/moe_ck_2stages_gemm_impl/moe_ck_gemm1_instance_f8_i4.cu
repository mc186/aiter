#include "moe_ck_gemm.hpp"
template<>
void ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScale, 128>(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
                        int topk,
                        void *&hidden_states,           // [m, k], input token
                        void *&w1,                      // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                        void *&w2,                      // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                        void *&sorted_token_ids,        // [max_num_tokens_padded]
                        void *&sorted_expert_ids,       // [max_num_m_blocks]
                        void *&num_valid_ids,           //[1]
                        void *&out,                     // [max_num_tokens_padded, inter_dim]
                        std::optional<void *> w1_scale, // [e, 1, n], gate(up) scale
                        std::optional<void *> a1_scale  // [m, 1], token scale
)
{
    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;
    ck::index_t KBatch = 1;
    using A0DataType       = F8;
    using B0DataType       = I4;
    using EDataType        = F16;
    using AccDataType      = F32;
    using CShuffleDataType = F32;
    using D0DataType       = F32;
    using D1DataType       = F32;
    using DsDataType       = ck::Tuple<D0DataType, D1DataType>;

    using A0Layout = Row;
    using B0Layout = Col;
    using ELayout  = Row;
    using D0Layout = Row;
    using D1Layout = Col;
    using DsLayout = ck::Tuple<D0Layout, D1Layout>;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CDEElementOp = MulABScale;
    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

    static constexpr ck::index_t MPerBlock = 128;
    static constexpr ck::index_t Nswizzle  = false;
    using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemm<
                Row, Col, DsLayout, ELayout, 
                A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
                AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   
                256,   MPerBlock,   128,    64,
                16,   32,
                32,   32,
                4,    1,
                S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
                S<2, 128, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 32, 32, 0,
                4,    1,   S<1, 32, 1, 8>, S<4, 1, 1>,
                ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, Nswizzle, true, A0DataType>;
    
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
                               hidden_states,
                               w1,
                               std::array<const void *, NumDTensor>{a1_scale.has_value() ? a1_scale.value() : nullptr,
                                                                    w1_scale.has_value() ? w1_scale.value() : nullptr},
                               out,
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, NumDTensor>{I0, I0},
                               StrideE,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if (!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_op with the specified compilation parameters does "
            "not support this MOE stage1 problem");
    }

    invoker.Run(argument, StreamConfig{stream});
}
