// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_gemm_utils.cuh"

struct opus_ptpc_pipe_p4d {};
struct opus_ptpc_pipe_p8 {};

template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename TILE_,
        typename PIPE_,
        int PREFETCH_ = 2,
        int WG_PER_CU_ = 1,
        bool ALIGNED_AB_ = true,
        bool ALIGNED_C_ = true>
struct opus_gemm_a8w8_ptpc_bpreshuffle_traits_gfx950 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using TILE  = opus::remove_cvref_t<TILE_>;
    using PIPE  = opus::remove_cvref_t<PIPE_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A   = opus::tuple_element_t<0, DTYPE>;
    using D_B   = opus::tuple_element_t<1, DTYPE>;
    using D_C   = opus::tuple_element_t<2, DTYPE>;
    using D_ACC = opus::tuple_element_t<3, DTYPE>;
    using D_SF  = opus::tuple_element_t<4, DTYPE>;
    static_assert(std::is_same<D_A, fp8_t>::value && std::is_same<D_B, fp8_t>::value,
                  "PTPC bpreshuffle expects fp8 A/B");
    static_assert(std::is_same<D_C, bf16_t>::value, "PTPC bpreshuffle outputs bf16");
    static_assert(std::is_same<D_ACC, fp32_t>::value, "PTPC bpreshuffle accumulates fp32");
    static_assert(std::is_same<D_SF, fp32_t>::value, "PTPC scales are fp32 [M]/[N]");

    static constexpr int T_M = opus::get<0>(TILE{});
    static constexpr int T_N = opus::get<1>(TILE{});
    static constexpr int T_K = 1;

    static constexpr bool IS_P4D = std::is_same<PIPE, opus_ptpc_pipe_p4d>::value;
    static constexpr bool IS_P8  = std::is_same<PIPE, opus_ptpc_pipe_p8>::value;
    static_assert(IS_P4D || IS_P8, "unknown PIPE tag");
    static_assert(!IS_P4D || (BLOCK_SIZE == 256 && T_M == 1 && T_N == 4),
                  "P4D is 4-wave direct-register: 256 threads, T_M=1, T_N=4");
    static_assert(!IS_P8 || (BLOCK_SIZE == 512 && T_M == 2 && T_N == 4),
                  "P8 is the real 8-wave pipeline: 512 threads, T_M=2, T_N=4");

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__)
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K,
                  "wave mapping must cover BLOCK_SIZE");
#endif
    static_assert(T_K == 1);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;

    static_assert(!IS_P4D || B_M % (W_M * T_M) == 0);
    static_assert(!IS_P4D || B_N % (W_N * T_N) == 0);
    static_assert(!IS_P8 || HALF_B_M % (W_M * T_M) == 0);
    static_assert(!IS_P8 || HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = (IS_P8 ? HALF_B_M : B_M) / (W_M * T_M);
    static constexpr int E_N = (IS_P8 ? HALF_B_N : B_N) / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});
    static_assert(VEC_A == 16 && VEC_B == 16, "fp8 dwordx4 (16B) operand chunks");
    static_assert(VEC_C == 4, "16x16 fp32 C fragment is 4 registers");

    static constexpr int A_LEN = W_M * W_K / 64;
    static constexpr int B_LEN = W_N * W_K / 64;
    static constexpr int C_LEN = W_M * W_N / 64;

    static constexpr int PREFETCH = PREFETCH_;
    static constexpr int WG_PER_CU = WG_PER_CU_;

    static constexpr bool ALIGNED_AB = ALIGNED_AB_;
    static constexpr bool ALIGNED_C  = ALIGNED_C_;
    static constexpr int ab_issue_vec = ALIGNED_AB ? 16 : 1;

    static constexpr int a_loads_per_tile = E_M * E_K * 2 * (ALIGNED_AB ? 1 : 16);
    static constexpr int b_loads_per_tile = E_N * E_K * 2 * (ALIGNED_AB ? 1 : 16);
    static constexpr int loads_per_tile = a_loads_per_tile + b_loads_per_tile;

    static_assert(!IS_P8 || HALF_B_M % (W_M * T_M) == 0);
    static_assert(!IS_P8 || HALF_B_N % (W_N * T_N) == 0);

    static constexpr int smem_linear_wave = opus::get_warp_size() * 16 / sizeof(D_A);
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = IS_P8 ? (HALF_B_M / smem_sub) : 0;
    static constexpr int smem_padding = 2 * 16 / sizeof(D_A);
    static constexpr int smem_a_byte = IS_P8 ? (smem_m_rep * (smem_linear_wave + smem_padding)) : 0;

    static constexpr int smem_b_byte = IS_P8 ? ((HALF_B_N / 16) * 2048) : 0;

    static constexpr int a_buffer_load_insts = IS_P8 ? (HALF_B_M * B_K / (BLOCK_SIZE * ab_issue_vec)) : 0;
    static constexpr int b_buffer_load_insts = IS_P8 ? (HALF_B_N * B_K / (BLOCK_SIZE * ab_issue_vec)) : 0;
    static constexpr int a_ds_read_insts = IS_P8 ? ((E_M * E_K * W_M * W_K) / (64 * VEC_A)) : 0;
    static constexpr int b_ds_read_insts = IS_P8 ? ((E_N * E_K * W_N * W_K) / (64 * VEC_B)) : 0;
};

struct opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    const void* __restrict__ ptr_sa;
    const void* __restrict__ ptr_sb;
    void* __restrict__ ptr_c;
    void* __restrict__ ptr_ws;
    int m;
    int n;
    int k;
    int stride_a;
    int stride_c;
    int split_k;
    int iters_per_split;
    int padded_m;
    int padded_n;
};
