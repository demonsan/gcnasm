// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_gemm_utils.cuh"

template<int VEC_ = 16, int BLOCK_ = 64, typename D_OUT = __bf16, bool ALIGNED_C_ = true>
__global__ void splitk_reduce_ptpc_gfx950(
    const void*   __restrict__ ws_ptr,
    const float*  __restrict__ sa_ptr,
    const float*  __restrict__ sb_ptr,
    D_OUT*        __restrict__ c_out,
    int split_k, int M, int N, int stride_c,
    int padded_M, int padded_N)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using D_WS = float;
    constexpr int VEC = VEC_;
    constexpr int BLOCK = BLOCK_;
    constexpr int STEP = 16 / (int)sizeof(D_OUT);
    static_assert(STEP * (int)sizeof(D_OUT) == 16);
    static_assert(VEC % STEP == 0);

    const D_WS* __restrict__ workspace = reinterpret_cast<const D_WS*>(ws_ptr);
    const int m      = int(opus::block_id_y());
    const int tid    = int(opus::thread_id_x());
    const int n_base = (int(opus::block_id_x()) * BLOCK + tid) * VEC;

    float sb_v[VEC];
    auto g_sb = opus::make_gmem(sb_ptr, (unsigned int)((size_t)N * sizeof(float)));
    #pragma unroll
    for (int g = 0; g < VEC / 4; ++g) {
        auto v4 = g_sb.template load<4>(n_base + g * 4);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            sb_v[g * 4 + j] = (n_base + g * 4 + j) < N ? v4[j] : 0.0f;
    }
    auto g_sa = opus::make_gmem(sa_ptr + m, (unsigned int)sizeof(float));
    const float sa = g_sa.template load<1>(0)[0];

    const long split_stride = (long)padded_M * padded_N;
    const long ws_row       = (long)m * padded_N + n_base;
    auto g_ws = opus::make_gmem(workspace,
                                (unsigned int)((size_t)split_k * split_stride * sizeof(D_WS)));
    float acc[VEC];
    #pragma unroll
    for (int t = 0; t < VEC; ++t) acc[t] = 0.0f;
    for (int s = 0; s < split_k; ++s) {
        const long ws_idx = ws_row + (long)s * split_stride;
        #pragma unroll
        for (int g = 0; g < VEC / 4; ++g) {
            auto v4 = g_ws.template load<4>((int)(ws_idx + g * 4));
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[g * 4 + j] += v4[j];
        }
    }

    D_OUT out[VEC];
    #pragma unroll
    for (int t = 0; t < VEC; ++t) out[t] = static_cast<D_OUT>((acc[t] * sa) * sb_v[t]);

    auto g_c = opus::make_gmem(c_out,
                               (unsigned int)(((size_t)(M - 1) * stride_c + N) * sizeof(D_OUT)));
    const int c_idx = m * stride_c + n_base;

    if (ALIGNED_C_ && n_base + VEC <= N) {
        #pragma unroll
        for (int g = 0; g < VEC / STEP; ++g) {
            opus::vector_t<D_OUT, STEP> chunk;
            #pragma unroll
            for (int j = 0; j < STEP; ++j) chunk[j] = out[g * STEP + j];
            g_c.template store<STEP>(chunk, c_idx + g * STEP);
        }
    } else if (n_base < N) {
        #pragma unroll
        for (int t = 0; t < VEC; ++t) {
            if (n_base + t < N) g_c.template store<1>(out[t], c_idx + t);
        }
    }

#endif
#endif
}
