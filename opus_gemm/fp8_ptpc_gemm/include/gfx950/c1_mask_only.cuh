// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_gemm_traits_a8w8_ptpc_bpreshuffle_gfx950.cuh"

#define OPUS_PTPC_OOB_SENTINEL ((int)0x80000000)

template<typename Traits, bool MASK>
__device__ __forceinline__ void p4d_c1_mask_body(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::IS_P4D, "P4D kernel requires the P4D pipe tag");

    static_assert(T::ALIGNED_AB, "c1-mask-only is aligned-only");
    static_assert(T::B_M == 16 && T::B_N == 64 && T::B_K == 256 && T::PREFETCH == 2,
                  "c1-mask-only is C1 PF2/B_K256/E geometry only");
    using D_A   = typename T::D_A;
    using D_B   = typename T::D_B;
    using D_C   = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    using D_SF  = typename T::D_SF;

    const int num_tiles_n = ceil_div_constexpr(kargs.n, T::B_N);
    const int wgid_full = opus::block_id_x();
    const int split_id  = wgid_full % kargs.split_k;
    const int wgid      = wgid_full / kargs.split_k;
    const int row_wg    = (wgid / num_tiles_n) * T::B_M;
    const int col_wg    = (wgid % num_tiles_n) * T::B_N;

    const int t_total   = (kargs.k + T::B_K - 1) / T::B_K;
    const int ips       = kargs.iters_per_split;
    const int k_tile_lo = split_id * ips;
    const int rem_t     = t_total - k_tile_lo;
    const int my_loops  = rem_t > 0 ? (ips < rem_t ? ips : rem_t) : 0;
    const int k_end_s   = min((k_tile_lo + my_loops) * T::B_K, kargs.k);

    const int wave_id = opus::thread_id_x() / opus::get_warp_size();
    const int lane_id = opus::thread_id_x() % opus::get_warp_size();
    const int wave_id_n = wave_id;
    const int wave_id_m = 0;

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a0, v_a1;
    typename decltype(mma)::vtype_b v_b0, v_b1;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    const int rows_avail = min(T::B_M, kargs.m - row_wg);
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + (size_t)row_wg * kargs.stride_a,
                         (rows_avail - 1) * kargs.stride_a + kargs.k);
    const int w_cols  = min(T::B_N, kargs.n - col_wg);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + (size_t)(col_wg / 16) * (16L * kargs.k),
                         w_cols * kargs.k);

    const int lg = lane_id / 16;
    const int lr = lane_id % 16;
    const int b_chunk_c0 = (lg / 2) * 512 + (lg % 2) * 256 + lr * 16;
    const long b_n_stride = 16L * kargs.k;
    static_assert(T::E_N == 1, "P4D expects one 16-col B block per wave");

    auto issue_tile = [&](int t, auto& va, auto& vb) {
        const int tg = k_tile_lo + t;
        static_ford<T::E_M, T::E_K>([&](auto i_m, auto i_k) {
            constexpr int e_m = decltype(i_m)::value;
            constexpr int e_k = decltype(i_k)::value;
            const int k0 = tg * T::B_K + e_k * T::W_K + lg * 16;
            const int k1 = k0 + 64;
            const int a_base = (e_m * 16 + lr) * kargs.stride_a + tg * T::B_K + e_k * T::W_K + lg * 16;
            vector_t<D_A, 16> c0, c1;
            if (MASK) {
                c0 = g_a.template load<T::VEC_A>(k0 < k_end_s ? a_base : OPUS_PTPC_OOB_SENTINEL);
                c1 = g_a.template load<T::VEC_A>(k1 < k_end_s ? a_base + 64 : OPUS_PTPC_OOB_SENTINEL);
                if (k0 >= k_end_s) c0 = decltype(c0){0};
                if (k1 >= k_end_s) c1 = decltype(c1){0};
            } else {
                c0 = g_a.template load<T::VEC_A>(a_base);
                c1 = g_a.template load<T::VEC_A>(a_base + 64);
            }
            set_slice(va, c0, number<(e_m * T::E_K + e_k) * T::A_LEN>{},
                              number<(e_m * T::E_K + e_k) * T::A_LEN + 16>{});
            set_slice(va, c1, number<(e_m * T::E_K + e_k) * T::A_LEN + 16>{},
                              number<(e_m * T::E_K + e_k) * T::A_LEN + 32>{});
        });
        static_ford<T::E_K>([&](auto i_k) {
            constexpr int e_k = decltype(i_k)::value;
            const int k0 = tg * T::B_K + e_k * T::W_K + lg * 16;
            const int k1 = k0 + 64;
            const long b_base = (long)wave_id_n * b_n_stride
                              + (long)(tg * T::E_K + e_k) * 2048 + b_chunk_c0;
            vector_t<D_B, 16> c0, c1;
            if (MASK) {
                c0 = g_b.template load<T::VEC_B>((int)(k0 < k_end_s ? b_base : (long)OPUS_PTPC_OOB_SENTINEL));
                c1 = g_b.template load<T::VEC_B>((int)(k1 < k_end_s ? b_base + 1024 : (long)OPUS_PTPC_OOB_SENTINEL));
                if (k0 >= k_end_s) c0 = decltype(c0){0};
                if (k1 >= k_end_s) c1 = decltype(c1){0};
            } else {
                c0 = g_b.template load<T::VEC_B>((int)b_base);
                c1 = g_b.template load<T::VEC_B>((int)(b_base + 1024));
            }
            set_slice(vb, c0, number<e_k * T::B_LEN>{}, number<e_k * T::B_LEN + 16>{});
            set_slice(vb, c1, number<e_k * T::B_LEN + 16>{}, number<e_k * T::B_LEN + 32>{});
        });
    };

    if (my_loops > 0) issue_tile(0, v_a0, v_b0);
    for (int t = 0; t < my_loops; t++) {

        const bool full_wait = MASK || (t + 1 >= my_loops);
        if (t & 1) {
            if (t + 1 < my_loops) issue_tile(t + 1, v_a0, v_b0);
            if (full_wait) s_waitcnt_vmcnt(0_I);
            else           s_waitcnt_vmcnt(number<T::loads_per_tile>{});
            v_c = mma(v_a1, v_b1, v_c, 127, 127);
        } else {
            if (t + 1 < my_loops) issue_tile(t + 1, v_a1, v_b1);
            if (full_wait) s_waitcnt_vmcnt(0_I);
            else           s_waitcnt_vmcnt(number<T::loads_per_tile>{});
            v_c = mma(v_a0, v_b0, v_c, 127, 127);
        }
    }

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      wave_id_n, lane_id / mma.grpn_c);
    if (kargs.split_k == 1) {

        auto u_gc   = partition_layout_c<1>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
        auto u_gc_m = partition_layout_c<1>(mma, opus::make_tuple(1_I, 0_I), p_coord_c);
        auto u_gc_n = partition_layout_c<1>(mma, opus::make_tuple(0_I, 1_I), p_coord_c);
        auto g_c    = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c)
                              + (size_t)row_wg * kargs.stride_c + col_wg);
        auto g_sa   = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sa) + row_wg,
                                (kargs.m - row_wg) * (int)sizeof(D_SF));
        auto g_sb   = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sb) + col_wg,
                                (kargs.n - col_wg) * (int)sizeof(D_SF));

        constexpr int nelem = T::E_M * T::E_N * T::C_LEN;
        vector_t<bf16_t, nelem> out;
        constexpr auto issue_space = layout_to_issue_space<decltype(u_gc)>();
        static_ford(issue_space, [&](auto... ids) {
            constexpr index_t idx = make_layout<-1>(issue_space)(ids...);
            const int m_rel = u_gc_m(ids...);
            const int n_rel = u_gc_n(ids...);
            const float sa = g_sa.template load<1>(m_rel)[0];
            const float sb = g_sb.template load<1>(n_rel)[0];
            out[idx] = static_cast<bf16_t>((v_c[idx] * sa) * sb);
        });
        auto pred = [&](auto... ids) {
            return (row_wg + u_gc_m(ids...)) < kargs.m && (col_wg + u_gc_n(ids...)) < kargs.n;
        };
        store_if<1>(g_c, pred, out, u_gc, 0);
    } else {

        auto u_ws = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.padded_n, 1_I), p_coord_c);
        auto g_ws = make_gmem(reinterpret_cast<D_ACC*>(kargs.ptr_ws)
                            + (size_t)split_id * kargs.padded_m * kargs.padded_n
                            + (size_t)row_wg * kargs.padded_n + col_wg);
        store<T::VEC_C>(g_ws, v_c, u_ws, 0);
    }
#endif
#endif
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_ptpc_c1_mask_gfx950(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::IS_P4D, "P4D kernel requires the P4D pipe tag");
    const int num_tiles_n = ceil_div_constexpr(kargs.n, T::B_N);
    const int wgid_full = opus::block_id_x();
    const int wgid      = wgid_full / kargs.split_k;
    const int row_wg    = (wgid / num_tiles_n) * T::B_M;
    const int col_wg    = (wgid % num_tiles_n) * T::B_N;
    const bool full_w   = (kargs.k % T::B_K != 0)
                      || (row_wg + T::B_M > kargs.m) || (col_wg + T::B_N > kargs.n);
    if (full_w) p4d_c1_mask_body<Traits, true>(kargs);
    else        p4d_c1_mask_body<Traits, false>(kargs);
#endif
#endif
}
