// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_gemm_traits_a8w8_ptpc_bpreshuffle_gfx950.cuh"

#define OPUS_PTPC_OOB_SENTINEL ((int)0x80000000)

template<typename Traits, bool EDGE>
__device__ __forceinline__ void p8_epi_body(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 kargs,
                                            char* smem_a, char* smem_b) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::IS_P8, "P8 kernel requires the P8 pipe tag");

    static_assert(T::ALIGNED_AB, "EPI-C is aligned-only");
    static_assert(T::B_M == 128 && T::B_N == 256 && T::B_K == 128 && T::PREFETCH == 2
                  && T::BLOCK_SIZE == 512, "EPI-C is C3-A 128x256x128 PF2 512-thread only");

    static_assert(EDGE || T::ALIGNED_AB, "counted body is aligned-only");
    static_assert(T::E_K == 1, "P8 fixes B_K = 128 per K-tile");
    using D_A   = typename T::D_A;
    using D_B   = typename T::D_B;
    using D_C   = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    using D_SF  = typename T::D_SF;
    using GEOM  = opus_ptpc_ab_geom<T>;

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
    const int n_s       = rem_t > 0 ? (ips < rem_t ? ips : rem_t) : 0;
    const int loops     = n_s < 2 ? 2 : ((n_s + 1) & ~1);
    const int k_end_s   = min((k_tile_lo + n_s) * T::B_K, kargs.k);

    const int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    const int lane_id = opus::thread_id_x() % get_warp_size();
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;

    const int rows_avail = min(T::B_M, kargs.m - row_wg);
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + (size_t)row_wg * kargs.stride_a,
                         (rows_avail - 1) * kargs.stride_a + kargs.k);
    const int w_cols  = min(T::B_N, kargs.n - col_wg);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + (size_t)(col_wg / 16) * (16L * kargs.k),
                         w_cols * kargs.k);

    smem<D_A> s_a[2][2] = {
        {make_smem(reinterpret_cast<D_A*>(smem_a)),
         make_smem(reinterpret_cast<D_A*>(smem_a + T::smem_a_byte))},
        {make_smem(reinterpret_cast<D_A*>(smem_a + 2 * T::smem_a_byte)),
         make_smem(reinterpret_cast<D_A*>(smem_a + 3 * T::smem_a_byte))}
    };
    smem<D_B> s_b[2][2] = {
        {make_smem(reinterpret_cast<D_B*>(smem_b)),
         make_smem(reinterpret_cast<D_B*>(smem_b + T::smem_b_byte))},
        {make_smem(reinterpret_cast<D_B*>(smem_b + 2 * T::smem_b_byte)),
         make_smem(reinterpret_cast<D_B*>(smem_b + 3 * T::smem_b_byte))}
    };

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b[2];
    typename decltype(mma)::vtype_c v_c[2][2];
    clear(v_c[0][0]);
    clear(v_c[0][1]);
    clear(v_c[1][0]);
    clear(v_c[1][1]);

    const int tid = opus::thread_id_x();

    auto wait_v = [&](auto cnt) {
        if (EDGE) { s_waitcnt_vmcnt(0_I); s_waitcnt_lgkmcnt(0_I); }
        else      s_waitcnt_vmcnt(cnt);
    };
    auto wait_l = [&](auto cnt) {
        if (EDGE) s_waitcnt_lgkmcnt(0_I);
        else      s_waitcnt_lgkmcnt(cnt);
    };
    auto full_barrier = [&]() {
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };

    auto issue_a = [&](int half_m, int t_local, int slot) {
        GEOM::issue_a(g_a, s_a[slot][half_m].ptr, tid, half_m, t_local, k_tile_lo,
                      k_end_s, kargs.stride_a, T::HALF_B_M);
    };
    auto issue_b = [&](int half_n, int t_local, int slot) {

        GEOM::issue_b(g_b, s_b[slot][half_n].ptr, tid,
                      half_n * (T::HALF_B_N / 16),
                      t_local, k_tile_lo, k_end_s, kargs.k, w_cols);
    };
    auto read_a = [&](auto mask_tag, int half_m, int t_local, int slot) {
        GEOM::template read_a<decltype(mask_tag)::value>(s_a[slot][half_m], v_a, lane_id, wave_id_m, t_local, k_tile_lo, k_end_s);
    };
    auto read_b = [&](auto mask_tag, int half_n, int t_local, int slot, auto& vb) {
        GEOM::template read_b<decltype(mask_tag)::value>(s_b[slot][half_n], vb, lane_id, wave_id_n, t_local, k_tile_lo, k_end_s, w_cols);
    };

    int tic = 0, toc = 1;

    issue_b(0, 0, tic);
    issue_a(0, 0, tic);
    issue_b(1, 0, tic);
    issue_a(1, 0, tic);

    wait_v(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    issue_b(0, 1, toc);
    issue_a(0, 1, toc);
    issue_b(1, 1, toc);

    wait_v(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    read_b(number<true>{}, 0, 0, tic, v_b[0]);
    __builtin_amdgcn_s_barrier();

    for (int tile = 0; tile < loops - 2; tile += 2) {

        read_a(number<false>{}, 0, tile, tic);
        issue_a(1, tile + 1, toc);
        wait_l(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_b(number<false>{}, 1, tile, tic, v_b[1]);
        issue_b(0, tile + 2, tic);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_a(number<false>{}, 1, tile, tic);
        issue_a(0, tile + 2, tic);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        issue_b(1, tile + 2, tic);
        wait_v(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        read_b(number<false>{}, 0, tile + 1, toc, v_b[0]);
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_a(number<false>{}, 0, tile + 1, toc);
        issue_a(1, tile + 2, tic);
        wait_l(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_b(number<false>{}, 1, tile + 1, toc, v_b[1]);
        issue_b(0, tile + 3, toc);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_a(number<false>{}, 1, tile + 1, toc);
        issue_a(0, tile + 3, toc);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        issue_b(1, tile + 3, toc);
        wait_v(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        read_b(number<false>{}, 0, tile + 2, tic, v_b[0]);
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    {
        int tile = loops - 2;

        read_a(number<true>{}, 0, tile, tic);
        issue_a(1, tile + 1, toc);

        read_b(number<true>{}, 0, tile, tic, v_b[0]);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);

        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_b(number<true>{}, 1, tile, tic, v_b[1]);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_a(number<true>{}, 1, tile, tic);
        full_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0], 127, 127);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1;
        toc ^= 1;
    }

    {
        read_b(number<true>{}, 0, loops - 1, tic, v_b[0]);
        read_a(number<true>{}, 0, loops - 1, tic);
        full_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_b(number<true>{}, 1, loops - 1, tic, v_b[1]);
        full_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        read_a(number<true>{}, 1, loops - 1, tic);
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0], 127, 127);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1], 127, 127);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      wave_id_n, lane_id / mma.grpn_c);
    if (kargs.split_k == 1) {

        auto u_gc   = partition_layout_c<1>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
        auto u_gc_m = partition_layout_c<1>(mma, opus::make_tuple(1_I, 0_I), p_coord_c);
        auto u_gc_n = partition_layout_c<1>(mma, opus::make_tuple(0_I, 1_I), p_coord_c);
        auto g_sa   = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sa) + row_wg,
                                (kargs.m - row_wg) * (int)sizeof(D_SF));
        auto g_sb   = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sb) + col_wg,
                                (kargs.n - col_wg) * (int)sizeof(D_SF));

        constexpr int nelem = T::E_M * T::E_N * T::C_LEN;
        constexpr auto issue_space = layout_to_issue_space<decltype(u_gc)>();
        bf16_t* epi_smem = reinterpret_cast<bf16_t*>(smem_b);

        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        static_ford<4>([&](auto q) {
            constexpr int hm = decltype(q)::value >> 1;
            constexpr int hn = decltype(q)::value & 1;
            constexpr int buf = decltype(q)::value & 1;

            vector_t<bf16_t, nelem> stage;
            static_ford(issue_space, [&](auto... ids) {
                constexpr index_t idx = make_layout<-1>(issue_space)(ids...);
                const float sa = g_sa.template load<1>(hm * T::HALF_B_M + u_gc_m(ids...))[0];
                const float sb = g_sb.template load<1>(hn * T::HALF_B_N + u_gc_n(ids...))[0];
                stage[idx] = static_cast<bf16_t>((v_c[hm][hn][idx] * sa) * sb);
            });

            auto epi_sm = make_smem(reinterpret_cast<bf16_t*>(smem_b) + buf * 8704);
            auto u_epi = partition_layout_c<1>(mma, opus::make_tuple(136, 1_I), p_coord_c);
            static_ford(issue_space, [&](auto... ids) {
                constexpr index_t idx = make_layout<-1>(issue_space)(ids...);
                epi_sm.template store<1>(stage[idx], u_epi(ids...));
            });
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();

            const int m_base_q = row_wg + hm * T::HALF_B_M;
            const int n_base_q = col_wg + hn * T::HALF_B_N;

            auto g_cq = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c)
                                  + (size_t)m_base_q * kargs.stride_c + n_base_q);
            static_ford<2>([&](auto r) {
                const int lin = (decltype(r)::value * T::BLOCK_SIZE + tid) * 8;
                const int row = lin / 128;
                const int col = lin % 128;
                auto rd = epi_sm.template load<8>(row * 136 + col);
                if ((m_base_q + row) < kargs.m && (n_base_q + col) < kargs.n) {

                    g_cq.template store<8>(rd, row * kargs.stride_c + col);
                }
            });
            s_waitcnt_lgkmcnt(0_I);
        });
    } else {

        auto u_ws = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.padded_n, 1_I), p_coord_c);
        auto g_ws = make_gmem(reinterpret_cast<D_ACC*>(kargs.ptr_ws)
                            + (size_t)split_id * kargs.padded_m * kargs.padded_n
                            + (size_t)row_wg * kargs.padded_n + col_wg);
        auto c_offset_ws = [&](int half_tile_m, int half_tile_n) {
            return half_tile_m * T::HALF_B_M * kargs.padded_n + half_tile_n * T::HALF_B_N;
        };
        store<T::VEC_C>(g_ws, v_c[0][0], u_ws, c_offset_ws(0, 0));
        store<T::VEC_C>(g_ws, v_c[0][1], u_ws, c_offset_ws(0, 1));
        store<T::VEC_C>(g_ws, v_c[1][0], u_ws, c_offset_ws(1, 0));
        store<T::VEC_C>(g_ws, v_c[1][1], u_ws, c_offset_ws(1, 1));
    }
#else

#endif
#endif
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_ptpc_c3_epi_gfx950(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::IS_P8, "P8 kernel requires the P8 pipe tag");
    const int num_tiles_n = ceil_div_constexpr(kargs.n, T::B_N);
    const int wgid_full = opus::block_id_x();
    const int split_id  = wgid_full % kargs.split_k;
    const int wgid      = wgid_full / kargs.split_k;
    const int row_wg    = (wgid / num_tiles_n) * T::B_M;
    const int col_wg    = (wgid % num_tiles_n) * T::B_N;
    const bool edge_wg  = (row_wg + T::B_M > kargs.m) || (col_wg + T::B_N > kargs.n);
    const int t_total   = (kargs.k + T::B_K - 1) / T::B_K;
    const int k_tile_lo = split_id * kargs.iters_per_split;
    const int rem_t     = t_total - k_tile_lo;
    const int n_s_wg    = rem_t > 0 ? (kargs.iters_per_split < rem_t ? kargs.iters_per_split : rem_t) : 0;
    __shared__ char smem_a[T::smem_a_byte * 4];
    __shared__ char smem_b[T::smem_b_byte * 4];
    if constexpr (!T::ALIGNED_AB) {
        p8_epi_body<Traits, true>(kargs, smem_a, smem_b);
    } else {
        const bool full = edge_wg || (kargs.k % T::B_K != 0) || ((n_s_wg & 1) != 0);
        if (full) p8_epi_body<Traits, true>(kargs, smem_a, smem_b);
        else      p8_epi_body<Traits, false>(kargs, smem_a, smem_b);
    }
#endif
#endif
}
