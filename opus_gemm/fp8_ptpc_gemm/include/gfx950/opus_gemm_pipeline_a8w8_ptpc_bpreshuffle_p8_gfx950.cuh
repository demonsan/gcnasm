// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_gemm_traits_a8w8_ptpc_bpreshuffle_gfx950.cuh"

#define OPUS_PTPC_OOB_SENTINEL ((int)0x80000000)

#ifdef __HIP_DEVICE_COMPILE__

template<typename T>
struct opus_ptpc_ab_geom {

    static constexpr int A_BLOCK = 1056;

    static __device__ int b_chunk_c0(int lane_id) {
        const int lg = lane_id / 16;
        const int lr = lane_id % 16;
        return (lg / 2) * 512 + (lg % 2) * 256 + lr * 16;
    }

    template<typename G>
    static __device__ void issue_a(G& g_a, void* smem_slot, int tid,
                                   int half_m, int t, int k_tile_lo, int k_end_s,
                                   int stride_a, int half_rows) {
        if (T::ALIGNED_AB) {
            opus::static_ford<T::a_buffer_load_insts>([&](auto i) {
                const int L    = (decltype(i)::value * T::BLOCK_SIZE + tid) * 16;
                const int row  = L / T::B_K;
                const int k16  = (L % T::B_K) / 16;
                const int smem = (row / 8) * A_BLOCK + (row % 8) * T::B_K + (L % T::B_K);
                const int va_lane = (half_m * half_rows + row) * stride_a + k16 * 16;
                const int s_t  = (k_tile_lo + t) * T::B_K;
                const int kk   = s_t + k16 * 16;
                const int v_os = (kk < k_end_s) ? va_lane : OPUS_PTPC_OOB_SENTINEL;
                g_a.template async_load<16>(
                    reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_slot) + smem),
                    v_os, s_t);
            });
        } else {
            opus::static_ford<T::HALF_B_M * T::B_K / (T::BLOCK_SIZE * 16)>([&](auto i) {
                const int L    = (decltype(i)::value * T::BLOCK_SIZE + tid) * 16;
                const int row  = L / T::B_K;
                const int k16  = (L % T::B_K) / 16;
                const int smem = (row / 8) * A_BLOCK + (row % 8) * T::B_K + (L % T::B_K);
                const int va_lane = (half_m * half_rows + row) * stride_a + k16 * 16;
                const int s_t  = (k_tile_lo + t) * T::B_K;
                const int kk   = s_t + k16 * 16;
                opus::vector_t<typename G::scalar_type, 16> chunk;
                opus::static_ford<16>([&](auto j) {
                    chunk[decltype(j)::value] =
                        g_a.template load<1>((kk < k_end_s) ? (va_lane + s_t + decltype(j)::value)
                                                            : OPUS_PTPC_OOB_SENTINEL)[0];
                });
                if (kk >= k_end_s) chunk = decltype(chunk){0};
                *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<typename G::scalar_type, 16>*>(
                    reinterpret_cast<__UINTPTR_TYPE__>(smem_slot) + smem) = chunk;
            });
        }
    }

    template<typename G>
    static __device__ void issue_b(G& g_b, void* smem_slot, int tid,
                                   int n_blk_base, int t, int k_tile_lo, int k_end_s,
                                   int k_total, int w_cols) {
        if (T::ALIGNED_AB) {
            opus::static_ford<T::b_buffer_load_insts>([&](auto i) {
                const int L     = (decltype(i)::value * T::BLOCK_SIZE + tid) * 16;
                const int j     = L >> 11;
                const int rem   = L & 2047;
                const int kblk  = rem >> 9;
                const int rem2  = rem & 511;
                const int hf    = rem2 >> 8;
                const int row16 = (rem2 & 255) >> 4;
                const int vb_lane = (n_blk_base + j) * (16 * k_total) + kblk * 512 + hf * 256 + row16 * 16;
                const int s_t  = (k_tile_lo + t) * 2048;
                const int kk   = (k_tile_lo + t) * T::B_K + kblk * 32 + hf * 16;
                const bool valid = (kk < k_end_s) && ((n_blk_base + j) * 16 < w_cols);
                const int v_os = valid ? vb_lane : OPUS_PTPC_OOB_SENTINEL;
                g_b.template async_load<16>(
                    reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_slot) + L),
                    v_os, s_t);
            });
        } else {
            opus::static_ford<T::HALF_B_N * T::B_K / (T::BLOCK_SIZE * 16)>([&](auto i) {
                const int L     = (decltype(i)::value * T::BLOCK_SIZE + tid) * 16;
                const int j     = L >> 11;
                const int rem   = L & 2047;
                const int kblk  = rem >> 9;
                const int rem2  = rem & 511;
                const int hf    = rem2 >> 8;
                const int row16 = (rem2 & 255) >> 4;
                const int vb_lane = (n_blk_base + j) * (16 * k_total) + kblk * 512 + hf * 256 + row16 * 16;
                const int s_t  = (k_tile_lo + t) * 2048;
                const int kk   = (k_tile_lo + t) * T::B_K + kblk * 32 + hf * 16;
                const bool valid = (kk < k_end_s) && ((n_blk_base + j) * 16 < w_cols);
                opus::vector_t<typename G::scalar_type, 16> chunk;
                opus::static_ford<16>([&](auto jx) {
                    chunk[decltype(jx)::value] =
                        g_b.template load<1>(valid ? (vb_lane + s_t + decltype(jx)::value)
                                                  : (long)OPUS_PTPC_OOB_SENTINEL)[0];
                });
                if (!valid) chunk = decltype(chunk){0};
                *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<typename G::scalar_type, 16>*>(
                    reinterpret_cast<__UINTPTR_TYPE__>(smem_slot) + L) = chunk;
            });
        }
    }

    template<bool MASK, typename SMEM, typename VA>
    static __device__ void read_a(SMEM& slot, VA& v_a, int lane_id, int wave_id_m,
                                  int t, int k_tile_lo, int k_end_s) {
        const int lg = lane_id / 16;
        const int lr = lane_id % 16;
        opus::static_ford<T::E_M, T::E_K>([&](auto i_m, auto i_k) {
            constexpr int e_m = decltype(i_m)::value;
            constexpr int e_k = decltype(i_k)::value;
            const int row  = (e_m * T::T_M + wave_id_m) * 16 + lr;
            const int base = (row / 8) * A_BLOCK + (row % 8) * T::B_K + lg * 16;
            auto c0 = slot.template load<T::VEC_A>(base);
            auto c1 = slot.template load<T::VEC_A>(base + 64);
            if (MASK) {
                const int k0 = (k_tile_lo + t) * T::B_K + e_k * T::W_K + lg * 16;
                const int k1 = k0 + 64;
                auto z0 = __builtin_bit_cast(opus::vector_t<opus::u32_t, 4>, c0);
                auto z1 = __builtin_bit_cast(opus::vector_t<opus::u32_t, 4>, c1);
                if (k0 >= k_end_s) z0 = decltype(z0){0};
                if (k1 >= k_end_s) z1 = decltype(z1){0};
                c0 = __builtin_bit_cast(decltype(c0), z0);
                c1 = __builtin_bit_cast(decltype(c1), z1);
            }
            opus::set_slice(v_a, c0, opus::number<(e_m * T::E_K + e_k) * T::A_LEN>{},
                              opus::number<(e_m * T::E_K + e_k) * T::A_LEN + 16>{});
            opus::set_slice(v_a, c1, opus::number<(e_m * T::E_K + e_k) * T::A_LEN + 16>{},
                              opus::number<(e_m * T::E_K + e_k) * T::A_LEN + 32>{});
        });
    }

    template<bool MASK, typename SMEM, typename VB>
    static __device__ void read_b(SMEM& slot, VB& v_b, int lane_id, int wave_id_n,
                                  int t, int k_tile_lo, int k_end_s, int w_cols) {
        const int base0 = b_chunk_c0(lane_id);
        const int lg = lane_id / 16;
        opus::static_ford<T::E_N, T::E_K>([&](auto i_n, auto i_k) {
            constexpr int e_n = decltype(i_n)::value;
            constexpr int e_k = decltype(i_k)::value;
            const int j_w = e_n * T::T_N + wave_id_n;
            static_assert(T::E_K == 1, "P8 fixes B_K = 128 (one 128-K step per tile)");
            const int base = j_w * 2048 + base0;
            auto c0 = slot.template load<T::VEC_B>(base);
            auto c1 = slot.template load<T::VEC_B>(base + 1024);
            if (MASK) {
                const int k0 = (k_tile_lo + t) * T::B_K + lg * 16;
                const int k1 = k0 + 64;
                const bool n_valid = (j_w * 16) < w_cols;
                auto z0 = __builtin_bit_cast(opus::vector_t<opus::u32_t, 4>, c0);
                auto z1 = __builtin_bit_cast(opus::vector_t<opus::u32_t, 4>, c1);
                if (k0 >= k_end_s || !n_valid) z0 = decltype(z0){0};
                if (k1 >= k_end_s || !n_valid) z1 = decltype(z1){0};
                c0 = __builtin_bit_cast(decltype(c0), z0);
                c1 = __builtin_bit_cast(decltype(c1), z1);
            }
            opus::set_slice(v_b, c0, opus::number<(e_n * T::E_K + e_k) * T::B_LEN>{},
                              opus::number<(e_n * T::E_K + e_k) * T::B_LEN + 16>{});
            opus::set_slice(v_b, c1, opus::number<(e_n * T::E_K + e_k) * T::B_LEN + 16>{},
                              opus::number<(e_n * T::E_K + e_k) * T::B_LEN + 32>{});
        });
    }
};
#endif

template<typename Traits, bool EDGE>
__device__ __forceinline__ void p8_body(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 kargs,
                                        char* smem_a, char* smem_b) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    static_assert(T::IS_P8, "P8 kernel requires the P8 pipe tag");

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
        auto store_c = [&](auto& vc, int half_tile_m, int half_tile_n) {
            const int m_base = row_wg + half_tile_m * T::HALF_B_M;
            const int n_base = col_wg + half_tile_n * T::HALF_B_N;
            auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c)
                                 + (size_t)m_base * kargs.stride_c + n_base);
            vector_t<bf16_t, nelem> out;
            constexpr auto issue_space = layout_to_issue_space<decltype(u_gc)>();
            static_ford(issue_space, [&](auto... ids) {
                constexpr index_t idx = make_layout<-1>(issue_space)(ids...);
                const float sa = g_sa.template load<1>(half_tile_m * T::HALF_B_M + u_gc_m(ids...))[0];
                const float sb = g_sb.template load<1>(half_tile_n * T::HALF_B_N + u_gc_n(ids...))[0];
                out[idx] = static_cast<bf16_t>((vc[idx] * sa) * sb);
            });
            auto pred = [&](auto... ids) {
                return (m_base + u_gc_m(ids...)) < kargs.m && (n_base + u_gc_n(ids...)) < kargs.n;
            };
            store_if<1>(g_c, pred, out, u_gc, 0);
        };
        store_c(v_c[0][0], 0, 0);
        store_c(v_c[0][1], 0, 1);
        store_c(v_c[1][0], 1, 0);
        store_c(v_c[1][1], 1, 1);
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
void gemm_a8w8_ptpc_bpreshuffle_p8_gfx950(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 kargs) {
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
        p8_body<Traits, true>(kargs, smem_a, smem_b);
    } else {
        const bool full = edge_wg || (kargs.k % T::B_K != 0) || ((n_s_wg & 1) != 0);
        if (full) p8_body<Traits, true>(kargs, smem_a, smem_b);
        else      p8_body<Traits, false>(kargs, smem_a, smem_b);
    }
#endif
#endif
}
