#pragma once

// Full direct-read pipeline copy derived from a2a_gemm_kernel_template.hpp.
// Mode is fixed to 3: no dedicated communication workgroups; each compute WG
// reads A shards directly from the rotated rank's LSA window.

#include "gemm_defs.h"
#include "../../../opus_dist_gemm/gemm_a16w16_quad_subtile_kernel_template.hpp"

namespace a2a_gemm_direct_read {

using opus::operator""_I;
static constexpr int kRotatePutWgs = 256;
static constexpr int kRotatePutWgStride = 1;

__device__ inline unsigned flat_load_u32(const unsigned* ptr) {
    return *reinterpret_cast<const volatile unsigned*>(ptr);
}

template<typename T>
__device__ inline void warmup_remote_shard(opus_a2a_gemm_kargs kargs, int part, int tid, int wave_id) {
    if (part == kargs.my_rank) {
        return;
    }
    if (wave_id != 0) {
        constexpr size_t kWarmupStrideBytes = 2ULL * 1024ULL * 1024ULL;
        const size_t shard_bytes = static_cast<size_t>(kargs.m) * kargs.k_shard * sizeof(typename T::D_A);
        const char* peer = static_cast<const char*>(cco_lsa_peer_c(kargs.recv_a_win, part));
        const char* shard_base = peer + static_cast<size_t>(part) * shard_bytes;
        const int warmup_thread = tid - opus::get_warp_size();
        const size_t offset = static_cast<size_t>(warmup_thread) * kWarmupStrideBytes;
        if (offset + sizeof(unsigned) <= shard_bytes) {
            const unsigned value = flat_load_u32(reinterpret_cast<const unsigned*>(shard_base + offset));
            asm volatile("" : : "v"(value));
        }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
}

__device__ inline void wait_rotate_ready(opus_a2a_gemm_kargs kargs, int part, int tid) {
    if (tid == 0 && part != kargs.my_rank) {
        auto* ready = static_cast<unsigned*>(kargs.ready_local);
        while (flat_load_u32(ready + part) == 0u) {
        }
    }
    __builtin_amdgcn_s_barrier();
}

__device__ inline void local_arrive_and_signal(opus_a2a_gemm_kargs kargs,
                                               int dst_rank,
                                               int part,
                                               int shard_iter,
                                               int tid) {
    if (tid == 0) {
        auto* local_counters = static_cast<unsigned*>(kargs.ready_local) + kargs.rank_count;
        const unsigned old = __atomic_fetch_add(local_counters + shard_iter, 1u, __ATOMIC_ACQ_REL);
        if (old + 1u == static_cast<unsigned>(kRotatePutWgs)) {
            auto* peer_ready = static_cast<unsigned*>(cco_lsa_peer_c(kargs.ready_win, dst_rank));
            __atomic_store_n(peer_ready + part, 1u, __ATOMIC_RELAXED);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
    }
}

}  // namespace a2a_gemm_direct_read

template<typename UserTraits>
__global__ __launch_bounds__(UserTraits::BLOCK_SIZE, 2)
void a2a_gemm_lsa_direct_read_kernel(opus_a2a_gemm_kargs kargs) {
    using namespace opus;
    using namespace gemm_quad_subtile;
    using namespace a2a_gemm_direct_read;
    using opus::operator""_I;
    using T = kernel_traits<opus::remove_cvref_t<UserTraits>>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    const int bx = opus::block_id_x();
    const int tid = opus::thread_id_x();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int lane_id = tid % get_warp_size();

    using rotate_vec_t = unsigned int __attribute__((ext_vector_type(4)));
    auto rotate_put_shard = [&](int part, int shard_iter) {
        if ((bx % kRotatePutWgStride) != 0) {
            return;
        }
        const size_t shard_bytes = static_cast<size_t>(kargs.m) * kargs.k_shard * sizeof(D_A);
        const size_t vec_count = shard_bytes / sizeof(rotate_vec_t);
        const auto* src = reinterpret_cast<const rotate_vec_t*>(
            static_cast<const char*>(kargs.recv_a_local) + static_cast<size_t>(part) * shard_bytes);
        const int dst_rank = (kargs.my_rank - 1 + kargs.rank_count) & 7;
        char* peer = static_cast<char*>(cco_lsa_peer_c(kargs.recv_a_win, dst_rank));
        auto* dst = reinterpret_cast<rotate_vec_t*>(peer + static_cast<size_t>(part) * shard_bytes);
        const int put_wg = bx / kRotatePutWgStride;
        const size_t elems_per_put_wg = (vec_count + kRotatePutWgs - 1) / kRotatePutWgs;
        const size_t begin = static_cast<size_t>(put_wg) * elems_per_put_wg;
        const size_t end = begin + elems_per_put_wg < vec_count ? begin + elems_per_put_wg : vec_count;
        for (size_t i = begin + tid; i < end; i += T::BLOCK_SIZE) {
            dst[i] = src[i];
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        local_arrive_and_signal(kargs, dst_rank, part, shard_iter, tid);
    };

    constexpr int kNumMTiles = 2048 / T::B_M;
    constexpr int kNumNTiles = 8192 / T::B_N;
    static_assert(kNumMTiles == 8);
    static_assert(kNumNTiles == 32);
    static_assert((kNumNTiles & (kNumNTiles - 1)) == 0);

    const int compute_task = bx;
    constexpr int kTileCount = kNumMTiles * kNumNTiles;
    if (compute_task >= kTileCount) {
        return;
    }

    const int m_tile = compute_task >> 5;
    const int n_tile_initial = compute_task & (kNumNTiles - 1);

    const int row = m_tile * T::B_M;
    auto compute_one_n_tile = [&](int n_tile) {
    int col = n_tile * T::B_N;
    int k_base = 0;

    auto make_g_a = [&](int part) {
        const size_t shard_bytes = static_cast<size_t>(kargs.m) * kargs.k_shard * sizeof(D_A);
        const char* local = static_cast<const char*>(kargs.recv_a_local);
        const D_A* a_base = reinterpret_cast<const D_A*>(local + static_cast<size_t>(part) * shard_bytes);
        const unsigned int a_bytes = static_cast<unsigned int>((kargs.m - row) * kargs.stride_a * sizeof(D_A));
        return make_gmem(a_base + row * kargs.stride_a, a_bytes);
    };
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b + k_base,
                         static_cast<unsigned int>((kargs.n - col) * kargs.stride_b * sizeof(D_B)));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + row * kargs.stride_c,
                         kargs.output_bytes - static_cast<unsigned int>(row * kargs.stride_c * sizeof(D_C)));

    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 4];
    constexpr int smem_b_byte = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 4];
#define A_TILE(slot, half) make_smem(reinterpret_cast<D_A*>(smem_a + (((slot) << 1) + (half)) * smem_a_byte))
#define B_TILE(slot, half) make_smem(reinterpret_cast<D_B*>(smem_b + (((slot) << 1) + (half)) * smem_b_byte))
#define LOAD_A_TILE(slot, half) ([&]() { auto mem = A_TILE(slot, half); return load<T::VEC_A>(mem, u_ra); }())
#define LOAD_B_TILE(slot, half) ([&]() { auto mem = B_TILE(slot, half); return load<T::VEC_B>(mem, u_rb); }())

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
    constexpr int kFullLocalShardTiles = 1024 / T::B_K;
    static_assert((kFullLocalShardTiles & (kFullLocalShardTiles - 1)) == 0);
    auto ordered_part = [&](int tile_k) {
        return (kargs.my_rank + (tile_k >> 4)) & 7;
    };
    auto a_offset = [&](int half_tile_m, int tile_k) {
        const int shard_tile_k = tile_k & (kFullLocalShardTiles - 1);
        return half_tile_m * T::HALF_B_M * kargs.stride_a + shard_tile_k * T::B_K;
    };
    auto b_offset = [&](int half_tile_n, int tile_k) {
        const int shard_tile_k = tile_k & (kFullLocalShardTiles - 1);
        return half_tile_n * T::HALF_B_N * kargs.stride_b +
               ordered_part(tile_k) * kargs.k_shard + shard_tile_k * T::B_K;
    };

    const int shard_outer_loops = kargs.rank_count;
    const int shard_loops = kFullLocalShardTiles;
    int tic = 0, toc = 1;

    for (int shard_iter = 0; shard_iter < shard_outer_loops; ++shard_iter) {
    const int shard_base_tile = shard_iter * kFullLocalShardTiles;
    const int part = ordered_part(shard_base_tile);
    auto g_a = make_g_a(part);
    if (shard_iter != 0) {
        s_waitcnt_vmcnt(0_I);
        if (wave_id_m == 1) __builtin_amdgcn_s_barrier();
    }
    wait_rotate_ready(kargs, part, tid);
    warmup_remote_shard<T>(kargs, part, tid, wave_id);

    async_load<T::VEC_B>(g_b, B_TILE(tic, 0).ptr, u_gb, u_sb, b_offset(0, shard_base_tile + 0));
    async_load<T::VEC_A>(g_a, A_TILE(tic, 0).ptr, u_ga, u_sa, a_offset(0, shard_base_tile + 0));
    async_load<T::VEC_B>(g_b, B_TILE(tic, 1).ptr, u_gb, u_sb, b_offset(1, shard_base_tile + 0));
    async_load<T::VEC_A>(g_a, A_TILE(tic, 1).ptr, u_ga, u_sa, a_offset(1, shard_base_tile + 0));

    if (shard_iter == 0 && wave_id_m == 1) __builtin_amdgcn_s_barrier();

    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    async_load<T::VEC_B>(g_b, B_TILE(toc, 0).ptr, u_gb, u_sb, b_offset(0, shard_base_tile + 1));
    async_load<T::VEC_A>(g_a, A_TILE(toc, 0).ptr, u_ga, u_sa, a_offset(0, shard_base_tile + 1));
    async_load<T::VEC_B>(g_b, B_TILE(toc, 1).ptr, u_gb, u_sb, b_offset(1, shard_base_tile + 1));

    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    v_b[0] = LOAD_B_TILE(tic, 0);
    __builtin_amdgcn_s_barrier();

    for(int tile = 0; tile < shard_loops - 2; tile += 2) {
        v_a = LOAD_A_TILE(tic, 0);
        async_load<T::VEC_A>(g_a, A_TILE(toc, 1).ptr, u_ga, u_sa, a_offset(1, shard_base_tile + tile + 1));
        s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = LOAD_B_TILE(tic, 1);
        async_load<T::VEC_B>(g_b, B_TILE(tic, 0).ptr, u_gb, u_sb, b_offset(0, shard_base_tile + tile + 2));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        v_a = LOAD_A_TILE(tic, 1);
        async_load<T::VEC_A>(g_a, A_TILE(tic, 0).ptr, u_ga, u_sa, a_offset(0, shard_base_tile + tile + 2));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[0] = LOAD_B_TILE(toc, 0);
        async_load<T::VEC_B>(g_b, B_TILE(tic, 1).ptr, u_gb, u_sb, b_offset(1, shard_base_tile + tile + 2));
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        v_a = LOAD_A_TILE(toc, 0);
        async_load<T::VEC_A>(g_a, A_TILE(tic, 1).ptr, u_ga, u_sa, a_offset(1, shard_base_tile + tile + 2));
        s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = LOAD_B_TILE(toc, 1);
        async_load<T::VEC_B>(g_b, B_TILE(toc, 0).ptr, u_gb, u_sb, b_offset(0, shard_base_tile + tile + 3));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        v_a = LOAD_A_TILE(toc, 1);
        async_load<T::VEC_A>(g_a, A_TILE(toc, 0).ptr, u_ga, u_sa, a_offset(0, shard_base_tile + tile + 3));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[0] = LOAD_B_TILE(tic, 0);
        async_load<T::VEC_B>(g_b, B_TILE(toc, 1).ptr, u_gb, u_sb, b_offset(1, shard_base_tile + tile + 3));
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        int tile = shard_loops - 2;
        v_a = LOAD_A_TILE(tic, 0);
        async_load<T::VEC_A>(g_a, A_TILE(toc, 1).ptr, u_ga, u_sa, a_offset(1, shard_base_tile + tile + 1));
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        v_b[1] = LOAD_B_TILE(tic, 1);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        v_a = LOAD_A_TILE(tic, 1);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        tic ^= 1;
        toc ^= 1;
    }

    {
        v_b[0] = LOAD_B_TILE(tic, 0);
        v_a = LOAD_A_TILE(tic, 0);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        v_b[1] = LOAD_B_TILE(tic, 1);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        v_a = LOAD_A_TILE(tic, 1);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }
    if (shard_iter + 1 < shard_outer_loops) {
        s_waitcnt_vmcnt(0_I);
        if (wave_id_m == 0) __builtin_amdgcn_s_barrier();
        rotate_put_shard(part, shard_iter);
    }
    }

    if (wave_id_m == 0) __builtin_amdgcn_s_barrier();

    auto u_gc = make_layout_gc<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_c);
    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * kargs.stride_c + half_tile_n * T::HALF_B_N + col;
    };

    auto direct_store = [&](auto& v_c_in, int half_tile_m, int half_tile_n) {
        auto v_c_f16 = cast<D_C>(v_c_in);
        static_assert(sizeof(D_C) * 8 % sizeof(u32_t) == 0);
        constexpr int u32_per_chunk = sizeof(D_C) * 8 / sizeof(u32_t);
        constexpr int num_chunks = sizeof(v_c_f16) / (sizeof(u32_t) * u32_per_chunk);
        auto* p_u32 = reinterpret_cast<u32_t*>(&v_c_f16);
        static_for<num_chunks>([&](auto c) {
            auto* p = p_u32 + c.value * u32_per_chunk;
            auto r0 = __builtin_amdgcn_permlane16_swap(p[0], p[2], false, true);
            auto r1 = __builtin_amdgcn_permlane16_swap(p[1], p[3], false, true);
            p[0] = r0[0]; p[2] = r0[1];
            p[1] = r1[0]; p[3] = r1[1];
        });
        g_c.template store<T::VEC_C>(v_c_f16, u_gc, c_offset(half_tile_m, half_tile_n));
    };

    direct_store(v_c[0][0], 0, 0);
    direct_store(v_c[0][1], 0, 1);
    direct_store(v_c[1][0], 1, 0);
    direct_store(v_c[1][1], 1, 1);
    };

    compute_one_n_tile(n_tile_initial);
}

#ifdef A_TILE
#undef A_TILE
#endif
#ifdef B_TILE
#undef B_TILE
#endif
#ifdef LOAD_A_TILE
#undef LOAD_A_TILE
#endif
#ifdef LOAD_B_TILE
#undef LOAD_B_TILE
#endif
