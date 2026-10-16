// SPDX-License-Identifier: MIT
#pragma once
#include "opus_gemm_traits_a8w8_ptpc_bpreshuffle_gfx950.cuh"
struct C11Traits {
    using D_A=fp8_t; using D_B=fp8_t; using D_ACC=fp32_t;
    static constexpr int BLOCK_SIZE=256,WG_PER_CU=2,B_M=32,B_N=64,B_K=256;
    static constexpr int T_M=2,T_N=1,T_K=1,W_M=16,W_N=16,W_K=128;
    static constexpr int VEC_A=16,VEC_B=16,VEC_C=4;
    static constexpr int LOAD_GROUP_M=32,LOAD_GROUP_N=32,LOAD_GROUP_K=128;
    static constexpr int LOAD_GROUP_M_LANE=1,LOAD_GROUP_N_LANE=1;
    static constexpr int NUM_LOAD_GROUPS_PER_BM=1,NUM_LOAD_GROUPS_PER_BN=2,NUM_LOAD_GROUPS_PER_BK=2;
    static constexpr int COM_REP_M=1,COM_REP_N=4,COM_REP_K=2;
    static constexpr int slots=4,smem_linear_wave_per_async_load=1024,smem_padding=32,smem_per_group_load_size=4224,prefetch_k_iter=3;
    static constexpr int a_buffer_load_insts=4,b_buffer_load_insts=8,a_ds_read_insts=4,b_ds_read_insts=16;
    static constexpr int LOAD_WAVES=2;
    static constexpr bool B_PRESHUFFLE=true,B_DIRECT_REG=false,ALL_WAVE=false;
};
#ifdef __HIP_DEVICE_COMPILE__
template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_group_load_c11(int lane_id, int wave_id, int stride) {
    constexpr int threads_k = T::LOAD_GROUP_K / T::VEC_A;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_m = threads_m_per_wave / T::LOAD_GROUP_M_LANE;
    constexpr int repeat_m = T::slots / WAVES;

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<interlanegroup_m>{},
        opus::number<repeat_m>{},
        opus::number<WAVES>{},
        opus::number<T::LOAD_GROUP_M_LANE>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(g_block_dim,
            opus::tuple{lane_id / threads_k / T::LOAD_GROUP_M_LANE,
                        wave_id % WAVES,
                        (lane_id / threads_k) % T::LOAD_GROUP_M_LANE,
                        lane_id % threads_k}));
}

template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_group_load_b_preshuffle_c11(int lane_id, int wave_id, int stride_b) {
    constexpr int threads_k = T::LOAD_GROUP_K / T::VEC_B;
    constexpr int threads_n_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_n = threads_n_per_wave / T::LOAD_GROUP_N_LANE;
    constexpr int repeat_n = T::slots / WAVES;

    constexpr int n_per_lane_group = repeat_n * WAVES * T::LOAD_GROUP_N_LANE;
    static_assert(n_per_lane_group <= 16 && 16 % n_per_lane_group == 0,
                  "one lane's n range must stay inside a single 16-row preshuffle block");
    static_assert(interlanegroup_n * n_per_lane_group == T::LOAD_GROUP_N);
    constexpr int repeat_n_stride = WAVES * T::LOAD_GROUP_N_LANE * 16;

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<repeat_n>{},
        opus::number<T::VEC_B>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    const int lane_n = lane_id / threads_k;
    const int lane_k = lane_id % threads_k;

    const int n0 = (lane_n / T::LOAD_GROUP_N_LANE) * n_per_lane_group
                 + (wave_id % WAVES) * T::LOAD_GROUP_N_LANE
                 + lane_n % T::LOAD_GROUP_N_LANE;

    auto u_gb = opus::make_layout<0>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape,
            opus::tuple{opus::number<repeat_n_stride>{}, 1_I}),
        opus::unfold_p_coord(g_block_dim, opus::tuple{0}));
    u_gb += (n0 / 16) * stride_b * 16 + (n0 % 16) * 16
          + (lane_k / 2) * 512 + (lane_k % 2) * 256;
    return u_gb;
}

template<typename T>
inline constexpr int b_preshuffle_chunk_bytes_c11 = opus::get_warp_size() * T::VEC_B;
template<typename T>
inline constexpr int b_preshuffle_k_chunks_c11 =
    T::LOAD_GROUP_K * 16 / b_preshuffle_chunk_bytes_c11<T>;
template<typename T>
inline constexpr int b_preshuffle_n_blocks_c11 = T::LOAD_GROUP_N / 16;

template<typename T>
inline constexpr bool b_preshuffle_contig_c11 =
    T::B_PRESHUFFLE && !T::B_DIRECT_REG
    && T::VEC_B == 16 && T::LOAD_GROUP_N % 16 == 0
    && (T::LOAD_GROUP_K * 16) % b_preshuffle_chunk_bytes_c11<T> == 0
    && b_preshuffle_k_chunks_c11<T> % 2 == 0
    && b_preshuffle_n_blocks_c11<T> * b_preshuffle_k_chunks_c11<T> == T::slots
    && T::COM_REP_N % b_preshuffle_n_blocks_c11<T> == 0

    && (!T::ALL_WAVE
        || (T::LOAD_WAVES <= b_preshuffle_k_chunks_c11<T>
                ? b_preshuffle_k_chunks_c11<T> % T::LOAD_WAVES == 0
                : T::LOAD_WAVES % b_preshuffle_k_chunks_c11<T> == 0
                      && b_preshuffle_n_blocks_c11<T>
                                 % (T::LOAD_WAVES / b_preshuffle_k_chunks_c11<T>)
                             == 0));

template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_group_load_b_preshuffle_contig_c11(
        int lane_id, int wave_id, int stride_b) {
    constexpr int chunk_bytes = b_preshuffle_chunk_bytes_c11<T>;
    constexpr int k_chunks    = b_preshuffle_k_chunks_c11<T>;
    constexpr int n_blocks    = b_preshuffle_n_blocks_c11<T>;

    constexpr bool WAVE_SPLITS_N = WAVES > k_chunks;
    static_assert(WAVE_SPLITS_N ? WAVES % k_chunks == 0 : k_chunks % WAVES == 0,
                  "the staging waves and the load group's k chunks must divide "
                  "one way or the other");
    constexpr int n_per_wave = WAVE_SPLITS_N ? WAVES / k_chunks : 1;
    static_assert(n_blocks % n_per_wave == 0,
                  "the waves past the k chunks must divide the n blocks");
    constexpr int n_blocks_per_wave  = n_blocks / n_per_wave;
    constexpr int k_chunks_per_wave  = WAVE_SPLITS_N ? 1 : k_chunks / WAVES;
    static_assert(n_blocks_per_wave * k_chunks_per_wave == T::slots / WAVES,
                  "one wave's issues must cover its share of the load group");

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<1>{},
        opus::number<n_blocks_per_wave>{},
        opus::number<k_chunks_per_wave>{},
        opus::number<T::VEC_B>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    constexpr int k_repeat_bytes = (WAVE_SPLITS_N ? 1 : WAVES) * chunk_bytes;
    auto u_gb = opus::make_layout<0>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape,
            opus::tuple{n_per_wave * stride_b * 16,
                        opus::number<k_repeat_bytes>{},
                        1_I}),
        opus::unfold_p_coord(g_block_dim, opus::tuple{0}));
    const int chunk = wave_id % WAVES;
    u_gb += (WAVE_SPLITS_N ? (chunk / k_chunks) * stride_b * 16 : 0)
          + (WAVE_SPLITS_N ? chunk % k_chunks : chunk) * chunk_bytes
          + lane_id * T::VEC_B;
    return u_gb;
}

template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_b_c11(int lane_id, int wave_id, int stride_b) {
    if constexpr (b_preshuffle_contig_c11<T>)
        return make_layout_gmem_group_load_b_preshuffle_contig_c11<T, WAVES>(lane_id, wave_id, stride_b);
    else if constexpr (T::B_PRESHUFFLE)
        return make_layout_gmem_group_load_b_preshuffle_c11<T, WAVES>(lane_id, wave_id, stride_b);
    else
        return make_layout_gmem_group_load_c11<T, WAVES>(lane_id, wave_id, stride_b);
}

template<typename T>
inline __device__ size_t b_gmem_tile_base_c11(int col, int k_start, int stride_b) {
    const size_t k_term = T::B_PRESHUFFLE ? (size_t)k_start * 16 : (size_t)k_start;
    return (size_t)col * (size_t)stride_b + k_term;
}

template<typename T>
inline __device__ auto make_layout_gmem_b_direct_c11(int lane_id, int stride_b, int nbc) {
    constexpr int grpk_b = opus::get_warp_size() / T::W_N;
    static_assert(grpk_b * T::W_N * T::VEC_B * 2 == T::W_N * T::W_K,
                  "one 16x16x128 B fragment is 2 halves x 64 lanes x VEC_B bytes");

    constexpr auto b_block_shape = opus::make_tuple(
        opus::number<T::COM_REP_N>{},
        opus::number<T::COM_REP_K>{},
        opus::number<2>{},
        opus::number<grpk_b>{},
        opus::number<T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto b_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto u_gb = opus::make_layout<0>(
        b_block_shape,
        opus::unfold_x_stride(b_block_dim, b_block_shape,
            opus::tuple{stride_b * 16, 1_I}),
        opus::unfold_p_coord(b_block_dim,
            opus::tuple{lane_id / T::W_N, lane_id % T::W_N}));
    u_gb += nbc * stride_b * 16;
    return u_gb;
}

template<typename T>
inline __device__ int b_direct_iter_offset_c11(int loop_k_idx) {
    return loop_k_idx * T::B_K * 16;
}

template<typename T>
inline __device__ int b_gmem_group_offset_c11(int loop_k_idx, int group_load_idx, int k_group, int stride_b) {
    const int k_off = (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    return group_load_idx * T::LOAD_GROUP_N * stride_b + (T::B_PRESHUFFLE ? k_off * 16 : k_off);
}

template<typename T, int WAVES>
inline __device__ auto make_layout_smem_group_load_c11(int lane_id, int wave_id) {
    constexpr int repeat_m = T::slots / WAVES;

    constexpr auto s_block_shape = opus::make_tuple(
        opus::number<repeat_m>{},
        opus::number<WAVES>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto s_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        s_block_shape,
        opus::unfold_x_stride(s_block_dim, s_block_shape,
            opus::tuple{T::smem_linear_wave_per_async_load + T::smem_padding, 1_I}),
        opus::unfold_p_coord(s_block_dim, opus::tuple{wave_id % WAVES, lane_id}));
}

template<typename T>
inline constexpr int ra_m_block_stride_c11 =
    T::NUM_LOAD_GROUPS_PER_BK * T::slots
    * (T::smem_linear_wave_per_async_load + T::smem_padding);

template<typename T, int M_REPS = T::COM_REP_M, bool WAVE_PAIR = false>
inline __device__ auto make_layout_ra_c11(int lane_id, int wave_id_m) {
    constexpr int threads_k = opus::get_warp_size() / T::W_M;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_m = threads_m_per_wave / T::LOAD_GROUP_M_LANE;
    constexpr int per_block_load = T::slots * (T::smem_linear_wave_per_async_load + T::smem_padding);
    constexpr int m_block_stride = ra_m_block_stride_c11<T>;

    if constexpr (WAVE_PAIR) {
        static_assert(T::T_M == 2, "the wave M remap hands each wave T_M*W_M "
                      "contiguous rows and takes the half from the subtile index; "
                      "only T_M=2 is derived (and only it is WIDE at sub=16)");
        static_assert(M_REPS % T::T_M == 0,
                      "the remap gives each wave whole 2*W_M row blocks");

        constexpr auto ra_block_shape = opus::make_tuple(
            opus::number<M_REPS / T::T_M>{},
            opus::number<T::T_M>{},
            opus::number<T::slots>{},
            opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
            opus::number<T::T_M>{},
            opus::number<interlanegroup_m / T::slots>{},
            opus::number<T::LOAD_GROUP_M_LANE>{},
            opus::number<2>{},
            opus::number<threads_k>{},
            opus::number<T::VEC_A>{});

        constexpr auto ra_block_dim = opus::make_tuple(
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
            opus::make_tuple(opus::p_dim{}),
            opus::make_tuple(opus::y_dim{}),
            opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{},
                             opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

        auto lane_id_m = lane_id % T::W_M;

        return opus::make_layout<0>(
            ra_block_shape,
            opus::unfold_x_stride(ra_block_dim, ra_block_shape,
                opus::tuple{opus::number<m_block_stride>{},
                            opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                            opus::number<per_block_load>{},
                            1_I}),
            opus::unfold_p_coord(ra_block_dim,
                opus::tuple{wave_id_m,
                            lane_id_m % T::slots,
                            lane_id_m / T::slots,
                            lane_id_m % T::LOAD_GROUP_M_LANE,
                            lane_id / T::W_M}));
    } else {

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<M_REPS>{},
        opus::number<T::slots>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<T::T_M>{},
        opus::number<interlanegroup_m / T::slots>{},
        opus::number<T::LOAD_GROUP_M_LANE>{},
        opus::number<2>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{},
                         opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_m = lane_id % T::W_M;

    return opus::make_layout<0>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<m_block_stride>{},
                        opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                        opus::number<per_block_load>{},
                        1_I}),
        opus::unfold_p_coord(ra_block_dim,
            opus::tuple{lane_id_m % T::slots,
                        wave_id_m,
                        lane_id_m / T::slots,
                        lane_id_m % T::LOAD_GROUP_M_LANE,
                        lane_id / T::W_M}));
    }
}

template<typename T>
inline __device__ auto make_layout_rb_staged_c11(int lane_id) {
    constexpr int grpk_b = opus::get_warp_size() / T::W_N;
    constexpr int interlanegroup_n = T::W_N / T::LOAD_GROUP_N_LANE;
    constexpr int loops_b = interlanegroup_n / T::slots;
    constexpr int tiles_per_block_n = T::LOAD_GROUP_N / T::W_N;
    constexpr int num_blocks_n = T::COM_REP_N / tiles_per_block_n;
    constexpr int per_block_load = T::slots * (T::smem_linear_wave_per_async_load + T::smem_padding);
    constexpr int n_block_stride = T::NUM_LOAD_GROUPS_PER_BK * per_block_load;
    constexpr int n_intra_stride = T::LOAD_GROUP_N_LANE * 2 * grpk_b * T::VEC_B;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<num_blocks_n>{},
        opus::number<T::slots>{},
        opus::number<tiles_per_block_n>{},
        opus::number<loops_b>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<T::LOAD_GROUP_N_LANE>{},
        opus::number<2>{},
        opus::number<grpk_b>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout<0>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<n_block_stride>{},
                        opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                        opus::number<n_intra_stride>{},
                        opus::number<per_block_load>{},
                        1_I}),
        opus::unfold_p_coord(rb_block_dim,
            opus::tuple{lane_id_n % T::slots,
                        lane_id_n / T::slots,
                        lane_id_n % T::LOAD_GROUP_N_LANE,
                        lane_id / T::W_N}));
}

template<typename T>
inline __device__ auto make_layout_rb_preshuffle_contig_c11(int lane_id) {
    constexpr int slot_stride    = T::smem_linear_wave_per_async_load + T::smem_padding;
    constexpr int per_block_load = T::slots * slot_stride;
    constexpr int k_chunks       = b_preshuffle_k_chunks_c11<T>;
    constexpr int n_blocks       = b_preshuffle_n_blocks_c11<T>;
    constexpr int num_groups_n   = T::COM_REP_N / n_blocks;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<num_groups_n>{},
        opus::number<n_blocks>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<k_chunks>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<T::NUM_LOAD_GROUPS_PER_BK * per_block_load>{},
                        opus::number<k_chunks * slot_stride>{},
                        opus::number<per_block_load>{},
                        opus::number<slot_stride>{},
                        1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{lane_id}));
}

template<typename T>
inline __device__ auto make_layout_rb_c11(int lane_id) {
    if constexpr (b_preshuffle_contig_c11<T>)
        return make_layout_rb_preshuffle_contig_c11<T>(lane_id);
    else
        return make_layout_rb_staged_c11<T>(lane_id);
}

#endif

template<bool DIRECT_C>
__global__ __launch_bounds__(C11Traits::BLOCK_SIZE,C11Traits::WG_PER_CU)
void gemm_ptpc_c11_pc_gfx950(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using T=C11Traits; using D_A=typename T::D_A;using D_B=typename T::D_B;using D_ACC=typename T::D_ACC;
    int wgid_full = opus::block_id_x();
    int split_id = wgid_full % kargs.split_k;
    int wgid = wgid_full / kargs.split_k;
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    const int total_iters = ceil_div(kargs.k, T::B_K);
    const int ips = kargs.iters_per_split;
    const int k_start = split_id * ips * T::B_K;
    const int my_loops = min(ips, total_iters - split_id * ips);
    if (my_loops < T::prefetch_k_iter) __builtin_trap();
    const int rows_avail = min(T::B_M, kargs.m - row);
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + (size_t)row * kargs.stride_a + k_start,
                         (rows_avail - 1) * kargs.stride_a + kargs.k - k_start);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + b_gmem_tile_base_c11<T>(col, k_start, kargs.k),
                         T::B_N * kargs.k - k_start * 16);
    const int loops = my_loops;
    const int role = ((wave_id & 1) ^ ((wgid >> 8) & 1));
    constexpr int mb = T::a_buffer_load_insts + T::b_buffer_load_insts;
    __shared__ __align__(16) char smem_a[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BM * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ __align__(16) char smem_b[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BN * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    auto smem_a_at = [&](int sk, int mg, int kg) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a + ((sk*T::NUM_LOAD_GROUPS_PER_BM+mg)*T::NUM_LOAD_GROUPS_PER_BK+kg)*T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int sk, int ng, int kg) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b + ((sk*T::NUM_LOAD_GROUPS_PER_BN+ng)*T::NUM_LOAD_GROUPS_PER_BK+kg)*T::smem_per_group_load_size);
    };
    auto a_offset = [&](int kt, int mg, int kg) {
        return mg*T::LOAD_GROUP_M*kargs.stride_a + (kt*T::NUM_LOAD_GROUPS_PER_BK+kg)*T::LOAD_GROUP_K;
    };
    auto b_offset = [&](int kt, int ng, int kg) { return b_gmem_group_offset_c11<T>(kt,ng,kg,kargs.k); };
    if (role == 0) {
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_c11<T, 2>(lane_id, wave_id_prod, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_c11<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_b_c11<T, 2>(lane_id, wave_id_prod, kargs.k);
        auto u_sb = make_layout_smem_group_load_c11<T, 2>(lane_id, wave_id_prod);

        auto stage_b = [&](int slot, int issue_k, auto kg_c) {
            if constexpr (!T::B_DIRECT_REG) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb,
                                         b_offset(issue_k, n, kg));
                });
            }
        };

        opus::static_for<T::prefetch_k_iter>([&](auto p_c) {
            constexpr int p = decltype(p_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                    constexpr int m = decltype(m_c)::value;
                    async_load<T::VEC_A>(g_a, smem_a_at(p, m, kg), u_ga, u_sa, a_offset(p, m, kg));
                });
                stage_b(p, p, kg_c);
            });

        });

        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = T::prefetch_k_iter - 1 - decltype(i_c)::value;
            s_waitcnt_vmcnt(number<mb * p>{});
            __builtin_amdgcn_s_barrier();
        });

        if constexpr (T::prefetch_k_iter == 3) {
            s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();
            for (int i = T::prefetch_k_iter - 1; i < loops - 1; i++) {
                int issue_k = i + 1;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    stage_b(slot, issue_k, kg_c);
                });

                s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
            }
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        } else {
            for (int i = T::prefetch_k_iter - 2; i < loops - 2; i++) {
                int issue_k = i + 2;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    stage_b(slot, issue_k, kg_c);
                });

                s_waitcnt_vmcnt(number<2 * mb>{});
                __builtin_amdgcn_s_barrier();
            }
            s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
    } else {
        const int wave_id_m = wave_id / 2;
        const int wave_id_n_cons = 0;
        auto u_ra = make_layout_ra_c11<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_c11<T>(lane_id);
        auto stage_barrier = [&](int) {

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        };
        auto stage_barrier_c = [&](int p) { stage_barrier(p); };
        auto mma = make_tiled_mma<D_A,D_B,D_ACC>(seq<T::COM_REP_M,T::COM_REP_N,T::COM_REP_K>{},
                      seq<T::T_M,T::T_N,T::T_K>{},seq<T::W_M,T::W_N,T::W_K>{},mfma_adaptor_swap_ab{});

        auto half_mma=make_tiled_mma<D_A,D_B,D_ACC>(seq<T::COM_REP_M,T::COM_REP_N,1>{},
            seq<T::T_M,T::T_N,T::T_K>{},seq<T::W_M,T::W_N,T::W_K>{},mfma_adaptor_swap_ab{});
        typename decltype(half_mma)::vtype_a v_a0,v_a1;
        typename decltype(half_mma)::vtype_b v_b0,v_b1;
        typename decltype(mma)::vtype_c v_c;
        static_assert(decltype(half_mma)::tile_c_len==decltype(mma)::tile_c_len);
        static_assert(sizeof(typename decltype(half_mma)::vtype_a)==32);
        static_assert(sizeof(typename decltype(half_mma)::vtype_b)==128);
        static_assert(sizeof(typename decltype(half_mma)::vtype_c)==sizeof(typename decltype(mma)::vtype_c));
        clear(v_c);
        const auto oa=layout_to_offsets<T::VEC_A>(u_ra);
        const auto ob=layout_to_offsets<T::VEC_B>(u_rb);
        auto read_half=[&](int kt,auto ik_c,auto& va,auto& vb) {
            constexpr int ik=decltype(ik_c)::value;
            const int sk=kt%T::prefetch_k_iter;
            auto sa=make_smem(smem_a_at(sk,0,0));
            auto sb=make_smem(smem_b_at(sk,0,0));
            static_ford<2>([&](auto h_c) {
                constexpr int h=decltype(h_c)::value;
                auto chunk=sa.template load<16>(oa[ik*2+h]);
                set_slice(va,chunk,number<h*16>{},number<h*16+16>{});
            });
            static_ford<T::COM_REP_N,2>([&](auto in_c,auto h_c) {
                constexpr int in=decltype(in_c)::value,h=decltype(h_c)::value;
                auto chunk=sb.template load<16>(ob[in*4+ik*2+h]);
                set_slice(vb,chunk,number<in*32+h*16>{},number<in*32+h*16+16>{});
            });
        };
        constexpr int half_reads=10;
        stage_barrier(0);
        read_half(0,0_I,v_a0,v_b0);
        read_half(0,1_I,v_a1,v_b1);
        s_waitcnt_lgkmcnt(number<half_reads>{});
        __builtin_amdgcn_s_setprio(1);
        v_c=half_mma(v_a0,v_b0,v_c,127,127);
        __builtin_amdgcn_s_setprio(0);
        for(int kt=1;kt<loops;++kt) {
            stage_barrier(kt);
            read_half(kt,0_I,v_a0,v_b0);
            s_waitcnt_lgkmcnt(number<half_reads>{});
            __builtin_amdgcn_s_setprio(1);
            v_c=half_mma(v_a1,v_b1,v_c,127,127);
            __builtin_amdgcn_s_setprio(0);
            read_half(kt,1_I,v_a1,v_b1);
            s_waitcnt_lgkmcnt(number<half_reads>{});
            __builtin_amdgcn_s_setprio(1);
            v_c=half_mma(v_a0,v_b0,v_c,127,127);
            __builtin_amdgcn_s_setprio(0);
        }
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c=half_mma(v_a1,v_b1,v_c,127,127);
        __builtin_amdgcn_s_setprio(0);

        auto p_coord_c=opus::make_tuple(wave_id_m,lane_id%mma.grpn_c,wave_id_n_cons,lane_id/mma.grpn_c);
        if constexpr (DIRECT_C) {
            auto uc=partition_layout_c<1>(mma,opus::make_tuple(kargs.stride_c,1_I),p_coord_c);
            auto um=partition_layout_c<1>(mma,opus::make_tuple(1_I,0_I),p_coord_c);
            auto un=partition_layout_c<1>(mma,opus::make_tuple(0_I,1_I),p_coord_c);
            auto gc=make_gmem(reinterpret_cast<bf16_t*>(kargs.ptr_c)+(size_t)row*kargs.stride_c+col);
            auto sa=make_gmem(reinterpret_cast<const float*>(kargs.ptr_sa)+row,rows_avail*(int)sizeof(float));
            auto sb=make_gmem(reinterpret_cast<const float*>(kargs.ptr_sb)+col,T::B_N*(int)sizeof(float));
            vector_t<bf16_t,T::COM_REP_M*T::COM_REP_N*decltype(mma)::mma_c_len> y;
            constexpr auto issues=layout_to_issue_space<decltype(uc)>();
            static_ford(issues,[&](auto... ids) { constexpr index_t j=make_layout<-1>(issues)(ids...);
                y[j]=static_cast<bf16_t>((v_c[j]*sa.template load<1>(um(ids...))[0])*sb.template load<1>(un(ids...))[0]); });
            auto pred=[&](auto...ids) {return row+um(ids...)<kargs.m && col+un(ids...)<kargs.n;};
            store_if<1>(gc,pred,y,uc,0);
        } else {
            auto uc=partition_layout_c<T::VEC_C>(mma,opus::make_tuple(kargs.padded_n,1_I),p_coord_c);
            auto gc=make_gmem(reinterpret_cast<float*>(kargs.ptr_ws)+(size_t)split_id*kargs.padded_m*kargs.padded_n+(size_t)row*kargs.padded_n+col);
            store<T::VEC_C>(gc,v_c,uc,0);
        }
    }
#endif
#endif
}
