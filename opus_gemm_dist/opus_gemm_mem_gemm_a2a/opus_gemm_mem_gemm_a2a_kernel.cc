#include "opus_gemm_mem_gemm_a2a.h"

template __global__ void gemm_a16w16_flatmm_splitk_kernel<opus_mem_gemm_a2a_traits>(
    opus_gemm_flatmm_splitk_kargs_gfx950);

template __global__ void splitk_reduce_kernel<
    OPUS_MEM_GEMM_A2A_REDUCE_VEC,
    OPUS_MEM_GEMM_A2A_REDUCE_BLOCK,
    bf16_t,
    false,
    bf16_t,
    false>(
    const opus_splitk_ws_handle*,
    bf16_t*,
    int,
    int,
    int,
    int,
    int,
    int,
    const bf16_t*,
    int);

template __global__ void splitk_reduce_a2a_kernel<
    OPUS_MEM_GEMM_A2A_REDUCE_VEC,
    OPUS_MEM_GEMM_A2A_REDUCE_BLOCK,
    bf16_t>(
    const opus_splitk_ws_handle*,
    void*,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int);

