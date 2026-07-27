#pragma once

#include "../../../opus_dist_gemm/gemm_defs.h"

struct opus_a2a_gemm_kargs {
    const void* __restrict__ ptr_b = nullptr;     // [N, K], bf16, replicated on every rank
    void* __restrict__ ptr_c = nullptr;           // [M, N], bf16

    void* recv_a_win = nullptr;                   // LSA window: [rank_count, M, K_SHARD], bf16
    void* recv_a_local = nullptr;                 // local device pointer for this rank's recv_a window
    void* ready_win = nullptr;                    // LSA window: [rank_count], uint32 rotate-put flags
    void* ready_local = nullptr;                  // local device pointer for this rank's rotate-put flags

    int m = 2048;
    int n = 8192;
    int k_shard = 1024;
    int rank_count = 8;
    int my_rank = 0;

    int stride_a = 1024;      // local/recv A shard row stride
    int stride_b = 8192;      // full B row stride
    int stride_c = 8192;      // output C row stride

    unsigned int output_bytes = 0;
};
