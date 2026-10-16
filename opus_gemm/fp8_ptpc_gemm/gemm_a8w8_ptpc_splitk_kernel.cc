// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#include "opus_gemm_traits_a8w8_ptpc_bpreshuffle_gfx950.cuh"
#include "opus_gemm_pipeline_a8w8_ptpc_bpreshuffle_p4d_gfx950.cuh"
#include "c1_mask_only.cuh"
#include "c7_c2_mask.cuh"
#include "c11_mature.cuh"
#include "splitk_reduce_ptpc_gfx950.cuh"

using DT_PTPC = opus::tuple<fp8_t, fp8_t, bf16_t, fp32_t, fp32_t>;
using VEC_PTPC = opus::seq<16, 16, 4>;

using TrP4D_C1_A = opus_gemm_a8w8_ptpc_bpreshuffle_traits_gfx950<
    256, opus::seq<16, 64, 256>, DT_PTPC, VEC_PTPC, opus::seq<1, 4>, opus_ptpc_pipe_p4d, 2, 1, true, true>;
using TrP4D_C1_U = opus_gemm_a8w8_ptpc_bpreshuffle_traits_gfx950<
    256, opus::seq<16, 64, 256>, DT_PTPC, VEC_PTPC, opus::seq<1, 4>, opus_ptpc_pipe_p4d, 2, 1, false, true>;
using TrP4D_C2_A = opus_gemm_a8w8_ptpc_bpreshuffle_traits_gfx950<
    256, opus::seq<32, 64, 256>, DT_PTPC, VEC_PTPC, opus::seq<1, 4>, opus_ptpc_pipe_p4d, 2, 1, true, true>;
using TrP4D_C2_U = opus_gemm_a8w8_ptpc_bpreshuffle_traits_gfx950<
    256, opus::seq<32, 64, 256>, DT_PTPC, VEC_PTPC, opus::seq<1, 4>, opus_ptpc_pipe_p4d, 2, 1, false, true>;

template __global__ void gemm_ptpc_c1_mask_gfx950<TrP4D_C1_A>(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950);
template __global__ void gemm_ptpc_c7_mask_gfx950<TrP4D_C2_A>(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_ptpc_bpreshuffle_p4d_gfx950<TrP4D_C1_U>(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950);
template __global__ void gemm_a8w8_ptpc_bpreshuffle_p4d_gfx950<TrP4D_C2_U>(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950);
template __global__ void splitk_reduce_ptpc_gfx950<16, 64, __bf16, true>(
    const void*, const float*, const float*, __bf16*, int, int, int, int, int, int);
template __global__ void splitk_reduce_ptpc_gfx950<16, 64, __bf16, false>(
    const void*, const float*, const float*, __bf16*, int, int, int, int, int, int);

namespace {

template<typename Tr>
int prepare_ptpc(opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950& kargs, dim3& grid, dim3& block,
                 const void* A, const void* Wp, const float* sa, const float* sb,
                 void* C, void* ws, size_t ws_bytes,
                 int m, int n, int k, int stride_a, int stride_c, int split_k_req)
{

    if (m < 1 || n < 16 || k < 32) return 2;
    if (n % 16 != 0) return 2;
    if (k % 32 != 0) return 2;
    if (stride_a < k) return 2;
    if (stride_c < n) return 2;
    if (split_k_req < 1) return 2;

    if (reinterpret_cast<uintptr_t>(C) % 2 != 0) return 2;
    if (reinterpret_cast<uintptr_t>(sa) % 4 != 0 || reinterpret_cast<uintptr_t>(sb) % 4 != 0) return 2;

    const int t_total = (k + Tr::B_K - 1) / Tr::B_K;
    int ips;
    if (Tr::IS_P8) {

        ips = t_total >= 2 ? ((((t_total + split_k_req - 1) / split_k_req) + 1) & ~1) : 1;
    } else {
        ips = (t_total + split_k_req - 1) / split_k_req;
    }
    const int s_eff = (t_total + ips - 1) / ips;

    const int tiles_m = (m + Tr::B_M - 1) / Tr::B_M;
    const int tiles_n = (n + Tr::B_N - 1) / Tr::B_N;
    const int padded_m = tiles_m * Tr::B_M;
    const int padded_n = tiles_n * Tr::B_N;

    const size_t a_span   = (size_t)(Tr::B_M - 1) * stride_a + k;
    const size_t w_span   = (size_t)Tr::B_N * k;
    const size_t y_span   = ((size_t)(m - 1) * stride_c + n) * sizeof(__bf16);
    const size_t ws_split = (size_t)padded_m * padded_n * sizeof(float);
    const size_t sa_span  = (size_t)Tr::B_M * sizeof(float);
    const size_t sb_span  = (size_t)Tr::B_N * sizeof(float);
    const size_t max_sos  = (size_t)(t_total > 0 ? t_total - 1 : 0) * (Tr::B_K > 2048 ? Tr::B_K : 2048);
    const size_t LIMIT = (size_t)1 << 31;
    if (a_span >= LIMIT || w_span >= LIMIT || y_span >= LIMIT || ws_split >= LIMIT
        || sa_span >= LIMIT || sb_span >= LIMIT || max_sos >= LIMIT) return 4;

    if (s_eff > 1) {
        const size_t ws_need = (size_t)s_eff * padded_m * padded_n * sizeof(float);
        if (ws == nullptr) return 2;
        if (ws_bytes < ws_need) return 5;
    }

    kargs.ptr_a = A;
    kargs.ptr_b = Wp;
    kargs.ptr_sa = sa;
    kargs.ptr_sb = sb;
    kargs.ptr_c = C;
    kargs.ptr_ws = ws;
    kargs.m = m;
    kargs.n = n;
    kargs.k = k;
    kargs.stride_a = stride_a;
    kargs.stride_c = stride_c;
    kargs.split_k = s_eff;
    kargs.iters_per_split = ips;
    kargs.padded_m = padded_m;
    kargs.padded_n = padded_n;

    grid = dim3((unsigned)(s_eff * tiles_m * tiles_n), 1, 1);
    block = dim3((unsigned)Tr::BLOCK_SIZE, 1, 1);
    return 0;
}

int finish_ptpc(const opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950& kargs, bool aligned_c,
                hipStream_t stream)
{
    if (kargs.split_k > 1) {
        const dim3 rgrid((unsigned)((kargs.n + 16 * 64 - 1) / (16 * 64)), (unsigned)kargs.m, 1);
        if (aligned_c) {
            splitk_reduce_ptpc_gfx950<16, 64, __bf16, true><<<rgrid, dim3(64, 1, 1), 0, stream>>>(
                kargs.ptr_ws, reinterpret_cast<const float*>(kargs.ptr_sa),
                reinterpret_cast<const float*>(kargs.ptr_sb), reinterpret_cast<__bf16*>(kargs.ptr_c),
                kargs.split_k, kargs.m, kargs.n, kargs.stride_c, kargs.padded_m, kargs.padded_n);
        } else {
            splitk_reduce_ptpc_gfx950<16, 64, __bf16, false><<<rgrid, dim3(64, 1, 1), 0, stream>>>(
                kargs.ptr_ws, reinterpret_cast<const float*>(kargs.ptr_sa),
                reinterpret_cast<const float*>(kargs.ptr_sb), reinterpret_cast<__bf16*>(kargs.ptr_c),
                kargs.split_k, kargs.m, kargs.n, kargs.stride_c, kargs.padded_m, kargs.padded_n);
        }
    }
    return 0;
}

}

#define PTPC_ENTRY(NAME, TR_A, TR_U, KERNEL_A, KERNEL_U)                                      \
extern "C" int NAME(const void* A, const void* Wp, const float* sa, const float* sb,          \
                    void* C, void* ws, size_t ws_bytes,                                        \
                    int m, int n, int k, int stride_a, int stride_c, int split_k_req,          \
                    int* s_eff_out, void* stream)                                              \
{                                                                                              \
    opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 kargs;                                        \
    dim3 grid, block;                                                                          \
    const int rc = prepare_ptpc<TR_A>(kargs, grid, block, A, Wp, sa, sb, C, ws, ws_bytes,      \
                                      m, n, k, stride_a, stride_c, split_k_req);               \
    if (rc != 0) return rc;                                                                    \
    if (s_eff_out) *s_eff_out = kargs.split_k;                                                 \
    hipStream_t st = reinterpret_cast<hipStream_t>(stream);                                    \
    const bool aligned_ab = (reinterpret_cast<uintptr_t>(A) % 16 == 0) && (stride_a % 16 == 0) \
                            && (reinterpret_cast<uintptr_t>(Wp) % 16 == 0);                    \
    const bool aligned_c  = (reinterpret_cast<uintptr_t>(C) % 16 == 0) && (stride_c % 8 == 0); \
    if (aligned_ab) KERNEL_A<TR_A><<<grid, block, 0, st>>>(kargs);                             \
    else            KERNEL_U<TR_U><<<grid, block, 0, st>>>(kargs);                             \
    return finish_ptpc(kargs, aligned_c, st);                                                  \
}

PTPC_ENTRY(ptpc_c1_launch, TrP4D_C1_A, TrP4D_C1_U, gemm_ptpc_c1_mask_gfx950, gemm_a8w8_ptpc_bpreshuffle_p4d_gfx950)

extern "C" int ptpc_c11_launch(const void* A,const void* Wp,const float* sa,const float* sb,
 void* C,void* ws,size_t ws_bytes,int m,int n,int k,int stride_a,int stride_c,int split_req,int* se_out,void* stream) {
    opus_gemm_a8w8_ptpc_bpreshuffle_kargs_gfx950 args; dim3 grid,block;
    int rc=prepare_ptpc<TrP4D_C2_A>(args,grid,block,A,Wp,sa,sb,C,ws,ws_bytes,m,n,k,stride_a,stride_c,split_req);
    if(rc) return rc;
    if(se_out)*se_out=args.split_k;
    hipStream_t st=reinterpret_cast<hipStream_t>(stream);
    const bool ab=(reinterpret_cast<uintptr_t>(A)%16==0)&&(stride_a%16==0)&&(reinterpret_cast<uintptr_t>(Wp)%16==0);
    const int tiles=(k+255)/256;
    const int last=tiles-(args.split_k-1)*args.iters_per_split;
    const bool fast=ab&&(m%32==0)&&(n%64==0)&&(k%256==0)&&args.iters_per_split>=3&&last>=3;
    if(fast) {
       if(args.split_k==1)gemm_ptpc_c11_pc_gfx950<true><<<grid,block,0,st>>>(args);
       else gemm_ptpc_c11_pc_gfx950<false><<<grid,block,0,st>>>(args);
    } else if(ab)gemm_ptpc_c7_mask_gfx950<TrP4D_C2_A><<<grid,block,0,st>>>(args);
    else gemm_a8w8_ptpc_bpreshuffle_p4d_gfx950<TrP4D_C2_U><<<grid,block,0,st>>>(args);
    const bool ac=(reinterpret_cast<uintptr_t>(C)%16==0)&&(stride_c%8==0);
    return finish_ptpc(args,ac,st);
}

extern "C" int ptpc_last_hip_error(void)
{
    return (int)hipGetLastError();
}
