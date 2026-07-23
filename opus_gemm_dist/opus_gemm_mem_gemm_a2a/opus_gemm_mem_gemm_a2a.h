#pragma once

#include "gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh"

// Baseline split-K shape for the memory-bound GEMM + A2A experiment:
// opus_gemm_flatmm_splitk_256x64x64x128_2x1_16x16x32_0x0x0_wgpcu1_nooob.
// The main GEMM writes fp32 partials to workspace; splitk_reduce_kernel casts
// the final C tile to bf16 before the RCCL reference all-to-all.
using opus_mem_gemm_a2a_traits = opus_flatmm_splitk_traits_gfx950<
    256,
    opus::seq<64, 64, 128>,
    opus::tuple<bf16_t, bf16_t, float, float, bf16_t>,
    opus::seq<8, 8, 4>,
    opus::seq<16, 16, 32>,
    1,
    false,
    false>;

static constexpr int OPUS_MEM_GEMM_A2A_REDUCE_VEC = 16;
static constexpr int OPUS_MEM_GEMM_A2A_REDUCE_BLOCK = 64;

