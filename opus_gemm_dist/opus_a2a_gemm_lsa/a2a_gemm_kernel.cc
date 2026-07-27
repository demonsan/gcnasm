#include <opus/hip_minimal.hpp>
#include "gemm_defs.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits>
__global__ void a2a_gemm_lsa_direct_read_kernel(opus_a2a_gemm_kargs kargs) {}
template __global__ void a2a_gemm_lsa_direct_read_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>>(opus_a2a_gemm_kargs);
#else
#include "a2a_gemm_direct_read_pipeline.hpp"
template __global__ void a2a_gemm_lsa_direct_read_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>>(opus_a2a_gemm_kargs);
#endif
