// SPDX-License-Identifier: MIT
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ptpc_launch_fn)(
    const void* A, const void* W, const float* sa, const float* sb,
    void* Y, void* workspace, size_t workspace_bytes,
    int M, int N, int K, int lda, int ldc, int requested_split,
    int* effective_split, void* stream);

#define PTPC_DECLARE(name) \
    int name(const void*, const void*, const float*, const float*, \
             void*, void*, size_t, int, int, int, int, int, int, int*, void*)

PTPC_DECLARE(ptpc_c1_launch);
PTPC_DECLARE(ptpc_c11_launch);
PTPC_DECLARE(ptpc_c3_launch);
PTPC_DECLARE(ptpc_c18_launch);
PTPC_DECLARE(ptpc_c19_launch);
int ptpc_last_hip_error(void);
#undef PTPC_DECLARE

#ifdef __cplusplus
}
#endif
