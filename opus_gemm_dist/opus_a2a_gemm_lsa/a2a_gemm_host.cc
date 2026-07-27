#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <omp.h>
#include <hip/hip_runtime.h>
#include "mori/cco/cco.hpp"

#include "gemm_defs.h"

using namespace mori::cco;

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                 \
        }                                                                                                 \
    } while (0)

#define CHECK_CCO(call)                                                                    \
    do {                                                                                   \
        int cco_status_ = (call);                                                          \
        if (cco_status_ != 0) {                                                            \
            fprintf(stderr, "cco error %d (%s:%d): %s\n", cco_status_, __FILE__, __LINE__, \
                    #call);                                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

template<typename Traits>
__global__ void a2a_gemm_lsa_direct_read_kernel(opus_a2a_gemm_kargs kargs);

static constexpr int kRanks = 8;
static constexpr size_t kPerRankVmm = 512ULL * 1024 * 1024;

static float a_value(int src_rank, int row, int k_local) {
    return 0.001f * float(src_rank + 1) + 0.0002f * float((row % 17) - 8) +
           0.0001f * float((k_local % 29) - 14);
}

static float b_value(int col, int global_k) {
    return 0.0003f * float((col % 23) - 11) + 0.0001f * float((global_k % 31) - 15);
}

static void fill_a(bf16_t* a, int rank, int m, int k_shard) {
#pragma omp parallel for collapse(2)
    for (int r = 0; r < m; ++r) {
        for (int k = 0; k < k_shard; ++k) {
            a[r * k_shard + k] = static_cast<bf16_t>(a_value(rank, r, k));
        }
    }
}

static void fill_b(bf16_t* b, int n, int k) {
#pragma omp parallel for collapse(2)
    for (int c = 0; c < n; ++c) {
        for (int kk = 0; kk < k; ++kk) {
            b[c * k + kk] = static_cast<bf16_t>(b_value(c, kk));
        }
    }
}

static float sample_ref(int row, int col, int k_shard, int ranks) {
    float acc = 0.0f;
    for (int src = 0; src < ranks; ++src) {
        for (int kk = 0; kk < k_shard; ++kk) {
            const int global_k = src * k_shard + kk;
            const float av = static_cast<float>(static_cast<bf16_t>(a_value(src, row, kk)));
            const float bv = static_cast<float>(static_cast<bf16_t>(b_value(col, global_k)));
            acc += av * bv;
        }
    }
    return static_cast<float>(static_cast<bf16_t>(acc));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int M = 2048;
    int N = 8192;
    int K_SHARD = 1024;
    int warmup = 1;
    int iters = 3;
    bool validate = true;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--m") == 0) && i + 1 < argc) M = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n") == 0) && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k-shard") == 0 && i + 1 < argc) K_SHARD = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--no-validate") == 0) validate = false;
    }

    if (nranks != kRanks) {
        if (rank == 0) fprintf(stderr, "requires exactly %d ranks\n", kRanks);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (M != 2048 || N != 8192 || K_SHARD != 1024) {
        if (rank == 0) fprintf(stderr, "direct-read ideal path requires M=2048 N=8192 K_SHARD=1024\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    using Traits = opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>;
    const int K = K_SHARD * nranks;
    if (K_SHARD % Traits::B_K != 0 || ((K_SHARD / Traits::B_K) % 2) != 0) {
        if (rank == 0) fprintf(stderr, "unsupported K shard\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int ndev = 0;
    CHECK_HIP(hipGetDeviceCount(&ndev));
    CHECK_HIP(hipSetDevice(rank % ndev));

    ccoUniqueId uid;
    if (rank == 0) CHECK_CCO(ccoGetUniqueId(&uid));
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    ccoComm* comm = nullptr;
    CHECK_CCO(ccoCommCreate(uid, nranks, rank, kPerRankVmm, &comm));

    const size_t a_elems = static_cast<size_t>(M) * K_SHARD;
    const size_t b_elems = static_cast<size_t>(N) * K;
    const size_t c_elems = static_cast<size_t>(M) * N;
    static constexpr int kGridWgs = 256;
    const size_t recv_elems = static_cast<size_t>(nranks) * a_elems;
    const size_t ready_elems = static_cast<size_t>(nranks) * 2;

    auto h_a = std::make_unique<bf16_t[]>(a_elems);
    auto h_b = std::make_unique<bf16_t[]>(b_elems);
    fill_a(h_a.get(), rank, M, K_SHARD);
    fill_b(h_b.get(), N, K);

    bf16_t* d_a = nullptr;
    bf16_t* d_b = nullptr;
    bf16_t* d_c = nullptr;
    CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_c, c_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMemcpy(d_a, h_a.get(), a_elems * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.get(), b_elems * sizeof(bf16_t), hipMemcpyHostToDevice));

    ccoWindow_t recv_win = nullptr;
    ccoWindow_t ready_win = nullptr;
    void* recv_local = nullptr;
    void* ready_local = nullptr;
    CHECK_CCO(ccoWindowRegister(comm, recv_elems * sizeof(bf16_t), &recv_win, &recv_local));
    CHECK_CCO(ccoWindowRegister(comm, ready_elems * sizeof(unsigned int), &ready_win, &ready_local));

    opus_a2a_gemm_kargs kargs{};
    kargs.ptr_b = d_b;
    kargs.ptr_c = d_c;
    kargs.recv_a_win = recv_win;
    kargs.recv_a_local = recv_local;
    kargs.ready_win = ready_win;
    kargs.ready_local = ready_local;
    kargs.m = M;
    kargs.n = N;
    kargs.k_shard = K_SHARD;
    kargs.rank_count = nranks;
    kargs.my_rank = rank;
    kargs.stride_a = K_SHARD;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.output_bytes = static_cast<unsigned int>(c_elems * sizeof(bf16_t));

    dim3 grid(kGridWgs, 1, 1);
    dim3 block(Traits::BLOCK_SIZE, 1, 1);

    auto clear_for_launch = [&]() {
        CHECK_HIP(hipMemset(recv_local, 0, recv_elems * sizeof(bf16_t)));
        CHECK_HIP(hipMemset(ready_local, 0, ready_elems * sizeof(unsigned int)));
        CHECK_HIP(hipMemset(d_c, 0, c_elems * sizeof(bf16_t)));
        CHECK_HIP(hipMemcpy(static_cast<char*>(recv_local) + static_cast<size_t>(rank) * a_elems * sizeof(bf16_t),
                            d_a, a_elems * sizeof(bf16_t), hipMemcpyDeviceToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_CCO(ccoBarrierAll(comm));
    };

    auto launch_once = [&]() {
        a2a_gemm_lsa_direct_read_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP(hipGetLastError());
    };

    for (int i = 0; i < warmup; ++i) {
        clear_for_launch();
        launch_once();
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_CCO(ccoBarrierAll(comm));
    }

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        clear_for_launch();
        CHECK_HIP(hipEventRecord(start));
        launch_once();
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        CHECK_CCO(ccoBarrierAll(comm));
    }
    const double local_ms = static_cast<double>(total_ms) / iters;
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int mism = 0;
    if (validate) {
        auto h_c = std::make_unique<bf16_t[]>(c_elems);
        CHECK_HIP(hipMemcpy(h_c.get(), d_c, c_elems * sizeof(bf16_t), hipMemcpyDeviceToHost));
        const int sample_rows[] = {0, 17, 255, 511, 1023, 1536, 2047};
        const int sample_cols[] = {0, 127, 255, 1024, 4096, 8191};
        for (int r : sample_rows) {
            for (int c : sample_cols) {
                const float got = static_cast<float>(h_c[static_cast<size_t>(r) * N + c]);
                const float ref = sample_ref(r, c, K_SHARD, nranks);
                const float diff = std::fabs(got - ref);
                if (diff > 0.1f) {
                    if (mism < 8) {
                        printf("[rank %d] mismatch row=%d col=%d got=%f ref=%f diff=%f\n",
                               rank, r, c, got, ref, diff);
                    }
                    ++mism;
                }
            }
        }
    }

    int total_mism = 0;
    MPI_Reduce(&mism, &total_mism, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double flops = 2.0 * double(M) * double(N) * double(K);
        printf("a2a_gemm_direct_read M=%d N=%d K=%d ranks=%d grid_wgs=%d max_rank_time=%.4f ms %.2f TFLOP/s %s\n",
               M, N, K, nranks, kGridWgs, max_ms, flops / (max_ms * 1.0e9),
               (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_CCO(ccoWindowDeregister(comm, recv_win));
    CHECK_CCO(ccoWindowDeregister(comm, ready_win));
    CHECK_CCO(ccoCommDestroy(comm));
    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_c));
    MPI_Finalize();
    return (!validate || total_mism == 0) ? 0 : 1;
}
