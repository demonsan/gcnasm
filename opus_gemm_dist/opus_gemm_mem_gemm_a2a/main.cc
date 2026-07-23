#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "mori/cco/cco.hpp"
#include "opus_gemm_mem_gemm_a2a.h"

using namespace mori::cco;

using ncclResult_t = int;
using ncclComm_t = void*;
static constexpr ncclResult_t ncclSuccess = 0;
static constexpr int ncclBfloat16 = 9;
struct ncclUniqueId {
    char internal[128];
};
extern "C" ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
extern "C" ncclResult_t ncclCommDestroy(ncclComm_t comm);
extern "C" const char* ncclGetErrorString(ncclResult_t result);
extern "C" ncclResult_t ncclGroupStart();
extern "C" ncclResult_t ncclGroupEnd();
extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count, int datatype, int peer, ncclComm_t comm, hipStream_t stream);
extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count, int datatype, int peer, ncclComm_t comm, hipStream_t stream);

#define CHECK_HIP_ABORT(call)                                                                              \
    do {                                                                                                   \
        hipError_t status_ = (call);                                                                       \
        if (status_ != hipSuccess) {                                                                       \
            std::fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                  \
        }                                                                                                  \
    } while (0)

#define CHECK_NCCL(call)                                                                                   \
    do {                                                                                                   \
        ncclResult_t status_ = (call);                                                                     \
        if (status_ != ncclSuccess) {                                                                      \
            std::fprintf(stderr, "RCCL error (%s:%d): %s\n", __FILE__, __LINE__, ncclGetErrorString(status_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                  \
        }                                                                                                  \
    } while (0)

#define CHECK_CCO(call)                                                                    \
    do {                                                                                   \
        int cco_status_ = (call);                                                          \
        if (cco_status_ != 0) {                                                            \
            std::fprintf(stderr, "CCO error %d (%s:%d): %s\n", cco_status_, __FILE__, __LINE__, \
                         #call);                                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                   \
        }                                                                                  \
    } while (0)

static constexpr size_t kPerRankVmm = 512ULL * 1024 * 1024;

__global__ void pack_c_shards_kernel(const bf16_t* __restrict__ c,
                                     bf16_t* __restrict__ packed,
                                     int m,
                                     int n,
                                     int shard_n,
                                     int ranks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * shard_n * ranks;
    const int grid_threads = gridDim.x * blockDim.x;
    for (int idx = tid; idx < total; idx += grid_threads) {
        const int local_col = idx % shard_n;
        const int t0 = idx / shard_n;
        const int row = t0 % m;
        const int peer = t0 / m;
        const int global_col = peer * shard_n + local_col;
        packed[(static_cast<size_t>(peer) * m + row) * shard_n + local_col] =
            c[static_cast<size_t>(row) * n + global_col];
    }
}

static float a_value(int rank, int row, int k) {
    return 0.001f * float(rank + 1) + 0.0002f * float((row % 17) - 8) +
           0.0001f * float((k % 29) - 14);
}

static float b_value(int col, int k) {
    return 0.0003f * float((col % 23) - 11) + 0.0001f * float((k % 31) - 15);
}

static void fill_a(bf16_t* a, int rank, int m, int k) {
#pragma omp parallel for collapse(2)
    for (int row = 0; row < m; ++row) {
        for (int kk = 0; kk < k; ++kk) {
            a[static_cast<size_t>(row) * k + kk] = static_cast<bf16_t>(a_value(rank, row, kk));
        }
    }
}

static void fill_b(bf16_t* b, int n, int k) {
#pragma omp parallel for collapse(2)
    for (int col = 0; col < n; ++col) {
        for (int kk = 0; kk < k; ++kk) {
            b[static_cast<size_t>(col) * k + kk] = static_cast<bf16_t>(b_value(col, kk));
        }
    }
}

static float sample_ref(int src_rank, int row, int col, int k) {
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const float av = static_cast<float>(static_cast<bf16_t>(a_value(src_rank, row, kk)));
        const float bv = static_cast<float>(static_cast<bf16_t>(b_value(col, kk)));
        acc += av * bv;
    }
    return static_cast<float>(static_cast<bf16_t>(acc));
}

static void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [-m M] [-n N] [-k K] [--split-k S] [--shard-n SN] "
        "[--warmup W] [--iters I] [--mode 0|1] [--no-validate]\n"
        "  mode 0: splitk_reduce direct LSA A2A\n"
        "  mode 1: splitk_reduce + RCCL A2A reference\n",
        prog);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int M = 1024;
    int N = 512;
    int K = 7168;
    int split_k = 2;
    int shard_n = 0;
    int warmup = 1;
    int iters = 5;
    int mode = 0;
    bool validate = true;

    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--m") == 0) && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n") == 0) && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-k") == 0 || std::strcmp(argv[i], "--k") == 0) && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--split-k") == 0 && i + 1 < argc) {
            split_k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--shard-n") == 0 && i + 1 < argc) {
            shard_n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-validate") == 0) {
            validate = false;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    if (shard_n == 0) {
        if (N % nranks != 0) {
            if (rank == 0) std::fprintf(stderr, "N must be divisible by ranks when --shard-n is omitted\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        shard_n = N / nranks;
    }

    using Traits = opus_mem_gemm_a2a_traits;
    const int padded_M = ceil_div(M, Traits::B_M) * Traits::B_M;
    const int padded_N = ceil_div(N, Traits::B_N) * Traits::B_N;
    const int num_tiles_m = ceil_div(M, Traits::B_M);
    const int num_tiles_n = ceil_div(N, Traits::B_N);
    const int scatter_n = shard_n * nranks;

    if (M % Traits::B_M != 0 || N % Traits::B_N != 0 || K % Traits::B_K != 0 ||
        split_k <= 0 || split_k > ceil_div(K, Traits::B_K) || scatter_n != N ||
        (shard_n % OPUS_MEM_GEMM_A2A_REDUCE_VEC) != 0 || warmup < 0 || iters <= 0 ||
        (mode != 0 && mode != 1)) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "unsupported shape/config: M,N,K must align to (%d,%d,%d), "
                         "split_k in [1,%d], shard_n*ranks == N, shard_n %% %d == 0, "
                         "iters > 0, mode in {0,1}\n",
                         Traits::B_M, Traits::B_N, Traits::B_K, ceil_div(K, Traits::B_K),
                         OPUS_MEM_GEMM_A2A_REDUCE_VEC);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int ndev = 0;
    CHECK_HIP_ABORT(hipGetDeviceCount(&ndev));
    CHECK_HIP_ABORT(hipSetDevice(rank % ndev));

    ncclComm_t nccl_comm = nullptr;
    if (mode == 1) {
        ncclUniqueId nccl_uid;
        if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&nccl_uid));
        MPI_Bcast(&nccl_uid, sizeof(nccl_uid), MPI_BYTE, 0, MPI_COMM_WORLD);
        CHECK_NCCL(ncclCommInitRank(&nccl_comm, nranks, nccl_uid, rank));
    }

    ccoComm* cco_comm = nullptr;
    ccoWindow_t a2a_win = nullptr;
    void* a2a_local = nullptr;
    if (mode == 0) {
        ccoUniqueId cco_uid;
        if (rank == 0) CHECK_CCO(ccoGetUniqueId(&cco_uid));
        MPI_Bcast(&cco_uid, sizeof(cco_uid), MPI_BYTE, 0, MPI_COMM_WORLD);
        CHECK_CCO(ccoCommCreate(cco_uid, nranks, rank, kPerRankVmm, &cco_comm));
    }

    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(N) * K;
    const size_t c_elems = static_cast<size_t>(M) * N;
    const size_t ws_elems = static_cast<size_t>(split_k) * padded_M * padded_N;
    const size_t shard_elems = static_cast<size_t>(M) * shard_n;
    const size_t a2a_elems = static_cast<size_t>(nranks) * shard_elems;

    if (mode == 0) {
        CHECK_CCO(ccoWindowRegister(cco_comm, a2a_elems * sizeof(bf16_t), &a2a_win, &a2a_local));
    }

    auto h_a = std::make_unique<bf16_t[]>(a_elems);
    auto h_b = std::make_unique<bf16_t[]>(b_elems);
    fill_a(h_a.get(), rank, M, K);
    fill_b(h_b.get(), N, K);

    bf16_t* d_a = nullptr;
    bf16_t* d_b = nullptr;
    bf16_t* d_c = nullptr;
    bf16_t* d_send = nullptr;
    bf16_t* d_recv = nullptr;
    float* d_workspace = nullptr;
    opus_splitk_ws_handle* d_ws_handle = nullptr;

    CHECK_HIP_ABORT(hipMalloc(&d_a, a_elems * sizeof(bf16_t)));
    CHECK_HIP_ABORT(hipMalloc(&d_b, b_elems * sizeof(bf16_t)));
    CHECK_HIP_ABORT(hipMalloc(&d_c, c_elems * sizeof(bf16_t)));
    CHECK_HIP_ABORT(hipMalloc(&d_send, a2a_elems * sizeof(bf16_t)));
    CHECK_HIP_ABORT(hipMalloc(&d_recv, a2a_elems * sizeof(bf16_t)));
    CHECK_HIP_ABORT(hipMalloc(&d_workspace, ws_elems * sizeof(float)));
    CHECK_HIP_ABORT(hipMalloc(&d_ws_handle, sizeof(opus_splitk_ws_handle)));
    CHECK_HIP_ABORT(hipMemcpy(d_a, h_a.get(), a_elems * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP_ABORT(hipMemcpy(d_b, h_b.get(), b_elems * sizeof(bf16_t), hipMemcpyHostToDevice));

    opus_splitk_ws_handle h_ws_handle{};
    h_ws_handle.ptr = d_workspace;
    h_ws_handle.bytes = ws_elems * sizeof(float);
    CHECK_HIP_ABORT(hipMemcpy(d_ws_handle, &h_ws_handle, sizeof(h_ws_handle), hipMemcpyHostToDevice));

    opus_gemm_flatmm_splitk_kargs_gfx950 gemm_args{};
    gemm_args.ptr_a = d_a;
    gemm_args.ptr_b = d_b;
    gemm_args.ws_handle = d_ws_handle;
    gemm_args.ptr_c = d_c;
    gemm_args.ptr_bias = nullptr;
    gemm_args.m = M;
    gemm_args.n = N;
    gemm_args.k = K;
    gemm_args.batch = 1;
    gemm_args.split_k = split_k;
    gemm_args.stride_a = K;
    gemm_args.stride_b = K;
    gemm_args.stride_ws = padded_N;
    gemm_args.stride_c = N;
    gemm_args.stride_a_batch = static_cast<int>(a_elems);
    gemm_args.stride_b_batch = static_cast<int>(b_elems);
    gemm_args.stride_ws_batch = padded_M * padded_N;
    gemm_args.stride_c_batch = M * N;
    gemm_args.stride_bias_batch = 0;

    dim3 gemm_grid(split_k * num_tiles_m * num_tiles_n, 1, 1);
    dim3 gemm_block(Traits::BLOCK_SIZE, 1, 1);
    dim3 reduce_grid(ceil_div(N, OPUS_MEM_GEMM_A2A_REDUCE_VEC * OPUS_MEM_GEMM_A2A_REDUCE_BLOCK), M, 1);
    dim3 reduce_block(OPUS_MEM_GEMM_A2A_REDUCE_BLOCK, 1, 1);
    dim3 pack_grid(ceil_div(static_cast<int>(a2a_elems), 256), 1, 1);
    dim3 pack_block(256, 1, 1);

    auto clear_buffers = [&]() {
        CHECK_HIP_ABORT(hipMemset(d_workspace, 0, ws_elems * sizeof(float)));
        CHECK_HIP_ABORT(hipMemset(d_c, 0, c_elems * sizeof(bf16_t)));
        CHECK_HIP_ABORT(hipMemset(d_send, 0, a2a_elems * sizeof(bf16_t)));
        CHECK_HIP_ABORT(hipMemset(d_recv, 0, a2a_elems * sizeof(bf16_t)));
        if (mode == 0) {
            CHECK_HIP_ABORT(hipMemset(a2a_local, 0, a2a_elems * sizeof(bf16_t)));
        }
    };

    auto launch_once = [&]() {
        gemm_a16w16_flatmm_splitk_kernel<Traits><<<gemm_grid, gemm_block>>>(gemm_args);
        CHECK_HIP_ABORT(hipGetLastError());
        if (mode == 0) {
            splitk_reduce_a2a_kernel<
                OPUS_MEM_GEMM_A2A_REDUCE_VEC,
                OPUS_MEM_GEMM_A2A_REDUCE_BLOCK,
                bf16_t><<<reduce_grid, reduce_block>>>(
                d_ws_handle, a2a_win, split_k, M, N, 1, padded_M, padded_N, shard_n, nranks);
            CHECK_HIP_ABORT(hipGetLastError());
        } else {
            splitk_reduce_kernel<
                OPUS_MEM_GEMM_A2A_REDUCE_VEC,
                OPUS_MEM_GEMM_A2A_REDUCE_BLOCK,
                bf16_t,
                false,
                bf16_t,
                false><<<reduce_grid, reduce_block>>>(
                d_ws_handle, d_c, split_k, M, N, 1, padded_M, padded_N, nullptr, 0);
            CHECK_HIP_ABORT(hipGetLastError());
            pack_c_shards_kernel<<<pack_grid, pack_block>>>(d_c, d_send, M, N, shard_n, nranks);
            CHECK_HIP_ABORT(hipGetLastError());
            CHECK_NCCL(ncclGroupStart());
            for (int peer = 0; peer < nranks; ++peer) {
                CHECK_NCCL(ncclSend(d_send + static_cast<size_t>(peer) * shard_elems,
                                    shard_elems, ncclBfloat16, peer, nccl_comm, nullptr));
                CHECK_NCCL(ncclRecv(d_recv + static_cast<size_t>(peer) * shard_elems,
                                    shard_elems, ncclBfloat16, peer, nccl_comm, nullptr));
            }
            CHECK_NCCL(ncclGroupEnd());
        }
    };

    for (int i = 0; i < warmup; ++i) {
        clear_buffers();
        CHECK_HIP_ABORT(hipDeviceSynchronize());
        if (mode == 0) CHECK_CCO(ccoBarrierAll(cco_comm));
        launch_once();
        CHECK_HIP_ABORT(hipDeviceSynchronize());
        if (mode == 0) CHECK_CCO(ccoBarrierAll(cco_comm));
        MPI_Barrier(MPI_COMM_WORLD);
    }

    hipEvent_t start, stop;
    CHECK_HIP_ABORT(hipEventCreate(&start));
    CHECK_HIP_ABORT(hipEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        clear_buffers();
        CHECK_HIP_ABORT(hipDeviceSynchronize());
        if (mode == 0) CHECK_CCO(ccoBarrierAll(cco_comm));
        CHECK_HIP_ABORT(hipEventRecord(start));
        launch_once();
        CHECK_HIP_ABORT(hipEventRecord(stop));
        CHECK_HIP_ABORT(hipEventSynchronize(stop));
        if (mode == 0) CHECK_CCO(ccoBarrierAll(cco_comm));
        float ms = 0.0f;
        CHECK_HIP_ABORT(hipEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    const double local_ms = static_cast<double>(total_ms) / iters;
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int mism = 0;
    if (validate) {
        auto h_recv = std::make_unique<bf16_t[]>(a2a_elems);
        void* result_dev = mode == 0 ? a2a_local : static_cast<void*>(d_recv);
        CHECK_HIP_ABORT(hipMemcpy(h_recv.get(), result_dev, a2a_elems * sizeof(bf16_t), hipMemcpyDeviceToHost));
        const int sample_rows[] = {0, 17, 255, 511, 1023, 1536, 2047};
        const int sample_cols[] = {0, 127, 255, 511, 1023};
        for (int src = 0; src < nranks; ++src) {
            for (int row : sample_rows) {
                if (row >= M) continue;
                for (int local_col : sample_cols) {
                    if (local_col >= shard_n) continue;
                    const int global_col = rank * shard_n + local_col;
                    const float got = static_cast<float>(
                        h_recv[(static_cast<size_t>(src) * M + row) * shard_n + local_col]);
                    const float ref = sample_ref(src, row, global_col, K);
                    const float diff = std::fabs(got - ref);
                    if (diff > 0.2f) {
                        if (mism < 8) {
                            std::printf("[rank %d] mismatch src=%d row=%d col=%d got=%f ref=%f diff=%f\n",
                                        rank, src, row, global_col, got, ref, diff);
                        }
                        ++mism;
                    }
                }
            }
        }
    }

    int total_mism = 0;
    MPI_Reduce(&mism, &total_mism, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double flops = 2.0 * double(M) * double(N) * double(K) * double(nranks);
        const double a2a_gb = double(nranks) * double(a2a_elems) * sizeof(bf16_t) / 1.0e9;
        std::printf("opus_gemm_mem_gemm_a2a M=%d N=%d K=%d ranks=%d split_k=%d shard_n=%d mode=%d "
                    "max_rank_time=%.4f ms aggregate=%.2f TFLOP/s a2a_payload=%.3f GB %s\n",
                    M, N, K, nranks, split_k, shard_n, mode, max_ms, flops / (max_ms * 1.0e9), a2a_gb,
                    (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
    }

    CHECK_HIP_ABORT(hipEventDestroy(start));
    CHECK_HIP_ABORT(hipEventDestroy(stop));
    if (nccl_comm) CHECK_NCCL(ncclCommDestroy(nccl_comm));
    if (a2a_win) CHECK_CCO(ccoWindowDeregister(cco_comm, a2a_win));
    if (cco_comm) CHECK_CCO(ccoCommDestroy(cco_comm));
    CHECK_HIP_ABORT(hipFree(d_a));
    CHECK_HIP_ABORT(hipFree(d_b));
    CHECK_HIP_ABORT(hipFree(d_c));
    CHECK_HIP_ABORT(hipFree(d_send));
    CHECK_HIP_ABORT(hipFree(d_recv));
    CHECK_HIP_ABORT(hipFree(d_workspace));
    CHECK_HIP_ABORT(hipFree(d_ws_handle));
    MPI_Finalize();
    return (!validate || total_mism == 0) ? 0 : 1;
}

