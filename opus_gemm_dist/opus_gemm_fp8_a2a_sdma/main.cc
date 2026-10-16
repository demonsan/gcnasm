// Standalone 4-rank MPI host for the FP8 PTPC GEMM + SDMA all-to-all (poster block) kernels of this directory.
//   mpirun -n 4 build/fp8_a2a_sdma_<plan>.exe --case dense --M 2048 --puts-per-peer 1 --warmup 5 --iters 50
// One epoch = GEMM(grid = 1 poster + items WGs x 512) [+ quiet_ready(1 WG x 256) in the F0 build], one stream.
// The golden receiver (Triton norm_rope_cache / AITER fused_qknorm_idxrqknorm) is Python and is NOT part of this directory: this program validates the RECEIVED PACKED SLABS (R), not the receiver outputs.
#include <hip/hip_runtime.h>
#include <opus/hip_minimal.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>
#include "mori/cco/cco.hpp"

#include "lg_comm_abi_v2.h"

using namespace mori::cco;

#define CHECK_HIP(call)                                                                                 \
    do {                                                                                                \
        hipError_t status_ = (call);                                                                    \
        if (status_ != hipSuccess) {                                                                    \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                               \
        }                                                                                               \
    } while (0)
#define CHECK_CCO(call)                                                                                              \
    do {                                                                                                             \
        int cco_status_ = (call);                                                                                    \
        if (cco_status_ != 0) {                                                                                      \
            fprintf(stderr, "cco error %d (%s:%d): %s\n", cco_status_, __FILE__, __LINE__, #call);                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                            \
        }                                                                                                            \
    } while (0)

// Kernel entry points (defined in gemm_fp8_a2a_sdma_kernel.cc; extern "C" symbols, launched with hipLaunchKernelGGL like the sibling directories).
extern "C" __global__ void lg_r6x_gemm_n1_v1(LgCommArgsV2);
extern "C" __global__ void lg_r6x_gemm_n2_v1(LgCommArgsV2);
extern "C" __global__ void lg_r6x_gemm_n4_v1(LgCommArgsV2);
extern "C" __global__ void lg_r7_gemm_n2h_v1(LgCommArgsV2);
extern "C" __global__ void lg_r7_gemm_n4h_v1(LgCommArgsV2);
extern "C" __global__ void lg_quiet_ready_v2(LgCommArgsV2);
extern "C" __global__ void lg_drain_all_v2(LgCommArgsV2);

namespace {
constexpr int kNumRanks = 4, kK = 6144, kDenseN = 9216;
constexpr uint32_t kN = LG_X_N_RAW;       // 9856 (indexer build) / 9216 (dense build): the weight rows actually computed
constexpr uint32_t kF = LG_COMM_F;        // per-destination slab columns: 2560 (indexer build) / 2304 (dense build)

enum class Case { Dense, Indexer, Reuse };
const char* case_name(Case c) { return c == Case::Dense ? "dense" : (c == Case::Indexer ? "indexer" : "reuse"); }

// ---- deterministic inputs (splitmix64; identical on every rank for the same (tag, a, b)) ----
uint64_t mix(uint64_t x) { x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31); }
uint64_t hsh(uint64_t tag, uint64_t a, uint64_t b) { return mix(mix(mix(tag) ^ a) ^ b); }

// FP8 E4M3FN code. exact mode: 6 values {+-0.5, +-1, +-2} (every dot product of 6144 terms is exact in FP32 and in double); random mode: finite codes, exponent field 5..9 (|v| ~ 0.25..4).
uint8_t fp8_code(uint64_t h, bool exact) {
    if (exact) { static const uint8_t t[6] = {0x38, 0xB8, 0x40, 0xC0, 0x30, 0xB0}; return t[h % 6]; }
    return static_cast<uint8_t>(((h & 1) << 7) | ((5 + (h >> 1) % 5) << 3) | ((h >> 8) & 7));
}
float fp8_to_f(uint8_t v) {                       // E4M3FN decode (no inf; 0x7F/0xFF = NaN, never generated)
    const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    const float r = (e == 0) ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -r : r;
}
uint16_t f_to_bf16_rne(float f) {                 // IEEE RNE FP32 -> BF16 (matches opus::fp32_to_bf16)
    uint32_t b; std::memcpy(&b, &f, 4);
    if (~b & 0x7f800000u) b += 0x7fffu + ((b >> 16) & 1u); else if (b & 0xffffu) b |= 0x10000u;
    return static_cast<uint16_t>(b >> 16);
}
float bf16_to_f(uint16_t v) { uint32_t b = static_cast<uint32_t>(v) << 16; float f; std::memcpy(&f, &b, 4); return f; }

uint8_t a_code(int src, int m, int k, bool exact, int gen) { return fp8_code(hsh(0xA0 + gen, static_cast<uint64_t>(src) * 100003 + m, k), exact); }
uint8_t w_code(int n, int k, bool exact) { return fp8_code(hsh(0xB0, n, k), exact); }          // W is identical on every rank
float a_scale_v(int src, int m, bool exact, int gen) { return exact ? 1.0f : 0.5f + (hsh(0xA5 + gen, src, m) % 1000) / 777.0f; }
float w_scale_v(int n, bool exact) { return exact ? 1.0f : 0.25f + (hsh(0xB5, n, 0) % 1000) / 913.0f; }

// shuffle_weight(layout=(16,16)) physical offset of logical (n, k) in W[N, K]: view (N/16, K/32, 2, 16, 16) strides (16K, 512, 256, 16, 1)  [aiter shuffle_weight, byte-for-byte]
size_t shuf_off(int n, int k) { return static_cast<size_t>(n / 16) * 16 * kK + static_cast<size_t>(k / 32) * 512 + static_cast<size_t>((k % 32) / 16) * 256 + static_cast<size_t>(n % 16) * 16 + (k % 16); }

// golden select_qkv_from_peers column map: destination d's packed column c in [0, kF) -> raw GEMM column (Q | K | V [| IQ | IK], IK identical for every destination)
int packed_to_raw(int d, int c) {
    if (c < 2048) return d * 2048 + c;
    if (c < 2176) return 8192 + d * 128 + (c - 2048);
    if (c < 2304) return 8704 + d * 128 + (c - 2176);
    if (c < 2432) return 9216 + d * 128 + (c - 2304);         // IQ (indexer build only)
    return 9728 + (c - 2432);                                  // IK
}

uint32_t default_puts_per_peer(uint32_t M) { return M <= 2048u ? 1u : (M <= 4096u ? 2u : 4u); }   // r6b/r7 measured best n per T (dense); Evidence's per-T choice overrides

void usage(int rank) {
    if (rank != 0) return;
    fprintf(stderr,
        "usage: mpirun -n 4 fp8_a2a_sdma_<build>.exe [options]\n"
        "  --case dense|indexer|reuse   plan (must match the build: indexer needs the LG_X_INDEXER=1 build, dense/reuse the LG_X_INDEXER=0 build). default: dense\n"
        "  --M <rows per rank>          M = T/4, multiple of 256, <= 8192 (T = 4M: 8192/16384/32768 -> M 2048/4096/8192). default 2048\n"
        "  --puts-per-peer 1|2|4        SDMA puts (put groups) per peer per epoch; 0 = default by M (1/2/4). default 0\n"
        "  --half-first                 r7 half-N first group (needs --puts-per-peer 2|4)\n"
        "  --warmup <n> --iters <n>     default 5 / 50\n"
        "  --check-rows <n>             validated rows per source rank (all columns). default 16\n"
        "  --exact                      exact-arithmetic inputs ({+-0.5,+-1,+-2}, unit scales): byte-exact validation (default: random inputs, tolerance rtol=atol=0.02)\n"
        "  --poll-budget-us <us>        poster wait budget (default 50000)\n");
}
}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    Case cs = Case::Dense;
    uint32_t M = 2048, puts = 0, warmup = 5, iters = 50, check_rows = 16;
    uint64_t budget_us = 0;
    bool half_first = false, exact = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* { if (i + 1 >= argc) { if (rank == 0) fprintf(stderr, "%s needs a value\n", name); usage(rank); MPI_Abort(MPI_COMM_WORLD, 2); } return argv[++i]; };
        if (a == "--case" || a == "-c") {
            const std::string v = next("--case");
            if (v == "dense") cs = Case::Dense; else if (v == "indexer") cs = Case::Indexer; else if (v == "reuse") cs = Case::Reuse;
            else { if (rank == 0) fprintf(stderr, "--case must be dense, indexer or reuse\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
        } else if (a == "--M" || a == "-M") M = static_cast<uint32_t>(std::atoi(next("--M")));
        else if (a == "--puts-per-peer") puts = static_cast<uint32_t>(std::atoi(next("--puts-per-peer")));
        else if (a == "--half-first") half_first = true;
        else if (a == "--warmup") warmup = static_cast<uint32_t>(std::atoi(next("--warmup")));
        else if (a == "--iters") iters = static_cast<uint32_t>(std::atoi(next("--iters")));
        else if (a == "--check-rows") check_rows = static_cast<uint32_t>(std::atoi(next("--check-rows")));
        else if (a == "--poll-budget-us") budget_us = static_cast<uint64_t>(std::atoll(next("--poll-budget-us")));
        else if (a == "--exact") exact = true;
        else if (a == "-h" || a == "--help") { usage(rank); MPI_Finalize(); return 0; }
        else { if (rank == 0) fprintf(stderr, "unknown option %s\n", a.c_str()); usage(rank); MPI_Abort(MPI_COMM_WORLD, 2); }
    }
    if (nranks != kNumRanks) { if (rank == 0) fprintf(stderr, "this program needs exactly 4 ranks (got %d)\n", nranks); MPI_Abort(MPI_COMM_WORLD, 2); }
    if (M == 0 || M % 256 != 0 || M > 8192) { if (rank == 0) fprintf(stderr, "--M must be a multiple of 256 and <= 8192\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    const bool build_indexer = (LG_X_INDEXER != 0);
    if ((cs == Case::Indexer) != build_indexer) {
        if (rank == 0) fprintf(stderr, "case/build mismatch: --case %s needs the %s build (this binary: LG_X_INDEXER=%d, F=%u). Rebuild with `make PLAN=%s`.\n",
                               case_name(cs), cs == Case::Indexer ? "indexer" : "dense", static_cast<int>(LG_X_INDEXER), kF, cs == Case::Indexer ? "indexer" : "dense");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const uint32_t npp = puts ? puts : default_puts_per_peer(M);
    const uint32_t chunks = M / LG_CHUNK_ROWS;
    if (npp != 1u && npp != 2u && npp != 4u) { if (rank == 0) fprintf(stderr, "--puts-per-peer must be 1, 2 or 4\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    if (half_first && npp == 1u) { if (rank == 0) fprintf(stderr, "--half-first needs --puts-per-peer 2 or 4\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    if (chunks % npp != 0u) { if (rank == 0) fprintf(stderr, "chunks (M/256 = %u) must be divisible by --puts-per-peer %u\n", chunks, npp); MPI_Abort(MPI_COMM_WORLD, 2); }

    int ndev = 0;
    CHECK_HIP(hipGetDeviceCount(&ndev));
    CHECK_HIP(hipSetDevice(rank % ndev));                      // the kernels require rank == lsaRank == HIP device id: run on GPUs 0..3 of one node (use HIP_VISIBLE_DEVICES to pick the 4 GPUs)

    // ---------------- communicator + three equal-size windows (S staging, R receive, C control) ----------------
    ccoUniqueId uid;
    if (rank == 0) CHECK_CCO(ccoGetUniqueId(&uid));
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    ccoComm* comm = nullptr;
    constexpr size_t kPerRankVmm = 4ull << 30;                 // 4 GiB per-rank VMM slot (CCO aligns it to 4 GiB)
    CHECK_CCO(ccoCommCreate(uid, nranks, rank, kPerRankVmm, &comm));
    if (comm->worldSize != 4 || comm->lsaSize != 4 || static_cast<uint32_t>(comm->lsaRank) != static_cast<uint32_t>(rank)) {
        fprintf(stderr, "rank %d: topology guard failed (world %d lsa %d lsaRank %d): need 4 ranks on ONE node with rank == lsaRank\n", rank, comm->worldSize, comm->lsaSize, comm->lsaRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Window sizing (r8g): S, R and C get the SAME size = roundup(max(P + pad, control/stamp area), G), G = max(comm granularity, 4 MiB). The VMM allocator failed with hipMemSetAccess "invalid argument" for a
    // window whose size is not a multiple of 4 MiB (the reported 4096-B granule is queried without gpuDirectRDMACapable, which ccoMemAlloc sets); every size that worked is a multiple of 4 MiB.
    const uint64_t P = 8ull * M * kF;                          // 4*M*F BF16 bytes
    const uint64_t stamp_bytes = LGC_STAMP_OFF + 64ull * 4096ull;
    const uint64_t cap = std::max<uint64_t>(P, stamp_bytes);
    uint64_t gran = std::max<uint64_t>(static_cast<uint64_t>(comm->vmmGranularity), 4ull << 20);
    if (const char* ge = std::getenv("LG_WINDOW_GRAN_MIB")) { const uint64_t m = std::strtoull(ge, nullptr, 10); if (m > 0 && (m << 20) > gran) gran = m << 20; }
    const uint64_t need = std::max<uint64_t>(P + LG_R_PAD_BYTES, cap);
    const uint64_t Ws = (need + gran - 1) / gran * gran;
    ccoWindow_t S = nullptr, R = nullptr, C = nullptr;
    void *Sva = nullptr, *Rva = nullptr, *Cva = nullptr;
    CHECK_CCO(ccoWindowRegister(comm, static_cast<size_t>(Ws), &S, &Sva));
    CHECK_CCO(ccoWindowRegister(comm, static_cast<size_t>(Ws), &R, &Rva));
    CHECK_CCO(ccoWindowRegister(comm, static_cast<size_t>(Ws), &C, &Cva));
    ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
    reqs.sdmaQueueCount = 1;                                   // one SDMA queue per peer (more queues were slower in the sdma bench)
    ccoDevComm dev{};
    CHECK_CCO(ccoDevCommCreate(comm, &reqs, &dev));
    if (dev.sdma.sdmaNumQueue != 1) { fprintf(stderr, "rank %d: sdmaNumQueue %d != 1\n", rank, static_cast<int>(dev.sdma.sdmaNumQueue)); MPI_Abort(MPI_COMM_WORLD, 1); }

    // ---------------- deterministic per-rank inputs ----------------
    const size_t a_bytes = static_cast<size_t>(M) * kK, w_bytes = static_cast<size_t>(kN) * kK;
    std::vector<uint8_t> hA(a_bytes), hW(w_bytes);
    std::vector<float> hAs(M), hWs(kN);
    #pragma omp parallel for
    for (int m = 0; m < static_cast<int>(M); ++m) {
        for (int k = 0; k < kK; ++k) hA[static_cast<size_t>(m) * kK + k] = a_code(rank, m, k, exact, 0);
        hAs[m] = a_scale_v(rank, m, exact, 0);
    }
    #pragma omp parallel for
    for (int n = 0; n < static_cast<int>(kN); ++n) {
        for (int k = 0; k < kK; ++k) hW[shuf_off(n, k)] = w_code(n, k, exact);
        hWs[n] = w_scale_v(n, exact);
    }
    void *dA = nullptr, *dW = nullptr, *dAs = nullptr, *dWs = nullptr;
    CHECK_HIP(hipMalloc(&dA, a_bytes)); CHECK_HIP(hipMalloc(&dW, w_bytes));
    CHECK_HIP(hipMalloc(&dAs, M * sizeof(float))); CHECK_HIP(hipMalloc(&dWs, kN * sizeof(float)));
    CHECK_HIP(hipMemcpy(dA, hA.data(), a_bytes, hipMemcpyHostToDevice)); CHECK_HIP(hipMemcpy(dW, hW.data(), w_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dAs, hAs.data(), M * sizeof(float), hipMemcpyHostToDevice)); CHECK_HIP(hipMemcpy(dWs, hWs.data(), kN * sizeof(float), hipMemcpyHostToDevice));

    // ---------------- control window init (ep = 1, magic) + R sentinel ----------------
    auto init_control = [&]() {
        std::vector<uint8_t> img(LGC_BYTES, 0);
        *reinterpret_cast<uint64_t*>(img.data() + LGC_EP_OFF) = 1ull;
        *reinterpret_cast<uint64_t*>(img.data() + LGC_MAGIC_OFF) = LGC_MAGIC;
        CHECK_HIP(hipMemcpy(Cva, img.data(), img.size(), hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
    };
    init_control();
    CHECK_HIP(hipMemset(Rva, 0xFF, Ws));                       // sentinel: an unwritten R element can never pass the check (0xFFFF is a BF16 NaN)
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);                               // all windows/control initialised before any rank PUTs or releases credit

    // ---------------- launch plumbing ----------------
    LgCommArgsV2 args; std::memset(&args, 0, sizeof args);
    args.dev = dev; args.Swin = S; args.Rwin = R; args.Cwin = C;
    args.A = dA; args.W = dW; args.Ascale = static_cast<const float*>(dAs); args.Wscale = static_cast<const float*>(dWs);
    args.M = M; args.N = kN; args.K = kK; args.strideA = kK; args.F = kF; args.rank = static_cast<uint32_t>(rank);
    args.chunks = chunks; args.abi_version = LG_COMM_ABI_VERSION; args.poll_iterations = 2000000;
    args.sdma_queues = 1; args.items_per_group = LG_ITEMS_PER_GROUP;
    args.reserved[0] = LG_R6_PLAN_CODE_N(npp, half_first ? 1 : 0);
    args.reserved[1] = budget_us ? budget_us : LG_POLL_BUDGET_US_DEFAULT;
    const uint32_t hg = half_first ? 1u : 0u;
    // grid = 1 poster block + items (36 dense / 41 indexer per chunk; half-first adds 3*9*(chunks/n) items)
    const uint32_t grid = LG_X_ITEMS_PER_CHUNK * chunks + (hg ? LG_NUM_PEERS * LG_TILES_N_PER_DEST * (chunks / npp) : 0u) + 1u;
    hipStream_t stream; CHECK_HIP(hipStreamCreate(&stream));
    auto launch_epoch = [&]() {
        if (hg)  { if (npp == 2u) hipLaunchKernelGGL(lg_r7_gemm_n2h_v1, dim3(grid), dim3(512), 0, stream, args); else hipLaunchKernelGGL(lg_r7_gemm_n4h_v1, dim3(grid), dim3(512), 0, stream, args); }
        else     { if (npp == 1u) hipLaunchKernelGGL(lg_r6x_gemm_n1_v1, dim3(grid), dim3(512), 0, stream, args); else if (npp == 2u) hipLaunchKernelGGL(lg_r6x_gemm_n2_v1, dim3(grid), dim3(512), 0, stream, args); else hipLaunchKernelGGL(lg_r6x_gemm_n4_v1, dim3(grid), dim3(512), 0, stream, args); }
#if !LG_R8_FOLD
        hipLaunchKernelGGL(lg_quiet_ready_v2, dim3(1), dim3(256), 0, stream, args);      // F0 (production): W2 + device reset + epoch advance in a separate 1-WG kernel
#endif
    };
    auto read_report = [&](uint64_t* ep, uint32_t* poison, uint64_t* epochs_done) {
        std::vector<uint8_t> img(LGC_BYTES);
        CHECK_HIP(hipMemcpy(img.data(), Cva, img.size(), hipMemcpyDeviceToHost));
        *ep = *reinterpret_cast<uint64_t*>(img.data() + LGC_EP_OFF);
        *poison = *reinterpret_cast<uint32_t*>(img.data() + LGC_POISON_OFF);
        *epochs_done = *reinterpret_cast<uint64_t*>(img.data() + LGC_REPORT_OFF + 8 * 5);
    };
    // A2-H recovery (mandatory after ANY poisoned epoch on ANY rank): drain_all on every rank -> barrier -> init_control on every rank (zeroes the whole control window) -> barrier.
    auto recover = [&]() {
        LgCommArgsV2 d; std::memset(&d, 0, sizeof d);
        d.dev = dev; d.Swin = S; d.Rwin = R; d.Cwin = C; d.rank = static_cast<uint32_t>(rank); d.sdma_queues = 1; d.abi_version = LG_COMM_ABI_VERSION;
        hipLaunchKernelGGL(lg_drain_all_v2, dim3(1), dim3(64), 0, stream, d);
        CHECK_HIP(hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        init_control();
        MPI_Barrier(MPI_COMM_WORLD);
    };
    auto check_poison_all = [&](const char* where) -> bool {
        uint64_t ep, done; uint32_t poison;
        read_report(&ep, &poison, &done);
        uint32_t any = 0, mine = poison;
        MPI_Allreduce(&mine, &any, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
        if (poison) fprintf(stderr, "rank %d: POISON %u at %s (ep=%llu epochs_done=%llu)\n", rank, poison, where, static_cast<unsigned long long>(ep), static_cast<unsigned long long>(done));
        return any != 0;
    };

    // ---------------- one validated epoch ----------------
    launch_epoch();
    CHECK_HIP(hipStreamSynchronize(stream));
    if (check_poison_all("validation epoch")) { recover(); if (rank == 0) fprintf(stderr, "FAILED: poisoned validation epoch (control window re-initialised; see per-rank POISON lines)\n"); MPI_Abort(MPI_COMM_WORLD, 3); }

    // Validate R: for every source rank s and a sample of rows, ALL kF packed columns of the slab this rank received. Reference = FP64 accumulate over the dequantised inputs, then (acc*Ws)*As in FP32 and ONE BF16 RNE.
    // (exact inputs: the dot product is exact, expect byte-equality; random inputs: tolerance rtol=atol=0.02, the golden compare_outputs rule). The reuse case reads the compact [4M,2304] R in the strided [T,2560] view layout of the golden receiver: row stride 2304.
    uint64_t checked = 0, bad = 0, byte_diff = 0;
    double max_abs = 0.0;
    const uint32_t rows_per_src = std::min<uint32_t>(check_rows, M);
    std::vector<uint16_t> hrow(kF);
    for (int s = 0; s < kNumRanks; ++s) {
        for (uint32_t r = 0; r < rows_per_src; ++r) {
            const int m = static_cast<int>((r * 2654435761u + 17u * static_cast<uint32_t>(s)) % M);     // deterministic spread, includes first/last chunks over rows
            std::vector<float> arow(kK);
            for (int k = 0; k < kK; ++k) arow[k] = fp8_to_f(a_code(s, m, k, exact, 0));
            const float as = a_scale_v(s, m, exact, 0);
            // device row: R[(s*M + m)][0..kF)
            CHECK_HIP(hipMemcpy(hrow.data(), static_cast<uint8_t*>(Rva) + (static_cast<size_t>(s) * M + m) * kF * sizeof(uint16_t), kF * sizeof(uint16_t), hipMemcpyDeviceToHost));
            std::vector<uint16_t> want(kF);
            #pragma omp parallel for
            for (int c = 0; c < static_cast<int>(kF); ++c) {
                const int n = packed_to_raw(rank, c);
                double acc = 0.0;
                for (int k = 0; k < kK; ++k) acc += static_cast<double>(arow[k]) * static_cast<double>(fp8_to_f(w_code(n, k, exact)));
                const float v = static_cast<float>(acc);                                              // FP32 accumulator value
                const float sc = static_cast<float>(v * w_scale_v(n, exact));                         // (acc*Ws)
                want[c] = f_to_bf16_rne(static_cast<float>(sc * as));                                 // *As, one BF16 RNE
            }
            for (uint32_t c = 0; c < kF; ++c) {
                ++checked;
                const float g = bf16_to_f(hrow[c]), w = bf16_to_f(want[c]);
                if (hrow[c] != want[c]) ++byte_diff;
                const double d = std::fabs(static_cast<double>(g) - static_cast<double>(w));
                max_abs = std::max(max_abs, std::isfinite(d) ? d : 1e30);
                const bool ok = std::isfinite(g) && d <= 0.02 + 0.02 * std::fabs(static_cast<double>(w));
                if (!ok) ++bad;
            }
        }
    }
    uint64_t tot[3] = {checked, bad, byte_diff}, all[3] = {0, 0, 0};
    MPI_Allreduce(tot, all, 3, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    double gmax = 0.0; MPI_Allreduce(&max_abs, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const bool valid = (all[1] == 0) && (!exact || all[2] == 0);
    if (rank == 0) printf("validate [%s, %s inputs]: %llu elements checked over %u rows/source x %d sources; tolerance failures %llu; bytes differing from the reference %llu; max |diff| %.4g -> %s\n",
                          case_name(cs), exact ? "exact" : "random", static_cast<unsigned long long>(all[0]), rows_per_src, kNumRanks, static_cast<unsigned long long>(all[1]), static_cast<unsigned long long>(all[2]), gmax, valid ? "PASS" : "FAIL");
    if (!valid) MPI_Abort(MPI_COMM_WORLD, 4);

    // ---------------- timing: hipEvents around `iters` back-to-back epochs (GEMM + SDMA chain, one stream) ----------------
    hipEvent_t e0, e1; CHECK_HIP(hipEventCreate(&e0)); CHECK_HIP(hipEventCreate(&e1));
    for (uint32_t i = 0; i < warmup; ++i) launch_epoch();
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    CHECK_HIP(hipEventRecord(e0, stream));
    for (uint32_t i = 0; i < iters; ++i) launch_epoch();
    CHECK_HIP(hipEventRecord(e1, stream));
    CHECK_HIP(hipEventSynchronize(e1));
    float ms = 0.f; CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
    const double us_epoch = 1000.0 * ms / std::max<uint32_t>(iters, 1);
    double us_max = 0.0, us_sum = 0.0;
    MPI_Allreduce(&us_epoch, &us_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&us_epoch, &us_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double out_bytes = 3.0 * M * kF * 2.0;               // bytes each rank PUTs per epoch (3 remote slabs of M x F BF16)
    const bool poisoned = check_poison_all("timing");
    uint64_t ep, done; uint32_t poison; read_report(&ep, &poison, &done);
    if (rank == 0) {
        printf("config: case=%s build(F=%u N=%u fold=%d) M=%u (T=%u) puts_per_peer=%u half_first=%d grid=%u warmup=%u iters=%u\n",
               case_name(cs), kF, kN, static_cast<int>(LG_R8_FOLD), M, 4 * M, npp, half_first ? 1 : 0, grid, warmup, iters);
        printf("epoch (GEMM + SDMA a2a + %s): max over ranks %.1f us, mean %.1f us; per-rank PUT %.2f MB/epoch -> %.1f GB/s per rank (%.1f GB/s aggregate over 4 ranks)\n",
               LG_R8_FOLD ? "folded W2/reset" : "quiet_ready", us_max, us_sum / kNumRanks, out_bytes / 1e6, out_bytes / (us_max * 1e3), 4.0 * out_bytes / (us_max * 1e3));
        printf("rank 0 report: ep=%llu epochs_done=%llu poison=%u\n", static_cast<unsigned long long>(ep), static_cast<unsigned long long>(done), poison);
    }
    (void)dev; (void)kDenseN;
    CHECK_HIP(hipEventDestroy(e0)); CHECK_HIP(hipEventDestroy(e1)); CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dW)); CHECK_HIP(hipFree(dAs)); CHECK_HIP(hipFree(dWs));
    MPI_Barrier(MPI_COMM_WORLD);
    CHECK_CCO(ccoWindowDeregister(comm, C)); CHECK_CCO(ccoWindowDeregister(comm, R)); CHECK_CCO(ccoWindowDeregister(comm, S));
    CHECK_CCO(ccoCommDestroy(comm));
    MPI_Finalize();
    return poisoned ? 5 : 0;
}