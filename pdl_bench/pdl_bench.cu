#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

constexpr int BLOCK_SIZE = 256;

// Prevent compiler from optimizing away loops
__device__ __forceinline__ float dummy_compute(float x, int iters) {
    float acc = x;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        acc = fmaf(acc, 1.0001f, float(i & 0xFF));
    }
    return acc;
}

// Producer: small write, trigger PDL, then long tail compute
__global__ void producer_kernel(float* __restrict__ out, int N, int tail_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = 1.0f;
    }
    cudaTriggerProgrammaticLaunchCompletion();
    if (tail_iters > 0) {
        float acc = dummy_compute(__int_as_float(idx), tail_iters);
        if (acc == 1.23456789e30f && idx < N) {
            out[idx] = acc;
        }
    }
}

// Consumer: long head compute (independent), sync, then small dependent work
__global__ void consumer_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                int N, int head_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    if (head_iters > 0) {
        acc = dummy_compute(__int_as_float(idx ^ 0x1A2B3C4D), head_iters);
    }
    cudaGridDependencySynchronize();
    if (idx < N) {
        out[idx] = in[idx] + acc;
    }
}

static void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CHECK(x) checkCuda(x, __FILE__, __LINE__)

// Helper: build a graph containing \'iters\' kernel pairs via stream capture
void build_graph(bool use_pdl, int iters,
                 dim3 gdim, dim3 block,
                 float* d_producer_out, float* d_consumer_out,
                 int N, int tail_iters, int head_iters,
                 cudaGraph_t& graph, cudaGraphExec_t& exec) {
    cudaStream_t capStream;
    CHECK(cudaStreamCreate(&capStream));
    CHECK(cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal));

    for (int i = 0; i < iters; ++i) {
        producer_kernel<<<gdim, block, 0, capStream>>>(d_producer_out, N, tail_iters);
        if (use_pdl) {
            cudaLaunchConfig_t config = {};
            config.gridDim = gdim;
            config.blockDim = block;
            config.dynamicSmemBytes = 0;
            config.stream = capStream;
            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            config.attrs = &attr;
            config.numAttrs = 1;
            CHECK(cudaLaunchKernelEx(&config, consumer_kernel,
                                     d_producer_out, d_consumer_out, N, head_iters));
        } else {
            consumer_kernel<<<gdim, block, 0, capStream>>>(d_producer_out, d_consumer_out, N, head_iters);
        }
    }

    CHECK(cudaStreamEndCapture(capStream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    CHECK(cudaStreamDestroy(capStream));
}

int main(int argc, char** argv) {
    int N = 1 << 16;
    int tail_iters = 16384;
    int head_iters = 16384;
    int warmup = 10;
    int iters = 100;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) tail_iters = atoi(argv[2]);
    if (argc > 3) head_iters = atoi(argv[3]);
    if (argc > 4) iters = atoi(argv[4]);

    int dev = 0;
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (sm%d%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        printf("ERROR: PDL requires Hopper (sm90+) or newer.\n");
        return 1;
    }

    size_t bytes = static_cast<size_t>(N) * sizeof(float);
    float *d_producer_out, *d_consumer_out;
    CHECK(cudaMalloc(&d_producer_out, bytes));
    CHECK(cudaMalloc(&d_consumer_out, bytes));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 gdim(grid);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    auto run_baseline = [&]() {
        producer_kernel<<<gdim, block, 0, stream>>>(d_producer_out, N, tail_iters);
        consumer_kernel<<<gdim, block, 0, stream>>>(d_producer_out, d_consumer_out, N, head_iters);
        CHECK(cudaGetLastError());
    };

    auto run_pdl = [&]() {
        producer_kernel<<<gdim, block, 0, stream>>>(d_producer_out, N, tail_iters);
        cudaLaunchConfig_t config = {};
        config.gridDim = gdim;
        config.blockDim = block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        config.attrs = &attr;
        config.numAttrs = 1;
        CHECK(cudaLaunchKernelEx(&config, consumer_kernel,
                                 d_producer_out, d_consumer_out, N, head_iters));
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        run_baseline();
    }
    CHECK(cudaStreamSynchronize(stream));

    // --- Baseline (no PDL, no Graph) ---
    CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        run_baseline();
    }
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaStreamSynchronize(stream));
    float baseline_ms = 0;
    CHECK(cudaEventElapsedTime(&baseline_ms, start, stop));

    // --- PDL only ---
    CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        run_pdl();
    }
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaStreamSynchronize(stream));
    float pdl_ms = 0;
    CHECK(cudaEventElapsedTime(&pdl_ms, start, stop));

    // --- CUDA Graph baseline ---
    cudaGraph_t graph_base;
    cudaGraphExec_t exec_base;
    build_graph(false, iters, gdim, block, d_producer_out, d_consumer_out,
                N, tail_iters, head_iters, graph_base, exec_base);
    // Warmup graph
    CHECK(cudaGraphLaunch(exec_base, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaEventRecord(start, stream));
    CHECK(cudaGraphLaunch(exec_base, stream));
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaStreamSynchronize(stream));
    float graph_base_ms = 0;
    CHECK(cudaEventElapsedTime(&graph_base_ms, start, stop));

    // --- CUDA Graph + PDL ---
    cudaGraph_t graph_pdl;
    cudaGraphExec_t exec_pdl;
    build_graph(true, iters, gdim, block, d_producer_out, d_consumer_out,
                N, tail_iters, head_iters, graph_pdl, exec_pdl);
    // Warmup graph
    CHECK(cudaGraphLaunch(exec_pdl, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaEventRecord(start, stream));
    CHECK(cudaGraphLaunch(exec_pdl, stream));
    CHECK(cudaEventRecord(stop, stream));
    CHECK(cudaStreamSynchronize(stream));
    float graph_pdl_ms = 0;
    CHECK(cudaEventElapsedTime(&graph_pdl_ms, start, stop));

    // --- Print results ---
    printf("\n=== PDL & CUDA Graph Microbenchmark ===\n");
    printf("N=%d, tail_iters=%d, head_iters=%d, iterations=%d\n",
           N, tail_iters, head_iters, iters);
    printf("Baseline (no PDL, no Graph): %.3f ms total, %.3f ms avg\n",
           baseline_ms, baseline_ms / iters);
    printf("PDL only                   : %.3f ms total, %.3f ms avg (%.2fx)\n",
           pdl_ms, pdl_ms / iters, baseline_ms / pdl_ms);
    printf("Graph only                 : %.3f ms total, %.3f ms avg (%.2fx)\n",
           graph_base_ms, graph_base_ms / iters, baseline_ms / graph_base_ms);
    printf("Graph + PDL                : %.3f ms total, %.3f ms avg (%.2fx)\n",
           graph_pdl_ms, graph_pdl_ms / iters, baseline_ms / graph_pdl_ms);

    CHECK(cudaFree(d_producer_out));
    CHECK(cudaFree(d_consumer_out));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaGraphDestroy(graph_base));
    CHECK(cudaGraphExecDestroy(exec_base));
    CHECK(cudaGraphDestroy(graph_pdl));
    CHECK(cudaGraphExecDestroy(exec_pdl));

    return 0;
}
