# Opus A2A GEMM CCO SDMA

Experimental 8-rank A2A+GEMM benchmark using CCO SDMA copy engines for the A-shard allgather.

Target shape per rank:

- local input shard: `[2048, 1024]` bf16
- replicated weight: `[8192, 8192]` bf16, stored as `[N, K]`
- output: `[2048, 8192]` bf16
- split-K partitions: 8, one input shard per rank

## Modes

The executable reports both:

- `fused`: one kernel overlaps SDMA all-peer A-shard puts with GEMM.
- `split`: standalone SDMA A2A allgather kernel followed by compute-only GEMM kernel.

The fused path currently uses a direct all-peer owner send:

1. block0 issues SDMA puts of this rank's local A shard to the other 7 ranks at kernel start.
2. GEMM computes K shards in rotate order from local `recv_a`.
3. after each shard, block0 quiets the corresponding destination and writes that peer's ready flag.
4. receivers poll their local ready flags before consuming non-local shards.

The split path uses the same all-peer SDMA put pattern in a standalone A2A kernel, then launches the same GEMM kernel with `sdma_fused=0`.

## Build

Build inside the MORI/ROCm container:

```bash
cd /shared/amdgpu/home/jiahao_zhou_qle/blyu/gcnasm/opus_gemm_dist/Opus_a2a_gemm_cco_sdma
make
```

The Makefile defines `BUILD_CCO_SDMA=1`; MORI itself must also have been built with SDMA support.

## Run

Best current setting on the MI355X box is one SDMA channel:

```bash
export MORI_ENABLE_SDMA=1
export MORI_SDMA_NUM_CHANNELS=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MORI_SOCKET_IFNAME=enp193s0f0np0
export LD_LIBRARY_PATH=/shared/amdgpu/home/jiahao_zhou_qle/blyu/mori/python/mori:${LD_LIBRARY_PATH:-}

mpirun --allow-run-as-root -n 8 \
  -x MORI_ENABLE_SDMA -x MORI_SDMA_NUM_CHANNELS \
  -x HIP_VISIBLE_DEVICES -x MORI_SOCKET_IFNAME -x LD_LIBRARY_PATH \
  ./build/a2a_gemm_lsa.exe --warmup 1 --iters 3
```

Recent result:

```text
a2a_gemm_sdma fused=0.2883 ms 953.55 TFLOP/s SUCCESS
a2a_gemm_sdma split_a2a=0.1095 ms split_gemm=0.2108 ms split_total=0.3202 ms 858.34 TFLOP/s SUCCESS
a2a_gemm_sdma fusion_win=9.98%
```

The harness also prints per-rank fused times and block0 start/end cycle deltas for quick imbalance checks.
