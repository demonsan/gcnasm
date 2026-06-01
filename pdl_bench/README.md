# pdl_bench -- Programmatic Dependent Launch Microbenchmark

A CUDA microbenchmark for measuring the **Programmatic Dependent Launch (PDL)**
overlap on NVIDIA Hopper (sm90+) GPUs, with optional **CUDA Graph** capture.

## What is PDL?

Programmatic Dependent Launch is a Hopper+ feature that allows two back-to-back
kernels on the same CUDA stream to execute concurrently. The producer kernel
calls `cudaTriggerProgrammaticLaunchCompletion()` to unblock the consumer,
while the consumer calls `cudaGridDependencySynchronize()` before accessing
data produced by the first kernel.

The consumer kernel must be launched via `cudaLaunchKernelEx` with the
`cudaLaunchAttributeProgrammaticStreamSerialization` attribute set.

## CUDA Graph Support

The benchmark also tests **CUDA Graphs** to show how PDL interacts with
graph capture:

| Mode | Description |
|------|-------------|
| Baseline | Normal sequential kernel launches |
| PDL only | Sequential launches with PDL overlap |
| Graph only | CUDA Graph capture of 100 kernel pairs (no PDL) |
| Graph + PDL | CUDA Graph capture with PDL edges |

## Build

```bash
make
```

Requires:
- CUDA toolkit 12.x or newer (tested with 12.8 on H20)
- `nvcc` with `-arch=sm_90` (Hopper H100 / H20 / H200)

## Run

```bash
./pdl_bench.exe [N] [tail_iters] [head_iters] [iterations]
```

| Argument | Default | Description |
|----------|---------|-------------|
| N | 65536 | Number of elements (grid = N / 256) |
| tail_iters | 16384 | Producer post-trigger FMA iterations |
| head_iters | 16384 | Consumer pre-sync FMA iterations |
| iterations | 100 | Number of kernel-pair launches captured/measured |

Example:

```bash
./pdl_bench.exe 65536 16384 16384 100
```

## Expected Output (NVIDIA H20)

### Heavy compute (tail=16384, head=16384)

```
Device: NVIDIA H20 (sm90)

=== PDL & CUDA Graph Microbenchmark ===
N=65536, tail_iters=16384, head_iters=16384, iterations=100
Baseline (no PDL, no Graph): 203.219 ms total, 2.032 ms avg
PDL only                   : 163.746 ms total, 1.637 ms avg (1.24x)
Graph only                 : 200.389 ms total, 2.004 ms avg (1.01x)
Graph + PDL                : 164.076 ms total, 1.641 ms avg (1.24x)
```

For heavy kernels, **PDL gives ~1.24x speedup** by overlapping producer tail
with consumer head. CUDA Graph alone does not help much because CPU launch
overhead is negligible compared to kernel execution time.

### Light compute (tail=1024, head=1024)

```
N=65536, tail_iters=1024, head_iters=1024, iterations=100
Baseline (no PDL, no Graph): 13.767 ms total, 0.138 ms avg
PDL only                   : 10.070 ms total, 0.101 ms avg (1.37x)
Graph only                 : 13.481 ms total, 0.135 ms avg (1.02x)
Graph + PDL                :  9.988 ms total, 0.100 ms avg (1.38x)
```

Even for lighter kernels, PDL overlap is the dominant optimization.

## Files

| File | Description |
|------|-------------|
| `pdl_bench.cu` | Benchmark source (producer + consumer + host timing + graph capture) |
| `Makefile` | Build rules |
| `README.md` | This file |
