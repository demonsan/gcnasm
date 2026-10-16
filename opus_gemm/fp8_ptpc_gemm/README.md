# FP8 PTPC preshuffle GEMM (gfx950)

A standalone OPUS-based HIP example with per-token activation scales,
per-channel weight scales, FP32 accumulation and BF16 output. It is independent
of aiter's public dispatch and uses caller-owned output buffers.

## Contract

```text
Y[m,n] = bf16_RNE((sum_k(A[m,k] * W[n,k]) * sa[m]) * sb[n])
```

- A: FP8 E4M3FN `[M,K]`, unshuffled, unit inner stride.
- W: logical FP8 E4M3FN `[N,K]`, contiguous in the canonical `(16,16)`
  preshuffled layout produced by aiter `shuffle_weight`.
- sa / sb: contiguous FP32 `[M]` or `[M,1]` and `[N]` or `[N,1]`.
- Y: caller-owned BF16 `[M,N]`, unit inner stride and row stride at least N.
- M >= 1, N >= 16 divisible by 16, K >= 32 divisible by 32; no bias or batching.
- Split-K writes unscaled FP32 partials to padded workspace, reduces them in
  fixed split order, scales once and converts to BF16 once. The MFMA scale
  operands are neutral 127,127.

Finite scales, correct shuffled weight bytes and non-aliasing input/output
buffers are preconditions. The Python adapter validates dimensions, dtypes,
strides, device and output aliasing. The host launchers reject over-range
32-bit buffer spans and insufficient workspace. These checks do not certify
all possible input distributions or scheduling conditions.

## Files and designs

| File | Role |
|---|---|
| [gemm_defs.h](gemm_defs.h) | Caller-owned-buffer C ABI declarations. |
| [gemm_a8w8_ptpc_splitk_kernel.cc](gemm_a8w8_ptpc_splitk_kernel.cc) | C1/C11 launchers and general small-M fallbacks. |
| [gemm_a8w8_ptpc_8wave_kernel.cc](gemm_a8w8_ptpc_8wave_kernel.cc) | C3/C18/C19 launchers and general eight-wave fallbacks. |
| [include/gfx950/](include/gfx950/) | Traits, device templates and FP32 workspace reducer. |
| [candidate.py](candidate.py) | Relocatable two-library Python interface. |
| [policy.py](policy.py) | Example selection, split partition and derived selector metadata. |
| [gemm_host.py](gemm_host.py) | Synthetic mathematical check and optional unprofiled smoke benchmark. |
| [tests/test_package.py](tests/test_package.py) | CPU-only policy, loader and source-structure tests. |

The small library uses a four-wave direct-register path for C1 and a C11
pipeline with two producer and two consumer waves, three-stage LDS staging
and half-K operand ping-pong. When C11's alignment/tile/interval requirements
are not met, it uses the C7 or natural-load fallback.

The eight-wave library uses 512-thread wave64 workgroups. C18 computes
256x256 output tiles with full-K expansion at K = 6144 or 8192. C19 uses a
256x192 tile, TM4xTN2 wave arrangement and EM2xEN3 per-quarter expansion,
with 96 FP32 accumulators per lane. Its physical B staging remains 128 rows
for 96 consumed rows. Other shapes use the C18 or 128x256 general C3 path.

The example policy selects C1 at M<=16, C11 at M<=32 and C19 otherwise,
with a few explicit small-M split choices in `policy.py`. It selects C19's
256x192 kernel only at packed/aligned `(4096,9216,6144)`, requested split 1.
`--cfg` and `--split` override the policy. This policy is an example, not a
claim of optimal performance at any shape.

`fn.derived` contains input-derived selector metadata. `fn.reported` contains
wrapper reports, retaining `None` for fields that are not reported (notably
C1 tile/path). Neither is a device-execution trace.

## Build

Linux, a ROCm toolchain with gfx950 support and an aiter checkout providing OPUS:

```sh
make OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
# When the utility header is not at the default sibling location:
make OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include \
     OPUS_GEMM_INCLUDE_DIR=/path/to/aiter/csrc/opus_gemm/include
```

Outputs:

```text
build/ptpc_c11_mature_gfx950.so
build/ptpc_c19_256192_gfx950.so
```

`HIPCC`, `BUILD`, `PYTHON` and the include directories can be overridden.
Only gfx950 is supported. The builds use C++17 and O3; the eight-wave build
also disables FMA contraction. No fast-math flag is added. `bash rebuild.sh`
forwards arguments to make. `make clean` removes only these two libraries.

The libraries are separate because they each export `ptpc_last_hip_error`;
do not link both translation units into one shared library.

## Check and run

CPU-only tests and policy listing (no Torch or GPU required):

```sh
python3 -m unittest discover -s tests -v
python3 gemm_host.py --plan-only --matrix17
```

With ROCm PyTorch and an available gfx950 device:

```sh
python3 gemm_host.py --matrix17 --verify-only
python3 gemm_host.py -m4096 -n9216 -k6144
python3 gemm_host.py -m32 -n9216 -k6144 --cfg c11 --split 3
```

The verifier uses synthetic inputs, an independent FP32 reference from the
unshuffled weights, all-element `rtol=atol=0.02`, output guard rows and three
bit-identical repeats. Optional benchmark output uses 101-call rotated HIP
graphs and events; it does not compare competing libraries or establish a
performance guarantee. Rebuilds and different OPUS/compiler versions require
fresh correctness and resource validation; no historical timings are included.

```python
from candidate import make_candidate
fn = make_candidate(build_dir="build")
effective_split = fn(A_fp8, W_preshuffled, sa, sb, Y_bf16)
```

Warm each exact shape/config/split outside graph capture to prepare library
and workspace pointers. Calls use the current PyTorch stream on A's device.
The returned effective split can be smaller than the requested split:
four-wave partitioning uses ceil(K/256) tiles, while the eight-wave partition
rounds iterations per split up to an even value when it has at least two tiles.

This is an experimental source example. The publication candidate has CPU
checks but no GPU validation of its cleaned source layout or rebuilt binaries.
It includes no benchmark results, binary artifacts or environment captures.
