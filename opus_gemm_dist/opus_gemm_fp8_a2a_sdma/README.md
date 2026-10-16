# opus_gemm_fp8_a2a_sdma

Standalone experiment (no MPI needed by default) for an FP8 PTPC GEMM fused with an SDMA all-to-all for the SP4 large-T QKV projection
(T = 8192 / 16384 / 32768 tokens, 4 ranks, per-rank M = T/4 = 2048 / 4096 / 8192).

Each rank computes `C = A x W^T` for its own `M` tokens over the full QKV weight (`K = 6144`; `N = 9216` dense, `N = 9856` with the
indexer columns) and **sends every destination rank only the columns that destination needs**, directly from the GEMM store path into a
per-destination staging slab, then PUTs those slabs to the peers' receive buffer with SDMA while the GEMM is still running.

**The golden receiver is not part of this directory.** The receiver that consumes the received packed tensor (Triton `norm_rope_cache` for
`qkv_dense` / `qkv_indexer_reuse`, AITER `fused_qknorm_idxrqknorm` for `qkv_indexer`) is Python and lives in the model/test harness. This
directory builds only the GEMM + SDMA chain and validates the **received packed slabs (R)** against a host reference. The full-chain numbers in
the results section were measured in the private harness with the unchanged golden receiver.

## What it computes

- `A`: FP8 E4M3FN `[M, 6144]`, per-token FP32 scale `Ascale[M]`.
- `W`: FP8 E4M3FN `[N, 6144]` in the `aiter shuffle_weight(layout=(16,16))` order, per-channel FP32 scale `Wscale[N]` (raw n order).
- FP32 accumulation, epilogue `bf16_rne(fp32(fp32(acc * Wscale[n]) * Ascale[m]))`: two separate multiplies, one BF16 rounding.
  MFMA `16x16x128` FP8 via `mma(a, b, c, 127, 127, 0, 0)` (scaled `f8f6f4` form; the plain 3-argument `mma` has no FP8 16x16x128 dispatch).
- Output per destination `d`: a slab `S_d[M][F]` whose columns follow the golden `select_qkv_from_peers` order:
  `Q[d*2048:+2048] | K[8192+d*128:+128] | V[8704+d*128:+128]` (`F = 2304`), plus for the indexer `| IQ[9216+d*128:+128] | IK[9728:9856]` (`F = 2560`).
  `IK` is the same 128 raw columns for every destination: it is computed once and rounded once, then written to all four destinations.
- Rank `r` receives into `R[4*M][F]` (source-rank-major rows: row `s*M + m` is source `s`'s token `m`), i.e. the packed tensor the receiver reads as `[T, F]`.

## Plans (`--case`) and options

| `--case` | build (`PLAN=`) | N | F | what is transferred |
|---|---|---|---|---|
| `dense` | `dense` | 9216 | 2304 | Q, K, V |
| `indexer` | `indexer` | 9856 | 2560 | Q, K, V, IQ, IK (IK replicated to all 4 destinations) |
| `reuse` | `dense` | 9216 | 2304 | **q/k/v only** (compact `[4M, 2304]` receive layout, see below) |

- `--puts-per-peer 1|2|4` (n1/n2/n4): number of SDMA PUTs (put groups) per peer per epoch. The best n differs per T (see results). `0` = default by M (1 / 2 / 4).
- `--half-first` (n2h/n4h): the first put group of every peer is computed as half-N (256x128) items so the first PUT starts earlier (needs `--puts-per-peer 2|4`; M = 2048 is where it matters).
- `--quarter-first` (**n3q**, dense build and `--M 2048` only): the first put group of every peer is computed as **128x128 quarter items** (one row half of one column half of a tile) and the chain uses **three put groups of 2 / 1 / 5 chunks** per peer
  (implies `--puts-per-peer 3`; host plan code `half_first = 2`). Group 0 (chunks 0-1, 216 quarter items) completes after about a quarter-tile time so the first PUT starts earlier; group 1 (chunk 2) is ready while group 0 is still on the wire, so the link stays busy; group 2 (chunks 3-7)
  follows. Arrivals per remote (destination, group) are 72 / 9 / 45; 450 items + 1 poster block. Exactly the dense numerics (same MFMA chain and epilogue per element). Entry `lg_r9_gemm_n3q_v1`; it traps in indexer builds.
- `--M`, `--warmup`, `--iters`, `--check-rows`, `--exact`, `--poll-budget-us`: see `--help`.

## Design notes

- **Resident poster block.** The GEMM grid is `1 + items` workgroups. Block 0 is a *poster*: three waves, one per remote peer, each issuing only the
  PUTs of its own peer (no lock, no CAS: exactly-once is structural). Each poster wave waits (bounded, relaxed spin + one acquire fence) until every
  item workgroup of its next put group has published, then issues one no-signal SDMA PUT of that group's rows (`n` groups per peer, contiguous byte
  ranges of the peer's slab), and after the last PUT runs `quietQueue` and releases an arrival flag to the peer. Item workgroups never wait on a remote
  GPU or on the poster. A separate 1-workgroup `quiet_ready` kernel (F0, production) waits for the 3 peers' arrivals, resets the device counters and
  advances the epoch; the receiver runs after it. (`FOLD=1` folds this into the poster block; it measured slower at T8192 and is not the default.)
- **L2b release.** Every item workgroup publishes with one system release: per-wave `s_waitcnt vmcnt(0)`, workgroup barrier, thread 0
  `buffer_wbl2` (system release fence) + `s_waitcnt vmcnt(0)`, then a RELAXED system `atomic_add` on the `(destination, put-group)` counter. One write-back
  per workgroup covers every wave's staging stores and every destination.
- **Group order and half-first.** Remote items are ordered by put group (g-major), so the first wave completes group 0 of all three peers together; self tiles
  run last and are written straight into `R[rank]`. With `--half-first`, group 0 uses 256x128 half items (each about 0.77 of a tile time), making group 0 ready
  after about half a tile time.
- **Quarter-first (n3q).** `quarter_body` is the half-N pipeline with ONE A half and one accumulator (128 rows x 128 columns; half the MFMAs and A traffic of a half item, same B layout), in its own inlined copy so the other item kinds' registers are unaffected. A true 256x64 quarter-N was not used because the B layouts
  are fixed at 128 columns. A pipe-continuity model (`tests/` of the private candidate) shows that a two-group split [2 quarter chunks][6 full chunks] would leave the link idle before the second put, hence the 2 / 1 / 5 split. Static proof: `r9_order_ok(8)` (bijection, every 128x128 quarter once, per-group arrivals 72 / 9 / 45).
- **No dispatch-packet read on the entry path (`NO_GRIDREAD=1`, default).** The original config guard read the AQL dispatch packet (`grid_size_x`, an `s_load` at packet offset 0xc) in **every** workgroup, including the poster, before any work, to check that the launched grid equals the plan. That scalar load sits on the critical path
  of every WG entry (about 3.5-4 us at small M). The guard now uses the compile-time plan grid, and ONE item block (item 0, thread 0) compares the real packet value **after its work** and poisons CONFIG on a mismatch. `NO_GRIDREAD=0` restores the entry read (A/B), `NO_GRIDREAD=2` removes the packet read entirely (the host launcher computes the same grid; the device-side backstop is lost).
  A mismatched grid is therefore detected at the end of item 0 instead of before any work: the host launcher always computes the plan grid, so this only matters for a buggy caller.
- **IK owner and IQ.** The IK column segment is computed by one item per chunk (full tile through the same pipeline, one BF16 rounding) and stored to the
  local `R` and to the three remote staging slabs; IQ is one single-destination item per destination.
- **Reuse compact layout (human-approved: reuse transfers only q/k/v).** The reuse receiver reads only `packed.split([2048,128,128,...])[:3]`. CCO has only
  linear SDMA copies (no strided/2D put), so the transfer saving needs a compact receive buffer: the dense build (`F = 2304`, no IQ/IK compute or transfer) receives
  `[4M, 2304]`, and the receiver is given `torch.as_strided(R, (T, 2560), (2304, 1))` (row stride 2304; the 256 IQ|IK columns of a row alias the next row's first columns and are never read).
  The R window carries a 4 KiB pad so the last row of that view stays in bounds. This standalone program validates the compact buffer; the strided view is built by the Python harness.
- **Windows (equal size, multiple of 4 MiB).** `S`, `R` and `C` are registered with the same size `roundup(max(P + pad, control area), 4 MiB)`. The VMM allocator fails with
  `hipMemSetAccess ... invalid argument` for a window whose size is not a multiple of 4 MiB (the reported 4096-byte granularity is queried without `gpuDirectRDMACapable`,
  which the allocation sets). `LG_WINDOW_GRAN_MIB=8` raises the granule for experiments.
- **Numerics are unchanged by the plan and by n.** Only the order of tiles and the grouping of puts differ; every output element is the same MFMA chain and the same epilogue.

## Memory-model rules (read before changing the device code)

- **Explicit release form everywhere.** A bare `ACQ_REL` RMW or release store can be lowered as `buffer_wbl2` -> `lgkmcnt` wait -> atomic with *no* `s_waitcnt vmcnt(0)` between the
  write-back and the atomic. Every release site here is `fence(release)` -> `asm volatile("s_waitcnt vmcnt(0)" ::: "memory")` -> RELAXED system atomic/store (acquire sites: RELAXED spin + one
  `fence(acquire)`). Verify in the ISA: `buffer_wbl2 sc0 sc1` -> `s_waitcnt vmcnt(0)` -> atomic.
- **Poster waits are relaxed spins with one acquire fence** after success (an `ACQUIRE` load per spin would issue a `buffer_inv` per spin on the poster's XCD for the whole GEMM).
- **Recovery rule (A2-H).** After *any* epoch with a poison on *any* rank (a put-phase failure leaves stale group counters/state), the next epoch must not be launched until:
  `drain_all` on every rank -> barrier -> `init_control` on every rank (zeroes the whole control window) -> barrier. `main.cc` implements this (`recover()`, using the CCO host barrier or MPI); a poisoned validation epoch aborts after recovery.
- **Bounded waits only.** Every remote wait has a budget in microseconds (`--poll-budget-us`, default 50 ms, must exceed worst-case GEMM time + rank skew); a timeout poisons the epoch instead of hanging.
- The kernels require **rank == lsaRank == HIP device id**, 4 ranks on one node (GPUs 0..3 or `HIP_VISIBLE_DEVICES`), and `BUILD_CCO_SDMA=1` (the MORI header default `0` turns PUTs into silent no-ops).

## Build

Inside the ROCm/MORI container (same variables as the sibling directories):

```bash
cd opus_gemm_dist/opus_gemm_fp8_a2a_sdma
make PLAN=dense   OPUS_INCLUDE_DIR=<path to aiter/csrc/include> MORI_INCLUDE_DIR=<mori>/include MORI_LIB_DIR=<mori>/build/src/cco
make PLAN=indexer OPUS_INCLUDE_DIR=<path to aiter/csrc/include> MORI_INCLUDE_DIR=<mori>/include MORI_LIB_DIR=<mori>/build/src/cco
```

By default the host is built **without MPI** (`USE_MPI=0`): 4 processes are started by `run_4rank.sh`. `make ... USE_MPI=1` builds the `mpirun` variant instead (needs `mpi.h`, `libmpi`, `mpirun`; output in `build_<plan>_mpi/`).

`OPUS_INCLUDE_DIR` must point at an aiter checkout's `csrc/include` containing `opus/opus.hpp` and `opus/hip_minimal.hpp` (the private Opus headers; the kernel TU is built with `-D__HIPCC_RTC__`,
its host pass defines empty `__global__` stubs). Builds go to `build_dense/` and `build_indexer/`; `-Rpass-analysis=kernel-resource-usage` prints VGPR/spill/LDS (expect no spill, no scratch, LDS 135168 B).
Production flags are fixed by the Makefile (`FOLD=0`, L2b release, full remote tiles, no debug seams); `FOLD=1`, `DEBUG_POSTCOUNT=1`, `STAMPS=1` are available for experiments and are never used for timing.

## Run

### Default: no MPI (`run_4rank.sh`)

```bash
# 4 ranks on GPUs 0..3 of one node (DEVICES=0,1,2,3 is the default; pick other GPUs with DEVICES=4,5,6,7)
./run_4rank.sh build_dense/fp8_a2a_sdma_dense.exe   --case dense   --M 2048 --puts-per-peer 2 --half-first   # T=8192 (half-first needs n = 2 or 4)
./run_4rank.sh build_dense/fp8_a2a_sdma_dense.exe   --case dense   --M 2048 --quarter-first                 # T=8192, n3q (dense/reuse build, M = 2048 only)
./run_4rank.sh build_dense/fp8_a2a_sdma_dense.exe   --case dense   --M 4096 --puts-per-peer 2 --warmup 5 --iters 50
./run_4rank.sh build_dense/fp8_a2a_sdma_dense.exe   --case reuse   --M 8192 --puts-per-peer 4
./run_4rank.sh build_indexer/fp8_a2a_sdma_indexer.exe --case indexer --M 2048 --puts-per-peer 2 --half-first
# byte-exact validation with exact-arithmetic inputs
./run_4rank.sh build_dense/fp8_a2a_sdma_dense.exe --case dense --M 2048 --exact
```

`run_4rank.sh` starts the 4 processes (`--rank r --world 4 --uid-file <workdir>/cco.uid --device r`), exports `HIP_VISIBLE_DEVICES=$DEVICES` to **every** rank (each process must see all 4 GPUs because CCO maps peer
memory; rank `r` uses visible device `r`), waits for them, and prints the 4 per-rank logs. Rank 0 creates the CCO unique id and publishes it atomically (write + rename) to the uid file; the other ranks poll
for it with a bounded timeout (`TIMEOUT_S`, default 120 s). After the communicator exists, barriers use CCO's host barrier (`ccoBarrierAll`); the few small reductions (poison flag, validation counts, timings) are exchanged through
atomic per-rank files in `<uid-file>.d/`. Every wait is bounded; if a rank fails, the others report which peer did not answer and exit. `WORKDIR` defaults to a fresh `mktemp -d` (a stale uid file would be misread, so reuse
a directory only after deleting it). Rank 0 prints the validation and timing summary; the other ranks print their own epoch time. Manual launch (e.g. under a scheduler): start 4 processes with
`HIP_VISIBLE_DEVICES=0,1,2,3 exe --rank r --world 4 --uid-file /shared/path/cco.uid --device r ...`.

### With MPI (`make USE_MPI=1`)

```bash
make PLAN=dense USE_MPI=1 OPUS_INCLUDE_DIR=... MORI_INCLUDE_DIR=... MORI_LIB_DIR=...
mpirun --allow-run-as-root -n 4 build_dense_mpi/fp8_a2a_sdma_dense.exe --case dense --M 2048 --puts-per-peer 2 --half-first
```

Identical options and output; `--rank/--world/--uid-file` are ignored (MPI provides them) and the barriers/reductions are MPI collectives.
`--M` is the per-rank row count (`M = T/4`). The program (either launch mode) creates the CCO communicator and the three windows, builds deterministic per-rank inputs, runs one validated epoch, then
times `--iters` back-to-back epochs with hipEvents and prints microseconds per epoch and per-rank / aggregate PUT GB/s.

**Validation.** For every source rank and `--check-rows` sampled rows, all `F` packed columns of the slab this rank received are compared with a host reference
(FP64 accumulation over the dequantised inputs, then `(acc*Ws)*As` in FP32 and one BF16 round-to-nearest-even). With `--exact` the inputs are `{+-0.5, +-1, +-2}` with unit scales,
every dot product is exact, and the result must be **byte-identical**; with random inputs the tolerance is `rtol = atol = 0.02` (the golden `compare_outputs` rule). R is pre-filled with `0xFF`
(a BF16 NaN) so an unwritten element can never pass.

## Debug flags (default off, never for timing)

| Make variable | effect |
|---|---|
| `FOLD=1` | poster block folds W2/reset/epoch; no `quiet_ready` launch (measured slower at T8192) |
| `NO_GRIDREAD=0` / `2` | A/B: restore the entry-path `grid_size_x` read / remove the packet read entirely (default `1`: read only once, after item 0's work) |
| `DEBUG_POSTCOUNT=1` | per-`(peer, group)` post counters (exactly-once check) |
| `STAMPS=1` | same-GPU `s_memrealtime` stamps (never compare stamps across ranks) |

## Results

**These are full-chain numbers from the private SP4 harness, not standalone numbers of this program.** Metric: collector-off GPU span of the whole chain (A2A GEMM + SDMA + unchanged golden
receiver, graph-captured) against the original Torch/RCCL baseline chain (GEMM -> RCCL all-to-all -> receiver); `B/C` = baseline time / candidate time. `n` is the best puts-per-peer for that cell.
Source: the private project ledger (confirmations by an independent evidence role, two windows where noted; the golden receiver outputs and caches passed the original all-element `rtol = atol = 0.02` checks byte-exactly on the actual graph outputs).

| Case | T = 8192 (M = 2048) | T = 16384 (M = 4096) | T = 32768 (M = 8192) |
|---|---|---|---|
| dense | **1.2251 / 1.2182** (r11 n3q: quarter-first + no grid read; r8g F0 n2h was 1.148) | 1.311 / 1.299 (n2) | 1.378 / 1.384 (n4) |
| indexer | 1.2185 / 1.2219 (n2h) | 1.4240 (n4) | 1.4745 (n4) |
| reuse (q/k/v only) | 1.2191 / 1.2189 (n2h) | 1.3363 (n4); 1.3884 in a later window (n2) | 1.3372 (n4); 1.4816 in a later window (n4) |

Two numbers in a cell are two independent windows (ledger r115 dense, r123 indexer T8192, r128/r129 reuse T8192); a single number is the confirming window of r118; "later window" values are the r128 first-window F0 results quoted in the ledger. Dense T = 8192 (r11 n3q, two windows) now also clears 1.20: all 9 large-T cells (3 cases x 3 T) are at or above 1.20 (the indexer/reuse T = 8192 cells are as confirmed with r7 n2h and r8g F0 n2h). Small M (T = 64 / 128) is not part of this directory.
### Standalone results of this program

GEMM + SDMA chain only (no receiver), MI355X gfx950, 4 ranks on GPUs 4–7, default non-MPI launch (`run_4rank.sh`), default build (`NO_GRIDREAD=1`), eager launches timed with hipEvents, `--warmup 10 --iters 101`, one repeat; reported value is µs per epoch, max over ranks. Validation samples 16 rows per source rank × all packed columns × 4 destinations: `--exact` inputs are byte-equal to the host reference, random inputs pass `rtol = atol = 0.02`, and the poison word is 0 on all ranks. The device code objects are byte-identical to the r11 candidate.

| case | M (rows/rank) | plan | exact-input µs | random-input µs | previous build (r8g, grid read) exact µs |
|---|---:|---|---:|---:|---:|
| dense | 2048 | n1 | 261.85 | 262.38 | 273.55 |
| dense | 2048 | n2h | 243.93 | 245.87 | 254.59 |
| dense | 2048 | **n3q (`--quarter-first`)** | **231.99** | **233.05** | — |
| dense | 4096 | n2 | 423.72 | 422.27 | 441.54 |
| dense | 8192 | n4 | 748.53 | 746.96 | 763.26 |
| indexer | 2048 | n2h | 264.27 | 266.61 | 274.50 |
| indexer | 4096 | n4 | 441.87 | 452.14 | 458.01 |
| indexer | 8192 | n4 | 875.20 | 874.50 | 884.48 |
| reuse (compact q/k/v) | 2048 | n2h | 242.51 | 244.63 | 254.27 |

These numbers are not comparable to the full-chain ratios above, which include the receiver and the graph-replay metric.

## Provenance

Ported without algorithm changes from the private playground candidate LG-QSDMA-r11 (manifest `9874fe1ecb3b6a997523c96210a960434179c8eab84fffc00537170d49836643`) = r10 (`LG_NO_GRIDREAD`) + r9 (quarter-first n3q) on the r8g base (`a6e3988b19be3b0d59c38eaf2a984fa9a7e22ec821cf1e2b886868fc78859892`), production configuration `FOLD=0`:
`gemm_fp8_a2a_sdma_kernel.cc` is `device/gemm_quad_fp8_comm.cu`; `lg_comm_device.hpp`, `lg_r6x.hpp`, `epilogue_ptpc_pairstore.hpp`, `layout/*`, `lg_comm_abi_v2.h` are copied unchanged from r11. The Python bridge and the
ctypes driver are replaced by `main.cc` (default non-MPI process group; MPI behind `USE_MPI=1`).