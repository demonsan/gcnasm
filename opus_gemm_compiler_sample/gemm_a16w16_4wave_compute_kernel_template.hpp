// gfx1250 BF16 GEMM -- symmetric 4-wave "compute" pipeline (device-only; see the
// stub TU). C[M, N] = A[M, K] @ B[N, K]^T, bf16 stored directly.
//
// Every wave both TDM-loads and runs WMMA; there is no producer/consumer split.
// The 4 waves split M (32 rows each) and each accumulates its own 32(M) x B_N
// strip over the full K loop, with N carved into kNumNSub sub-tiles so all
// accumulators (~256 fp32 VGPR/lane) stay live.
//
// TDM_LOAD_PART = 2: all 4 waves issue one half-row TDM per K step -- w0/w1 take
// the upper/lower half of A, w2/w3 the upper/lower half of B. Halving the tile
// halves the per-instruction DMA latency.
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "gemm_defs.h"

#include <cstdint>

#if !defined(__gfx1250__)
#error "the 4wave_compute pipeline is gfx1250-only (WMMA 16x16x32 + TDM); build with --offload-arch=gfx1250"
#endif

namespace gemm_4wave_compute {

using opus::operator""_I;

// Kernel-internal derived traits: WMMA decomposition, N-subtiles, TDM windows.
template<typename UT>
struct kernel_traits {
    static constexpr int BLOCK_SIZE = UT::BLOCK_SIZE;

    using DataA   = typename UT::D_A;
    using DataB   = typename UT::D_B;
    using DataC   = typename UT::D_C;
    using DataAcc = typename UT::D_ACC;

    static constexpr int kVecA = UT::VEC_A;
    static constexpr int kVecB = UT::VEC_B;
    static constexpr int kCVec = UT::VEC_C;

    static constexpr int kBlockM = UT::B_M;
    static constexpr int kBlockN = UT::B_N;
    static constexpr int kBlockK = UT::B_K;

    // WMMA 16x16x32 (gfx1250 bf16).
    static constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 32;

    static constexpr int kWarp = 32;                       // gfx1250 wave size
    // Runtime-pass warp (32 device / 64 host); the WMMA register decomposition
    // is derived from this so the device and host passes agree on vtype_c.
    static constexpr int kWarpRt = opus::get_warp_size();
    static constexpr int kNumWaves = BLOCK_SIZE / kWarp;
    static_assert(kNumWaves == 4, "4wave_compute is locked to 4 waves");

    // The 4 waves split M; each owns the full N.
    static constexpr int kTileM = 4;
    static constexpr int kTileN = 1;
    static constexpr int kTileK = 1;

    static constexpr int kExpM = kBlockM / (kWmmaM * kTileM);
    static constexpr int kExpN = kBlockN / (kWmmaN * kTileN);
    static_assert(kExpM * (kWmmaM * kTileM) == kBlockM, "B_M must be a multiple of kWmmaM*kTileM");
    static_assert(kExpN * (kWmmaN * kTileN) == kBlockN, "B_N must be a multiple of kWmmaN*kTileN");

    // N-subtiles (msb): kNumNSub accumulator groups. Splitting N keeps each
    // step's ds count inside the 6-bit DScnt budget, which is what lets ds_read
    // overlap WMMA without sched_barrier walls.
    static constexpr int kNumNSub = 4;
    static_assert(kExpN % kNumNSub == 0, "kExpN must split into kNumNSub msb groups");
    static constexpr int kExpNPerSub = kExpN / kNumNSub;

    static constexpr int kExpKHalf   = 2;                        // K-tiles per ds half
    static constexpr int kKHalfElems = kWmmaK * kExpKHalf;
    static_assert(kBlockK % kKHalfElems == 0, "B_K must be a multiple of kWmmaK*kExpKHalf");
    static constexpr int kHalvesPerSlot = kBlockK / kKHalfElems;

    static constexpr int kTdmK      = kBlockK;                   // one TDM per B_K slot
    static constexpr int kSmemPitch = UT::SMEM_PITCH;

    // Half-row TDM: w0+w1 split the A rows, w2+w3 split the B rows.
    static constexpr int kTdmLoadPart = 2;
    static constexpr int kARowsTdm    = kBlockM / kTdmLoadPart;
    static constexpr int kBRowsTdm    = kBlockN / kTdmLoadPart;
    static constexpr int kSlotBytesTdmA = kARowsTdm * kSmemPitch * (int)sizeof(DataA);
    static constexpr int kSlotBytesTdmB = kBRowsTdm * kSmemPitch * (int)sizeof(DataB);

    static constexpr int kSlotElemsA = kBlockM * kSmemPitch;
    static constexpr int kSlotElemsB = kBlockN * kSmemPitch;
    static constexpr int kSlotBytesA = UT::SLOT_BYTES_A;
    static constexpr int kSlotBytesB = UT::SLOT_BYTES_B;
    static constexpr int kSegBytesA  = UT::SEG_BYTES_A;
    static constexpr int kLdsTotalBytes = UT::LDS_BYTES;

    static constexpr int kNumSlots   = UT::NUM_SLOTS;
    static constexpr int kClusterWgM = UT::CLUSTER_WG_M;
    static constexpr int kClusterWgN = UT::CLUSTER_WG_N;

    // WMMA register decomposition (from kWarpRt so dev/host agree).
    static constexpr int kReptA = kWmmaM * kWmmaK / kWarpRt / kVecA;
    static constexpr int kReptB = kWmmaN * kWmmaK / kWarpRt / kVecB;
    static constexpr int kGrpKA = kWarpRt / kWmmaM;
    static constexpr int kGrpKB = kWarpRt / kWmmaN;

    // Automatic tier: a row is kBlockK elements and gets one b128 read vector of pad, which is what
    // breaks the power-of-two bank conflict. The dtype-sensitive arithmetic lives in the tag.
    using Padding = opus::tdm_traits::padding_auto<DataA, kBlockK>;
    // The auto tier is only shorthand for the hand-written one; pin that so a change to either is
    // caught here rather than as a silently different D#.
    static_assert(opus::is_same_v<Padding, opus::tdm_traits::padding<DataA, kBlockK, UT::PAD_ELEMS>>,
                  "tdm_traits::padding_auto must agree with the hand-written tier for this dtype and row length");
    // The single source of LDS row pitch: gemm_defs sizes the segments from SMEM_PITCH while the D#
    // is programmed from the policy, so they have to be the same number.
    static_assert(Padding::pitch_elements == kSmemPitch, "LDS row pitch must match the padding policy");

    // One window type covers both operands; only tile_dim1 differs and that is
    // patched into the D# at issue time (see the kernel body).
    using Window  = opus::tdm<DataA, opus::seq<kTdmK, kARowsTdm>, Padding>;
};

// kExpN-shadowed view of the traits so make_layout_rb builds the B read layout
// for ONE msb (kExpNPerSub N-tiles) instead of the full N.
template<typename T>
struct sub_traits : T { static constexpr int kExpN = T::kExpNPerSub; };

// A operand (M x K) smem read layout; wave_m selects this wave's M sub-tile.
template<typename T>
inline __device__ auto make_layout_ra(int lane_id, int wave_m) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpM>{}, opus::number<T::kTileM>{}, opus::number<T::kWmmaM>{},
        opus::number<T::kExpKHalf>{}, opus::number<T::kTileK>{},
        opus::number<T::kReptA>{}, opus::number<T::kGrpKA>{}, opus::number<T::kVecA>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<0>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSmemPitch, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_m, lane_id % T::kWmmaM, 0, lane_id / T::kWmmaM}));
}

// B operand (N x K) smem read layout. The N-dim order must be (kExpN outer,
// kTileN inner, kWmmaN) to match the C-store layout.
template<typename T>
inline __device__ auto make_layout_rb(int lane_id, int wave_n) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpN>{}, opus::number<T::kTileN>{}, opus::number<T::kWmmaN>{},
        opus::number<T::kExpKHalf>{}, opus::number<T::kTileK>{},
        opus::number<T::kReptB>{}, opus::number<T::kGrpKB>{}, opus::number<T::kVecB>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<0>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSmemPitch, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_n, lane_id % T::kWmmaN, 0, lane_id / T::kWmmaN}));
}

}  // namespace gemm_4wave_compute

template<typename UserTraits>
__global__ __launch_bounds__(UserTraits::BLOCK_SIZE, 1)
__attribute__((amdgpu_num_vgpr(1024)))
__cluster_dims__(UserTraits::CLUSTER_WG_M, UserTraits::CLUSTER_WG_N, 1)
void gemm_a16w16_4wave_compute_kernel(opus_gemm_kargs kargs) {
    using namespace opus;
    using namespace gemm_4wave_compute;
    using opus::operator""_I;
    using T = kernel_traits<opus::remove_cvref_t<UserTraits>>;
    using DataA   = typename T::DataA;
    using DataB   = typename T::DataB;
    using DataC   = typename T::DataC;
    using DataAcc = typename T::DataAcc;

    constexpr int P = T::kNumSlots;

    const int wave_id = __builtin_amdgcn_readfirstlane((int)opus::waveid_in_workgroup());
    const int lane_id = (int)opus::lane_id();

    // Cluster (CWGM x CWGN): cluster_id = super-tile, cluster_workgroup_id = local tile.
    const int cluster_x = (int)__builtin_amdgcn_cluster_id_x();
    const int cluster_y = (int)__builtin_amdgcn_cluster_id_y();
    const int local_x   = (int)__builtin_amdgcn_cluster_workgroup_id_x();
    const int local_y   = (int)__builtin_amdgcn_cluster_workgroup_id_y();
    const int batch_id  = (int)__builtin_amdgcn_workgroup_id_z();
    const int tile_row  = (cluster_x * T::kClusterWgM + local_x) * T::kBlockM;
    const int tile_col  = (cluster_y * T::kClusterWgN + local_y) * T::kBlockN;

    // Whether this workgroup has a tile at all. `tile_row >= m` is exactly
    // `tile_index >= ceil_div(m, B_M)`, because tile_row is that index times
    // B_M, so the test costs neither of the two runtime divisions the tile counts
    // would. It is workgroup-uniform, and it is acted on down in the prologue
    // rather than here: a workgroup without a tile still owes the cluster barrier
    // its one arrival.
    const bool tile_oob = tile_row >= kargs.m || tile_col >= kargs.n;

    // Multicast workgroup_mask: A is shared by the CWGN peers that fix M, B by
    // the CWGM peers that fix N. The producers fold a fan-out of <=1 to "no
    // multicast" -- required for correctness under a non-cluster launch, not
    // just a saving; both fan-outs are compile-time here so the fold is free.
    const auto mask_a = opus::tdm_traits::peers_along_y<T::kClusterWgM, T::kClusterWgN>();
    const auto mask_b = opus::tdm_traits::peers_along_x<T::kClusterWgM, T::kClusterWgN>();

    const int stride_a = kargs.stride_a;
    const int stride_b = kargs.stride_b;
    const int stride_c = kargs.stride_c;

    const DataA* base_a = reinterpret_cast<const DataA*>(kargs.ptr_a)
                        + (size_t)batch_id * (size_t)kargs.stride_a_batch;
    const DataB* base_b = reinterpret_cast<const DataB*>(kargs.ptr_b)
                        + (size_t)batch_id * (size_t)kargs.stride_b_batch;
    DataC*       base_c = reinterpret_cast<DataC*>(kargs.ptr_c)
                        + (size_t)batch_id * (size_t)kargs.stride_c_batch;

    const int k_steps = ceil_div(kargs.k, T::kBlockK);
    if (k_steps <= 0) return;

    __shared__ char lds_buf[T::kLdsTotalBytes];
    DataA* smem_a = reinterpret_cast<DataA*>(lds_buf);
    DataB* smem_b = reinterpret_cast<DataB*>(lds_buf + T::kSegBytesA);
    constexpr int slot_a = T::kSlotElemsA;
    constexpr int slot_b = T::kSlotElemsB;

    // ONE window type for both operands. The A and B windows are the same tdm<>
    // instantiation apart from DataType and TileDim1, and TileDim1 is patchable
    // at issue time (D# sg1[4] low16). Keeping them as two types forces an
    // `if (is_operand_a) ... else ...` around every issue, and that branch ends the
    // scheduling region: the descriptor SALU then land in a basic block with no
    // WMMA in it, so no amount of sched_group_barrier can pair them with one.
    using WindowU = typename T::Window;
    // DataType is the one parameter that is NOT patchable at issue time: it
    // feeds the D#'s element-size field and the sizeof() in make()/move() that
    // scales global_offset_bytes.
    static_assert(opus::is_same_v<DataA, DataB>,
                  "folding the A/B TDM windows patches only tile_dim1; "
                  "mixed A/B element types need two windows");

    // ── Per-wave TDM stream: all 4 waves load every K step, half the rows each.
    //   w0 (half 0): A rows [0, kARowsTdm)        w1 (half 1): A rows [kARowsTdm, B_M)
    //   w2 (half 0): B rows [0, kBRowsTdm)        w3 (half 1): B rows [kBRowsTdm, B_N)
    const bool is_operand_a = (wave_id <= 1);
    const int  row_half_id  = wave_id & 1;       // which half of the operand's rows this wave takes
    constexpr auto KStep = opus::number<T::kBlockK>{};

    // One window per operand, each stated in its own geometry, and a single select over the finished
    // windows rather than seven selects over their inputs. Measured cost of the split: five extra
    // prologue SALU (the second window's saturating_sub and offset multiply, which cannot sink past
    // the select because both feed it) and +52 code bytes. The WMMA mainloop is byte-identical and
    // register pressure is unchanged, so the whole difference executes once per launch.
    //
    // The layout is opus (C) order -- rows first, contiguous extent last -- and make_tdm() reverses it
    // into D# order, where dim0 is the contiguous one. Its innermost stride is never read (dim0 is the
    // fastest axis by definition) but has to be there to match the rank.
    // The extents are the WHOLE tensor's, not "rows left from here": the window clamps the tile against
    // them, which is what makes the ragged last tile need no handling here.
    auto window_a = opus::make_tdm<WindowU>(
        (uint32_t)(reinterpret_cast<uintptr_t>(smem_a) + row_half_id * T::kSlotBytesTdmA),
        (const void*)base_a,
        opus::make_layout<-1>(
            opus::make_tuple((uint32_t)kargs.m, (uint32_t)kargs.k),
            opus::make_tuple((uint64_t)stride_a, 1_I),
            opus::make_tuple((uint32_t)(tile_row + row_half_id * T::kARowsTdm), 0u)));
    window_a.set_workgroup_mask(mask_a);
    window_a.set_tile_dim1((uint32_t)T::kARowsTdm);

    auto window_b = opus::make_tdm<WindowU>(
        (uint32_t)(reinterpret_cast<uintptr_t>(smem_b) + row_half_id * T::kSlotBytesTdmB),
        (const void*)base_b,
        opus::make_layout<-1>(
            opus::make_tuple((uint32_t)kargs.n, (uint32_t)kargs.k),
            opus::make_tuple((uint64_t)stride_b, 1_I),
            opus::make_tuple((uint32_t)(tile_col + row_half_id * T::kBRowsTdm), 0u)));
    window_b.set_workgroup_mask(mask_b);
    window_b.set_tile_dim1((uint32_t)T::kBRowsTdm);

    auto tdm_window = is_operand_a ? window_a : window_b;
    // Slot size is not window state -- the ring walk scales by it -- so it stays a plain value select.
    const int slot_elems = is_operand_a ? T::kSlotElemsA : T::kSlotElemsB;

    // Why make_descriptor() and async_load() are kept apart wherever there are WMMAs to put between
    // them is documented on those two methods in opus.hpp. What is specific to here: there is
    // deliberately no asm-volatile anchor between the two. An empty asm volatile is only a
    // BarrierChain for MEMORY order, and a WMMA is not a memory op, so the WMMAs slide across it and
    // the anchor sinks down to the tensor_load with the whole chain in tow. A sched_barrier at the
    // call site enforces the separation instead.
    //
    // ld() fires the current descriptor, then advances global by one K step and
    // the LDS slot by +1 (mod P). All waves call ld() at the same steps.
    // `loaded` stays here rather than in opus: it is what sets the s_wait_tensorcnt depth, and that
    // depth is a WMMA-scheduling decision.
    int loaded = 0;
    // Index and element offset are carried side by side, rather than the offset alone wrapping on a
    // runtime P * slot_elems. Both are loop-carried, and keeping them separate keeps both chains short:
    // the wrap test is against the compile-time P, and the offset is a plain accumulate whose addend
    // the select has already produced. Folding them into one cursor costs 16 fewer instructions and
    // measures 0.5% slower, because add-compare-select then sits in series on one value.
    int slot_index = 0, slot_offset = 0;
    auto advance_slot = [&]() __attribute__((always_inline)) {
        const bool wrap = (slot_index + 1 == P);
        slot_offset += (wrap ? 1 - P : 1) * slot_elems;
        slot_index   = wrap ? 0 : slot_index + 1;
    };
    auto ld = [&]() __attribute__((always_inline)) {
        if (loaded > 0) {
            tdm_window.move(KStep);
            advance_slot();
        }
        tdm_window.async_load(slot_offset);   // fused: the prologue has no WMMA stream to spread across
        ++loaded;
    };

    // ── WMMA setup: per-msb sub-tile mma, M-split with wave_m = wave_id. ──
    auto mma = make_tiled_mma<DataA, DataB, DataAcc>(
        seq<T::kExpM, T::kExpNPerSub, T::kExpKHalf>{},
        seq<T::kTileM, T::kTileN, T::kTileK>{},
        seq<T::kWmmaM, T::kWmmaN, T::kWmmaK>{}, wmma_adaptor_swap_ab{});
    auto u_ra = make_layout_ra<T>(lane_id, wave_id);              // A: wave's M rows
    auto u_rb = make_layout_rb<sub_traits<T>>(lane_id, 0);        // B: one msb

    constexpr int NS  = T::kNumNSub;
    constexpr int HS  = T::kHalvesPerSlot;
    constexpr int NSUBROWS = T::kExpNPerSub * T::kTileN * T::kWmmaN;

    using VA = typename decltype(mma)::vtype_a;

    using bf16x16  = __attribute__((__vector_size__(16 * sizeof(DataA)))) DataA;
    // ext_vector_type, not vector_size: only the former satisfies opus::is_vector_v,
    // which is what smem::store<> dispatches on.
    using cx8      = opus::vector_t<DataC, 8>;      // one C store unit: 8 N, 16 B
    using u32x4    = opus::vector_t<unsigned, 4>;   // the same 16 B, as select-able DWORD
    static_assert(sizeof(cx8) == sizeof(u32x4), "the C store unit must be a whole number of DWORD");
    using f32x8    = __attribute__((__vector_size__(8 * sizeof(float)))) float;
    using bf16x256 = __attribute__((__vector_size__(256 * sizeof(DataA)))) DataA;
    using f32x256  = __attribute__((__vector_size__(256 * sizeof(float)))) float;

    // Both K-halves live at once. The allocator overlaps them, which leaves the
    // two remaining depctr_va_vdst(0) drains in the K loop; pinning them onto
    // disjoint ranges removes those drains but measured as a wash (+0.6% at
    // 4096/8192 cubed, -0.7% at K=16384) for 64 extra VGPRs, so they stay here.
    VA v_a[2];

    // Double-buffered B as per-WMMA-tile pinned sub-tiles. Each sub-tile is one
    // 16x16x32 WMMA's B operand (bf16x16 = 8 VGPR) so the WMMA reads it in place
    // -- no shuffle, hence no copy, hence the pin survives.
    //
    // One pinned variable per (buffer, sub-tile), assigned whole. A single wide
    // pinned vector written element-wise pins nothing: clang pins the value of
    // a store whose destination is the pinned local itself, and an element
    // write goes through a GEP. With neither buffer pinned the allocator sees
    // the sub-tiles die one at a time and overlaps the two buffers, so the
    // prefetch for tile t+1 lands on registers a tile-t WMMA has issued but not
    // yet read -- a WAR the hardware can only resolve by draining VA_VDST to 0
    // before the ds_read. That drain was every s_wait_alu in the K loop.
    //
    // Both banks sit in the same 256-VGPR window so the WMMA src0 field never
    // needs an S_SET_VGPR_MSB switch between sub-tiles.
#ifndef OPUS_B_PIN_BASE
#define OPUS_B_PIN_BASE 512
#endif
    constexpr int kBBuf0 = OPUS_B_PIN_BASE;        // buf0: 8 sub-tiles x 8 VGPR
    constexpr int kBBuf1 = OPUS_B_PIN_BASE + 64;   // buf1: the next 64
    // The pin rides on the written value, not on the declaration: OPUS_PIN_B
    // wraps what is stored into the sub-tile, so which writes carry a placement
    // request is visible here in the source. -DOPUS_NO_PIN=1 drops the request
    // and leaves everything else -- declarations, control flow, scheduling
    // groups -- identical, which is what makes the two builds comparable.
#define OPUS_BFRAG(b, k) bf16x16 Bt##b##_##k
#if OPUS_NO_PIN
#define OPUS_PIN_B(b, k, v) (v)
#else
#define OPUS_PIN_B(b, k, v) __builtin_amdgcn_pin_vgpr(v, kBBuf##b + (k) * 8)
#endif
    OPUS_BFRAG(0, 0); OPUS_BFRAG(0, 1); OPUS_BFRAG(0, 2); OPUS_BFRAG(0, 3);
    OPUS_BFRAG(0, 4); OPUS_BFRAG(0, 5); OPUS_BFRAG(0, 6); OPUS_BFRAG(0, 7);
    OPUS_BFRAG(1, 0); OPUS_BFRAG(1, 1); OPUS_BFRAG(1, 2); OPUS_BFRAG(1, 3);
    OPUS_BFRAG(1, 4); OPUS_BFRAG(1, 5); OPUS_BFRAG(1, 6); OPUS_BFRAG(1, 7);
#undef OPUS_BFRAG

    // buf/ksub are compile-time everywhere, so the dispatch folds away and each
    // access names one variable -- which is what keeps the store pinned.
#define OPUS_BFRAG_SET(b, k) if constexpr (bi == (b) && ki == (k)) Bt##b##_##k = OPUS_PIN_B(b, k, v); else
#define OPUS_BFRAG_GET(b, k) if constexpr (bi == (b) && ki == (k)) return Bt##b##_##k; else
    auto Bt_set = [&](auto bufN, auto ksubN, bf16x16 v) __attribute__((always_inline)) {
        constexpr int bi = decltype(bufN)::value, ki = decltype(ksubN)::value;
        OPUS_BFRAG_SET(0, 0) OPUS_BFRAG_SET(0, 1) OPUS_BFRAG_SET(0, 2) OPUS_BFRAG_SET(0, 3)
        OPUS_BFRAG_SET(0, 4) OPUS_BFRAG_SET(0, 5) OPUS_BFRAG_SET(0, 6) OPUS_BFRAG_SET(0, 7)
        OPUS_BFRAG_SET(1, 0) OPUS_BFRAG_SET(1, 1) OPUS_BFRAG_SET(1, 2) OPUS_BFRAG_SET(1, 3)
        OPUS_BFRAG_SET(1, 4) OPUS_BFRAG_SET(1, 5) OPUS_BFRAG_SET(1, 6) OPUS_BFRAG_SET(1, 7)
        static_assert(bi >= 0 && bi < 2 && ki >= 0 && ki < 8, "B sub-tile out of range");
    };
    auto Bt_get = [&](auto bufN, auto ksubN) __attribute__((always_inline)) -> bf16x16 {
        constexpr int bi = decltype(bufN)::value, ki = decltype(ksubN)::value;
        OPUS_BFRAG_GET(0, 0) OPUS_BFRAG_GET(0, 1) OPUS_BFRAG_GET(0, 2) OPUS_BFRAG_GET(0, 3)
        OPUS_BFRAG_GET(0, 4) OPUS_BFRAG_GET(0, 5) OPUS_BFRAG_GET(0, 6) OPUS_BFRAG_GET(0, 7)
        OPUS_BFRAG_GET(1, 0) OPUS_BFRAG_GET(1, 1) OPUS_BFRAG_GET(1, 2) OPUS_BFRAG_GET(1, 3)
        OPUS_BFRAG_GET(1, 4) OPUS_BFRAG_GET(1, 5) OPUS_BFRAG_GET(1, 6) OPUS_BFRAG_GET(1, 7)
        return bf16x16{};
    };
#undef OPUS_BFRAG_SET
#undef OPUS_BFRAG_GET
#undef OPUS_PIN_B

    // Hoist the per-lane ds_read element offsets OUT of the K loop: u_ra/u_rb do
    // not depend on slot/msb/half (those only move the smem base), so the offset
    // arrays are loop-invariant. Inside the loop each ds_read is
    // (base_elem + offs[i]); msb/half are compile-time and fold into the
    // ds_read `offset:` immediate, leaving only the runtime slot in the base
    // VGPR. The expression must be left in exactly this shape -- the immediates
    // are picked by the address-mode matcher on the `uniform + lane + static_i`
    // form, not by constant folding.
    constexpr int rElemA = opus::layout_load_traits<decltype(u_ra), T::kVecA>::r_elem.value;
    constexpr int rElemB = opus::layout_load_traits<decltype(u_rb), T::kVecB>::r_elem.value;
    auto offs_a = opus::layout_to_offsets<T::kVecA>(u_ra);
    auto offs_b = opus::layout_to_offsets<T::kVecB>(u_rb);

    auto raw_load_a = [&](int base_elem) __attribute__((always_inline)) {
        auto sm = make_smem(smem_a);
        VA r;
        opus::static_for<rElemA>([&](auto iN) __attribute__((always_inline)) {
            constexpr int i = iN.value;
            auto t = sm.template load<T::kVecA>(base_elem + offs_a[i]);
            #pragma unroll
            for (int j = 0; j < T::kVecA; ++j) r[i * T::kVecA + j] = t[j];
        });
        return r;
    };

    auto load_A = [&](int s, int half, int abuf) __attribute__((always_inline)) {
        v_a[abuf] = raw_load_a(s * slot_a + half * T::kKHalfElems);
    };

    // Load ONE B sub-tile (bf16x16 = one WMMA's B = 2 ds_read issues) directly
    // into the pinned Bt[buf][ksub] slot, so the K loop can interleave
    // individual B prefetches with individual WMMAs.
    auto load_B_sub = [&](int s, int half, int msb, auto bufN, auto ksubN) __attribute__((always_inline)) {
        constexpr int buf  = decltype(bufN)::value;
        constexpr int ksub = decltype(ksubN)::value;
        const int be = s * slot_b + msb * NSUBROWS * T::kSmemPitch + half * T::kKHalfElems;
        auto sm = make_smem(smem_b);
        bf16x16 sub;
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
            auto t = sm.template load<T::kVecB>(be + offs_b[2 * ksub + e]);
            #pragma unroll
            for (int j = 0; j < T::kVecB; ++j) sub[e * T::kVecB + j] = t[j];
        }
        Bt_set(bufN, ksubN, sub);
    };
    // Load one (half, msb) B tile as 8 per-WMMA sub-tiles.
    auto load_B = [&](int s, int half, int msb, auto bufN) __attribute__((always_inline)) {
        constexpr int buf = decltype(bufN)::value;
        opus::static_for<8>([&](auto kN) __attribute__((always_inline)) {
            load_B_sub(s, half, msb, opus::number<buf>{}, kN);
        });
    };

    constexpr int EM = T::kExpM, EN = T::kExpNPerSub, EK = T::kExpKHalf;
    constexpr int AL = 16, CL = 8;   // per-WMMA element counts (bf16 A, f32 C)

    // Persistent accumulators tiling v256-511, one pinned variable per WMMA
    // accumulator slice. A pinned slot has to be exactly as wide as the value
    // that uses it: a single f32x256 pinned at kAccBase carries a pin only on
    // its initialiser, because the per-element writes that follow are not
    // stores to the variable, so all 32 loop-carried definitions reach the
    // allocator unmarked and the accumulators land wherever they fit.
    constexpr int kAccBase = 256;
    constexpr int NSlice = EM * EN;              // c-slices per msb
    constexpr int NAcc   = NS * NSlice;
    static_assert(NAcc * CL == 256, "accumulators must tile v256..v511 exactly");
#if OPUS_NO_PIN
#define OPUS_ACC(i) f32x8 Acc##i = {}
#define OPUS_PIN_ACC(i, v) (v)
#else
#define OPUS_ACC(i)                                                            \
    f32x8 Acc##i = __builtin_amdgcn_pin_vgpr(f32x8{}, kAccBase + (i) * CL)
#define OPUS_PIN_ACC(i, v) __builtin_amdgcn_pin_vgpr(v, kAccBase + (i) * CL)
#endif
    OPUS_ACC(0);  OPUS_ACC(1);  OPUS_ACC(2);  OPUS_ACC(3);
    OPUS_ACC(4);  OPUS_ACC(5);  OPUS_ACC(6);  OPUS_ACC(7);
    OPUS_ACC(8);  OPUS_ACC(9);  OPUS_ACC(10); OPUS_ACC(11);
    OPUS_ACC(12); OPUS_ACC(13); OPUS_ACC(14); OPUS_ACC(15);
    OPUS_ACC(16); OPUS_ACC(17); OPUS_ACC(18); OPUS_ACC(19);
    OPUS_ACC(20); OPUS_ACC(21); OPUS_ACC(22); OPUS_ACC(23);
    OPUS_ACC(24); OPUS_ACC(25); OPUS_ACC(26); OPUS_ACC(27);
    OPUS_ACC(28); OPUS_ACC(29); OPUS_ACC(30); OPUS_ACC(31);
    // Like the B sub-tiles, the dispatch has to end in an assignment that names
    // the variable: a store reached through a returned reference is not a store
    // to the variable as far as the pin is concerned, and loses it.
#define OPUS_ACC_SET(i) if constexpr (ai == (i)) Acc##i = OPUS_PIN_ACC(i, v); else
#define OPUS_ACC_GET(i) if constexpr (ai == (i)) return Acc##i; else
    auto acc_set = [&](auto aiN, f32x8 v) __attribute__((always_inline)) {
        constexpr int ai = decltype(aiN)::value;
        OPUS_ACC_SET(0)  OPUS_ACC_SET(1)  OPUS_ACC_SET(2)  OPUS_ACC_SET(3)
        OPUS_ACC_SET(4)  OPUS_ACC_SET(5)  OPUS_ACC_SET(6)  OPUS_ACC_SET(7)
        OPUS_ACC_SET(8)  OPUS_ACC_SET(9)  OPUS_ACC_SET(10) OPUS_ACC_SET(11)
        OPUS_ACC_SET(12) OPUS_ACC_SET(13) OPUS_ACC_SET(14) OPUS_ACC_SET(15)
        OPUS_ACC_SET(16) OPUS_ACC_SET(17) OPUS_ACC_SET(18) OPUS_ACC_SET(19)
        OPUS_ACC_SET(20) OPUS_ACC_SET(21) OPUS_ACC_SET(22) OPUS_ACC_SET(23)
        OPUS_ACC_SET(24) OPUS_ACC_SET(25) OPUS_ACC_SET(26) OPUS_ACC_SET(27)
        OPUS_ACC_SET(28) OPUS_ACC_SET(29) OPUS_ACC_SET(30) OPUS_ACC_SET(31)
        static_assert(ai >= 0 && ai < NAcc, "accumulator index out of range");
    };
    auto acc_get = [&](auto aiN) __attribute__((always_inline)) -> f32x8 {
        constexpr int ai = decltype(aiN)::value;
        OPUS_ACC_GET(0)  OPUS_ACC_GET(1)  OPUS_ACC_GET(2)  OPUS_ACC_GET(3)
        OPUS_ACC_GET(4)  OPUS_ACC_GET(5)  OPUS_ACC_GET(6)  OPUS_ACC_GET(7)
        OPUS_ACC_GET(8)  OPUS_ACC_GET(9)  OPUS_ACC_GET(10) OPUS_ACC_GET(11)
        OPUS_ACC_GET(12) OPUS_ACC_GET(13) OPUS_ACC_GET(14) OPUS_ACC_GET(15)
        OPUS_ACC_GET(16) OPUS_ACC_GET(17) OPUS_ACC_GET(18) OPUS_ACC_GET(19)
        OPUS_ACC_GET(20) OPUS_ACC_GET(21) OPUS_ACC_GET(22) OPUS_ACC_GET(23)
        OPUS_ACC_GET(24) OPUS_ACC_GET(25) OPUS_ACC_GET(26) OPUS_ACC_GET(27)
        OPUS_ACC_GET(28) OPUS_ACC_GET(29) OPUS_ACC_GET(30) OPUS_ACC_GET(31)
        return f32x8{};
    };

    // Builtin (not asm) WMMA is required: asm plus a pinned physreg tuple hits a
    // copyPhysReg subreg assert. swap_ab: hardware wmma(D, src0=B, src1=A, C).
    auto wmma1 = [](bf16x16 a, bf16x16 b, f32x8 c) __attribute__((always_inline)) -> f32x8 {
        return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, b, false, a, (short)0, c, false, false);
    };

    // sched_group_barrier mask bits (AMDGPUIGroupLP): MFMA (WMMA on the XDL
    // pipe) = 1<<3, SALU = 1<<2, DS_READ = 1<<8.
    constexpr unsigned DSR = 0x100u, MFM = 0x08u, SAL = 0x04u;
    constexpr int kIlWmma = 1;    // WMMA per interleave group
    constexpr int kIlDs   = 2;    // ds_read per interleave group
    // SALU per group is a MAX. Ask for exactly one: the scalar pipe only
    // co-executes with the matrix pipe if the work is spread one op per gap, and
    // a larger value lets the solver dump several scalar ops into one WMMA gap
    // and leave the neighbouring gaps empty.
    constexpr int kIlSalu = 1;

    // Compute ONE 32x32x64 tile = EM x ENsub x EK = 8 WMMAs over 4 pinned B
    // sub-tiles. `nsub` selects which 32-N half of the msb's 64 N.
    constexpr int ENsub = 2;
    auto mma_tile32 = [&](VA& va, auto bufN, auto msbN, auto nsubN, auto&& prefetch)
                       __attribute__((always_inline)) {
        constexpr int msb  = decltype(msbN)::value;
        constexpr int buf  = decltype(bufN)::value;
        constexpr int nsub = decltype(nsubN)::value;
        opus::static_for<8>([&](auto wN) __attribute__((always_inline)) {
            constexpr int w   = wN.value;
            constexpr int ik  = w / (EM * ENsub);
            constexpr int im  = (w / ENsub) % EM;
            constexpr int inl = w % ENsub;
            constexpr int in  = nsub * ENsub + inl;
            constexpr int ia  = (im * EK + ik) * AL;
            constexpr int ai  = msb * NSlice + (im * EN + in);
            constexpr int bsub = in * EK + ik;
            bf16x16 a = __builtin_shufflevector(va, va,
                ia+0,ia+1,ia+2,ia+3,ia+4,ia+5,ia+6,ia+7,
                ia+8,ia+9,ia+10,ia+11,ia+12,ia+13,ia+14,ia+15);
            bf16x16 b = Bt_get(opus::number<buf>{}, opus::number<bsub>{});
            acc_set(opus::number<ai>{}, wmma1(a, b, acc_get(opus::number<ai>{})));
            prefetch(wN);
        });
        // Tile boundary. The wall stops the scheduler sinking a prefetch ds_read
        // past it down to its consumer in the next tile -- that sink is what
        // forced a per-WMMA dscnt(0) drain. Keep it fully closed: opening it to
        // SALU lets the list scheduler sink the descriptor chain toward the
        // tile-7 DMA where it bunches up, which measured ~1.5% slower.
        __builtin_amdgcn_sched_barrier(0);
    };

    // Software pipeline over NT = HS*NS msb-tiles, each two 32x32x64 sub-tiles.
    constexpr int NT = HS * NS;
    auto tile_half = [](int t){ return t / NS; };
    auto tile_msb  = [](int t){ return t % NS; };
    // front_load issues slot s's tile-0 dependencies. A half1 is not needed
    // until t == NT/2, so it goes last and stays in flight. Split out of the
    // compute body so it can be hoisted into the previous iteration's epilogue:
    // the ds for the next body then land while the current one runs, and the
    // body's tile-0 wait covers loads already in flight instead of a cold
    // dscnt(rElemA).
    auto front_load = [&](int s) __attribute__((always_inline)) {
        load_A(s, 0, 0);
        load_B(s, tile_half(0), tile_msb(0), opus::number<0>{});
        load_A(s, 1, 1);
        // Force the tile-0 front loads to ISSUE before the body's WMMAs, so
        // tile 0's counted wait covers in-flight loads instead of the scheduler
        // sinking the first-consumed B sub-tiles into tile 0's WMMAs.
        __builtin_amdgcn_sched_barrier(0);
    };

    // ── Hot loop ───────────────────────────────────────────────────────────
    //   Tile 0:      dscnt(rElemA) + WMMA + B prefetch
    //   Tiles 1-6:   dscnt(rElemB) + WMMA + B prefetch
    //   Tile NT-1:   dscnt(rElemB) + wait_tensorcnt + barrier_signal, then WMMA
    //                interleaved with {barrier_wait, front_load, next TDM}
    //   Epilogue:    A half1 + loop latch
    constexpr int kBarrierAhead = 3;   // WMMAs between barrier signal and wait
    // Where in the last tile the descriptor is built. Its SGPRs (12) stay live
    // until the fire on the last WMMA; the kernel uses ~36 SGPR, so this is free.
    constexpr int kTdmBuildW = 4;
    auto compute_body = [&](int s, int next_s) __attribute__((always_inline)) {
        typename WindowU::descriptor tdm_payload;
        tdm_window.move(KStep);
        advance_slot();
        opus::static_for<NT>([&](auto tN) __attribute__((always_inline)) {
            constexpr int t    = tN.value;
            constexpr int half = t / NS, msb = t % NS, buf = t & 1;
            if constexpr (t == 0) {
                opus::s_wait_dscnt(opus::number<rElemA>{});
            } else {
                opus::s_wait_dscnt(opus::number<rElemB>{});
            }
            if constexpr (t == NT - 1) {
                // The prologue primes exactly P and the body issues exactly one
                // per step, so loaded == P + g always and tensorcnt(1) is the
                // right wait unconditionally. Branch-free matters here: a
                // diamond would split this tile from its descriptor SALU.
                __builtin_amdgcn_s_wait_tensorcnt(1);
                __builtin_amdgcn_s_barrier_signal(-1);
            }
            opus::static_for<2>([&](auto nsN) __attribute__((always_inline)) {
                constexpr int nsub = nsN.value;
                mma_tile32(v_a[half], opus::number<buf>{}, opus::number<msb>{},
                           opus::number<nsub>{},
                    [&](auto wN) __attribute__((always_inline)) {
                        constexpr int w  = wN.value;
                        constexpr int gw = nsub * 8 + w;
                        if constexpr (t + 1 < NT) {
                            // Tiles 0..NT-2: prefetch the next tile's B sub-tiles.
                            //
                            // All eight up front, even though that is what
                            // leaves the K loop its counted s_wait_alu. A tile
                            // prefetches into the buffer the PREVIOUS tile read,
                            // so each ds_read carries a WAR against that tile's
                            // last WMMA on the same sub-tile, over a distance of
                            // (16 - last_read_slot) + prefetch_slot. Loading
                            // sub-tile k at slot k gives 14 11 15 12 10 7 11 8
                            // against a read order of 0 2 0 2 1 3 1 3 4 6 4 6 5
                            // 7 5 7, and the compiler spends six exactly-tight
                            // waits per tile on it. Moving each prefetch to the
                            // slot before that sub-tile's own last read pushes
                            // every distance to 15, which is as loose as va_vdst
                            // encodes, and drops the tight waits from 134 to 27
                            // -- and measured -0.4 to -3.0% on all but one
                            // shape, because it also flattens load-to-use from
                            // 14-22 WMMAs down to a uniform 14. The counted
                            // waits are free; the ds slack is not.
                            constexpr int nh = (t + 1) / NS, nm = (t + 1) % NS, nb = (t + 1) & 1;
                            if constexpr (gw < 8) {
                                load_B_sub(s, nh, nm, opus::number<nb>{}, opus::number<gw>{});
                                __builtin_amdgcn_sched_group_barrier(SAL, kIlSalu, 0);
                                __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                                __builtin_amdgcn_sched_group_barrier(DSR, kIlDs, 0);
                            } else {
                                // No ds left in this tile, so these WMMA slots
                                // are the cheapest place to put scalar work.
                                __builtin_amdgcn_sched_group_barrier(SAL, kIlSalu, 0);
                                __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                            }
                        } else {
                            // Last tile: barrier_wait + next slot's front_load +
                            // the next K step's TDM, all branch-free.
                            if constexpr (nsub == 0) {
                                if constexpr (w < kBarrierAhead - 1) {
                                    __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                                } else if constexpr (w == kBarrierAhead - 1) {
                                    __builtin_amdgcn_s_barrier_wait(-1);
                                    load_A(next_s, 0, 0);
                                    __builtin_amdgcn_sched_group_barrier(SAL, kIlSalu, 0);
                                    __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                                    __builtin_amdgcn_sched_group_barrier(DSR, rElemA, 0);
                                } else {
                                    load_B_sub(next_s, tile_half(0), tile_msb(0),
                                               opus::number<0>{}, opus::number<w - kBarrierAhead>{});
                                    __builtin_amdgcn_sched_group_barrier(SAL, kIlSalu, 0);
                                    __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                                    __builtin_amdgcn_sched_group_barrier(DSR, kIlDs, 0);
                                }
                            } else {
                                if constexpr (w < 3) {
                                    load_B_sub(next_s, tile_half(0), tile_msb(0),
                                               opus::number<0>{}, opus::number<w + 5>{});
                                    __builtin_amdgcn_sched_group_barrier(SAL, kIlSalu, 0);
                                    __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                                    __builtin_amdgcn_sched_group_barrier(DSR, kIlDs, 0);
                                } else {
                                    // No ds left: this is where the next K
                                    // step's TDM goes. Its descriptor SALU are
                                    // ready at body entry (they only depend on
                                    // the move) and the consumer is emitted on
                                    // the last WMMA, so the chain is free to
                                    // spread over these slots. The WMMAs in
                                    // between are 8 cycles apart, which covers
                                    // the SALU -> memory write-back for free.
                                    if constexpr (w == kTdmBuildW) {
                                        tdm_payload = tdm_window.make_descriptor(slot_offset);
                                        // Keep the descriptor SALU above and the
                                        // remaining WMMAs below, so the last of
                                        // them is ~18 cycles ahead of the DMA
                                        // issue. Cheap: no ds_read left to reorder.
                                        __builtin_amdgcn_sched_barrier(0);
                                    }
                                    if constexpr (w == 7) { WindowU::async_load(tdm_payload); ++loaded; }
                                    __builtin_amdgcn_sched_group_barrier(SAL, kIlSalu, 0);
                                    __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                                }
                            }
                        }
                    });
            });
        });
    };

#ifndef OPUS_C_TDM_STORE
#define OPUS_C_TDM_STORE 1
#endif
#if OPUS_C_TDM_STORE
    // ── C staging, hoisted above the loop so the last K step can fuse it ────
    // Everything here depends only on the lane's identity, so it costs nothing
    // to have it live across the loop; `smem_c` is the one runtime piece and is
    // pointed at an idle ring slot just before the step that first writes it.
    constexpr int kCRowsTdm   = T::kBlockM / T::kNumWaves;
    constexpr int kCTileElems = T::kBlockM * T::kBlockN;
    static_assert(kCTileElems * (int)sizeof(DataC) <= slot_b * (int)sizeof(DataB),
                  "the C tile must fit in the one B ring slot it stages through");

    // The row pitch is kBlockN * 2 = 512 B and 512/4 is a multiple of 32, so M
    // does not move the LDS bank at all: a store where all 16 lanes of a row
    // group share one N lands every one of them on the same bank pair. Measured
    // at 3-23% of total runtime depending on shape.
    //
    // Each lane holds one row (M = im*64 + wave*16 + lane_m) and four 8-element
    // N groups (N = in*16 + lane_n, in = 0..3). Storing group (lane_m + s) % 4
    // on step s puts the 32 lanes on N/2 = {0,8,16,24} and the same +4, so the
    // four b128 dwords per lane tile all 32 banks exactly 4 deep -- the minimum
    // a 32-lane b128 store can reach. The rotation is per lane and a register
    // number is not, so the groups are rotated into place first; that is what
    // the two v_cndmask layers below buy.
    //
    // swap_ab transposes the WMMA C layout, so grpn_c is the group that walks M
    // and grpm_c splits N; the arithmetic below reads both as masks.
    using mma_t = opus::remove_cvref_t<decltype(mma)>;
    static_assert(mma_t::grpn_c == T::kWmmaM, "swap_ab must put grpn_c on the M axis");
    static_assert(mma_t::grpm_c * (T::kWmmaN / mma_t::grpm_c) == T::kWmmaN, "N must split evenly over grpm_c");
    static_assert((EN & (EN - 1)) == 0, "the group rotation masks with EN - 1");
    static_assert(EN * T::kWmmaN == NSUBROWS, "a msb sub-tile is exactly EN groups of kWmmaN");

    const int lane_m = lane_id % mma_t::grpn_c;                              // this lane's row, 0..15
    const int lane_n = (lane_id / mma_t::grpn_c) * (T::kWmmaN / mma_t::grpm_c);
    const int rot    = lane_m & (EN - 1);
    const int c_base = (wave_id * T::kWmmaM + lane_m) * T::kBlockN + lane_n;
    DataC* smem_c = nullptr;

    auto sel4 = [](bool c, u32x4 a, u32x4 b) __attribute__((always_inline)) {
        u32x4 r;
        #pragma unroll
        for (int i = 0; i < 4; ++i) r[i] = c ? a[i] : b[i];
        return r;
    };

    // Cast one msb's accumulators to bf16 and stage them, rotated. msb walks N,
    // which is the contiguous axis, so it folds into the base.
    //
    // The ds_store reads a register the cvt just wrote, so each store burst is
    // fronted with a depctr_va_vdst(0) drain -- 10 of them across the peeled
    // step and the epilogue. Both ways of buying distance measured worse than
    // leaving them: draining the cvt into the first half of a tile's free slots
    // and the stores into the second half fragments the bursts into more drains
    // (13) and costs 0.4-1.6%, and moving the cvt a whole tile ahead of its
    // store does cut them to 8 but pushes one msb's store out of the loop
    // entirely, costing 0.1-2.1%. The drains are not what this region is
    // waiting on.
    auto store_msb = [&](auto mN) __attribute__((always_inline)) {
        constexpr int msb = decltype(mN)::value;
        auto s_c = make_smem(smem_c + msb * NSUBROWS);
        opus::static_for<EM>([&](auto imN) __attribute__((always_inline)) {
            constexpr int im = decltype(imN)::value;
            u32x4 g[EN];
            opus::static_for<EN>([&](auto inN) __attribute__((always_inline)) {
                constexpr int in = decltype(inN)::value;
                cx8 v;
                const f32x8 av = acc_get(opus::number<msb * NSlice + (im * EN + in)>{});
                #pragma unroll
                for (int j = 0; j < CL; ++j)
                    v[j] = (DataC)av[j];
                g[in] = __builtin_bit_cast(u32x4, v);
            });
            // Rotate the four groups left by `rot`, as two conditional stages.
            u32x4 t[EN], o[EN];
            #pragma unroll
            for (int q = 0; q < EN; ++q) t[q] = sel4(rot & 1, g[(q + 1) & (EN - 1)], g[q]);
            #pragma unroll
            for (int q = 0; q < EN; ++q) o[q] = sel4(rot & 2, t[(q + 2) & (EN - 1)], t[q]);
            opus::static_for<EN>([&](auto sN) __attribute__((always_inline)) {
                constexpr int s = decltype(sN)::value;
                const int n_off = (rot * T::kWmmaN + s * T::kWmmaN) & (NSUBROWS - 1);
                s_c.template store<8>(__builtin_bit_cast(cx8, o[s]),
                                      c_base + im * (T::kTileM * T::kWmmaM) * T::kBlockN + n_off);
            });
        });
    };
#endif
    // msb m's accumulators are final once tile NS+m retires, so the staging for
    // the first kFusedMsb sub-tiles fits under the WMMAs that follow. The last
    // msb only completes on the very last tile and has to stay in the epilogue.
#ifndef OPUS_C_FUSED_MSB
#define OPUS_C_FUSED_MSB (NS - 1)
#endif
    constexpr int kFusedMsb = OPUS_C_TDM_STORE ? (OPUS_C_FUSED_MSB) : 0;
    static_assert(kFusedMsb < NS, "the last msb only completes on the very last tile");

    // ── Prologue: prime the P-deep ring, 2 TDMs stay in flight ──────────────
    //   step 0 - issued and drained here, published by the barrier
    //   step 1 - issued here, drained at the end of iteration g = 0
    //   step 2 - issued here, drained at the end of iteration g = 1
    // Each half-TDM therefore gets two full compute bodies to complete.
#define OPUS_ACC_ZERO(i) Acc##i = f32x8{}
    OPUS_ACC_ZERO(0);  OPUS_ACC_ZERO(1);  OPUS_ACC_ZERO(2);  OPUS_ACC_ZERO(3);
    OPUS_ACC_ZERO(4);  OPUS_ACC_ZERO(5);  OPUS_ACC_ZERO(6);  OPUS_ACC_ZERO(7);
    OPUS_ACC_ZERO(8);  OPUS_ACC_ZERO(9);  OPUS_ACC_ZERO(10); OPUS_ACC_ZERO(11);
    OPUS_ACC_ZERO(12); OPUS_ACC_ZERO(13); OPUS_ACC_ZERO(14); OPUS_ACC_ZERO(15);
    OPUS_ACC_ZERO(16); OPUS_ACC_ZERO(17); OPUS_ACC_ZERO(18); OPUS_ACC_ZERO(19);
    OPUS_ACC_ZERO(20); OPUS_ACC_ZERO(21); OPUS_ACC_ZERO(22); OPUS_ACC_ZERO(23);
    OPUS_ACC_ZERO(24); OPUS_ACC_ZERO(25); OPUS_ACC_ZERO(26); OPUS_ACC_ZERO(27);
    OPUS_ACC_ZERO(28); OPUS_ACC_ZERO(29); OPUS_ACC_ZERO(30); OPUS_ACC_ZERO(31);
    // Cluster sync before the first multicast TDM, so the peers of a cluster
    // arrive together and GL1 can merge their requests instead of releasing each
    // one on its timeout.
    __builtin_amdgcn_s_barrier();
    if (wave_id == 0) {
        __builtin_amdgcn_s_barrier_signal(-3);
        __builtin_amdgcn_s_barrier_wait(-3);
    }
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    // Workgroups with no tile leave here, having paid for nothing but the
    // barrier: the grid is rounded up to whole clusters -- the runtime rejects a
    // grid that is not a multiple of the cluster dims -- so an edge cluster
    // carries surplus workgroups, and without this they run the entire K loop on
    // zero-extent DMAs. Nothing above touches LDS or the TDM, and the first ld()
    // is below, so the exit costs the cluster no work it has to undo.
    //
    // This is the earliest point they may go, not the latest they can be caught:
    // -3 counts one arrival PER WORKGROUP, so skipping the barrier above would
    // hang every peer of the cluster forever. It is also the only -3 in the
    // kernel -- every later barrier is workgroup-scope (-1), and all 4 waves of
    // a workgroup take this workgroup-uniform branch together -- so once past it
    // no peer ever waits on a workgroup that has gone.
    //
    // The survivors still name the departed in their multicast masks, which
    // neither hangs nor writes into the dead LDS: GL1 returns only to the waves
    // that made a request. Such a request merges with fewer peers than the mask
    // claims and so waits out the timeout, which is why an edge cluster gains
    // less than the K loop it skips would suggest.
    if (tile_oob) return;
    // Unconditional P-deep prime. Once origin0 passes K the descriptor's
    // tensor_dim0 saturates to 0, so a step beyond the last is a zero-extent
    // DMA: it touches no memory and only bumps tensorcnt. That makes
    // loaded == P + g hold for every shape, which is what lets the hot loop use
    // an unconditional wait_tensorcnt(1) and issue the TDM without a guard.
    opus::static_for<P>([&](auto) __attribute__((always_inline)) { ld(); });
    __builtin_amdgcn_s_wait_tensorcnt(P - 1);   // land step 0, keep the rest in flight
    __builtin_amdgcn_s_barrier_signal(-1);      // publish step 0
    __builtin_amdgcn_s_barrier_wait(-1);
    front_load(0);

    // The peeled last K step. Nothing follows it, so it drops everything the
    // body does on behalf of the next one -- the next slot's front_load, the
    // next TDM, and the barrier that hands the slot back -- and spends the
    // freed slots on the C staging instead. Tiles NS+1..NT-1 each carry the
    // sub-tile the previous tile finished, so the cvt and ds_store issue in the
    // shadow of the remaining WMMAs rather than after all of them.
    auto final_step = [&](int s) __attribute__((always_inline)) {
        opus::static_for<NT>([&](auto tN) __attribute__((always_inline)) {
            constexpr int t    = tN.value;
            constexpr int half = t / NS, msb = t % NS, buf = t & 1;
            constexpr int fused = t - NS - 1;   // the msb tile t-1 completed
            if constexpr (t == 0) {
                opus::s_wait_dscnt(opus::number<rElemA>{});
            } else {
                opus::s_wait_dscnt(opus::number<rElemB>{});
            }
            opus::static_for<2>([&](auto nsN) __attribute__((always_inline)) {
                constexpr int nsub = nsN.value;
                mma_tile32(v_a[half], opus::number<buf>{}, opus::number<msb>{},
                           opus::number<nsub>{},
                    [&](auto wN) __attribute__((always_inline)) {
                        constexpr int w  = wN.value;
                        constexpr int gw = nsub * 8 + w;
                        if constexpr (gw < 8 && t + 1 < NT) {
                            constexpr int nh = (t + 1) / NS, nm = (t + 1) % NS, nb = (t + 1) & 1;
                            load_B_sub(s, nh, nm, opus::number<nb>{}, opus::number<gw>{});
                            __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                            __builtin_amdgcn_sched_group_barrier(DSR, kIlDs, 0);
                        } else if constexpr (fused >= 0 && fused < kFusedMsb) {
                            // One msb is 32 cvt and 8 ds_store; the slots left in
                            // this tile are 8 (or 16 on the last, which has no
                            // prefetch), so ask for the per-WMMA share of each.
                            constexpr int kSlots = (t + 1 < NT) ? 8 : 16;
                            constexpr int kCvt   = 32 / kSlots;
                            if constexpr (gw == 16 - kSlots) store_msb(opus::number<fused>{});
                            __builtin_amdgcn_sched_group_barrier(0x002u, kCvt, 0);
                            __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                            __builtin_amdgcn_sched_group_barrier(0x200u, 8 / kSlots, 0);
                        } else {
                            __builtin_amdgcn_sched_group_barrier(MFM, kIlWmma, 0);
                        }
                    });
            });
        });
    };

    // One K step: the body, then the tail the group barriers below order.
    auto one_step = [&](int s, int next_s) __attribute__((always_inline)) {
        compute_body(s, next_s);
        // Keep the tail region holding the A-half1 ds_reads together with the
        // loop latch so the group barriers below can order them, while still
        // stopping the last tile from sinking into the tail.
        __builtin_amdgcn_sched_barrier(0);
        // Unconditional: on the last step this reads a ring slot whose contents
        // are never consumed. Guarding it fences these ds_reads off into their
        // own branch target, where there is no WMMA to hide them behind.
        load_A(next_s, 1, 1);
        // The SALU does not forward into branches either, so the
        // latch's s_cmp eats the full SGPR write-back when it lands next to
        // s_cbranch -- which is exactly where a bottom-up scheduler puts it.
        // Order the tail as [latch SALU][ds_read] so the loads cover the delay.
        __builtin_amdgcn_sched_group_barrier(SAL, 4, 0);
        __builtin_amdgcn_sched_group_barrier(DSR, rElemA, 0);
    };

    // Unroll by P so the ring index is a LITERAL in every copy. With `s` at
    // runtime, `s * slot_elems` is a VALU inside the loop, and because it feeds
    // a ds_read address the compiler guards it with s_wait_alu
    // depctr_va_vdst(0) -- which drains the ENTIRE in-flight WMMA pipe, not just
    // the one write being waited on. With `s` constant the multiply folds into
    // the ds_read `offset:` immediates and the few bases that overflow the
    // 16-bit field become loop-invariant, so LICM hoists them out.
    //
    // The remainder runs AFTER the unrolled loop, not before: n_full is a
    // multiple of P, so the ring index is back at 0 on exit and the tail can
    // carry it at runtime. Peeling in front would enter the unrolled loop at a
    // runtime slot, which defeats the whole transform.
    //
    // Spelled out rather than static_for<P>: the accumulators stay register-resident only
    // as long as SROA can see through every layer wrapping it, and the extra
    // lambda a static_for introduces was enough to lose it -- the whole
    // accumulator went to scratch. Keep the nesting depth identical to the
    // pre-unroll loop.
    // The last step is peeled off so it can run final_step, and it takes a
    // runtime slot rather than a three-way dispatch on one.
    //
    // The remainder is dispatched rather than looped. A loop here would cost
    // the accumulators their pinned tuples: the main loop's values have to stay
    // live across it, because a remainder that runs zero times leaves them as
    // the result, so its range overlaps the remainder's own and the two cannot
    // share a register. Only one of them then gets the tuple and the other is
    // placed wherever, which the epilogue pays for in copies to bring the two
    // halves back together. Dispatching leaves each accumulator one live range:
    // pinned placement goes from 444 of 640 WMMA to all of them, and the copies
    // -- 607 v_mov, none of them in the K loop -- drop to 126. That pays for
    // the second copy of the body outright (129 fewer instructions overall, and
    // 747 VGPRs against 848), though it does not move the clock: the copies it
    // removes all sat in the epilogue, which runs once.
    //
    // The dispatch is exact because the remainder is at most P-1 steps and the
    // ring index is back at 0 on entry, so the slots it visits are a compile
    // time constant -- the same property the unrolled loop above relies on.
    static_assert(P == 3, "the unrolled body below is written out for P == 3");
    const int n_body = k_steps - 1;
    const int n_full = n_body - n_body % P;
    for (int g = 0; g < n_full; g += P) {
        one_step(0, 1);
        one_step(1, 2);
        one_step(2, 0);
    }
    int s_cur = 0;
    const int n_rem = n_body - n_full;
    if (n_rem >= 1) { one_step(0, 1); s_cur = 1; }
    if (n_rem >= 2) { one_step(1, 2); s_cur = 2; }
#if OPUS_C_TDM_STORE
    // The peeled step computes out of slot s_cur, so the free one is the next.
    // Free rather than merely finished: the only TDM ever aimed there is for a
    // K step past the last, which the D# saturates to zero extent. That is what
    // lets a wave stage C mid-step with no barrier in front of it -- staging
    // into a slot the group still owns would need one, and would push the whole
    // cvt and ds_store block back behind the slowest peer.
    smem_c = reinterpret_cast<DataC*>(smem_b + (size_t)((s_cur + 1 == P) ? 0 : (s_cur + 1)) * slot_b);
#endif
    final_step(s_cur);
    // Retire the trailing zero-extent DMAs before the store reuses the SGPRs.
    __builtin_amdgcn_s_wait_tensorcnt(0);

    // ── Epilogue: cast the fp32 accumulators to bf16 and write the B_M x B_N
    // tile out, NS msb sub-tiles at a time. Two routes, same C layout: a TDM
    // store through an LDS staging copy, or the per-lane buffer_store it
    // replaced. ──
#if OPUS_C_TDM_STORE
    // Stage C in the A/B ring -- dead once the last DMA retired -- and hand the
    // global write to one TDM per wave. The DMA moves whole B_N rows, so the
    // ragged tile needs no guard (the D# clamps origin against the extents,
    // exactly as the load side relies on) and no per-store byte count.
    using StoreWin = opus::tdm<DataC, opus::seq<T::kBlockN, kCRowsTdm>>;

    // Whatever the peeled step did not already stage. Same grouping as the
    // buffer_store path below, for the same reason: left interleaved, every
    // ds_store inherits the VALU->LDS operand hazard.
    constexpr unsigned VAL = 0x002u, DSW = 0x200u;
    opus::static_for<NS - kFusedMsb>([&](auto mN) __attribute__((always_inline)) {
        store_msb(opus::number<kFusedMsb + decltype(mN)::value>{});
        __builtin_amdgcn_sched_group_barrier(VAL, 48, 0);
        __builtin_amdgcn_sched_group_barrier(DSW, 8, 0);
    });

    // The barrier orders the waves against each other but not against their own
    // in-flight LDS traffic, so the ds_writes have to be retired explicitly
    // before a peer's DMA is allowed to read the rows they land in.
    opus::s_wait_dscnt(opus::number<0>{});
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    // The store partition is by contiguous M, independent of the compute
    // partition (a wave's own accumulator rows are two strided 16-row chunks).
    auto window_c = opus::make_tdm<StoreWin>(
        (uint32_t)(reinterpret_cast<uintptr_t>(smem_c)
                   + (size_t)wave_id * kCRowsTdm * T::kBlockN * sizeof(DataC)),
        (void*)base_c,
        opus::make_layout<-1>(
            opus::make_tuple((uint32_t)kargs.m, (uint32_t)kargs.n),
            opus::make_tuple((uint64_t)stride_c, 1_I),
            opus::make_tuple((uint32_t)(tile_row + wave_id * kCRowsTdm), (uint32_t)tile_col)));
    window_c.async_store();
    __builtin_amdgcn_s_wait_tensorcnt(0);
#else
    auto p_coord_c = opus::make_tuple(wave_id, lane_id % mma.grpn_c, 0, lane_id / mma.grpn_c);

    // Slice msb out of the accumulators into one WMMA-C-shaped register tile.
    auto gather_msb = [&](auto mN) __attribute__((always_inline)) {
        constexpr int msb = decltype(mN)::value;
        typename decltype(mma)::vtype_c reg_c;
        opus::static_for<NSlice>([&](auto sN) __attribute__((always_inline)) {
            constexpr int islice = sN.value;
            const f32x8 av = acc_get(opus::number<msb * NSlice + islice>{});
            #pragma unroll
            for (int j = 0; j < CL; ++j)
                reg_c[islice * CL + j] = av[j];
        });
        return reg_c;
    };

    auto u_gc = partition_layout_c<T::kCVec>(mma, opus::make_tuple((int)stride_c, 1_I), p_coord_c);

    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    if (tile_row < kargs.m && tile_col < kargs.n) {
        opus::static_for<NS>([&](auto mN) __attribute__((always_inline)) {
            constexpr int msb = decltype(mN)::value;
            const size_t c_base = (size_t)tile_row * (size_t)stride_c
                                + (size_t)tile_col + (size_t)msb * NSUBROWS;
            const unsigned int c_bytes =
                (unsigned int)(((size_t)kargs.m * (size_t)stride_c - c_base) * sizeof(DataC));
            auto g_c = make_gmem<DataC>(base_c + c_base, c_bytes);
            store<T::kCVec>(g_c, cast<DataC>(gather_msb(mN)), u_gc, 0);
        });
        // Left alone the scheduler interleaves each v_cvt_pk_bf16_f32 with the
        // buffer_store that consumes it. Every store then hits the VALU->VMEM
        // operand hazard and gets fronted with an s_wait_alu depctr_va_vdst(0),
        // which drains the VALU pipe 64 times and stops the stores clausing --
        // ~2us per workgroup, and with one workgroup per CU (306KB LDS) nothing
        // overlaps it. Group each sub-tile as [all VALU][all stores] instead.
        constexpr unsigned VAL = 0x002u, VMW = 0x040u;
        opus::static_for<NS>([&](auto) __attribute__((always_inline)) {
            __builtin_amdgcn_sched_group_barrier(VAL, 48, 0);
            __builtin_amdgcn_sched_group_barrier(VMW, 16, 0);
        });
    }
#endif
}
