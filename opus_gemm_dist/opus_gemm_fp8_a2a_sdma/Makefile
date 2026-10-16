OPUS_INCLUDE_DIR ?= ../../mla_agentic_v0/third_party/aiter/csrc/include
MORI_INCLUDE_DIR ?= /workspace/mori/include
MORI_LIB_DIR     ?= /workspace/mori/build/src/cco
HIPCC            ?= /opt/rocm/bin/hipcc
ARCH             ?= gfx950
# PLAN selects the build: dense (N=9216, F=2304; also serves the REUSE case, which is the dense plan with the compact receive layout)
# or indexer (N=9856, F=2560). The plan is a compile-time constant of the kernels and of main.cc (it fixes the window size, grid and item order).
PLAN             ?= dense
# FOLD=0 is the production configuration (separate quiet_ready kernel). FOLD=1 folds W2/reset/epoch into the poster block (measured slower at T8192; kept for experiments).
FOLD             ?= 0
# Optional debug seams (default off, never for timing): see README "Debug flags".
DEBUG_POSTCOUNT  ?= 0
STAMPS           ?= 0

ifeq ($(PLAN),dense)
INDEXER := 0
else ifeq ($(PLAN),indexer)
INDEXER := 1
else
$(error PLAN must be dense or indexer)
endif

MPI_INCLUDE_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB         ?= -lmpi_cxx -lmpi

BUILD = build_$(PLAN)
OUT   = $(BUILD)/fp8_a2a_sdma_$(PLAN).exe

# Flags shared by every object. -fno-fast-math is REQUIRED: the epilogue is bf16_rne(fp32(fp32(acc*Wscale)*Ascale)) (two separate multiplies, one rounding), and the
# MFMA/barrier schedule must not be reassociated. BUILD_CCO_SDMA=1 is required (the MORI header default 0 turns SDMA puts into silent no-ops).
LG_FLAGS     = -DBUILD_CCO_SDMA=1 -DLG_X_INDEXER=$(INDEXER) -DLG_R8_FOLD=$(FOLD) -DLG_REMOTE_HALF=0 -DLG_R6_POSTER=1 -DLG_R6_ARRIVAL_ACQ=0 \
               -DLG_L2B_ONE_RELEASE=1 -DLG_L2A_BYPASS=0 -DLG_L1C_SWIZZLE=0 -DLG_DEBUG_POSTCOUNT=$(DEBUG_POSTCOUNT) -DLG_STAMPS=$(STAMPS)
COMMON_FLAGS = -I. -I$(abspath $(OPUS_INCLUDE_DIR)) -std=c++20 -fopenmp -O3 -Wall -fno-fast-math --offload-arch=$(ARCH) -DOPUS_TILE_CONTAINER=0 $(LG_FLAGS)
HOST_INCLUDES = -I$(MORI_INCLUDE_DIR) -I$(MPI_INCLUDE_DIR)
HOST_LIBS = -L$(MORI_LIB_DIR) -lmori_cco -Wl,-rpath,$(MORI_LIB_DIR) $(MPI_LIB)

HDRS = lg_comm_abi_v2.h lg_comm_device.hpp lg_r6x.hpp epilogue_ptpc_pairstore.hpp layout/quad_fp8_fieldmap.hpp layout/quad_fp8_layouts.hpp layout/quad_fp8_traits.hpp
OBJS = $(BUILD)/gemm_fp8_a2a_sdma_kernel.o $(BUILD)/main.o

.PHONY: all dense indexer clean run

all: $(OUT)

dense:
	$(MAKE) PLAN=dense

indexer:
	$(MAKE) PLAN=indexer

$(BUILD):
	mkdir -p $(BUILD)

# The kernel TU is compiled in RTC mode (-D__HIPCC_RTC__, like opus_gemm_a2a_lsa): the device pass uses the private Opus headers only; the host pass of the same file
# defines empty __global__ stubs so main.o can launch the kernels with hipLaunchKernelGGL.
$(BUILD)/gemm_fp8_a2a_sdma_kernel.o: gemm_fp8_a2a_sdma_kernel.cc $(HDRS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/main.o: main.cc lg_comm_abi_v2.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -c -o $@

$(OUT): $(OBJS)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp $(HOST_LIBS) -o $@

# 4 ranks on GPUs 0..3 of one node (rank == lsaRank == HIP device id is required by the kernels).
run: $(OUT)
	mpirun --allow-run-as-root -n 4 $(OUT) --case $(if $(filter indexer,$(PLAN)),indexer,dense)

clean:
	rm -rf build_dense build_indexer