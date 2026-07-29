OPUS_INCLUDE_DIR      ?= /shared/amdgpu/home/jiahao_zhou_qle/blyu/aiter/csrc/include
OPUS_GEMM_INCLUDE_DIR ?= /shared/amdgpu/home/jiahao_zhou_qle/blyu/aiter/csrc/opus_gemm/include
MORI_INCLUDE_DIR      ?= /shared/amdgpu/home/jiahao_zhou_qle/blyu/mori/include
MORI_LIB_DIR          ?= /shared/amdgpu/home/jiahao_zhou_qle/blyu/mori/python/mori
HIPCC                 ?= /opt/rocm/bin/hipcc
ARCH                  ?= gfx950

MPI_INCLUDE_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB         ?= -lmpi_cxx -lmpi

BUILD = build
OUT   = $(BUILD)/opus_gemm_mem_gemm_a2a.exe

COMMON_FLAGS = -I. -Iinclude -I$(OPUS_INCLUDE_DIR) -I$(OPUS_GEMM_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math
HOST_INCLUDES = -I$(MORI_INCLUDE_DIR) -I$(MPI_INCLUDE_DIR)
HOST_LIBS = -L$(MORI_LIB_DIR) -lmori_cco -Wl,-rpath,$(MORI_LIB_DIR) -lrccl $(MPI_LIB)

OBJS = $(BUILD)/opus_gemm_mem_gemm_a2a_kernel.o $(BUILD)/main.o

.PHONY: all clean run

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

LOCAL_HEADERS = \
	include/opus_gemm_utils.cuh \
	include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh \
	include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh \
	include/gfx950/splitk_reduce_gfx950.cuh

$(BUILD)/opus_gemm_mem_gemm_a2a_kernel.o: opus_gemm_mem_gemm_a2a_kernel.cc opus_gemm_mem_gemm_a2a.h $(LOCAL_HEADERS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/main.o: main.cc opus_gemm_mem_gemm_a2a.h $(LOCAL_HEADERS) | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -c -o $@

$(OUT): $(OBJS)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp $(HOST_LIBS) -o $@

run: $(OUT)
	mpirun --allow-run-as-root -n 8 $(OUT)

clean:
	rm -rf $(BUILD)

