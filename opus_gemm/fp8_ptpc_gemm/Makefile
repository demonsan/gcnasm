OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
OPUS_GEMM_INCLUDE_DIR ?= $(OPUS_INCLUDE_DIR)/../opus_gemm/include
HIPCC ?= /opt/rocm/bin/hipcc
PYTHON ?= python3
ARCH ?= gfx950
BUILD ?= build

COMMON_FLAGS = -std=c++17 -O3 --offload-arch=$(ARCH) -fPIC -shared \
	-Iinclude/gfx950 -I$(OPUS_GEMM_INCLUDE_DIR) -I$(OPUS_INCLUDE_DIR)
HEADERS = $(wildcard include/gfx950/*.cuh)
SMALL = $(BUILD)/ptpc_c11_mature_gfx950.so
LARGE = $(BUILD)/ptpc_c19_256192_gfx950.so

.PHONY: all check check-includes test benchmark clean
all: $(SMALL) $(LARGE)

check-includes:
	@test "$(ARCH)" = gfx950 || { echo "Only gfx950 is supported (other device bodies are stubs)" >&2; exit 2; }
	@test -f "$(OPUS_INCLUDE_DIR)/opus/opus.hpp" || { echo "Set OPUS_INCLUDE_DIR to aiter/csrc/include" >&2; exit 2; }
	@test -f "$(OPUS_GEMM_INCLUDE_DIR)/opus_gemm_utils.cuh" || { echo "Set OPUS_GEMM_INCLUDE_DIR to aiter/csrc/opus_gemm/include" >&2; exit 2; }

$(BUILD):
	mkdir -p "$@"

$(SMALL): gemm_a8w8_ptpc_splitk_kernel.cc $(HEADERS) | $(BUILD) check-includes
	$(HIPCC) $(COMMON_FLAGS) $< -o "$@"

$(LARGE): gemm_a8w8_ptpc_8wave_kernel.cc $(HEADERS) | $(BUILD) check-includes
	$(HIPCC) $(COMMON_FLAGS) -ffp-contract=off $< -o "$@"

check:
	$(PYTHON) -m unittest discover -s tests -v

test: all
	$(PYTHON) gemm_host.py --build-dir "$(BUILD)" --matrix17 --verify-only

benchmark: all
	$(PYTHON) gemm_host.py --build-dir "$(BUILD)" --matrix17

clean:
	rm -f "$(SMALL)" "$(LARGE)"
