CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

CUDA_ARCH ?= -arch=sm_90

CXXFLAGS  := -std=c++17 -O3
NVCCFLAGS := $(CUDA_ARCH) $(CXXFLAGS)

TARGET    := pdl_bench.exe
SRC       := pdl_bench.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
