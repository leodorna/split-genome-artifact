# Build the split-genome generation kernels and their benchmark.
#   make CUDA_ARCH=120 MATHDX_DIR=/path/to/mathdx/25.12/include
# CUDA_ARCH is the SM architecture of the GPU (120: RTX PRO 6000 Blackwell, 86: RTX A6000).

CUDA_ARCH  ?= 120
MATHDX_DIR ?= external/mathdx/25.12/include
NVCC       ?= nvcc
BIN        := bin

NVCCFLAGS := -std=c++20 -x cu -O3 -diag-suppress 3056 \
             -arch=sm_$(CUDA_ARCH) -DTARGET_SM=$(CUDA_ARCH)0 \
             -Iinclude -I$(MATHDX_DIR)
HEADERS   := $(wildcard include/*.hpp include/*.ipp include/kernels/*.hpp)
# Changing CUDA_ARCH rebuilds every binary: the stamp names the architecture they were built for.
ARCH_STAMP := $(BIN)/.arch_sm_$(CUDA_ARCH)

.PHONY: all bench roofline test clean check-mathdx
all: bench

check-mathdx:
	@test -f $(MATHDX_DIR)/curanddx.hpp || { \
	  echo "cuRANDDx not found in $(MATHDX_DIR)."; \
	  echo "Download NVIDIA MathDx 25.12 and set MATHDX_DIR (see README.md)."; exit 1; }

$(BIN)/bench_generation_v2: apps/bench_generation_v2.cpp $(HEADERS) $(ARCH_STAMP) | check-mathdx
	@mkdir -p $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN)/roofline_ceilings_v3: apps/roofline_ceilings_v3.cu $(HEADERS) $(ARCH_STAMP) | check-mathdx
	@mkdir -p $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN)/test_rng_streams: tests/test_rng_streams.cu include/common.ipp $(ARCH_STAMP) | check-mathdx
	@mkdir -p $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(ARCH_STAMP):
	@mkdir -p $(BIN)
	@rm -f $(BIN)/.arch_sm_*
	@touch $@

bench: $(BIN)/bench_generation_v2
roofline: $(BIN)/roofline_ceilings_v3

test: $(BIN)/test_rng_streams $(BIN)/bench_generation_v2
	./$(BIN)/test_rng_streams
	./tests/test_generation_v2.sh
	./tests/test_fitness_matrix.sh

clean:
	rm -rf $(BIN)
