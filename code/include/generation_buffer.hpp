#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <cuda_runtime.h>

// Double-buffered population/fitness + highdim survivor scratch for the v2
// generation kernels. The value type is the operator's (float or packed uint32);
// a binary buffer stores nRows * d/32 words per population.

namespace evoopt {
namespace bench {
namespace v2 {

template <typename T>
struct GenerationBuffer {
    static constexpr bool binary = std::is_same_v<T, uint32_t>;

    static size_t elems(size_t d) { return binary ? d / 32 : d; }

    T* pop[2] = {nullptr, nullptr};
    T* trial = nullptr;            // offspring scratch (decomposed pipeline)
    float* fit[2] = {nullptr, nullptr};
    float* accum = nullptr;        // per-row trial-fitness accumulator (highdim)
    int* arrival = nullptr;        // per-row arrival counter (highdim)
    uint8_t* revert_mask = nullptr; // D4: per-row "child lost, restore parent" flag

    // nsums: per-row sums of the fitness (2 for Ackley); accum holds nRows * nsums floats.
    void init(size_t nRows, size_t d, size_t nsums = 1) {
        const size_t popBytes = nRows * elems(d) * sizeof(T);
        cudaMalloc(&pop[0], popBytes);
        cudaMalloc(&pop[1], popBytes);
        cudaMalloc(&trial, popBytes);
        cudaMalloc(&fit[0], nRows * sizeof(float));
        cudaMalloc(&fit[1], nRows * sizeof(float));
        cudaMalloc(&accum, nRows * nsums * sizeof(float));
        cudaMalloc(&arrival, nRows * sizeof(int));
        cudaMalloc(&revert_mask, nRows * sizeof(uint8_t));
        if (!pop[0] || !pop[1] || !trial || !fit[0] || !fit[1] || !accum || !arrival || !revert_mask) {
            std::fprintf(stderr, "FATAL: GenerationBuffer::init cudaMalloc failed\n");
            std::exit(1);
        }
        cudaMemset(accum, 0, nRows * nsums * sizeof(float));
        cudaMemset(arrival, 0, nRows * sizeof(int));
        cudaMemset(revert_mask, 0, nRows * sizeof(uint8_t));
    }

    void free_all() {
        cudaFree(pop[0]); cudaFree(pop[1]); cudaFree(trial);
        cudaFree(fit[0]); cudaFree(fit[1]);
        cudaFree(accum); cudaFree(arrival); cudaFree(revert_mask);
    }
};

} // v2
} // bench
} // evoopt
