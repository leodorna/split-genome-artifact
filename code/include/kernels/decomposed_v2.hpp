#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>
#include "common.ipp"
#include "operators.hpp"
#include "kernels/fused_v2.hpp"

// Decomposed generation pipeline:
//   SelectionKernel  ->  FusedVariation_*  ->  ReplacementKernel
//
// Each stage owns its own launch configuration and is independent of the
// variation layout. Selection works on fitness/indices only (no genome), so the
// individual/highdim axis does not apply to it.

namespace evoopt {
namespace bench {
namespace v2 {

// One thread per couple; writes island-local parent indices into `parents`.
template <typename SELECTION, typename EnginePolicy>
__global__ void SelectionKernel(
    const float* __restrict__ fit_src,
    size_t* __restrict__ parents,
    const size_t nCouples,
    const size_t rowsPI,
    const size_t islands,
    const unsigned long long gen_seed,
    const unsigned long long gen_offset
) {
    static_assert(SELECTION::has_selection, "SelectionKernel requires a selecting policy");
    constexpr int NP = SELECTION::num_parents;
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= islands * nCouples) return;
    const size_t island = g / nCouples;
    const size_t local = g - island * nCouples;
    auto sel_rng = EnginePolicy::make_couple(gen_seed, gen_offset,
                                             (unsigned long long)RNG_STREAM::STREAM_SELECTION + g);
    size_t idx[NP];
    SELECTION::select(fit_src + island * rowsPI, rowsPI, local, idx, sel_rng);
    #pragma unroll
    for (int k = 0; k < NP; ++k) parents[(size_t)NP * g + k] = idx[k];
}

// Winner-select: pop_dst[row] = keep ? trial[row] : pop_src[row] (parallel copy),
// fit_dst[row] set once per row.
template <typename REPLACEMENT, typename T>
__global__ void ReplacementKernel(
    const T* __restrict__ pop_src,
    const T* __restrict__ trial,
    const float* __restrict__ accum,
    const float* __restrict__ fit_src,
    T* __restrict__ pop_dst,
    float* __restrict__ fit_dst,
    const size_t nUnitsPerRow,
    const size_t nRows,
    const unsigned blocksPerRow
) {
    const size_t total = nRows * (size_t)blocksPerRow;
    const size_t stride = (size_t)blocksPerRow * blockDim.x;
    for (size_t idx = blockIdx.x; idx < total; idx += gridDim.x) {
        const size_t row = idx / blocksPerRow;
        const size_t sub = idx - row * blocksPerRow;
        const bool take = REPLACEMENT::keep(accum[row], fit_src[row]);
        const T* src = take ? trial : pop_src;
        const size_t base = row * nUnitsPerRow;
        for (size_t u = sub * blockDim.x + threadIdx.x; u < nUnitsPerRow; u += stride)
            pop_dst[base + u] = src[base + u];
        if (sub == 0 && threadIdx.x == 0)
            fit_dst[row] = take ? accum[row] : fit_src[row];
    }
}

// Blocks-per-row for the replacement copy, and the (grid, blocksPerRow) to use.
inline unsigned v2_bpr(size_t nRows, unsigned cap = 256) {
    return (unsigned)std::min<size_t>(cap, std::max<size_t>(1, (size_t)65536 / std::max<size_t>(1, nRows)));
}
inline unsigned v2_replace_blocks(size_t nRows, unsigned bpr) {
    const size_t total = nRows * (size_t)bpr;
    return (unsigned)std::min<size_t>(total, (size_t)(1u << 13));
}

// One decomposed generation on `stream`: select -> fused variation -> replace.
template <typename SELECTION, typename REPLACEMENT, typename CROSSOVER, typename MUTATION,
          typename FITNESS, typename EnginePolicy, bool VEC4, bool HIGHDIM>
inline void generationStepDecomposed(
    const typename CROSSOVER::value_type* pop_src,
    typename CROSSOVER::value_type* pop_dst,
    typename CROSSOVER::value_type* trial,
    const float* fit_src,
    float* fit_dst,
    float* accum,
    size_t* parents,
    const typename CROSSOVER::value_type* lb,
    const typename CROSSOVER::value_type* ub,
    const size_t nCols,
    const size_t nCouples,
    const size_t islands,
    const unsigned long long gen_seed,
    const unsigned long long gen_offset,
    const int tx,
    const int ty,
    cudaStream_t stream
) {
    using T = typename CROSSOVER::value_type;
    constexpr bool binary = std::is_same_v<T, uint32_t>;
    const size_t rowsPI = CROSSOVER::NUM_OFFSPRING * nCouples;
    const size_t nRows = islands * rowsPI;
    const size_t nUnits = VEC4 ? (binary ? nCols / 128 : nCols / 4) : (binary ? nCols / 32 : nCols);
    const size_t elemsT = binary ? nCols / 32 : nCols;
    const unsigned tyb = (unsigned)std::max(1, ty);
    const size_t smem = (size_t)tx * tyb * sizeof(float);

    // 1. selection (own launch config: one thread per couple)
    if constexpr (SELECTION::has_selection) {
        const size_t total = islands * nCouples;
        const unsigned blocks = (unsigned)((total + 255) / 256);
        SelectionKernel<SELECTION, EnginePolicy><<<blocks, 256, 0, stream>>>(
            fit_src, parents, nCouples, rowsPI, islands, gen_seed, gen_offset);
    }

    // 2. fused variation + fitness (writes trial genes and trial fitness)
    cudaMemsetAsync(accum, 0, nRows * sizeof(float), stream);
    if constexpr (HIGHDIM) {
        dim3 grid(v2_hd_grid_x(nUnits, tx), (unsigned)((nCouples + tyb - 1) / tyb), (unsigned)islands);
        dim3 block((unsigned)tx, tyb);
        FusedVariation_HighDim<CROSSOVER, MUTATION, FITNESS, EnginePolicy, VEC4><<<grid, block, smem, stream>>>(
            pop_src, trial, lb, ub, parents, nCols, nCouples, islands,
            gen_seed, gen_offset, accum, CROSSOVER{}, MUTATION{});
    } else {
        dim3 grid((unsigned)((nCouples + tyb - 1) / tyb), (unsigned)islands);
        dim3 block((unsigned)tx, tyb);
        FusedVariation_Individual<CROSSOVER, MUTATION, FITNESS, EnginePolicy, VEC4><<<grid, block, smem, stream>>>(
            pop_src, trial, lb, ub, parents, nCols, nCouples, islands,
            gen_seed, gen_offset, accum, CROSSOVER{}, MUTATION{});
    }

    // 3. replacement (winner-select, parallel copy)
    const unsigned bpr = v2_bpr(nRows);
    const unsigned rgx = v2_replace_blocks(nRows, bpr);
    ReplacementKernel<REPLACEMENT, T><<<rgx == 0 ? 1u : rgx, 256, 0, stream>>>(
        pop_src, trial, accum, fit_src, pop_dst, fit_dst, elemsT, nRows, bpr);
}

} // v2
} // bench
} // evoopt
