#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include "common.ipp"
#include "operators.hpp"

// v2 fused variation + fitness kernels (scalar and vec4), island-modelled.
//
//   VEC4=false : one unit per thread / per loop step (float or uint32).
//   VEC4=true  : one 4-wide vector per thread / per loop step (float4 or uint4).
//
// Island-major layout: rowsPI = NUM_OFFSPRING * nCouples rows per island; the
// global row of child k of (island, local couple) is
//   island*rowsPI + k*nCouples + local.
//
// HIGHDIM (unified, U1): block = (xthreads, yblock), grid =
// (min(ceil(nUnits/xthreads), cap), ceil(nCouples/yblock), islands). Each
// threadIdx.y lane owns one couple and strides units with gridDim.x*blockDim.x.
//   * yblock=1            -> dimension-axis only (old highdim; cap controls it);
//   * cap=1, yblock>1     -> population-axis only (equivalent to `individual`);
//   * cap>1 and yblock>1  -> hybrid (genes across grid.x, couples across block.y).
// INDIVIDUAL: kept as the block-local reference (grid.x = ceil(nCouples/yblock)).
// Fitness is minimization-only; the kernel reduces FITNESS::cost[_vec4] into
// fitness[global_row], which is CALLER-OWNED and zeroed (highdim atomicAdds).
//
// Dynamic shared memory: xthreads * yblock * sizeof(float) bytes (highdim: yblock=1).

namespace evoopt {
namespace bench {
namespace v2 {

// UNIFIED (highdim): block = (tx, ty), grid = (gx, ceil(nCouples/ty), islands).
// gx = min(ceil(nUnits/tx), cap). Threads of lane threadIdx.y own one couple and
// stride the units with stride = gridDim.x*blockDim.x; genes are distributed over
// grid.x. cap==1,ty>1 reproduces `individual`; ty==1 reproduces the old highdim.
template <typename CROSSOVER, typename MUTATION, typename FITNESS, typename EnginePolicy, bool VEC4>
__global__ void FusedVariation_HighDim(
    const typename CROSSOVER::value_type* __restrict__ population,
    typename CROSSOVER::value_type* __restrict__ offspring,
    const typename CROSSOVER::value_type* __restrict__ lb,
    const typename CROSSOVER::value_type* __restrict__ ub,
    const size_t* __restrict__ parents,
    const size_t nCols,
    const size_t nCouples,
    const size_t islands,
    const unsigned long long gen_seed,
    const unsigned long long gen_offset,
    float* __restrict__ fitness,
    CROSSOVER crossover,
    MUTATION mutation
) {
    using T = typename CROSSOVER::value_type;
    using V = typename CROSSOVER::vec_type;
    constexpr size_t NP = CROSSOVER::NUM_PARENTS;
    constexpr size_t NO = CROSSOVER::NUM_OFFSPRING;
    constexpr bool binary = std::is_same_v<T, uint32_t>;
    static_assert(std::is_same_v<T, typename MUTATION::value_type>, "MUTATION::value_type must match");
    static_assert(std::is_same_v<T, typename FITNESS::value_type>, "FITNESS::value_type must match");
    extern __shared__ float s_mem[];

    const size_t local = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const bool active = (local < nCouples);
    const size_t island = blockIdx.z;
    const size_t rowsPI = NO * nCouples;
    const size_t gbase = island * rowsPI;
    const size_t gcouple = island * nCouples + local;
    const size_t nUnits = VEC4 ? (binary ? nCols / 128 : nCols / 4) : (binary ? nCols / 32 : nCols);
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    size_t idx[NP];
    #pragma unroll
    for (size_t k = 0; k < NP; ++k) idx[k] = active ? parents[NP * gcouple + k] : 0;

    constexpr unsigned long long CS = (unsigned long long)CROSSOVER::OP_STREAM;
    constexpr unsigned long long MS = (unsigned long long)MUTATION::OP_STREAM;
    auto couple_rng = EnginePolicy::make_couple(gen_seed, gen_offset, CS + gcouple);
    const typename CROSSOVER::CoupleState c_state = crossover.setup_couple(couple_rng, nCols);

    float cost[NO];
    #pragma unroll
    for (size_t k = 0; k < NO; ++k) cost[k] = 0.0f;

    if (active) {
        for (size_t unit = (size_t)blockIdx.x * blockDim.x + threadIdx.x; unit < nUnits; unit += stride) {
            auto gene_rng_c = EnginePolicy::make_vector(gen_seed, gen_offset, CS + gcouple * nUnits + unit);
            auto gene_rng_m = EnginePolicy::make_vector(gen_seed, gen_offset, MS + gcouple * nUnits + unit);
            if constexpr (VEC4) {
                const V* srcv = reinterpret_cast<const V*>(population);
                V* dstv = reinterpret_cast<V*>(offspring);
                V pop[NP];
                #pragma unroll
                for (size_t k = 0; k < NP; ++k) pop[k] = srcv[(gbase + idx[k]) * nUnits + unit];
                V lo, hi;
                if constexpr (binary) { lo = make_uint4(0, 0, 0, 0); hi = lo; }
                else {
                    const T* lp = lb + 4 * unit; const T* up = ub + 4 * unit;
                    lo = make_float4(lp[0], lp[1], lp[2], lp[3]);
                    hi = make_float4(up[0], up[1], up[2], up[3]);
                }
                V off[NO];
                crossover.apply_vec4(pop, off, lo, hi, unit, c_state, gene_rng_c);
                #pragma unroll
                for (size_t k = 0; k < NO; ++k) {
                    mutation.apply_vec4(off[k], lo, hi, unit, nCols, gene_rng_m);
                    dstv[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                    cost[k] += FITNESS::cost_vec4(off[k]);
                }
            } else {
                T pop[NP];
                #pragma unroll
                for (size_t k = 0; k < NP; ++k) pop[k] = population[(gbase + idx[k]) * nUnits + unit];
                T lo = T(0), hi = T(0);
                if constexpr (!binary) { lo = lb[unit]; hi = ub[unit]; }
                T off[NO];
                crossover.apply(pop, off, lo, hi, unit, c_state, gene_rng_c);
                #pragma unroll
                for (size_t k = 0; k < NO; ++k) {
                    mutation.apply(off[k], lo, hi, unit, nCols, gene_rng_m);
                    offspring[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                    cost[k] += FITNESS::cost(off[k]);
                }
            }
        }
    }

    #pragma unroll
    for (size_t k = 0; k < NO; ++k) {
        const float s = block_sum_y(cost[k], s_mem);
        if (active && threadIdx.x == 0) atomicAdd(&fitness[gbase + k * nCouples + local], s);
    }
}

// INDIVIDUAL: grid = (ceil(nCouples/yblock), islands); block = (xthreads, yblock).
template <typename CROSSOVER, typename MUTATION, typename FITNESS, typename EnginePolicy, bool VEC4>
__global__ void FusedVariation_Individual(
    const typename CROSSOVER::value_type* __restrict__ population,
    typename CROSSOVER::value_type* __restrict__ offspring,
    const typename CROSSOVER::value_type* __restrict__ lb,
    const typename CROSSOVER::value_type* __restrict__ ub,
    const size_t* __restrict__ parents,
    const size_t nCols,
    const size_t nCouples,
    const size_t islands,
    const unsigned long long gen_seed,
    const unsigned long long gen_offset,
    float* __restrict__ fitness,
    CROSSOVER crossover,
    MUTATION mutation
) {
    using T = typename CROSSOVER::value_type;
    using V = typename CROSSOVER::vec_type;
    constexpr size_t NP = CROSSOVER::NUM_PARENTS;
    constexpr size_t NO = CROSSOVER::NUM_OFFSPRING;
    constexpr bool binary = std::is_same_v<T, uint32_t>;
    static_assert(std::is_same_v<T, typename MUTATION::value_type>, "MUTATION::value_type must match");
    static_assert(std::is_same_v<T, typename FITNESS::value_type>, "FITNESS::value_type must match");
    extern __shared__ float s_mem[];

    const size_t local = (size_t)blockIdx.x * blockDim.y + threadIdx.y;
    const size_t island = blockIdx.y;
    const bool active = (local < nCouples);
    const size_t rowsPI = NO * nCouples;
    const size_t gbase = island * rowsPI;
    const size_t gcouple = island * nCouples + local;
    const size_t nUnits = VEC4 ? (binary ? nCols / 128 : nCols / 4) : (binary ? nCols / 32 : nCols);

    size_t idx[NP];
    #pragma unroll
    for (size_t k = 0; k < NP; ++k) idx[k] = active ? parents[NP * gcouple + k] : 0;

    constexpr unsigned long long CS = (unsigned long long)CROSSOVER::OP_STREAM;
    constexpr unsigned long long MS = (unsigned long long)MUTATION::OP_STREAM;
    auto couple_rng = EnginePolicy::make_couple(gen_seed, gen_offset, CS + gcouple);
    const typename CROSSOVER::CoupleState c_state = crossover.setup_couple(couple_rng, nCols);

    float acc[NO];
    #pragma unroll
    for (size_t k = 0; k < NO; ++k) acc[k] = 0.0f;

    if constexpr (VEC4) {
        const V* srcv = reinterpret_cast<const V*>(population);
        V* dstv = reinterpret_cast<V*>(offspring);
        for (size_t unit = threadIdx.x; unit < nUnits; unit += blockDim.x) {
            if (!active) break;
            auto gene_rng_c = EnginePolicy::make_vector(gen_seed, gen_offset, CS + gcouple * nUnits + unit);
            auto gene_rng_m = EnginePolicy::make_vector(gen_seed, gen_offset, MS + gcouple * nUnits + unit);
            V pop[NP];
            #pragma unroll
            for (size_t k = 0; k < NP; ++k) pop[k] = srcv[(gbase + idx[k]) * nUnits + unit];
            V lo, hi;
            if constexpr (binary) { lo = make_uint4(0, 0, 0, 0); hi = lo; }
            else {
                const T* lp = lb + 4 * unit; const T* up = ub + 4 * unit;
                lo = make_float4(lp[0], lp[1], lp[2], lp[3]);
                hi = make_float4(up[0], up[1], up[2], up[3]);
            }
            V off[NO];
            crossover.apply_vec4(pop, off, lo, hi, unit, c_state, gene_rng_c);
            #pragma unroll
            for (size_t k = 0; k < NO; ++k) {
                mutation.apply_vec4(off[k], lo, hi, unit, nCols, gene_rng_m);
                dstv[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                acc[k] += FITNESS::cost_vec4(off[k]);
            }
        }
    } else {
        for (size_t unit = threadIdx.x; unit < nUnits; unit += blockDim.x) {
            if (!active) break;
            auto gene_rng_c = EnginePolicy::make_vector(gen_seed, gen_offset, CS + gcouple * nUnits + unit);
            auto gene_rng_m = EnginePolicy::make_vector(gen_seed, gen_offset, MS + gcouple * nUnits + unit);
            T pop[NP];
            #pragma unroll
            for (size_t k = 0; k < NP; ++k) pop[k] = population[(gbase + idx[k]) * nUnits + unit];
            T lo = T(0), hi = T(0);
            if constexpr (!binary) { lo = lb[unit]; hi = ub[unit]; }
            T off[NO];
            crossover.apply(pop, off, lo, hi, unit, c_state, gene_rng_c);
            #pragma unroll
            for (size_t k = 0; k < NO; ++k) {
                mutation.apply(off[k], lo, hi, unit, nCols, gene_rng_m);
                offspring[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                acc[k] += FITNESS::cost(off[k]);
            }
        }
    }

    #pragma unroll
    for (size_t k = 0; k < NO; ++k) {
        const float s = block_sum_y(acc[k], s_mem);
        if (active && threadIdx.x == 0) fitness[gbase + k * nCouples + local] = s;
    }
}

} // v2
} // bench
} // evoopt
