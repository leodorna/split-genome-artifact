#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>
#include "common.ipp"
#include "operators.hpp"

// Donor probe (scripts/donor_probe.sh; off by default, which leaves the kernels unchanged):
// PROBE_CONST_BOUNDS reads the bounds of unit 0 for every unit, so the two bound arrays
// stop being read streams of the Sphere/OneMax loops.
#ifdef PROBE_CONST_BOUNDS
#define V2_BOUND_UNIT(u) ((size_t)0)
#else
#define V2_BOUND_UNIT(u) (u)
#endif

// v2 fused *generation* kernels: selection -> crossover -> mutation -> fitness
// -> replacement, one launch per generation, island-modelled.
//
//   SELECTION     picks CROSSOVER::NUM_PARENTS island-local parent rows per
//                 couple from fit_src (RandomDonors / TournamentPairMin /
//                 NoneSelection = external `parents`).
//   REPLACEMENT   keep-parent vs keep-child for the row the child is written to
//                 (island*rowsPI + k*nCouples + local); ties keep the parent.
//
// Island-major layout: rowsPI = NUM_OFFSPRING * nCouples rows per island; the
// global row of child k of (island, local couple) is island*rowsPI +
// k*nCouples + local. HighDim (unified, U1): grid = (min(ceil(nUnits/xthreads),
// cap), ceil(nCouples/yblock), islands), block = (xthreads, yblock); each
// threadIdx.y lane owns one couple. Individual grid = (nCouples, islands). The
// kernel writes pop_dst/fit_dst for every row (child, or the parent copied back
// on a loss).
//
// HighDim: when gridDim.x > 1 the last-arriving block of a row (per-row arrival
// counter) does the replacement and resets accum/arrival for the next
// generation; when gridDim.x == 1 finalize is block-local (no atomics, equal to
// individual). accum/arrival must be zeroed once before the first generation.
//
// Dynamic shared memory: launch with blockDim.x * blockDim.y * sizeof(float).

namespace evoopt {
namespace bench {
namespace v2 {

// HIGHDIM (unified, U1): block = (tx, ty), grid = (gx, ceil(nCouples/ty), islands),
// gx = min(ceil(nUnits/tx), cap), stride = gridDim.x*blockDim.x. Each threadIdx.y
// lane owns one couple. Cross-block reduction uses per-row accum/arrival; only the
// last-arriving block of a row finalizes it.
//   * gridDim.x == 1 -> no cross-block reduction: block-local finalize (== individual)
//   * gridDim.x >  1 -> atomicAdd into accum, arrival counter, then finalize
// Barriers are unconditional; finalize uses per-lane predicates with no interior
// barrier, so ty>1 lanes may diverge safely (no `continue` across a __syncthreads).
template <typename SELECTION, typename REPLACEMENT, typename CROSSOVER, typename MUTATION,
          typename FITNESS, typename EnginePolicy, bool VEC4>
__global__ void FusedGeneration_HighDim(
    const typename CROSSOVER::value_type* __restrict__ pop_src,
    typename CROSSOVER::value_type* __restrict__ pop_dst,
    const float* __restrict__ fit_src,
    float* __restrict__ fit_dst,
    float* __restrict__ accum,
    int* __restrict__ arrival,
    uint8_t* __restrict__ revert_mask,
    const int defer_restore,
    const typename CROSSOVER::value_type* __restrict__ lb,
    const typename CROSSOVER::value_type* __restrict__ ub,
    const size_t* __restrict__ parents,
    const size_t nCols,
    const size_t nCouples,
    const size_t islands,
    const unsigned long long gen_seed,
    const unsigned long long gen_offset,
    CROSSOVER crossover,
    MUTATION mutation
) {
    using T = typename CROSSOVER::value_type;
    using V = typename CROSSOVER::vec_type;
    constexpr size_t NP = CROSSOVER::NUM_PARENTS;
    constexpr size_t NO = CROSSOVER::NUM_OFFSPRING;
    constexpr bool binary = std::is_same_v<T, uint32_t>;
    static_assert(std::is_same_v<T, typename MUTATION::value_type>, "MUTATION::value_type mismatch");
    static_assert(std::is_same_v<T, typename FITNESS::value_type>, "FITNESS::value_type mismatch");
    static_assert(!SELECTION::has_selection || SELECTION::num_parents == (int)NP, "SELECTION parent count mismatch");
    extern __shared__ float s_mem[];
    __shared__ int s_last[NO * 32];

    const size_t local = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const bool active = (local < nCouples);
    const size_t island = blockIdx.z;
    const size_t rowsPI = NO * nCouples;
    const size_t gbase = island * rowsPI;
    const size_t gcouple = island * nCouples + local;
    const size_t nUnits = VEC4 ? (binary ? nCols / 128 : nCols / 4) : (binary ? nCols / 32 : nCols);
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    size_t idx[NP];
    if constexpr (SELECTION::has_selection) {
        auto sel_rng = EnginePolicy::make_couple(gen_seed, gen_offset,
                                                 (unsigned long long)RNG_STREAM::STREAM_SELECTION + gcouple);
        if (active) SELECTION::select(fit_src + gbase, rowsPI, local, idx, sel_rng);
        else { for (size_t k = 0; k < NP; ++k) idx[k] = 0; }
    } else {
        #pragma unroll
        for (size_t k = 0; k < NP; ++k) idx[k] = active ? parents[NP * gcouple + k] : 0;
    }

    constexpr unsigned long long CS = (unsigned long long)CROSSOVER::OP_STREAM;
    constexpr unsigned long long MS = (unsigned long long)MUTATION::OP_STREAM;
    auto couple_rng = EnginePolicy::make_couple(gen_seed, gen_offset, CS + gcouple);
    const typename CROSSOVER::CoupleState c_state = crossover.setup_couple(couple_rng, nCols);

    constexpr int NS = num_sums_v<FITNESS>;
    static_assert(!positional_v<FITNESS> || !sync_loop_v<FITNESS>, "positional fitness and the barrier probe");
    float cost[NO][NS];
    #pragma unroll
    for (size_t k = 0; k < NO; ++k) {
        #pragma unroll
        for (int m = 0; m < NS; ++m) cost[k][m] = 0.0f;
    }

    if constexpr (!needs_prev_v<FITNESS> && !sync_loop_v<FITNESS>) {
        if (active) {
            for (size_t unit = (size_t)blockIdx.x * blockDim.x + threadIdx.x; unit < nUnits; unit += stride) {
                auto gene_rng_c = EnginePolicy::make_vector(gen_seed, gen_offset, CS + gcouple * nUnits + unit);
                auto gene_rng_m = EnginePolicy::make_vector(gen_seed, gen_offset, MS + gcouple * nUnits + unit);
                if constexpr (VEC4) {
                    const V* srcv = reinterpret_cast<const V*>(pop_src);
                    V* dstv = reinterpret_cast<V*>(pop_dst);
                    V pop[NP];
                    #pragma unroll
                    for (size_t k = 0; k < NP; ++k) pop[k] = srcv[(gbase + idx[k]) * nUnits + unit];
                    V lo, hi;
                    if constexpr (binary) { lo = make_uint4(0, 0, 0, 0); hi = lo; }
                    else {
                        const T* lp = lb + 4 * V2_BOUND_UNIT(unit); const T* up = ub + 4 * V2_BOUND_UNIT(unit);
                        lo = make_float4(lp[0], lp[1], lp[2], lp[3]);
                        hi = make_float4(up[0], up[1], up[2], up[3]);
                    }
                    V off[NO];
                    crossover.apply_vec4(pop, off, lo, hi, unit, c_state, gene_rng_c);
                    #pragma unroll
                    for (size_t k = 0; k < NO; ++k) {
                        mutation.apply_vec4(off[k], lo, hi, unit, nCols, gene_rng_m);
                        dstv[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                        fit_add_vec4<FITNESS>(cost[k], off[k], unit);
                    }
                } else {
                    T pop[NP];
                    #pragma unroll
                    for (size_t k = 0; k < NP; ++k) pop[k] = pop_src[(gbase + idx[k]) * nUnits + unit];
                    T lo = T(0), hi = T(0);
                    if constexpr (!binary) { lo = lb[V2_BOUND_UNIT(unit)]; hi = ub[V2_BOUND_UNIT(unit)]; }
                    T off[NO];
                    crossover.apply(pop, off, lo, hi, unit, c_state, gene_rng_c);
                    #pragma unroll
                    for (size_t k = 0; k < NO; ++k) {
                        mutation.apply(off[k], lo, hi, unit, nCols, gene_rng_m);
                        pop_dst[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                        fit_add<FITNESS>(cost[k], off[k], unit);
                    }
                }
            }
        }
    } else if constexpr (sync_loop_v<FITNESS>) {
        // Barrier probe (experimental): the loop above with a block-uniform trip count
        // and one barrier per iteration, so that the warps of a block advance together.
        for (size_t base = (size_t)blockIdx.x * blockDim.x; base < nUnits; base += stride) {
            const size_t unit = base + threadIdx.x;
            if (active && unit < nUnits) {
                auto gene_rng_c = EnginePolicy::make_vector(gen_seed, gen_offset, CS + gcouple * nUnits + unit);
                auto gene_rng_m = EnginePolicy::make_vector(gen_seed, gen_offset, MS + gcouple * nUnits + unit);
                if constexpr (VEC4) {
                    const V* srcv = reinterpret_cast<const V*>(pop_src);
                    V* dstv = reinterpret_cast<V*>(pop_dst);
                    V pop[NP];
                    #pragma unroll
                    for (size_t k = 0; k < NP; ++k) pop[k] = srcv[(gbase + idx[k]) * nUnits + unit];
                    V lo, hi;
                    if constexpr (binary) { lo = make_uint4(0, 0, 0, 0); hi = lo; }
                    else {
                        const T* lp = lb + 4 * V2_BOUND_UNIT(unit); const T* up = ub + 4 * V2_BOUND_UNIT(unit);
                        lo = make_float4(lp[0], lp[1], lp[2], lp[3]);
                        hi = make_float4(up[0], up[1], up[2], up[3]);
                    }
                    V off[NO];
                    crossover.apply_vec4(pop, off, lo, hi, unit, c_state, gene_rng_c);
                    #pragma unroll
                    for (size_t k = 0; k < NO; ++k) {
                        mutation.apply_vec4(off[k], lo, hi, unit, nCols, gene_rng_m);
                        dstv[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                        fit_add_vec4<FITNESS>(cost[k], off[k], unit);
                    }
                } else {
                    T pop[NP];
                    #pragma unroll
                    for (size_t k = 0; k < NP; ++k) pop[k] = pop_src[(gbase + idx[k]) * nUnits + unit];
                    T lo = T(0), hi = T(0);
                    if constexpr (!binary) { lo = lb[V2_BOUND_UNIT(unit)]; hi = ub[V2_BOUND_UNIT(unit)]; }
                    T off[NO];
                    crossover.apply(pop, off, lo, hi, unit, c_state, gene_rng_c);
                    #pragma unroll
                    for (size_t k = 0; k < NO; ++k) {
                        mutation.apply(off[k], lo, hi, unit, nCols, gene_rng_m);
                        pop_dst[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                        fit_add<FITNESS>(cost[k], off[k], unit);
                    }
                }
            }
            __syncthreads();
        }
    } else {
        // Fitness that needs the gene before each unit (Rosenbrock). Every thread runs
        // every iteration, because that gene comes from the previous lane (warp
        // shuffle), the previous warp (shared memory), the same block's previous
        // iteration (unsplit rows), or, for the first unit of a split block, is
        // recomputed from the parents and the unit's streams, with no communication
        // between blocks.
        static_assert(!binary, "needs_prev fitness is defined for real-valued genomes");
        using U = std::conditional_t<VEC4, V, T>;
        __shared__ float s_warp_last[NO][32];   // last gene of each warp, per offspring
        __shared__ float s_carry[NO][32];       // last gene of the previous iteration, per couple lane
        const unsigned lane = threadIdx.x & 31u;
        const unsigned warp = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
        auto make_offspring = [&](size_t u, U (&off)[NO]) {
            auto gene_rng_c = EnginePolicy::make_vector(gen_seed, gen_offset, CS + gcouple * nUnits + u);
            auto gene_rng_m = EnginePolicy::make_vector(gen_seed, gen_offset, MS + gcouple * nUnits + u);
            U pop[NP], lo, hi;
            if constexpr (VEC4) {
                const V* srcv = reinterpret_cast<const V*>(pop_src);
                #pragma unroll
                for (size_t k = 0; k < NP; ++k) pop[k] = srcv[(gbase + idx[k]) * nUnits + u];
                const T* lp = lb + 4 * u; const T* up = ub + 4 * u;
                lo = make_float4(lp[0], lp[1], lp[2], lp[3]);
                hi = make_float4(up[0], up[1], up[2], up[3]);
                crossover.apply_vec4(pop, off, lo, hi, u, c_state, gene_rng_c);
                #pragma unroll
                for (size_t k = 0; k < NO; ++k) mutation.apply_vec4(off[k], lo, hi, u, nCols, gene_rng_m);
            } else {
                #pragma unroll
                for (size_t k = 0; k < NP; ++k) pop[k] = pop_src[(gbase + idx[k]) * nUnits + u];
                lo = lb[u]; hi = ub[u];
                crossover.apply(pop, off, lo, hi, u, c_state, gene_rng_c);
                #pragma unroll
                for (size_t k = 0; k < NO; ++k) mutation.apply(off[k], lo, hi, u, nCols, gene_rng_m);
            }
        };
        auto last_gene = [](const U& x) -> float { if constexpr (VEC4) return x.w; else return x; };

        for (size_t base = (size_t)blockIdx.x * blockDim.x; base < nUnits; base += stride) {
            const size_t unit = base + threadIdx.x;
            const bool valid = active && unit < nUnits;
            U off[NO];
            float last[NO], prev[NO];
            #pragma unroll
            for (size_t k = 0; k < NO; ++k) last[k] = 0.0f;
            if (valid) {
                make_offspring(unit, off);
                #pragma unroll
                for (size_t k = 0; k < NO; ++k) {
                    reinterpret_cast<U*>(pop_dst)[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                    last[k] = last_gene(off[k]);
                }
            }
            #pragma unroll
            for (size_t k = 0; k < NO; ++k) {
                prev[k] = __shfl_up_sync(0xffffffffu, last[k], 1);
                if (lane == 31) s_warp_last[k][warp] = last[k];
            }
            __syncthreads();
            if (valid && lane == 0 && unit > 0) {
                if (threadIdx.x > 0) {
                    #pragma unroll
                    for (size_t k = 0; k < NO; ++k) prev[k] = s_warp_last[k][warp - 1];
                } else if (gridDim.x == 1) {
                    #pragma unroll
                    for (size_t k = 0; k < NO; ++k) prev[k] = s_carry[k][threadIdx.y];
                } else {
                    U before[NO];
                    make_offspring(unit - 1, before);
                    #pragma unroll
                    for (size_t k = 0; k < NO; ++k) prev[k] = last_gene(before[k]);
                }
            }
            if (valid) {
                #pragma unroll
                for (size_t k = 0; k < NO; ++k)
                    cost[k][0] += unit > 0 ? fit_cost_prev<FITNESS>(prev[k], off[k], unit)
                                           : fit_cost_first<FITNESS>(off[k], unit);
            }
            __syncthreads();
            if (threadIdx.x == blockDim.x - 1) {
                #pragma unroll
                for (size_t k = 0; k < NO; ++k) s_carry[k][threadIdx.y] = last[k];
            }
        }
    }

    // gridDim.x == 1: single block per row -> block-local finalize (no atomics).
    if (gridDim.x == 1) {
        #pragma unroll
        for (size_t k = 0; k < NO; ++k) {
            float sums[NS];
            #pragma unroll
            for (int m = 0; m < NS; ++m) sums[m] = block_sum_y(cost[k][m], s_mem);
            const float s = fit_finish<FITNESS>(sums, nCols);
            if (!active) continue;
            const size_t row = gbase + k * nCouples + local;
            const bool take = REPLACEMENT::keep(s, fit_src[row]);
            if (threadIdx.x == 0) {
                fit_dst[row] = take ? s : fit_src[row];
                revert_mask[row] = (!take && defer_restore) ? 1 : 0;
            }
            if (!defer_restore && !take) {
                if constexpr (VEC4) {
                    const V* srcv = reinterpret_cast<const V*>(pop_src);
                    V* dstv = reinterpret_cast<V*>(pop_dst);
                    for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x)
                        dstv[row * nUnits + u] = srcv[row * nUnits + u];
                } else {
                    for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x)
                        pop_dst[row * nUnits + u] = pop_src[row * nUnits + u];
                }
            }
        }
        return;
    }

    #pragma unroll
    for (size_t k = 0; k < NO; ++k) {
        #pragma unroll
        for (int m = 0; m < NS; ++m) {
            const float s = block_sum_y(cost[k][m], s_mem);
            if (active && threadIdx.x == 0) atomicAdd(&accum[(gbase + k * nCouples + local) * NS + m], s);
        }
    }
    // Lane leader (threadIdx.x == 0) of the last-arriving block decides the row:
    // s_last = 0 (not last), 1 (offspring kept), 2 (parent restored). The leader
    // alone reads and resets accum/arrival, so no other thread of the block can
    // read accum after the reset (the earlier code read it in every thread and
    // reset it in thread 0 without a barrier, a race between warps).
    if (active && threadIdx.x == 0) {
        __threadfence();   // release: this block's offspring writes and partial sums
        #pragma unroll
        for (size_t k = 0; k < NO; ++k) {
            const size_t row = gbase + k * nCouples + local;
            s_last[k * blockDim.y + threadIdx.y] =
                (atomicAdd(&arrival[row], 1) == (int)gridDim.x - 1) ? 1 : 0;
        }
        __threadfence();   // acquire: the other blocks' writes to this row
        #pragma unroll
        for (size_t k = 0; k < NO; ++k) {
            if (!s_last[k * blockDim.y + threadIdx.y]) continue;
            const size_t row = gbase + k * nCouples + local;
            const float f = fit_finish<FITNESS>(&accum[row * NS], nCols);
            const bool take = REPLACEMENT::keep(f, fit_src[row]);
            fit_dst[row] = take ? f : fit_src[row];
            revert_mask[row] = (!take && defer_restore) ? 1 : 0;
            #pragma unroll
            for (int m = 0; m < NS; ++m) accum[row * NS + m] = 0.0f;
            arrival[row] = 0;
            s_last[k * blockDim.y + threadIdx.y] = take ? 1 : 2;
        }
    }
    __syncthreads();

    #pragma unroll
    for (size_t k = 0; k < NO; ++k) {
        if (!active) continue;
        if (s_last[k * blockDim.y + threadIdx.y] != 2) continue;
        const size_t row = gbase + k * nCouples + local;
        if (!defer_restore) {
            if constexpr (VEC4) {
                const V* srcv = reinterpret_cast<const V*>(pop_src);
                V* dstv = reinterpret_cast<V*>(pop_dst);
                for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x)
                    dstv[row * nUnits + u] = srcv[row * nUnits + u];
            } else {
                for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x)
                    pop_dst[row * nUnits + u] = pop_src[row * nUnits + u];
            }
        }
    }
}

template <typename SELECTION, typename REPLACEMENT, typename CROSSOVER, typename MUTATION,
          typename FITNESS, typename EnginePolicy, bool VEC4>
__global__ void FusedGeneration_Individual(
    const typename CROSSOVER::value_type* __restrict__ pop_src,
    typename CROSSOVER::value_type* __restrict__ pop_dst,
    const float* __restrict__ fit_src,
    float* __restrict__ fit_dst,
    float* __restrict__ accum,
    int* __restrict__ arrival,
    uint8_t* __restrict__ revert_mask,
    const int defer_restore,
    const typename CROSSOVER::value_type* __restrict__ lb,
    const typename CROSSOVER::value_type* __restrict__ ub,
    const size_t* __restrict__ parents,
    const size_t nCols,
    const size_t nCouples,
    const size_t islands,
    const unsigned long long gen_seed,
    const unsigned long long gen_offset,
    CROSSOVER crossover,
    MUTATION mutation
) {
    using T = typename CROSSOVER::value_type;
    using V = typename CROSSOVER::vec_type;
    constexpr size_t NP = CROSSOVER::NUM_PARENTS;
    constexpr size_t NO = CROSSOVER::NUM_OFFSPRING;
    constexpr bool binary = std::is_same_v<T, uint32_t>;
    static_assert(std::is_same_v<T, typename MUTATION::value_type>, "MUTATION::value_type mismatch");
    static_assert(std::is_same_v<T, typename FITNESS::value_type>, "FITNESS::value_type mismatch");
    static_assert(!SELECTION::has_selection || SELECTION::num_parents == (int)NP, "SELECTION parent count mismatch");
    (void)accum; (void)arrival;  // individual replacement is block-local
    extern __shared__ float s_mem[];

    const size_t local = (size_t)blockIdx.x * blockDim.y + threadIdx.y;
    const bool active = (local < nCouples);
    const size_t island = blockIdx.y;
    const size_t rowsPI = NO * nCouples;
    const size_t gbase = island * rowsPI;
    const size_t gcouple = island * nCouples + local;
    const size_t nUnits = VEC4 ? (binary ? nCols / 128 : nCols / 4) : (binary ? nCols / 32 : nCols);

    size_t idx[NP];
    if constexpr (SELECTION::has_selection) {
        auto sel_rng = EnginePolicy::make_couple(gen_seed, gen_offset,
                                                 (unsigned long long)RNG_STREAM::STREAM_SELECTION + gcouple);
        if (active) SELECTION::select(fit_src + gbase, rowsPI, local, idx, sel_rng);
        else { for (size_t k = 0; k < NP; ++k) idx[k] = 0; }
    } else {
        #pragma unroll
        for (size_t k = 0; k < NP; ++k) idx[k] = active ? parents[NP * gcouple + k] : 0;
    }

    constexpr unsigned long long CS = (unsigned long long)CROSSOVER::OP_STREAM;
    constexpr unsigned long long MS = (unsigned long long)MUTATION::OP_STREAM;
    auto couple_rng = EnginePolicy::make_couple(gen_seed, gen_offset, CS + gcouple);
    const typename CROSSOVER::CoupleState c_state = crossover.setup_couple(couple_rng, nCols);

    float acc[NO];
    #pragma unroll
    for (size_t k = 0; k < NO; ++k) acc[k] = 0.0f;

    if constexpr (VEC4) {
        const V* srcv = reinterpret_cast<const V*>(pop_src);
        V* dstv = reinterpret_cast<V*>(pop_dst);
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
                const T* lp = lb + 4 * V2_BOUND_UNIT(unit); const T* up = ub + 4 * V2_BOUND_UNIT(unit);
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
            for (size_t k = 0; k < NP; ++k) pop[k] = pop_src[(gbase + idx[k]) * nUnits + unit];
            T lo = T(0), hi = T(0);
            if constexpr (!binary) { lo = lb[V2_BOUND_UNIT(unit)]; hi = ub[V2_BOUND_UNIT(unit)]; }
            T off[NO];
            crossover.apply(pop, off, lo, hi, unit, c_state, gene_rng_c);
            #pragma unroll
            for (size_t k = 0; k < NO; ++k) {
                mutation.apply(off[k], lo, hi, unit, nCols, gene_rng_m);
                pop_dst[(gbase + k * nCouples + local) * nUnits + unit] = off[k];
                acc[k] += FITNESS::cost(off[k]);
            }
        }
    }

    #pragma unroll
    for (size_t k = 0; k < NO; ++k) {
        const float s = block_sum_y(acc[k], s_mem);
        if (!active) continue;
        const size_t row = gbase + k * nCouples + local;
        const bool take = REPLACEMENT::keep(s, fit_src[row]);
        if (!defer_restore && !take) {
            if constexpr (VEC4) {
                const V* srcv = reinterpret_cast<const V*>(pop_src);
                V* dstv = reinterpret_cast<V*>(pop_dst);
                for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x)
                    dstv[row * nUnits + u] = srcv[row * nUnits + u];
            } else {
                for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x)
                    pop_dst[row * nUnits + u] = pop_src[row * nUnits + u];
            }
        }
        if (threadIdx.x == 0) {
            fit_dst[row] = take ? s : fit_src[row];
            revert_mask[row] = (!take && defer_restore) ? 1 : 0;
        }
    }
}

// D4: restore the parent row for offspring that lost (revert_mask[row] != 0).
// 1D grid-stride over `nRows * blocksPerRow` (row, chunk) pairs, so it is not
// limited by grid.y <= 65535 and launches a bounded number of blocks. Winning
// rows are skipped immediately.
template <typename T>
__global__ void ParallelRestore(
    const T* __restrict__ pop_src,
    T* __restrict__ pop_dst,
    const uint8_t* __restrict__ revert_mask,
    const size_t nUnitsPerRow,
    const size_t nRows,
    const unsigned blocksPerRow
) {
    const size_t total = nRows * (size_t)blocksPerRow;
    const size_t stride = (size_t)blocksPerRow * blockDim.x;
    for (size_t idx = blockIdx.x; idx < total; idx += gridDim.x) {
        const size_t row = idx / blocksPerRow;
        if (!revert_mask[row]) continue;
        const size_t sub = idx - row * blocksPerRow;
        const T* src = pop_src + row * nUnitsPerRow;
        T* dst = pop_dst + row * nUnitsPerRow;
        for (size_t u = sub * blockDim.x + threadIdx.x; u < nUnitsPerRow; u += stride)
            dst[u] = src[u];
    }
}

// Row-wise fitness evaluation (initial population). Grid = (totalRows, 1, 1).
template <typename FITNESS, bool VEC4 = false>
__global__ void Evaluate(
    const typename FITNESS::value_type* __restrict__ pop,
    float* __restrict__ fit,
    const size_t nRows,
    const size_t nCols
) {
    using T = typename FITNESS::value_type;
    using V = std::conditional_t<std::is_same_v<T, uint32_t>, uint4, float4>;
    constexpr bool binary = std::is_same_v<T, uint32_t>;
    extern __shared__ float s_mem[];

    const size_t row = blockIdx.x;
    if (row >= nRows) return;
    const size_t nUnits = VEC4 ? (binary ? nCols / 128 : nCols / 4) : (binary ? nCols / 32 : nCols);

    constexpr int NS = num_sums_v<FITNESS>;
    float acc[NS];
    #pragma unroll
    for (int m = 0; m < NS; ++m) acc[m] = 0.0f;
    if constexpr (needs_prev_v<FITNESS>) {
        static_assert(!VEC4, "Evaluate: needs_prev fitness is evaluated gene by gene");
        for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x)
            acc[0] += u > 0 ? fit_cost_prev<FITNESS>(pop[row * nUnits + u - 1], pop[row * nUnits + u], u)
                            : fit_cost_first<FITNESS>(pop[row * nUnits + u], u);
    } else if constexpr (VEC4) {
        const V* pv = reinterpret_cast<const V*>(pop);
        for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x) fit_add_vec4<FITNESS>(acc, pv[row * nUnits + u], u);
    } else {
        for (size_t u = threadIdx.x; u < nUnits; u += blockDim.x) fit_add<FITNESS>(acc, pop[row * nUnits + u], u);
    }
    float sums[NS];
    #pragma unroll
    for (int m = 0; m < NS; ++m) sums[m] = block_sum(acc[m], s_mem);
    if (threadIdx.x == 0) fit[row] = fit_finish<FITNESS>(sums, nCols);
}

} // v2
} // bench
} // evoopt
