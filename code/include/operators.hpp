#pragma once

#include <algorithm>
#include <type_traits>

#include <cuda_runtime.h>
#include "common.ipp"
#include "operators_aux.hpp"

// v2 operator policies (EvoX-faithful).
//
// A policy exposes:
//   value_type, vec_type              gene type (float/uint32) and its 4-wide vector
//   NUM_PARENTS, NUM_OFFSPRING        compile-time sizes (crossover)
//   OP_STREAM                         stream id so operators never share draws
//   setup_couple / CoupleState        couple phase (crossover only)
//   apply / apply_vec4                gene/word phase, in the operator's stream
//
// Hyperparameters are runtime instance fields; only the members that size arrays
// or select streams are constexpr. Bounds are per-variable and handed to the
// operator one unit at a time (scalar for apply, 4-wide for apply_vec4).
//
// Real-valued operations follow EvoX exactly (de.py, sbx.py, pm_mutation.py).
// Binary operations follow the standard word-mask formulation.

namespace evoopt {
namespace bench {
namespace v2 {

// D1: bound the number of HighDim blocks per row so the per-row fitness
// `atomicAdd` count is O(1) in D. Each block strides over its share of the
// units, so grid.x = min(ceil(nUnits/xthreads), V2_HD_MAX_BLOCKS_PER_ROW).
inline constexpr unsigned V2_HD_MAX_BLOCKS_PER_ROW = 256;

inline unsigned v2_hd_grid_x(size_t nUnits, int xthreads,
                             unsigned cap = V2_HD_MAX_BLOCKS_PER_ROW) {
    const size_t gx = (nUnits + (size_t)xthreads - 1) / (size_t)xthreads;
    return (unsigned)std::min<size_t>(gx == 0 ? 1 : gx, cap == 0 ? 1u : cap);
}

enum class RNG_STREAM : unsigned long long {
    STREAM_SELECTION   = 0x100000000ULL,
    STREAM_CROSSOVER   = 0x200000000ULL,
    STREAM_MUTATION    = 0x300000000ULL,
    STREAM_REPLACEMENT = 0x400000000ULL
};

// ---------------------------------------------------------------------------
// DE/rand/1/bin. Parents = {target, base, diff1, diff2}; one trial offspring.
// ---------------------------------------------------------------------------
struct DE1bin {
    using value_type = float;
    using vec_type = float4;
    static constexpr size_t NUM_PARENTS = 4;
    static constexpr size_t NUM_OFFSPRING = 1;
    static constexpr RNG_STREAM OP_STREAM = RNG_STREAM::STREAM_CROSSOVER;

    float CR = 0.9f;
    float F = 0.5f;

    struct CoupleState { size_t j_star; };  // guaranteed crossover dimension

    template <typename RandomEngine>
    __device__ CoupleState setup_couple(RandomEngine& couple_rng, size_t nCols) const {
        quad4 q;
        size_t j = static_cast<size_t>(draw_uniform(couple_rng, q) * static_cast<float>(nCols));
        if (j >= nCols) j = nCols - 1;
        return CoupleState{j};
    }

    template <typename RandomEngine>
    __device__ void apply(
        const float (&parents)[NUM_PARENTS], float (&offspring)[NUM_OFFSPRING],
        float lower, float upper, size_t unit,
        const CoupleState& state, RandomEngine& gene_rng
    ) const {
        uquad4 uq;
        const bool do_cx = (draw_uint32(gene_rng, uq) < prob_to_threshold(CR)) || (unit == state.j_star);
        offspring[0] = de_evox_gene(parents[0], parents[1], parents[2], parents[3],
                                    F, do_cx, lower, upper);
    }

    template <typename RandomEngine>
    __device__ void apply_vec4(
        const float4 (&parents)[NUM_PARENTS], float4 (&offspring)[NUM_OFFSPRING],
        float4 lower, float4 upper, size_t unit4,
        const CoupleState& state, RandomEngine& gene_rng
    ) const {
        const uint4 cr = gene_rng.next_uint4();
        const uint32_t thr = prob_to_threshold(CR);
        const b4 do_cx = make_b4((cr.x < thr) || (4 * unit4 + 0 == state.j_star),
                                 (cr.y < thr) || (4 * unit4 + 1 == state.j_star),
                                 (cr.z < thr) || (4 * unit4 + 2 == state.j_star),
                                 (cr.w < thr) || (4 * unit4 + 3 == state.j_star));
        offspring[0] = de_evox4(parents[0], parents[1], parents[2], parents[3], F, do_cx, lower, upper);
    }
};

// ---------------------------------------------------------------------------
// EvoX SBX. Parents = {p1, p2}; two children (mid +/- beta*h). No bounds clamp.
// ---------------------------------------------------------------------------
struct SBXCrossover {
    using value_type = float;
    using vec_type = float4;
    static constexpr size_t NUM_PARENTS = 2;
    static constexpr size_t NUM_OFFSPRING = 2;
    static constexpr RNG_STREAM OP_STREAM = RNG_STREAM::STREAM_CROSSOVER;

    float PRO_C = 1.0f;    // EvoX pro_c (per-gene no-spread gate)
    float DIS_C = 20.0f;   // distribution index

    struct CoupleState { };  // EvoX SBX has no couple-level draw

    template <typename RandomEngine>
    __device__ CoupleState setup_couple(RandomEngine& /*couple_rng*/, size_t /*nCols*/) const {
        return CoupleState{};
    }

    template <typename RandomEngine>
    __device__ void apply(
        const float (&parents)[NUM_PARENTS], float (&offspring)[NUM_OFFSPRING],
        float /*lower*/, float /*upper*/, size_t /*unit*/,
        const CoupleState& /*state*/, RandomEngine& gene_rng
    ) const {
        quad4 q; uquad4 uq;
        const float mu = draw_uniform(gene_rng, q);
        const uint32_t s = draw_uint32(gene_rng, uq);
        const float rh = draw_uniform(gene_rng, q);
        const float rp = draw_uniform(gene_rng, q);
        offspring[0] = sbx_evox_from(mu, s, rh, rp, parents[0], parents[1], PRO_C, DIS_C, true);
        offspring[1] = sbx_evox_from(mu, s, rh, rp, parents[0], parents[1], PRO_C, DIS_C, false);
    }

    template <typename RandomEngine>
    __device__ void apply_vec4(
        const float4 (&parents)[NUM_PARENTS], float4 (&offspring)[NUM_OFFSPRING],
        float4 /*lower*/, float4 /*upper*/, size_t /*unit4*/,
        const CoupleState& /*state*/, RandomEngine& gene_rng
    ) const {
        const float4 mu = gene_rng.next_uniform4();
        const uint4 sign = gene_rng.next_uint4();
        const float4 rh = gene_rng.next_uniform4();
        const float4 rp = gene_rng.next_uniform4();
        sbx_evox4(mu, sign, rh, rp, parents[0], parents[1], PRO_C, DIS_C, offspring[0], offspring[1]);
    }
};

// ---------------------------------------------------------------------------
// EvoX polynomial mutation (in place on one gene).
// ---------------------------------------------------------------------------
struct PolynomialMutation {
    using value_type = float;
    using vec_type = float4;
    static constexpr RNG_STREAM OP_STREAM = RNG_STREAM::STREAM_MUTATION;

    float PRO_M = 1.0f;    // numerator; per-gene site probability = PRO_M / D
    float DIS_M = 20.0f;

    template <typename RandomEngine>
    __device__ void apply(
        float& x, float lower, float upper, size_t /*unit*/, size_t nCols,
        RandomEngine& gene_rng
    ) const {
        quad4 q;
        const float site = draw_uniform(gene_rng, q);
        const float mu = draw_uniform(gene_rng, q);
        x = pm_evox_from(x, site, mu, lower, upper, PRO_M, DIS_M, nCols);
    }

    template <typename RandomEngine>
    __device__ void apply_vec4(
        float4& x, float4 lower, float4 upper, size_t /*unit4*/, size_t nCols,
        RandomEngine& gene_rng
    ) const {
        const float4 site = gene_rng.next_uniform4();
        const float4 mu = gene_rng.next_uniform4();
        x = pm_evox4(x, site, mu, lower, upper, PRO_M, DIS_M, nCols);
    }
};

// ---------------------------------------------------------------------------
// Binary uniform crossover. Parents = {p1, p2}; two children. pc is a
// couple-level gate; one uint32 mask per 32 genes (uint4 per 128 in vec4).
// ---------------------------------------------------------------------------
struct UX {
    using value_type = uint32_t;
    using vec_type = uint4;
    static constexpr size_t NUM_PARENTS = 2;
    static constexpr size_t NUM_OFFSPRING = 2;
    static constexpr RNG_STREAM OP_STREAM = RNG_STREAM::STREAM_CROSSOVER;

    float PC = 1.0f;

    struct CoupleState { bool is_active; };

    template <typename RandomEngine>
    __device__ CoupleState setup_couple(RandomEngine& couple_rng, size_t /*nCols*/) const {
        uquad4 uq;
        return CoupleState{draw_uint32(couple_rng, uq) < prob_to_threshold(PC)};
    }

    template <typename RandomEngine>
    __device__ void apply(
        const uint32_t (&parents)[NUM_PARENTS], uint32_t (&offspring)[NUM_OFFSPRING],
        uint32_t /*lower*/, uint32_t /*upper*/, size_t /*unit*/,
        const CoupleState& state, RandomEngine& gene_rng
    ) const {
        if (!state.is_active) {
            offspring[0] = parents[0];
            offspring[1] = parents[1];
            return;
        }
        uquad4 uq;
        const uint32_t mask = draw_uint32(gene_rng, uq);
        offspring[0] = ux_apply(parents[0], parents[1], mask, true);
        offspring[1] = ux_apply(parents[0], parents[1], mask, false);
    }

    template <typename RandomEngine>
    __device__ void apply_vec4(
        const uint4 (&parents)[NUM_PARENTS], uint4 (&offspring)[NUM_OFFSPRING],
        uint4 /*lower*/, uint4 /*upper*/, size_t /*unit4*/,
        const CoupleState& state, RandomEngine& gene_rng
    ) const {
        if (!state.is_active) {
            offspring[0] = parents[0];
            offspring[1] = parents[1];
            return;
        }
        const uint4 mask = gene_rng.next_uint4();
        offspring[0] = ux_apply4(parents[0], parents[1], mask, true);
        offspring[1] = ux_apply4(parents[0], parents[1], mask, false);
    }
};

// ---------------------------------------------------------------------------
// Bitflip mutation (in place on one packed word).
// ---------------------------------------------------------------------------
struct BF {
    using value_type = uint32_t;
    using vec_type = uint4;
    static constexpr RNG_STREAM OP_STREAM = RNG_STREAM::STREAM_MUTATION;

    float PROB = 0.01f;  // per-bit flip probability

    template <typename RandomEngine>
    __device__ void apply(
        uint32_t& word, uint32_t /*lower*/, uint32_t /*upper*/, size_t /*unit*/, size_t /*nCols*/,
        RandomEngine& gene_rng
    ) const {
        word ^= bf_mask(gene_rng, prob_to_threshold(PROB));
    }

    template <typename RandomEngine>
    __device__ void apply_vec4(
        uint4& word, uint4 /*lower*/, uint4 /*upper*/, size_t /*unit4*/, size_t /*nCols*/,
        RandomEngine& gene_rng
    ) const {
        const uint4 mask = bf_mask4(gene_rng, prob_to_threshold(PROB));
        word = make_uint4(word.x ^ mask.x, word.y ^ mask.y, word.z ^ mask.z, word.w ^ mask.w);
    }
};

// No-op mutation (for DE, whose operator already produces the trial).
template <typename T>
struct NoMutationT {
    using value_type = T;
    using vec_type = std::conditional_t<std::is_same_v<T, uint32_t>, uint4, float4>;
    static constexpr RNG_STREAM OP_STREAM = RNG_STREAM::STREAM_MUTATION;

    template <typename RandomEngine>
    __device__ void apply(T&, T, T, size_t, size_t, RandomEngine&) const {}

    template <typename RandomEngine>
    __device__ void apply_vec4(vec_type&, vec_type, vec_type, size_t, size_t, RandomEngine&) const {}
};
using NoMutation = NoMutationT<float>;

// ---------------------------------------------------------------------------
// Selection policies. Produce CROSSOVER::NUM_PARENTS parent indices for a
// couple, reading fit_src and drawing from STREAM_SELECTION.
// ---------------------------------------------------------------------------

// DE: idx[0] = target (the couple's own slot), idx[1..] = uniform donors with
// replacement (EvoX de.py draws each donor independently, replacement allowed).
template <int NP>
struct RandomDonors {
    static constexpr bool has_selection = true;
    static constexpr int num_parents = NP;
    template <typename RandomEngine>
    __device__ static void select(const float* /*fit*/, size_t n, size_t couple_id,
                                  size_t (&idx)[NP], RandomEngine& rng) {
        idx[0] = couple_id;
        quad4 q;
        #pragma unroll
        for (int k = 1; k < NP; ++k) {
            size_t r = static_cast<size_t>(draw_uniform(rng, q) * static_cast<float>(n));
            if (r >= n) r = n - 1;
            idx[k] = r;
        }
#ifdef PROBE_ONE_DONOR
        // Donor probe (scripts/donor_probe.sh): the donors share one row, so an offspring
        // reads two genomes instead of four. This changes the algorithm (the trial is
        // the first donor where it crosses), so it is a timing probe only.
        #pragma unroll
        for (int k = 2; k < NP; ++k) idx[k] = idx[1];
#endif
    }
};

// GA / ux-bf: two independent binary tournaments (min) over the population.
// The SBX pair semantics (c, c+n/2) are realized by the crossover's two
// children, not by the selection. Ties pick the first drawn.
struct TournamentPairMin {
    static constexpr bool has_selection = true;
    static constexpr int num_parents = 2;

    template <typename RandomEngine>
    __device__ static size_t winner(const float* fit, size_t n, RandomEngine& rng) {
        quad4 q;
        size_t a = static_cast<size_t>(draw_uniform(rng, q) * static_cast<float>(n));
        size_t b = static_cast<size_t>(draw_uniform(rng, q) * static_cast<float>(n));
        if (a >= n) a = n - 1;
        if (b >= n) b = n - 1;
        return (fit[a] <= fit[b]) ? a : b;
    }

    template <typename RandomEngine>
    __device__ static void select(const float* fit, size_t n, size_t couple_id,
                                  size_t (&idx)[2], RandomEngine& rng) {
        (void)couple_id;
        idx[0] = winner(fit, n, rng);
        idx[1] = winner(fit, n, rng);
    }
};

// Operator-only mode: parents come from an external array.
struct NoneSelection {
    static constexpr bool has_selection = false;
    static constexpr int num_parents = 0;
};

// ---------------------------------------------------------------------------
// Replacement policies. The child always competes against the row it is written
// to (k*nCouples + couple_id); ties keep the parent.
// ---------------------------------------------------------------------------
struct GreedyMin {
    __device__ static bool keep(float child_fit, float parent_fit) { return child_fit < parent_fit; }
};

// ---------------------------------------------------------------------------
// Fitness policies (minimization only; per-unit cost reduced by the kernel).
// ---------------------------------------------------------------------------
struct Sphere {
    using value_type = float;
    __device__ static float cost(float g) { return g * g; }
    __device__ static float cost_vec4(float4 g) { return g.x*g.x + g.y*g.y + g.z*g.z + g.w*g.w; }
};

// OneMax over packed uint32 words: per-word cost is 32 - popcount, so the row
// total is D - ones (minimizing maximizes the number of set bits).
struct OneMax {
    using value_type = uint32_t;
    __device__ static float cost(uint32_t w) { return (float)(32 - __popc(w)); }
    __device__ static float cost_vec4(uint4 w) {
        return (float)(128 - (__popc(w.x) + __popc(w.y) + __popc(w.z) + __popc(w.w)));
    }
};

// Rastrigin, 10 D + sum_i (x_i^2 - 10 cos(2 pi x_i)), as evox.problems.numerical.Rastrigin.
// Separable, so it uses the same per-unit path as Sphere; cosf is the accurate cosine.
struct Rastrigin {
    using value_type = float;
    __host__ __device__ static float cost(float g) {
        return g * g - 10.0f * cosf(2.0f * 3.14159265358979f * g) + 10.0f;
    }
    __device__ static float cost_vec4(float4 g) { return cost(g.x) + cost(g.y) + cost(g.z) + cost(g.w); }
};

// Rosenbrock, sum_{i<D-1} 100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2, as
// evox.problems.numerical.Rosenbrock. Not separable: the term of the pair (i-1, i)
// needs the gene before a unit (needs_prev), which the kernel hands to cost_prev;
// the first unit of a genome has no gene before it (cost_first).
struct Rosenbrock {
    using value_type = float;
    static constexpr bool needs_prev = true;
    __host__ __device__ static float pair(float x, float xn) {
        const float t = xn - x * x, u = x - 1.0f;
        return 100.0f * t * t + u * u;
    }
    __device__ static float cost_prev(float prev, float g) { return pair(prev, g); }
    __device__ static float cost_prev(float prev, float4 g) {
        return pair(prev, g.x) + pair(g.x, g.y) + pair(g.y, g.z) + pair(g.z, g.w);
    }
    __device__ static float cost_first(float) { return 0.0f; }
    __device__ static float cost_first(float4 g) { return pair(g.x, g.y) + pair(g.y, g.z) + pair(g.z, g.w); }
};

// True for fitness policies whose per-unit cost needs the gene before the unit.
template <typename F>
constexpr bool needs_prev_of() {
    if constexpr (requires { F::needs_prev; }) return F::needs_prev;
    else return false;
}
template <typename F>
inline constexpr bool needs_prev_v = needs_prev_of<F>();

// Barrier probe (experimental): Sphere through block-synchronous loops. SphereSync keeps
// Sphere's loop and adds one barrier per iteration; SpherePrev runs Sphere through the
// neighbor-exchange loop of Rosenbrock, with the previous gene fetched but unused.
struct SphereSync : Sphere { static constexpr bool sync_loop = true; };
struct SpherePrev : Sphere {
    static constexpr bool needs_prev = true;
    __device__ static float cost_prev(float, float g) { return cost(g); }
    __device__ static float cost_prev(float, float4 g) { return cost_vec4(g); }
    __device__ static float cost_first(float g) { return cost(g); }
    __device__ static float cost_first(float4 g) { return cost_vec4(g); }
};
template <typename F>
constexpr bool sync_loop_of() {
    if constexpr (requires { F::sync_loop; }) return F::sync_loop;
    else return false;
}
template <typename F>
inline constexpr bool sync_loop_v = sync_loop_of<F>();

// ---------------------------------------------------------------------------
// Position-dependent fitness and fitness with several sums.
//   positional  the cost of gene j depends on j (shift o_j, elliptic weight); the
//               policy provides add_at(acc, g, j) and add_vec4_at(acc, g4, unit)
//   num_sums    number of per-row sums the kernel reduces (default 1); the
//               fitness is finish(sums, D) (default: the single sum)
// Functions without these traits keep the original path: acc[0] += cost(g).
// ---------------------------------------------------------------------------
template <typename F>
constexpr bool positional_of() {
    if constexpr (requires { F::positional; }) return F::positional;
    else return false;
}
template <typename F>
inline constexpr bool positional_v = positional_of<F>();

template <typename F>
constexpr int num_sums_of() {
    if constexpr (requires { F::num_sums; }) return F::num_sums;
    else return 1;
}
template <typename F>
inline constexpr int num_sums_v = num_sums_of<F>();

template <typename F, typename T>
__device__ __forceinline__ void fit_add(float* acc, T g, size_t unit) {
    if constexpr (positional_v<F>) F::add_at(acc, g, unit);
    else acc[0] += F::cost(g);
}
template <typename F, typename V>
__device__ __forceinline__ void fit_add_vec4(float* acc, V g, size_t unit) {
    if constexpr (positional_v<F>) F::add_vec4_at(acc, g, unit);
    else acc[0] += F::cost_vec4(g);
}
// Neighbor-coupled terms (needs_prev): positional policies receive the unit index.
template <typename F, typename T>
__device__ __forceinline__ float fit_cost_prev(float prev, T g, size_t unit) {
    if constexpr (positional_v<F>) return F::cost_prev_at(prev, g, unit);
    else return F::cost_prev(prev, g);
}
template <typename F, typename T>
__device__ __forceinline__ float fit_cost_first(T g, size_t unit) {
    if constexpr (positional_v<F>) return F::cost_first_at(g, unit);
    else return F::cost_first(g);
}
template <typename F>
__host__ __device__ __forceinline__ float fit_finish(const float* s, size_t d) {
    if constexpr (requires(const float* p, size_t n) { F::finish(p, n); }) return F::finish(s, d);
    else return s[0];
}

// Parameters of the positional functions, set by the host before the first launch:
// the shift vector o (device memory, D floats, 16-byte aligned) and 1 / (D - 1), the
// step of the relative gene position t_j = j / (D - 1) used by the transformations.
struct FitParams {
    const float* shift;
    float inv_dm1;
    const uint32_t* kp_items;   // knapsack: item j packed as (weight << 16) | profit
    float kp_capacity;          // knapsack: capacity c
    float kp_rho;               // knapsack: penalty factor, max_j profit_j / weight_j
};
__constant__ FitParams c_fit;

// ---------------------------------------------------------------------------
// CEC 2013 LSGO f1, f2 and f3 (Li, Tang, Omidvar, Yang, Qin, 2013), as in the
// organizers' C++ code (Benchmarks.cpp): z = x - o, then element-wise
//   T_osz(z)     = sign(z) exp(h + 0.049 (sin(c1 h) + sin(c2 h))), h = log|z| (0 if z = 0),
//                  c1 = 10, c2 = 7.9 if z > 0, else c1 = 5.5, c2 = 3.1
//   T_asy^b(z)_j = z_j^(1 + b t_j sqrt(z_j)) if z_j > 0, else z_j
//   Lambda^a(z)_j = a^(t_j / 2) z_j,  t_j = j / (D - 1), j = 0..D-1
// f1 = sum_j 10^(6 t_j) T_osz(z)_j^2                                  on [-100, 100]^D
// f2 = sum_j (y_j^2 - 10 cos(2 pi y_j) + 10),  y = Lambda^10 T_asy^0.2 T_osz(z)   on [-5, 5]^D
// f3 = -20 exp(-0.2 sqrt(mean y^2)) - exp(mean cos(2 pi y)) + 20 + e,  y as f2   on [-32, 32]^D
// All three are fully separable and unrotated in the suite; their only fixed-size data is
// the shift (D = 1000 in the suite; generated for other D).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float cec_t(size_t j) { return (float)j * c_fit.inv_dm1; }

__device__ __forceinline__ float cec_osz(float z) {
    if (z == 0.0f) return 0.0f;
    const float h = logf(fabsf(z));
    const float c1 = z > 0.0f ? 10.0f : 5.5f;
    const float c2 = z > 0.0f ? 7.9f : 3.1f;
    return copysignf(expf(h + 0.049f * (sinf(c1 * h) + sinf(c2 * h))), z);
}

// Lambda^10 T_asy^0.2 T_osz of one shifted gene z at relative position t
__device__ __forceinline__ float cec_y(float z, float t) {
    float y = cec_osz(z);
    if (y > 0.0f) y = powf(y, 1.0f + 0.2f * t * sqrtf(y));
    return y * exp2f(0.5f * 3.32192809488736f * t);   // 10^(t/2)
}

template <typename Derived>
struct CecSeparable {
    using value_type = float;
    static constexpr bool positional = true;
    __device__ static void add_at(float* acc, float g, size_t j) {
        Derived::add_term(acc, g - c_fit.shift[j], cec_t(j));
    }
    __device__ static void add_vec4_at(float* acc, float4 g, size_t u) {
        const float4 o = reinterpret_cast<const float4*>(c_fit.shift)[u];
        const size_t j = 4 * u;
        Derived::add_term(acc, g.x - o.x, cec_t(j));
        Derived::add_term(acc, g.y - o.y, cec_t(j + 1));
        Derived::add_term(acc, g.z - o.z, cec_t(j + 2));
        Derived::add_term(acc, g.w - o.w, cec_t(j + 3));
    }
};

// f1, shifted elliptic. The weight 10^(6 t) is recomputed (one exp2f) instead of read.
struct CecF1 : CecSeparable<CecF1> {
    __device__ static void add_term(float* acc, float z, float t) {
        const float y = cec_osz(z);
        acc[0] += exp2f(6.0f * 3.32192809488736f * t) * y * y;
    }
};

// f2, shifted Rastrigin.
struct CecF2 : CecSeparable<CecF2> {
    __device__ static void add_term(float* acc, float z, float t) {
        const float y = cec_y(z, t);
        acc[0] += y * y - 10.0f * cospif(2.0f * y) + 10.0f;
    }
};

// f3, shifted Ackley: two sums per row, combined by finish once the row is complete.
struct CecF3 : CecSeparable<CecF3> {
    static constexpr int num_sums = 2;
    __device__ static void add_term(float* acc, float z, float t) {
        const float y = cec_y(z, t);
        acc[0] += y * y;
        acc[1] += cospif(2.0f * y);
    }
    __host__ __device__ static float finish(const float* s, size_t d) {
        const float inv = 1.0f / (float)d;
        return -20.0f * expf(-0.2f * sqrtf(s[0] * inv)) - expf(s[1] * inv) + 20.0f + 2.718281828459045f;
    }
};

// 0/1 knapsack on packed genomes (binary GA): bit b of word u selects item 32 u + b.
//   f(x) = -P + rho max(0, W - c),  P = sum p_j x_j,  W = sum w_j x_j,  rho = max_j p_j / w_j
// (minimized; a feasible solution scores -P, and the linear penalty makes every infeasible
// solution worse than removing its excess weight). Two sums per row. The instance is one
// 32-bit word per item, read as uint4 (eight loads per genome word). The sums are exact
// integers within a word and accumulated in float, like every other fitness: totals above
// 2^24 are rounded to float precision (relative 6e-8).
struct Knapsack {
    using value_type = uint32_t;
    static constexpr bool positional = true;
    static constexpr int num_sums = 2;
    __device__ static void add_word(float* acc, uint32_t x, size_t word) {
        const uint4* it = reinterpret_cast<const uint4*>(c_fit.kp_items) + 8 * word;
        uint32_t P = 0, W = 0;
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            const uint4 v = it[q];
            const uint32_t item[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const uint32_t bit = (x >> (4 * q + r)) & 1u;
                P += bit * (item[r] & 0xFFFFu);
                W += bit * (item[r] >> 16);
            }
        }
        acc[0] += (float)P;
        acc[1] += (float)W;
    }
    __device__ static void add_at(float* acc, uint32_t x, size_t u) { add_word(acc, x, u); }
    __device__ static void add_vec4_at(float* acc, uint4 x, size_t u) {
        add_word(acc, x.x, 4 * u); add_word(acc, x.y, 4 * u + 1);
        add_word(acc, x.z, 4 * u + 2); add_word(acc, x.w, 4 * u + 3);
    }
    __device__ static float finish(const float* s, size_t) {
        return -s[0] + c_fit.kp_rho * fmaxf(0.0f, s[1] - c_fit.kp_capacity);
    }
};

// f12, shifted Rosenbrock: z = x - o (no other transformation), then
//   sum_{j<D-1} 100 (z_j^2 - z_{j+1})^2 + (z_j - 1)^2               on [-100, 100]^D.
// Neighbor-coupled like Rosenbrock: the kernel hands the raw gene before the unit, and the
// policy shifts it with o of that position.
struct CecF12 {
    using value_type = float;
    static constexpr bool positional = true;
    static constexpr bool needs_prev = true;
    __device__ static float cost_prev_at(float prev, float g, size_t u) {
        return Rosenbrock::pair(prev - c_fit.shift[u - 1], g - c_fit.shift[u]);
    }
    __device__ static float cost_first_at(float, size_t) { return 0.0f; }
    __device__ static float in_unit(float4 z) {
        return Rosenbrock::pair(z.x, z.y) + Rosenbrock::pair(z.y, z.z) + Rosenbrock::pair(z.z, z.w);
    }
    __device__ static float4 shifted(float4 g, size_t u) {
        const float4 o = reinterpret_cast<const float4*>(c_fit.shift)[u];
        return make_float4(g.x - o.x, g.y - o.y, g.z - o.z, g.w - o.w);
    }
    __device__ static float cost_prev_at(float prev, float4 g, size_t u) {
        const float4 z = shifted(g, u);
        return Rosenbrock::pair(prev - c_fit.shift[4 * u - 1], z.x) + in_unit(z);
    }
    __device__ static float cost_first_at(float4 g, size_t u) { return in_unit(shifted(g, u)); }
};

} // v2
} // bench
} // evoopt
