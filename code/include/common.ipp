#pragma once

// Shared stateless-device helpers for the divergence benchmark kernels.
//
// RNG: two stateless engines, PCG (scalar) and Philox4x32 (vector), wrapped in
// Pcg32 / Philox4x32 with a symmetric scalar/vector draw interface. A
// RandomEnginePair<CoupleEngine, VectorEngine> composes the couple-decision
// engine with the gene engine. A kernel seeds a fresh register-local engine from
// (seed, subsequence, offset) and draws from it directly; there is no per-thread
// state array nor pre-generated buffer.

#include <cstdint>
#include <curanddx.hpp>
#include <cuda_runtime.h>

namespace evoopt {
namespace bench {

// Number of independent sub-sequences a single "stream id" maps onto.
constexpr unsigned int RNG_SUBSEQUENCES = 65536;

// curanddx thread-execution engine types (require TARGET_SM at definition).
using PcgEngine = decltype(
        curanddx::Generator<curanddx::pcg>()
        + curanddx::SM<TARGET_SM>()
        + curanddx::Thread()
);
using PhiloxEngine = decltype(
        curanddx::Generator<curanddx::philox4_32>()
        + curanddx::PhiloxRounds<7>()
        + curanddx::SM<TARGET_SM>()
        + curanddx::Thread()
);

// splitmix64 finalizer: a well-mixed 64-bit value from any input
__host__ __device__ inline unsigned long long mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Key of one generation: splitmix64 of (seed, generation). Consecutive generations
// therefore get unrelated 64-bit keys, never the sequence 0, 1, 2, ... of the
// generation counter itself.
__host__ __device__ inline unsigned long long generation_key(unsigned long long seed,
                                                             unsigned long long gen_offset) {
    return mix64(seed + 0x9E3779B97F4A7C15ULL * (gen_offset + 1));
}

// Build a fresh stateless engine for the given stream id (adapted from evoopt
// src/evoopt/core/statelessRng.hpp). Each generation (`gen_offset`) gets its own
// key and each element its own subsequence (`stream_tag + id`, offset 0), so
// streams never overlap: not across generations (adding gen_offset to the stream
// index made element k of generation g+1 reuse the stream of element k+1 of
// generation g) and not within one (splitting the index into subsequence
// id % 65536 and offset id / 65536 made element k + 65536 reuse element k's stream
// shifted by one block). Both engines take a 64-bit subsequence at constant cost
// (Philox: high half of the counter; PCG: the stream increment).
// Philox is counter-based, so a distinct subsequence already gives a distinct
// counter and the generation key is used as is. curanddx's PCG instead starts
// every stream from the key itself, so with a shared key the first draw would be
// the same for every element whatever its subsequence; PCG streams therefore also
// mix the element into the key. `stream_tag` (default 0) is a per-operator salt;
// the step benchmark already folds its operator stream into `id`.
template <typename RandomEngine>
__device__ inline RandomEngine make_rng(unsigned long long seed,
                                        unsigned long long gen_offset,
                                        unsigned long long id,
                                        unsigned long long stream_tag = 0) {
    const unsigned long long sid = stream_tag + id;
    const unsigned long long key = generation_key(seed, gen_offset);
    if constexpr (RandomEngine::is_generator_philox) {
        return RandomEngine(key, sid, 0);
    } else {
        return RandomEngine(mix64(key ^ (sid * 0xD1B54A32D192ED03ULL)), sid, 0);
    }
}

// ---------------------------------------------------------------------------
// engine wrappers: symmetric scalar/vector draw interface for both generators.
// Either can serve as the couple engine, the vector engine, or both.
// ---------------------------------------------------------------------------

struct Pcg32 {
    static constexpr bool is_generator_philox = false;
    PcgEngine e;

    __device__ inline Pcg32(unsigned long long seed, unsigned long long sub, unsigned long long off)
        : e(seed, sub, off) {}

    __device__ inline uint32_t generate()    { return e.generate(); }
    __device__ inline uint32_t next_uint32() { return e.generate(); }
    __device__ inline float    next_float()  {
        curanddx::uniform<float> d;
        return d.generate(e);
    }

    // 4 sequential 32-bit state advances packed into a uint4.
    __device__ inline uint4 generate4() {
        uint4 r;
        r.x = e.generate(); r.y = e.generate(); r.z = e.generate(); r.w = e.generate();
        return r;
    }
    __device__ inline uint4  next_uint4()   { return generate4(); }
    __device__ inline float4 next_uniform4() {
        curanddx::uniform<float> d;
        return make_float4(d.generate(e), d.generate(e), d.generate(e), d.generate(e));
    }
};

struct Philox4x32 {
    static constexpr bool is_generator_philox = true;
    PhiloxEngine e;

    __device__ inline Philox4x32(unsigned long long seed, unsigned long long sub, unsigned long long off)
        : e(seed, sub, off) {}

    __device__ inline uint4  generate4()   { return e.generate4(); }
    __device__ inline uint4  next_uint4()  { return e.generate4(); }
    __device__ inline float4 next_uniform4() {
        curanddx::uniform<float> d;
        return d.generate4(e);
    }
    // convenience scalar draws (a full Feistel round each; use the cached
    // draw_uint32/draw_uniform helpers for repeated scalar draws instead).
    __device__ inline uint32_t next_uint32() { return e.generate4().x; }
    __device__ inline float    next_float()  {
        curanddx::uniform<float> d;
        return d.generate4(e).x;
    }
};

// ---------------------------------------------------------------------------
// composite engine policy: couple engine + vector (gene) engine.
// ---------------------------------------------------------------------------

// Couple streams carry this tag (top bit of the subsequence) so that they never
// coincide with gene streams: the kernels number couple streams tag + i and gene
// streams tag + i * nUnits + u, which overlap as integers, and with the same engine
// on both sides couple i would replay gene u = i of couple 0.
constexpr unsigned long long COUPLE_STREAM_TAG = 1ULL << 63;

template <typename CoupleEngine, typename VectorEngine>
struct RandomEnginePair {
    using CoupleRNG = CoupleEngine;
    using VectorRNG = VectorEngine;

    __device__ static inline CoupleRNG make_couple(
        unsigned long long seed, unsigned long long offset, size_t couple_idx) {
        return make_rng<CoupleRNG>(seed, offset, couple_idx, COUPLE_STREAM_TAG);
    }

    __device__ static inline VectorRNG make_vector(
        unsigned long long seed, unsigned long long offset, size_t flat_idx) {
        return make_rng<VectorRNG>(seed, offset, flat_idx);
    }
};

// default policy: PCG for couple decisions, Philox for the vector (gene) engine.
using DefaultEnginePolicy = RandomEnginePair<Pcg32, Philox4x32>;

// explicit pair combinations used by the benchmark harness.
using PairPCGPCG         = RandomEnginePair<Pcg32, Pcg32>;
using PairPCGPhilox      = RandomEnginePair<Pcg32, Philox4x32>;
using PairPhiloxPCG      = RandomEnginePair<Philox4x32, Pcg32>;
using PairPhiloxPhilox   = RandomEnginePair<Philox4x32, Philox4x32>;

// per-thread lane caches: one generate4() serves four scalar draws.
struct quad4 {
    float4 v;
    int n = 4;
};
struct uquad4 {
    uint4 v;
    int n = 4;
};

// one scalar uniform in [0,1) from either generator (cached for Philox).
template <typename RandomEngine>
__device__ inline float draw_uniform(RandomEngine& rng, quad4& q) {
    if constexpr (RandomEngine::is_generator_philox) {
        if (q.n == 4) {
            q.v = rng.next_uniform4();
            q.n = 0;
        }
        float val = (q.n == 0) ? q.v.x : (q.n == 1) ? q.v.y : (q.n == 2) ? q.v.z : q.v.w;
        ++q.n;
        return val;
    } else {
        return rng.next_float();
    }
}

// one scalar raw uint32 from either generator (cached for Philox).
template <typename RandomEngine>
__device__ inline uint32_t draw_uint32(RandomEngine& rng, uquad4& q) {
    if constexpr (RandomEngine::is_generator_philox) {
        if (q.n == 4) {
            q.v = rng.next_uint4();
            q.n = 0;
        }
        uint32_t val = (q.n == 0) ? q.v.x : (q.n == 1) ? q.v.y : (q.n == 2) ? q.v.z : q.v.w;
        ++q.n;
        return val;
    } else {
        return rng.next_uint32();
    }
}

// ---------------------------------------------------------------------------
// determinism policy
// ---------------------------------------------------------------------------
// THREAD_DETERMINISTIC (default, throughput): a stream is bound to the executing
// thread; the engine is built once and auto-increments across the stride loop.
// DATA_DETERMINISTIC (strict reproducibility): a stream is bound to the global
// data coordinate; the engine is rebuilt inside the loop via make_rng(flat_idx).
#ifndef EVOOPT_DATA_DET
#define EVOOPT_DATA_DET false
#endif

// compile-time label for reporting (ternary, since #if can't evaluate true/false)
constexpr const char* EVOOPT_DET_NAME = EVOOPT_DATA_DET ? "data" : "thread";

// A per-element draw handle: engine + scalar caches. Supports multiple sequential
// draws from a single element's stream (required by bitflip/SBX/PM which draw
// more than once per gene/chunk).
template <typename RandomEngine>
struct GeneStream {
    RandomEngine rng;
    uquad4 uq;
    quad4 fq;

    __device__ inline GeneStream(RandomEngine e) : rng(e) {}

    __device__ inline uint32_t next_uint32()  { return draw_uint32(rng, uq); }
    __device__ inline float    next_uniform() { return draw_uniform(rng, fq); }
    __device__ inline uint4    next_uint4()   { return rng.next_uint4(); }
    __device__ inline float4   next_uniform4(){ return rng.next_uniform4(); }
};

template <typename EnginePolicy, bool DataDet = EVOOPT_DATA_DET>
struct GeneRng;

// THREAD mode: one persistent stream per thread; counter auto-increments.
template <typename EnginePolicy>
struct GeneRng<EnginePolicy, false> {
    GeneStream<typename EnginePolicy::VectorRNG> s;

    __device__ inline GeneRng(unsigned long long sd, unsigned long long o,
                              size_t thread_id, size_t /*data_base*/)
        : s(EnginePolicy::make_vector(sd, o, (unsigned long long)thread_id)) {}

    // bind `auto&&` to the returned reference -> zero-copy persistent stream.
    __device__ inline GeneStream<typename EnginePolicy::VectorRNG>& stream(size_t /*j*/) { return s; }

    __device__ inline uint32_t next_uint32(size_t /*j*/)  { return s.next_uint32(); }
    __device__ inline float    next_uniform(size_t /*j*/) { return s.next_uniform(); }
    __device__ inline uint4    next_uint4(size_t /*j*/)   { return s.next_uint4(); }
    __device__ inline float4   next_uniform4(size_t /*j*/){ return s.next_uniform4(); }
};

// DATA mode: stateless, one fresh stream per data coordinate (no persistent
// RNG state, so no register waste).
template <typename EnginePolicy>
struct GeneRng<EnginePolicy, true> {
    unsigned long long seed;
    unsigned long long offset;
    size_t base;

    __device__ inline GeneRng(unsigned long long sd, unsigned long long o,
                              size_t /*thread_id*/, size_t data_base)
        : seed(sd), offset(o), base(data_base) {}

    // bind `auto&&` to the returned prvalue -> temporary lifetime extends to
    // the loop-body scope, so the fresh stream survives across its draws.
    __device__ inline GeneStream<typename EnginePolicy::VectorRNG> stream(size_t j) {
        return GeneStream<typename EnginePolicy::VectorRNG>(
            EnginePolicy::make_vector(seed, offset, (unsigned long long)(base + j)));
    }
};

inline __device__ __host__ uint32_t prob_to_threshold(float prob) {
    if (prob >= 1.0f) return 0xFFFFFFFF; // max representable uint32_t
    if (prob <= 0.0f) return 0;
    // multiplying by 2^32 uniformly scales [0.0, 1.0) into the integer domain
    return static_cast<uint32_t>(prob * 4294967296.0f);
}

// per-lane boolean mask (CUDA has no built-in bool4)
struct b4 {
    bool x, y, z, w;
};

__device__ inline b4 make_b4(bool x, bool y, bool z, bool w) {
    b4 b;
    b.x = x; b.y = y; b.z = z; b.w = w;
    return b;
}

// ---------------------------------------------------------------------------
// scalar per-variable math (verbatim from the production operators)
// ---------------------------------------------------------------------------

__device__ inline float calc_beta(float u, float gamma, float ef) {
    const float one_minus_gamma = 1.0f - gamma;
    const float eps = 1.19209290e-07F;
    const float exponent = 1.0f / (ef + 1.0f);

    if (one_minus_gamma <= eps) {
        return powf(2.0f * u, exponent);
    }
    const float a = fmaxf(2.0f * u * one_minus_gamma, eps);
    const float b = fmaxf(2.0f * (1.0f - u * one_minus_gamma), eps);
    const float base = fminf(a, 1.0f / b);
    return powf(base, exponent);
}

// SBX per-variable pair of children. `swap_enabled && u_bin < prob_bin_thr`
// swaps the two children (prob_exch/prob_bin exchange), otherwise c1 is the
// "first" child. The caller applies the `u_skip < prob_var` activation gate.
__device__ inline void sbx_var2(
    float p1, float p2, float u, uint32_t u_bin, bool swap_enabled,
    uint32_t prob_bin_thr, float ef, float lb, float ub,
    float& out_c1, float& out_c2
) {
    float min_p = fminf(p1, p2);
    float max_p = fmaxf(p1, p2);
    float rspread = 1.0f / fmaxf(max_p - min_p, 1e-9f);

    float alpha1 = 1.0f + (min_p - lb) * rspread;
    float alpha2 = 1.0f + (ub - max_p) * rspread;

    float gamma1 = powf(alpha1, -(ef + 1.0f));
    float gamma2 = powf(alpha2, -(ef + 1.0f));

    float beta1 = calc_beta(u, gamma1, ef);
    float beta2 = calc_beta(u, gamma2, ef);

    float c1 = 0.5f * (1.0f + beta1) * min_p + 0.5f * (1.0f - beta1) * max_p;
    float c2 = 0.5f * (1.0f - beta2) * min_p + 0.5f * (1.0f + beta2) * max_p;

    c1 = fminf(fmaxf(c1, lb), ub);
    c2 = fminf(fmaxf(c2, lb), ub);

    if (swap_enabled && (u_bin < prob_bin_thr)) {
        out_c1 = c2;
        out_c2 = c1;
    } else {
        out_c1 = c1;
        out_c2 = c2;
    }
}

// DE/rand/1/bin per-variable trial.
__device__ inline float de_var(
    float x_target, float x_r1, float x_r2, float x_r3,
    float F, bool do_crossover, float lb, float ub
) {
    float v = x_r1 + F * (x_r2 - x_r3);
    float u = do_crossover ? v : x_target;
    return (u < lb) ? lb : ((u > ub) ? ub : u);
}

// Polynomial mutation per-variable (branchless).
__device__ inline float pm_var(float x, float u, float ef, float lb, float ub, bool isMutate) {
    float delta1 = (x - lb) / (ub - lb);
    float delta2 = (ub - x) / (ub - lb);

    bool isUHigh = u > 0.5f;

    float deltaqLess = powf(2 * u + (1 - 2 * u) * powf(1 - delta1, ef + 1), 1 / (ef + 1)) - 1;
    float deltaqHigher = 1 - powf(2 * (1 - u) + 2 * (u - 0.5f) * powf(1 - delta2, ef + 1), 1 / (ef + 1));

    float xm = x + (deltaqLess * !isUHigh + deltaqHigher * isUHigh) * (ub - lb);
    xm = fminf(fmaxf(xm, lb), ub);

    return isMutate ? xm : x;
}

// ---------------------------------------------------------------------------
// float4 per-variable math (vec4 layouts; scalar bounds broadcast to all lanes)
// ---------------------------------------------------------------------------

__device__ inline float4 f4min(float4 a, float4 b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
__device__ inline float4 f4max(float4 a, float4 b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
__device__ inline float4 f4pow(float4 a, float p) {
    return make_float4(__powf(a.x, p), __powf(a.y, p), __powf(a.z, p), __powf(a.w, p));
}

__device__ inline float4 calc_beta4(float4 u, float4 gamma, float ef) {
    const float eps = 1.19209290e-07F;
    const float exponent = 1.0f / (ef + 1.0f);
    float4 o = make_float4(1.0f - gamma.x, 1.0f - gamma.y, 1.0f - gamma.z, 1.0f - gamma.w);
    float4 a = make_float4(fmaxf(2.0f * u.x * o.x, eps), fmaxf(2.0f * u.y * o.y, eps),
                           fmaxf(2.0f * u.z * o.z, eps), fmaxf(2.0f * u.w * o.w, eps));
    float4 b = make_float4(fmaxf(2.0f * (1.0f - u.x * o.x), eps), fmaxf(2.0f * (1.0f - u.y * o.y), eps),
                           fmaxf(2.0f * (1.0f - u.z * o.z), eps), fmaxf(2.0f * (1.0f - u.w * o.w), eps));
    float4 base = make_float4(fminf(a.x, 1.0f / b.x), fminf(a.y, 1.0f / b.y),
                              fminf(a.z, 1.0f / b.z), fminf(a.w, 1.0f / b.w));
    float4 res;
    res.x = (o.x <= eps) ? powf(2.0f * u.x, exponent) : powf(base.x, exponent);
    res.y = (o.y <= eps) ? powf(2.0f * u.y, exponent) : powf(base.y, exponent);
    res.z = (o.z <= eps) ? powf(2.0f * u.z, exponent) : powf(base.z, exponent);
    res.w = (o.w <= eps) ? powf(2.0f * u.w, exponent) : powf(base.w, exponent);
    return res;
}

// SBX pair of children, vectorized. gate.x/y/z/w disable (per-variable
// u_skip < prob_var); swap_enabled/prob_bin_thr exchange the two children.
__device__ inline void sbx_var4(
    float4 p1, float4 p2, float4 u, uint4 u_bin, bool swap_enabled,
    uint32_t prob_bin_thr, float ef, float lb, float ub,
    float4& out_c1, float4& out_c2
) {
    float4 min_p = f4min(p1, p2);
    float4 max_p = f4max(p1, p2);
    float4 rspread = make_float4(1.0f / fmaxf(max_p.x - min_p.x, 1e-9f),
                                 1.0f / fmaxf(max_p.y - min_p.y, 1e-9f),
                                 1.0f / fmaxf(max_p.z - min_p.z, 1e-9f),
                                 1.0f / fmaxf(max_p.w - min_p.w, 1e-9f));
    float4 alpha1 = make_float4(1.0f + (min_p.x - lb) * rspread.x, 1.0f + (min_p.y - lb) * rspread.y,
                                1.0f + (min_p.z - lb) * rspread.z, 1.0f + (min_p.w - lb) * rspread.w);
    float4 alpha2 = make_float4(1.0f + (ub - max_p.x) * rspread.x, 1.0f + (ub - max_p.y) * rspread.y,
                                1.0f + (ub - max_p.z) * rspread.z, 1.0f + (ub - max_p.w) * rspread.w);
    float4 gamma1 = f4pow(alpha1, -(ef + 1.0f));
    float4 gamma2 = f4pow(alpha2, -(ef + 1.0f));
    float4 beta1 = calc_beta4(u, gamma1, ef);
    float4 beta2 = calc_beta4(u, gamma2, ef);

    float4 c1 = make_float4(0.5f * (1.0f + beta1.x) * min_p.x + 0.5f * (1.0f - beta1.x) * max_p.x,
                            0.5f * (1.0f + beta1.y) * min_p.y + 0.5f * (1.0f - beta1.y) * max_p.y,
                            0.5f * (1.0f + beta1.z) * min_p.z + 0.5f * (1.0f - beta1.z) * max_p.z,
                            0.5f * (1.0f + beta1.w) * min_p.w + 0.5f * (1.0f - beta1.w) * max_p.w);
    float4 c2 = make_float4(0.5f * (1.0f - beta2.x) * min_p.x + 0.5f * (1.0f + beta2.x) * max_p.x,
                            0.5f * (1.0f - beta2.y) * min_p.y + 0.5f * (1.0f + beta2.y) * max_p.y,
                            0.5f * (1.0f - beta2.z) * min_p.z + 0.5f * (1.0f + beta2.z) * max_p.z,
                            0.5f * (1.0f - beta2.w) * min_p.w + 0.5f * (1.0f + beta2.w) * max_p.w);

    c1 = make_float4(fminf(fmaxf(c1.x, lb), ub), fminf(fmaxf(c1.y, lb), ub),
                     fminf(fmaxf(c1.z, lb), ub), fminf(fmaxf(c1.w, lb), ub));
    c2 = make_float4(fminf(fmaxf(c2.x, lb), ub), fminf(fmaxf(c2.y, lb), ub),
                     fminf(fmaxf(c2.z, lb), ub), fminf(fmaxf(c2.w, lb), ub));

    bool swap_x = swap_enabled && (u_bin.x < prob_bin_thr);
    bool swap_y = swap_enabled && (u_bin.y < prob_bin_thr);
    bool swap_z = swap_enabled && (u_bin.z < prob_bin_thr);
    bool swap_w = swap_enabled && (u_bin.w < prob_bin_thr);

    out_c1 = make_float4(swap_x ? c2.x : c1.x, swap_y ? c2.y : c1.y,
                         swap_z ? c2.z : c1.z, swap_w ? c2.w : c1.w);
    out_c2 = make_float4(swap_x ? c1.x : c2.x, swap_y ? c1.y : c2.y,
                         swap_z ? c1.z : c2.z, swap_w ? c1.w : c2.w);
}

// DE/rand/1/bin trial, vectorized (per-lane crossover decision).
__device__ inline float4 de_var4(float4 x, float4 x_r1, float4 x_r2, float4 x_r3,
                                 float F, b4 do_crossover, float lb, float ub) {
    float4 v = make_float4(x_r1.x + F * (x_r2.x - x_r3.x), x_r1.y + F * (x_r2.y - x_r3.y),
                           x_r1.z + F * (x_r2.z - x_r3.z), x_r1.w + F * (x_r2.w - x_r3.w));
    float4 u = make_float4(do_crossover.x ? v.x : x.x, do_crossover.y ? v.y : x.y,
                           do_crossover.z ? v.z : x.z, do_crossover.w ? v.w : x.w);
    return make_float4(fminf(fmaxf(u.x, lb), ub), fminf(fmaxf(u.y, lb), ub),
                       fminf(fmaxf(u.z, lb), ub), fminf(fmaxf(u.w, lb), ub));
}

// Polynomial mutation, vectorized (per-lane mutate decision).
__device__ inline float4 pm_var4(float4 x, float4 u, uint4 u_mut,
                                 uint32_t prob_var_thr, float ef, float lb, float ub) {
    const float p_pow = 1.0f / (ef + 1.0f);
    const float range = ub - lb;
    float4 delta1 = make_float4((x.x - lb) / range, (x.y - lb) / range,
                                (x.z - lb) / range, (x.w - lb) / range);
    float4 delta2 = make_float4((ub - x.x) / range, (ub - x.y) / range,
                                (ub - x.z) / range, (ub - x.w) / range);

    float4 dl = make_float4(
        powf(2 * u.x + (1 - 2 * u.x) * powf(1 - delta1.x, ef + 1), p_pow) - 1,
        powf(2 * u.y + (1 - 2 * u.y) * powf(1 - delta1.y, ef + 1), p_pow) - 1,
        powf(2 * u.z + (1 - 2 * u.z) * powf(1 - delta1.z, ef + 1), p_pow) - 1,
        powf(2 * u.w + (1 - 2 * u.w) * powf(1 - delta1.w, ef + 1), p_pow) - 1);
    float4 dh = make_float4(
        1 - powf(2 * (1 - u.x) + 2 * (u.x - 0.5f) * powf(1 - delta2.x, ef + 1), p_pow),
        1 - powf(2 * (1 - u.y) + 2 * (u.y - 0.5f) * powf(1 - delta2.y, ef + 1), p_pow),
        1 - powf(2 * (1 - u.z) + 2 * (u.z - 0.5f) * powf(1 - delta2.z, ef + 1), p_pow),
        1 - powf(2 * (1 - u.w) + 2 * (u.w - 0.5f) * powf(1 - delta2.w, ef + 1), p_pow));

    float4 xm = make_float4(
        x.x + (u.x > 0.5f ? dh.x : dl.x) * range,
        x.y + (u.y > 0.5f ? dh.y : dl.y) * range,
        x.z + (u.z > 0.5f ? dh.z : dl.z) * range,
        x.w + (u.w > 0.5f ? dh.w : dl.w) * range);
    xm = make_float4(fminf(fmaxf(xm.x, lb), ub), fminf(fmaxf(xm.y, lb), ub),
                     fminf(fmaxf(xm.z, lb), ub), fminf(fmaxf(xm.w, lb), ub));

    return make_float4((u_mut.x < prob_var_thr) ? xm.x : x.x,
                       (u_mut.y < prob_var_thr) ? xm.y : x.y,
                       (u_mut.z < prob_var_thr) ? xm.z : x.z,
                       (u_mut.w < prob_var_thr) ? xm.w : x.w);
}

// ---------------------------------------------------------------------------
// stream compaction pipeline (stateless). Shared by the SBX and UX launchers.
// ---------------------------------------------------------------------------

// mask[t] = rand < p  for t in [0, n). Stateless: each couple re-seeds itself.
template <typename EnginePolicy>
__global__ void activation_mask(bool* mask, float p, size_t n,
                                unsigned long long gen_seed, unsigned long long gen_offset) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    auto rng = EnginePolicy::make_couple(gen_seed, gen_offset, t);
    uquad4 q;
    mask[t] = draw_uint32(rng, q) < prob_to_threshold(p);
}

// atomic-counter based compaction; no hidden allocations.
__global__ void compact_by_mask(const bool* mask, unsigned long long* counter,
                                size_t* selected, size_t n) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    if (mask[t]) {
        unsigned long long slot = atomicAdd(counter, 1ULL);
        selected[slot] = t;
    }
}

} // bench
} // evoopt
