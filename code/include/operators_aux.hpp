#pragma once

#include <cuda_runtime.h>
#include "common.ipp"

// Auxiliary device math for the v2 operator policies.
//
// Kept separate from operators.hpp so the policy structs read as pure interface
// (parents in -> offspring out). Three families:
//   * EvoX-faithful real-valued per-gene math (DE/rand/1/bin, SBX, PM), scalar
//     and float4;
//   * binary word math (uniform crossover masks, bitflip masks), scalar and uint4;
//   * the in-block reduction used by the fused kernels.

namespace evoopt {
namespace bench {
namespace v2 {

// ---------------------------------------------------------------------------
// EvoX DE/rand/1/bin (de.py, base_vector="rand", K=1). parents =
// {target, base, diff1, diff2}; do_cx = rand < CR || j == j*.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float de_evox_gene(
    float x_i, float x_r1, float x_r2, float x_r3,
    float F, bool do_cx, float lb, float ub
) {
    const float v = x_r1 + F * (x_r2 - x_r3);
    const float u = do_cx ? v : x_i;
    return fminf(fmaxf(u, lb), ub);
}

__device__ __forceinline__ float4 de_evox4(
    float4 x_i, float4 x_r1, float4 x_r2, float4 x_r3,
    float F, b4 do_cx, float4 lb, float4 ub
) {
    const float4 v = make_float4(x_r1.x + F * (x_r2.x - x_r3.x),
                                 x_r1.y + F * (x_r2.y - x_r3.y),
                                 x_r1.z + F * (x_r2.z - x_r3.z),
                                 x_r1.w + F * (x_r2.w - x_r3.w));
    const float4 u = make_float4(do_cx.x ? v.x : x_i.x, do_cx.y ? v.y : x_i.y,
                                 do_cx.z ? v.z : x_i.z, do_cx.w ? v.w : x_i.w);
    return make_float4(fminf(fmaxf(u.x, lb.x), ub.x), fminf(fmaxf(u.y, lb.y), ub.y),
                       fminf(fmaxf(u.z, lb.z), ub.z), fminf(fmaxf(u.w, lb.w), ub.w));
}

// ---------------------------------------------------------------------------
// EvoX SBX (operators/crossover/sbx.py). One beta from mu, random sign, two
// "no-spread" overrides; children are mid +/- beta*h. Bounds are NOT used.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sbx_evox_from(
    float mu, uint32_t sign_bit, float r_half, float r_pc,
    float p1, float p2, float pro_c, float dis_c, bool first
) {
    const float ex = 1.0f / (dis_c + 1.0f);
    float beta = (mu <= 0.5f) ? powf(2.0f * mu, ex) : powf(2.0f - 2.0f * mu, -ex);
    beta *= (sign_bit & 1u) ? -1.0f : 1.0f;
    if (r_half < 0.5f) beta = 1.0f;
    if (r_pc > pro_c) beta = 1.0f;
    const float mid = 0.5f * (p1 + p2);
    const float h = 0.5f * (p1 - p2);
    return first ? (mid + beta * h) : (mid - beta * h);
}

__device__ __forceinline__ void sbx_evox4(
    float4 mu, uint4 sign, float4 r_half, float4 r_pc,
    float4 p1, float4 p2, float pro_c, float dis_c,
    float4& c1, float4& c2
) {
    const float ex = 1.0f / (dis_c + 1.0f);
    const float m[4]  = {mu.x, mu.y, mu.z, mu.w};
    const uint32_t s[4] = {sign.x, sign.y, sign.z, sign.w};
    const float rh[4] = {r_half.x, r_half.y, r_half.z, r_half.w};
    const float rp[4] = {r_pc.x, r_pc.y, r_pc.z, r_pc.w};
    const float a[4]  = {p1.x, p1.y, p1.z, p1.w};
    const float b[4]  = {p2.x, p2.y, p2.z, p2.w};
    float o1[4], o2[4];
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        float beta = (m[l] <= 0.5f) ? powf(2.0f * m[l], ex) : powf(2.0f - 2.0f * m[l], -ex);
        beta *= (s[l] & 1u) ? -1.0f : 1.0f;
        if (rh[l] < 0.5f) beta = 1.0f;
        if (rp[l] > pro_c) beta = 1.0f;
        const float mid = 0.5f * (a[l] + b[l]);
        const float h = 0.5f * (a[l] - b[l]);
        o1[l] = mid + beta * h;
        o2[l] = mid - beta * h;
    }
    c1 = make_float4(o1[0], o1[1], o1[2], o1[3]);
    c2 = make_float4(o2[0], o2[1], o2[2], o2[3]);
}

// ---------------------------------------------------------------------------
// EvoX polynomial mutation (operators/mutation/pm_mutation.py). Clamp x, gate
// with site < pro_m/D, then the two mu branches.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float pm_evox_from(
    float x, float site_r, float mu, float lb, float ub,
    float pro_m, float dis_m, size_t nCols
) {
    float out = fminf(fmaxf(x, lb), ub);
    if (site_r < pro_m / (float)nCols) {
        const float range = ub - lb;
        const float ex = 1.0f / (dis_m + 1.0f);
        if (mu <= 0.5f) {
            const float norm = (out - lb) / range;
            const float t = powf(2.0f * mu + (1.0f - 2.0f * mu) * powf(1.0f - norm, dis_m + 1.0f), ex);
            out = out + (t - 1.0f) * range;
        } else {
            const float norm = (ub - out) / range;
            const float t = 1.0f - powf(2.0f * (1.0f - mu) + 2.0f * (mu - 0.5f) * powf(1.0f - norm, dis_m + 1.0f), ex);
            out = out + t * range;
        }
    }
    return out;
}

__device__ __forceinline__ float4 pm_evox4(
    float4 x, float4 site, float4 mu, float4 lb, float4 ub,
    float pro_m, float dis_m, size_t nCols
) {
    const float xs[4] = {x.x, x.y, x.z, x.w};
    const float ss[4] = {site.x, site.y, site.z, site.w};
    const float ms[4] = {mu.x, mu.y, mu.z, mu.w};
    const float lo[4] = {lb.x, lb.y, lb.z, lb.w};
    const float hi[4] = {ub.x, ub.y, ub.z, ub.w};
    float o[4];
    #pragma unroll
    for (int l = 0; l < 4; ++l)
        o[l] = pm_evox_from(xs[l], ss[l], ms[l], lo[l], hi[l], pro_m, dis_m, nCols);
    return make_float4(o[0], o[1], o[2], o[3]);
}

// ---------------------------------------------------------------------------
// Binary uniform crossover (one uint32 mask per 32 genes).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t ux_apply(uint32_t v1, uint32_t v2, uint32_t mask, bool first) {
    return first ? ((~mask & v1) | (mask & v2)) : ((mask & v1) | (~mask & v2));
}

__device__ __forceinline__ uint4 ux_apply4(uint4 v1, uint4 v2, uint4 mask, bool first) {
    return make_uint4(ux_apply(v1.x, v2.x, mask.x, first),
                      ux_apply(v1.y, v2.y, mask.y, first),
                      ux_apply(v1.z, v2.z, mask.z, first),
                      ux_apply(v1.w, v2.w, mask.w, first));
}

// ---------------------------------------------------------------------------
// Bitflip mutation masks (one uint32 decision per bit).
// ---------------------------------------------------------------------------
template <typename RandomEngine>
__device__ __forceinline__ uint32_t bf_mask(RandomEngine& rng, uint32_t thr) {
    uquad4 q;
    uint32_t mask = 0;
    #pragma unroll
    for (int bit = 0; bit < 32; ++bit)
        mask |= ((draw_uint32(rng, q) < thr) ? 1u : 0u) << bit;
    return mask;
}

template <typename RandomEngine>
__device__ __forceinline__ uint4 bf_mask4(RandomEngine& rng, uint32_t thr) {
    uint4 mask = make_uint4(0, 0, 0, 0);
    #pragma unroll
    for (int bit = 0; bit < 32; ++bit) {
        const uint4 r = rng.next_uint4();
        mask.x |= ((r.x < thr) ? 1u : 0u) << bit;
        mask.y |= ((r.y < thr) ? 1u : 0u) << bit;
        mask.z |= ((r.z < thr) ? 1u : 0u) << bit;
        mask.w |= ((r.w < thr) ? 1u : 0u) << bit;
    }
    return mask;
}

// ---------------------------------------------------------------------------
// In-block sum over `blockDim.x` values; scratch is `blockDim.x` floats. Ends
// with a barrier so the scratch is reusable for the next reduction.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_sum(float v, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float r = smem[0];
    __syncthreads();
    return r;
}

// Per-lane reduction: each threadIdx.y lane reduces over its blockDim.x threads.
// `smem` must hold blockDim.x * blockDim.y floats.
__device__ __forceinline__ float block_sum_y(float v, float* smem) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    float* s = smem + (size_t)ty * blockDim.x;
    s[tx] = v;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

} // v2
} // bench
} // evoopt
