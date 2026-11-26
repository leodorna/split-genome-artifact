// Ceilings for the instruction roofline (scripts/run_roofline_v3.sh).
//
// Each kernel isolates one limit of the GPU; scripts/ncu_ceilings_v3.sh profiles
// them with the same Nsight Compute counters as the generation kernels, so the
// roofs and the measured points share units and clocks.
//
//   copy_dram     copy between two 4 GiB buffers (read + write)    -> DRAM bandwidth
//   copy_l2       ping-pong copy of a 2 x 16 MiB working set, loads
//                 and stores cached at L2 only                      -> L2 bandwidth
//   issue_fp32    independent FFMA chains                           -> warp issue rate
//   issue_int     independent IADD/SHF/LOP3 chains
//   issue_imad    independent IMAD chains (Philox multiplies run here)
//   issue_mix     FFMA and integer chains interleaved
//   philox_draws  Philox4x32-7 blocks from one engine per thread    -> random-number throughput
//   binary_rng    ux-bf crossover + bit-flip mutation on parents held in registers,
//                 with the per-unit streams of FusedGeneration_HighDim; scalar
//                 (binary_rng) or vec4 (binary_rng_vec4). The random-number work
//                 of one binary generation of --words 32-bit words, without memory
//                 traffic, selection, fitness or replacement.
//
// Usage: roofline_ceilings_v3 --kernel <name|all> [--warmup 3] [--reps 5] [--words 2^30]
// Prints kernel,best_ms,work,work_unit (best of --reps timed launches, CUDA events).
// Under ncu, profile launch --warmup of the kernel (0-based) to skip the warm-up.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include "operators.hpp"

using namespace evoopt::bench::v2;
using EP = evoopt::bench::PairPhiloxPhilox;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

// One element per thread: the fastest of the copy variants tried (grid-stride with
// 1-8 waves, 4x unrolled, cudaMemcpy D2D: 1.43-1.47 TB/s against 1.48 TB/s).
__global__ void copy_dram(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

// Each thread copies the same indices every pass and reads back only what it wrote,
// so the passes cannot be merged and there is no race between threads.
__global__ void copy_l2(float4* a, float4* b, size_t n, int passes) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int p = 0; p < passes; ++p) {
        const float4* src = (p & 1) ? b : a;
        float4* dst = (p & 1) ? a : b;
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
            __stcg(dst + i, __ldcg(src + i));
    }
}

__global__ void issue_fp32(float* out, int iters) {
    float a[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = (float)(threadIdx.x + j);
    for (int k = 0; k < iters; ++k) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = fmaf(a[j], 0.999f, 0.001f);
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += a[j];
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void issue_int(uint32_t* out, int iters) {
    uint32_t a[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = threadIdx.x * 8 + j;
    for (int k = 0; k < iters; ++k) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = (a[j] + (uint32_t)k) ^ (a[j] >> 7);
    }
    uint32_t s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s ^= a[j];
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void issue_imad(uint32_t* out, int iters) {
    uint32_t a[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = threadIdx.x * 8 + j;
    for (int k = 0; k < iters; ++k) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = a[j] * 0x9E3779B1u + (uint32_t)k;
    }
    uint32_t s = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s ^= a[j];
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void issue_mix(uint32_t* out, int iters) {
    float f[4];
    uint32_t a[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) { f[j] = (float)(threadIdx.x + j); a[j] = threadIdx.x * 4 + j; }
    for (int k = 0; k < iters; ++k) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            f[j] = fmaf(f[j], 0.999f, 0.001f);
            a[j] = (a[j] + (uint32_t)k) ^ (a[j] >> 7);
        }
    }
    uint32_t s = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) s ^= a[j] ^ __float_as_uint(f[j]);
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void philox_draws(uint32_t* out, int blocks_per_thread, unsigned long long seed) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    auto rng = EP::make_vector(seed, 0, (unsigned long long)RNG_STREAM::STREAM_MUTATION + tid);
    uint32_t acc = 0;
    for (int k = 0; k < blocks_per_thread; ++k) {
        const uint4 r = rng.next_uint4();
        acc ^= r.x ^ r.y ^ r.z ^ r.w;
    }
    out[tid] = acc;
}

// n = couple-units (one unit = one word, or one uint4 in vec4); each yields two
// offspring units, as in FusedGeneration_HighDim with nUnits = n and one couple.
template <bool VEC4>
__global__ void binary_rng(uint32_t* out, size_t n, unsigned long long seed) {
    constexpr unsigned long long CS = (unsigned long long)UX::OP_STREAM;
    constexpr unsigned long long MS = (unsigned long long)BF::OP_STREAM;
    const UX crossover{};
    const BF mutation{};
    const UX::CoupleState c_state{true};
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (size_t unit = tid; unit < n; unit += stride) {
        auto gene_rng_c = EP::make_vector(seed, 0, CS + unit);
        auto gene_rng_m = EP::make_vector(seed, 0, MS + unit);
        const uint32_t u = (uint32_t)unit;
        if constexpr (VEC4) {
            const uint4 pop[2] = {make_uint4(u, u ^ 0x55555555u, ~u, u * 3u),
                                  make_uint4(~u, u * 5u, u ^ 0xAAAAAAAAu, u + 1u)};
            const uint4 z = make_uint4(0, 0, 0, 0);
            uint4 off[2];
            crossover.apply_vec4(pop, off, z, z, unit, c_state, gene_rng_c);
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                mutation.apply_vec4(off[k], z, z, unit, n, gene_rng_m);
                acc ^= off[k].x ^ off[k].y ^ off[k].z ^ off[k].w;
            }
        } else {
            const uint32_t pop[2] = {u, ~u * 3u};
            uint32_t off[2];
            crossover.apply(pop, off, 0u, 0u, unit, c_state, gene_rng_c);
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                mutation.apply(off[k], 0u, 0u, unit, n, gene_rng_m);
                acc ^= off[k];
            }
        }
    }
    out[tid] = acc;
}

template <typename K>
static unsigned full_wave(K kernel, unsigned threads) {
    int dev = 0, sms = 0, per_sm = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, (int)threads, 0));
    return (unsigned)(sms * std::max(per_sm, 1));
}

template <typename F>
static float best_ms(F launch, int warmup, int reps) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < warmup; ++i) launch();
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(e0));
        launch();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms);
    }
    CK(cudaGetLastError());
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return best;
}

int main(int argc, char** argv) {
    std::string which = "all";
    int warmup = 3, reps = 5;
    size_t words = (size_t)1 << 30;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", s.c_str()); std::exit(1); } return std::string(argv[++i]); };
        if (s == "--kernel") which = next();
        else if (s == "--warmup") warmup = std::stoi(next());
        else if (s == "--reps") reps = std::stoi(next());
        else if (s == "--words") words = std::stoull(next());
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); return 1; }
    }
    auto on = [&](const char* k) { return which == "all" || which == k; };
    const unsigned T = 256;
    std::printf("kernel,best_ms,work,work_unit\n");

    if (on("copy_dram")) {
        const size_t n = ((size_t)4 << 30) / sizeof(float4);
        float4 *a, *b;
        CK(cudaMalloc(&a, n * sizeof(float4))); CK(cudaMalloc(&b, n * sizeof(float4)));
        CK(cudaMemset(a, 0, n * sizeof(float4)));
        const unsigned g = (unsigned)((n + T - 1) / T);
        const float ms = best_ms([&] { copy_dram<<<g, T>>>(a, b, n); }, warmup, reps);
        std::printf("copy_dram,%.4f,%zu,bytes\n", ms, 2 * n * sizeof(float4));
        cudaFree(a); cudaFree(b);
    }
    if (on("copy_l2")) {
        const size_t n = ((size_t)16 << 20) / sizeof(float4);
        const int passes = 256;
        float4 *a, *b;
        CK(cudaMalloc(&a, n * sizeof(float4))); CK(cudaMalloc(&b, n * sizeof(float4)));
        CK(cudaMemset(a, 0, n * sizeof(float4)));
        const unsigned g = full_wave(copy_l2, T);
        const float ms = best_ms([&] { copy_l2<<<g, T>>>(a, b, n, passes); }, warmup, reps);
        std::printf("copy_l2,%.4f,%zu,bytes\n", ms, 2 * n * sizeof(float4) * passes);
        cudaFree(a); cudaFree(b);
    }
    {
        const int iters = 1 << 16;
        int dev = 0, sms = 0, per_sm = 0;
        CK(cudaGetDevice(&dev));
        CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        CK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
        uint32_t* o;   // one value per resident thread, enough for any full-wave grid
        CK(cudaMalloc(&o, (size_t)sms * per_sm * sizeof(uint32_t)));
        auto run = [&](const char* name, auto kernel, auto* out) {
            if (!on(name)) return;
            const unsigned g = full_wave(kernel, T);
            const float ms = best_ms([&] { kernel<<<g, T>>>(out, iters); }, warmup, reps);
            std::printf("%s,%.4f,%zu,thread_chain_steps\n", name, ms, (size_t)g * T * iters * 8);
        };
        run("issue_fp32", issue_fp32, reinterpret_cast<float*>(o));
        run("issue_int", issue_int, o);
        run("issue_imad", issue_imad, o);
        run("issue_mix", issue_mix, o);
        cudaFree(o);
    }
    if (on("philox_draws")) {
        const unsigned g = full_wave(philox_draws, T);
        const size_t threads = (size_t)g * T;
        const int per = (int)(((size_t)1 << 31) / threads);   // about 2^31 blocks = 2^33 draws
        uint32_t* o; CK(cudaMalloc(&o, threads * sizeof(uint32_t)));
        const float ms = best_ms([&] { philox_draws<<<g, T>>>(o, per, 42ULL); }, warmup, reps);
        std::printf("philox_draws,%.4f,%zu,draws\n", ms, threads * per * 4);
        cudaFree(o);
    }
    auto run_binary = [&](const char* name, auto kernel, size_t per_unit) {
        if (!on(name)) return;
        const unsigned g = full_wave(kernel, T);
        const size_t n = words / 2 / per_unit;                     // couple-units
        uint32_t* o; CK(cudaMalloc(&o, (size_t)g * T * sizeof(uint32_t)));
        const float ms = best_ms([&] { kernel<<<g, T>>>(o, n, 42ULL); }, warmup, reps);
        // per offspring word: 32 bit-flip draws; per couple-unit: one crossover mask
        std::printf("%s,%.4f,%zu,draws\n", name, ms, words * 32 + n * per_unit);
        cudaFree(o);
    };
    run_binary("binary_rng", binary_rng<false>, 1);
    run_binary("binary_rng_vec4", binary_rng<true>, 4);
    return 0;
}
