// v2 benchmark: policy-based, island-modelled, scalar/vec4 kernels.
//
//   --mode variation   Phase 2: crossover + mutation + fitness (FusedVariation),
//                      external parents, no selection/replacement.
//   --mode generation  Phase 3: selection + crossover + mutation + fitness +
//                      replacement (FusedGeneration), one launch per generation.
//
//   --algo   de-gen | ga-gen | ux-bf
//   --layout individual | individual_vec4 | highdim | highdim_vec4
//   --islands I   --n N (per-island population rows)   --d D
//   --no-sel   use externally supplied parents (NoneSelection) in generation mode
//   --fitness sphere | rastrigin | rosenbrock | cec-f1 | cec-f2 | cec-f3 | cec-f12   (de-gen, ga-gen;
//              all but sphere only with --mode generation and the highdim layouts).
//              cec-f1..f3 are CEC 2013 LSGO f1 (elliptic), f2 (Rastrigin) and f3 (Ackley),
//              shifted, with T_osz / T_asy / Lambda; cec-f12 is f12 (shifted Rosenbrock,
//              neighbor-coupled). The shift comes from --shift-seed
//              (default 1) or --shift-file (D float32 values, or the suite's .txt xopt file).
//              knapsack (ux-bf only): 0/1 knapsack with D items, generated from --kp-seed
//              (default 1), --kp-type strong (default) | uncorrelated.
//   --fitness (cont.)                     (de-gen, ga-gen; rastrigin and rosenbrock only with
//              --mode generation and the highdim layouts), --lb/--ub the gene bounds
//              (defaults: sphere [0,1], rastrigin [-5.12,5.12], rosenbrock [-2.048,2.048])
//
// Emits the paper CSV: op,layout,I,N,D,total,xthreads,yblock,time_us
// (total = I*N; time_us = per-launch for variation, per-generation for generation).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include <cuda_runtime.h>

#include "operators.hpp"
#include "generation_buffer.hpp"
#include "kernels/fused_v2.hpp"
#include "kernels/fused_generation_v2.hpp"
#include "kernels/decomposed_v2.hpp"

using namespace evoopt::bench::v2;
// Philox4x32 for every stream (couple, selection and gene draws), as in the paper.
using EP = evoopt::bench::PairPhiloxPhilox;

// Seed-dependent, device-side population init. Replaces the host-side fill so
// each (seed) invocation is cheap and starts from an independent population.
// Fills 4 elements per draw with the vector engine; handles a possible tail.
template <typename T, typename EnginePolicy>
__global__ void InitPopulationKernel(T* __restrict__ pop, size_t total,
                                     unsigned long long seed) {
    constexpr unsigned long long INIT_OFFSET = 0x500000000ULL;
    const size_t nvec = total / 4;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < nvec; i += stride) {
        auto rng = EnginePolicy::make_vector(seed, INIT_OFFSET, i);
        if constexpr (std::is_same_v<T, uint32_t>) {
            const uint4 r = rng.next_uint4();
            pop[4 * i + 0] = r.x; pop[4 * i + 1] = r.y;
            pop[4 * i + 2] = r.z; pop[4 * i + 3] = r.w;
        } else {
            const float4 r = rng.next_uniform4();
            pop[4 * i + 0] = r.x; pop[4 * i + 1] = r.y;
            pop[4 * i + 2] = r.z; pop[4 * i + 3] = r.w;
        }
    }
    if (total % 4 != 0) {
        const size_t base = nvec * 4;
        for (size_t i = base + tid; i < total; i += stride) {
            auto rng = EnginePolicy::make_vector(seed, INIT_OFFSET, i);
            if constexpr (std::is_same_v<T, uint32_t>) pop[i] = (T)rng.next_uint32();
            else pop[i] = (T)rng.next_float();
        }
    }
}

// Maps the [0, 1) initial population to [lo, hi); launched only for bounds other than
// [0, 1], so the Sphere runs keep their initial population.
__global__ void ScalePopulation(float* __restrict__ pop, size_t total, float lo, float hi) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride)
        pop[i] = lo + (hi - lo) * pop[i];
}

// Shift vector of the positional functions (host copy, for --check).
static std::vector<float> g_shift;

// Shift o_j = lo + (hi - lo) (0.1 + 0.8 u_j), u_j = top 24 bits of splitmix64(seed * 2^32 + j)
// / 2^24: inside the central 80% of the range. scripts/shift.py computes the same values.
static std::vector<float> make_shift(size_t d, double lo, double hi, unsigned long long seed) {
    std::vector<float> o(d);
    for (size_t j = 0; j < d; ++j) {
        unsigned long long z = (seed << 32) + j + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        const double u = (double)(z >> 40) / 16777216.0;
        o[j] = (float)(lo + (hi - lo) * (0.1 + 0.8 * u));
    }
    return o;
}

// 0/1 knapsack instance: item j has weight w_j = 1 + (h mod R) and profit p_j = 1 + ((h >> 32)
// mod R) (uncorrelated) or w_j + R/10 (strongly correlated), h = splitmix64(seed * 2^32 + j),
// R = 1000; capacity c = floor(sum_j w_j / 2). step_bench/knapsack.py builds the same instance.
struct KnapsackInstance {
    std::vector<uint32_t> items;   // (w << 16) | p
    std::vector<uint32_t> w, p;
    double capacity = 0, rho = 0;
};
static KnapsackInstance g_kp;
static KnapsackInstance make_knapsack(size_t n, unsigned long long seed, bool strong) {
    KnapsackInstance k;
    k.items.resize(n); k.w.resize(n); k.p.resize(n);
    const unsigned long long R = 1000;
    double sw = 0;
    for (size_t j = 0; j < n; ++j) {
        unsigned long long z = (seed << 32) + j + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        const uint32_t w = (uint32_t)(1 + z % R);
        const uint32_t p = strong ? w + (uint32_t)(R / 10) : (uint32_t)(1 + (z >> 32) % R);
        k.w[j] = w; k.p[j] = p; k.items[j] = (w << 16) | p;
        sw += w;
        k.rho = std::max(k.rho, (double)p / w);
    }
    k.capacity = std::floor(sw / 2);
    return k;
}

// Host fitness of one row, for --check.
template <typename FIT>
static float host_fitness(const float* x, size_t d) {
    float s = 0;
    if constexpr (std::is_same_v<FIT, CecF12>) {
        // Benchmarks.cpp rosenbrock() on z = x - o.
        double t = 0;
        for (size_t j = 0; j + 1 < d; ++j) {
            const double z = (double)x[j] - g_shift[j], zn = (double)x[j + 1] - g_shift[j + 1];
            const double a = z * z - zn, b = z - 1.0;
            t += 100.0 * a * a + b * b;
        }
        return (float)t;
    }
    if constexpr (std::is_same_v<FIT, CecF1> || std::is_same_v<FIT, CecF2> || std::is_same_v<FIT, CecF3>) {
        // Double-precision transcription of the organizers' C++ code (Benchmarks.cpp):
        // transform_osz, transform_asy(0.2), Lambda(10), then elliptic / rastrigin / ackley.
        auto osz = [](double z) {
            if (z == 0.0) return 0.0;
            const double h = std::log(std::fabs(z));
            const double c1 = z > 0 ? 10.0 : 5.5, c2 = z > 0 ? 7.9 : 3.1;
            return (z > 0 ? 1.0 : -1.0) * std::exp(h + 0.049 * (std::sin(c1 * h) + std::sin(c2 * h)));
        };
        const double dm1 = (double)(d > 1 ? d - 1 : 1);
        double s1 = 0, s2 = 0;
        for (size_t j = 0; j < d; ++j) {
            const double t = (double)j / dm1;
            double y = osz((double)x[j] - g_shift[j]);
            if constexpr (std::is_same_v<FIT, CecF1>) {
                s1 += std::pow(1.0e6, t) * y * y;
                continue;
            }
            if (y > 0) y = std::pow(y, 1.0 + 0.2 * t * std::sqrt(y));
            y *= std::pow(10.0, 0.5 * t);
            if constexpr (std::is_same_v<FIT, CecF2>) s1 += y * y - 10.0 * std::cos(2.0 * M_PI * y) + 10.0;
            else { s1 += y * y; s2 += std::cos(2.0 * M_PI * y); }
        }
        if constexpr (std::is_same_v<FIT, CecF3>)
            return (float)(-20.0 * std::exp(-0.2 * std::sqrt(s1 / d)) - std::exp(s2 / d) + 20.0 + M_E);
        return (float)s1;
    }
    if constexpr (std::is_same_v<FIT, Rosenbrock>) { for (size_t j = 1; j < d; ++j) s += Rosenbrock::pair(x[j - 1], x[j]); }
    else if constexpr (std::is_same_v<FIT, Rastrigin>) { for (size_t j = 0; j < d; ++j) s += Rastrigin::cost(x[j]); }
    else { for (size_t j = 0; j < d; ++j) s += x[j] * x[j]; }
    return s;
}

struct Args {
    std::string algo = "de-gen";
    std::string layout = "highdim";
    std::string mode = "generation";
    size_t n = 1024;          // per-island population rows
    size_t d = 128;
    size_t islands = 1;
    int generations = 50;
    int warmup = 5;
    int reps = 20;
    unsigned long long seed = 42;
    int xthreads = 0;
    int yblock = 0;
    int hd_cap = (int)V2_HD_MAX_BLOCKS_PER_ROW;   // highdim blocks-per-row cap
    int defer_rows = 64;      // use ParallelRestore when nRows <= defer_rows
    bool defer_always = false;
    bool defer_layout = false; // layout-aware: highdim always defers
    bool sel_prepass = false;  // generation: precompute parent indices in a tiny kernel
    bool no_sel = false;
    bool check = false;
    std::string out;
    std::string dump_fit;
    std::string dump_times;
    std::string trace;          // per-generation best/mean fitness (runs --generations)
    std::string fitness = "sphere";
    unsigned long long shift_seed = 1;  // shifted functions: seed of the generated shift
    unsigned long long kp_seed = 1;     // knapsack: seed of the generated instance
    std::string kp_type = "strong";     // knapsack: uncorrelated | strong (correlated)
    std::string shift_file;             // shifted functions: D float32 values instead
    double lo = NAN, hi = NAN;  // gene bounds of the real-coded operators
    double check_tol = 1e-2;    // --check: relative tolerance of the recomputed fitness
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (const char* er = std::getenv("DEFER_ROWS")) a.defer_rows = std::atoi(er);
    if (const char* ea = std::getenv("DEFER_ALWAYS")) a.defer_always = std::atoi(ea) != 0;
    if (const char* el = std::getenv("DEFER_LAYOUT")) a.defer_layout = std::atoi(el) != 0;
    if (const char* sp = std::getenv("SEL_PREPASS")) a.sel_prepass = std::atoi(sp) != 0;
    auto next = [&](int& i, const char* f) -> std::string {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", f); std::exit(1); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--algo") a.algo = next(i, "--algo");
        else if (arg == "--layout") a.layout = next(i, "--layout");
        else if (arg == "--mode") a.mode = next(i, "--mode");
        else if (arg == "--n") a.n = std::stoul(next(i, "--n"));
        else if (arg == "--islands") a.islands = std::stoul(next(i, "--islands"));
        else if (arg == "--d") a.d = std::stoul(next(i, "--d"));
        else if (arg == "--generations") a.generations = std::stoi(next(i, "--generations"));
        else if (arg == "--warmup") a.warmup = std::stoi(next(i, "--warmup"));
        else if (arg == "--reps") a.reps = std::stoi(next(i, "--reps"));
        else if (arg == "--seed") a.seed = std::stoull(next(i, "--seed"));
        else if (arg == "--xthreads") a.xthreads = std::stoi(next(i, "--xthreads"));
        else if (arg == "--yblock") a.yblock = std::stoi(next(i, "--yblock"));
        else if (arg == "--hd-cap") a.hd_cap = std::stoi(next(i, "--hd-cap"));
        else if (arg == "--defer-rows") a.defer_rows = std::stoi(next(i, "--defer-rows"));
        else if (arg == "--defer-always") a.defer_always = true;
        else if (arg == "--defer-layout") a.defer_layout = true;
        else if (arg == "--sel-prepass") a.sel_prepass = true;
        else if (arg == "--no-sel") a.no_sel = true;
        else if (arg == "--check") a.check = true;
        else if (arg == "--out") a.out = next(i, "--out");
        else if (arg == "--dump-fit") a.dump_fit = next(i, "--dump-fit");
        else if (arg == "--dump-times") a.dump_times = next(i, "--dump-times");
        else if (arg == "--trace") a.trace = next(i, "--trace");
        else if (arg == "--fitness") a.fitness = next(i, "--fitness");
        else if (arg == "--shift-seed") a.shift_seed = std::stoull(next(i, "--shift-seed"));
        else if (arg == "--shift-file") a.shift_file = next(i, "--shift-file");
        else if (arg == "--kp-seed") a.kp_seed = std::stoull(next(i, "--kp-seed"));
        else if (arg == "--kp-type") a.kp_type = next(i, "--kp-type");
        else if (arg == "--lb") a.lo = std::stod(next(i, "--lb"));
        else if (arg == "--ub") a.hi = std::stod(next(i, "--ub"));
        else if (arg == "--check-tol") a.check_tol = std::stod(next(i, "--check-tol"));
        else { std::fprintf(stderr, "unknown arg: %s\n", arg.c_str()); std::exit(1); }
    }
    return a;
}

static bool is_vec4(const std::string& l) { return l == "individual_vec4" || l == "highdim_vec4"; }
static bool is_highdim(const std::string& l) { return l == "highdim" || l == "highdim_vec4"; }
static unsigned pow2_floor(unsigned x) { unsigned p = 1; while (p * 2 <= x) p *= 2; return p; }

template <bool VEC4, bool HIGHDIM, int MODE,
          typename SEL, typename REP, typename CROSS, typename MUT, typename FIT>
static int run(const Args& a) {
    using T = typename CROSS::value_type;
    constexpr bool binary = std::is_same_v<T, uint32_t>;
    constexpr size_t NO = CROSS::NUM_OFFSPRING;
    constexpr size_t NP = CROSS::NUM_PARENTS;

    const size_t islands = a.islands;
    if (islands == 0 || islands > 65535) { std::fprintf(stderr, "islands must be in [1,65535]\n"); return 2; }
    if (a.n == 0 || a.n % NO != 0 || a.n / NO > 65535) {
        std::fprintf(stderr, "n must give nCouples = n/%zu in [1,65535]\n", NO);
        return 2;
    }
    const size_t nCouples = a.n / NO;
    if constexpr (binary) if (a.d % 32 != 0) { std::fprintf(stderr, "binary needs D %% 32 == 0\n"); return 2; }
    if constexpr (VEC4) {
        const size_t req = binary ? 128 : 4;
        if (a.d % req != 0) { std::fprintf(stderr, "vec4 needs D %% %zu == 0\n", req); return 2; }
    }

    const size_t rowsPI = NO * nCouples;
    const size_t nRows = islands * rowsPI;
    const size_t nUnits = VEC4 ? (binary ? a.d / 128 : a.d / 4) : (binary ? a.d / 32 : a.d);

    int tx = a.xthreads;
    if (tx == 0) {
        if constexpr (HIGHDIM) tx = (int)pow2_floor(std::min<unsigned>((unsigned)nUnits, 256u));
        else tx = 128;
        if (tx < 1) tx = 1;
    }
    if ((tx & (tx - 1)) != 0) { std::fprintf(stderr, "xthreads must be a power of two\n"); return 2; }
    int ty = (a.yblock > 0) ? a.yblock : 1;
    if (ty < 1 || ty > 32) { std::fprintf(stderr, "yblock must be in [1,32]\n"); return 2; }
    if (tx * ty > 1024) { std::fprintf(stderr, "xthreads*yblock must be <= 1024\n"); return 2; }
    if constexpr (needs_prev_v<FIT>)
        if (tx < 32) { std::fprintf(stderr, "%s needs xthreads >= 32 (whole warps)\n", a.fitness.c_str()); return 2; }
    const size_t smem = (size_t)tx * ty * sizeof(float);

    GenerationBuffer<T> buf;
    buf.init(nRows, a.d, (size_t)num_sums_v<FIT>);
    float* dshift = nullptr;
    uint32_t* ditems = nullptr;
    if constexpr (positional_v<FIT> && binary) {
        if (a.kp_type != "strong" && a.kp_type != "uncorrelated") {
            std::fprintf(stderr, "--kp-type must be strong or uncorrelated\n"); return 2;
        }
        g_kp = make_knapsack(a.d, a.kp_seed, a.kp_type == "strong");
        cudaMalloc(&ditems, a.d * sizeof(uint32_t));
        cudaMemcpy(ditems, g_kp.items.data(), a.d * sizeof(uint32_t), cudaMemcpyHostToDevice);
        FitParams fp{nullptr, 0.0f, ditems, (float)g_kp.capacity, (float)g_kp.rho};
        cudaMemcpyToSymbol(c_fit, &fp, sizeof(fp));
    } else if constexpr (positional_v<FIT>) {
        if (!a.shift_file.empty()) {
            // .txt: the suite's xopt files (comma- or newline-separated decimals); otherwise
            // D raw float32 values.
            g_shift.assign(a.d, 0.0f);
            const bool text = a.shift_file.size() > 4 && a.shift_file.substr(a.shift_file.size() - 4) == ".txt";
            FILE* f = std::fopen(a.shift_file.c_str(), text ? "r" : "rb");
            size_t got = 0;
            if (f && text) { double v; while (got < a.d && std::fscanf(f, " %lf ,", &v) == 1) g_shift[got++] = (float)v; }
            else if (f) got = std::fread(g_shift.data(), sizeof(float), a.d, f);
            if (f) std::fclose(f);
            if (got != a.d) {
                std::fprintf(stderr, "cannot read %zu shift values from %s\n", a.d, a.shift_file.c_str());
                return 2;
            }
        } else {
            g_shift = make_shift(a.d, a.lo, a.hi, a.shift_seed);
        }
        cudaMalloc(&dshift, a.d * sizeof(float));
        cudaMemcpy(dshift, g_shift.data(), a.d * sizeof(float), cudaMemcpyHostToDevice);
        FitParams fp{dshift, (float)(1.0 / (double)(a.d > 1 ? a.d - 1 : 1)), nullptr, 0.0f, 0.0f};
        cudaMemcpyToSymbol(c_fit, &fp, sizeof(fp));
    }
    T* dlb = nullptr; T* dub = nullptr;
    cudaMalloc(&dlb, a.d * sizeof(T));
    cudaMalloc(&dub, a.d * sizeof(T));
    {
        std::vector<T> lb(a.d, (T)0), ub(a.d, (T)0);
        if constexpr (!binary) for (size_t j = 0; j < a.d; ++j) { lb[j] = (T)a.lo; ub[j] = (T)a.hi; }
        cudaMemcpy(dlb, lb.data(), a.d * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dub, ub.data(), a.d * sizeof(T), cudaMemcpyHostToDevice);
    }
    {
        const size_t total = nRows * buf.elems(a.d);
        const unsigned threads = 256;
        const size_t nvec = (total + 3) / 4;
        const unsigned blocks = (unsigned)std::min<size_t>((nvec + threads - 1) / threads, (size_t)(1u << 22));
        InitPopulationKernel<T, EP><<<blocks == 0 ? 1u : blocks, threads>>>(buf.pop[0], total, a.seed);
        if constexpr (!binary)
            if (a.lo != 0.0 || a.hi != 1.0)
                ScalePopulation<<<blocks == 0 ? 1u : blocks, threads>>>(buf.pop[0], total, (float)a.lo, (float)a.hi);
    }
    // external parents (island-local rows), indexed gcouple*NP + k
    size_t* dpar = nullptr;
    {
        std::vector<size_t> par(islands * nCouples * NP);
        for (size_t is = 0; is < islands; ++is)
            for (size_t c = 0; c < nCouples; ++c)
                for (size_t k = 0; k < NP; ++k) {
                    size_t v;
                    if (k == 0) v = c;
                    else if (NP == 2) v = c + nCouples;
                    else v = (c + k) % rowsPI;
                    par[NP * (is * nCouples + c) + k] = v;
                }
        cudaMalloc(&dpar, par.size() * sizeof(size_t));
        cudaMemcpy(dpar, par.data(), par.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    }

    size_t* dsel = nullptr;   // selection pre-pass output (generation mode)
    if (a.sel_prepass) cudaMalloc(&dsel, (size_t)islands * nCouples * NP * sizeof(size_t));

    Evaluate<FIT><<<(unsigned)nRows, (unsigned)tx, smem>>>(buf.pop[0], buf.fit[0], nRows, a.d);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::fprintf(stderr, "init error: %s\n", cudaGetErrorString(err)); return 1; }

    auto launch = [&](size_t g, int s, int dd) {
        if constexpr (MODE == 0) {
            // Phase 2: variation + fitness only
            cudaMemsetAsync(buf.fit[1], 0, nRows * sizeof(float));
            if constexpr (HIGHDIM) {
                dim3 grid(v2_hd_grid_x(nUnits, tx, (unsigned)a.hd_cap), (unsigned)((nCouples + ty - 1) / ty), (unsigned)islands);
                dim3 block2((unsigned)tx, (unsigned)ty);
                FusedVariation_HighDim<CROSS, MUT, FIT, EP, VEC4><<<grid, block2, smem>>>(
                    buf.pop[0], buf.pop[1], dlb, dub, dpar, a.d, nCouples, islands,
                    a.seed, (unsigned long long)g, buf.fit[1], CROSS{}, MUT{});
            } else {
                dim3 grid((unsigned)((nCouples + ty - 1) / ty), (unsigned)islands);
                dim3 block2((unsigned)tx, (unsigned)ty);
                FusedVariation_Individual<CROSS, MUT, FIT, EP, VEC4><<<grid, block2, smem>>>(
                    buf.pop[0], buf.pop[1], dlb, dub, dpar, a.d, nCouples, islands,
                    a.seed, (unsigned long long)g, buf.fit[1], CROSS{}, MUT{});
            }
        } else if constexpr (MODE == 1) {
            // Fused single-kernel generation (+ D4 parallel restore)
            int defer;
            if (a.defer_always) defer = 1;
            else if (a.defer_layout && HIGHDIM) defer = 1;
            else defer = (nRows <= (size_t)a.defer_rows) ? 1 : 0;
            if constexpr (HIGHDIM) {
                dim3 grid(v2_hd_grid_x(nUnits, tx, (unsigned)a.hd_cap), (unsigned)((nCouples + ty - 1) / ty), (unsigned)islands);
                dim3 block2((unsigned)tx, (unsigned)ty);
                if constexpr (SEL::has_selection) {
                    if (a.sel_prepass) {
                        const size_t stot = (size_t)islands * nCouples;
                        SelectionKernel<SEL, EP><<<(unsigned)((stot + 255) / 256), 256>>>(
                            buf.fit[s], dsel, nCouples, rowsPI, islands, a.seed, (unsigned long long)g);
                        FusedGeneration_HighDim<NoneSelection, REP, CROSS, MUT, FIT, EP, VEC4><<<grid, block2, smem>>>(
                            buf.pop[s], buf.pop[dd], buf.fit[s], buf.fit[dd], buf.accum, buf.arrival,
                            buf.revert_mask, defer, dlb, dub, dsel, a.d, nCouples, islands, a.seed,
                            (unsigned long long)g, CROSS{}, MUT{});
                    } else {
                        FusedGeneration_HighDim<SEL, REP, CROSS, MUT, FIT, EP, VEC4><<<grid, block2, smem>>>(
                            buf.pop[s], buf.pop[dd], buf.fit[s], buf.fit[dd], buf.accum, buf.arrival,
                            buf.revert_mask, defer, dlb, dub, dpar, a.d, nCouples, islands, a.seed,
                            (unsigned long long)g, CROSS{}, MUT{});
                    }
                } else {
                    FusedGeneration_HighDim<SEL, REP, CROSS, MUT, FIT, EP, VEC4><<<grid, block2, smem>>>(
                        buf.pop[s], buf.pop[dd], buf.fit[s], buf.fit[dd], buf.accum, buf.arrival,
                        buf.revert_mask, defer, dlb, dub, dpar, a.d, nCouples, islands, a.seed,
                        (unsigned long long)g, CROSS{}, MUT{});
                }
            } else {
                dim3 grid((unsigned)((nCouples + ty - 1) / ty), (unsigned)islands);
                dim3 block2((unsigned)tx, (unsigned)ty);
                if constexpr (SEL::has_selection) {
                    if (a.sel_prepass) {
                        const size_t stot = (size_t)islands * nCouples;
                        SelectionKernel<SEL, EP><<<(unsigned)((stot + 255) / 256), 256>>>(
                            buf.fit[s], dsel, nCouples, rowsPI, islands, a.seed, (unsigned long long)g);
                        FusedGeneration_Individual<NoneSelection, REP, CROSS, MUT, FIT, EP, VEC4><<<grid, block2, smem>>>(
                            buf.pop[s], buf.pop[dd], buf.fit[s], buf.fit[dd], buf.accum, buf.arrival,
                            buf.revert_mask, defer, dlb, dub, dsel, a.d, nCouples, islands, a.seed,
                            (unsigned long long)g, CROSS{}, MUT{});
                    } else {
                        FusedGeneration_Individual<SEL, REP, CROSS, MUT, FIT, EP, VEC4><<<grid, block2, smem>>>(
                            buf.pop[s], buf.pop[dd], buf.fit[s], buf.fit[dd], buf.accum, buf.arrival,
                            buf.revert_mask, defer, dlb, dub, dpar, a.d, nCouples, islands, a.seed,
                            (unsigned long long)g, CROSS{}, MUT{});
                    }
                } else {
                    FusedGeneration_Individual<SEL, REP, CROSS, MUT, FIT, EP, VEC4><<<grid, block2, smem>>>(
                        buf.pop[s], buf.pop[dd], buf.fit[s], buf.fit[dd], buf.accum, buf.arrival,
                        buf.revert_mask, defer, dlb, dub, dpar, a.d, nCouples, islands, a.seed,
                        (unsigned long long)g, CROSS{}, MUT{});
                }
            }
            if (defer) {
                const size_t elemsT = buf.elems(a.d);
                const unsigned bpr = (unsigned)std::min<size_t>(
                    256, std::max<size_t>(1, (size_t)65536 / std::max<size_t>(1, nRows)));
                const size_t rtotal = nRows * (size_t)bpr;
                const unsigned rgx = (unsigned)std::min<size_t>(rtotal, (size_t)(1u << 13));
                ParallelRestore<T><<<rgx == 0 ? 1u : rgx, 256>>>(
                    buf.pop[s], buf.pop[dd], buf.revert_mask, elemsT, nRows, bpr);
            }
        } else {
            // Decomposed: selection + fused variation + replacement
            generationStepDecomposed<SEL, REP, CROSS, MUT, FIT, EP, VEC4, HIGHDIM>(
                buf.pop[s], buf.pop[dd], buf.trial, buf.fit[s], buf.fit[dd], buf.accum, dpar,
                dlb, dub, a.d, nCouples, islands, a.seed, (unsigned long long)g, tx, ty, 0);
        }
    };

    std::vector<float> times;
    float prevBest = 1e30f;
    int failures = 0;
    const int nExec = (a.check || !a.trace.empty()) ? a.generations : (a.warmup + a.reps);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // --trace: gen,best,mean of the population fitness; gen 0 is the initial population.
    FILE* trace = a.trace.empty() ? nullptr : std::fopen(a.trace.c_str(), "w");
    auto trace_row = [&](int gen, const float* dfit) {
        std::vector<float> hf(nRows);
        cudaMemcpy(hf.data(), dfit, nRows * sizeof(float), cudaMemcpyDeviceToHost);
        double best = hf[0], sum = 0;
        for (float x : hf) { best = std::min(best, (double)x); sum += x; }
        std::fprintf(trace, "%d,%.9g,%.9g\n", gen, best, sum / (double)nRows);
    };
    if (trace) { std::fprintf(trace, "gen,best,mean\n"); trace_row(0, buf.fit[0]); }
    for (int g = 0; g < nExec; ++g) {
        const int s = (MODE != 0) ? (g & 1) : 0;
        const int dd = (MODE != 0) ? ((g + 1) & 1) : 1;
        cudaEventRecord(e0);
        launch((size_t)g, s, dd);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        err = cudaGetLastError();
        if (err != cudaSuccess) { std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(err)); return 1; }
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        if (g >= a.warmup || a.check) times.push_back(ms * 1000.0f);
        if (trace) trace_row(g + 1, buf.fit[(MODE != 0) ? dd : 1]);

        if (a.check) {
            const int f = (MODE != 0) ? dd : 1;
            std::vector<float> hf(nRows);
            cudaMemcpy(hf.data(), buf.fit[f], nRows * sizeof(float), cudaMemcpyDeviceToHost);
            float best = hf[0]; for (float x : hf) best = std::fmin(best, x);
            if ((MODE != 0) && best > prevBest + 1e-3f * std::fmax(1.0f, std::fabs(prevBest))) {
                std::fprintf(stderr, "FAIL monotone gen %d: %f > %f\n", g, best, prevBest);
                ++failures; break;
            }
            prevBest = best;
            // recompute fitness from the produced population
            const int pr = (MODE != 0) ? dd : 1;
            std::vector<T> hp(nRows * buf.elems(a.d));
            cudaMemcpy(hp.data(), buf.pop[pr], hp.size() * sizeof(T), cudaMemcpyDeviceToHost);
            for (size_t r = 0; r < nRows; ++r) {
                float host = 0;
                if constexpr (binary) {
                    const uint32_t* row = reinterpret_cast<const uint32_t*>(&hp[r * buf.elems(a.d)]);
                    if constexpr (std::is_same_v<FIT, Knapsack>) {
                        double P = 0, W = 0;
                        for (size_t j = 0; j < a.d; ++j)
                            if ((row[j / 32] >> (j % 32)) & 1u) { P += g_kp.p[j]; W += g_kp.w[j]; }
                        host = (float)(-P + g_kp.rho * std::max(0.0, W - g_kp.capacity));
                    } else {
                        for (size_t j = 0; j < buf.elems(a.d); ++j) host += (float)(32 - __builtin_popcount(row[j]));
                    }
                } else {
                    host = host_fitness<FIT>(&hp[r * buf.elems(a.d)], a.d);
                }
                float diff = std::fabs(host - hf[r]);
                float sc = std::fmax(1.0f, std::fabs(host));
                if (diff > (float)a.check_tol * sc) { std::fprintf(stderr, "FAIL fit row %zu: %f vs %f\n", r, hf[r], host); ++failures;
                    if constexpr (!binary) if (std::getenv("DBG_CHECK")) {
                        std::vector<T> hs(nRows * buf.elems(a.d));
                        cudaMemcpy(hs.data(), buf.pop[pr ^ 1], hs.size() * sizeof(T), cudaMemcpyDeviceToHost);
                        std::fprintf(stderr, "  host(src row) = %f; differing genes:", host_fitness<FIT>(&hs[r * buf.elems(a.d)], a.d));
                        int shown = 0; size_t cnt = 0;
                        for (size_t j = 0; j < a.d; ++j) if (hs[r*a.d+j] != hp[r*a.d+j]) { ++cnt; if (j % 4 == 0 && shown++ < 200) std::fprintf(stderr, " %zu", j / 4); }
                        std::fprintf(stderr, " (total %zu)\n", cnt);
                    }
                    break; }
            }
            if (failures) break;
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (trace) std::fclose(trace);

    if (a.check) {
        std::printf(failures ? "check FAIL (%d)\n" : "check PASS\n", failures);
    } else {
        double total = 0; for (float t : times) total += t;
        std::vector<float> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const double med = times.empty() ? 0.0 : sorted[sorted.size() / 2];
        const double mn = times.empty() ? 0.0 : sorted.front();
        const double mean = times.empty() ? 0.0 : total / times.size();

        FILE* out = stdout;
        if (!a.out.empty()) out = std::fopen(a.out.c_str(), "w");
        std::fprintf(out, "op,layout,I,N,D,total,xthreads,yblock,time_us\n");
        std::fprintf(out, "%s,%s,%zu,%zu,%zu,%zu,%d,%d,%.3f\n",
                     a.algo.c_str(), a.layout.c_str(), islands, a.n, a.d, islands * a.n, tx, ty, med);
        if (out != stdout) std::fclose(out);
        (void)mean; (void)mn;
    }

    if (!a.dump_times.empty()) {
        FILE* tp = std::fopen(a.dump_times.c_str(), "w");
        if (tp) { for (float t : times) std::fprintf(tp, "%.6f\n", t); std::fclose(tp); }
    }

    if (!a.dump_fit.empty()) {
        const int f = (MODE != 0) ? (int)(nExec & 1) : 1;
        std::vector<float> hf(nRows);
        cudaMemcpy(hf.data(), buf.fit[f], nRows * sizeof(float), cudaMemcpyDeviceToHost);
        FILE* fp = std::fopen(a.dump_fit.c_str(), "w");
        if (fp) { for (size_t r = 0; r < nRows; ++r) std::fprintf(fp, "%.9g\n", hf[r]); std::fclose(fp); }
    }

    cudaFree(dlb); cudaFree(dub); cudaFree(dpar); cudaFree(dsel); cudaFree(dshift); cudaFree(ditems);
    buf.free_all();
    return failures ? 1 : 0;
}

template <int MODE, typename SEL, typename REP, typename CROSS, typename MUT, typename FIT>
static int runLayout(const Args& a) {
    const bool v4 = is_vec4(a.layout), hd = is_highdim(a.layout);
    if (hd && v4) return run<true, true, MODE, SEL, REP, CROSS, MUT, FIT>(a);
    if (hd)       return run<false, true, MODE, SEL, REP, CROSS, MUT, FIT>(a);
    if (v4)       return run<true, false, MODE, SEL, REP, CROSS, MUT, FIT>(a);
    return run<false, false, MODE, SEL, REP, CROSS, MUT, FIT>(a);
}

template <typename SEL, typename REP, typename CROSS, typename MUT, typename FIT>
static int runMode(int mode, const Args& a) {
    if (mode == 0) return runLayout<0, SEL, REP, CROSS, MUT, FIT>(a);
    if (mode == 1) return runLayout<1, SEL, REP, CROSS, MUT, FIT>(a);
    return runLayout<2, SEL, REP, CROSS, MUT, FIT>(a);
}

// Rastrigin and Rosenbrock: fused generation with the split-genome layouts only
// (the other modes and layouts are not used with them).
template <typename SEL, typename CROSS, typename MUT, typename FIT>
static int runSplitGenome(int mode, const Args& a) {
    if (mode != 1 || !is_highdim(a.layout)) {
        std::fprintf(stderr, "--fitness %s needs --mode generation and a highdim layout\n", a.fitness.c_str());
        return 2;
    }
    if (is_vec4(a.layout)) return run<true, true, 1, SEL, GreedyMin, CROSS, MUT, FIT>(a);
    return run<false, true, 1, SEL, GreedyMin, CROSS, MUT, FIT>(a);
}

template <typename SEL, typename CROSS, typename MUT>
static int runFitness(int mode, const Args& a) {
    if (a.fitness == "rastrigin") return runSplitGenome<SEL, CROSS, MUT, Rastrigin>(mode, a);
    if (a.fitness == "rosenbrock") return runSplitGenome<SEL, CROSS, MUT, Rosenbrock>(mode, a);
    if (a.fitness == "cec-f1") return runSplitGenome<SEL, CROSS, MUT, CecF1>(mode, a);
    if (a.fitness == "cec-f2") return runSplitGenome<SEL, CROSS, MUT, CecF2>(mode, a);
    if (a.fitness == "cec-f3") return runSplitGenome<SEL, CROSS, MUT, CecF3>(mode, a);
    if (a.fitness == "cec-f12") return runSplitGenome<SEL, CROSS, MUT, CecF12>(mode, a);
    if (a.fitness == "sphere-sync") return runSplitGenome<SEL, CROSS, MUT, SphereSync>(mode, a);  // barrier probe
    if (a.fitness == "sphere-prev") return runSplitGenome<SEL, CROSS, MUT, SpherePrev>(mode, a);  // barrier probe
    return runMode<SEL, GreedyMin, CROSS, MUT, Sphere>(mode, a);
}

int main(int argc, char** argv) {
    Args a = parseArgs(argc, argv);
    const int mode = (a.mode == "variation") ? 0
                   : (a.mode == "generation") ? 1
                   : (a.mode == "decomposed") ? 2 : -1;
    if (mode < 0) {
        std::fprintf(stderr, "unknown mode: %s (variation|generation|decomposed)\n", a.mode.c_str());
        return 2;
    }

    if (a.fitness != "sphere" && a.fitness != "rastrigin" && a.fitness != "rosenbrock" &&
        a.fitness != "cec-f1" && a.fitness != "cec-f2" && a.fitness != "cec-f3" && a.fitness != "cec-f12" &&
        a.fitness != "knapsack" &&
        a.fitness != "sphere-sync" && a.fitness != "sphere-prev") {
        std::fprintf(stderr, "unknown fitness: %s (sphere|rastrigin|rosenbrock|cec-f1|cec-f2|cec-f3|cec-f12|knapsack|sphere-sync|sphere-prev)\n",
                     a.fitness.c_str());
        return 2;
    }
    const double dlo = a.fitness == "rastrigin" ? -5.12 : a.fitness == "rosenbrock" ? -2.048
                     : (a.fitness == "cec-f1" || a.fitness == "cec-f12") ? -100.0 : a.fitness == "cec-f2" ? -5.0
                     : a.fitness == "cec-f3" ? -32.0 : 0.0;
    const double dhi = a.fitness == "rastrigin" ? 5.12 : a.fitness == "rosenbrock" ? 2.048
                     : (a.fitness == "cec-f1" || a.fitness == "cec-f12") ? 100.0 : a.fitness == "cec-f2" ? 5.0
                     : a.fitness == "cec-f3" ? 32.0 : 1.0;
    if (std::isnan(a.lo)) a.lo = dlo;
    if (std::isnan(a.hi)) a.hi = dhi;
    if (a.fitness == "knapsack" && a.algo != "ux-bf") {
        std::fprintf(stderr, "knapsack is a binary problem (ux-bf only)\n"); return 2;
    }
    if (a.algo == "de-gen") {
        if (a.no_sel) return runFitness<NoneSelection, DE1bin, NoMutation>(mode, a);
        return runFitness<RandomDonors<4>, DE1bin, NoMutation>(mode, a);
    }
    if (a.algo == "ga-gen") {
        if (a.no_sel) return runFitness<NoneSelection, SBXCrossover, PolynomialMutation>(mode, a);
        return runFitness<TournamentPairMin, SBXCrossover, PolynomialMutation>(mode, a);
    }
    if (a.fitness != "sphere" && a.fitness != "knapsack") {
        std::fprintf(stderr, "ux-bf evaluates OneMax (default) or knapsack\n"); return 2;
    }
    if (a.algo == "ux-bf") {
        if (a.fitness == "knapsack") {
            if (a.no_sel) { std::fprintf(stderr, "knapsack needs selection\n"); return 2; }
            return runSplitGenome<TournamentPairMin, UX, BF, Knapsack>(mode, a);
        }
        if (a.no_sel) return runMode<NoneSelection, GreedyMin, UX, BF, OneMax>(mode, a);
        return runMode<TournamentPairMin, GreedyMin, UX, BF, OneMax>(mode, a);
    }
    std::fprintf(stderr, "unknown algo: %s (de-gen|ga-gen|ux-bf)\n", a.algo.c_str());
    return 2;
}
