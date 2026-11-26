// Stream-independence test for the stateless RNG (make_rng in
// benchmarks/divergence/kernels/common.ipp).
//
// For both engines make_rng supports (Philox4x32, which the benchmark uses for
// every stream, and Pcg32) it draws the first two 128-bit blocks of every stream
// (generation g, stream id) over G generations and a set of ids that includes
// ids 2^16 apart and ids near the operator stream tags (2^32 ...), and checks:
//   (a) all first blocks are distinct;
//   (b) no cross-generation reuse: stream (g+1, id) != stream (g, id+1);
//   (c) no within-generation overlap: block 1 of id != block 0 of id + 2^16;
//   (e) the same stream id in consecutive generations is uncorrelated (about
//       half of the bits of its first block agree).
// It also checks (d) that the per-generation key is not a sequence: over 10,000
// consecutive generations the keys are distinct, consecutive keys differ in about
// half of their 64 bits, and their differences are not constant; and (f) that
// couple streams never replay gene streams. The kernels number couple streams
// tag + i and gene streams tag + i * nUnits + u, so the same integer names a couple
// and a gene stream; with Philox on both sides they must still differ.
// The same checks run on the previous make_rng (kept verbatim below as
// make_rng_legacy) and must FAIL there, which shows the test detects the bugs.
//
// Build and run (inside the blackwell_env conda environment), from the repo root:
//   make test_rng_streams
// or
//   nvcc -std=c++20 -O3 -arch=sm_120 -DTARGET_SM=1200 \
//        -Iexternal/mathdx/25.12/include/ -Ibenchmarks/divergence/kernels \
//        -o bin/test_rng_streams benchmarks/step_bench/cpp/tests/test_rng_streams.cu
//   ./bin/test_rng_streams
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "common.ipp"

using namespace evoopt::bench;

// The previous scheme, kept only to demonstrate that the checks catch it.
template <typename RandomEngine>
__device__ inline RandomEngine make_rng_legacy(unsigned long long seed,
                                               unsigned long long gen_offset,
                                               unsigned long long id) {
    return RandomEngine(seed,
                        (gen_offset + id) % RNG_SUBSEQUENCES,
                        (gen_offset + id) / RNG_SUBSEQUENCES);
}

template <typename E> __device__ inline uint4 block(E& e) { return e.next_uint4(); }

template <typename E, bool LEGACY>
__global__ void draw(const unsigned long long* gens, const unsigned long long* ids, size_t n,
                     unsigned long long seed, uint4* b0, uint4* b1) {
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    E e = LEGACY ? make_rng_legacy<E>(seed, gens[k], ids[k]) : make_rng<E>(seed, gens[k], ids[k]);
    b0[k] = block(e);
    b1[k] = block(e);
}

using Key = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
static Key K(uint4 v) { return {v.x, v.y, v.z, v.w}; }

template <typename E, bool LEGACY>
static bool run(const char* name) {
    const unsigned long long G = 4, seed = 42, TAG = 0x200000000ULL, SH = 65536;
    std::vector<unsigned long long> base;
    for (unsigned long long i = 0; i < 4096; ++i) base.push_back(i);            // consecutive ids
    for (unsigned long long i = 0; i < 4096; ++i) base.push_back(SH + i);       // ids 2^16 apart
    for (unsigned long long i = 0; i < 4096; ++i) base.push_back(TAG + i);      // operator tag range
    std::vector<unsigned long long> gens, ids;
    for (unsigned long long g = 0; g < G; ++g)
        for (auto id : base) { gens.push_back(g); ids.push_back(id); }
    const size_t n = ids.size();

    unsigned long long *dg, *di; uint4 *d0, *d1;
    cudaMalloc(&dg, n * 8); cudaMalloc(&di, n * 8); cudaMalloc(&d0, n * 16); cudaMalloc(&d1, n * 16);
    cudaMemcpy(dg, gens.data(), n * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(di, ids.data(), n * 8, cudaMemcpyHostToDevice);
    draw<E, LEGACY><<<(unsigned)((n + 255) / 256), 256>>>(dg, di, n, seed, d0, d1);
    std::vector<uint4> b0(n), b1(n);
    cudaMemcpy(b0.data(), d0, n * 16, cudaMemcpyDeviceToHost);
    cudaMemcpy(b1.data(), d1, n * 16, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    cudaFree(dg); cudaFree(di); cudaFree(d0); cudaFree(d1);
    if (err != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(err)); return false; }

    std::map<std::pair<unsigned long long, unsigned long long>, size_t> at;  // (g, id) -> index
    for (size_t k = 0; k < n; ++k) at[{gens[k], ids[k]}] = k;

    std::map<Key, int> seen;
    size_t dup = 0;
    for (size_t k = 0; k < n; ++k) if (seen[K(b0[k])]++) ++dup;

    size_t cross = 0, cross_n = 0;   // (b) stream (g+1, id) == stream (g, id+1)
    for (unsigned long long g = 0; g + 1 < G; ++g)
        for (auto id : base) {
            auto a = at.find({g + 1, id}), b = at.find({g, id + 1});
            if (a == at.end() || b == at.end()) continue;
            ++cross_n;
            if (K(b0[a->second]) == K(b0[b->second])) ++cross;
        }
    size_t over = 0, over_n = 0;     // (c) block 1 of id == block 0 of id + 2^16 (same generation)
    for (unsigned long long g = 0; g < G; ++g)
        for (unsigned long long i = 0; i < 4096; ++i) {
            auto a = at.find({g, i}), b = at.find({g, i + SH});
            if (a == at.end() || b == at.end()) continue;
            ++over_n;
            if (K(b1[a->second]) == K(b0[b->second])) ++over;
        }
    unsigned long long agree = 0, bits = 0;   // (e) same id, generations g and g+1
    for (unsigned long long g = 0; g + 1 < G; ++g)
        for (auto id : base) {
            const uint4 a = b0[at[{g, id}]], b = b0[at[{g + 1, id}]];
            agree += 128 - (__builtin_popcount(a.x ^ b.x) + __builtin_popcount(a.y ^ b.y)
                          + __builtin_popcount(a.z ^ b.z) + __builtin_popcount(a.w ^ b.w));
            bits += 128;
        }
    const double frac = (double)agree / (double)bits;
    const bool ok = dup == 0 && cross == 0 && over == 0 && frac > 0.49 && frac < 0.51;
    std::printf("%-26s streams %zu | (a) duplicate first blocks %zu | (b) cross-generation reuse %zu/%zu"
                " | (c) within-generation overlap %zu/%zu | (e) bit agreement g vs g+1 %.4f -> %s\n",
                name, n, dup, cross, cross_n, over, over_n, frac, ok ? "PASS" : "FAIL");
    return ok;
}

// (f) couple stream id vs gene stream id with the same integer value, as the
// kernels produce them (couple i == gene u = i of couple 0, tag included).
template <bool TAGGED>
__global__ void draw_couple_gene(const unsigned long long* ids, size_t n, unsigned long long seed,
                                 unsigned long long gen, uint4* bc, uint4* bg) {
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    using P = PairPhiloxPhilox;
    auto c = TAGGED ? P::make_couple(seed, gen, ids[k]) : make_rng<Philox4x32>(seed, gen, ids[k]);
    auto g = P::make_vector(seed, gen, ids[k]);
    bc[k] = c.next_uint4();
    bg[k] = g.next_uint4();
}

template <bool TAGGED>
static size_t couple_gene_collisions(size_t& n_out) {
    std::vector<unsigned long long> ids;
    for (unsigned long long tag : {0x100000000ULL, 0x200000000ULL, 0x300000000ULL})
        for (unsigned long long i = 0; i < 4096; ++i) ids.push_back(tag + i);
    const size_t n = ids.size();
    unsigned long long* di; uint4 *dc, *dg;
    cudaMalloc(&di, n * 8); cudaMalloc(&dc, n * 16); cudaMalloc(&dg, n * 16);
    cudaMemcpy(di, ids.data(), n * 8, cudaMemcpyHostToDevice);
    draw_couple_gene<TAGGED><<<(unsigned)((n + 255) / 256), 256>>>(di, n, 42, 3, dc, dg);
    std::vector<uint4> c(n), g(n);
    cudaMemcpy(c.data(), dc, n * 16, cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data(), dg, n * 16, cudaMemcpyDeviceToHost);
    cudaFree(di); cudaFree(dc); cudaFree(dg);
    // compare every couple block against every gene block, not only the same id
    std::map<Key, int> genes;
    for (auto v : g) genes[K(v)]++;
    size_t hit = 0;
    for (auto v : c) if (genes.count(K(v))) ++hit;
    n_out = n;
    return hit;
}

static bool check_couple_gene() {
    size_t n = 0;
    const size_t tagged = couple_gene_collisions<true>(n);
    const size_t untagged = couple_gene_collisions<false>(n);
    const bool ok = tagged == 0 && untagged == n;
    std::printf("(f) couple vs gene streams, %zu ids: collisions with COUPLE_STREAM_TAG %zu,"
                " without the tag %zu (must be all) -> %s\n", n, tagged, untagged, ok ? "PASS" : "FAIL");
    return ok;
}

// (d) the key of each generation is a hash, not a sequence.
static bool check_generation_keys() {
    const unsigned long long seed = 42, G = 10000;
    std::map<unsigned long long, int> keys, diffs;
    int min_hd = 64, max_hd = 0;
    double mean_hd = 0;
    for (unsigned long long g = 0; g < G; ++g) {
        const unsigned long long k = generation_key(seed, g), k1 = generation_key(seed, g + 1);
        const int hd = __builtin_popcountll(k ^ k1);
        min_hd = std::min(min_hd, hd); max_hd = std::max(max_hd, hd); mean_hd += hd;
        ++keys[k]; ++diffs[k1 - k];
    }
    mean_hd /= G;
    const bool ok = keys.size() == G && diffs.size() > G - 10 && mean_hd > 31 && mean_hd < 33
                    && min_hd >= 12 && max_hd <= 52;
    std::printf("(d) generation keys g=0..%llu: %zu distinct, %zu distinct consecutive differences,"
                " bits changed between consecutive keys: mean %.2f, min %d, max %d -> %s\n",
                G - 1, keys.size(), diffs.size(), mean_hd, min_hd, max_hd, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = check_generation_keys();
    ok &= check_couple_gene();
    ok &= run<Philox4x32, false>("Philox4x32 make_rng");
    ok &= run<Pcg32, false>("Pcg32 make_rng");
    // The legacy scheme must fail; if it passed, the checks would be blind.
    const bool legacy_philox = run<Philox4x32, true>("Philox4x32 legacy");
    const bool legacy_pcg = run<Pcg32, true>("Pcg32 legacy");
    const bool detects = !legacy_philox && !legacy_pcg;
    std::printf("legacy scheme detected as broken: %s\n", detects ? "yes" : "NO");
    std::printf("%s\n", ok && detects ? "OK" : "FAILED");
    return ok && detects ? 0 : 1;
}
