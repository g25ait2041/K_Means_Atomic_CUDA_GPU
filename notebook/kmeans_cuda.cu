/*
 * kmeans_cuda.cu  --  GPU-Based K-Means Clustering (CUDA)
 * PGD AI (IIT Jodhpur) Project: "GPU-Based K-Means Clustering: Atomic vs. Reduction
 *                  Strategies for Parallel Centroid Update"
 *
 * Two centroid-update strategies are implemented and compared:
 *   MODE_ATOMIC    : every thread atomically adds its point into global
 *                    per-cluster sums (simple, contended, non-deterministic).
 *   MODE_REDUCTION : block-level privatisation in shared memory, then a
 *                    deterministic hierarchical reduction of per-block partials
 *                    (no global atomics on the hot path).
 *
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

enum Mode { MODE_ATOMIC = 0, MODE_REDUCTION = 1 };

static const int BLOCK = 256;            // threads per block (assign/update)
static const int MAX_SHARED_FLOATS = 12000; // 48 KB / 4 bytes, keep margin

/* ------------------------------------------------------------------------ */
/* Kernel 1: assignment step. One thread per point; centroids in shared mem. */
/* ------------------------------------------------------------------------ */
__global__ void assign_kernel(const float* __restrict__ points,
                              const float* __restrict__ centroids,
                              int* __restrict__ assign,
                              int* __restrict__ changed_count,
                              int N, int D, int K, int use_shared)
{
    extern __shared__ float s_cent[];       // K*D floats when use_shared

    const float* cent = centroids;
    if (use_shared) {
        for (int idx = threadIdx.x; idx < K * D; idx += blockDim.x)
            s_cent[idx] = centroids[idx];
        cent = s_cent;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_changed = 0;
    if (i < N) {
        const float* p = points + (size_t)i * D;
        float best = FLT_MAX;
        int best_k = 0;
        for (int k = 0; k < K; ++k) {
            const float* c = cent + k * D;
            float dist = 0.f;
            for (int j = 0; j < D; ++j) {
                float diff = p[j] - c[j];
                dist = fmaf(diff, diff, dist);
            }
            if (dist < best) { best = dist; best_k = k; }
        }
        if (assign[i] != best_k) { assign[i] = best_k; local_changed = 1; }
    }

    /* block-level count of changed assignments, then ONE global atomic */
    local_changed = __syncthreads_count(local_changed);
    if (threadIdx.x == 0 && local_changed > 0) atomicAdd(changed_count, local_changed);
}

/* ------------------------------------------------------------------------ */
/* Kernel 2a: ATOMIC update. Each thread atomically adds its coordinates.   */
/* ------------------------------------------------------------------------ */
__global__ void update_atomic_kernel(const float* __restrict__ points,
                                     const int* __restrict__ assign,
                                     float* __restrict__ sums,
                                     int* __restrict__ counts,
                                     int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int k = assign[i];
    const float* p = points + (size_t)i * D;
    for (int j = 0; j < D; ++j) atomicAdd(&sums[k * D + j], p[j]);
    atomicAdd(&counts[k], 1);
}

/* ------------------------------------------------------------------------ */
/* Kernel 2b: REDUCTION update, stage 1. Block-private accumulation in      */
/* shared memory (shared-memory atomics are on-chip and cheap), partial      */
/* sums written to a per-block slot -> no global atomics at all.             */
/* ------------------------------------------------------------------------ */
__global__ void update_partial_kernel(const float* __restrict__ points,
                                      const int* __restrict__ assign,
                                      float* __restrict__ partial_sums,   // [gridDim.x][K*D]
                                      int* __restrict__ partial_counts,   // [gridDim.x][K]
                                      int N, int D, int K)
{
    extern __shared__ float s_sums[];             // K*D floats
    int* s_counts = (int*)(s_sums + K * D);       // K ints after the floats

    for (int idx = threadIdx.x; idx < K * D; idx += blockDim.x) s_sums[idx] = 0.f;
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) s_counts[idx] = 0;
    __syncthreads();

    /* grid-stride loop: each block owns a contiguous stripe of points */
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        int k = assign[i];
        const float* p = points + (size_t)i * D;
        for (int j = 0; j < D; ++j) atomicAdd(&s_sums[k * D + j], p[j]);
        atomicAdd(&s_counts[k], 1);
    }
    __syncthreads();

    float* out_s = partial_sums + (size_t)blockIdx.x * K * D;
    int*   out_c = partial_counts + (size_t)blockIdx.x * K;
    for (int idx = threadIdx.x; idx < K * D; idx += blockDim.x) out_s[idx] = s_sums[idx];
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) out_c[idx] = s_counts[idx];
}

/* Stage 2: deterministic reduction over blocks. One thread per (k,j) cell. */
__global__ void reduce_partials_kernel(const float* __restrict__ partial_sums,
                                       const int* __restrict__ partial_counts,
                                       float* __restrict__ sums,
                                       int* __restrict__ counts,
                                       int num_blocks, int D, int K)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;     // 0 .. K*D-1
    if (cell < K * D) {
        float acc = 0.f;
        for (int b = 0; b < num_blocks; ++b) acc += partial_sums[(size_t)b * K * D + cell];
        sums[cell] = acc;
    }
    if (cell < K) {
        int acc = 0;
        for (int b = 0; b < num_blocks; ++b) acc += partial_counts[(size_t)b * K + cell];
        counts[cell] = acc;
    }
}

/* ------------------------------------------------------------------------ */
/* Kernel 3: finalise centroids = sums / counts and reduce max shift^2.      */
/* atomicMax on float bits is valid because shifts are non-negative.         */
/* ------------------------------------------------------------------------ */
__global__ void finalize_kernel(float* __restrict__ centroids,
                                const float* __restrict__ sums,
                                const int* __restrict__ counts,
                                float* __restrict__ max_shift_sq,
                                int D, int K)
{
    __shared__ float s_max[BLOCK];
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    float shift = 0.f;
    if (k < K) {
        int c = counts[k];
        if (c > 0) {
            float inv = 1.f / (float)c;
            for (int j = 0; j < D; ++j) {
                float newv = sums[k * D + j] * inv;
                float diff = newv - centroids[k * D + j];
                shift = fmaf(diff, diff, shift);
                centroids[k * D + j] = newv;
            }
        } /* empty cluster: keep previous centroid */
    }
    s_max[threadIdx.x] = shift;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax((int*)max_shift_sq, __float_as_int(s_max[0]));
}

/* ------------------------------------------------------------------------ */
/* Host helpers                                                              */
/* ------------------------------------------------------------------------ */
static bool read_bin(const std::string& path, std::vector<float>& data, int& N, int& D)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    if (fread(&N, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1) { fclose(f); return false; }
    data.resize((size_t)N * D);
    size_t got = fread(data.data(), 4, data.size(), f);
    fclose(f);
    return got == data.size();
}

static void write_bin_f(const std::string& path, const std::vector<float>& data, int N, int D)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(&N, 4, 1, f); fwrite(&D, 4, 1, f);
    fwrite(data.data(), 4, data.size(), f);
    fclose(f);
}

static void write_bin_i(const std::string& path, const std::vector<int>& data, int N)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(&N, 4, 1, f);
    fwrite(data.data(), 4, data.size(), f);
    fclose(f);
}

static void gen_blobs(int N, int D, int K, unsigned seed, std::vector<float>& pts)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uc(-10.f, 10.f);
    std::normal_distribution<float> noise(0.f, 1.0f);
    std::vector<float> centers((size_t)K * D);
    for (auto& v : centers) v = uc(rng);
    pts.resize((size_t)N * D);
    for (int i = 0; i < N; ++i) {
        int k = i % K;
        for (int j = 0; j < D; ++j) pts[(size_t)i * D + j] = centers[k * D + j] + noise(rng);
    }
    /* shuffle rows so clusters are interleaved */
    std::vector<int> perm(N);
    for (int i = 0; i < N; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<float> out((size_t)N * D);
    for (int i = 0; i < N; ++i)
        memcpy(&out[(size_t)i * D], &pts[(size_t)perm[i] * D], D * sizeof(float));
    pts.swap(out);
}

int main(int argc, char** argv)
{
    std::string data_path, init_path, out_assign = "assign.bin", out_cent = "centroids.bin";
    std::string dump_path;
    int K = 0, max_iters = 100, N = 0, D = 0;
    float tol = 1e-4f;
    Mode mode = MODE_REDUCTION;
    unsigned seed = 42;
    bool gen = false, verbose = false;
    int genN = 0, genD = 0;

    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        auto next = [&](const char* what) -> const char* {
            if (a + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(1); }
            return argv[++a];
        };
        if      (s == "--data")       data_path  = next("--data");
        else if (s == "--init")       init_path  = next("--init");
        else if (s == "--k")          K          = atoi(next("--k"));
        else if (s == "--iters")      max_iters  = atoi(next("--iters"));
        else if (s == "--tol")        tol        = (float)atof(next("--tol"));
        else if (s == "--seed")       seed       = (unsigned)atoi(next("--seed"));
        else if (s == "--out-assign") out_assign = next("--out-assign");
        else if (s == "--out-cent")   out_cent   = next("--out-cent");
        else if (s == "--dump")       dump_path  = next("--dump");
        else if (s == "--verbose")    verbose    = true;
        else if (s == "--mode") {
            std::string m = next("--mode");
            mode = (m == "atomic") ? MODE_ATOMIC : MODE_REDUCTION;
        }
        else if (s == "--gen") {
            gen = true;
            genN = atoi(next("--gen N")); genD = atoi(next("D")); K = atoi(next("K"));
        }
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 1; }
    }
    if (K <= 0) { fprintf(stderr, "--k required\n"); return 1; }

    std::vector<float> h_points, h_cent;
    if (gen) {
        N = genN; D = genD;
        gen_blobs(N, D, K, seed, h_points);
        if (!dump_path.empty()) write_bin_f(dump_path, h_points, N, D);
    } else {
        if (!read_bin(data_path, h_points, N, D)) return 1;
    }
    if (!init_path.empty()) {
        int kk, dd;
        if (!read_bin(init_path, h_cent, kk, dd) || kk != K || dd != D) {
            fprintf(stderr, "bad init file\n"); return 1;
        }
    } else {
        /* deterministic init: first K points (same rule as the CPU reference) */
        h_cent.assign(h_points.begin(), h_points.begin() + (size_t)K * D);
    }

    const int use_shared = (K * D <= MAX_SHARED_FLOATS) ? 1 : 0;
    const int grid_assign = (N + BLOCK - 1) / BLOCK;
    int dev = 0; cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    /* for the reduction path use ~8 blocks per SM so partials stay small */
    int grid_partial = prop.multiProcessorCount * 8;
    if (grid_partial > grid_assign) grid_partial = grid_assign;
    const size_t shm_cent  = use_shared ? (size_t)K * D * sizeof(float) : 0;
    const size_t shm_part  = (size_t)K * D * sizeof(float) + (size_t)K * sizeof(int);
    if (mode == MODE_REDUCTION && shm_part > 48 * 1024) {
        fprintf(stderr, "K*D too large for shared-memory privatisation; use --mode atomic\n");
        return 1;
    }

    float *d_points, *d_cent, *d_sums, *d_partial_sums, *d_max_shift;
    int   *d_assign, *d_counts, *d_partial_counts, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_points, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cent,   (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums,   (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assign, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_shift, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_sums,   (size_t)grid_partial * K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_counts, (size_t)grid_partial * K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent,   h_cent.data(),   (size_t)K * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_assign, 0xFF, (size_t)N * sizeof(int)));   // -1 = unassigned

    cudaEvent_t ev0, ev1, evA, evB;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&evA)); CUDA_CHECK(cudaEventCreate(&evB));
    float assign_ms = 0.f, update_ms = 0.f, t;
    const float tol_sq = tol * tol;
    int iters = 0, changed = 0; float max_shift_sq = 0.f;

    if (verbose)
        fprintf(stderr, "[host] gpu=%s SMs=%d | N=%d D=%d K=%d | mode=%s | grid_assign=%d grid_partial=%d BLOCK=%d | shared centroids=%s (%zu B) | tol=%.1e\n",
                prop.name, prop.multiProcessorCount, N, D, K, mode == MODE_ATOMIC ? "atomic" : "reduction",
                grid_assign, grid_partial, BLOCK, use_shared ? "yes" : "no", shm_cent, tol);
    CUDA_CHECK(cudaEventRecord(ev0));
    for (iters = 1; iters <= max_iters; ++iters) {
        CUDA_CHECK(cudaMemsetAsync(d_changed, 0, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_max_shift, 0, sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_sums, 0, (size_t)K * D * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_counts, 0, (size_t)K * sizeof(int)));

        CUDA_CHECK(cudaEventRecord(evA));
        assign_kernel<<<grid_assign, BLOCK, shm_cent>>>(d_points, d_cent, d_assign, d_changed, N, D, K, use_shared);
        CUDA_CHECK(cudaEventRecord(evB));
        CUDA_CHECK(cudaEventSynchronize(evB));
        CUDA_CHECK(cudaEventElapsedTime(&t, evA, evB)); assign_ms += t; float t_assign_iter = t;

        CUDA_CHECK(cudaEventRecord(evA));
        if (mode == MODE_ATOMIC) {
            update_atomic_kernel<<<grid_assign, BLOCK>>>(d_points, d_assign, d_sums, d_counts, N, D);
        } else {
            update_partial_kernel<<<grid_partial, BLOCK, shm_part>>>(d_points, d_assign, d_partial_sums, d_partial_counts, N, D, K);
            int cells = K * D, gr = (cells + BLOCK - 1) / BLOCK;
            reduce_partials_kernel<<<gr, BLOCK>>>(d_partial_sums, d_partial_counts, d_sums, d_counts, grid_partial, D, K);
        }
        finalize_kernel<<<(K + BLOCK - 1) / BLOCK, BLOCK>>>(d_cent, d_sums, d_counts, d_max_shift, D, K);
        CUDA_CHECK(cudaEventRecord(evB));
        CUDA_CHECK(cudaEventSynchronize(evB));
        CUDA_CHECK(cudaEventElapsedTime(&t, evA, evB)); update_ms += t;
        CUDA_CHECK(cudaGetLastError());

        /* ONLY two scalars cross the PCIe bus per iteration */
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&max_shift_sq, d_max_shift, sizeof(float), cudaMemcpyDeviceToHost));
        if (verbose)
            fprintf(stderr, "[iter %3d] assign=%.3fms update=%.3fms changed=%d max_shift=%.3e%s\n",
                    iters, t_assign_iter, t, changed, sqrtf(max_shift_sq),
                    (changed == 0) ? "  => CONVERGED (no change)" : (max_shift_sq < tol_sq) ? "  => CONVERGED (shift < tol)" : "");
        if (changed == 0 || max_shift_sq < tol_sq) break;
    }
    if (iters > max_iters) iters = max_iters;
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float total_ms; CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));

    std::vector<int> h_assign(N);
    CUDA_CHECK(cudaMemcpy(h_assign.data(), d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cent.data(), d_cent, (size_t)K * D * sizeof(float), cudaMemcpyDeviceToHost));

    /* inertia (sum of squared distances) on host for reporting */
    double inertia = 0.0;
    for (int i = 0; i < N; ++i) {
        int k = h_assign[i];
        for (int j = 0; j < D; ++j) {
            double diff = (double)h_points[(size_t)i * D + j] - h_cent[k * D + j];
            inertia += diff * diff;
        }
    }
    write_bin_i(out_assign, h_assign, N);
    write_bin_f(out_cent, h_cent, K, D);

    printf("RESULT mode=%s N=%d D=%d K=%d iters=%d changed_last=%d max_shift=%.3e "
           "time_ms=%.3f assign_ms=%.3f update_ms=%.3f inertia=%.6e shared_centroids=%d gpu=\"%s\"\n",
           mode == MODE_ATOMIC ? "atomic" : "reduction", N, D, K, iters, changed, sqrtf(max_shift_sq),
           total_ms, assign_ms, update_ms, inertia, use_shared, prop.name);

    cudaFree(d_points); cudaFree(d_cent); cudaFree(d_sums); cudaFree(d_assign); cudaFree(d_counts);
    cudaFree(d_changed); cudaFree(d_max_shift); cudaFree(d_partial_sums); cudaFree(d_partial_counts);
    return 0;
}
