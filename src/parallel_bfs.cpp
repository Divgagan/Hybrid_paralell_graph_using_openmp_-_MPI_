/**
 * @file parallel_bfs.cpp
 * @brief Hybrid level-synchronous BFS: Optimized with thread-safe Bitmap and two-phase checks.
 *
 * Performance Optimization (v3):
 * 1. Bitmap Visited Set: Uses std::atomic<uint64_t> words to pack 64 vertices per word.
 *    For 2M vertices, this reduces the "visited" structure from 8MB down to ~250KB.
 *    This allows the entire visited set to stay in the CPU's L2 cache.
 *
 * 2. Two-Phase Visited Check:
 *    - Phase 1: load(relaxed) to see if bit is already set.
 *    - Phase 2: fetch_or(relaxed) only if Phase 1 was false.
 *    This prevents "cache-line bouncing" where multiple threads try to write to
 *    the same memory word simultaneously.
 *
 * 3. Static Scheduling: Uses OpenMP static scheduling to minimize tasking overhead.
 */

#include "parallel_bfs.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <vector>
#include <cstdint>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Bitmap helpers
inline uint64_t bit_mask(int v) { return 1ULL << (static_cast<uint64_t>(v) & 63ULL); }
inline size_t word_idx(int v) { return static_cast<size_t>(v) >> 6; }

BFSResult bfs_hybrid(MPI_Comm comm,
                     const std::vector<int>& offsets,
                     const std::vector<int>& neighbors,
                     int num_vertices, int source,
                     int lo, int hi) {

    int rank = 0, nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    assert(source >= 1 && source <= num_vertices);

#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif

    // BFS state (parent/level still use full ints)
    std::vector<int> parent(static_cast<size_t>(num_vertices + 1), -1);
    std::vector<int> level (static_cast<size_t>(num_vertices + 1), -1);

    // BITMAP: 1 bit per vertex. Footprint: (N/64)*8 bytes.
    // For 1.9M vertices, this is ~246 KB.
    size_t num_words = (static_cast<size_t>(num_vertices) >> 6) + 1;
    std::vector<std::atomic<uint64_t>> visited(num_words);
    for (size_t i = 0; i < num_words; ++i) {
        visited[i].store(0, std::memory_order_relaxed);
    }

    long long edges_traversed = 0;
    double compute_time = 0.0, comm_time = 0.0;
    int current_level = 0;

    // Seed source
    std::vector<int> global_frontier;
    global_frontier.reserve(std::max(1024, num_vertices / 100)); // Heuristic
    
    // Mark source in bitmap
    visited[word_idx(source)].store(bit_mask(source), std::memory_order_relaxed);
    parent[source] = 0;
    level[source]  = 0;
    global_frontier.push_back(source);

    // Per-thread buffers (resized once)
    std::vector<std::vector<int>> thread_next(static_cast<size_t>(max_threads));
    for (auto& buf : thread_next) {
        buf.reserve(1024); 
    }

    // ─────────────────────────────────────────────────────────────────────────
    while (!global_frontier.empty()) {
        ++current_level;
        for (auto& buf : thread_next) buf.clear();

        double t_comp = MPI_Wtime();
        long long local_edges = 0;
        const int frontier_size = static_cast<int>(global_frontier.size());

        // Dynamic reservation hint per thread
        const size_t next_hint = std::max(static_cast<size_t>(100), (static_cast<size_t>(frontier_size) * 4) / max_threads);
        for (auto& buf : thread_next) {
            if (buf.capacity() < next_hint) buf.reserve(next_hint);
        }

#pragma omp parallel reduction(+:local_edges)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
#pragma omp for schedule(static) nowait
            for (int idx = 0; idx < frontier_size; ++idx) {
                const int v = global_frontier[idx];
                for (int j = offsets[v]; j < offsets[v + 1]; ++j) {
                    const int w = neighbors[j];
                    ++local_edges;

                    const uint64_t mask = bit_mask(w);
                    const size_t idx_w = word_idx(w);

                    // TWO-PHASE CHECK:
                    // 1. Relaxed load: Is someone likely to have visited w?
                    if (!(visited[idx_w].load(std::memory_order_relaxed) & mask)) {
                        // 2. Atomic claim: Try to be the one who marks it.
                        uint64_t old_bits = visited[idx_w].fetch_or(mask, std::memory_order_relaxed);
                        if (!(old_bits & mask)) {
                            // We won the race for vertex w
                            parent[w] = v;
                            level[w]  = current_level;
                            if (w >= lo && w < hi) {
                                thread_next[tid].push_back(w);
                            }
                        }
                    }
                }
            }
        } // end omp parallel

        edges_traversed += local_edges;
        compute_time += MPI_Wtime() - t_comp;

        // Serialization: merge thread buffers
        std::vector<int> local_next;
        size_t total_local = 0;
        for (const auto& buf : thread_next) total_local += buf.size();
        local_next.reserve(total_local);
        for (auto& buf : thread_next) {
            local_next.insert(local_next.end(), buf.begin(), buf.end());
        }

        double t_comm = MPI_Wtime();
        if (nranks == 1) {
            // Single-node fast path
            if (local_next.empty()) {
                global_frontier.clear();
            } else {
                global_frontier.swap(local_next);
            }
        } else {
            // MPI Multi-node path
            int send_count = static_cast<int>(local_next.size());
            std::vector<int> all_counts(static_cast<size_t>(nranks));
            MPI_Allgather(&send_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);

            int total_next = 0;
            std::vector<int> displs(static_cast<size_t>(nranks));
            for (int r = 0; r < nranks; ++r) {
                displs[r] = total_next;
                total_next += all_counts[r];
            }

            if (total_next == 0) {
                global_frontier.clear();
            } else {
                global_frontier.resize(static_cast<size_t>(total_next));
                MPI_Allgatherv(local_next.data(), send_count, MPI_INT,
                               global_frontier.data(), all_counts.data(), displs.data(), MPI_INT, comm);
            }
        }
        comm_time += MPI_Wtime() - t_comm;
    }

    return BFSResult{std::move(parent), std::move(level), edges_traversed, current_level, compute_time, comm_time};
}
