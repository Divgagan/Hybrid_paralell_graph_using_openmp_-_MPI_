/**
 * @file serial_bfs.cpp
 * @brief Single-threaded BFS baseline implementation.
 *
 * Uses a pre-allocated std::vector as a flat queue (head/tail indices) for
 * cache-friendly memory access — mirroring the Julia implementation.
 *
 * Timing uses std::chrono::steady_clock for nanosecond resolution.
 */

#include "serial_bfs.hpp"

#include <cassert>
#include <chrono>
#include <vector>

SerialBFSResult serial_bfs(const std::vector<int>& offsets,
                            const std::vector<int>& neighbors,
                            int num_vertices, int source) {

    assert(source >= 1 && source <= num_vertices && "source out of range");

    auto t_start = std::chrono::steady_clock::now();

    // Allocate BFS state (1-indexed; index 0 is unused padding)
    std::vector<int>  parent(static_cast<size_t>(num_vertices + 1), -1);
    std::vector<int>  level (static_cast<size_t>(num_vertices + 1), -1);
    std::vector<bool> visited(static_cast<size_t>(num_vertices + 1), false);

    // Pre-allocated flat queue
    std::vector<int> queue(static_cast<size_t>(num_vertices));
    int q_head = 0;
    int q_tail = 0;

    // Seed
    visited[source] = true;
    parent[source]  = 0;     // root sentinel
    level[source]   = 0;
    queue[q_tail++] = source;

    long long edges_traversed = 0;
    int       current_level   = 0;
    int       level_end       = 0;   // last queue index in current level

    while (q_head < q_tail) {
        const int v  = queue[q_head++];

        for (int j = offsets[v]; j < offsets[v + 1]; ++j) {
            const int w = neighbors[j];
            ++edges_traversed;

            if (!visited[w]) {
                visited[w] = true;
                parent[w]  = v;
                level[w]   = level[v] + 1;
                queue[q_tail++] = w;
            }
        }

        // Advance level counter when current level is exhausted
        if (q_head > level_end && q_head < q_tail) {
            ++current_level;
            level_end = q_tail - 1;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    return SerialBFSResult{
        std::move(parent),
        std::move(level),
        edges_traversed,
        current_level,
        elapsed_s
    };
}
