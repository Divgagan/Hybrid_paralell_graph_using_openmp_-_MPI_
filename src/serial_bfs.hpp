/**
 * @file serial_bfs.hpp
 * @brief Single-threaded BFS baseline used as the speedup denominator.
 *
 * Operates on the same CSR data structure as the parallel implementation
 * to ensure a fair wall-clock comparison.
 */

#pragma once

#include <vector>

/**
 * @brief Result of a serial BFS run.
 *
 * @member parent           BFS parent array (0 = root, -1 = unvisited).
 * @member level            BFS level per vertex (-1 = unvisited).
 * @member edges_traversed  Total edge relaxations.
 * @member num_levels       Number of BFS levels (depth of search tree).
 * @member elapsed_s        Wall-clock time of the BFS in seconds.
 */
struct SerialBFSResult {
    std::vector<int> parent;
    std::vector<int> level;
    long long        edges_traversed;
    int              num_levels;
    double           elapsed_s;
};

/**
 * @brief Perform a standard iterative BFS from `source` on a CSR graph.
 *
 * Uses a pre-allocated queue (std::vector used as a ring buffer) for
 * cache-friendly traversal.  Timing begins immediately before queue
 * initialization and ends after the last vertex is dequeued.
 *
 * @param offsets      CSR offset array.
 * @param neighbors    CSR neighbor array.
 * @param num_vertices Total vertices (1-indexed).
 * @param source       BFS source vertex (1-indexed).
 * @return             Populated SerialBFSResult.
 */
SerialBFSResult serial_bfs(const std::vector<int>& offsets,
                            const std::vector<int>& neighbors,
                            int num_vertices, int source);
