/**
 * @file partition.hpp
 * @brief Vertex-range partitioning utilities for distributing graph workload
 *        across MPI ranks.
 *
 * Strategy: rank r owns the contiguous half-open vertex range [lo, hi).
 * Vertices are always 1-indexed throughout the system.
 */

#pragma once

#include <utility>
#include <vector>
#include <unordered_map>

/**
 * @brief Compute the half-open vertex range [lo, hi) owned by `rank`.
 *
 * Distributes `num_vertices` as evenly as possible: base = num_vertices /
 * nranks, and the first (num_vertices % nranks) ranks each get one extra.
 *
 * @param rank         0-indexed MPI rank (0 ≤ rank < nranks).
 * @param nranks       Total number of MPI processes.
 * @param num_vertices Total graph vertices (1-indexed).
 * @return             (lo, hi) — rank owns vertices lo, lo+1, …, hi-1.
 *
 * @note Asserts: 0 ≤ rank < nranks, 1 ≤ lo ≤ hi ≤ num_vertices+1.
 */
std::pair<int, int> get_local_range(int rank, int nranks, int num_vertices);

/**
 * @brief Return the 0-indexed MPI rank that owns vertex v.
 *
 * Uses binary search over partition boundaries — O(log P).
 *
 * @param v            1-indexed vertex ID.
 * @param nranks       Total MPI ranks.
 * @param num_vertices Total vertices.
 * @return             Owning rank (0-indexed).
 */
int owner_of(int v, int nranks, int num_vertices);

/**
 * @brief Ghost-vertex metadata for one MPI rank.
 *
 * Ghost vertices are vertices adjacent to locally-owned vertices but owned
 * by another rank.  They participate in frontier exchange during BFS.
 *
 * @member ghost_vertices  Sorted list of ghost vertex IDs (1-indexed).
 * @member ghost_owner     Parallel array: rank owning ghost_vertices[i].
 * @member ghost_index     Map from global vertex ID to index in ghost_vertices.
 */
struct GhostInfo {
    std::vector<int>              ghost_vertices;
    std::vector<int>              ghost_owner;
    std::unordered_map<int, int>  ghost_index;
};

/**
 * @brief Identify all ghost vertices for the local range [lo, hi).
 *
 * Iterates over the adjacency lists of every locally-owned vertex and
 * collects neighbors outside [lo, hi).
 *
 * @param rank         This rank's 0-indexed ID.
 * @param nranks       Total MPI ranks.
 * @param num_vertices Total graph vertices.
 * @param offsets      CSR offset array.
 * @param neighbors    CSR neighbor array.
 * @param lo           First locally-owned vertex (inclusive, 1-indexed).
 * @param hi           One past last locally-owned vertex (exclusive).
 * @return             Populated GhostInfo.
 */
GhostInfo build_ghost_map(int rank, int nranks, int num_vertices,
                           const std::vector<int>& offsets,
                           const std::vector<int>& neighbors,
                           int lo, int hi);
