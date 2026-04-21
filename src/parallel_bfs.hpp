/**
 * @file parallel_bfs.hpp
 * @brief Hybrid BFS combining MPI (inter-process) and OpenMP (intra-process).
 *
 * Algorithm: level-synchronous BFS.
 *   Outer loop  — BFS levels (sequential across levels)
 *   Inner loop  — frontier vertex expansion (#pragma omp parallel for)
 *   Sync        — MPI_Allgatherv to exchange next-frontiers across all ranks
 *   Thread-safe — std::atomic<int> for visited marking (compare_exchange_strong)
 */

#pragma once

#include <vector>
#include <mpi.h>

/**
 * @brief Result returned by one hybrid BFS run.
 *
 * @member parent           BFS parent array (0 = root, -1 = unvisited), 1-indexed.
 * @member level            BFS depth per vertex (-1 = unvisited), 1-indexed.
 * @member edges_traversed  Total edge relaxations performed by this rank.
 * @member num_levels       Number of BFS levels reached (graph diameter to source).
 * @member compute_time_s   Seconds in OpenMP frontier expansion.
 * @member comm_time_s      Seconds in MPI communication calls.
 */
struct BFSResult {
    std::vector<int> parent;
    std::vector<int> level;
    long long        edges_traversed;
    int              num_levels;
    double           compute_time_s;
    double           comm_time_s;
};

/**
 * @brief Run one level-synchronous hybrid BFS from `source`.
 *
 * All ranks must call this function collectively with identical arguments
 * (except lo/hi which are rank-specific).
 *
 * @param comm         MPI communicator.
 * @param offsets      CSR offset array (replicated on all ranks).
 * @param neighbors    CSR neighbor array (replicated on all ranks).
 * @param num_vertices Total graph vertices (1-indexed).
 * @param source       BFS source vertex (1-indexed).
 * @param lo           First vertex owned by this rank (inclusive, 1-indexed).
 * @param hi           One past last vertex owned by this rank (exclusive).
 * @return             BFSResult for this rank.
 */
BFSResult bfs_hybrid(MPI_Comm comm,
                     const std::vector<int>& offsets,
                     const std::vector<int>& neighbors,
                     int num_vertices, int source,
                     int lo, int hi);
