/**
 * @file benchmark.hpp
 * @brief Timing harness for the hybrid parallel BFS.
 *
 * Methodology:
 *  1. Run BFS NUM_TRIALS (5) times.
 *  2. Discard the first WARMUP (1) trials (JIT / first-touch effects).
 *  3. Average trials 2..NUM_TRIALS.
 *  4. Reduce across ranks: wall time via MPI_MAX, compute/comm via MPI_SUM.
 *  5. Compute TEPS = total_edges / mean_time.
 *  6. Compare against serial baseline for speedup.
 *  7. Append one CSV row to <output_dir>/benchmark.csv.
 */

#pragma once

#include <string>
#include <mpi.h>
#include "graph_io.hpp"

/// Total number of BFS trials per benchmark run.
static constexpr int NUM_TRIALS = 5;

/// Number of initial trials to discard as warm-up.
static constexpr int WARMUP = 1;

/**
 * @brief Configuration for one benchmark run.
 *
 * @member graph_path   Path to SNAP edge-list (for metadata only after load).
 * @member source       BFS source vertex (1-indexed).
 * @member output_dir   Directory to write benchmark.csv.
 */
struct BenchmarkConfig {
    std::string graph_path;
    int         source;
    std::string output_dir;
};

/**
 * @brief Run the BFS timing harness and append results to benchmark.csv.
 *
 * All ranks participate; only rank 0 writes the CSV file.
 *
 * @param comm         MPI communicator.
 * @param cfg          Benchmark configuration.
 * @param offsets      CSR offsets (replicated on all ranks).
 * @param neighbors    CSR neighbors (replicated on all ranks).
 * @param num_vertices Total graph vertices.
 * @param num_edges    Total directed arcs (2 × undirected edges).
 * @param lo           This rank's first owned vertex (1-indexed, inclusive).
 * @param hi           This rank's first un-owned vertex (1-indexed, exclusive).
 */
void run_benchmark(MPI_Comm comm,
                   const BenchmarkConfig& cfg,
                   const std::vector<int>& offsets,
                   const std::vector<int>& neighbors,
                   int num_vertices, int num_edges,
                   int lo, int hi);
