/**
 * @file benchmark.cpp
 * @brief BFS timing harness: 5 trials, discard 1 warmup, MPI_Reduce aggregation,
 *        TEPS computation, and CSV output.
 *
 * CSV schema (appended row):
 *   ranks, threads_per_rank, num_vertices, num_edges,
 *   mean_time_s, stddev_time_s, teps,
 *   mean_compute_s, mean_comm_s,
 *   serial_time_s, speedup
 */

#include "benchmark.hpp"
#include "parallel_bfs.hpp"
#include "serial_bfs.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute mean of a vector of doubles.
 */
static double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

/**
 * @brief Compute population standard deviation (corrected=false).
 */
static double stddev_of(const std::vector<double>& v) {
    double m = mean_of(v);
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - m) * (x - m);
    return std::sqrt(sq_sum / static_cast<double>(v.size()));
}

// ─────────────────────────────────────────────────────────────────────────────

void run_benchmark(MPI_Comm comm,
                   const BenchmarkConfig& cfg,
                   const std::vector<int>& offsets,
                   const std::vector<int>& neighbors,
                   int num_vertices, int num_edges,
                   int lo, int hi) {

    int rank   = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif

    // ── Run trials ──────────────────────────────────────────────────────────
    std::vector<double> times_s;
    std::vector<double> compute_ss;
    std::vector<double> comm_ss;
    long long last_edges = 0;

    times_s.reserve(NUM_TRIALS);
    compute_ss.reserve(NUM_TRIALS);
    comm_ss.reserve(NUM_TRIALS);

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        MPI_Barrier(comm);   // synchronise before each trial

        double t0 = MPI_Wtime();
        BFSResult result = bfs_hybrid(comm, offsets, neighbors,
                                      num_vertices, cfg.source, lo, hi);
        double t1 = MPI_Wtime();

        times_s.push_back(t1 - t0);
        compute_ss.push_back(result.compute_time_s);
        comm_ss.push_back(result.comm_time_s);
        last_edges = result.edges_traversed;
    }

    // ── Discard warmup trials ────────────────────────────────────────────────
    std::vector<double> valid_times   (times_s.begin()    + WARMUP, times_s.end());
    std::vector<double> valid_compute (compute_ss.begin() + WARMUP, compute_ss.end());
    std::vector<double> valid_comm    (comm_ss.begin()    + WARMUP, comm_ss.end());

    double local_mean_time    = mean_of(valid_times);
    double local_stddev_time  = stddev_of(valid_times);
    double local_mean_compute = mean_of(valid_compute);
    double local_mean_comm    = mean_of(valid_comm);

    // ── Cross-rank reduction ─────────────────────────────────────────────────
    // Wall time  : MPI_MAX  (critical path dominates)
    // Compute/Comm: MPI_SUM then divide by nranks (average across ranks)
    double max_time     = 0.0;
    double max_stddev   = 0.0;
    double sum_compute  = 0.0;
    double sum_comm     = 0.0;
    long long total_edges = 0;

    MPI_Reduce(&local_mean_time,    &max_time,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_stddev_time,  &max_stddev,  1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_mean_compute, &sum_compute, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_mean_comm,    &sum_comm,    1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&last_edges,         &total_edges, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    // ── Serial baseline and CSV write (rank 0 only) ──────────────────────────
    if (rank == 0) {
        // Run serial BFS twice: first for warm-up, second for timing
        serial_bfs(offsets, neighbors, num_vertices, cfg.source);
        SerialBFSResult s = serial_bfs(offsets, neighbors, num_vertices, cfg.source);

        double avg_compute = sum_compute / static_cast<double>(nranks);
        double avg_comm    = sum_comm    / static_cast<double>(nranks);
        double teps        = static_cast<double>(total_edges) / max_time;
        double speedup     = s.elapsed_s / max_time;

        // ── Console summary ──────────────────────────────────────────────────
        std::cout << std::string(60, '=') << "\n";
        std::cout << "  Ranks          : " << nranks                          << "\n";
        std::cout << "  Threads/rank   : " << nthreads                        << "\n";
        std::cout << "  Vertices       : " << num_vertices                    << "\n";
        std::cout << "  Edges (undir)  : " << (num_edges / 2)                 << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Mean wall time : " << max_time * 1000.0  << " ms\n";
        std::cout << "  TEPS           : " << teps / 1e6         << " M-TEPS\n";
        std::cout << "  Serial time    : " << s.elapsed_s * 1000 << " ms\n";
        std::cout << std::setprecision(3);
        std::cout << "  Speedup        : " << speedup             << "x\n";
        std::cout << std::setprecision(2);
        std::cout << "  Compute        : " << avg_compute * 1000  << " ms\n";
        std::cout << "  Comm (MPI)     : " << avg_comm * 1000     << " ms\n";
        std::cout << std::string(60, '=') << "\n";

        // ── Write CSV ────────────────────────────────────────────────────────
        std::filesystem::create_directories(cfg.output_dir);
        std::string csv_path = cfg.output_dir + "/benchmark.csv";

        bool write_header = !std::filesystem::exists(csv_path);
        std::ofstream csv(csv_path, std::ios::app);

        if (write_header) {
            csv << "ranks,threads_per_rank,num_vertices,num_edges,"
                   "mean_time_s,stddev_time_s,teps,"
                   "mean_compute_s,mean_comm_s,"
                   "serial_time_s,speedup\n";
        }

        csv << std::fixed << std::setprecision(8);
        csv << nranks          << ","
            << nthreads        << ","
            << num_vertices    << ","
            << (num_edges / 2) << ","
            << max_time        << ","
            << max_stddev      << ","
            << teps            << ","
            << avg_compute     << ","
            << avg_comm        << ","
            << s.elapsed_s     << ","
            << speedup         << "\n";

        csv.close();
        std::cout << "  CSV written to: " << csv_path << "\n";
    }
}
