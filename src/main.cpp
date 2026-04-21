/**
 * @file main.cpp
 * @brief MPI entry point for the hybrid parallel BFS benchmark.
 *
 * Usage:
 *   mpirun -np <P> ./hybrid_bfs --graph <path> [--source <id>] [--output <dir>]
 *
 * Execution flow:
 *   1. MPI_Init
 *   2. Parse CLI flags (cross-platform: manual argv parsing, no getopt dependency)
 *   3. Rank 0 loads graph from disk
 *   4. Broadcast: num_vertices, num_edges, offsets[], neighbors[]
 *   5. All ranks compute local vertex range via get_local_range()
 *   6. run_benchmark() — warm-up + timed trials + CSV write
 *   7. MPI_Finalize
 *
 * No global mutable state outside the MPI rank context.
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>

#include "graph_io.hpp"
#include "partition.hpp"
#include "benchmark.hpp"

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Parse CLI arguments from argv.
 *
 * Supported flags:
 *   --graph  <path>    Path to SNAP edge-list (required)
 *   --source <int>     BFS source vertex, 1-indexed (default: 1)
 *   --output <dir>     Output directory for CSV (default: ./results)
 *
 * @param argc  Argument count from main().
 * @param argv  Argument vector from main().
 * @param graph_path  [out] Graph file path.
 * @param source      [out] Source vertex.
 * @param output_dir  [out] Output directory.
 * @throws std::invalid_argument if --graph is missing.
 */
static void parse_args(int argc, char** argv,
                       std::string& graph_path,
                       int& source,
                       std::string& output_dir) {
    graph_path  = "";
    source      = 1;
    output_dir  = "./results";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--graph" && i + 1 < argc) {
            graph_path = argv[++i];
        } else if (arg == "--source" && i + 1 < argc) {
            source = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
                "Usage: mpirun -np <P> ./hybrid_bfs \\\n"
                "         --graph <path> [--source <id>] [--output <dir>]\n"
                "\n"
                "  --graph  <path>  SNAP edge-list file (required)\n"
                "  --source <int>   BFS source vertex, 1-indexed (default: 1)\n"
                "  --output <dir>   Output directory for CSV (default: ./results)\n";
            std::exit(0);
        }
    }

    if (graph_path.empty()) {
        throw std::invalid_argument("--graph <path> is required. Use --help for usage.");
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Broadcast a single int from root to all ranks.
 */
static int bcast_int(MPI_Comm comm, int val, int root = 0) {
    MPI_Bcast(&val, 1, MPI_INT, root, comm);
    return val;
}

/**
 * @brief Broadcast a std::vector<int> from root to all ranks.
 *
 * Non-root ranks provide an empty vector; this function resizes and fills it.
 */
static void bcast_intvec(MPI_Comm comm, std::vector<int>& vec,
                          int n, int root = 0) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != root) {
        vec.resize(static_cast<size_t>(n));
    }
    MPI_Bcast(vec.data(), n, MPI_INT, root, comm);
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // ── MPI initialisation ────────────────────────────────────────────────────
    MPI_Init(&argc, &argv);

    MPI_Comm comm   = MPI_COMM_WORLD;
    int      rank   = 0;
    int      nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    // ── Parse CLI ─────────────────────────────────────────────────────────────
    std::string graph_path;
    int         source     = 1;
    std::string output_dir = "./results";

    // Rank 0 parses; errors are printed then all ranks abort cleanly.
    int parse_ok = 1;
    if (rank == 0) {
        try {
            parse_args(argc, argv, graph_path, source, output_dir);
        } catch (const std::exception& ex) {
            std::cerr << "[ERROR] " << ex.what() << "\n";
            parse_ok = 0;
        }
    }
    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, comm);
    if (!parse_ok) {
        MPI_Finalize();
        return 1;
    }

    // Broadcast string arguments (length first, then chars)
    auto bcast_string = [&](std::string& s) {
        int len = (rank == 0) ? static_cast<int>(s.size()) : 0;
        MPI_Bcast(&len, 1, MPI_INT, 0, comm);
        s.resize(static_cast<size_t>(len));
        MPI_Bcast(s.data(), len, MPI_CHAR, 0, comm);
    };
    bcast_string(graph_path);
    bcast_string(output_dir);
    source = bcast_int(comm, source);

    // ── Load graph (rank 0) and broadcast ─────────────────────────────────────
    int              num_vertices = 0;
    int              num_edges    = 0;
    std::vector<int> offsets;
    std::vector<int> neighbors;

    if (rank == 0) {
        try {
            if (rank == 0) std::cout << "[Rank 0] Loading graph: " << graph_path << " ...\n";
            CSRGraph g   = load_snap_graph(graph_path);
            num_vertices = g.num_vertices;
            num_edges    = g.num_edges;
            offsets      = std::move(g.offsets);
            neighbors    = std::move(g.neighbors);
            std::cout << "[Rank 0] V=" << num_vertices
                      << " E=" << (num_edges / 2) << " (undirected)\n";
        } catch (const std::exception& ex) {
            std::cerr << "[ERROR] " << ex.what() << "\n";
            int err = 0;
            MPI_Bcast(&err, 1, MPI_INT, 0, comm);
            MPI_Finalize();
            return 1;
        }
    }

    // Signal all ranks that load succeeded
    {
        int ok = 1;
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    }

    // Broadcast graph dimensions
    num_vertices = bcast_int(comm, num_vertices);
    num_edges    = bcast_int(comm, num_edges);

    // Broadcast CSR arrays
    bcast_intvec(comm, offsets,   num_vertices + 1);
    bcast_intvec(comm, neighbors, num_edges);

    // ── Validate source ───────────────────────────────────────────────────────
    if (source < 1 || source > num_vertices) {
        if (rank == 0) {
            std::cerr << "[ERROR] Source vertex " << source
                      << " out of range [1, " << num_vertices << "]\n";
        }
        MPI_Finalize();
        return 1;
    }

    // ── Compute local vertex range ────────────────────────────────────────────
    auto [lo, hi] = get_local_range(rank, nranks, num_vertices);

    if (rank == 0) {
        std::cout << "[Rank " << rank << "] owns vertices [" << lo
                  << ", " << hi - 1 << "] (" << (hi - lo) << " vertices)\n";
    }

    // ── Run benchmark ─────────────────────────────────────────────────────────
    BenchmarkConfig cfg{graph_path, source, output_dir};
    run_benchmark(comm, cfg, offsets, neighbors, num_vertices, num_edges, lo, hi);

    MPI_Finalize();
    return 0;
}
