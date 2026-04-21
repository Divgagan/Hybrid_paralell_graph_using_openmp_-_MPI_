/**
 * @file graph_io.hpp
 * @brief Graph I/O utilities: loads SNAP-format edge lists into a
 *        Compressed Sparse Row (CSR) adjacency representation.
 *
 * SNAP edge-list format:
 *  - Lines beginning with '#' are comments and are skipped.
 *  - Each data line: "src\tdst" (tab-separated, any whitespace accepted).
 *  - Vertex IDs may be 0-indexed or 1-indexed; auto-detected.
 *  - Undirected: both (u→v) and (v→u) arcs are inserted.
 */

#pragma once

#include <string>
#include <vector>

/**
 * @brief Immutable Compressed Sparse Row graph representation.
 *
 * Vertices are stored 1-indexed internally regardless of the source file's
 * indexing convention.
 *
 * @member num_vertices  Total number of vertices.
 * @member num_edges     Total directed arcs stored (2 × undirected edges).
 * @member offsets       Length (num_vertices + 1).  offsets[v] is the index
 *                       in `neighbors` where vertex v's adjacency list starts.
 * @member neighbors     Flat array of all neighbor vertex IDs.
 */
struct CSRGraph {
    int              num_vertices;
    int              num_edges;
    std::vector<int> offsets;    ///< size = num_vertices + 1
    std::vector<int> neighbors;  ///< size = num_edges
};

/**
 * @brief Load a SNAP-format edge-list file and return a CSRGraph.
 *
 * Steps:
 *  1. Skip comment/blank lines.
 *  2. Detect 0-indexed vs 1-indexed vertex IDs and shift to 1-indexed.
 *  3. Insert both directions for undirected semantics.
 *  4. Build CSR (prefix-sum over degree array).
 *  5. Sort each adjacency list for deterministic traversal.
 *
 * @param path  Filesystem path to the edge-list file.
 * @return      Populated CSRGraph.
 * @throws      std::runtime_error if the file cannot be opened or is empty.
 */
CSRGraph load_snap_graph(const std::string& path);

/**
 * @brief Return a pointer to vertex v's adjacency list and its length.
 *
 * @param g       The CSR graph.
 * @param v       1-indexed vertex ID.
 * @param degree  [out] number of neighbors.
 * @return        Pointer into g.neighbors; valid until g is destroyed.
 */
inline const int* get_neighbors(const CSRGraph& g, int v, int& degree) {
    degree = g.offsets[v + 1] - g.offsets[v];
    return g.neighbors.data() + g.offsets[v];
}
