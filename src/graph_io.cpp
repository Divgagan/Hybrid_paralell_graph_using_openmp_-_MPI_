/**
 * @file graph_io.cpp
 * @brief Implementation: SNAP edge-list → CSR graph.
 */

#include "graph_io.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

CSRGraph load_snap_graph(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open graph file: " + path);
    }

    // ── Pass 1: collect raw edges, find vertex ID range ──────────────────────
    std::vector<std::pair<int, int>> raw_edges;
    raw_edges.reserve(4'000'000);   // initial capacity; will grow if needed

    int min_id = std::numeric_limits<int>::max();
    int max_id = std::numeric_limits<int>::min();

    std::string line;
    while (std::getline(fin, line)) {
        // Skip blank lines and comments
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;   // malformed line

        raw_edges.emplace_back(u, v);
        min_id = std::min({min_id, u, v});
        max_id = std::max({max_id, u, v});
    }
    fin.close();

    if (raw_edges.empty()) {
        throw std::runtime_error("No valid edges found in: " + path);
    }

    // ── Detect 0-indexed vs 1-indexed, shift to 1-indexed ────────────────────
    const int shift      = (min_id == 0) ? 1 : 0;
    const int num_vertices = max_id + shift;  // now 1-indexed max

    // ── Pass 2: count out-degree (both directions, skip self-loops) ───────────
    std::vector<int> degree(static_cast<size_t>(num_vertices + 1), 0);

    for (auto& e : raw_edges) {
        const int u1 = e.first  + shift;
        const int v1 = e.second + shift;
        if (u1 == v1) continue;   // skip self-loops
        degree[u1]++;
        degree[v1]++;
    }

    // ── Build CSR offsets (exclusive prefix sum, 0-indexed into neighbors) ────
    // We use 0-based indexing inside the CSR arrays for simplicity in C++.
    std::vector<int> offsets(static_cast<size_t>(num_vertices + 1), 0);
    // degree[v] holds the degree of 1-indexed vertex v; use degree[i] not degree[i-1]
    for (int i = 1; i <= num_vertices; ++i) {
        offsets[i] = offsets[i - 1] + degree[i];
    }
    const int num_edges = offsets[num_vertices];

    // ── Fill neighbors ────────────────────────────────────────────────────────
    std::vector<int> neighbors(static_cast<size_t>(num_edges), 0);
    // cursor[v-1] must start at offsets[v] (the start of vertex v's adj list)
    std::vector<int> cursor(offsets.begin() + 1, offsets.end());

    for (auto& e : raw_edges) {
        const int u1 = e.first  + shift;
        const int v1 = e.second + shift;
        if (u1 == v1) continue;
        // Convert to 0-indexed vertex IDs for array indexing
        neighbors[cursor[u1 - 1]++] = v1;
        neighbors[cursor[v1 - 1]++] = u1;
    }

    // ── Sort each adjacency list for determinism ──────────────────────────────
    // Vertex v's adj list: [offsets[v], offsets[v+1])
    for (int v = 1; v <= num_vertices; ++v) {
        const int lo = offsets[v];
        const int hi = offsets[v + 1];
        if (hi - lo > 1) {
            std::sort(neighbors.begin() + lo, neighbors.begin() + hi);
        }
    }

    return CSRGraph{num_vertices, num_edges, std::move(offsets), std::move(neighbors)};
}
