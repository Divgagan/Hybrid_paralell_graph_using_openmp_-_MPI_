/**
 * @file partition.cpp
 * @brief Vertex-range partitioning implementation.
 */

#include "partition.hpp"

#include <algorithm>
#include <cassert>
#include <set>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────

std::pair<int, int> get_local_range(int rank, int nranks, int num_vertices) {
    assert(rank >= 0 && rank < nranks   && "rank out of range");
    assert(nranks >= 1                  && "nranks must be positive");
    assert(num_vertices >= 0            && "num_vertices must be non-negative");

    const int base = num_vertices / nranks;
    const int rem  = num_vertices % nranks;

    // Ranks 0..(rem-1) get one extra vertex
    const int lo = rank * base + std::min(rank, rem) + 1;          // 1-indexed
    const int hi = lo + base + (rank < rem ? 1 : 0);               // exclusive

    assert(lo >= 1 && lo <= hi && hi <= num_vertices + 1 &&
           "Partition boundary violated");

    return {lo, hi};
}

// ─────────────────────────────────────────────────────────────────────────────

int owner_of(int v, int nranks, int num_vertices) {
    assert(v >= 1 && v <= num_vertices && "vertex out of range");

    // Binary search: find r such that lo(r) <= v < hi(r)
    int lo_r = 0, hi_r = nranks - 1;
    while (lo_r < hi_r) {
        int mid = (lo_r + hi_r) / 2;
        auto range = get_local_range(mid, nranks, num_vertices);
        int hi_mid = range.second;
        if (v < hi_mid) {
            hi_r = mid;
        } else {
            lo_r = mid + 1;
        }
    }
    return lo_r;
}

// ─────────────────────────────────────────────────────────────────────────────

GhostInfo build_ghost_map(int rank, int nranks, int num_vertices,
                           const std::vector<int>& offsets,
                           const std::vector<int>& neighbors,
                           int lo, int hi) {
    assert(lo >= 1 && lo <= hi && hi <= num_vertices + 1 &&
           "Invalid local range");

    std::set<int> ghost_set;

    for (int v = lo; v < hi; ++v) {
        // Vertex v is 1-indexed; offsets[v]..offsets[v+1] is its adj list
        for (int j = offsets[v]; j < offsets[v + 1]; ++j) {
            const int w = neighbors[j];
            if (w < lo || w >= hi) {
                ghost_set.insert(w);
            }
        }
    }

    GhostInfo info;
    info.ghost_vertices.reserve(ghost_set.size());
    info.ghost_owner.reserve(ghost_set.size());

    int idx = 0;
    for (int g : ghost_set) {   // already sorted (std::set)
        info.ghost_vertices.push_back(g);
        info.ghost_owner.push_back(owner_of(g, nranks, num_vertices));
        info.ghost_index[g] = idx++;
    }

    return info;
}
