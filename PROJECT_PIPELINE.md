# Hybrid Parallel BFS Project Pipeline

This file explains the complete execution pipeline of the project from input graph to benchmark dashboard.

## 1. Project Goal

The project implements Breadth-First Search (BFS) on a large graph using hybrid parallel programming.

- MPI is used for process-level parallelism across ranks.
- OpenMP is used for thread-level parallelism inside each MPI rank.
- A serial BFS baseline is also implemented to compare performance.
- Benchmark results are stored in CSV format and visualized using a Streamlit dashboard.

The main graph used is the Stanford SNAP `roadNet-CA` road network dataset.

## 2. Input Data Pipeline

Input file:

```text
data/roadNet-CA.txt
```

The graph is stored as a SNAP edge-list file:

```text
source_vertex destination_vertex
```

Comment lines start with `#` and are ignored.

The graph loader is implemented in:

```text
src/graph_io.hpp
src/graph_io.cpp
```

Loading steps:

1. Open the graph file.
2. Skip blank lines and comment lines.
3. Read all valid edges.
4. Detect whether vertex IDs are 0-indexed or 1-indexed.
5. Convert the graph to a 1-indexed internal representation.
6. Treat the graph as undirected by inserting both directions:

```text
u -> v
v -> u
```

7. Build a Compressed Sparse Row representation.
8. Sort each adjacency list for deterministic traversal.

## 3. CSR Graph Representation

The project stores the graph in CSR format:

```cpp
struct CSRGraph {
    int num_vertices;
    int num_edges;
    std::vector<int> offsets;
    std::vector<int> neighbors;
};
```

CSR is used because it is compact and cache-friendly.

For a vertex `v`, its neighbors are stored in:

```text
neighbors[offsets[v] ... offsets[v + 1] - 1]
```

Example:

```text
offsets[v]     = start index of v's adjacency list
offsets[v + 1] = one position after the end
degree(v)      = offsets[v + 1] - offsets[v]
```

This avoids storing separate vectors for every vertex, which would use more memory and reduce cache locality.

## 4. Program Startup Pipeline

The executable starts from:

```text
src/main.cpp
```

Command example:

```bash
OMP_NUM_THREADS=4 mpirun -np 2 ./build/hybrid_bfs --graph data/roadNet-CA.txt --source 1 --output ./results
```

Startup steps:

1. `MPI_Init` starts the MPI environment.
2. Each process gets its rank and total number of ranks:

```cpp
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &nranks);
```

3. Rank 0 parses command-line arguments:

```text
--graph   graph file path
--source  BFS starting vertex
--output  output directory
```

4. Rank 0 broadcasts parsed values to all other ranks.
5. Rank 0 loads the graph from disk.
6. Rank 0 broadcasts the CSR graph arrays to every rank:

```text
num_vertices
num_edges
offsets[]
neighbors[]
```

All ranks then have a full copy of the CSR graph.

## 5. Partitioning Pipeline

Partitioning logic is implemented in:

```text
src/partition.hpp
src/partition.cpp
```

Each MPI rank owns a contiguous range of vertices:

```text
[lo, hi)
```

For example, if there are 1000 vertices and 4 ranks:

```text
rank 0 owns vertices 1   to 250
rank 1 owns vertices 251 to 500
rank 2 owns vertices 501 to 750
rank 3 owns vertices 751 to 1000
```

The function:

```cpp
get_local_range(rank, nranks, num_vertices)
```

computes this ownership range.

If the number of vertices does not divide evenly, the first few ranks get one extra vertex.

## 6. Hybrid BFS Pipeline

Hybrid BFS is implemented in:

```text
src/parallel_bfs.hpp
src/parallel_bfs.cpp
```

The BFS is level-synchronous. This means BFS processes one depth level at a time:

```text
level 0: source vertex
level 1: all neighbors of source
level 2: all unvisited neighbors of level 1
...
```

The main BFS state includes:

```text
parent[]  = parent of each vertex in BFS tree
level[]   = distance level from source
visited   = bitmap marking visited vertices
frontier  = current BFS level
```

The frontier contains vertices that need to be expanded at the current BFS level.

## 7. OpenMP Thread-Level Work

Inside each MPI rank, OpenMP expands the current frontier in parallel:

```cpp
#pragma omp parallel reduction(+:local_edges)
{
    #pragma omp for schedule(static) nowait
    for (int idx = 0; idx < frontier_size; ++idx) {
        ...
    }
}
```

Each thread:

1. Takes some vertices from the current frontier.
2. Reads their adjacency lists from CSR.
3. Checks each neighbor.
4. Atomically marks unvisited neighbors.
5. Stores newly discovered local vertices in a thread-local buffer.

Thread-local buffers reduce lock contention because threads do not push directly into one shared vector.

## 8. Visited Bitmap Optimization

Instead of using one integer or boolean per vertex, the parallel BFS uses a bitmap:

```cpp
std::vector<std::atomic<uint64_t>> visited;
```

One 64-bit word stores visited information for 64 vertices.

Benefits:

- Less memory usage.
- Better cache locality.
- Atomic updates are possible.
- Multiple threads can safely race to discover vertices.

The code uses a two-phase check:

1. First, read the bitmap word.
2. If the bit is not set, use atomic `fetch_or` to claim the vertex.

Only the thread that successfully sets the bit first becomes responsible for assigning the parent and level.

## 9. MPI Communication Pipeline

After OpenMP expansion, each rank has a local `local_next` frontier.

Each rank only contributes vertices that belong to its owned range:

```text
if w belongs to this rank:
    add w to local_next
```

Then MPI combines all ranks' next frontiers.

Communication steps:

1. Each rank sends the number of discovered vertices:

```cpp
MPI_Allgather(...)
```

2. Ranks compute displacements for receiving variable-length data.
3. All ranks exchange frontier data:

```cpp
MPI_Allgatherv(...)
```

4. Every rank receives the complete next global frontier.
5. BFS continues to the next level.

The loop stops when the global frontier becomes empty.

## 10. Benchmark Pipeline

Benchmarking is implemented in:

```text
src/benchmark.hpp
src/benchmark.cpp
```

Benchmark steps:

1. Run hybrid BFS 5 times.
2. Treat the first run as warm-up.
3. Average the remaining 4 trials.
4. Use `MPI_Barrier` before each trial for synchronization.
5. Use `MPI_Reduce` to combine timing data.
6. Run serial BFS on rank 0 for baseline comparison.
7. Compute performance metrics.
8. Append one result row to:

```text
results/benchmark.csv
```

## 11. Timing Aggregation

The benchmark records:

```text
mean_time_s
stddev_time_s
mean_compute_s
mean_comm_s
serial_time_s
speedup
teps
```

Wall time uses `MPI_MAX` because the slowest rank determines the total parallel runtime.

Compute and communication times are summed across ranks and then averaged.

## 12. Serial Baseline Pipeline

Serial BFS is implemented in:

```text
src/serial_bfs.hpp
src/serial_bfs.cpp
```

It uses the same CSR graph representation as the parallel version.

This makes the comparison fair because the graph format is the same.

The serial BFS uses:

- A normal visited array.
- A pre-allocated vector queue.
- Standard level-based traversal.

Speedup is calculated as:

```text
speedup = serial_time_s / parallel_mean_time_s
```

## 13. Output Pipeline

The benchmark writes results to:

```text
results/benchmark.csv
```

CSV columns:

```text
ranks
threads_per_rank
num_vertices
num_edges
mean_time_s
stddev_time_s
teps
mean_compute_s
mean_comm_s
serial_time_s
speedup
```

This file is the bridge between the C++ benchmark and the Python dashboard.

## 14. Dashboard Pipeline

The dashboard is implemented in:

```text
dashboard/app.py
```

Run it with:

```bash
streamlit run dashboard/app.py
```

The dashboard:

1. Reads `results/benchmark.csv`.
2. Cleans duplicate header rows if present.
3. Converts columns to numeric types.
4. Builds configuration labels such as:

```text
2R x 4T
```

5. Displays metric cards.
6. Plots speedup curves.
7. Plots TEPS bar charts.
8. Plots compute vs communication time.
9. Optionally previews a graph sample using NetworkX and Plotly.

## 15. Complete End-to-End Flow

The full project pipeline is:

```text
SNAP edge-list graph
        |
        v
Rank 0 loads graph
        |
        v
Convert edge list to CSR
        |
        v
Broadcast CSR to all MPI ranks
        |
        v
Partition vertices by rank
        |
        v
Run hybrid BFS
        |
        +--> OpenMP expands frontier inside each rank
        |
        +--> MPI exchanges next frontier across ranks
        |
        v
Repeat until frontier is empty
        |
        v
Run multiple benchmark trials
        |
        v
Compare with serial BFS baseline
        |
        v
Write results/benchmark.csv
        |
        v
Visualize results in Streamlit dashboard
```

## 16. Demo Pipeline For Evaluation

Build:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..
```

Run small test graph:

```bash
OMP_NUM_THREADS=2 mpirun -np 2 ./build/hybrid_bfs --graph data/test_graph.txt --source 1 --output ./results
```

Run real graph:

```bash
OMP_NUM_THREADS=4 mpirun -np 2 ./build/hybrid_bfs --graph data/roadNet-CA.txt --source 1 --output ./results
```

Open dashboard:

```bash
streamlit run dashboard/app.py
```

## 17. One-Minute Pipeline Explanation

This project takes a large SNAP graph as input, loads it into a CSR representation, broadcasts that graph to all MPI ranks, partitions vertices among ranks, and runs level-synchronous BFS. Inside each rank, OpenMP threads expand frontier vertices in parallel. Across ranks, MPI collects and redistributes the next frontier using `MPI_Allgather` and `MPI_Allgatherv`. The BFS is repeated for multiple trials, compared against a serial baseline, and metrics such as time, speedup, TEPS, compute time, and communication time are written to a CSV file. Finally, a Streamlit dashboard reads the CSV and visualizes the performance results.
