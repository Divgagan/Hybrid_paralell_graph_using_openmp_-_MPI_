# Complete Project Explanation

## 1. Project Title

Hybrid Parallel Graph Processing using MPI and OpenMP.

The project focuses on implementing and benchmarking Breadth-First Search on a large real-world graph.

## 2. Problem Statement

Graphs are used to model roads, social networks, web links, communication networks, and many other real-world systems.

Breadth-First Search is one of the most important graph algorithms. It is used to find shortest paths in unweighted graphs, graph levels, connected components, and reachability.

For small graphs, serial BFS is enough. But for large graphs with millions of vertices and edges, serial BFS becomes slow. The goal of this project is to speed up BFS using hybrid parallelism.

The project uses:

- MPI for distributed process-level parallelism.
- OpenMP for shared-memory thread-level parallelism.
- C++ for high-performance implementation.
- Streamlit and Plotly for visualization.

## 3. Main Objective

The main objective is to compare serial BFS and hybrid parallel BFS on a large graph and analyze performance using:

- Execution time.
- Speedup.
- TEPS, meaning Traversed Edges Per Second.
- Compute time.
- Communication time.
- Effect of MPI ranks and OpenMP threads.

## 4. Technologies Used

### C++

C++ is used for the core BFS implementation because it gives high performance and direct control over memory layout.

### MPI

MPI is used to run multiple processes. Each process is called a rank.

MPI helps distribute graph traversal work across multiple ranks.

Important MPI calls used:

```cpp
MPI_Init
MPI_Comm_rank
MPI_Comm_size
MPI_Bcast
MPI_Barrier
MPI_Allgather
MPI_Allgatherv
MPI_Reduce
MPI_Finalize
```

### OpenMP

OpenMP is used inside each MPI rank to use multiple CPU threads.

The main OpenMP parallel region is used to expand the BFS frontier.

### CMake

CMake builds the C++ project and links MPI and OpenMP.

### Python Streamlit

Streamlit is used to create the dashboard.

### Plotly

Plotly is used for interactive charts.

### NetworkX

NetworkX is used in the dashboard to draw a small graph preview.

## 5. Dataset

The main dataset is `roadNet-CA` from Stanford SNAP.

It represents the California road network.

Dataset details:

```text
Vertices: about 1.96 million
Edges: about 2.77 million undirected edges
Format: SNAP edge list
```

The project also includes a small test graph:

```text
data/test_graph.txt
```

This is useful for quickly checking whether the program runs correctly.

## 6. Project Folder Structure

```text
project/
|-- src/
|   |-- main.cpp
|   |-- graph_io.hpp
|   |-- graph_io.cpp
|   |-- partition.hpp
|   |-- partition.cpp
|   |-- parallel_bfs.hpp
|   |-- parallel_bfs.cpp
|   |-- serial_bfs.hpp
|   |-- serial_bfs.cpp
|   |-- benchmark.hpp
|   |-- benchmark.cpp
|
|-- dashboard/
|   |-- app.py
|
|-- data/
|   |-- roadNet-CA.txt
|   |-- test_graph.txt
|   |-- README.md
|
|-- results/
|   |-- benchmark.csv
|
|-- CMakeLists.txt
|-- requirements.txt
|-- README.md
```

## 7. Role of Each Source File

### `src/main.cpp`

This is the entry point of the C++ program.

Responsibilities:

- Start MPI.
- Parse command-line arguments.
- Load the graph on rank 0.
- Broadcast graph data to all ranks.
- Validate source vertex.
- Compute vertex ownership range for each rank.
- Call the benchmark function.
- Finalize MPI.

### `src/graph_io.hpp` and `src/graph_io.cpp`

These files load the graph from disk.

Responsibilities:

- Read SNAP edge-list format.
- Ignore comments and invalid lines.
- Detect 0-indexed or 1-indexed vertex IDs.
- Convert graph to 1-indexed internal format.
- Insert reverse edges to make the graph undirected.
- Convert edge list to CSR representation.
- Sort adjacency lists.

### `src/partition.hpp` and `src/partition.cpp`

These files handle graph partitioning.

Responsibilities:

- Divide vertices among MPI ranks.
- Find which rank owns a vertex.
- Build ghost vertex information.

The current BFS mainly uses range ownership and global frontier exchange.

### `src/parallel_bfs.hpp` and `src/parallel_bfs.cpp`

These files implement the hybrid parallel BFS.

Responsibilities:

- Maintain BFS parent and level arrays.
- Maintain visited bitmap.
- Expand the frontier using OpenMP.
- Use atomic operations to avoid duplicate visits.
- Exchange next frontier using MPI.
- Measure compute and communication time.

### `src/serial_bfs.hpp` and `src/serial_bfs.cpp`

These files implement serial BFS.

Responsibilities:

- Run BFS using one process and one thread.
- Use the same CSR graph format.
- Provide baseline runtime for speedup calculation.

### `src/benchmark.hpp` and `src/benchmark.cpp`

These files run experiments.

Responsibilities:

- Run BFS multiple times.
- Discard warm-up trial.
- Average valid trials.
- Reduce timing data across MPI ranks.
- Run serial baseline.
- Calculate TEPS and speedup.
- Write benchmark results to CSV.

### `dashboard/app.py`

This file provides the visualization dashboard.

Responsibilities:

- Read benchmark CSV.
- Display metric cards.
- Plot speedup.
- Plot TEPS.
- Plot compute vs communication time.
- Show optional graph preview.

## 8. Why BFS Is Parallelized By Levels

BFS works level by level.

For example:

```text
level 0: source
level 1: neighbors of source
level 2: neighbors of level 1
```

Vertices within the same level can be processed in parallel because they are independent frontier vertices.

However, level 2 cannot start until level 1 is complete. This is why the algorithm is level-synchronous.

## 9. Serial BFS Explanation

Serial BFS uses a queue.

Steps:

1. Mark source as visited.
2. Push source into queue.
3. Pop one vertex from queue.
4. Visit all unvisited neighbors.
5. Push newly discovered neighbors into queue.
6. Repeat until queue is empty.

The serial BFS stores:

```text
visited[]
parent[]
level[]
queue[]
```

This gives the baseline time.

## 10. Hybrid Parallel BFS Explanation

Hybrid BFS combines MPI and OpenMP.

Each MPI rank gets a range of vertices. Each rank has a copy of the graph, but it only contributes newly discovered vertices from its owned range.

At every BFS level:

1. All ranks start with the same current global frontier.
2. OpenMP threads inside each rank expand frontier vertices in parallel.
3. Threads check neighbors and atomically mark newly discovered vertices.
4. A rank keeps only the discovered vertices that belong to its ownership range.
5. MPI exchanges all local next frontiers.
6. Every rank receives the new global frontier.
7. The next BFS level begins.

This repeats until no rank discovers new vertices.

## 11. Why MPI Is Used

MPI allows the program to use multiple processes.

In a cluster or multi-node environment, each node can run one or more MPI ranks.

MPI is useful because:

- It scales beyond one process.
- It provides communication primitives.
- It lets us model distributed graph processing.

In this project, MPI is mainly used for:

- Broadcasting graph data.
- Synchronizing trials.
- Exchanging frontier data.
- Reducing benchmark results.

## 12. Why OpenMP Is Used

OpenMP is used because each MPI process may have multiple CPU cores available.

Instead of one rank processing the frontier serially, multiple threads process frontier vertices in parallel.

This improves CPU utilization inside each rank.

## 13. Why CSR Format Is Used

CSR stands for Compressed Sparse Row.

Graph adjacency is stored using two arrays:

```text
offsets[]
neighbors[]
```

CSR advantages:

- Memory efficient.
- Good cache locality.
- Fast traversal of neighbors.
- Suitable for large sparse graphs.

Road networks are sparse graphs, so CSR is a good fit.

## 14. Important Data Structures

### `parent[]`

Stores the BFS parent of each vertex.

```text
parent[source] = 0
parent[unvisited] = -1
```

### `level[]`

Stores the BFS depth of each vertex from the source.

```text
level[source] = 0
level[unvisited] = -1
```

### `visited` bitmap

Stores whether a vertex is visited.

The parallel BFS uses atomic 64-bit words.

### `global_frontier`

Stores the current BFS level that all ranks process.

### `local_next`

Stores newly discovered vertices owned by the current rank.

### `thread_next`

Stores each thread's local discoveries before merging.

## 15. Atomic Visited Marking

In parallel BFS, two threads may discover the same vertex at the same time.

To avoid duplicate processing, the project uses atomic operations.

The visited bitmap allows safe claiming of a vertex:

```text
if bit is not set:
    atomically set bit
    if this thread set it first:
        assign parent and level
        add to next frontier
```

This prevents race conditions in the visited state.

## 16. Benchmark Metrics

### Mean Time

Average wall-clock time of valid BFS trials.

### Standard Deviation

Shows variation across trials.

### TEPS

TEPS means Traversed Edges Per Second.

Formula:

```text
TEPS = total_edges_traversed / mean_time_s
```

Higher TEPS means better graph traversal throughput.

### Speedup

Formula:

```text
speedup = serial_time_s / parallel_time_s
```

If speedup is greater than 1, the parallel version is faster than serial.

If speedup is less than 1, overhead is higher than benefit.

### Compute Time

Time spent expanding BFS frontiers.

### Communication Time

Time spent in MPI communication.

## 17. Why Speedup Can Be Less Than 1

Parallel programs have overhead.

Overheads include:

- MPI communication.
- Synchronization.
- Atomic operations.
- Thread management.
- Memory bandwidth pressure.
- Small frontier sizes in road networks.

Road networks often have low average degree, so there may not be enough work per level to fully hide communication overhead.

This is why some configurations may not outperform serial BFS.

## 18. Build Process

The build system is defined in:

```text
CMakeLists.txt
```

Build commands:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..
```

CMake finds:

- MPI
- OpenMP
- C++ compiler

Then it builds the executable:

```text
build/hybrid_bfs
```

## 19. Run Commands

Small graph test:

```bash
OMP_NUM_THREADS=2 mpirun -np 2 ./build/hybrid_bfs --graph data/test_graph.txt --source 1 --output ./results
```

Real graph run:

```bash
OMP_NUM_THREADS=4 mpirun -np 2 ./build/hybrid_bfs --graph data/roadNet-CA.txt --source 1 --output ./results
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

## 20. Dashboard Explanation

The dashboard reads:

```text
results/benchmark.csv
```

It displays:

- Latest rank and thread configuration.
- Best speedup.
- Peak TEPS.
- Speedup vs ranks.
- TEPS by configuration.
- Compute vs communication breakdown.
- Optional graph preview.

This helps evaluate how the parallel algorithm behaves under different configurations.

## 21. End-to-End Execution Summary

End-to-end flow:

```text
User runs mpirun command
        |
MPI starts multiple ranks
        |
Rank 0 parses arguments
        |
Rank 0 loads graph
        |
Graph is converted to CSR
        |
CSR is broadcast to all ranks
        |
Each rank computes its vertex range
        |
Benchmark starts
        |
Hybrid BFS runs 5 times
        |
First run is discarded as warm-up
        |
MPI reduces timing metrics
        |
Rank 0 runs serial BFS
        |
Speedup and TEPS are calculated
        |
CSV row is written
        |
Dashboard reads CSV
        |
Charts are displayed
```

## 22. Evaluation Explanation Script

You can say this during evaluation:

```text
My project is a hybrid parallel BFS implementation for large graph processing. The input graph is a SNAP edge list, and rank 0 loads it from disk. The graph is converted into CSR format because CSR is memory efficient and cache-friendly for sparse graphs. After loading, rank 0 broadcasts the CSR arrays to all MPI ranks.

Each MPI rank owns a contiguous range of vertices. The BFS is level-synchronous, so every iteration processes one frontier level. Inside each rank, OpenMP threads expand the current frontier in parallel. The visited structure is implemented as an atomic bitmap, so if multiple threads discover the same vertex, only one thread successfully marks it and assigns its parent and level.

After each level, every rank has a local next frontier containing the newly discovered vertices it owns. MPI_Allgather and MPI_Allgatherv are used to exchange these local frontiers so every rank receives the next global frontier. The algorithm continues until the frontier becomes empty.

For benchmarking, the project runs BFS five times, discards the first warm-up run, averages the remaining runs, and compares the hybrid BFS time with a serial BFS baseline. It calculates speedup and TEPS, then writes the results to benchmark.csv. Finally, the Streamlit dashboard reads the CSV and visualizes speedup, TEPS, and compute versus communication time.
```

## 23. Questions You May Be Asked

### What is BFS?

BFS is Breadth-First Search. It explores a graph level by level from a source vertex.

### Why use MPI and OpenMP together?

MPI gives process-level parallelism and OpenMP gives thread-level parallelism. Together they allow hybrid parallel execution.

### Why use CSR?

CSR is compact and fast for sparse graph traversal. It stores all neighbors in one flat array, which improves cache locality.

### What is a frontier?

A frontier is the set of vertices at the current BFS level that need to be expanded.

### What is TEPS?

TEPS means Traversed Edges Per Second. It measures graph traversal throughput.

### Why does the program use atomics?

Atomics prevent multiple threads from marking the same vertex as newly visited at the same time.

### Why is `MPI_Allgatherv` used?

Each rank may discover a different number of vertices, so the next frontier size is variable. `MPI_Allgatherv` supports variable receive counts.

### Why is there a serial BFS?

Serial BFS provides a baseline runtime, which is needed to calculate speedup.

### Why can communication time increase with more ranks?

More ranks require more synchronization and frontier exchange, so MPI overhead can increase.

### Why can speedup be low on road networks?

Road networks are sparse and often have small frontiers. The available parallel work may not be enough to overcome communication and synchronization overhead.

## 24. Limitations

- The graph is replicated on every rank, so memory usage increases with graph size.
- Frontier exchange uses all-gather, so communication cost can become high.
- Road networks may not expose enough parallelism at every BFS level.
- Parent and level arrays are stored on every rank.

## 25. Possible Future Improvements

- Store only local graph partitions instead of replicating the full graph.
- Use point-to-point communication instead of all-gather.
- Use direction-optimizing BFS for large frontiers.
- Use better load balancing based on edge counts instead of only vertex counts.
- Add validation comparing serial and parallel BFS levels.
- Add more datasets for stronger scaling analysis.

## 26. Final Short Summary

This project implements BFS for large graphs using hybrid MPI plus OpenMP parallelism. It loads a SNAP graph, converts it to CSR, partitions vertices across MPI ranks, expands BFS frontiers with OpenMP threads, exchanges frontier data with MPI, benchmarks performance against serial BFS, writes results to CSV, and visualizes the results using a Streamlit dashboard.
