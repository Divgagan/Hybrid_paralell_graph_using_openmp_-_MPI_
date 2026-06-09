<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Hybrid%20Parallel%20Graph&fontSize=32&fontColor=fff&animation=twinkling&desc=OpenMP%20%2B%20MPI%20%7C%20High%20Performance%20Computing&descSize=16&descAlignY=75" width="100%"/>

<div align="center">

# Hybrid Parallel Graph -- OpenMP + MPI

[![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org)
[![OpenMP](https://img.shields.io/badge/OpenMP-0071C5?style=for-the-badge&logo=intel&logoColor=white)](https://www.openmp.org)
[![MPI](https://img.shields.io/badge/MPI-FF6600?style=for-the-badge&logo=gnu&logoColor=white)](https://www.open-mpi.org)

</div>

---

## Overview

A **High Performance Computing (HPC)** project implementing parallel graph algorithms using a hybrid **OpenMP + MPI** approach. Developed during the HPC course, demonstrating multi-node, multi-thread parallelism for large-scale graph processing.

---

## Architecture

```
MPI Process 0          MPI Process 1          MPI Process N
+-------------+       +-------------+       +-------------+
| Thread 0    |       | Thread 0    |       | Thread 0    |
| Thread 1    | <---> | Thread 1    | <---> | Thread 1    |
| Thread 2    |  MPI  | Thread 2    |  MPI  | Thread 2    |
| Thread N    |       | Thread N    |       | Thread N    |
+-------------+       +-------------+       +-------------+
   OpenMP                 OpenMP                OpenMP
```

---

## Algorithms Implemented

- **Parallel BFS** - Breadth-First Search with OpenMP thread parallelism
- **Parallel Dijkstra** - Shortest path with MPI process distribution
- **Parallel MST** - Minimum Spanning Tree (Kruskal / Prim)
- **Graph Partitioning** - Distributed graph across MPI processes

---

## Build & Run

```bash
# Compile
make all

# Run with MPI (4 processes) + OpenMP (8 threads each)
export OMP_NUM_THREADS=8
mpirun -np 4 ./parallel_graph input_graph.txt

# Run benchmarks
make benchmark
```

---

## Performance Results

| Algorithm | Sequential | OpenMP (8T) | MPI (4P) | Hybrid (4P x 8T) |
|---|---|---|---|---|
| BFS | 1.0x | 4.2x | 3.1x | **11.8x** |
| Dijkstra | 1.0x | 3.8x | 2.9x | **10.2x** |

---

## Requirements

- GCC 9+
- OpenMPI 4.0+
- OpenMP 4.5+

---

**Author:** [Gagan Diwakar](https://portfolio-gagan-nu.vercel.app/) | HPC Course Project

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>