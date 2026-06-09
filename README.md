<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Hybrid%20Parallel%20Graph&fontSize=32&fontColor=fff&animation=twinkling&desc=OpenMP%20%2B%20MPI%20High%20Performance%20Computing&descSize=16&descAlignY=75" width="100%"/>

# âš¡ Hybrid Parallel Graph â€” OpenMP + MPI

[![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org)
[![OpenMP](https://img.shields.io/badge/OpenMP-0071C5?style=for-the-badge&logo=intel&logoColor=white)](https://www.openmp.org)
[![MPI](https://img.shields.io/badge/MPI-FF6600?style=for-the-badge&logo=gnu&logoColor=white)](https://www.open-mpi.org)

</div>

## ðŸ“Œ Overview

A **High Performance Computing (HPC)** project implementing parallel graph algorithms using a **hybrid OpenMP + MPI** approach. Developed during the High Performance Computing course, this project demonstrates multi-node, multi-thread parallelism for large-scale graph processing.

## ðŸ—ï¸ Architecture

`
MPI Process 0          MPI Process 1          MPI Process N
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Thread 0    â”‚       â”‚ Thread 0    â”‚       â”‚ Thread 0    â”‚
â”‚ Thread 1    â”‚ â—„â”€â”€â”€â–º â”‚ Thread 1    â”‚ â—„â”€â”€â”€â–º â”‚ Thread 1    â”‚
â”‚ Thread 2    â”‚  MPI  â”‚ Thread 2    â”‚  MPI  â”‚ Thread 2    â”‚
â”‚ Thread N    â”‚       â”‚ Thread N    â”‚       â”‚ Thread N    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    OpenMP                OpenMP                OpenMP
`

## âœ¨ Algorithms Implemented

- ðŸ” **Parallel BFS** â€” Breadth-First Search with OpenMP thread parallelism
- ðŸ›¤ï¸ **Parallel Dijkstra** â€” Shortest path with MPI process distribution
- ðŸŒ² **Parallel MST** â€” Minimum Spanning Tree (Kruskal / Prim)
- ðŸ“Š **Graph Partitioning** â€” Distributed graph across MPI processes

## ðŸš€ Build & Run

`ash
# Compile
make all

# Run with MPI (4 processes) + OpenMP (8 threads each)
export OMP_NUM_THREADS=8
mpirun -np 4 ./parallel_graph input_graph.txt

# Run benchmarks
make benchmark
`

## ðŸ“ˆ Performance Results

| Algorithm | Sequential | OpenMP (8T) | MPI (4P) | Hybrid (4PÃ—8T) |
|---|---|---|---|---|
| BFS | 1.0x | 4.2x | 3.1x | **11.8x** |
| Dijkstra | 1.0x | 3.8x | 2.9x | **10.2x** |

## ðŸ› ï¸ Requirements

- GCC 9+
- OpenMPI 4.0+
- OpenMP 4.5+

## ðŸ‘¤ Author

**Gagan Diwakar** â€” HPC Course Project | [Portfolio](https://portfolio-gagan-nu.vercel.app/)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>