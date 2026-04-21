# Hybrid Parallel BFS — C++ (MPI + OpenMP)

A production-quality hybrid parallel Breadth-First Search (BFS) implementation
exploiting **two levels of parallelism simultaneously**:

| Level | Technology | Scope |
|-------|------------|-------|
| Inter-process | MPI (`MPI_Allgatherv`) | Across nodes / sockets |
| Intra-process | OpenMP (`#pragma omp parallel for`) | Cores within one MPI process |

Graph: [roadNet-CA](https://snap.stanford.edu/data/roadNet-CA.html) from Stanford SNAP  
(1.96M vertices, 2.77M undirected edges).

---

## Project Structure

```
├── src/
│   ├── main.cpp            # MPI entry point + CLI
│   ├── graph_io.hpp/.cpp   # SNAP loader → CSR graph
│   ├── partition.hpp/.cpp  # Vertex-range partitioning + ghost map
│   ├── parallel_bfs.hpp/.cpp  # Hybrid BFS (MPI_Allgatherv + OpenMP)
│   ├── benchmark.hpp/.cpp  # 5-trial harness, TEPS, CSV output
│   └── serial_bfs.hpp/.cpp # Single-threaded baseline for speedup
├── dashboard/
│   └── app.py              # Streamlit + Plotly results dashboard
├── data/
│   └── README.md           # Graph download instructions
├── results/                # Created at build/runtime; holds benchmark.csv
├── CMakeLists.txt          # CMake build system (MPI + OpenMP)
├── requirements.txt        # Python deps for dashboard
└── README.md               # ← you are here
```

---

## Prerequisites

### C++ compiler + MPI + CMake

**Linux / WSL (recommended):**
```bash
sudo apt update
sudo apt install build-essential cmake libopenmpi-dev openmpi-bin
```

**macOS (Homebrew):**
```bash
brew install cmake open-mpi
```

**Windows native (MS-MPI):**
1. Install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
2. Install [CMake](https://cmake.org/download/) and Visual Studio 2022 (with C++ workload)

### Python (for dashboard)
```bash
pip install -r requirements.txt
```

---

## Build

### Linux / WSL / macOS

```bash
# From the project root:
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
```

### Windows (Developer Command Prompt / PowerShell)

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ..
```

The executable is created at `./build/hybrid_bfs` (Linux) or
`./build/Release/hybrid_bfs.exe` (Windows).

---

## Download the Graph

```bash
# Linux / WSL:
mkdir -p data
wget -P data/ https://snap.stanford.edu/data/roadNet-CA.txt.gz
gunzip data/roadNet-CA.txt.gz

# Windows PowerShell:
New-Item -ItemType Directory -Force -Path data
Invoke-WebRequest -Uri https://snap.stanford.edu/data/roadNet-CA.txt.gz `
    -OutFile data\roadNet-CA.txt.gz
tar -xzf data\roadNet-CA.txt.gz -C data\
```

See `data/README.md` for Python / curl alternatives and smaller test graphs.

---

## Running

### Full hybrid run (4 MPI ranks × 4 OpenMP threads = 16 cores)

```bash
mpirun -np 4 ./build/hybrid_bfs \
    --graph data/roadNet-CA.txt \
    --source 1 \
    --output ./results
```

> Pass `OMP_NUM_THREADS` to control threads per rank:
> ```bash
> OMP_NUM_THREADS=4 mpirun -np 4 ./build/hybrid_bfs --graph data/roadNet-CA.txt
> ```

### Serial baseline (1 rank, 1 thread)

```bash
OMP_NUM_THREADS=1 mpirun -np 1 ./build/hybrid_bfs --graph data/roadNet-CA.txt
```

### Quick smoke test (synthetic chain graph)

```bash
# Generate a 1001-vertex chain
python3 -c "
lines = ['# synthetic chain graph']
for i in range(1000):
    lines.append(f'{i}\t{i+1}')
open('data/test_graph.txt','w').write('\n'.join(lines))
"

OMP_NUM_THREADS=2 mpirun -np 2 ./build/hybrid_bfs --graph data/test_graph.txt
```

### Scaling study (sweep ranks × threads)

```bash
for ranks in 1 2 4 8; do
  for threads in 1 2 4; do
    OMP_NUM_THREADS=$threads mpirun -np $ranks \
      ./build/hybrid_bfs --graph data/roadNet-CA.txt
  done
done
```

Results are appended to `./results/benchmark.csv` after each run.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`. Charts update automatically when new CSV rows are added.

---

## Algorithm Overview

```
Level-synchronous Hybrid BFS (C++ / MPI + OpenMP)
──────────────────────────────────────────────────
While global_frontier ≠ ∅:

  ┌── #pragma omp parallel for schedule(dynamic,64) ──────────────┐
  │  For each v in global_frontier:                                │
  │    For each neighbour w of v:                                  │
  │      atomic.compare_exchange_strong(visited[w], 0→1)           │
  │      If w is locally-owned and unvisited → thread_next[tid]    │
  └────────────────────────────────────────────────────────────────┘
  Merge thread_next[0..T-1] → local_next
  MPI_Allgather(send_count → all_counts)
  MPI_Allgatherv(local_next → global_frontier_next)
```

---

## CSV Schema (`results/benchmark.csv`)

| Column | Description |
|--------|-------------|
| `ranks` | MPI process count |
| `threads_per_rank` | OpenMP threads per process |
| `num_vertices` | Graph vertices |
| `num_edges` | Undirected edge count |
| `mean_time_s` | Average BFS wall time (seconds) |
| `stddev_time_s` | Standard deviation across 4 valid trials |
| `teps` | Traversed Edges Per Second |
| `mean_compute_s` | Average compute (OpenMP) phase time |
| `mean_comm_s` | Average MPI communication time |
| `serial_time_s` | Serial BFS wall time |
| `speedup` | `serial_time_s / mean_time_s` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `MPI not found` by CMake | `export MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi` then re-run cmake |
| `OpenMP not found` | Install `libomp-dev` (Ubuntu) or `libomp` (Homebrew) |
| Linker: `undefined reference to MPI_Init` | Ensure you're using `mpicxx` or that CMake found MPI correctly |
| `mpi.h` missing in IDE | Expected — IDE doesn't know MPI path. Build with CMake, not directly from IDE |
| Segfault on large graph | Check `OMP_STACKSIZE=64M` for deep call stacks |
| `speedup < 1` on small graph | Normal — MPI overhead dominates for small graphs / few levels |
| Windows: `hybrid_bfs.exe` not found | Look in `build\Release\` not `build\` |

---

## License

MIT — educational / research use.
