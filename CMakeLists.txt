cmake_minimum_required(VERSION 3.16)
project(HybridBFS CXX)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Find MPI ──────────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)
message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
message(STATUS "MPI include dirs: ${MPI_CXX_INCLUDE_DIRS}")

# ── Find OpenMP ───────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")

# ── Source files ──────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/graph_io.cpp
    src/partition.cpp
    src/parallel_bfs.cpp
    src/serial_bfs.cpp
    src/benchmark.cpp
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(hybrid_bfs ${SOURCES})

target_include_directories(hybrid_bfs PRIVATE
    src
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(hybrid_bfs PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    # Windows / Visual Studio
    target_compile_options(hybrid_bfs PRIVATE
        /W3         # Warning level 3
        /O2         # Optimise for speed
        /openmp     # Enable OpenMP (MSVC built-in)
        /EHsc       # C++ exceptions
        /arch:AVX2  # AVX2 vectorisation
        /GL         # Whole-program optimisation
    )
else()
    # GCC / Clang (Linux, macOS, WSL)
    target_compile_options(hybrid_bfs PRIVATE
        -Wall
        -Wextra
        -O3
        -march=native        # CPU-specific vectorisation + AVX2/AVX-512
        -funroll-loops
        -ffast-math          # Aggressive FP optimisation (safe for BFS ints)
        -falign-functions=32 # Align hot functions to cache-line boundary
        -falign-loops=32     # Align inner loops to avoid cache straddle
    )
endif()

# ── Ensure results directory exists at build time ────────────────────────────
add_custom_command(TARGET hybrid_bfs POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_SOURCE_DIR}/results
    COMMENT "Creating results/ directory"
)

# ── Print build summary ───────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== HybridBFS build summary ===")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  MPI        : ${MPI_CXX_COMPILER}")
message(STATUS "  OpenMP     : ${OpenMP_CXX_FLAGS}")
message(STATUS "================================")
