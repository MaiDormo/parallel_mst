# HPC Project: Distributed Parallel MST

![MPI](https://img.shields.io/badge/MPI-MPICH%203.2-blue?style=flat-square&logo=mpi)
![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-green?style=flat-square&logo=c)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A highly optimized, hybrid **MPI + OpenMP** implementation of **Boruvka's Algorithm** for finding the Minimum Spanning Tree (MST) on distributed clusters. Designed for speed, scalability, and memory efficiency.

---

## 🚀 Performance Highlights

| Metric | Baseline | Optimized | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Execution (20k Nodes)** | 94.0s | **20.0s** | **4.7x Faster** |
| **I/O Overhead** | 93s | **~1.2s** | **77x Faster** |
| **Memory Access** | Indirect (Ptr-to-Ptr) | **Flat 1D (Contiguous)** | Cache Optimal |

## 📖 Table of Contents

- [HPC Project: Distributed Parallel MST](#hpc-project-distributed-parallel-mst)
  - [🚀 Performance Highlights](#-performance-highlights)
  - [📖 Table of Contents](#-table-of-contents)
  - [✨ Key Features](#-key-features)
  - [🛠 Architecture \& Optimizations](#-architecture--optimizations)
    - [1. Memory Layout](#1-memory-layout)
    - [2. Computation](#2-computation)
    - [3. Communication](#3-communication)
  - [📂 Project Structure](#-project-structure)
  - [⚙️ Build \& Run](#️-build--run)
    - [Prerequisites](#prerequisites)
    - [1. Compile](#1-compile)
    - [2. Generate Data](#2-generate-data)
    - [3. Run Locally](#3-run-locally)
  - [☁️ Cluster Deployment](#️-cluster-deployment)
  - [📊 Analysis \& Benchmarking](#-analysis--benchmarking)
  - [🛣 Future Roadmap](#-future-roadmap)

---

## ✨ Key Features

- **Hybrid Parallelism:** Uses MPI for inter-node communication and OpenMP for intra-node threading.
- **Vectorized Compute:** SIMD-friendly loops utilizing AVX2/AVX-512 instructions.
- **Zero-Copy I/O:** Memory-mapped file reading (`mmap`) for instant graph loading.
- **Automated Tooling:** Complete suite for graph generation, benchmarking, and result plotting.

## 🛠 Architecture & Optimizations

This project moves beyond standard implementations by addressing hardware-level bottlenecks:

### 1. Memory Layout
* **Flattened 1D Arrays:** Replaced `uint16_t**` with `uint16_t*` aligned to 64-byte boundaries. This enables hardware prefetching and eliminates pointer-chasing overhead.
* **Branchless Logic:** Uses a symmetric $N \times N$ matrix to remove `if (row < col)` branching inside hot loops, preventing pipeline flushes.

### 2. Computation
* **SIMD Vectorization:** Inner loops are refactored to be autovectorizable, processing 16-32 edge weights per CPU cycle.
* **Logical Edge Pruning:** Edges connecting vertices within the same component are marked as `MAX_VALUE` to skip expensive union-find lookups in subsequent iterations.
* **Static Scheduling:** OpenMP `schedule(static)` is used to eliminate dynamic scheduling lock contention.

### 3. Communication
* **Bulk Broadcasting:** Replaced row-by-row broadcasting with a single bulk `MPI_Bcast`, saturating the interconnect bandwidth.
* **Tree Flattening:** Explicit path compression before parallel regions ensures $O(1)$ component lookups.

---

## 📂 Project Structure

```bash
├── src/
│   ├── new_main.c                  # Optimized MPI+OpenMP implementation
│   └── main.c                      # Baseline implementation
├── utils/
│   ├── plot_tools.py               # Visualization tools (example)
├── hpc_tools.sh                    # Unified script for cluster/remote usage
├── hpc_tools_local.sh              # Unified script for local usage
├── Makefile                        # Build automation
└── README.md

```

---

## ⚙️ Build & Run

### Prerequisites

* GCC (with OpenMP support)
* MPICH or OpenMPI

### 1. Compile

```bash

# For cluster/remote usage:
./hpc_tools.sh compile

# For local usage:
./hpc_tools_local.sh compile

```

*Compiler Flags Used:* `-O3 -fopenmp -march=native -ftree-vectorize -funroll-loops -std=gnu11`

### 2. Generate Data

```bash

# To generate graphs (cluster/remote):
./hpc_tools.sh generate-graphs

# To generate graphs (local):
./hpc_tools_local.sh generate-graphs

```

### 3. Run Locally

```bash

# To benchmark (cluster/remote):
./hpc_tools.sh benchmark

# To benchmark (local):
./hpc_tools_local.sh benchmark

# To compare implementations (cluster/remote):
./hpc_tools.sh compare

# To compare implementations (local):
./hpc_tools_local.sh compare

```

---

## ☁️ Cluster Deployment

Designed for PBS/Torque clusters (e.g., typically found in HPC environments).

| Command | Description |
| --- | --- |
| `./hpc_tools.sh pbs-job serial|parallel|graph-gen` | Print a PBS job script for the given type |


**PBS job script generation example:**

```bash
# Print a PBS script for a parallel job:
./hpc_tools.sh pbs-job parallel > submit_parallel.pbs
```

---


## 📊 Analysis & Benchmarking

Use the `compare` command in the unified scripts to compare implementations and collect results. You can further analyze output logs or results as needed.

---

## 🛣 Future Roadmap

* [ ] **Sparse Matrix Support:** Switch to CSR (Compressed Sparse Row) format to support N > 100,000 graphs.
* [ ] **Asynchronous MPI:** Implement `MPI_Iallreduce` to overlap communication with computation.
* [ ] **Hierarchical Merging:** Implement a divide-and-conquer strategy for massive scale (>1M nodes).
* [ ] **GPU Offloading:** Port local edge scanning to CUDA/OpenACC.
