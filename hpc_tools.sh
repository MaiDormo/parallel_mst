#!/bin/bash

# hpc_tools.sh: Unified script for compilation, graph generation, benchmarking, and PBS job submission
# Usage: ./hpc_tools.sh <command> [args]
# Commands:
#   compile                Compile all C implementations
#   generate-graphs        Generate graphs for all sizes/densities
#   benchmark              Run all benchmarks (serial and parallel)
#   compare                Compare implementations

set -e

PROJECT_DIR="$HOME/parallel_mst"
SRC_DIR="$PROJECT_DIR/src"

function compile_all() {
    echo "Compiling serial implementation..."
    gcc -O3 "$PROJECT_DIR/serial.c" -o "$PROJECT_DIR/serial.o"
    echo "Compiling parallel implementation..."
    mpicc -O3 -fopenmp -march=native -mtune=native -ftree-vectorize -funroll-loops -flto -o "$SRC_DIR/main.o" "$SRC_DIR/main.c"
    echo "Compiling graph generator..."
    gcc -O3 -fopenmp "$PROJECT_DIR/graph_generator.c" -o "$PROJECT_DIR/graph_generator"
    echo "Compilation complete."
}

function generate_graphs() {
    mkdir -p "$PROJECT_DIR/graphs"
    VERTICES=(1000 1500 2000 5000 10000 20000 40000 80000)
    for v in "${VERTICES[@]}"; do
        sparse_edges=$((v * 2))
        medium_edges=$((v * $(echo "l($v)/l(2)" | bc -l | cut -d'.' -f1)))
        dense_edges=$((v * v / 4))
        echo "Generating graphs for $v vertices..."
        "$PROJECT_DIR/graph_generator" $v $sparse_edges "$PROJECT_DIR/graphs/graph_${v}_sparse.txt"
        "$PROJECT_DIR/graph_generator" $v $medium_edges "$PROJECT_DIR/graphs/graph_${v}_medium.txt"
        "$PROJECT_DIR/graph_generator" $v $dense_edges "$PROJECT_DIR/graphs/graph_${v}_dense.txt"
    done
    echo "Graph generation complete!"
}

function benchmark() {
    for graph in "$PROJECT_DIR/graphs/graph_*_*.txt"; do
        num_vertices=$(head -n 1 "$graph" | cut -d' ' -f1)
        echo "============================================"
        echo "Testing $graph..."
        echo "Running serial implementation..."
        "$PROJECT_DIR/serial.o" "$num_vertices" "$graph"
        echo "Running parallel implementation..."
        mpirun -np 2 --bind-to none -x OMP_NUM_THREADS=4 "$SRC_DIR/main.o" "$num_vertices" "$graph"
    done
}

function compare() {
    for graph in "$PROJECT_DIR/graphs/graph_*_*.txt"; do
        num_vertices=$(head -n 1 "$graph" | cut -d' ' -f1)
        echo "============================================"
        echo "Testing $graph..."
        echo "Running serial implementation..."
        serial_output=$("$PROJECT_DIR/serial.o" "$num_vertices" "$graph")
        serial_weight=$(echo "$serial_output" | grep "MST Weight:" | head -n1 | grep -o '[0-9]\+')
        serial_time=$(echo "$serial_output" | grep "Computation Time:" | head -n1 | grep -o '[0-9.]*')
        echo "Running parallel implementation..."
        parallel_output=$(mpirun -np 2 --bind-to none -x OMP_NUM_THREADS=4 "$SRC_DIR/main.o" "$num_vertices" "$graph")
        parallel_weight=$(echo "$parallel_output" | grep "MST Weight:" | head -n1 | grep -o '[0-9]\+')
        parallel_time=$(echo "$parallel_output" | grep "Computation Time:" | head -n1 | grep -o '[0-9.]*')
        echo "Serial: weight=$serial_weight, time=$serial_time"
        echo "Parallel: weight=$parallel_weight, time=$parallel_time"
    done
}