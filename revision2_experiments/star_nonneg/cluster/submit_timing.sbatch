#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=factor_timing
#SBATCH --nodelist=n001
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --output=logs/timing_%j.out
#SBATCH --error=logs/timing_%j.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# The paper's timing column, measured properly: ONE pinned node, all eight
# models on all four datasets, model order RANDOMIZED within each replicate,
# medians reported.
#
# Why this is separate from the 3200-cell statistical sweep:
#   * the sweep's per-cell wall-clock is not comparable across models. Cells are
#     enumerated dataset-major then model-major, so `poisson` cells all run early
#     and `starnnf_sqrt` cells all run last -- any drift in node load over the
#     run biases models systematically.
#   * the sweep spreads over heterogeneous hardware (g-nodes 64 CPU/515G,
#     o001/q001 256+ CPU/1T, m002, ...), so seconds-per-fit mixes CPU
#     generations.
#   * fits, ESS and R-hat are hardware-independent, so the sweep needs no
#     controlled conditions -- only the timing does.
#
# Node choice: pinned to ONE machine so every model is measured on identical
# silicon; --exclusive is deliberately NOT used. Requesting a whole node would
# queue until it fully drains (every candidate had 24-64 CPUs already busy) and
# would block GPU users off a node this job does not use -- the code is pure
# CPU Julia. Residual contention is handled by the benchmark's design instead:
# model order is randomized within each replicate and medians are reported, so
# a noisy neighbour perturbs all models equally rather than biasing one.
#
# Pick the quietest node at submission time (check: sinfo -p general -N -o
# "%12N %9T %5C %8O" and read CPU_LOAD). n001 was at load 0.76 when this was
# written; g003/g009 ~10, m002 ~23, o001 ~212. Record the node in the paper
# caption: absolute seconds are node-dependent, the model RATIOS are not.
#
# Submit from this directory: sbatch submit_timing.sbatch
# ============================================================

set -u
# OpenBLAS sizes its thread pool from the node's physical core count, not the
# cgroup: measured 32 BLAS threads against 2 visible cores, turning a 4 s cell
# into >900 s. Julia's own threads are already 1; this is the separate BLAS pool.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "=== timing benchmark on $(hostname), $(date) ==="
lscpu 2>/dev/null | grep -E "Model name|CPU\(s\):|Thread" | head -3
echo

# single-threaded: the order-statistic models use internal @threads and the
# STAR models do not, so 1 thread is what makes wall-clock comparable
"$HOME/.juliaup/bin/julia" --project=../../.. --threads=1 \
    ../../perf_review/bench_by_dataset.jl 5 80

echo "=== done $(date) ==="
