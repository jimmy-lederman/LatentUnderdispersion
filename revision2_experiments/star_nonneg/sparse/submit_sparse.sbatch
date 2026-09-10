#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=sparse_prior
#SBATCH --array=1-20%20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --output=logs/sp_%a_%j.out
#SBATCH --error=logs/sp_%a_%j.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# Sparse-prior computational-cost grid:
#   5 models x 6 concentrations x 10 seeds = 300 cells
#   4 chains x 48000 after 48000 warmup (384k sweeps/cell)
#
# The budget comes from a pilot: STAR-NN needs 192k sweeps to reach q95 R-hat
# <= 1.05 at a = 0.01 with g(x) = x, and MORE than that with g(x) = sqrt(x),
# which is the variant worth reporting (it fits this data better than any other
# model). 384k covers both. ESS/s was measured to be budget-invariant across a
# 4x range, so the metric does not depend on this choice -- only the convergence
# claim does.
#
# ~28 CPU-hours total, ~15 cells and ~85 min per task.
#
# Submit from this directory: sbatch submit_sparse.sbatch
# Completed cells are skipped, so resubmission after failures is safe.
# ============================================================

set -u
# OpenBLAS sizes its pool from the NODE's core count, not the cgroup: measured
# 32 threads against 2 visible cores, turning a 4 s cell into >900 s. This cost
# an entire 12-hour sweep earlier in the project.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JULIA_CPU_TARGET="generic;znver3,clone_all"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs results

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname), $(date) ==="
"$HOME/.juliaup/bin/julia" --project=../../.. --threads=1 \
    run_task_sparse.jl "${SLURM_ARRAY_TASK_ID}" 20
echo "=== task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
