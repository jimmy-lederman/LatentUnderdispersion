#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=factor100
#SBATCH --array=1-21%21
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=logs/factor_%a_%j.out
#SBATCH --error=logs/factor_%a_%j.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# Section 6.3 production run: 4 datasets x 8 models x 100 seeds = 3200 cells.
#
# 21 array tasks, all 21 running at once (%21): the cluster's cap is 24
# concurrent jobs, the flights campaign holds 2, and one slot is deliberately
# left free. Round-robin via run_task.jl: task j runs ORDER[j], ORDER[j+21],
# ... = 152-153 cells each, sequentially, in ONE Julia process (packages load
# once, not per cell). At the 40-120 s/cell that 4 chains x 10000 sweeps
# implies, that is ~2-5 h per task, comfortably inside the 12 h wall.
#
# Chosen over submit_5nodes.sh (one job per A40 node) because at submission
# time only g006-g009 were idle -- g003/g004 were fully allocated at load ~50.
# Four nodes means 800 cells each, ~9-27 h, which overruns the wall. The
# one-node-per-job design existed to keep seconds-per-fit comparable across
# models, but the timing column now comes from submit_timing.sbatch on a single
# pinned node, and fits/ESS/R-hat are hardware-independent -- so the sweep does
# not need homogeneous hardware and can use whatever capacity is free.
#
# Cell -> (dataset, model, seed):
#   seed  = (cell-1) % 100 + 1
#   model = MODELS[((cell-1) / 100) % 8]
#   ds    = DATASETS[(cell-1) / 800]
#
# One-time setup on the cluster (from the repo root):
#   julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
#   mkdir -p revision2_experiments/star_nonneg/cluster/logs
# Submit from this directory:  sbatch submit_array.sbatch
# Completed cells are skipped, so resubmission after failures is safe.
# ============================================================

set -u
# OpenBLAS sizes its thread pool from the node's physical core count, not the
# cgroup: measured 32 BLAS threads against 2 visible cores, turning a 4 s cell
# into >900 s. Julia's own threads are already 1; this is the separate BLAS pool.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

NJOBS=21
TASK_ID=${SLURM_ARRAY_TASK_ID}

# multi-target cache: login node (precompile host) and compute nodes have
# different CPU archs (znver3 rejection otherwise)
export JULIA_CPU_TARGET="generic;znver3,clone_all"

echo "=== Job ${TASK_ID} on $(hostname), start $(date) ==="

# one Julia process for ALL of this task's cells: packages load once
# (~minutes) instead of once per cell. Single-threaded for timing fairness
# (the original models use internal @threads, the STAR variants do not).
"$HOME/.juliaup/bin/julia" --project=../../.. --threads=1 run_task.jl "$TASK_ID" "$NJOBS"

echo "=== Job ${TASK_ID} complete: $(date) ==="
