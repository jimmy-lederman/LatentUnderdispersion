#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=factor100
#SBATCH --array=1-23%23
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=logs/factor_%a_%j.out
#SBATCH --error=logs/factor_%a_%j.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# Section 6.3 production run: 4 datasets x 8 models x 100 seeds = 3200 cells.
#
# 199 array tasks, throttled to 23 RUNNING at once (%23) — one slot below
# the cluster's 24-concurrent-job cap stays free for other work. Remaining
# tasks wait in the array queue and start as slots free up.
# NOTE: if the 24-cap turns out to apply to SUBMITTED (not running) jobs and
# sbatch rejects the array, resize to --array=1-23%23 and set NJOBS=23 below
# (the round-robin then gives each job ~139 cells, ~2h each — still fine).
# Round-robin assignment as in SD/cluster/submit_all.sh: job j runs cells
# j, j+199, j+398, ... Each job handles 16-17 cells sequentially
# (~15-25 min total per job).
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
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

NJOBS=23
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
