#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=cos63
#SBATCH --array=1-10%10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --output=logs/cos_%a_%j.out
#SBATCH --error=logs/cos_%a_%j.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# Recompute the Section 6.3 diagnostics with MCMCDiagnosticTools.jl (bulk-ESS,
# tail-ESS, rank-normalized split R-hat) from the posterior samples the sweep
# saved. NO refitting: this reads results/samples/*.jld and writes
# results_cos/*.csv.
#
# Only 10 tasks, not 21: a cell costs 1-3 s, so 320 cells per task is ~10 min.
# Sizing this like the sweep would leave the queue full for no gain, and it
# keeps most of the 24-job cap free.
#
# Cheap enough to be worth stating plainly: the whole 3200-cell recompute is
# ~2 CPU-hours, versus ~24 CPU-hours to refit. That is the payoff for having
# saved U, V and the heldout log-density chain per cell.
#
# Submit from this directory: sbatch submit_diag.sbatch
# Completed cells are skipped, so resubmission after a failure is safe.
# ============================================================

set -u
# Same BLAS pinning as the sweep. The matrices here are tiny (20x3 by 3x20), so
# threading buys nothing, but an unpinned OpenBLAS still sizes its pool from the
# node's core count rather than the cgroup -- the failure that cost the first
# sweep 12 hours.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

NJOBS=10
TASK_ID=${SLURM_ARRAY_TASK_ID}

export JULIA_CPU_TARGET="generic;znver3,clone_all"

echo "=== diag task ${TASK_ID}/${NJOBS} on $(hostname), start $(date) ==="

"$HOME/.juliaup/bin/julia" --project=../../.. --threads=1 \
    recompute_cosine.jl "$TASK_ID" "$NJOBS"

echo "=== diag task ${TASK_ID} complete: $(date) ==="
