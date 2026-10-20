#!/bin/bash
#SBATCH --job-name=coviddiag
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16000
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=logs/coviddiag_%A_%a.out
#SBATCH --error=logs/coviddiag_%A_%a.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# Section 7.2 COVID diagnostics on the FULL matrix: 4 chains, one per array task.
#
# WHY AN ARRAY AND NOT ONE SEQUENTIAL JOB (unlike the flights campaign): chains are
# independent, so this finishes in one chain's wall clock instead of four, and a chain that
# dies costs one chain rather than the campaign. The flights job runs sequentially because
# its purpose is comparable TIMINGS, which requires no two of our runs competing. This job's
# purpose is diagnostics, not timing, so concurrency is free.
#
# THREADS: 16 per task, and OPENBLAS_NUM_THREADS=1. See configs.sh -- the second is not
# optional, OpenBLAS otherwise sizes its pool from the node's 64 physical cores regardless
# of the cgroup, which is what previously turned 31-second cells into hours.
#
# SAFE TO RESUBMIT: each chain checkpoints every 500 sweeps and resumes from the checkpoint.
# A task that hits the 12 h wall clock continues where it stopped. Completed chains are
# skipped.
#
# NO hardware constraint, unlike the flights job: that one pins --constraint=a40 because its
# purpose is comparable timings. This one is diagnostics, so pinning only lengthens the queue
# -- but it means the per-sweep costs in these logs span node types and are NOT usable for
# the paper's compute table.
#
# Submit from this directory:  sbatch submit_diag.sbatch
# Then:  julia --project=../../.. analyze_diag.jl
# ============================================================
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./configs.sh
REPO=../../..

export JULIA_NUM_THREADS=${CPUS}
export OPENBLAS_NUM_THREADS=1
export COLLAPSED=${COLLAPSED:-0}     # see configs.sh; read by run_diag_chain.jl
export DIRICHLET=${DIRICHLET:-0}
export ATHETA=${ATHETA:-1.0}
export KFAC=${KFAC:-5}
export QDIM=${QDIM:-25}
export THIN=${THIN:-1}
export DFIX=${DFIX:-0}
export CPHI=${CPHI:-100}
export FINAL=${FINAL:-0}
export ANNEAL=${ANNEAL:-0}
export NMFINIT=${NMFINIT:-0}

CHAIN=${SLURM_ARRAY_TASK_ID}
OUT=$(outfile_for "$CHAIN")

echo "=== covid diagnostics: chain ${CHAIN}, job ${SLURM_JOB_ID:-local}, $(date)"
echo "=== host $(hostname), cpus ${CPUS}, julia threads ${JULIA_NUM_THREADS}, openblas ${OPENBLAS_NUM_THREADS}, collapsed ${COLLAPSED}, K ${KFAC}, Q ${QDIM}, outdir ${OUTDIR}"
echo "=== cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null || echo unknown)"

if [ -s "$REPO/$OUT" ]; then
  echo "=== $OUT already complete, nothing to do"
  exit 0
fi

# --gcthreads=1: Julia 1.11 segfaulted in gc_sweep_pool_parallel under this workload on
# macOS (a known parallel-GC instability); one GC thread avoids the parallel sweep entirely.
"$JULIA" --project=$REPO --threads=${CPUS} --gcthreads=1 \
    "$REPO/revision2_experiments/covid/cluster/run_diag_chain.jl" \
    "$CHAIN" "$NSWEEPS" "$REPO/$OUTDIR"

echo "=== chain ${CHAIN} finished $(date)"
