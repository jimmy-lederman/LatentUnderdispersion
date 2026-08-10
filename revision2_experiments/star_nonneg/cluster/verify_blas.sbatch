#!/bin/bash
#SBATCH --job-name=blas_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint=a40
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --partition=general
#SBATCH --output=logs/blas_%j.out
#SBATCH --error=logs/blas_%j.err
# ============================================================
# Why the Section 6.3 sweep produced 4 cells per job in 12 hours while burning
# 24 CPU-hours (200%, pegged -- computing, not blocked).
#
# Hypothesis: OpenBLAS sizes its thread pool from the NODE's physical core count
# (64 here), not from the cgroup the job is confined to (--cpus-per-task=2), so
# every U*V product thrashes. Unverified: 200% CPU is equally consistent with two
# cores doing honest but slow work, so this job is meant to DISCRIMINATE, not to
# confirm.
#
# Runs under exactly the allocation shape the failed jobs used. Every earlier
# timing of mine was taken on the login node with all 64 cores unrestricted,
# which is why the cells looked fine (31 s) there and took hours here.
#
# Reports:
#   1. cores visible to the job vs BLAS.get_num_threads()  -- if BLAS reports far
#      more than 2, oversubscription is real
#   2. the same three cells timed with BLAS unset, then pinned to 1 thread
#      -- a large speedup implicates BLAS; equal times mean the cause is elsewhere
#
# Submit from this directory: sbatch verify_blas.sbatch
# ============================================================
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
JULIA=$HOME/.juliaup/bin/julia

echo "=== node $(hostname), $(date) ==="
echo "physical cores on node : $(nproc --all)"
echo "cores visible to job   : $(nproc)"
echo "SLURM_CPUS_PER_TASK    : ${SLURM_CPUS_PER_TASK:-unset}"
echo

echo "=== what BLAS thinks it has, inside this allocation ==="
$JULIA --project=../../.. --threads=1 -e '
using LinearAlgebra
println("  BLAS.get_num_threads() = ", BLAS.get_num_threads())
println("  Sys.CPU_THREADS        = ", Sys.CPU_THREADS)
println("  Julia threads          = ", Threads.nthreads())'
echo

# three cells spanning the cost range: cheap, medium, expensive
CELLS=("Poisson poisson 9901" "CMP medpois 9902" "CMP mednb 9903")

for MODE in unset pinned; do
    if [ "$MODE" = "pinned" ]; then
        export OPENBLAS_NUM_THREADS=1
        echo "=== BLAS PINNED to 1 thread ==="
    else
        unset OPENBLAS_NUM_THREADS
        echo "=== BLAS UNSET (as the failed sweep ran) ==="
    fi
    for c in "${CELLS[@]}"; do
        set -- $c
        rm -f "results/$1_$2_$3.csv" "results/samples/$1_$2_$3_samples.jld"
        START=$(date +%s)
        timeout 900 $JULIA --project=../../.. --threads=1 run_cell.jl "$1" "$2" "$3" > /dev/null 2>&1
        RC=$?
        END=$(date +%s)
        if [ $RC -eq 124 ]; then
            echo "  $2 on $1: TIMED OUT after 900s  <-- this is the failure mode"
        else
            echo "  $2 on $1: $((END - START))s"
        fi
        rm -f "results/$1_$2_$3.csv" "results/samples/$1_$2_$3_samples.jld"
    done
    echo
done

echo "reference: these same cells took 5-45 s each on the login node (64 free cores)"
echo "=== done $(date) ==="
