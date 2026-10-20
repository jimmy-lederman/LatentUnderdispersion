#!/bin/bash
#SBATCH --job-name=covidig
#SBATCH --array=1-120%23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16000
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=logs/covidig_%A_%a.out
#SBATCH --error=logs/covidig_%A_%a.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# Section 7.2 information-gain campaign (Table COVIDinfo), FULL matrix, K = 5, Q = 1.
#
# 120 tasks = 6 arms x 5 masks x 4 chains:
#   arms   D = 0 (inferred), 1 (the Poisson baseline), 3, 5, 7, 9  -- all from covid4
#   masks  101..105  (the table's SE is over these five forecasting splits)
#   chains 1..4      (pooled per split: logsumexp over chains per cell)
#
# THIS DRIVES run_diag_chain.jl, NOT analysis/covid/runcovid_heldout.jl. The two were doing
# the same job by different means: run_diag_chain.jl builds the identical forecasting mask
# (MASKSEED=101 reproduces it exactly), fits the same model with the same hybrid sampler and
# the same pinned eps, and evaluates the held-out likelihood INSIDE the sweep loop, storing
# only small functionals. runcovid_heldout.jl retains whole state dicts (7.5 MB/draw, mostly
# D_NM) and evaluates afterwards, which is what forced both a 32 GB request and thinning.
# In-loop evaluation needs neither: the chain is stored UNTHINNED and the 300-cell / full-
# surface diagnostics come out of the same run. diag_final_Q1 is therefore the (D inferred,
# mask 101) cell of this grid and does not need rerunning.
#
# IDENTICAL MCMC SETTINGS FOR EVERY CELL: 4 chains x 15000 sweeps, unthinned, same sampler,
# same hyperparameters, same eps. No per-arm asymmetry to explain in the paper. Burn-in is
# an ANALYSIS choice, not a run parameter -- every sweep is stored and analyze_diag.jl /
# summarize_covid.jl apply the 7500-15000 window, matching diag_final_Q1.
#
# WALL CLOCK: cost is O(D) per cell, so the arms are not equally expensive at a common
# sweep count -- roughly 4.6 h at D = 1 and ~19 h at D = 9 for 15000 sweeps. The D = 7 and
# D = 9 arms will exceed the 12 h limit and need ONE resubmission each; checkpointing every
# 500 sweeps makes that a no-op (completed cells are skipped). Budget ~3 days of wall clock
# for the grid at 23 concurrent tasks. The cluster allows 24, so %23 leaves one slot free
# for an interactive srun while the grid runs.
#
# STORAGE, and the two places it goes:
#   SCRATCH  gets the full chain file, ~1.1 GB/chain with THETAFULL=1 (the theta trace alone
#            is 3104 x 5 Float32 x 15000 = 931 MB). ~135 GB for the grid, against hundreds of
#            TB free. Scratch PURGES, so nothing the paper needs may live only here.
#   PROJECT  gets a durable companion: the same file minus the theta trace, ~190 MB/chain and
#            ~23 GB for the grid. Every table and figure is built from this. summarize_covid.jl
#            reads the companions, so a purge costs only the ability to recompute per-draw
#            theta diagnostics later -- not any reported number.
# FULLSURF=1 keeps the N x M statistics for mu and D (117 MB/chain), so the full-surface
# split-Rhat lines of the diagnostics table are available for EVERY cell of the grid rather
# than only the reported one. HOTHIN=10 stores the per-cell held-out densities every
# 10th sweep; the per-draw SUM is stored every sweep. That is a storage cadence, not a
# sampling one -- the chain itself is never thinned.
#
# CHECKPOINT I/O: with THETAFULL=1 a checkpoint is ~1.1 GB and is rewritten every 500 sweeps,
# so each chain writes ~33 GB over its life. That is why checkpoints live on scratch too.
#
# Submit from this directory:  sbatch submit_infogain.sbatch
# Then:  julia --project=../../.. ../../../analysis/covid/summarize_covid.jl
# ============================================================
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./configs.sh          # JULIA, JULIA_DEPOT_PATH, JULIA_CPU_TARGET, CPUS
REPO=../../..

export JULIA_NUM_THREADS=${CPUS}
export OPENBLAS_NUM_THREADS=1

# --- the settings that are identical for every cell ---------------------------------
export COLLAPSED=1       # hybrid: theta by the collapsed slice step, phi keeps the PGDS pass
export KFAC=5
export QDIM=1
export CPHI=100
export ATHETA=1.0
export DIRICHLET=0
export ANNEAL=0
export NMFINIT=0
export THIN=1            # the chain is NOT thinned
export HOTHIN=10         # per-cell held-out densities every 10th sweep (storage only)
export FULLSURF=1        # full-surface split-Rhat for mu and D on all 915,680 cells, every
                         # cell of the grid: +117 MB/chain durable (37 GB total) and <1% runtime
export THETAFULL=1       # theta for every county, every sweep (931 MB/chain, on scratch)
NSWEEPS=15000

# --- the three things that vary across the grid -------------------------------------
DLIST=(0 1 3 5 7 9)
MASKS=(101 102 103 104 105)
i=$((SLURM_ARRAY_TASK_ID - 1))
CHAIN=$(( i % 4 + 1 ))
MASK=${MASKS[$(( (i / 4) % 5 ))]}
DVAL=${DLIST[$(( i / 20 ))]}

export DFIX=${DVAL}
export MASKSEED=${MASK}

# full chain file (with the theta trace) on scratch; durable companion in project space
SCRATCHDIR=/net/scratch/$USER/orderstats/infogain/D${DVAL}_mask${MASK}
export DURABLEDIR="$(cd "$REPO" && pwd)/output/covid/infogain/D${DVAL}_mask${MASK}"
mkdir -p "$SCRATCHDIR" "$DURABLEDIR"
# completion is judged on the DURABLE file: the scratch copy may have been purged
OUT="$DURABLEDIR/summary_chain${CHAIN}.jld"

echo "=== covid info-gain: D=${DVAL} mask=${MASK} chain=${CHAIN} (task ${SLURM_ARRAY_TASK_ID}), job ${SLURM_JOB_ID:-local}, $(date)"
echo "=== host $(hostname), cpus ${CPUS}, threads ${JULIA_NUM_THREADS}, openblas ${OPENBLAS_NUM_THREADS}"
echo "=== scratch ${SCRATCHDIR}  durable ${DURABLEDIR}"
echo "=== cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null || echo unknown)"

if [ -s "$OUT" ]; then
  echo "=== $OUT already complete, nothing to do"
  exit 0
fi

"$JULIA" --project=$REPO --threads=${CPUS} --gcthreads=1 \
    "$REPO/revision2_experiments/covid/cluster/run_diag_chain.jl" \
    "$CHAIN" "$NSWEEPS" "$SCRATCHDIR"

echo "=== D=${DVAL} mask=${MASK} chain=${CHAIN} finished $(date)"
