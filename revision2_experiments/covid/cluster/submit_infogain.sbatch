#!/bin/bash
#SBATCH --job-name=covidig
#SBATCH --array=1-120%12
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
# IDENTICAL MCMC SETTINGS FOR EVERY CELL: 4 chains x 10000 sweeps, unthinned, 5000 burn-in,
# same sampler, same hyperparameters, same eps. No per-arm asymmetry to explain in the paper.
# Every sweep is stored, so the window stays revisitable; BURNIN only governs the derived
# posterior summaries the run writes.
#
# WHY 5000 + 5000. The draft reports 2000 burn-in and 2000 retained sweeps. 5000 burn-in is
# 2.5x that and is what the evidence supports: the slowest D climb seen in diag_final_Q1 took
# about 4000-5000 sweeps, and once D settles it stays (mean D was already 7.78 by sweep 500
# in the first cell of the cancelled 15000-sweep attempt). 5000 retained, unthinned, gives
# per-chain ESS of roughly 980 on D, 943 on theta, 893 on mu and 93 on the held-out density
# -- the latter from its measured autocorrelation time of ~54 sweeps. s_k*phi stays the weak
# one at ~17/chain; at tau ~ 288 sweeps even 15000 draws only reached 26, so that is a
# property of the phi block rather than of the run length.
#
# WALL CLOCK: cost per sweep is FLAT in D at Q = 1 (measured: 1.48-1.76 s locally across
# D = 1..9 and inferred, arm_cost.jl; 3.38 s on an EPYC 9115 at 16 threads). So 10000 sweeps
# is ~9.4 h per cell, inside the 12 h limit with ~2.6 h of margin -- ONE pass, no
# resubmission, no wall-clock kills -- PROVIDED tasks are not packed onto shared nodes.
#
# CONCURRENCY: %23 was tried and Slurm packed up to EIGHT tasks onto one node (r002), where
# they contend for memory bandwidth: 3.9-5.9 s/sweep against 3.38 s/sweep on a less crowded
# node. That pushed cells past the 12 h limit and produced 18 TIMEOUTs. %12 trades
# parallelism for per-task speed at roughly constant throughput and keeps cells inside the
# wall clock.
#
# RELIABILITY (learned the hard way in arrays 2204211 and 2205122):
#   * checkpoint WRITES are guarded in run_diag_chain.jl -- an HDF5 'Bad address' failure
#     logs a warning instead of killing a 10-hour chain. Roughly 1 chain in 4 hit this.
#   * checkpoint READS are guarded too, so a truncated checkpoint restarts that chain
#     rather than failing the task.
#   * this script checks Julia's exit code AND that the output file exists, so a crash can
#     never again be recorded as COMPLETED. In array 2205122, 9 of 15 "completed" tasks had
#     actually died and written nothing.
#
# (The earlier 15000-sweep attempt, array 2204211, was cancelled: at 3.38 s/sweep it needed
# 14.1 h and would have required a resubmission per cell for a window nobody needed.)
#
# STORAGE, and the two places it goes:
#   SCRATCH  gets the full chain file, ~880 MB/chain with THETAFULL=1 (the theta trace alone
#            is 3104 x 5 Float32 x 10000 = 620 MB). ~106 GB for the grid, against hundreds of
#            TB free. Scratch PURGES, so nothing the paper needs may live only here.
#   PROJECT  gets a durable companion: the same file minus the theta trace, ~260 MB/chain and
#            ~31 GB for the grid. Every table and figure is built from this. summarize_covid.jl
#            reads the companions, so a purge costs only the ability to recompute per-draw
#            theta diagnostics later -- not any reported number.
# FULLSURF=1 keeps the N x M statistics for mu and D (117 MB/chain), so the full-surface
# split-Rhat lines of the diagnostics table are available for EVERY cell of the grid rather
# than only the reported one. HOTHIN=10 stores the per-cell held-out densities every
# 10th sweep; the per-draw SUM is stored every sweep. That is a storage cadence, not a
# sampling one -- the chain itself is never thinned.
#
# CHECKPOINT I/O: with THETAFULL=1 a checkpoint is ~880 MB and is rewritten every 500 sweeps,
# so each chain writes ~18 GB over its life. That is why checkpoints live on scratch too.
# STALE CHECKPOINTS ARE A TRAP: run_diag_chain.jl resumes from any checkpoint it finds, so a
# leftover from a run with different settings would silently continue under the OLD config.
# Clear the scratch tree before relaunching with changed settings.
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
export THETAFULL=1       # theta for every county, every sweep (620 MB/chain, on scratch)
export BURNIN=5000       # warm-up excluded from the derived posterior summaries
export CKPT_EVERY=2500   # fewer 880 MB rewrites -> fewer chances to hit the Lustre HDF5 fault
NSWEEPS=10000

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

if ! "$JULIA" --project=$REPO --threads=${CPUS} --gcthreads=1 \
    "$REPO/revision2_experiments/covid/cluster/run_diag_chain.jl" \
    "$CHAIN" "$NSWEEPS" "$SCRATCHDIR"; then
  rc=$?
  echo "=== JULIA FAILED (exit ${rc}): D=${DVAL} mask=${MASK} chain=${CHAIN}"
  exit ${rc}
fi

if [ ! -s "$OUT" ]; then
  echo "=== NO DURABLE OUTPUT WRITTEN: D=${DVAL} mask=${MASK} chain=${CHAIN}"
  exit 1
fi

echo "=== D=${DVAL} mask=${MASK} chain=${CHAIN} finished $(date)"
