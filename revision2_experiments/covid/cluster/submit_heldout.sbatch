#!/bin/bash
#SBATCH --job-name=covidho
#SBATCH --array=1-120%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=logs/covidho_%A_%a.out
#SBATCH --error=logs/covidho_%A_%a.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# Section 7.2 information-gain campaign (Table COVIDinfo), FULL matrix, K = 5, Q = 1.
#
# 120 tasks = 6 arms x 5 masks x 4 chains, which is exactly what summarize_covid.jl reads:
#   arms   D = 0 (inferred), 1 (the Poisson baseline), 3, 5, 7, 9    -- ALL as type 4
#   masks  101..105   (the SE in the table is over these five train/test splits)
#   chains 1..4       (pooled per split: logsumexp over chains per cell)
#
# ALL SIX ARMS RUN FROM covid4, with D pinned via constantinit + skipupdate for the fixed
# rows, rather than covid3 for those rows. Every row then shares one kernel, so the table
# compares orders rather than samplers. Skipping is exact, not an approximation: with D
# pinned, beta and tau have no data and nothing else in the model depends on them.
#
# SAMPLER: the hybrid -- theta by the collapsed slice step, phi keeping the CRT / Gamma
# forward pass, the (theta, phi) scale move on, CRT tol 0.02 -- matching the diagnostics
# run in output/covid/diag_final_Q1. eps is PINNED at 0.5, also matching.
#
# RUN LENGTH: the inferred arm gets 5000 burn-in + 500 draws thinned by 20 (15000 sweeps);
# the pinned arms 2000 + 500 thinned by 8 (6000 sweeps). The asymmetry is deliberate and is
# explained in runcovid_heldout.jl -- only the inferred arm has rho's climb from 1/2, which
# took up to ~5000 sweeps in the final diagnostics run.
#
# %20 THROTTLE: the cluster allows 24 concurrent jobs; this leaves 4 free.
#
# MEMORY: `fit` accumulates whole state dicts and D_NM is 915,680 entries per draw, so 500
# retained draws are ~4 GB resident before working copies. 32 GB is the headroom for that.
# The same 4 GB is what each checkpoint weighs, which is why they go to scratch (see below).
#
# SAFE TO RESUBMIT: every task checkpoints every 500 sweeps (MatrixMF.fit ckptfile) and
# resumes where it stopped; completed tasks are skipped. Resuming is crash recovery, not
# reproducibility -- the per-task RNGs inside @threads loops cannot be restored.
#
# NO hardware constraint: this is an information-gain campaign, not a timing one. The
# paper's timing column comes from the diagnostics run, which records its own environment.
#
# Submit from this directory:  sbatch submit_heldout.sbatch
# Then:  julia --project=../../.. ../../../analysis/covid/summarize_covid.jl
# ============================================================
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./configs.sh          # JULIA, JULIA_DEPOT_PATH, JULIA_CPU_TARGET, CPUS
REPO=../../..

export JULIA_NUM_THREADS=${CPUS}
export OPENBLAS_NUM_THREADS=1

# Checkpoints on SCRATCH (hundreds of TB free, network-mounted so a resubmitted task finds
# its checkpoint on any node), outputs in the project space. A late-run checkpoint is ~4 GB
# because `fit` retains whole state dicts; 20 of those would not fit in the project quota.
# Scratch is purged, which is harmless -- runcovid_heldout.jl deletes each checkpoint as
# soon as that run's output is written.
export CKPTDIR=${CKPTDIR:-/net/scratch/$USER/orderstats/ckpt/heldout}
mkdir -p "$CKPTDIR"

KFAC=5
QDIM=1
DLIST=(0 1 3 5 7 9)
MASKS=(101 102 103 104 105)

i=$((SLURM_ARRAY_TASK_ID - 1))
CHAIN=$(( i % 4 + 1 ))
MASK=${MASKS[$(( (i / 4) % 5 ))]}
DVAL=${DLIST[$(( i / 20 ))]}

OUT="$REPO/output/covid/revisionsamples/covid4D${DVAL}K${KFAC}Q${QDIM}mask${MASK}chain${CHAIN}.jld"

echo "=== covid held-out: D=${DVAL} mask=${MASK} chain=${CHAIN} (task ${SLURM_ARRAY_TASK_ID}), job ${SLURM_JOB_ID:-local}, $(date)"
echo "=== host $(hostname), cpus ${CPUS}, julia threads ${JULIA_NUM_THREADS}, openblas ${OPENBLAS_NUM_THREADS}, ckptdir ${CKPTDIR}"
echo "=== cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null || echo unknown)"

if [ -s "$OUT" ]; then
  echo "=== $OUT already complete, nothing to do"
  exit 0
fi

"$JULIA" --project=$REPO --threads=${CPUS} --gcthreads=1 \
    "$REPO/analysis/covid/runcovid_heldout.jl" \
    "$MASK" "$CHAIN" "$DVAL" "$KFAC" "$QDIM" 4

echo "=== D=${DVAL} mask=${MASK} chain=${CHAIN} finished $(date)"
