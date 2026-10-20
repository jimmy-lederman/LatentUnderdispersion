#!/bin/bash
# Shared configuration for the Section 7.2 COVID diagnostic run.
#
# Purpose: everything measured so far is on 100-300 county subsets, and those are NOT a
# scaled-down version of the real problem -- Q = 25 is proportionate at N = 3104 and badly
# over-parameterised at N = 100. This campaign fits the FULL matrix so the diagnostics that
# go in the paper are the ones a reader would reproduce.

CHAINS=(1 2 3 4)
NSWEEPS=15000

# --- threading -------------------------------------------------------------------------
# Unlike Section 7.1, threading is FAIR here: every row of the Section 7.2 table is the same
# code, so there is no cross-model asymmetry to preserve. And the gain is large -- the
# sampler went 3.45 -> 0.67 s/sweep from threading alone on 8 cores.
#
# OPENBLAS_NUM_THREADS=1 is not optional. The sampler calls BLAS from inside @threads loops,
# and OpenBLAS sizes its pool from the machine's PHYSICAL CORE COUNT, not the cgroup
# allocation -- so on a 64-core node a job with few CPUs spawns 64 BLAS threads fighting
# over them. That is the failure mode that previously turned 31-second cells into hours.
# The model file also calls BLAS.set_num_threads(1), but only once it has loaded.
CPUS=16

# --- measured cost, for sizing the wall clock --------------------------------------------
# Apple M4 Pro, 8 threads, BLAS pinned, FULL matrix (N=3104, M=295, K=5, Q=25, Dmax=9):
#   0.67 s/sweep early in the chain, and cost RISES with the inferred D (the latent-sum
#   sampler takes D sequential steps per cell, and D climbs as the fit improves) -- roughly
#   1.5x from start to sweep 8000.
# MEASURED ON THE CLUSTER (srun, same full matrix):
#   4 threads on g005:     3.1 s/sweep
#   16 threads on i001-ds: 1.1 s/sweep
#   Cluster cores are much slower than the M4 Pro. At 16 threads and ~1.7 s/sweep converged,
#   15000 sweeps is ~7 h/chain, inside the 12 h request. Checkpointing every 500 sweeps means
#   a job that hits the wall clock can simply be resubmitted.
WALLTIME=12:00:00
# NO --constraint=a40 for this campaign. The flights job pins the hardware class because its
# purpose is comparable TIMINGS; this one is diagnostics, so pinning would only lengthen the
# queue. Consequence: the per-sweep costs recorded here span node types and must NOT be used
# for the paper's compute table.
MEMMB=16000

OUTDIR=output/covid/diag
# COLLAPSED=1 in the environment (e.g. `COLLAPSED=1 sbatch submit_diag.sbatch`; sbatch
# propagates the caller's environment) runs the HYBRID sampler: theta by the collapsed
# slice step, phi by the paper's CRT / forward pass on thinned counts (the composition
# claim of Section 7.2 is untouched). COLLAPSED=2 collapses phi and eps as well (no gain
# over the hybrid on the 100-county subset). Each writes to its own directory so the
# samplers' chains never mix. Per-sweep cost: hybrid ~1.5x thinning, full ~3x; ESS per
# sweep ~50x on synthetic. The checkpoint/resume path covers a job that runs out of time.
case "${COLLAPSED:-0}" in
    1) OUTDIR=output/covid/diag_hybrid ;;
    2) OUTDIR=output/covid/diag_collapsed ;;
esac
# DIRICHLET=1: theta on the simplex (implies the hybrid sampler); its own directory.
[ "${DIRICHLET:-0}" = "1" ] && OUTDIR=output/covid/diag_dirichlet
# ATHETA=X (default 1.0): shape of the theta prior; a sparse variant gets its own directory.
[ -n "${ATHETA:-}" ] && [ "${ATHETA}" != "1.0" ] && OUTDIR=${OUTDIR}_a${ATHETA}
# KFAC / QDIM (defaults 5 / 25): variants get their own directory.
[ -n "${KFAC:-}" ] && [ "${KFAC}" != "5" ] && OUTDIR=${OUTDIR}_K${KFAC}
[ -n "${QDIM:-}" ] && [ "${QDIM}" != "25" ] && OUTDIR=${OUTDIR}_Q${QDIM}
# DFIX=n (default 0 = inferred): D pinned; the D = 1 Poisson baseline gets its own directory.
[ -n "${DFIX:-}" ] && [ "${DFIX}" != "0" ] && OUTDIR=${OUTDIR}_D${DFIX}
# CPHI (default 100): the gamma-Markov chain's c.
[ -n "${CPHI:-}" ] && [ "${CPHI}" != "100" ] && OUTDIR=${OUTDIR}_c${CPHI}
# FINAL=1: the paper's run -- K = 5, Q = 1, c = 100, eps = 0.5, hybrid sampler, unthinned,
# full-surface statistics -- in its own directory, never mixed with the exploratory arrays.
[ "${FINAL:-0}" = "1" ] && OUTDIR=output/covid/diag_final_Q1
# ANNEAL=n (D held at 1 for the first n sweeps of warm-up) gets its own directory; this
# rule is last so it also applies to the FINAL run.
[ -n "${ANNEAL:-}" ] && [ "${ANNEAL}" != "0" ] && OUTDIR=${OUTDIR}_anneal${ANNEAL}
[ "${NMFINIT:-0}" = "1" ] && OUTDIR=${OUTDIR}_nmf

JULIA="$HOME/.juliaup/bin/julia"
[ -x "$JULIA" ] || JULIA="$(command -v julia)"
[ -n "$JULIA" ] || { echo "ERROR: no julia found" >&2; exit 1; }

# The cluster HOME is over quota, so Julia cannot write precompilation caches there. Prepend
# a writable depot in project space while KEEPING ~/.julia on the path: the packages are
# already installed there, and replacing the path outright would make Julia re-download and
# rebuild all of them. Julia writes to the first depot and reads from both.
if [ -d /net/projects/schein-lab/jimmy ]; then
    export JULIA_DEPOT_PATH="/net/projects/schein-lab/jimmy/juliadepot:$HOME/.julia"
fi
export JULIA_CPU_TARGET="generic;znver3,clone_all"

outfile_for() {  # arg: chainSeed
  echo "${OUTDIR}/diag_chain${1}.jld"
}
