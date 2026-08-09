#!/bin/bash
#SBATCH --job-name=flights
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint=a40
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=logs/flights_%j.out
#SBATCH --error=logs/flights_%j.err
#SBATCH --mail-user=jlederman@uchicago.edu
#SBATCH --mail-type=END,FAIL
# ============================================================
# Section 7.1 flights campaign: all 200 runs (10 configs x 5 masks x 4 chains),
# SEQUENTIALLY on ONE EXCLUSIVE NODE.
#
# Everything runs sequentially in one job, so no two of OUR runs ever compete: running
# them concurrently was measured to inflate wall-clock by 25-35%, larger than any
# difference the timing table for referee comment 2(b) is trying to report. There is
# therefore no "timing" vs "statistical" split -- FLIGHTS_DEDICATED=1 throughout.
#
# NOT --exclusive, deliberately. Checked with `sbatch --test-only` on 2026-08-08: an
# exclusive A40 node would not start until 09:50 the next morning (SLURM must drain all
# 64 cores), while a 2-core request starts within a couple of hours. Since the campaign
# takes 6-9 h, exclusive would push completion to the following evening.
# What we keep instead:
#   - --constraint=a40 pins every run to one hardware class (g003-g010 are identical
#     64-core nodes), so timings are comparable across models;
#   - the config order rotates each block, so no model is systematically co-scheduled
#     with a noisy neighbour;
#   - 20 replicates per config, medians reported.
# Co-tenant jobs on the node are typically GPU jobs using few CPU cores, and contention
# is random with respect to which of our models happens to be running, so it adds noise
# to absolute wall-clock but does not bias the comparison BETWEEN models. check_runs.jl
# verifies afterwards that every run really did land on the same CPU model.
#
# Measured 12.8 min per (mask, chain) block on an Apple M1; 20 blocks = 4.3 h there,
# so expect roughly 6-9 h on a cluster core. The 12 h limit leaves margin.
#
# SAFE TO RESUBMIT: each run writes its own .jld as it finishes and completed runs are
# skipped, so if the job hits the wall clock or dies, resubmitting resumes where it
# stopped. Nothing already computed is lost or repeated.
#
# The models are pure CPU Julia and use no GPU. The A40 constraint pins every run to one
# hardware class so the timings are comparable -- not for acceleration.
# VERIFY THE SELECTOR FOR YOUR CLUSTER FIRST:  sinfo -o "%20P %30N %25f %20G"
#   --constraint=a40       A40 exposed as a node FEATURE        <- assumed here
#   --gres=gpu:a40:1       A40 exposed as a GRES
#   --partition=<gpu>      A40 nodes in their own partition
#
# Submit from this directory:  sbatch submit_all.sbatch
# ============================================================
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./configs.sh
REPO=../../..

export FLIGHTS_DEDICATED=1   # every run is measured under dedicated conditions
export JULIA_NUM_THREADS=1   # single-threaded for ALL models: flights threads more loops
                             # than flights_STAR, so threaded numbers are not comparable

echo "=== flights campaign on $(hostname), job ${SLURM_JOB_ID:-local}, start $(date)"
echo "=== cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null || echo unknown)"
echo "=== ${#CONFIGS[@]} configs x ${#MASKS[@]} masks x ${#CHAINS[@]} chains"

ndone=0; nskip=0; nfail=0; block=0
for mask in "${MASKS[@]}"; do
  for chain in "${CHAINS[@]}"; do
    # Rotate the config order each block so no model is systematically last, in case the
    # node drifts thermally over the run.
    n=${#CONFIGS[@]}
    for ((k = 0; k < n; k++)); do
      cfg=${CONFIGS[$(( (k + block) % n ))]}
      read -r D TYPE G <<< "$cfg"
      OUT=$(outfile_for "$D" "$TYPE" "$G" "$mask" "$chain")
      if [ -s "$REPO/$OUT" ]; then
        nskip=$((nskip+1)); continue
      fi
      echo "--- $(date +%H:%M:%S)  D=$D type=$TYPE g=$G mask=$mask chain=$chain"
      if "$JULIA" --project=$REPO --threads=1 \
           $REPO/analysis/flights/runflights_heldout.jl "$mask" "$chain" "$D" "$TYPE" "$G"; then
        ndone=$((ndone+1))
      else
        echo "!!! FAILED: D=$D type=$TYPE g=$G mask=$mask chain=$chain"; nfail=$((nfail+1))
      fi
    done
    block=$((block+1))
    echo "=== block ${block}/20 done (mask=$mask chain=$chain), $(date)"
  done
done

echo "=== campaign complete: ${ndone} run, ${nskip} skipped, ${nfail} failed, $(date)"
echo "=== verify with: julia --project=$REPO $REPO/revision2_experiments/flights/cluster/check_runs.jl"
