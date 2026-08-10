#!/bin/bash
# Shared configuration for the Section 7.1 flights campaign.
# Each run is: runflights_heldout.jl <maskSeed> <chainSeed> <D> <type> <g>
#   type 1 = MedPoisson, hierarchical mu (default)      type 3 = MedPoisson, flat mu
#   type 2 = STAR,       hierarchical mu (default)      type 4 = STAR,       flat mu
#   D = 0 infers D_k; D > 0 fixes it. g: 1 = identity, 2 = sqrt.

# --- the Table 2 columns ---
CONFIGS=(
  "0 1 0"   # MedPois hier, D_k inferred
  "1 1 0"   # MedPois hier, D_k = 1   <- the Poisson baseline information gain is relative to
  "3 1 0"   # MedPois hier, D_k = 3
  "5 1 0"   # MedPois hier, D_k = 5
  "7 1 0"   # MedPois hier, D_k = 7
  "9 1 0"   # MedPois hier, D_k = 9
  "0 2 2"   # STAR hier, g = sqrt
  "0 5 0"   # MedNB, D_k inferred
  "9 5 0"   # MedNB, D_k = 9 fixed  <- identified; dispersion carried continuously by p
)
# MedNB is back in. The OOM that removed it was a numerics bug, not the model: when the
# parent pdf underflowed the grid gave up and the BigFloat fallback recursed at 5x
# precision forever (48 GB/sweep). Fixed by passing logpdf into the grid and capping the
# recursion; the same state now costs 0.4 MB. p is updated by a mean-preserving Metropolis
# move every 10th sweep -- its conjugate conditional cannot learn p at this data scale.
# The flat-prior models (types 3 and 4) are dropped. Every reported model now carries the
# hierarchical mu_k prior, so the whole table comes from one code version at one commit --
# which is what check_runs.jl verifies before it will endorse the timings. The flat runs
# from the previous campaign remain on disk but are no longer read.

MASKS=(101 102 103 104 105)     # the five train/test splits
CHAINS=(1 2 3 4)                # four chains per split

# Timing replicates: which (mask, chain) pairs get a dedicated exclusive node.
# Same seeds as the statistical grid, so these runs ARE part of it -- they are simply
# measured under conditions where the wall-clock is publishable.
TIMING_MASK=101
TIMING_CHAINS=(1 2 3)

JULIA="$HOME/.juliaup/bin/julia"
# fall back to whatever is on PATH if juliaup is not installed at that location
[ -x "$JULIA" ] || JULIA="$(command -v julia)"
[ -n "$JULIA" ] || { echo "ERROR: no julia found" >&2; exit 1; }

# The cluster HOME is over quota (it cannot absorb even a 5 MB write), so Julia cannot
# write precompilation caches there. Prepend a writable depot in project space while
# KEEPING ~/.julia on the path: the 255 packages of this environment are already
# installed there, and replacing the path outright would make Julia try to re-download
# and rebuild all of them. Julia writes to the first (project) depot and reads from both.
# Project space is NFS-mounted and visible from every compute node.
if [ -d /net/projects/schein-lab/jimmy ]; then
    export JULIA_DEPOT_PATH="/net/projects/schein-lab/jimmy/juliadepot:$HOME/.julia"
fi
# login node and compute nodes have different CPU archs; keep the multi-target cache.
# If the A40 nodes are Intel rather than AMD, add the matching target here.
export JULIA_CPU_TARGET="generic;znver3,clone_all"

# Output file that a given run produces, so completed runs can be skipped on resubmission.
outfile_for() {  # args: D type g mask chain
  local D=$1 type=$2 g=$3 mask=$4 chain=$5
  case $type in
    1) echo "output/flights/revisionsamples/MedPoissonD${D}mask${mask}chain${chain}.jld" ;;
    2) echo "output/flights/revisionsamples/STARg${g}mask${mask}chain${chain}.jld" ;;
    3) echo "output/flights/revisionsamples/MedPoissonFlatD${D}mask${mask}chain${chain}.jld" ;;
    4) echo "output/flights/revisionsamples/STARFlatg${g}mask${mask}chain${chain}.jld" ;;
    5) echo "output/flights/revisionsamples/MedNBD${D}mask${mask}chain${chain}.jld" ;;
  esac
}
