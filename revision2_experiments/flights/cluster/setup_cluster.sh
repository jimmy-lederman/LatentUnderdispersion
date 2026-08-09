#!/bin/bash
# One-time setup on the cluster. Run interactively:
#   cd ~/OrderStats && bash revision2_experiments/flights/cluster/setup_cluster.sh
set -e
cd "$(dirname "$0")"
source ./configs.sh
REPO=../../..

echo "=== 1. Julia ==="
"$JULIA" --version

echo ""
echo "=== 2. Instantiate + precompile (do this BEFORE submitting) ==="
# Precompiling once here matters: 20 array tasks starting simultaneously would otherwise
# race to populate the same depot cache.
"$JULIA" --project=$REPO -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

echo ""
echo "=== 3. Directories ==="
mkdir -p logs "$REPO/output/flights/revisionsamples"

echo ""
echo "=== 4. How does this cluster expose A40 nodes? ==="
echo "    Set --constraint / --gres / --partition in submit_timing.sbatch to match."
sinfo -o "%20P %30N %25f %20G" 2>/dev/null | head -25 || echo "    sinfo unavailable"

echo ""
echo "=== 5. Smoke test: one cheap run (D=1 fixed, a few seconds of sampling) ==="
FLIGHTS_DEDICATED=0 "$JULIA" --project=$REPO --threads=1 \
    $REPO/analysis/flights/runflights_heldout.jl 999 1 1 1 0
rm -f "$REPO/output/flights/revisionsamples/MedPoissonD1mask999chain1.jld"

echo ""
echo "Setup complete. Then:"
echo "  sbatch submit_all.sbatch     # all 200 runs, sequential, one exclusive node (~6-9 h)"
echo ""
echo "Afterwards, verify hardware homogeneity of the timed runs before believing the timings:"
echo "  julia --project=. revision2_experiments/flights/cluster/check_runs.jl"
