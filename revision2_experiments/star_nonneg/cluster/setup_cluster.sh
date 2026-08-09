#!/bin/bash
# ============================================================
# One-time setup on the cluster (pattern from SD/cluster/setup_cluster.sh).
# Run interactively after cloning/syncing the repo:
#   cd ~/OrderStats
#   bash revision2_experiments/star_nonneg/cluster/setup_cluster.sh
# ============================================================
set -e

echo "=== 1. Check Julia ==="
if ! command -v julia &> /dev/null; then
    echo "Julia not found. Install it first:"
    echo "  curl -fsSL https://install.julialang.org | sh"
    exit 1
fi
julia --version

echo ""
echo "=== 2. Instantiate Julia project ==="
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

echo ""
echo "=== 3. Create directories ==="
mkdir -p revision2_experiments/star_nonneg/cluster/logs
mkdir -p revision2_experiments/star_nonneg/cluster/results

echo ""
echo "=== 4. Smoke test one cell (small budget) ==="
cd revision2_experiments/star_nonneg/cluster
julia --project=../../.. run_cell.jl CMP starnnf_sqrt 999 100 500 5 2
rm -f results/CMP_starnnf_sqrt_999.csv
echo ""
echo "Setup complete. Submit with: sbatch submit_array.sbatch"
