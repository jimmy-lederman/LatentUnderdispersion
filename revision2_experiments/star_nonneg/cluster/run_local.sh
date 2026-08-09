#!/bin/bash
# Local fallback: run all 800 cells with P parallel workers (default 6).
# Skips completed cells, so it is safe to interrupt and rerun.
#   ./run_local.sh [P]
set -euo pipefail
cd "$(dirname "$0")"
P=${1:-6}

for ds in CMP GC Poisson NB; do
  for model in poisson medpois nb mednb star_id star_sqrt starnnf_id starnnf_sqrt; do
    for seed in $(seq 1 100); do
      if [ ! -s "results/${ds}_${model}_${seed}.csv" ]; then
        echo "$ds $model $seed"
      fi
    done
  done
done | xargs -P "$P" -L 1 sh -c 'julia --project=../../.. run_cell.jl $0 $1 $2 > logs/local_$0_$1_$2.log 2>&1 && echo done $0 $1 $2 || echo FAIL $0 $1 $2'
