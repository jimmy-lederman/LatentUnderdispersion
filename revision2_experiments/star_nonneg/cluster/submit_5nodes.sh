#!/bin/bash
# ============================================================
# Section 6.3 sweep: 3200 cells over 5 A40 nodes, ONE job per node, cells run
# SEQUENTIALLY within each job.
#
# Follows the methodology established by the flights campaign
# (revision2_experiments/flights/cluster/submit_all.sbatch):
#   * one hardware class only -- g003-g010 are identical 64-core A40 machines --
#     so seconds-per-fit is comparable across models;
#   * NOT --exclusive: an exclusive A40 request waits for all 64 cores to drain
#     (measured there as an overnight wait), while a 2-core request starts in
#     minutes. Co-tenants are typically GPU jobs using few cores, so they add
#     noise to absolute wall-clock without biasing the comparison BETWEEN models;
#   * one job per node via --nodelist, because the flights campaign measured that
#     running our OWN jobs concurrently on a node inflates wall-clock by 25-35%
#     -- larger than the differences the timing table reports. Five separate
#     submissions guarantee SLURM cannot co-schedule two of ours together;
#   * run_task.jl shuffles the cell order (fixed seed), so no model is
#     systematically scheduled early or late.
#
# Usage:  ./submit_5nodes.sh [node1 node2 ...]
# Default nodes: g003 g004 g006 g007 g009. Check load first with
#   sinfo -p general -N -o "%12N %9T %5C %8O" | grep '^g00'
# and prefer the quietest five.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results/samples

NODES=("$@")
if [ ${#NODES[@]} -eq 0 ]; then
    NODES=(g003 g004 g006 g007 g008)
fi
NJOBS=${#NODES[@]}

echo "submitting $NJOBS sequential jobs, one per node: ${NODES[*]}"
for i in "${!NODES[@]}"; do
    node=${NODES[$i]}
    task=$((i + 1))
    sbatch --job-name="f63_${task}" \
           --nodelist="$node" \
           --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32000 \
           --time=12:00:00 --partition=general \
           --output="logs/f63_${task}_%j.out" --error="logs/f63_${task}_%j.err" \
           --mail-user=jlederman@uchicago.edu --mail-type=END,FAIL \
           --wrap="cd \$SLURM_SUBMIT_DIR && echo \"=== task $task/$NJOBS on \$(hostname) \$(date) ===\" && lscpu | grep -E 'Model name' && \$HOME/.juliaup/bin/julia --project=../../.. --threads=1 run_task.jl $task $NJOBS && echo \"=== task $task done \$(date) ===\""
done
echo
echo "watch:   squeue -u \$USER"
echo "progress: ls results/*.csv | wc -l   (target 3200)"
