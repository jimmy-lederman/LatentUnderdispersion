# One array task of the production run: loads packages/models ONCE, then
# runs all of this task's cells in-process (round-robin: cells task_id,
# task_id + njobs, ...). Skips completed cells; a cell failure is logged and
# does not kill the task.
#
# Usage: julia --project=../../.. run_task.jl <task_id> <njobs>

include("cell_lib.jl")

using Random

task_id = parse(Int, ARGS[1])
njobs = parse(Int, ARGS[2])
const TOTAL = 3200
const NSEEDS_T = 100
DATASETS = ["CMP", "GC", "Poisson", "NB"]
MODELS = ["poisson", "medpois", "nb", "mednb", "star_id", "star_sqrt",
          "starnnf_id", "starnnf_sqrt"]

# Randomize the cell order (fixed seed, so the assignment is reproducible and
# every task computes the same permutation).
#
# The natural enumeration is dataset-major then model-major, which would run all
# `poisson` cells at the start of a task and all `starnnf_sqrt` cells at the end.
# Any drift in node load over a multi-hour run would then be confounded with
# MODEL, biasing the timing column. Shuffling decorrelates model from wall-clock
# position, so co-tenant noise averages out across models instead of loading onto
# whichever model happens to run last -- the same reasoning behind rotating the
# config order in the flights campaign.
const ORDER = shuffle(MersenneTwister(20260808), collect(1:TOTAL))

for cell in ORDER[task_id:njobs:TOTAL]
    idx = cell - 1
    seed = idx % NSEEDS_T + 1
    m = MODELS[(idx ÷ NSEEDS_T) % 8 + 1]
    ds = DATASETS[idx ÷ (NSEEDS_T * 8) + 1]
    out = joinpath(@__DIR__, "results", "$(ds)_$(m)_$(seed).csv")
    if isfile(out) && filesize(out) > 0
        println("skip $ds $m $seed (done)")
        continue
    end
    println("--- cell $cell/$TOTAL: $ds $m seed $seed")
    flush(stdout)
    try
        run_one(ds, m, seed)
    catch e
        msg = sprint(showerror, e)
        println("FAILED: $ds $m $seed: ", first(msg, 300))
    end
    flush(stdout)
end
println("task $task_id complete")
