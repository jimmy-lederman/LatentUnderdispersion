# One array task of the sparse-prior grid. Loads packages ONCE and runs its share
# of cells in-process (round-robin: task_id, task_id + njobs, ...). Julia startup
# plus compilation is ~20 s, so per-cell invocation would waste over an hour
# across 300 cells.
#
# Cell order is shuffled with a fixed seed so no model is systematically
# scheduled early or late -- wall-clock enters the ESS/s metric, so drift in node
# load must not correlate with model.
#
# Completed cells are skipped, so resubmission after a failure is safe.
#
# Usage: julia --project=../../.. run_task_sparse.jl <task_id> <njobs>

include("sparse_lib.jl")
using Random

task_id = parse(Int, ARGS[1])
njobs   = parse(Int, ARGS[2])

const DS      = get(ENV, "DS", "SparsePois")
const NS      = parse(Int, get(ENV, "NS", "48000"))
const NB      = parse(Int, get(ENV, "NB", "48000"))
const NC      = parse(Int, get(ENV, "NC", "4"))
const MODELS  = ["poisson", "medpois", "starnn_slice_sqrt", "starnn_slice", "starnn_mh_sqrt"]
const ACS     = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
const SEEDS   = 1:10

cells = [(m, a, s) for m in MODELS for a in ACS for s in SEEDS]
shuffle!(MersenneTwister(20260910), cells)
mine = cells[task_id:njobs:length(cells)]
println("task $task_id/$njobs: $(length(mine)) of $(length(cells)) cells, "
        * "$DS, $NC chains x $NS after $NB")
flush(stdout)

for (i, (m, a, s)) in enumerate(mine)
    out = joinpath(@__DIR__, "results", "$(DS)_$(m)_$(a)_$(s).csv")
    if isfile(out) && filesize(out) > 0
        println("skip $m $a $s"); flush(stdout); continue
    end
    try
        run_one(DS, m, a, s; NS=NS, NB=NB, NC=NC)
    catch e
        println("FAILED $m $a $s: ", first(sprint(showerror, e), 300))
    end
    flush(stdout)
end
println("task $task_id complete")
