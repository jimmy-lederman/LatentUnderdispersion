# Thin wrapper: run one cell. Implementation lives in sparse_lib.jl; the task
# driver run_task_sparse.jl runs many cells in ONE process so Julia startup and
# compilation are paid once rather than per cell.
#
# Usage: julia --project=../../.. run_sparse_cells.jl <dataset> <model> <ac> <seed> [nsamples nburnin nchains]
include("sparse_lib.jl")
run_one(ARGS[1], ARGS[2], parse(Float64, ARGS[3]), parse(Int, ARGS[4]);
        NS = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2000,
        NB = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2000,
        NC = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 4)
