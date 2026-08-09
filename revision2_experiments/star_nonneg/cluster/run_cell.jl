# Thin wrapper: run a single cell (see cell_lib.jl for the implementation,
# run_task.jl for the batched array-task driver).
#
# Usage: julia --project=../../.. run_cell.jl <dataset> <model> <seed> [nsamples] [nburnin] [nthin] [nchains]

include("cell_lib.jl")

run_one(ARGS[1], ARGS[2], parse(Int, ARGS[3]);
        NSAMPLES=length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5000,
        NBURNIN=length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 5000,
        NTHIN=length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1,
        NCHAINS=length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 4)
