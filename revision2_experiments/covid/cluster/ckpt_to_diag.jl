# Build analysable diag_chain*.jld files from mid-run checkpoints, so the diagnostics can be
# computed before the chains finish. Checkpoints carry every stored draw and the
# full-surface statistics but none of the run metadata that analyze_diag.jl prints, so the
# metadata is supplied here from the same constants the chain script uses. Completed
# chains are copied through unchanged. Output goes to a separate directory; nothing in the
# run's own directory is touched.
#
#   julia --project=. ckpt_to_diag.jl <rundir> <destdir> [N M K Q Dmax eps]
using JLD, Printf
rundir, dest = ARGS[1], ARGS[2]
N, M, K, Q, Dmax, epsf = length(ARGS) >= 8 ? (parse(Int,ARGS[3]), parse(Int,ARGS[4]), parse(Int,ARGS[5]),
    parse(Int,ARGS[6]), parse(Int,ARGS[7]), parse(Float64,ARGS[8])) : (3104, 295, 5, 1, 9, 0.5)
mkpath(dest)
for c in 1:4
    fin = joinpath(rundir, "diag_chain$c.jld"); ck = joinpath(rundir, "ckpt_chain$c.jld")
    out = joinpath(dest, "diag_chain$c.jld")
    if isfile(fin)
        cp(fin, out, force=true); @printf("chain %d: complete run copied\n", c); continue
    end
    isfile(ck) || (@printf("chain %d: no checkpoint\n", c); continue)
    d = load(ck)
    extra = Dict{String,Any}("chainSeed" => c, "nsweeps" => d["sweep"], "thin" => 1,
        "N" => N, "M" => M, "K" => K, "Q" => Q, "Dmax" => Dmax, "eps_fixed" => epsf,
        "nseg" => size(d["S1mu"], 3), "cells" => fill((0,0), length(d["mu"][1])),
        "seconds" => NaN, "cpu" => "midrun", "nthreads" => 0)
    # JLD.save takes alternating name, value -- not pairs.
    allk = merge(Dict{String,Any}(k => v for (k, v) in d if k != "state"), extra)
    args = Any[]; for (k, v) in allk; push!(args, k); push!(args, v); end
    save(out, args...)
    @printf("chain %d: checkpoint at sweep %d -> %s\n", c, d["sweep"], out)
end
