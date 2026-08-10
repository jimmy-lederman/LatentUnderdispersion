# Recompute Section 6.3 MCMC diagnostics with the STANDARD estimators, from the
# posterior samples saved by cell_lib.jl -- no refitting required.
#
# Why this exists. The sweep reported ESS from `ess_multichain` in
# revison_experiments/mcmc_stuff.jl, which CONCATENATES the four chains end to
# end and takes the autocorrelation of the concatenated series. The standard
# estimator takes autocorrelations WITHIN chains and combines them with the
# between-chain variance. Concatenation turns any difference in chain means into
# apparent long-range positive autocorrelation, which inflates the Geyer sum and
# so UNDER-states ESS. The bias is conservative, and with R-hat ~= 1.00-1.03 it is
# small here, but it is a homemade estimator and referee 1 specifically asked how
# ESS was computed.
#
# `rhat_split_multivariate` in that same file is, by contrast, textbook split
# R-hat (BDA3) and needs no correction; what it is not is the RANK-NORMALIZED
# R-hat of Vehtari et al. (2021), which additionally detects differences in scale
# and in the tails.
#
# This script uses MCMCDiagnosticTools.jl (the implementation behind Turing and
# ArviZ) for:
#   * bulk-ESS   -- rank-normalized, the headline "how many effective draws"
#   * tail-ESS   -- ESS for the 5%/95% quantiles, where heavier-tailed factor
#                   posteriors strain in a way bulk-ESS smooths over
#   * rank-normalized split R-hat
# on the same two identified functionals the sweep reported: the fitted means
# mu = (UV)_ij over all N*M entries, and l, the mean heldout log-density.
#
# Parameters are passed one at a time as (draws, chains) matrices, which is the
# unambiguous input layout -- no reliance on the 3-D dimension convention.
#
# Usage (array task, mirrors run_task.jl):
#   julia --project=../../.. recompute_diag.jl <task_id> <njobs>
# Writes one CSV per cell into results_diag/; completed cells are skipped, so
# resubmission after a failure is safe.

using JLD, MCMCDiagnosticTools, Statistics, Printf, DataFrames, CSV

task_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
njobs   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1

const SAMPDIR = joinpath(@__DIR__, "results", "samples")
const OUTDIR  = joinpath(@__DIR__, "results_diag")
mkpath(OUTDIR)

function diag_cell(path::String)
    d = JLD.load(path)
    U, V = d["U"], d["V"]
    ll = d["llik_chain"]                 # (nsamples, 1, nchains)
    nch, ns = d["nchains"], d["nsamples"]
    N, K = size(U, 2), size(U, 3)
    M = size(V, 3)
    P = N * M

    # mu[draw, chain, entry]; saved global index is (ch-1)*ns + s
    mu = Array{Float64}(undef, ns, nch, P)
    for ch in 1:nch, s in 1:ns
        g = (ch - 1) * ns + s
        mu[s, ch, :] = vec(Float64.(@view U[g, :, :]) * Float64.(@view V[g, :, :]))
    end

    bulk = Vector{Float64}(undef, P)
    tail = Vector{Float64}(undef, P)
    rh   = Vector{Float64}(undef, P)
    for p in 1:P
        A = @view mu[:, :, p]
        bulk[p] = ess(A; kind = :bulk)
        tail[p] = ess(A; kind = :tail)
        rh[p]   = rhat(A; kind = :rank)
    end

    L = dropdims(ll; dims = 2)           # (nsamples, nchains)
    return (essmu_bulk_med = median(bulk), essmu_bulk_min = minimum(bulk),
            essmu_tail_med = median(tail), essmu_tail_min = minimum(tail),
            rhatmu_rank_max = maximum(rh),
            essll_bulk = ess(L; kind = :bulk), essll_tail = ess(L; kind = :tail),
            rhatll_rank = rhat(L; kind = :rank))
end

files = sort(filter(f -> endswith(f, "_samples.jld"), readdir(SAMPDIR)))
mine = files[task_id:njobs:length(files)]
println("task $task_id/$njobs: $(length(mine)) of $(length(files)) cells")
flush(stdout)

for (i, f) in enumerate(mine)
    stem = replace(f, "_samples.jld" => "")
    out = joinpath(OUTDIR, stem * ".csv")
    (isfile(out) && filesize(out) > 0) && continue
    parts = split(stem, "_")
    seed = parse(Int, parts[end])
    ds = parts[1]
    mname = join(parts[2:end-1], "_")
    try
        r = diag_cell(joinpath(SAMPDIR, f))
        CSV.write(out, DataFrame(; dataset = ds, model = mname, seed = seed,
                                 pairs(r)...))
    catch e
        println("FAILED $stem: ", first(sprint(showerror, e), 300))
    end
    if i % 25 == 0
        println("  $i/$(length(mine))"); flush(stdout)
    end
end
println("task $task_id complete")
