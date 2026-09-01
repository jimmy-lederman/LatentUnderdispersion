# One cell of the non-conjugacy experiment: (dataset, model, a=c, seed).
#
# Every model is given the SAME Dirichlet concentration a and gamma shape c.
# The order statistic and Poisson models absorb them into conjugate draws --
# Dirichlet(a + counts) and Gamma(c + counts, ...) -- at no cost. STAR-NN must
# fall back to MH or slice for both updates. STAR-G is the control: its Gaussian
# priors have no a or c, so it is unaffected by construction.
#
# Usage: julia --project=../../.. run_sparse_cells.jl <dataset> <model> <ac> <seed> [nsamples nburnin nchains]
#   dataset: SparseCMP | CMP
#   model:   poisson | medpois | starnn_slice | starnn_mh | star_g

include("../../../models/other_models/PoissonMF.jl")
include("../../../models/other_models/OrderStatisticPoissonMFprior.jl")
include("../../../models/STARmodels/STARMF.jl")
include("../../../models/STARmodels/revision2/STARMFNNsparse.jl")
include("../../../revison_experiments/mcmc_stuff.jl")
import MCMCDiagnosticTools as MCD      # qualified: mcmc_stuff.jl defines its own `ess`
using CSV, DataFrames, Random, Statistics, LinearAlgebra, Printf, Combinatorics

ds     = ARGS[1]
mname  = ARGS[2]
ac     = parse(Float64, ARGS[3])
seed   = parse(Int, ARGS[4])
NS     = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2000
NB     = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 2000
NC     = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 4

const HERE = @__DIR__
DATA = Dict(
  "SparseCMP" => (joinpath(HERE, "data/SparseCMPfactor.csv"), joinpath(HERE, "data/SparseCMPU_NK.csv")),
  "CMP"       => (joinpath(HERE, "../../../revison_experiments/factorexperiment/data/CMPfactor.csv"),
                  joinpath(HERE, "../../../revison_experiments/factorexperiment/data/CMPU_NK2.csv")))

readm(f) = Matrix(CSV.read(f, DataFrame))[1:end, 2:end]
Y = readm(DATA[ds][1]); N, M = size(Y); K = 3
raw = readm(DATA[ds][2]); Utrue = size(raw, 1) == N ? Float64.(raw) : Float64.(permutedims(raw))
data = Dict("Y_NM" => Y)

function build(mname, ac)
    if mname == "poisson";      return PoissonMF(N, M, K, ac, 1.0, ac, 0.01)
    elseif mname == "medpois";  return OrderStatisticPoissonMF(N, M, K, ac, 1.0, ac, 0.01, 15, 2, 0.5)
    elseif mname == "starnn_slice"; return STARMFNNsp(N, M, K, ac, ac, 0.01, 1.0, 1.0, identity, identity, :slice)
    elseif mname == "starnn_mh";    return STARMFNNsp(N, M, K, ac, ac, 0.01, 1.0, 1.0, identity, identity, :mh)
    elseif mname == "star_g";   return STARMF(N, M, K, 1.0, 1.0, identity, identity)   # a,c do not apply
    else error("unknown model $mname") end
end

Random.seed!(seed)
mask = rand(N, M) .< 0.2
heldout = [(i, j) for i in 1:N for j in 1:M if mask[i, j]]
model = build(mname, ac)

samples = []
t = @elapsed for ch in 1:NC
    s = fit(model, copy(data); nsamples=NS, nburnin=NB, nthin=1, mask=mask,
            initseed=seed * 1000 + ch, verbose=false)
    push!(samples, s)
end

mu = Array{Float64}(undef, NS, NC, N * M)
ll = Array{Float64}(undef, NS, NC)
acc = Float64[]
for ch in 1:NC, s in 1:NS
    st = samples[ch][s]
    mu[s, ch, :] = vec(st["U_NK"] * st["V_KM"])
    ll[s, ch] = mean(evalulateLogLikelihood(model, st, data, nothing, i, j) for (i, j) in heldout)
    haskey(st, "accU") && !isnan(st["accU"]) && push!(acc, st["accU"])
end
essb = [MCD.ess(@view(mu[:, :, p]); kind=:bulk) for p in 1:(N * M)]
esst = [MCD.ess(@view(mu[:, :, p]); kind=:tail) for p in 1:(N * M)]
rh   = [MCD.rhat(@view(mu[:, :, p]); kind=:rank) for p in 1:(N * M)]
ir = evaluateInfoRate(model, data, vcat(samples...), mask=mask, verbose=false)

bp(Ut, U) = (S = abs.(cor(Ut, U)); best = -Inf; p0 = collect(1:K);
             for p in permutations(1:K); sc = sum(S[k, p[k]] for k in 1:K); sc > best && (best = sc; p0 = collect(p)); end; p0)
# posterior mean loadings, then permutation-aligned to the truth. Cosine is a
# secondary metric here (ESS/s is the headline), so no per-draw label-switch
# alignment is applied -- see cluster/recompute_cosine.jl for the careful version.
allst = vcat(samples...)
Um = zeros(N, K); for st in allst; Um .+= st["U_NK"]; end; Um ./= length(allst)
Um = Um[:, bp(Utrue, Um)]
cosine = mean(abs(dot(Utrue[:, k], Um[:, k])) / (norm(Utrue[:, k]) * norm(Um[:, k])) for k in 1:K)

row = DataFrame(dataset=ds, model=mname, ac=ac, seed=seed,
                inforate=ir, essmu_bulk=median(essb), essmu_tail=median(esst),
                rhat_max=maximum(rh), essll_bulk=MCD.ess(ll; kind=:bulk),
                cosine=cosine, time_s=t, ess_per_s=median(essb) / t,
                accept=isempty(acc) ? NaN : mean(acc), iters=NC * (NS + NB))
outdir = joinpath(HERE, "results"); mkpath(outdir)
CSV.write(joinpath(outdir, "$(ds)_$(mname)_$(ac)_$(seed).csv"), row)
@printf("done %s %s a=c=%.2f seed %d: ir %.4f essmu %.0f ess/s %.1f cos %.3f acc %.2f (%.0fs)\n",
        ds, mname, ac, seed, ir, median(essb), median(essb) / t, cosine,
        isempty(acc) ? NaN : mean(acc), t)
