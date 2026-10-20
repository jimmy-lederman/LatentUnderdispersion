# Is eps needed, and does its value matter? (Re-run of eps_necessity.jl at the CURRENT
# configuration: hybrid sampler, Q = 1. The August version used K=5/Q=25 under the thinning
# sampler, whose Rhat numbers are uninformative now.)
#
#   julia -t 6 --project=. revision2_experiments/covid/eps_necessity2.jl [ncty] [nsweeps] [nchains]
#
# A referee asked what eps is and why it is included. mu_{i,t} = Y_{i,t-1} + eps + theta'phi,
# so eps is one additive constant shared by all cells. Of the two reasons on record, the
# "anchors the factors" one is already dead (free eps collapses to 0.004 against a mean
# increment of ~14, i.e. 0.03% of the growth term). What is left is a POSITIVITY FLOOR: phi's
# gamma-Markov chain has a near-absorbing state at zero (shape 0.01 + 100 phi_{t-1}), so if
# every factor underflowed at some t, a county with Y_{i,t-1} = 0 would have mu = 0 exactly
# and a zero likelihood for any positive count.
#
# Arms, all with the hybrid sampler and eps re-pinned after every sweep where it is pinned:
#   free     eps drawn as the model specifies (confirms the collapse to ~0.004)
#   0.5      what the reported runs actually use
#   0.004    where free inference puts it
#   0.0      no floor at all -- does anything actually break?
#
# Judged on: held-out log density per cell (must be ~unchanged), the smallest phi and the
# smallest mu reached (does the floor ever bind?), and whether any non-finite likelihood
# occurs. If the fit is flat across arms and nothing breaks, eps is inert and the paper can
# state it as a fixed constant for positivity -- or drop it.

ROOTDIR = abspath(joinpath(@__DIR__, "../.."))
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra

ncty    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 150
nsweeps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1200
nchains = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
println("Julia ", VERSION, "  threads = ", Threads.nthreads(), "  BLAS = ", BLAS.get_num_threads())

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
step = max(1, size(cumdf,1) ÷ ncty)
cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
Y0_N = Int.(reshape(cumdf[:,1], :, 1))
Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)

# same held-out construction as the cluster driver: last 10 days of a random 20% of counties
Random.seed!(101)
mc = rand(N) .< 0.2
mask_NM = falses(N, M)
for t in (M-9):M; mask_NM[:, t] = mc; end
heldout = findall(mask_NM)

inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)
println("N = $N, M = $M, held-out cells = $(length(heldout)), mean increment = $(round(mean_inc, digits=2))")

K, Q, Dmax = 5, 1, 9
model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
               false,  # hier_b
               true,   # scale_move
               false,  # eps_mh
               true,   # use_offset
               false,  # shear_move
               true,   # collapsed: theta by the slice step
               false,  # iid_phi
               true,   # phi_da: HYBRID
               false)  # dir_theta

function run_chain(seed, epsval)          # epsval = nothing means "free"
    Random.seed!(seed)
    st = sample_prior(model, info)
    isnothing(epsval) || (st["eps"] = epsval)
    target = max(mean_inc - st["eps"], 1e-6)
    st["V_KM"] = st["V_KM"] .* (target / max(mean(st["U_NK"] * st["V_KM"]), 1e-12))
    ll = Float64[]; epstrace = Float64[]; phimin = Inf; mumin = Inf; nonfinite = 0
    for s in 1:nsweeps
        (_d, st) = backward_sample(model, data, st, mask_NM)
        isnothing(epsval) || (st["eps"] = epsval)
        if s > nsweeps ÷ 2
            U = st["U_NK"]::Matrix{Float64}; V = st["V_KM"]::Matrix{Float64}
            l = 0.0
            for ci in heldout; l += evalulateLogLikelihood(model, st, data, info, ci[1], ci[2]); end
            isfinite(l) ? push!(ll, l) : (nonfinite += 1)
            push!(epstrace, st["eps"])
            phimin = min(phimin, minimum(V))
            G = U * V
            # smallest rate anywhere: the offset is 0 only where Y_{i,t-1} = 0
            mn = Inf
            for m in 1:M, n in 1:N
                off = m == 1 ? Float64(Y0_N[n]) : Float64(Y_NM[n,m-1])
                mn = min(mn, off + st["eps"] + G[n,m])
            end
            mumin = min(mumin, mn)
        end
    end
    (ll, epstrace, phimin, mumin, nonfinite)
end

arms = [("free", nothing), ("0.5 (reported)", 0.5), ("0.004", 0.004), ("0.0", 0.0)]
@printf("\n%-16s %14s %12s %10s %12s %12s %10s\n",
        "arm", "held-out/cell", "spread", "eps(end)", "min phi", "min mu", "nonfinite")
for (label, ev) in arms
    res = [run_chain(600+c, ev) for c in 1:nchains]
    perc = [mean(r[1])/length(heldout) for r in res]
    @printf("%-16s %14.4f %12.4f %10.5f %12.3e %12.3e %10d\n",
            label, mean(perc), maximum(perc) - minimum(perc),
            mean(r[2][end] for r in res), minimum(r[3] for r in res),
            minimum(r[4] for r in res), sum(r[5] for r in res))
    flush(stdout)
end
