# Is eps still necessary?
#
#   julia -t 8 --project=. revision2_experiments/covid/eps_necessity.jl [nsweeps] [nchains] [ncounties]
#
# eps was kept in the model because, once phi got its time-varying gamma-Markov prior, it
# "anchored" the factors. Two things that plausibly meant:
#   (1) without eps one of the K factors is spent representing the baseline growth common to
#       all counties and times, leaving K-1 interpretable waves;
#   (2) eps floors mu strictly above Y_{i,t-1}. The phi chain has a near-absorbing state at
#       zero (its shape is c*phi_{t-1} + d), so if the factors collapsed there, a county at
#       zero cases reporting its first case would get a near-zero likelihood.
#
# What we now know argues against (1): on real data eps collapses from 0.2 at initialisation
# to 0.004, against a mean observed increment of ~14. It contributes ~0.03% of the growth
# term, so the factors have absorbed the baseline regardless. Meanwhile eps is measurably
# harmful -- unidentified in practice (its posterior is set by wherever it starts), too slow
# to move, and confounded with the overall factor level, which is one of the slow directions
# behind the convergence problem.
#
# (2) is the real question and it is empirical. Three arms:
#   free    eps sampled as now
#   zero    eps pinned at 0 -- removes the ridge entirely; does anything collapse?
#   fixed   eps pinned at a constant -- removes the ridge but KEEPS the floor. Since eps is
#           unidentified anyway, fixing it is arguably more honest than inferring it, and
#           the paper would simply state the value.
#
# Judged on held-out density (must not get worse) and on Rhat/ESS for the reported
# quantities (the hoped-for gain).

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra, LogExpFunctions
using MCMCDiagnosticTools, Combinatorics

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
ncty    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 150
THIN    = 8
println("Julia ", VERSION, "  threads = ", Threads.nthreads())

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
if ncty < size(cumdf, 1)
    step = max(1, size(cumdf,1) ÷ ncty)
    cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
end
Y0_N = Int.(reshape(cumdf[:,1], :, 1))
Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)
K, Q, Dmax = 5, 25, 9

Random.seed!(101)
mask_NM = falses(N, M)
mc = rand(N) .< 0.2
for t in (M-9):M; mask_NM[:, t] = mc; end
heldout = findall(mask_NM)
Random.seed!(20260809)
cells = [(rand(1:N), rand(1:M)) for _ in 1:150]
icty  = sort(randperm(N)[1:60])
println("N = $N, M = $M, held-out = $(length(heldout)), $nchains chains x $nsweeps sweeps\n")

inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)
model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false, true)

function run_chain(seed, epsmode)
    Random.seed!(seed)
    st = sample_prior(model, info)
    epsmode == :zero  && (st["eps"] = 0.0)
    epsmode == :fixed && (st["eps"] = 0.5)
    target = max(mean_inc - st["eps"], 1e-6)
    st["V_KM"] = st["V_KM"] .* (target / max(mean(st["U_NK"] * st["V_KM"]), 1e-12))
    ll = Float64[]; mu = Vector{Vector{Float64}}()
    th = Vector{Matrix{Float64}}(); ph = Vector{Matrix{Float64}}()
    phimin = Float64[]
    for s in 1:nsweeps
        (_d, st) = backward_sample(model, data, st, mask_NM)
        epsmode == :zero  && (st["eps"] = 0.0)
        epsmode == :fixed && (st["eps"] = 0.5)
        if s > nsweeps ÷ 2 && s % THIN == 0
            U = st["U_NK"]::Matrix{Float64}; V = st["V_KM"]::Matrix{Float64}
            l = 0.0
            for ci in heldout; l += evalulateLogLikelihood(model, st, data, info, ci[1], ci[2]); end
            push!(ll, l)
            push!(mu, Float64[sum(U[n,:] .* V[:,m]) for (n,m) in cells])
            sk = vec(sum(U, dims=1))
            push!(th, U[icty,:] ./ reshape(sk,1,K)); push!(ph, V .* reshape(sk,K,1))
            push!(phimin, minimum(V))
        end
    end
    return ll, mu, th, ph, phimin
end

safe(f,x) = try f(x) catch; NaN end
r3(A) = reshape(A, size(A,1), size(A,2), 1)
# Report the DISTRIBUTION of Rhat over parameters, not just the max. The max over 150
# cells is an extreme statistic -- a few genuinely hard cells (low-count counties) can sit
# at 3 while the median cell is converged, and quoting only the max reads as "nothing
# converged" when the truth may be "most things converged and a tail did not".
function dg(A)
    rs = filter(isfinite, [safe(_ -> rhat(r3(A[:,:,p]))[1], p) for p in 1:size(A,3)])
    e = mean(filter(isfinite, [safe(x -> ess(r3(reshape(x,:,1)); kind=:bulk)[1], A[:,c,p])
                               for c in 1:size(A,2), p in 1:size(A,3)]))
    return (median(rs), quantile(rs, 0.9), maximum(rs)), e
end

function arm(label, epsmode)
    res = [run_chain(600+c, epsmode) for c in 1:nchains]
    nk = length(res[1][1])
    L = Array{Float64}(undef, nk, nchains, 1)
    for c in 1:nchains; L[:,c,1] = res[c][1]; end
    MU = Array{Float64}(undef, nk, nchains, length(cells))
    for c in 1:nchains, s in 1:nk, p in eachindex(cells); MU[s,c,p] = res[c][2][s][p]; end
    prof = [mean(res[c][3]) for c in 1:nchains]
    perms = collect(permutations(1:K)); best = [collect(1:K) for _ in 1:nchains]
    for c in 2:nchains
        best[c] = perms[argmin([sum(abs2, prof[1] .- prof[c][:,p]) for p in perms])]
    end
    nI = length(icty)
    TH = Array{Float64}(undef, nk, nchains, nI*K); PH = Array{Float64}(undef, nk, nchains, K*M)
    for c in 1:nchains, s in 1:nk
        TH[s,c,:] = vec(res[c][3][s][:, best[c]]); PH[s,c,:] = vec(res[c][4][s][best[c], :])
    end
    # held-out information rate, pooled over chains
    allll = reduce(vcat, [res[c][1] for c in 1:nchains])
    ir = (logsumexp(allll) - log(length(allll))) / length(heldout)
    rl,el = dg(L); rm,em = dg(MU); rt,et = dg(TH); rp,ep = dg(PH)
    pmin = minimum(reduce(vcat, [res[c][5] for c in 1:nchains]))
    @printf("%-12s ir/cell %8.4f  Rhat med/p90/max:  mu %.2f/%.2f/%.2f   th %.2f/%.2f/%.2f   ph %.2f/%.2f/%.2f  | ESS mu %5.1f th %5.1f\n",
            label, ir, rm..., rt..., rp..., em, et)
    flush(stdout)
end

arm("eps free",  :free)
arm("eps = 0",   :zero)
arm("eps = 0.5", :fixed)
println("\nDropping or fixing eps removes a known ridge. It is a win if held-out density does")
println("not get worse AND Rhat/ESS improve. Watch min phi for the collapse eps was there to")
println("prevent.")
