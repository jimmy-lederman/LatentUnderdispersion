# Which parameters of the Section 7.2 model mix, and which do not — and WHY not.
#
#   julia -t 6 --project=. revision2_experiments/covid/diagnose_mixing.jl [nsweeps] [nchains] [ncounties]
#
# Three different pathologies all show up as a large Rhat, and they need different fixes,
# so this script separates them:
#
#   (1) SLOW MIXING       high within-chain autocorrelation. Per-chain ESS is low even
#                         though the chain is stationary. Fix = a better sampler.
#   (2) NOT CONVERGED     the chain is still drifting. Per-chain ESS may look fine while
#                         the chain's own first-half and second-half means differ.
#                         Fix = longer burn-in, or a better sampler if the drift is slow
#                         because of (1).
#   (3) NON-IDENTIFIED    each chain mixes fine but they disagree, because the parameter
#                         is only identified up to a transformation (per-factor scale,
#                         label permutation, rotation of the Polya-Gamma space).
#                         Fix = report an identified functional, or constrain the model.
#
# The discriminators reported per quantity:
#   ESS/chain   mean over chains of the SINGLE-chain bulk ESS  -> low means (1)
#   drift       |mean(2nd half) - mean(1st half)| / sd, per chain, max over chains
#                                                              -> large means (2)
#   Rhat        rank-normalized split-Rhat across chains        -> large with good
#                                                                 ESS/chain and small
#                                                                 drift means (3)
#
# Factor quantities are reported three ways to localise the non-identifiability:
#   raw         theta_{i,k}, phi_{k,t} as sampled
#   scaled      theta_{i,k}/s_k and s_k*phi_{k,t}, s_k = sum_i theta_{i,k}
#               (removes the per-factor multiplicative ridge)
#   aligned     scaled, after matching factor labels across chains by brute-force
#               permutation on the mean loading profiles (removes label switching)

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra, Combinatorics
using MCMCDiagnosticTools

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
ncty    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 300
nkeep   = nsweeps ÷ 2
println("Julia ", VERSION, "  threads = ", Threads.nthreads())
println("$nchains chains x $nsweeps sweeps, keeping the last $nkeep")

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
if ncty < size(cumdf, 1)
    step = max(1, size(cumdf,1) ÷ ncty)
    cumdf = cumdf[1:step:end, :][1:min(ncty, end), :]
end
Y0_N = Int.(reshape(cumdf[:,1], :, 1)); Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)
K, Q, Dmax = 5, 25, 9
println("N = $N counties, M = $M days, K = $K, Q = $Q, Dmax = $Dmax")

Random.seed!(20260809)
cells = [(rand(1:N), rand(1:M)) for _ in 1:150]      # for mu, rho, D
icty  = sort(randperm(N)[1:60])                      # counties whose theta we track

model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false)

function run_chain(model, data, info, seed, nsweeps, nkeep, cells, icty)
    Random.seed!(seed)
    state = sample_prior(model, info)
    sc = Dict(k => Float64[] for k in ("eps","s2tau","s2beta","Dmean"))
    MU  = Vector{Vector{Float64}}(); RHO = Vector{Vector{Float64}}()
    DC  = Vector{Vector{Float64}}()
    TH  = Vector{Matrix{Float64}}(); PH = Vector{Matrix{Float64}}()
    t0 = time()
    for s in 1:nsweeps
        (_d, state) = backward_sample(model, data, state)
        if s > nsweeps - nkeep
            U = state["U_NK"]::Matrix{Float64}; V = state["V_KM"]::Matrix{Float64}
            B = state["Beta_NQ"]::Matrix{Float64}; T = state["Tau_QM"]::Matrix{Float64}
            D = state["D_NM"]
            push!(sc["eps"], state["eps"]); push!(sc["s2tau"], state["sigma2_tau"])
            push!(sc["s2beta"], state["sigma2_beta"]); push!(sc["Dmean"], mean(D))
            push!(MU,  Float64[sum(U[n,:] .* V[:,m]) for (n,m) in cells])
            push!(RHO, Float64[logistic(dot(B[n,:], T[:,m])) for (n,m) in cells])
            push!(DC,  Float64[D[n,m] for (n,m) in cells])
            push!(TH,  U[icty, :]); push!(PH, copy(V))
        end
        s % 500 == 0 && (@printf("    seed %d: %d/%d (%.1f min)\n", seed, s, nsweeps, (time()-t0)/60); flush(stdout))
    end
    return sc, MU, RHO, DC, TH, PH
end

res = [run_chain(model, data, info, 1000+c, nsweeps, nkeep, cells, icty) for c in 1:nchains]

# ---------------------------------------------------------------- diagnostics helpers
safe(f, x) = try f(x) catch; NaN end
# stack a per-chain vector-of-vectors into (draws, chains, params)
function stack(per, P)
    A = Array{Float64}(undef, nkeep, nchains, P)
    for c in 1:nchains, s in 1:nkeep, p in 1:P
        A[s, c, p] = per[c][s][p]
    end
    A
end
perchain_ess(A) = mean(filter(isfinite,
    [safe(x -> ess(reshape(x, :, 1, 1); kind=:bulk)[1], A[:, c, p]) for c in 1:size(A,2), p in 1:size(A,3)]))
function drift(A)
    h = nkeep ÷ 2
    d = Float64[]
    for c in 1:size(A,2), p in 1:size(A,3)
        x = @view A[:, c, p]
        s = std(x)
        s > 0 && push!(d, abs(mean(x[h+1:end]) - mean(x[1:h])) / s)
    end
    isempty(d) ? NaN : maximum(d)
end
poolrhat(A) = maximum(filter(isfinite, vec([safe(x -> rhat(reshape(A[:,:,p], nkeep, nchains, 1))[1], p) for p in 1:size(A,3)])))

rows = Tuple{String,String,Array{Float64,3}}[]
for k in ("eps","s2tau","s2beta","Dmean")
    A = Array{Float64}(undef, nkeep, nchains, 1)
    for c in 1:nchains; A[:, c, 1] = res[c][1][k]; end
    push!(rows, ("scalar", k, A))
end
push!(rows, ("identified", "mu (cells)",  stack([r[2] for r in res], length(cells))))
push!(rows, ("identified", "rho (cells)", stack([r[3] for r in res], length(cells))))
push!(rows, ("identified", "D (cells)",   stack([r[4] for r in res], length(cells))))

# --- factors: raw, scaled, aligned
TH = [r[5] for r in res]; PH = [r[6] for r in res]
nI = length(icty)
raw_th = Array{Float64}(undef, nkeep, nchains, nI*K)
raw_ph = Array{Float64}(undef, nkeep, nchains, K*M)
scl_th = similar(raw_th); scl_ph = similar(raw_ph)
sk_all = Array{Float64}(undef, nkeep, nchains, K)
for c in 1:nchains, s in 1:nkeep
    th = TH[c][s]; ph = PH[c][s]
    sk = vec(sum(th, dims=1))                 # per-factor scale from the tracked counties
    sk_all[s, c, :] = sk
    raw_th[s, c, :] = vec(th)
    raw_ph[s, c, :] = vec(ph)
    scl_th[s, c, :] = vec(th ./ reshape(sk, 1, K))
    scl_ph[s, c, :] = vec(ph .* reshape(sk, K, 1))
end
push!(rows, ("factor raw",    "theta",       raw_th))
push!(rows, ("factor raw",    "phi",         raw_ph))
push!(rows, ("factor raw",    "s_k (scale)", sk_all))
push!(rows, ("factor scaled", "theta/s_k",   scl_th))
push!(rows, ("factor scaled", "s_k*phi",     scl_ph))

# label alignment: permute each chain's factors to best match chain 1's mean profile
prof = [reshape(mean(scl_th[:, c, :], dims=1), nI, K) for c in 1:nchains]
perms = collect(permutations(1:K))
best = [collect(1:K) for _ in 1:nchains]
for c in 2:nchains
    scores = [sum(abs2, prof[1] .- prof[c][:, p]) for p in perms]
    best[c] = perms[argmin(scores)]
end
println("\nfactor permutation chosen per chain (vs chain 1): ", best)
ali_th = similar(scl_th); ali_ph = similar(scl_ph)
for c in 1:nchains, s in 1:nkeep
    p = best[c]
    ali_th[s, c, :] = vec(reshape(scl_th[s, c, :], nI, K)[:, p])
    ali_ph[s, c, :] = vec(reshape(scl_ph[s, c, :], K, M)[p, :])
end
push!(rows, ("factor aligned", "theta/s_k", ali_th))
push!(rows, ("factor aligned", "s_k*phi",   ali_ph))

println()
@printf("%-15s %-14s %10s %10s %8s   %s\n", "group", "quantity", "ESS/chain", "drift", "Rhat", "verdict")
println("-"^92)
for (g, nm, A) in rows
    e = perchain_ess(A); d = drift(A); r = poolrhat(A)
    verdict = if !isfinite(r); "--"
        elseif e < 20 && d > 0.5;  "NOT CONVERGED + slow"
        elseif e < 20;             "SLOW MIXING"
        elseif r > 1.05 && d < 0.3;"NON-IDENTIFIED"
        elseif r > 1.05;           "not converged"
        else                       "ok" end
    @printf("%-15s %-14s %10.1f %10.2f %8.2f   %s\n", g, nm, e, d, r, verdict)
end
println("\ndrift = |mean(2nd half) - mean(1st half)| / sd, max over chains and components")
println("ESS/chain is SINGLE-chain bulk ESS out of $nkeep draws, averaged over chains")
