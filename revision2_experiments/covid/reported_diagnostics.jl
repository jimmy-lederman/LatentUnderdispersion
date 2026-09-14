# Diagnostics for the two quantities Section 7.2 actually reports.
#
#   julia -t 4 --project=. revision2_experiments/covid/reported_diagnostics.jl [nsweeps] [nchains] [ncounties]
#
# Under the framing that D is a computational device rather than an estimand, the reported
# claims rest on exactly two things:
#
#   HELD-OUT DENSITY   the forecasting claim. Aggregate and identified, so it should be the
#                      best behaved thing in the model.
#   THE FACTORS        Figure 10 interprets them as waves of the pandemic, so their
#                      diagnostics have to be reported on the INTERPRETED objects: the
#                      loading profile theta_{i,k}/s_k and the scaled trajectory
#                      s_k*phi_{k,t}, with s_k = sum_i theta_{i,k}, aligned across chains.
#
# Raw factor entries are not reported here because they are identified only up to the
# per-factor scale and the label permutation. But note the prior evidence that this may not
# be enough: on the real subset, per-cell mu sat at Rhat ~2 after 15000 sweeps, and mu is
# ALREADY invariant to both the scale and the permutation -- so if that reproduces here, the
# chains disagree about the fitted surface itself and no post-hoc normalisation rescues the
# factors.
#
# Everything is reported over three run-length windows so the number of sweeps needed can be
# read off for each quantity separately, rather than assumed common.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra, LogExpFunctions
using MCMCDiagnosticTools, Combinatorics

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
ncty    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 200
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
K       = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
# Shape of the county-factor prior theta_{i,k} ~ Gamma(a_theta, b). a = 1 is the current
# exponential, which exerts no pressure toward sparsity. a < 1 puts prior mass near zero, so
# each county loads on a FEW factors rather than all K -- which is the separability that
# makes a non-negative factorisation identifiable. Conjugate: a is only a shape, the update
# is unchanged.
a_theta = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0
Q       = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 25
sV1     = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.01
# ARGS[8] = 0 fits INCREMENTS with no offset in the rate. The data below is replaced by the
# clipped first differences, and mu = eps + theta'phi. This is the diagnostic for how much
# of the factors' poor identifiability is caused by Y_{i,t-1}.
useoff  = length(ARGS) >= 8 ? parse(Int, ARGS[8]) == 1 : true
# ARGS[9] = 1 enables the pairwise shear move (the off-diagonal partner of the scale move).
shear   = length(ARGS) >= 9 ? parse(Int, ARGS[9]) == 1 : false
# ARGS[10] = c, the gamma-Markov coupling on phi. c=100 (the paper) expects phi to move ~6%
# per day and makes the landscape over factorisations rugged: every mu-preserving shear
# costs 50-400 nats of prior. Weakening c is the direct test of whether that ruggedness is
# what keeps the chains in different factorisations.
cphi    = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 100.0
# ARGS[11] = 1 initialises theta, phi from a KL-NMF of the clipped increments, seeded PER
# CHAIN so the chains start from different NMF solutions (dispersed within what the data
# supports) rather than from prior draws. On the 100-county subset NMF solutions from 12
# random starts agree to cosine 0.999, while MCMC chains from prior draws disagree at phi
# Rhat ~2.3 -- so the test is whether data-informed starts remove that disagreement.
nmfinit = length(ARGS) >= 11 ? parse(Int, ARGS[11]) == 1 : false
# ARGS[12] = Dfix > 0 PINS D_{i,t} = Dfix and skips the whole D / Polya-Gamma / beta / tau
# branch. Diagnostic for whether that layer is what keeps theta, phi, mu from converging:
# rho had the worst Rhat of any quantity (6.26) and mean D sits at Rhat 3 on the full matrix
# even though D is drawn from its exact conditional.
dfix    = length(ARGS) >= 12 ? parse(Int, ARGS[12]) : 0
function nmf_kl(V, K, rng, iters=300)
    n, m = size(V); W = rand(rng, n, K) .+ 0.1; H = rand(rng, K, m) .+ 0.1
    for _ in 1:iters
        WH = max.(W*H, 1e-10); H .*= (transpose(W) * (V ./ WH)) ./ max.(transpose(sum(W, dims=1)), 1e-10)
        WH = max.(W*H, 1e-10); W .*= ((V ./ WH) * transpose(H)) ./ max.(transpose(sum(H, dims=2)), 1e-10)
    end
    return W, H
end
Dmax    = 9
println("K = ", K, "  a_theta = ", a_theta, "  Q = ", Q, "  start_V1 = ", sV1, "  shear = ", shear, "  c = ", cphi, "  nmfinit = ", nmfinit, "  Dfix = ", dfix)

Random.seed!(101)
missing_counties = rand(N) .< 0.2
mask_NM = falses(N, M)
for t in (M-9):M; mask_NM[:, t] = missing_counties; end
heldout = findall(mask_NM)
Random.seed!(20260809)
cells = [(rand(1:N), rand(1:M)) for _ in 1:150]
icty  = sort(randperm(N)[1:60])
println("N = $N, M = $M, held-out cells = $(length(heldout))")
println("$nchains chains x $nsweeps sweeps, thin $THIN\n")

inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)
if !useoff
    data["Y_NM"] = Int.(round.(inc))          # model the increments themselves
    Y_NM = data["Y_NM"]
    info["Y0_N"] = zeros(Int, N, 1)
    println("NO OFFSET: fitting increments directly, mean = ", round(mean(Y_NM), digits=2))
end

# ARG 13: 1 = the collapsed (thinning-free, slice-sampled) theta / phi / eps update.
# ARG 13: 1 = fully collapsed theta / phi / eps; 2 = hybrid (theta collapsed, phi by the PGDS pass)
collapsed = length(ARGS) >= 13 && ARGS[13] in ("1", "2")
hybrid    = length(ARGS) >= 13 && ARGS[13] == "2"
# ARG 14: 1 = Dirichlet theta (columns on the simplex; hybrid sampler). phi then carries
# the scale, so c comes from the scale rule c_for_scale(0.02, phibar) with phibar = the
# total daily increment / K, and E[phi_1] = phibar; ARG 10 (c) is ignored in that case.
dirichlet = length(ARGS) >= 14 && ARGS[14] == "1"
phibar = mean_inc * N / K
if dirichlet
    cphi = c_for_scale(0.02, phibar); sV1 = 1.0; sV2 = 1 / phibar; collapsed = true; hybrid = true
else
    sV2 = 1.0
end
println("sampler: ", hybrid ? "hybrid" : collapsed ? "collapsed" : "thinning",
        dirichlet ? @sprintf("   Dirichlet theta, phibar = %.0f, c = %.3f", phibar, cphi) : "")
model = covid4(N, M, K, Q, Dmax, a_theta, 1.0, cphi, 0.1, 0.5, 1.0,
               sV1, sV2, 1.0, 1.0, 1.0, 0.0, 0.0, false, true, false, useoff, shear, collapsed, false, hybrid, dirichlet)

function run_chain(seed)
    Random.seed!(seed)
    st = sample_prior(model, info)
    if nmfinit
        W, H = nmf_kl(inc, K, MersenneTwister(seed))
        # multiplicative updates leave exact zeros; the model's parameters are strictly positive
        st["U_NK"] = max.(W, 1e-6); st["V_KM"] = max.(H, 1e-6)
    else
        target = max(mean_inc - st["eps"], 1e-6)
        cur = mean(st["U_NK"] * st["V_KM"])
        st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))
    end
    dfix > 0 && (st["D_NM"] = fill(dfix, N, M))
    sw = Int[]; ll = Float64[]; acc = Float64[]
    mu = Vector{Vector{Float64}}(); th = Vector{Matrix{Float64}}(); ph = Vector{Matrix{Float64}}()
    t0 = time()
    for s in 1:nsweeps
        (_d, st) = dfix > 0 ? backward_sample(model, data, st, mask_NM; skipupdate=["D_NM"]) :
                              backward_sample(model, data, st, mask_NM)
        dfix > 0 && (st["D_NM"] = fill(dfix, N, M))
        shear && push!(acc, st["shear_acc"])
        if s % THIN == 0
            U = st["U_NK"]::Matrix{Float64}; V = st["V_KM"]::Matrix{Float64}
            l = 0.0
            for ci in heldout
                l += evalulateLogLikelihood(model, st, data, info, ci[1], ci[2])
            end
            push!(sw, s); push!(ll, l)
            push!(mu, Float64[sum(U[n,:] .* V[:,m]) for (n,m) in cells])
            sk = vec(sum(U, dims=1))
            push!(th, U[icty, :] ./ reshape(sk, 1, K))
            push!(ph, V .* reshape(sk, K, 1))
        end
        s % 2000 == 0 && (@printf("    seed %d: %d/%d (%.1f min)%s\n", seed, s, nsweeps, (time()-t0)/60,
                                   shear ? @sprintf("  shear acc %.2f", mean(acc[max(1,end-1999):end])) : ""); flush(stdout))
    end
    return sw, ll, mu, th, ph
end

res = Vector{Any}(undef, nchains)
for c in 1:nchains
    res[c] = run_chain(500 + c)
    @printf("  chain %d done\n", c); flush(stdout)
end

sw = res[1][1]
safe(f, x) = try f(x) catch; NaN end
r3(A) = reshape(A, size(A,1), size(A,2), 1)
# Rhat is reported as median / p90 / max over parameters. Quoting only the max over 150
# cells reads as "nothing converged" even when the median parameter is fine -- a handful of
# hard cells dominate it.
function diag3(A)   # (draws, chains, params) -> ((med,p90,max) Rhat, mean per-chain ESS)
    rs = filter(isfinite, [safe(_ -> rhat(r3(A[:,:,p]))[1], p) for p in 1:size(A,3)])
    e = mean(filter(isfinite,
        [safe(x -> ess(r3(reshape(x, :, 1)); kind=:bulk)[1], A[:,c,p])
         for c in 1:size(A,2), p in 1:size(A,3)]))
    return (median(rs), quantile(rs, 0.9), maximum(rs)), e
end

# Relabel EVERY DRAW, not every chain.
#
# Aligning a chain's MEAN profile to another chain's mean is only valid if the labels are
# stable within a chain. They are not: label switching across draws inside a chain is
# standard in factor models, and then the chain mean is a blend of permutations, matching
# against it is meaningless, and one permutation per chain fixes nothing. Inflated Rhat
# follows from the relabelling, not from the sampler.
#
# This is the usual Stephens-style scheme: start from a reference, assign each draw the
# permutation minimising distance to it, recompute the reference as the mean of the
# relabelled draws, iterate. Returns a permutation PER DRAW PER CHAIN.
function align_draws(thw, niter=8)   # thw[c][s] :: nI x K
    perms = collect(permutations(1:K))
    ref = mean(thw[1])
    assign = [[collect(1:K) for _ in eachindex(thw[c])] for c in 1:nchains]
    for _ in 1:niter
        acc = zeros(size(ref)); n = 0
        for c in 1:nchains, s in eachindex(thw[c])
            p = perms[argmin([sum(abs2, ref .- thw[c][s][:, pp]) for pp in perms])]
            assign[c][s] = p
            acc .+= thw[c][s][:, p]; n += 1
        end
        ref = acc ./ n
    end
    return assign
end

windows = [(1000, 2000), (2000, 4000), (4000, nsweeps)]
println()
@printf("%-14s %-26s %19s %10s\n", "window", "quantity", "Rhat med/p90/max", "ESS/chain")
println("-"^76)
for (lo, hi) in windows
    idx = findall(s -> lo < s <= hi, sw)
    length(idx) < 20 && continue
    nk = length(idx)
    L = Array{Float64}(undef, nk, nchains, 1)
    for c in 1:nchains; L[:,c,1] = res[c][2][idx]; end
    MU = Array{Float64}(undef, nk, nchains, length(cells))
    for c in 1:nchains, s in 1:nk, p in eachindex(cells); MU[s,c,p] = res[c][3][idx[s]][p]; end
    thw = [[res[c][4][i] for i in idx] for c in 1:nchains]
    phw = [[res[c][5][i] for i in idx] for c in 1:nchains]
    asg = align_draws(thw)
    nI = size(thw[1][1], 1)
    TH = Array{Float64}(undef, nk, nchains, nI*K)
    PH = Array{Float64}(undef, nk, nchains, K*M)
    for c in 1:nchains, s in 1:nk
        p = asg[c][s]
        TH[s,c,:] = vec(thw[c][s][:, p])
        PH[s,c,:] = vec(phw[c][s][p, :])
    end
    # COARSE factor summaries -- what Figure 10 actually claims. Computed per draw from the
    # relabelled scaled trajectory, so they inherit the alignment.
    COM = Array{Float64}(undef, nk, nchains, K)   # temporal centre of mass
    DUR = Array{Float64}(undef, nk, nchains, K)   # duration (sd of the time distribution)
    SHR = Array{Float64}(undef, nk, nchains, K)   # share of total activity
    for c in 1:nchains, s in 1:nk
        P = phw[c][s][asg[c][s], :]               # K x M, relabelled
        tot = vec(sum(P, dims=2))
        for k in 1:K
            wt = @view P[k, :]
            mass = max(tot[k], 1e-300)
            com = sum(t * wt[t] for t in 1:M) / mass
            COM[s,c,k] = com
            DUR[s,c,k] = sqrt(max(sum(wt[t]*(t-com)^2 for t in 1:M)/mass, 0.0))
        end
        SHR[s,c,:] = tot ./ max(sum(tot), 1e-300)
    end

    w = "$lo-$hi"
    for (nm, A) in (("held-out log density", L), ("mu (150 cells)", MU),
                    ("theta/s_k aligned", TH), ("s_k*phi aligned", PH),
                    ("FACTOR centre of mass", COM), ("FACTOR duration", DUR),
                    ("FACTOR activity share", SHR))
        r, e = diag3(A)
        @printf("%-14s %-26s %5.2f/%5.2f/%5.2f %10.1f\n", w, nm, r[1], r[2], r[3], e)
    end
    println()
end

println("Held-out density is the predictive claim; the aligned normalised factors are what")
println("Figure 10 interprets. mu is shown for reference: it is ALREADY scale- and")
println("permutation-invariant, so a large Rhat there means the chains disagree about the")
println("fitted surface itself, which post-hoc normalisation of the factors cannot fix.")
