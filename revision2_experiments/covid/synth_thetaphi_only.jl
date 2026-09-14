# The sharpest isolation of the theta / phi block: synthetic data FROM the model, D and eps
# pinned at their generating values, only theta / phi (and the latent counts) sampled.
#
#   julia -t 8 --gcthreads=1 --project=. revision2_experiments/covid/synth_thetaphi_only.jl [nsweeps] [nchains] [K] [init]
#
# K = 1 is the discriminating case: a single factor is identified up to the theta/phi scale,
# there is no permutation or shear to mix over, so a correct block MUST converge. init=truth
# starts every chain at the generating theta/phi: a correct kernel then stays in the typical
# set (bias ~0, coverage ~95%); a wrong one drifts away from the truth it was handed.
#
# A correct sampler for a correct model must, on data generated from that model with a
# prior-typical truth, give (i) Rhat -> 1 across dispersed chains and (ii) ~95% coverage of
# the truth. Earlier synthetic runs plateaued at 86% coverage / +2.7 bias after 27000 sweeps
# with everything free; that was attributed to eps being frozen. Here eps and D are held at
# truth, so any residual is the theta / phi block alone.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using Random, Printf, Statistics, LinearAlgebra
using MCMCDiagnosticTools, Combinatorics

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
Kin = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
initmode = length(ARGS) >= 4 ? ARGS[4] : "prior"
collapsed = length(ARGS) >= 5 && ARGS[5] in ("collapsed", "hybrid")
# "hybrid": theta collapsed, phi by the paper's CRT backward / Gamma forward pass on thinned counts
hybrid = length(ARGS) >= 5 && ARGS[5] == "hybrid"
# ARGS[6] == "D1": every D pinned at 1, i.e. a plain Poisson likelihood with NO order-statistic
# augmentation. The collapsed sampler is then an exact slice sampler on the posterior, so
# failure to converge here can only be the slice code or the posterior itself.
d1 = length(ARGS) >= 6 && ARGS[6] == "D1"
# ARGS[7] == "iid": iid Gamma prior on phi instead of the gamma-Markov chain (scale move off).
iid = length(ARGS) >= 7 && ARGS[7] == "iid"
# ARGS[8] == "waves": generate the true phi as K well-separated bumps (waves at different
# times) instead of prior draws. Prior draws of the gamma-Markov chain are near-collinear
# (pairwise cosine ~0.9-1.0), which leaves the split of activity between factors
# unidentified BY CONSTRUCTION; the real data's waves are separated in time.
waves = length(ARGS) >= 8 && ARGS[8] == "waves"
nkeep = nsweeps ÷ 2
N, M, K, Q, Dmax = 100, 60, Kin, 5, 9
# prior-typical truth in the high-offset regime: phi_1 ~ Gamma(1, rate 0.1) => E ~ 10
mk() = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 0.0, 0.0, false, !iid, false, true, false, collapsed, iid, hybrid)
model = mk()
info = Dict("Y0_N" => fill(20000, N, 1))
Random.seed!(4242)
st0 = sample_prior(model, info); st0["eps"] = 2.0
d1 && (st0["D_NM"] .= 1)
if waves
    for k in 1:K
        ctr = M * (k - 0.5) / K; wid = M / (3K)
        st0["V_KM"][k, :] = [10.0 * exp(-0.5 * ((t - ctr) / wid)^2) + 0.05 for t in 1:M]
    end
end
data, st_true = forward_sample(model; state=st0, info=info)
G_true = st_true["U_NK"] * st_true["V_KM"]; D_true = st_true["D_NM"]
@printf("init=%s  sampler=%s  D1=%s  iid_phi=%s  waves=%s  ", initmode, hybrid ? "hybrid" : collapsed ? "collapsed" : "thinning", d1, iid, waves); @printf("N=%d M=%d K=%d  mean growth %.1f  mean Y %.0f  mean D %.2f\n", N, M, K, mean(G_true), mean(data["Y_NM"]), mean(D_true))

Random.seed!(99); cells = [(rand(1:N), rand(1:M)) for _ in 1:300]
function run_chain(seed)
    Random.seed!(seed)
    st = sample_prior(model, info)
    st["eps"] = 2.0; st["D_NM"] = copy(D_true)                     # pinned at truth
    if initmode == "truth"
        st["U_NK"] = copy(st_true["U_NK"]); st["V_KM"] = copy(st_true["V_KM"])
    else
        st["V_KM"] .*= max(mean(G_true) , 1e-6) / max(mean(st["U_NK"]*st["V_KM"]), 1e-12)
    end
    g = Vector{Matrix{Float64}}(); th = Vector{Matrix{Float64}}(); ph = Vector{Matrix{Float64}}()
    for s in 1:nsweeps
        (_d, st) = backward_sample(model, data, st; skipupdate=["D_NM"])
        st["eps"] = 2.0; st["D_NM"] = D_true
        if s > nsweeps - nkeep
            U = st["U_NK"]; V = st["V_KM"]; sk = vec(sum(U, dims=1))
            push!(g, U * V)
            push!(th, U[1:min(N,60), :] ./ reshape(sk, 1, K))   # theta / s_k, scale-free
            push!(ph, V .* reshape(sk, K, 1))                    # s_k * phi, scale-free
        end
    end
    (g, th, ph)
end
out = [run_chain(800 + c) for c in 1:nchains]
res = [o[1] for o in out]; thw = [o[2] for o in out]; phw = [o[3] for o in out]

A = Array{Float64}(undef, nkeep, nchains, length(cells))
for c in 1:nchains, s in 1:nkeep, p in eachindex(cells); (n,m) = cells[p]; A[s,c,p] = res[c][s][n,m]; end
r3(x) = reshape(x, size(x,1), size(x,2), 1)
rs = [rhat(r3(A[:,:,p]))[1] for p in 1:length(cells)]
es = [ess(r3(A[:,:,p]); kind=:bulk)[1] for p in 1:length(cells)]
allg = vcat(res...)
ncov = 0; bias = mean(mean(allg) .- G_true)
for (p,(n,m)) in enumerate(cells)
    v = [gg[n,m] for gg in allg]
    quantile(v,.025) <= G_true[n,m] <= quantile(v,.975) && (global ncov += 1)
end
@printf("\nmu at 300 cells, %d chains x %d kept:  Rhat med/p90/max %.3f/%.3f/%.3f   pooled ESS med/min %.0f/%.0f\n",
        nchains, nkeep, median(rs), quantile(rs,.9), maximum(rs), median(es), minimum(es))
@printf("recovery: bias %+.2f (truth mean %.1f)   95%% coverage %.0f%%\n", bias, mean(G_true), 100ncov/length(cells))
println("\nWith D and eps at truth, a correct theta/phi block must give Rhat ~1 and coverage ~95%.")

# ---- the factors themselves, relabelled per draw (Stephens-style, as in reported_diagnostics.jl)
function factor_diag(thw, phw, st_true)
    perms = collect(permutations(1:K))
    ref = mean(thw[1]); asg = [[collect(1:K) for _ in 1:nkeep] for c in 1:nchains]
    for _ in 1:8
        acc = zeros(size(ref)); n = 0
        for c in 1:nchains, s in 1:nkeep
            pp = perms[argmin([sum(abs2, ref .- thw[c][s][:, q]) for q in perms])]
            asg[c][s] = pp; acc .+= thw[c][s][:, pp]; n += 1
        end
        ref = acc ./ n
    end
    nI = size(thw[1][1], 1)
    TH = Array{Float64}(undef, nkeep, nchains, nI*K); PH = Array{Float64}(undef, nkeep, nchains, K*M)
    COM = Array{Float64}(undef, nkeep, nchains, K); SHR = Array{Float64}(undef, nkeep, nchains, K)
    for c in 1:nchains, s in 1:nkeep
        pp = asg[c][s]
        TH[s,c,:] = vec(thw[c][s][:, pp]); P = phw[c][s][pp, :]; PH[s,c,:] = vec(P)
        tot = vec(sum(P, dims=2))
        for k in 1:K; COM[s,c,k] = sum(t * P[k,t] for t in 1:M) / max(tot[k], 1e-300); end
        SHR[s,c,:] = tot ./ max(sum(tot), 1e-300)
    end
    @printf("\n%-24s %19s %14s\n", "factors (aligned)", "Rhat med/p90/max", "pooled ESS med/min")
    for (nm, A) in (("theta/s_k", TH), ("s_k*phi", PH), ("centre of mass", COM), ("activity share", SHR))
        rr = [rhat(r3(A[:,:,p]))[1] for p in 1:size(A,3)]
        ee = [ess(r3(A[:,:,p]); kind=:bulk)[1] for p in 1:size(A,3)]
        @printf("%-24s %5.3f/%5.3f/%5.3f %7.0f/%6.0f\n", nm, median(rr), quantile(rr,.9), maximum(rr), median(ee), minimum(ee))
    end
    # how separated are the true factors? cosine similarity between true phi rows
    Vt = st_true["V_KM"]; cs = [dot(Vt[i,:], Vt[j,:]) / (norm(Vt[i,:]) * norm(Vt[j,:])) for i in 1:K for j in i+1:K]
    @printf("true phi rows: pairwise cosine similarity %s\n", join([@sprintf("%.2f", x) for x in cs], " "))
end
K > 1 && factor_diag(thw, phw, st_true)

