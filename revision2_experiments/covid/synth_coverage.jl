# Why does coverage top out at 70% instead of 95%?
#
#   julia -t 6 --project=. revision2_experiments/covid/synth_coverage.jl [nsweeps] [nchains]
#
# HYPOTHESIS: it is prior-truth mismatch, not a sampler defect. A Bayesian credible interval
# has nominal frequentist coverage only when averaged over truths drawn FROM THE PRIOR. The
# synthetic truth used so far is hand-set (phi in the tens), while the prior
# phi_{k,1} ~ Gamma(start_V1 = 0.01, start_V2 = 1) feeding a c = 100 random walk with drift
# d/c = 0.001 per step essentially cannot produce phi in the tens. So the prior fights the
# truth, shrinks the growth term, and coverage falls below nominal -- which is correct
# behaviour for a misspecified prior, not a broken sampler.
#
# The test is the pair of arms with a PRIOR-TYPICAL truth. To get one that is also in the
# high-offset regime the paper lives in, the phi prior is rescaled (start_V1 = 1,
# start_V2 = 0.1, so E[phi_1] = 10 and the growth term lands in the tens). Then:
#
#   truth ~ rescaled prior, fit with rescaled prior   -> coverage should be ~95%
#   truth ~ rescaled prior, fit with ORIGINAL prior   -> coverage should be well below
#
# If that pair separates, the 70% is the prior, and the honest fix is to rescale
# start_V1/start_V2 rather than to chase the sampler.
#
# All arms use the scale-corrected initialisation and Y0 = 20000.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using Random, Printf, Statistics, LinearAlgebra

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
nkeep   = nsweeps ÷ 2
N, M, K, Q, Dmax = 100, 60, 3, 5, 9
println("N=$N M=$M K=$K Q=$Q Dmax=$Dmax, $nchains chains x $nsweeps sweeps, keep $nkeep\n")

mk(sV1, sV2) = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
                      sV1, sV2, 1.0, 1.0, 1.0, 0.0, 0.0, false, false)
const ORIG = (0.01, 1.0)     # the paper's prior for phi_1
const RESC = (1.0,  0.1)     # E[phi_1] = 10, so the growth term lands in the tens

Y0 = fill(20000, N, 1)
info = Dict("Y0_N" => Y0)

bump(t, c, w) = exp(-0.5*((t-c)/w)^2)
function handset_truth(seed)
    Random.seed!(seed)
    th = fill(0.1, N, K)
    for i in 1:N; th[i, rand(1:K)] += 2.0*rand(); end
    ph = Matrix{Float64}(undef, K, M)
    for k in 1:K, t in 1:M; ph[k,t] = 40.0*bump(t, [25.0,30.0,35.0][k], 15.0) + 0.5; end
    return th, ph
end

# Generate data. `source` is :handset or a (sV1,sV2) pair meaning "draw the truth from that
# prior", which is the only way to get a truth the posterior is entitled to cover.
function gen(source, seed)
    genmodel = source === :handset ? mk(ORIG...) : mk(source...)
    Random.seed!(seed)
    st = sample_prior(genmodel, info)
    if source === :handset
        th, ph = handset_truth(seed + 1)
        st["U_NK"] = th; st["V_KM"] = ph
    end
    st["eps"] = 2.0
    data, st = forward_sample(genmodel; state=st, info=info)
    return data, st["U_NK"] * st["V_KM"]
end

function run_chain(model, data, inc, seed, nsw=nsweeps, nkp=nkeep, epstrue=false)
    Random.seed!(seed)
    st = sample_prior(model, info)
    epstrue && (st["eps"] = 2.0)      # the generating value: an oracle DIAGNOSTIC, not a fix
    target = max(mean(inc) - st["eps"], 1e-6)
    cur = mean(st["U_NK"] * st["V_KM"])
    st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))   # the shipped initialisation
    g = Vector{Matrix{Float64}}()
    for s in 1:nsw
        (_d, st) = backward_sample(model, data, st)
        s > nsw - nkp && push!(g, st["U_NK"] * st["V_KM"])
    end
    return g
end

Random.seed!(99)
cells = [(rand(1:N), rand(1:M)) for _ in 1:300]

function arm(label, source, fitprior; smove=false, nsw=nsweeps, nkp=nkeep, epstrue=false, epsmh=false)
    data, gtrue = gen(source, 4242)
    inc = Matrix{Float64}(undef, N, M)
    for i in 1:N
        inc[i,1] = max(data["Y_NM"][i,1] - Y0[i], 0)
        for t in 2:M; inc[i,t] = max(data["Y_NM"][i,t] - data["Y_NM"][i,t-1], 0); end
    end
    model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
                   fitprior[1], fitprior[2], 1.0, 1.0, 1.0, 0.0, 0.0, false, smove, epsmh)
    allg = vcat([run_chain(model, data, inc, 800+c, nsw, nkp, epstrue) for c in 1:nchains]...)
    cov = 0; wid = Float64[]
    for (n, m) in cells
        v = [gg[n,m] for gg in allg]
        lo, hi = quantile(v, 0.025), quantile(v, 0.975)
        lo <= gtrue[n,m] <= hi && (cov += 1)
        push!(wid, hi - lo)
    end
    ghat = mean(allg)
    bias = mean(ghat .- gtrue)
    @printf("%-44s  mean growth %6.1f (true %6.1f)  bias %+7.2f  median CI width %6.2f  cov95 %3.0f%%\n",
            label, mean(ghat), mean(gtrue), bias, median(wid), 100*cov/length(cells))
    flush(stdout)
end

println("truth prior-typical? -> coverage should only be nominal when it is\n")
# All three use the prior-typical truth, where a correct converged sampler MUST be at 95%.
# Run-length calibration. Coverage on a prior-typical truth is the honest target: it says
# what it means, unlike Rhat/ESS on this model. Scale move on throughout -- it is free and
# helps a little.
# Does the residual bias close if eps starts at the generating value? eps cannot move from
# wherever it is initialised, and the scale-corrected init sets the growth target to
# mean(increment) - eps, so an eps that lands low forces the growth term high by the deficit.
# The oracle rows are the target the MH step is trying to reach without oracle information.
arm("eps Gibbs only,   3000 sweeps", RESC, RESC; smove=true)
arm("eps + MH step,    3000 sweeps", RESC, RESC; smove=true, epsmh=true)
arm("eps AT TRUTH,     3000 sweeps (oracle)", RESC, RESC; smove=true, epstrue=true)
arm("eps + MH step,    9000 sweeps", RESC, RESC; smove=true, epsmh=true, nsw=9000, nkp=4500)
arm("eps AT TRUTH,     9000 sweeps (oracle)", RESC, RESC; smove=true, epstrue=true, nsw=9000, nkp=4500)

println("\nIntervals were too narrow, not misplaced, so this asks whether the missing spread is")
println("scale uncertainty the chains cannot traverse (scale move) or simply unexplored")
println("posterior (longer run). Whichever raises coverage is the cause.")
