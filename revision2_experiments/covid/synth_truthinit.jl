# Is the high-offset failure a burn-in problem or a posterior problem?
#
#   julia -t 6 --project=. revision2_experiments/covid/synth_truthinit.jl [nsweeps] [nchains]
#
# synth_ladder.jl showed that on data whose level dwarfs the growth term -- which is the
# regime the real cumulative COVID counts live in -- four chains of 4000 sweeps started
# from the prior recover the growth with 99% relative error and 0% coverage, while the
# SAME model on lower-offset data recovers it to 13-29% with ~88% coverage. The joint
# distribution test passes at these count magnitudes, so the kernel is correct.
#
# That leaves two possibilities, and they call for completely different responses:
#
#   BURN-IN     the posterior does concentrate near the truth, but a chain started from
#               the prior (phi ~ Gamma(0.01,1), i.e. growth ~ 0.01 against a true ~40)
#               cannot climb there in any feasible number of sweeps, because the gradient
#               is weak relative to the per-observation noise sqrt(20000) ~ 140.
#               => fix by initialising from the data. Costs nothing inferentially.
#
#   POSTERIOR   the posterior genuinely does not concentrate near the truth at this
#               signal-to-noise. => no sampler change helps; it is a statement about what
#               the data can support, and belongs in the paper as such.
#
# The discriminator: start the chains AT the truth. If they stay there with proper
# coverage, it is burn-in. If they drift away, it is the posterior.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using Random, Printf, Statistics, LinearAlgebra

nsweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3000
nchains = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
nkeep   = nsweeps ÷ 2
N, M, K, Q, Dmax = 100, 60, 3, 5, 9
println("N=$N M=$M K=$K Q=$Q Dmax=$Dmax, $nchains chains x $nsweeps sweeps, keep $nkeep")

bump(t, c, w) = exp(-0.5*((t-c)/w)^2)
function make_truth(seed)
    Random.seed!(seed)
    th = fill(0.1, N, K)
    for i in 1:N; th[i, rand(1:K)] += 2.0*rand(); end
    ph = Matrix{Float64}(undef, K, M)
    ctr = [25.0, 30.0, 35.0]; wid = [15.0, 15.0, 15.0]
    for k in 1:K, t in 1:M; ph[k,t] = 40.0*bump(t, ctr[k], wid[k]) + 0.5; end
    return th, ph
end

model = covid4(N, M, K, Q, Dmax, 1.0, 1.0, 100.0, 0.1, 0.5, 1.0,
               0.01, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, false, false)
Y0 = fill(20000, N, 1)                      # the failing, realistic regime
info = Dict("Y0_N" => Y0)
th_true, ph_true = make_truth(4243)
Random.seed!(4242)
st0 = sample_prior(model, info)
st0["U_NK"] = th_true; st0["V_KM"] = ph_true; st0["eps"] = 2.0
data, st_true = forward_sample(model; state=st0, info=info)
growth_true = th_true * ph_true
@printf("median growth/Y = %.5f   mean growth = %.1f   sqrt(mean Y) = %.1f\n",
        median(vec(growth_true) ./ max.(vec(Float64.(data["Y_NM"])), 1.0)),
        mean(growth_true), sqrt(mean(data["Y_NM"])))

Random.seed!(99)
cells = [(rand(1:N), rand(1:M)) for _ in 1:100]

# Clipped first differences. The model says E[Y_{i,t} - Y_{i,t-1}] ~ eps + sum_k theta phi,
# so this is a direct estimate of what the growth term has to reproduce. Negative
# increments are reporting corrections (the phenomenon the model is built around) and are
# clipped to zero.
inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(data["Y_NM"][i,1] - Y0[i], 0)
    for t in 2:M
        inc[i,t] = max(data["Y_NM"][i,t] - data["Y_NM"][i,t-1], 0)
    end
end

# KL-divergence NMF (multiplicative updates) -- the right divergence for count data, and
# the same factorisation the model is doing. Started from its own random point per chain so
# the chains stay dispersed: initialising every chain at one NMF solution would collapse
# Rhat to ~1 by construction and make the diagnostic worthless.
function nmf_kl(V, K, iters, rng)
    n, m = size(V)
    W = rand(rng, n, K) .+ 0.1
    H = rand(rng, K, m) .+ 0.1
    for _ in 1:iters
        WH = max.(W*H, 1e-10)
        H .*= (transpose(W) * (V ./ WH)) ./ max.(transpose(sum(W, dims=1)), 1e-10)
        WH = max.(W*H, 1e-10)
        W .*= ((V ./ WH) * transpose(H)) ./ max.(transpose(sum(H, dims=2)), 1e-10)
    end
    return W, H
end

# start = :prior  draw everything from the prior (what the "no tailored initialisation"
#                 convention of Sections 6.3 and 7.1 gives)
#         :truth  theta, phi, eps at the generating values -- an oracle start, used only to
#                 ask whether the posterior supports the truth at all
#         :seed1  phi seeded at 1.0 and then free, which is what every previous covid run
#                 actually did and what the run scripts now do. This is the practical
#                 candidate: it uses no knowledge of the truth.
function run_chain(seed, start)
    Random.seed!(seed)
    st = sample_prior(model, info)
    if start == :truth
        st["U_NK"] = copy(th_true); st["V_KM"] = copy(ph_true); st["eps"] = 2.0
    elseif start == :seed1
        st["V_KM"] = fill(1.0, K, M)
    elseif start == :scalecal
        # Keep the prior draw's STRUCTURE -- so chains stay as dispersed as before -- and
        # correct only its overall magnitude, which is the thing the diagnosis says fails.
        # One constant, no data structure injected into the factors.
        target = max(mean(inc) - st["eps"], 1e-6)
        cur = mean(st["U_NK"] * st["V_KM"])
        st["V_KM"] = st["V_KM"] .* (target / max(cur, 1e-12))
    elseif start == :nmf
        W, H = nmf_kl(inc, K, 60, Random.default_rng())
        st["U_NK"] = W; st["V_KM"] = H
    end
    g = Vector{Matrix{Float64}}(); e = Float64[]
    for s in 1:nsweeps
        (_d, st) = backward_sample(model, data, st)
        if s > nsweeps - nkeep
            push!(g, st["U_NK"] * st["V_KM"]); push!(e, st["eps"])
        end
    end
    return g, e
end

function report(label, start)
    res = [run_chain(700+c, start) for c in 1:nchains]
    allg = vcat([r[1] for r in res]...)
    ghat = mean(allg)
    relerr = mean(abs.(ghat .- growth_true) ./ max.(growth_true, 1e-9))
    cov = 0
    for (n, m) in cells
        v = [gg[n,m] for gg in allg]
        quantile(v, 0.025) <= growth_true[n,m] <= quantile(v, 0.975) && (cov += 1)
    end
    # did each chain STAY where it started, or move?
    firstlast = [(mean(r[1][1]), mean(r[1][end])) for r in res]
    epshat = mean(mean(r[2]) for r in res)
    @printf("%-22s growth relerr %5.2f   cov95 %3.0f%%   eps %5.2f (true 2.0)\n",
            label, relerr, 100*cov/length(cells), epshat)
    @printf("%-22s per-chain mean growth, first kept draw -> last: %s   (truth %.1f)\n", "",
            join([@sprintf("%.1f->%.1f", a, b) for (a,b) in firstlast], "  "), mean(growth_true))
    flush(stdout)
end

report("from prior",      :prior)
report("phi seeded at 1", :seed1)
report("scale-calibrated", :scalecal)
report("NMF per chain",   :nmf)
report("from truth",      :truth)
println("\n'from truth' says whether the posterior supports the truth at all.")
println("'phi seeded at 1' is the practical test: it uses no knowledge of the truth, and if")
println("it lands where the oracle start lands then seeding phi is a sufficient fix.")
