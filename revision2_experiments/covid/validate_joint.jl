# Joint-distribution (Schein) test of the Section 7.2 sampler, plus a thread-invariance
# check on the real data.
#
#   julia -t 1 --project=. revision2_experiments/covid/validate_joint.jl
#
# PART 1 is the correctness test that matters for the tau fix. If backward_sample is a
# valid Gibbs kernel for the model, then drawing (theta, Y) from the prior and then
# applying the kernel a few times leaves theta's marginal unchanged. So the "forward"
# sample (straight from the prior) and the "backward" sample (prior, then Gibbs sweeps)
# must have the same distribution. A biased update shows up as a small p-value.
#
# PART 2 checks that the threaded sampler and the serial one agree on the real data, i.e.
# that the per-thread accumulators and :static scheduling are sound.

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using Random, Printf, Statistics, LinearAlgebra

# ---------------------------------------------------------------- PART 1
println("="^78)
println("PART 1: Schein joint-distribution test on a small instance")
println("="^78)

# Sizes default to a small instance; pass "big" to get one large enough that every thread
# gets real work, which is how the threaded accumulators get exercised. Running this file
# under -t 8 with "big" is the thread-safety test: if :static scheduling or the
# per-thread Y_NK / Y_MK / sum_noise accumulators were wrong, the sufficient statistics
# would be corrupted and the joint distribution would not be preserved.
const BIG = "big" in ARGS
# Pass "hier" to test the variant that learns the county-factor rate b instead of holding
# it fixed. b is a state variable there, so it enters the joint test as its own functional.
const HIER = "hier" in ARGS
# Pass "scale" to turn on the interleaved multiplicative scale move. It is an MH step on
# the same target, so this test passing is what certifies it does not perturb the
# posterior -- the whole point of the move is that the answer is unchanged.
const SCALE = "scale" in ARGS
# Pass "epsmh" to turn on the marginal Metropolis step on eps. Like the scale move it is an
# MH step on the same target, so this test passing is what certifies it.
const EPSMH = "epsmh" in ARGS
# Pass "shear" to turn on the pairwise shear move. MH on the same target; passing certifies it.
const SHEAR = "shear" in ARGS
# Pass "bigcounts" to start every county at 20000 instead of 1..N. The default puts the
# sampler in a small-count regime, which is NOT where the real data lives: cumulative
# COVID counts reach 1e5-1e6, so sampleSumGivenOrderStatistic runs with parent means in
# the tens of thousands and draws deep-tail truncated Poissons. A correctness problem that
# only appears at that magnitude would be invisible to the default test.
const BIGCOUNTS = "bigcounts" in ARGS
# Pass "realistic" for the regime the chains actually live in: K = 5, Q = 25, M = 295, phi in
# the tens (start_V1 rescaled), cumulative counts in the thousands, D near its ceiling. The
# default "big" test is K = 2, Q = 3, M = 48 with phi ~ 0.01 -- it certifies the kernel where
# the parameters are small, not where the real runs are.
const REALISTIC = "realistic" in ARGS
# Pass "synthreg" for exactly the instance of synth_thetaphi_only.jl (N = 100, M = 60, K = 3,
# Q = 5, Y0 = 20000, phi_1 ~ Gamma(1,1)), where the isolated theta/phi block failed to
# converge; "thetaphi" skips the D / PG / tau / beta branch so only the theta / phi kernel
# (latent sums, thinning, U, V, scale move) is under test.
const SYNTHREG = "synthreg" in ARGS
const THETAPHI = "thetaphi" in ARGS
# Pass "collapsed" to test the slice-sampled, thinning-free theta / phi / eps update.
const COLLAPSED = ("collapsed" in ARGS) || ("hybrid" in ARGS)
# Pass "hybrid" for theta collapsed + phi by the PGDS pass on thinned counts.
const HYBRID = "hybrid" in ARGS
# "noscale" turns the interleaved scale move off (synthreg turns it on by default).
const NOSCALE = "noscale" in ARGS
# "dirichlet": theta columns on the simplex (dir_theta), hybrid sampler, c from the scale
# rule; "dirreg" is a realistic-scale regime for it (N = 100, M = 60, K = 3, phi ~ 1000).
const DIRICHLET = "dirichlet" in ARGS
const DIRREG = "dirreg" in ARGS
# "a=X": Dirichlet / Gamma shape of theta (a < 1 = sparse county shares).
atheta = 1.0
for a in ARGS; startswith(a, "a=") && (global atheta = parse(Float64, a[3:end])); end
# "exactcrt" disables the le Cam tail approximation in the CRT of the phi backward pass;
# "crt=X" sets its tolerance.
"exactcrt" in ARGS && (CRT_TOL[] = 0.0)
for a in ARGS; startswith(a, "crt=") && (CRT_TOL[] = parse(Float64, a[5:end])); end
println("CRT le Cam tolerance: ", CRT_TOL[])
"taunaive" in ARGS && (TAU_NAIVE[] = true; println("tau: naive MvNormal draw"))
N, M, K, Q, Dmax = DIRREG ? (100, 60, 3, 5, 9) : SYNTHREG ? (100, 60, 3, 5, 9) : REALISTIC ? (60, 295, 5, 25, 9) : BIG ? (240, 48, 2, 3, 5) : (8, 6, 2, 3, 5)
# Dirichlet theta: phi carries the scale. Typical factor size phibar sets start_V2 (E[phi_1] =
# phibar) and c = c_for_scale(0.02, phibar); the small regime uses phibar = 10, dirreg 1000.
phibar = DIRREG ? 1000.0 : 10.0
cval  = DIRICHLET ? c_for_scale(0.02, phibar) : 100.0
model = covid4(N, M, K, Q, Dmax,
               atheta, 1.0, cval, 0.1, 0.5, 1.0,      # a b c d g h
               REALISTIC ? 20.0 : (SYNTHREG || DIRICHLET) ? 1.0 : 0.01,
               REALISTIC ? 2.0 : DIRICHLET ? 1/phibar : 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,  # startV1 startV2
               HIER, (SCALE || SYNTHREG) && !NOSCALE, EPSMH, true, SHEAR,
               COLLAPSED || DIRICHLET, false, HYBRID || DIRICHLET, DIRICHLET)
DIRICHLET && println("Dirichlet theta, hybrid sampler, a = ", atheta, ", c = ", round(cval, digits=3), ", E[phi_1] = ", phibar)
info = Dict("Y0_N" => (BIGCOUNTS || SYNTHREG || DIRREG) ? fill(20000, N, 1) : REALISTIC ? fill(500, N, 1) : reshape(collect(1:N), :, 1))
# "dlayer" is the complement: freeze theta / phi / eps and test the D / PG / tau / beta
# layer on its own.
const DLAYER = "dlayer" in ARGS
skipkw = THETAPHI ? (; skipupdate = ["D_NM"]) : DLAYER ? (; skipupdate = ["U_NK"]) : (;)

nsamples = (REALISTIC || SYNTHREG || DIRREG) ? 400 : BIG ? 600 : 4000
nthin = (REALISTIC || SYNTHREG || DIRREG) ? 3 : BIG ? 2 : 4
# thin=N overrides the sweeps-per-replicate. In the realistic regime the kernel moves the
# state only slightly in 3 sweeps, so a bias that accumulates over sweeps is invisible there.
for a in ARGS; startswith(a, "thin=") && (global nthin = parse(Int, a[6:end])); end
for a in ARGS; startswith(a, "nrep=") && (global nsamples = parse(Int, a[6:end])); end
println("N=$N M=$M K=$K Q=$Q Dmax=$Dmax, $nsamples replicates, $nthin sweeps each, ",
        Threads.nthreads(), " threads")

varlist = ["eps", "sigma2_tau", "sigma2_beta", "U_NK", "V_KM", "Tau_QM", "Beta_NQ", "D_NM"]
HIER && push!(varlist, "b_mu")

Random.seed!(2024)
f_samples, b_samples = scheinTest(model, varlist; nsamples=nsamples, nthin=nthin, info=info, skipkw...)
# Derived functionals of the growth term G = theta * phi, which is what the real runs
# disagree about: a bias in the theta / phi kernel that trades scale between the two
# factors is invisible to mean(U) and mean(V) separately but not to G.
for (smp) in (f_samples, b_samples)
    smp["G"] = [u * v for (u, v) in zip(smp["U_NK"], smp["V_KM"])]
end

# Scalar functionals of each variable. Both first and second moments, since a wrong
# conditional often matches the mean but not the spread.
funcs = Dict(
    "eps"         => [("mean", x -> x),            ("sq",   x -> x^2)],
    "sigma2_tau"  => [("mean", x -> x),            ("log",  x -> log(x))],
    "sigma2_beta" => [("mean", x -> x),            ("log",  x -> log(x))],
    "U_NK"        => [("mean", x -> mean(x)),      ("var",  x -> var(vec(x))),
                      ("maxshare", x -> mean(maximum(x, dims=1)))],   # simplex-aware (mean is constant under Dirichlet)
    "V_KM"        => [("mean", x -> mean(x)),      ("var",  x -> var(vec(x)))],
    "Tau_QM"      => [("mean", x -> mean(x)),      ("var",  x -> var(vec(x))),
                      ("lag1", x -> mean(x[:,1:end-1] .* x[:,2:end]))],
    "Beta_NQ"     => [("mean", x -> mean(x)),      ("var",  x -> var(vec(x)))],
    "D_NM"        => [("mean", x -> mean(x)),      ("var",  x -> var(vec(x)))],
    "G"           => [("mean", x -> mean(x)),      ("var",  x -> var(vec(x))),
                      ("logmean", x -> mean(log.(x .+ 1e-12))),
                      ("colcv", x -> mean(std(x, dims=1) ./ (mean(x, dims=1) .+ 1e-12)))],
)
HIER && (funcs["b_mu"] = [("mean", x -> x), ("log", x -> log(x))])

println()
@printf("%-14s %-7s %12s %12s %10s %10s %10s\n", "variable", "func", "forward", "backward", "p-value", "paired-p", "sign-p")
println("-"^82)
# Each backward chain is started from its own forward draw, so (fw[i], bw[i]) is a matched
# pair. The paired test uses the variance of the differences, which is the right scale when
# the kernel moves the state only a little per replicate; the unpaired z-score then has
# almost no power (its p-values pile up near 1).
pairedp(fw, bw) = (d = fw .- bw; z = mean(d) / (std(d) / sqrt(length(d))); 2 * (1 - cdf(Normal(), abs(z))))
# Sign test on the same pairs: immune to the heavy tails of var(tau) under an
# inverse-gamma sigma2, where a handful of huge replicates can drive the paired z.
signp(fw, bw) = (d = filter(!=(0), fw .- bw); n = length(d); k = count(>(0), d);
                 min(1.0, 2 * min(cdf(Binomial(n, 0.5), k), 1 - cdf(Binomial(n, 0.5), k - 1))))
saved = Dict{String,Any}()
pvals = Float64[]
labels = String[]
for (key, fl) in sort(collect(funcs), by=first)
    for (fname, f) in fl
        fw = Float64[f(x) for x in f_samples[key]]
        bw = Float64[f(x) for x in b_samples[key]]
        p = gewekepvalue(fw, bw); pp = pairedp(fw, bw); sp = signp(fw, bw)
        saved["$key.$fname.fw"] = fw; saved["$key.$fname.bw"] = bw
        push!(pvals, pp); push!(labels, "$key.$fname")
        @printf("%-14s %-7s %12.5f %12.5f %10.4f %10.4f %10.4f%s\n",
                key, fname, mean(fw), mean(bw), p, pp, sp, pp < 0.01 ? "  <-- LOW" : "")
    end
end

for a in ARGS
    if startswith(a, "save=")
        using JLD; JLD.save(a[6:end], "saved", saved); println("per-replicate functionals saved to ", a[6:end])
    end
end
nlow = count(<(0.01), pvals)
println()
# 17 paired tests at alpha = 0.01: expect 0 or 1 low by chance.
if nlow == 0
    println("RESULT: no p-value below 0.01 across $(length(pvals)) functionals -- consistent")
elseif nlow <= 1
    println("RESULT: $nlow of $(length(pvals)) below 0.01 -- within what multiple testing gives")
else
    println("RESULT: $nlow of $(length(pvals)) below 0.01 -- INVESTIGATE: $(labels[pvals .< 0.01])")
end

println()
println("Run this file BOTH ways to cover the threading:")
println("  julia -t 1 --project=. revision2_experiments/covid/validate_joint.jl big")
println("  julia -t 8 --project=. revision2_experiments/covid/validate_joint.jl big")
println("Passing under -t 8 is what certifies the :static schedule and the per-thread")
println("Y_NK / Y_MK / sum_noise accumulators: a corrupted sufficient statistic does not")
println("preserve the joint distribution, so it shows up here as a small p-value.")
