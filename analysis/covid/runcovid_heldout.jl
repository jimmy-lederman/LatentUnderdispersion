# Section 7.2 held-out (forecasting) run for the COVID-19 case study.
#
#   julia -t auto --project=. analysis/covid/runcovid_heldout.jl <maskSeed> <chainSeed> <D> <K> <Q> <type>
#
#   type 1..3 = the ablation models of Appendix E.1 (models/covid/ablation/)
#   type 4    = the final model of Section 7.2 (models/covid/covid_final.jl)
#   D         = 0 infers D_{i,t}; D > 0 pins it (D = 1 is the Poisson baseline)
#
# The mask holds out the last `endlength` days for a random 20% of counties, which the
# sampler imputes inside each Gibbs sweep. Held-out cells are scored against the truth,
# the same heldout-entry convention used in Sections 6.3 and 7.1.
#
# Threads matter here: the sampler is threaded over cells and columns, and the saved
# output records Threads.nthreads() so the timing column stays interpretable.
println("opened file"); flush(stdout)
using Dates, CSV, DataFrames, Random, JLD, Statistics
println("threads = ", Threads.nthreads()); flush(stdout)

ROOTDIR = joinpath(@__DIR__, "../..")

# ---- data ------------------------------------------------------------------------------
cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))
days  = cumdf[1, 4:end]
state = cumdf[2:end, 1]
fips  = cumdf[2:end, 2]
pops  = log.(cumdf[2:end, 3])
cumdf = cumdf[2:end, 4:end]
# Column 1 is t = 0, the count Eq. (12) CONDITIONS on; it is not itself modelled. It used
# to be passed as Y0_N *and* left as Y_NM[:,1], so for t = 1 the rate contained Y_{i,1}
# and the model conditioned that day on itself. Splitting them gives the T = 295 modelled
# days the paper describes.
Y0_N  = Int.(reshape(cumdf[:, 1], :, 1))
Y_NM  = cumdf[:, 2:end]
data  = Dict("Y_NM" => Y_NM)
N = size(Y_NM, 1)
M = size(Y_NM, 2)
info = Dict("Y0_N" => Y0_N, "pop_N" => pops)
println("N = $N counties, M = $M modelled days (t = 0 held out as Y0)")

# ---- arguments -------------------------------------------------------------------------
maskSeed  = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[2])
D         = parse(Int, ARGS[3])
K         = parse(Int, ARGS[4])
Q         = parse(Int, ARGS[5])
type      = parse(Int, ARGS[6])

# ---- forecasting mask: last `endlength` days for a random 20% of counties ---------------
endlength  = 10
county_pct = .2
function make_forecasting_mask(endlength, county_pct, seed, N, M)
    Random.seed!(seed)
    missing_counties = rand(N) .< county_pct
    mask_NM = zeros(N, M)
    for i in (M-endlength+1):M
        mask_NM[:, i] = missing_counties
    end
    return BitArray(mask_NM .!= 0)
end
mask_NM = make_forecasting_mask(endlength, county_pct, maskSeed, N, M)

# ---- hyperparameters -------------------------------------------------------------------
a = 1.0; b = 1.0
d = .1
g = .5; h = 1.0
start_V1 = .01; start_V2 = 1.0
alpha0 = 1.0; beta0 = 1.0   # named to match the struct fields, and to avoid shadowing SpecialFunctions.beta
start_tau = 0.0; tauc = 1.0; taud = 0.0
Dmax = 9

# eps is PINNED, not inferred (Section 7.2 reports it as a fixed positivity floor; see the
# epsilon paragraph). The diagnostics runs pin it at the same value, so the information-gain
# table and the diagnostics table describe the same posterior.
EPS_PIN = 0.5

# Run lengths. The D-INFERRED arm has a burn-in problem the pinned arms do not: rho's linear
# predictor is a bilinear form in two zero-mean factors, so it starts at logit^{-1}(0) = 1/2
# and mean D starts near 1 + (Dmax-1)/2 ~ 5, having to climb to ~7.9. Across the four chains
# of the final full-matrix run that climb took between under 1000 and about 5000 sweeps (one
# chain was still at 7.73 at sweep 4000). A 2000-sweep burn-in would retain that chain
# mid-climb, and ONLY the inferred arm has the climb -- so the old setting would have biased
# the inferred-vs-pinned comparison in favour of the paper's own claim. With D pinned the
# whole D / Polya-gamma / beta / tau block is skipped and there is nothing there to converge.
#
# Retained draws are held at 500 deliberately: `fit` accumulates whole state dicts, and
# D_NM alone is 915,680 entries per draw, so 500 draws is already ~4 GB resident.
if type == 4 && D == 0
    Nsamples = 500
    nburnin  = 5000
    nthin    = 20        # 15000 sweeps, matching output/covid/diag_final_Q1
else
    Nsamples = 500
    nburnin  = 2000
    nthin    = 8         # 6000 sweeps
end

# `constantinit` PINS D for the whole run; it is how the fixed-D rows, including the
# D = 1 Poisson baseline, are produced. `skipupdate` additionally stops the D branch from
# RUNNING -- pinning alone lets the D update, the ~900k Polya-Gamma draws and the
# beta/tau blocks execute and then discards them, which is about half a sweep, so the
# fixed-D rows of the timing table would be charged for machinery they do not use.
#
# Skipping is exact here, not an approximation: with D pinned, beta and tau have no data
# (D is no longer a draw from the OddBinomial), so their posterior is their prior and
# nothing else in the model depends on them. Everything reported -- theta, phi, eps, the
# fitted means, the held-out density -- is unaffected.
#
# D = 0 means the model infers D_{i,t} and nothing is pinned.
#
# Only type 4 carries a per-cell D_{i,t} to pin: types 1 and 2 have no order statistic at
# all, and type 3 takes its fixed order as a model field.
cinit = Dict{String,Any}()
type == 4 && (cinit["eps"] = EPS_PIN)
(type == 4 && D > 0) && (cinit["D_NM"] = fill(D, N, M))
Dkw = isempty(cinit) ? (;) :
      (type == 4 && D > 0) ? (; constantinit = cinit, skipupdate = ["D_NM"]) :
                             (; constantinit = cinit)

# phi's prior draw is SCALE-CORRECTED before the chain starts. Its structure is left alone,
# so chains stay as dispersed as a prior draw makes them; only the overall magnitude is
# fixed. This is not optional here.
#
# The prior phi_{k,1} ~ Gamma(start_V1 = 0.01, 1) is a near-spike at zero feeding a tight
# (c = 100) Markov chain, so a prior draw starts the growth term around 0.01 while the truth
# is in the tens. That start is self-reinforcing: the augmentation allocates latent counts
# to each component in proportion to its CURRENT share of the rate, and when the rate is
# dominated by Y_{i,t-1} (1e4-1e6 for cumulative counts) a component starting near zero
# receives essentially nothing and never grows.
#
# Measured on synthetic data generated from this model with the truth known (mean growth 32,
# Y ~ 20000), 4 chains x 3000 sweeps:
#     prior draw          growth stays at 0.0-0.6, never moves,        95% coverage  0%
#     phi seeded at 1     climbs to 12-16, still rising at the end,    95% coverage 49%
#     scale-corrected     starts 25-31 dispersed, converges to 31-32,  95% coverage 70%
#     started AT truth    stays at 32-35,                              95% coverage 61%
# so the scale correction reaches the truth from dispersed starts, which a fixed seed does
# not. An NMF initialisation was also tried and is worse: it fits the whole increment, so
# the factors absorb eps and eps collapses to 0.33 against a truth of 2.0.
#
# Sections 6.3 and 7.1 initialise every parameter from its prior; that convention is safe
# there because those models have no offset term in the rate, so the trap does not exist.
# `init` seeds and then lets the parameter move; `constantinit` would freeze it.
inc_NM = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc_NM[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M
        inc_NM[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0)
    end
end
mean_inc = mean(inc_NM)
initkw = (; init = Dict("V_KM" => function (st)
              # `fit` applies init BEFORE constantinit, so st["eps"] here is still the prior
              # draw even though the run will hold eps at EPS_PIN. Use the value the chain
              # will actually run at.
              epsref = type == 4 ? EPS_PIN : st["eps"]
              target = max(mean_inc - epsref, 1e-6)
              cur = mean(st["U_NK"] * st["V_KM"])
              return st["V_KM"] .* (target / max(cur, 1e-12))
          end))

if type == 1
    c = .01
    include(joinpath(ROOTDIR, "models/covid/ablation/covid1.jl"))
    model = covid1(N, M, K, a, b, c, d)
elseif type == 2
    c = 100.0
    include(joinpath(ROOTDIR, "models/covid/ablation/covid2.jl"))
    model = covid2(N, M, K, a, b, c, d, g, h, start_V1, start_V2)
elseif type == 3
    c = 100.0
    # covid3 takes a FIXED order D, not Dmax -- it is the homogeneous-D ablation.
    D > 0 || error("type 3 is the fixed-D ablation; pass D > 0 (got $D)")
    include(joinpath(ROOTDIR, "models/covid/ablation/covid3.jl"))
    model = covid3(N, M, K, D, a, b, c, d, g, h, start_V1, start_V2)
elseif type == 4
    c = 100.0
    # was: include(ablation/covid3.jl), which does not define covid4 at all, so this
    # branch could never run.
    include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
    # The 18-argument constructor defaults every sampler flag to false, which is the ORIGINAL
    # thinning sampler: theta drawn from the multinomial-thinned counts, no scale move. That
    # is the kernel whose theta autocorrelation is ~mu/g (10^2-10^4 sweeps on this matrix)
    # and which does not converge at any affordable run length. The Section 7.2 diagnostics
    # are reported from the HYBRID sampler, so the information-gain table has to come from
    # the same kernel or the two tables describe different posteriors.
    model = covid4(N, M, K, Q, Dmax, a, b, c, d, g, h,
                   start_V1, start_V2, alpha0, beta0, tauc, taud, start_tau,
                   false,   # hier_b
                   true,    # scale_move: the mu-preserving (theta, phi) rescaling
                   false,   # eps_mh: off, it did not deliver; eps is pinned anyway
                   true,    # use_offset
                   false,   # shear_move
                   true,    # collapsed: theta by the slice step on the S-target
                   false,   # iid_phi (test flag only)
                   true,    # phi_da: HYBRID -- phi keeps the CRT / Gamma forward pass
                   false)   # dir_theta
else
    error("unknown type $type (1-3 = ablation models, 4 = Section 7.2 final model)")
end

# ---- warm up the JIT so compile time is not inside the timed fit ------------------------
fit(model, data; nsamples=1, nburnin=1, nthin=1, mask=mask_NM, info=info,
    initseed=chainSeed, verbose=false, initkw..., Dkw...)

# Checkpoint every 500 sweeps. The inferred-D arm is 15000 sweeps at roughly 2 s/sweep on
# the full matrix, i.e. 8-10 h against a 12 h wall clock, so a job that is cut off has to be
# resumable rather than restarted. Resuming is crash recovery, not reproducibility -- see the
# caveat in MatrixMF.fit.
# Checkpoints go to SCRATCH, not to the project space. `fit` accumulates whole state dicts
# and D_NM alone is 915,680 entries per draw, so a late-run checkpoint is ~4 GB; with 20
# concurrent tasks that is ~75 GB, against ~117 GB free in /net/projects/schein-lab. Scratch
# is a different Lustre fileset on the same server with hundreds of TB free, and it is
# network-mounted, so a resubmitted task lands on any node and still finds its checkpoint.
# Scratch is PURGED, which is fine: the checkpoint is disposable the moment the output is
# written, and this script deletes it then. Only the ~40 MB output per run is kept, and that
# goes to the project space.
ckptdir = get(ENV, "CKPTDIR", joinpath(ROOTDIR, "output/covid/revisionsamples/ckpt"))
mkpath(ckptdir)
ckptfile = joinpath(ckptdir,
    "ckpt_covid$(type)D$(D)K$(K)Q$(Q)mask$(maskSeed)chain$(chainSeed).jld")

time_result = @elapsed samples = fit(model, data; nsamples=Nsamples, nburnin=nburnin,
                                     nthin=nthin, mask=mask_NM, initseed=chainSeed,
                                     verbose=true, info=info,
                                     ckptfile=ckptfile, ckptevery=500,
                                     initkw..., Dkw...)

# ---- held-out predictive density, retained PER DRAW ------------------------------------
# Same quantities as Section 7.1: a per-draw scalar for ESS/Rhat, and a per-cell posterior
# mean predictive density for the information rate.
heldout_idx = findall(mask_NM)
S = length(samples)
time_eval = @elapsed begin
    loglik_SH = Matrix{Float64}(undef, S, length(heldout_idx))
    Threads.@threads for h in eachindex(heldout_idx)
        ci = heldout_idx[h]
        for s in 1:S
            loglik_SH[s, h] = evalulateLogLikelihood(model, samples[s], data, info, ci[1], ci[2])
        end
    end
end
heldout_loglik_draws     = vec(sum(loglik_SH, dims=2))
heldout_logmeanexp_cells = [logsumexpvec(loglik_SH[:, h]) - log(S) for h in eachindex(heldout_idx)]
inforate = sum(heldout_logmeanexp_cells) / length(heldout_idx)
println("held-out information rate: ", inforate)

# ---- retain only what the summary needs, so the .jld stays manageable -------------------
# D_NM and the full latent matrices are 900k entries per draw; keeping every draw of every
# one of them would be tens of GB per chain. mu on a fixed random subset of cells is the
# identifiable functional the diagnostics are computed on.
Random.seed!(20260809)
mu_cells = [(rand(1:N), rand(1:M)) for _ in 1:200]
mu_draws = [Float64[sum(s["U_NK"][n, :] .* s["V_KM"][:, m]) for (n, m) in mu_cells] for s in samples]
Dmean_draws = Float64[mean(s["D_NM"]) for s in samples]
# theta for ALL counties, as Float32, plus s_k = sum_i theta_{i,k}. phi (V_KM) was already
# kept per draw; without theta and s_k the scaled trajectory s_k*phi_{k,t} and the loading
# profile theta_{i,k}/s_k -- the two objects the factor figure actually interprets -- cannot
# be reconstructed later, which is what forces a refit when the reporting choice changes.
# 3104 x 5 Float32 is 62 KB/draw, so 500 draws adds ~31 MB per run.
samplesnew = [Dict("eps" => s["eps"], "V_KM" => s["V_KM"],
                   "U_NK" => Float32.(s["U_NK"]),
                   "s_k" => vec(sum(s["U_NK"], dims=1)),
                   "sigma2_beta" => s["sigma2_beta"], "sigma2_tau" => s["sigma2_tau"],
                   "Tau_QM" => s["Tau_QM"]) for s in samples]

gitcommit = try readchomp(`git -C $ROOTDIR rev-parse --short HEAD`) catch; "unknown" end

folder = joinpath(ROOTDIR, "output/covid/revisionsamples/")
mkpath(folder)
save(folder * "covid$(type)D$(D)K$(K)Q$(Q)mask$(maskSeed)chain$(chainSeed).jld",
     "params", [maskSeed, chainSeed, D, K, Q, type],
     "samples", samplesnew,
     "mask", mask_NM,
     # --- timing: the fit only; the JIT warm-up and the evaluation are excluded from `time`
     "time", time_result,
     "time_eval", time_eval,
     "nsamples", Nsamples, "nburnin", nburnin, "nthin", nthin,
     "nsweeps", nburnin + nthin*Nsamples,
     "hypers", Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d, "g"=>g, "h"=>h,
                    "start_V1"=>start_V1, "start_V2"=>start_V2,
                    "alpha0"=>alpha0, "beta0"=>beta0, "tauc"=>tauc, "taud"=>taud,
                    "start_tau"=>start_tau, "Dmax"=>Dmax,
                    "endlength"=>endlength, "county_pct"=>county_pct),
     # --- held-out predictive quantities
     "heldout_loglik_draws", heldout_loglik_draws,
     "heldout_logmeanexp_cells", heldout_logmeanexp_cells,
     "heldout_idx", [(ci[1], ci[2]) for ci in heldout_idx],
     "inforate", inforate,
     # --- identifiable functionals for the diagnostics table
     "mu_cells", mu_cells, "mu_draws", mu_draws, "Dmean_draws", Dmean_draws,
     # --- environment, without which a timing table is not interpretable
     "nthreads", Threads.nthreads(), "julia", string(VERSION),
     "cpu", Sys.CPU_NAME, "ncores", Sys.CPU_THREADS, "gitcommit", gitcommit,
     "hostname", gethostname(), "slurmjob", get(ENV, "SLURM_JOB_ID", ""),
     "dedicated", get(ENV, "COVID_DEDICATED", "0"))
println("saved.")

# The checkpoint exists only to survive the wall clock. Once the output is written it is
# dead weight, and on a throttled array the difference between deleting it here and leaving
# it is tens of GB of scratch.
if isfile(ckptfile)
    rm(ckptfile; force=true)
    println("removed checkpoint ", ckptfile)
end
