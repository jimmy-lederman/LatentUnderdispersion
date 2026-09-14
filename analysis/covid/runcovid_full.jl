# Section 7.2 full-data run for the COVID-19 case study (Figures 9 and 10).
#
#   julia -t auto --project=. analysis/covid/runcovid_full.jl <chainSeed> <K> <Q>
#
# Fits the final model of Section 7.2 to all counties and days with no mask, retaining
# the draws the factor figures and the forecast intervals are built from.
#
# This script previously (a) passed skipupdate= to fit(), which fit() no longer accepts,
# (b) pinned D_{i,t} = 1 and V_KM = 1 through constantinit for the whole run, which makes
# it a Poisson fit with the order-statistic layer switched off, and (c) kept a single
# draw (nsamples = 1, nthin = 1). Those were debugging leftovers. Initialisation is now a
# draw from the prior for every parameter, matching Sections 6.3 and 7.1.
println("opened file"); flush(stdout)
using Dates, CSV, DataFrames, Random, JLD, Statistics
println("threads = ", Threads.nthreads()); flush(stdout)

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))
days  = cumdf[1, 4:end]
state = cumdf[2:end, 1]
fips  = cumdf[2:end, 2]
pops  = log.(cumdf[2:end, 3])
cumdf = cumdf[2:end, 4:end]
# Column 1 is t = 0, the count Eq. (12) conditions on, not a modelled day. See the note in
# runcovid_heldout.jl: it used to be both Y0_N and Y_NM[:,1].
Y0_N  = Int.(reshape(cumdf[:, 1], :, 1))
Y_NM  = cumdf[:, 2:end]
data  = Dict("Y_NM" => Y_NM)
N = size(Y_NM, 1)
M = size(Y_NM, 2)
info = Dict("Y0_N" => Y0_N, "pop_N" => pops,
            "days_M" => Vector(days)[2:end], "state_N" => state)
println("N = $N counties, M = $M modelled days (t = 0 held out as Y0)")

chainSeed = parse(Int, ARGS[1])
K = parse(Int, ARGS[2])
Q = parse(Int, ARGS[3])

a = 1.0; b = 1.0
c = 100.0; d = .1
g = .5; h = 1.0
start_V1 = .01; start_V2 = 1.0
alpha0 = 1.0; beta0 = 1.0
start_tau = 0.0; tauc = 1.0; taud = 0.0
Dmax = 9

nsamples = 500
nburnin  = 2000
nthin    = 4

model = covid4(N, M, K, Q, Dmax, a, b, c, d, g, h,
               start_V1, start_V2, alpha0, beta0, tauc, taud, start_tau)

# phi's prior draw is SCALE-CORRECTED before the chain starts -- see the long note in
# runcovid_heldout.jl. Briefly: phi_{k,1} ~ Gamma(0.01, 1) is a near-spike at zero, and
# because the augmentation allocates latent counts in proportion to each component's CURRENT
# share of a rate dominated by Y_{i,t-1}, a start near zero never grows. On synthetic data
# from this model with the truth known, prior-initialised chains sit at a mean growth of
# 0.0-0.6 against a truth of 32 for 3000 sweeps and do not move; scale-corrected chains
# start dispersed and converge onto it. Only the magnitude is corrected, so the chains keep
# the dispersion of their own prior draws.
inc_NM = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc_NM[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M
        inc_NM[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0)
    end
end
mean_inc = mean(inc_NM)
initkw = (; init = Dict("V_KM" => function (st)
              target = max(mean_inc - st["eps"], 1e-6)
              cur = mean(st["U_NK"] * st["V_KM"])
              return st["V_KM"] .* (target / max(cur, 1e-12))
          end))

# warm up the JIT so compile time is not inside the timed fit
fit(model, data; nsamples=1, nburnin=1, nthin=1, info=info, initseed=chainSeed,
    verbose=false, initkw...)

time_result = @elapsed samples = fit(model, data; nsamples=nsamples, nburnin=nburnin,
                                     nthin=nthin, info=info, initseed=chainSeed,
                                     verbose=true, initkw...)

# Keep the quantities the figures need. D_NM is 900k entries per draw, so its posterior
# mean is accumulated rather than every draw being stored.
Dmean_NM = reduce(+, (Float64.(s["D_NM"]) for s in samples)) ./ length(samples)
samplesnew = [Dict("U_NK" => s["U_NK"], "V_KM" => s["V_KM"], "eps" => s["eps"],
                   "Tau_QM" => s["Tau_QM"], "Beta_NQ" => s["Beta_NQ"],
                   "sigma2_beta" => s["sigma2_beta"], "sigma2_tau" => s["sigma2_tau"])
              for s in samples]

gitcommit = try readchomp(`git -C $ROOTDIR rev-parse --short HEAD`) catch; "unknown" end

folder = joinpath(ROOTDIR, "output/covid/full_samples/")
mkpath(folder)
save(folder * "fullsample_seed$(chainSeed)K$(K)Q$(Q).jld",
     "params", [chainSeed, K, Q],
     "samples", samplesnew,
     "Dmean_NM", Dmean_NM,
     "time", time_result,
     "nsamples", nsamples, "nburnin", nburnin, "nthin", nthin,
     "nsweeps", nburnin + nthin*nsamples,
     "hypers", Dict("a"=>a, "b"=>b, "c"=>c, "d"=>d, "g"=>g, "h"=>h,
                    "start_V1"=>start_V1, "start_V2"=>start_V2,
                    "alpha0"=>alpha0, "beta0"=>beta0, "tauc"=>tauc, "taud"=>taud,
                    "start_tau"=>start_tau, "Dmax"=>Dmax),
     "nthreads", Threads.nthreads(), "julia", string(VERSION),
     "cpu", Sys.CPU_NAME, "ncores", Sys.CPU_THREADS, "gitcommit", gitcommit,
     "hostname", gethostname(), "slurmjob", get(ENV, "SLURM_JOB_ID", ""))
println("saved.")
