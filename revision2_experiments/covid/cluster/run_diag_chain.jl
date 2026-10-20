# ONE chain of the Section 7.2 model on the FULL matrix, saving the functionals the
# diagnostics are computed on. One chain per job; analyze_diag.jl combines them.
#
#   julia -t $SLURM_CPUS_PER_TASK --project=. run_diag_chain.jl <chainSeed> <nsweeps> [outdir]
#
# Everything measured so far is on 100-300 county subsets, which are not merely a
# scaled-down version of the real problem -- Q = 25 is proportionate at N = 3104 and badly
# over-parameterised at N = 100. This run is to see what the diagnostics actually look like
# on the whole matrix.
#
# Design notes:
#   * ONE CHAIN PER JOB. Chains are independent, so this parallelises across the cluster
#     instead of running 4x longer in one job, and a chain that dies costs one chain.
#   * CHECKPOINTS every `ckpt_every` sweeps, and resumes from the checkpoint if one exists.
#     A 6-hour job that hits the wall clock can be resubmitted and continues.
#   * Saves per-draw FUNCTIONALS, not states. A full state is ~40 MB; the functionals are
#     ~70 MB for the whole chain.
#   * OPENBLAS_NUM_THREADS=1 must be set in the job script. The model file also calls
#     BLAS.set_num_threads(1), but only once it loads, and OpenBLAS sizes its pool from the
#     machine's physical cores, not the cgroup -- a 2-CPU job on a 64-core node otherwise
#     spawns 64 BLAS threads.

ROOTDIR = abspath(joinpath(@__DIR__, "../../.."))
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using CSV, DataFrames, Random, Printf, Statistics, LinearAlgebra, JLD, LogExpFunctions

chainSeed = parse(Int, ARGS[1])
nsweeps   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12000
outdir    = length(ARGS) >= 3 ? ARGS[3] : joinpath(ROOTDIR, "output/covid/diag")
BURNIN    = parse(Int, get(ENV, "BURNIN", string(nsweeps ÷ 2)))
mkpath(outdir)

# Every sweep is stored by default (author's preference, 2026-09-16); THIN=n in the
# environment keeps every n-th. Unthinned, a 15000-sweep chain file is ~250 MB (mu at 300
# cells, the 60-county loading profiles, the K x M trajectories, D at the cells, and the
# full-surface sufficient statistics), which is fine in project space.
const THIN       = parse(Int, get(ENV, "THIN", "1"))
const NSEG       = 8       # segments of the run for the full-surface sufficient statistics
const CKPT_EVERY = 500     # checkpoint cadence
const NCELLS     = 300     # cells for mu
const NCTY       = 200     # counties whose theta profile is tracked
# THETAFULL=1: track theta for EVERY county per draw rather than a subset of NCTY. This is
# the only array in the file that scales with N x K x nsweeps -- 3104 x 5 Float32 is 62 KB a
# draw, so 931 MB a chain against the subset's 60 MB. It buys ESS (not just Rhat) on all
# 15,520 loadings: S1th/S2th already give split-Rhat for all of them from their first two
# moments, but ESS needs the draws themselves. Implemented by widening `icty`, so the
# analyser needs no change -- though aligning 3104 x 5 per draw instead of 200 x 5 makes
# analyze_diag.jl's permutation search ~15x slower, which is analysis time, not run time.
# MASKSEED=n: which forecasting split to hold out. The information-gain table reports its
# SE across these splits (summarize_covid.jl uses 101..105), so the campaign needs the seed
# to be a parameter; 101 is the split the diagnostics run used, which makes diag_final_Q1
# the (D inferred, mask 101) cell of that campaign rather than a separate artefact.
const MASKSEED   = parse(Int, get(ENV, "MASKSEED", "101"))
# HOTHIN=n: store the PER-CELL held-out log densities every n-th sweep. The per-draw SUM is
# stored every sweep regardless (it is the scalar the ESS/Rhat table reports). This is a
# STORAGE cadence, not a sampling one -- the chain itself is never thinned. The per-cell
# matrix is 6208 Float32 per draw, i.e. 372 MB/chain unthinned against a 310 MB file, and
# the information rate is a posterior mean whose autocorrelation time is ~54 sweeps, so one
# record per 10 sweeps retains effectively all of the independent information.
const HOTHIN     = parse(Int, get(ENV, "HOTHIN", "10"))
# FULLSURF=0 drops the N x M sufficient statistics for mu and D (~117 MB/chain), which are
# what give split-Rhat on all 915,680 cells -- wanted for the reported diagnostics cell and
# dead weight for the other 119 cells of the grid. The N x K statistics for theta are NOT
# dropped: they are ~0.5 MB and they are what Figure 10's per-county loadings are built
# from, for every county rather than the 200 tracked per draw.
const FULLSURF   = get(ENV, "FULLSURF", "1") == "1"
# BURNIN=n: sweeps at or below n are warm-up and are EXCLUDED from any derived posterior
# summary written here. The chain itself is stored in full (THIN=1) so the window can be
# revisited, but heldout_logmeanexp_cells is a posterior mean and must not average over
# warm-up. Default nsweeps/2, which is the 7500-of-15000 window every reported table uses.

println("Julia ", VERSION, "  threads = ", Threads.nthreads(),
        "  BLAS = ", BLAS.get_num_threads(), "  host = ", gethostname())
flush(stdout)

# ---- data: full matrix, t = 0 split off as the conditioning value --------------------
cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"), DataFrame))[2:end, 4:end]
Y0_N = Int.(reshape(cumdf[:,1], :, 1))
Y_NM = cumdf[:, 2:end]
data = Dict("Y_NM" => Y_NM); N, M = size(Y_NM); info = Dict("Y0_N" => Y0_N)
# KFAC / QDIM (env): number of factors and the dimension of the logistic factor model
# for the order prior rho_{it} = logistic(beta_i' tau_t). Paper: K = 5, Q = 25.
K    = parse(Int, get(ENV, "KFAC", "5"))
Q    = parse(Int, get(ENV, "QDIM", "25"))
Dmax = 9
println("N = $N, M = $M, K = $K, Q = $Q, Dmax = $Dmax, nsweeps = $nsweeps"); flush(stdout)

# ---- held-out mask: the same forecasting mask the run script uses ---------------------
# Identical construction to analysis/covid/runcovid_heldout.jl's make_forecasting_mask, so
# MASKSEED=101 reproduces that script's mask exactly.
Random.seed!(MASKSEED)
mc = rand(N) .< 0.2
mask_NM = falses(N, M)
for t in (M-9):M; mask_NM[:, t] = mc; end
heldout = findall(mask_NM)
const ENDLEN = 10                 # forecast horizon: the mask holds out the last 10 days
hocty = findall(mc)               # the counties those held-out days belong to

# fixed monitoring sets, identical across chains
Random.seed!(20260810)
cells = [(rand(1:N), rand(1:M)) for _ in 1:NCELLS]
THETAFULL = get(ENV, "THETAFULL", "0") == "1"
icty  = THETAFULL ? collect(1:N) : sort(randperm(N)[1:NCTY])
THETAFULL && println("theta tracked for ALL $N counties (", round(N*K*4*nsweeps/1e9, digits=2), " GB/chain)")

inc = Matrix{Float64}(undef, N, M)
for i in 1:N
    inc[i,1] = max(Y_NM[i,1] - Y0_N[i], 0)
    for t in 2:M; inc[i,t] = max(Y_NM[i,t] - Y_NM[i,t-1], 0); end
end
mean_inc = mean(inc)

# eps is FIXED, not inferred: its posterior is determined by wherever it is initialised, it
# is confounded with the overall factor level, and fixing it gave the best held-out density
# of the three arms tested. Stated as a constant in the paper rather than reported as an
# inference.
const EPS_FIXED = 0.5

# DIRICHLET=1: theta on the county simplex, phi carries the scale, c from the scale rule
# (relative step sd 2% at phibar = mean daily increment x N / K), E[phi_1] = phibar.
DIRICHLET = get(ENV, "DIRICHLET", "0") == "1"
inc_mean = sum(max(Y_NM[i,t] - (t == 1 ? Y0_N[i] : Y_NM[i,t-1]), 0) for i in 1:N, t in 1:M) / (N * M)
phibar = inc_mean * N / K
cphi = DIRICHLET ? c_for_scale(0.02, phibar) : parse(Float64, get(ENV, "CPHI", "100"))
# ATHETA: shape of the theta prior (Dirichlet / Gamma); < 1 gives sparse county shares.
ATHETA = parse(Float64, get(ENV, "ATHETA", "1.0"))
# PHIGROUP=1: tilt + curvature group moves on each factor trajectory after the theta step
PHI_GROUP[] = get(ENV, "PHIGROUP", "0") == "1"; println("phi group moves: ", PHI_GROUP[])
# DFIX=n: pin every D_it at n (n = 1 is the Poisson baseline for the information-gain
# headline); the D / PG / tau / beta block is skipped entirely.
DFIX = parse(Int, get(ENV, "DFIX", "0")); DFIX > 0 && println("D pinned at ", DFIX)
# ANNEAL=n: hold D at 1 for the first n sweeps of the warm-up, then infer it. The D = 1
# likelihood is flat enough that the factor posterior is not ridged, so chains settle into
# the same basin before the order sharpens; D barely affects the estimate of mu, and a
# D = 1 sweep costs ~1.1 s against ~4.5 s at D ~ 8, so the warm-up is nearly free. The
# annealed sweeps are burn-in, so the kernel producing the reported draws is the certified
# one. The rho layer (PG, tau, beta) is frozen while D is pinned: conditioning it on
# D == 1 in every cell would drive rho to zero, which is not a useful starting point.
ANNEAL = parse(Int, get(ENV, "ANNEAL", "0")); ANNEAL > 0 && println("annealing: D = 1 for the first ", ANNEAL, " sweeps")
# NMFINIT=1: start each chain at its own KL-NMF factorisation of the increments (300
# multiplicative updates from a per-chain random start). NMF is non-convex, so the chains
# still start in different places -- this is a data-informed dispersed init, not a common
# one -- but each starts at a plausible temporal structure instead of a prior draw.
NMFINIT = get(ENV, "NMFINIT", "0") == "1"; NMFINIT && println("init: per-chain KL-NMF of the increments")
function nmf_kl(V, K, rng, iters=300)
    n, m = size(V); W = rand(rng, n, K) .+ 0.1; H = rand(rng, K, m) .+ 0.1
    for _ in 1:iters
        WH = max.(W*H, 1e-10); H .*= (transpose(W) * (V ./ WH)) ./ max.(transpose(sum(W, dims=1)), 1e-10)
        WH = max.(W*H, 1e-10); W .*= ((V ./ WH) * transpose(H)) ./ max.(transpose(sum(H, dims=2)), 1e-10)
    end
    W, H
end
model = covid4(N, M, K, Q, Dmax, ATHETA, 1.0, cphi, 0.1, 0.5, 1.0,
               DIRICHLET ? 1.0 : 0.01, DIRICHLET ? 1 / phibar : 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
               false,   # hier_b
               !DIRICHLET,  # scale_move: moot on the simplex
               false,   # eps_mh: off, it did not deliver
               true,    # use_offset
               false,   # shear_move
               get(ENV, "COLLAPSED", "0") in ("1", "2") || DIRICHLET,   # theta by the collapsed slice step
               false,   # iid_phi (test flag only)
               get(ENV, "COLLAPSED", "0") == "1" || DIRICHLET,   # 1 = HYBRID: phi keeps the CRT/forward pass; 2 = full collapse
               DIRICHLET)
println("a_theta = ", ATHETA)
println("sampler: ", DIRICHLET ? "Dirichlet theta, hybrid, c = $(round(cphi, digits=3)), phibar = $(round(phibar))" :
                     get(ENV, "COLLAPSED", "0") == "1" ? "hybrid (theta collapsed, phi PGDS)" :
                     get(ENV, "COLLAPSED", "0") == "2" ? "fully collapsed" : "thinning")

ckptfile = joinpath(outdir, "ckpt_chain$(chainSeed).jld")
outfile  = joinpath(outdir, "diag_chain$(chainSeed).jld")

# ---- initialise, or resume ------------------------------------------------------------
if isfile(ckptfile)
    ck = load(ckptfile)
    state = ck["state"]; s0 = ck["sweep"]
    sw = ck["sw"]; ll = ck["ll"]; mu = ck["mu"]; th = ck["th"]; ph = ck["ph"]
    dm = ck["dm"]; dc = get(ck, "dc", Vector{Vector{Int8}}())
    hoc = get(ck, "hoc", Vector{Vector{Float32}}()); hosw = get(ck, "hosw", Int[])
    fc = get(ck, "fc", Vector{Matrix{Int32}}())
    S1mu = ck["S1mu"]; S2mu = ck["S2mu"]; S1d = ck["S1d"]; S2d = ck["S2d"]; segn = ck["segn"]
    S1th = ck["S1th"]; S2th = ck["S2th"]
    println("resuming from checkpoint at sweep $s0"); flush(stdout)
else
    Random.seed!(chainSeed)
    state = sample_prior(model, info)
    state["eps"] = EPS_FIXED
    DFIX > 0 && (state["D_NM"] = fill(DFIX, N, M))
    ANNEAL > 0 && (state["D_NM"] = fill(1, N, M))
    # phi's prior draw is SCALE-CORRECTED: Gamma(0.01, 1) starts the growth term ~0.01 while
    # the data needs ~14, and the augmentation allocates counts in proportion to current
    # values, so a start near zero never grows. Only the magnitude is corrected, so chains
    # stay dispersed and Rhat keeps its meaning.
    if NMFINIT
        incmat = Float64[max(Y_NM[i,t] - (t == 1 ? Y0_N[i] : Y_NM[i,t-1]), 0) for i in 1:N, t in 1:M]
        W, H = nmf_kl(incmat, K, MersenneTwister(chainSeed))
        # multiplicative updates leave exact zeros; every parameter here is strictly positive
        state["U_NK"] = max.(W, 1e-6); state["V_KM"] = max.(H, 1e-6)
    else
        target = max(mean_inc - state["eps"], 1e-6)
        state["V_KM"] = state["V_KM"] .* (target / max(mean(state["U_NK"] * state["V_KM"]), 1e-12))
    end
    s0 = 0
    sw = Int[]; ll = Float64[]
    mu = Vector{Vector{Float32}}(); th = Vector{Matrix{Float32}}(); ph = Vector{Matrix{Float32}}()
    dm = Float64[]
    dc = Vector{Vector{Int8}}()      # D at the tracked cells: the per-cell analogue of mu for Rhat
    # PER-CELL held-out log predictive densities, every HOTHIN-th sweep. summarize_covid.jl
    # needs these (not the sum) to form the pooled information rate: logsumexp over chains
    # per cell, then averaged over cells. `hosw` records which sweep each row came from.
    hoc  = Vector{Vector{Float32}}(); hosw = Int[]
    # POSTERIOR PREDICTIVE forecast trajectories for the held-out counties, for Figure 9's
    # credible intervals. predict_x chains the 10 steps forward from the last OBSERVED day,
    # feeding each draw into the next day's offset, which is the quantity the figure shows.
    # Stored on the HOTHIN cadence: 646 counties x 10 days as Int32 is ~26 KB per record.
    fc = Vector{Matrix{Int32}}()
    # FULL-SURFACE sufficient statistics: per-cell sum and sum of squares of the growth
    # term mu = theta*phi and of D over the stored draws, in NSEG equal segments of the run.
    # Split-Rhat on every one of the N*M cells follows from these alone (the analyser forms
    # any window from whole segments and splits it into halves); the 300-cell sample above
    # keeps the rank-normalised Rhat and ESS. Float32 keeps the file at ~120 MB.
    nsegalloc = FULLSURF ? NSEG : 0
    S1mu = zeros(Float32, N, M, nsegalloc); S2mu = zeros(Float32, N, M, nsegalloc)
    S1d  = zeros(Float32, N, M, nsegalloc); S2d  = zeros(Float32, N, M, nsegalloc)
    segn = zeros(Int, NSEG)
    # ... and of the normalised loadings theta_ik / s_k for ALL counties (N x K, ~1 MB).
    # Labels are aligned across chains at analysis time by one permutation per chain
    # (labels do not switch within a Gibbs chain); the 60-county draws keep the per-draw check.
    S1th = zeros(Float32, N, K, NSEG); S2th = zeros(Float32, N, K, NSEG)
end

t0 = time()
for s in (s0+1):nsweeps
    global state
    pin = DFIX > 0 ? DFIX : (s <= ANNEAL ? 1 : 0)
    (_d, state) = pin > 0 ? backward_sample(model, data, state, mask_NM; skipupdate=["D_NM"]) :
                            backward_sample(model, data, state, mask_NM)
    state["eps"] = EPS_FIXED
    pin > 0 && (state["D_NM"] = fill(pin, N, M))
    if s % THIN == 0
        U = state["U_NK"]::Matrix{Float64}; V = state["V_KM"]::Matrix{Float64}
        # one pass over the held-out cells serves both tables: the SUM every sweep (the
        # scalar whose ESS/Rhat is reported) and, every HOTHIN-th sweep, the per-cell vector
        # the information rate is built from.
        keepcell = (s % HOTHIN == 0)
        hv = keepcell ? Vector{Float32}(undef, length(heldout)) : Float32[]
        l = 0.0
        for (hi, ci) in enumerate(heldout)
            lh = evalulateLogLikelihood(model, state, data, info, ci[1], ci[2])
            l += lh
            keepcell && (hv[hi] = Float32(lh))
        end
        if keepcell
            push!(hoc, hv); push!(hosw, s)
            fcm = Matrix{Int32}(undef, length(hocty), ENDLEN)
            for (ci, n) in enumerate(hocty)
                fcm[ci, :] = Int32.(predict_x(model, state, info, n, M - ENDLEN,
                                              Y_NM[n, M - ENDLEN], ENDLEN))
            end
            push!(fc, fcm)
        end
        sk = vec(sum(U, dims=1))
        push!(sw, s); push!(ll, l); push!(dm, mean(state["D_NM"]))
        push!(mu, Float32[sum(U[n,:] .* V[:,m]) for (n,m) in cells])
        push!(dc, Int8[state["D_NM"][n,m] for (n,m) in cells])
        push!(th, Float32.(U[icty, :] ./ reshape(sk, 1, K)))   # loading profile
        push!(ph, Float32.(V .* reshape(sk, K, 1)))            # scaled trajectory
        seg = min(NSEG, 1 + (s - 1) * NSEG ÷ nsweeps)
        if FULLSURF
            G = U * V; Dm = state["D_NM"]
            @views begin
                S1mu[:, :, seg] .+= Float32.(G);  S2mu[:, :, seg] .+= Float32.(G .* G)
                S1d[:, :, seg]  .+= Float32.(Dm); S2d[:, :, seg]  .+= Float32.(Dm .* Dm)
            end
        end
        # theta for ALL counties, always: Figure 10 interprets the loadings geographically,
        # and `th` above only tracks NCTY of them.
        Tn = U ./ reshape(sk, 1, K)
        @views begin S1th[:, :, seg] .+= Float32.(Tn); S2th[:, :, seg] .+= Float32.(Tn .* Tn) end
        segn[seg] += 1
    end
    if s % CKPT_EVERY == 0
        save(ckptfile, "state", state, "sweep", s, "sw", sw, "ll", ll,
             "mu", mu, "th", th, "ph", ph, "dm", dm, "dc", dc,
             "hoc", hoc, "hosw", hosw, "fc", fc,
             "S1mu", S1mu, "S2mu", S2mu, "S1d", S1d, "S2d", S2d, "segn", segn, "S1th", S1th, "S2th", S2th)
        @printf("  sweep %d/%d  %.1f min  (%.2f s/sweep)  meanD %.2f\n",
                s, nsweeps, (time()-t0)/60, (time()-t0)/(s-s0), dm[end]); flush(stdout)
    end
end

# --dirty matters here: this campaign is run from an rsynced working tree that is AHEAD of
# the committed state, so a bare commit hash would misdescribe the code that produced the
# output. `git describe --always --dirty` appends "-dirty" when the tree is modified.
gitcommit = try readchomp(`git -C $ROOTDIR describe --always --dirty`) catch; "unknown" end
# heldout_logmeanexp_cells: log of the per-chain posterior MEAN predictive density per
# held-out cell, which is what summarize_covid.jl pools across chains for the information
# rate. Computed here so the analyser never has to hold the whole hoc matrix.
hokeep = findall(>(BURNIN), hosw)
isempty(hokeep) && @warn "no per-cell held-out draws past BURNIN=$BURNIN; information rate will be empty"
heldout_logmeanexp_cells = isempty(hokeep) ? Float64[] :
    [logsumexp(Float64[hoc[t][h] for t in hokeep]) - log(length(hokeep))
     for h in eachindex(heldout)]
inforate = isempty(heldout_logmeanexp_cells) ? NaN : mean(heldout_logmeanexp_cells)
println("held-out information rate: ", inforate)

save(outfile, "sw", sw, "ll", ll, "mu", mu, "th", th, "ph", ph, "dm", dm, "dc", dc,
     "hoc", hoc, "hosw", hosw, "hothin", HOTHIN,
     "fc", fc, "fc_counties", hocty, "fc_horizon", ENDLEN,
     "heldout_logmeanexp_cells", heldout_logmeanexp_cells, "inforate", inforate,
     "maskseed", MASKSEED, "dfix", DFIX, "fullsurf", FULLSURF, "burnin", BURNIN,
     "n_inforate_draws", length(hokeep),
     "S1mu", S1mu, "S2mu", S2mu, "S1d", S1d, "S2d", S2d, "segn", segn, "nseg", NSEG, "S1th", S1th, "S2th", S2th,
     "cells", cells, "icty", icty, "heldout", [(c[1], c[2]) for c in heldout],
     "chainSeed", chainSeed, "nsweeps", nsweeps, "thin", THIN,
     "N", N, "M", M, "K", K, "Q", Q, "Dmax", Dmax, "eps_fixed", EPS_FIXED,
     "sampler", DIRICHLET ? "dirichlet-hybrid" : get(ENV, "COLLAPSED", "0") == "1" ? "hybrid" :
                get(ENV, "COLLAPSED", "0") == "2" ? "collapsed" : "thinning",
     "crt_tol", CRT_TOL[], "a_theta", ATHETA, "c", cphi, "KFAC", K, "QDIM", Q, "DFIX", DFIX, "ANNEAL", ANNEAL, "NMFINIT", NMFINIT,
     "nthreads", Threads.nthreads(), "julia", string(VERSION),
     "cpu", Sys.CPU_NAME, "gitcommit", gitcommit, "hostname", gethostname(),
     "slurmjob", get(ENV, "SLURM_JOB_ID", ""), "seconds", time()-t0)
# ---- durable companion -------------------------------------------------------------
# DURABLEDIR, when set, gets a copy of this file WITHOUT `th`. With THETAFULL=1 the theta
# trace is 931 MB of a ~1.1 GB file, so the split is: the full file lives wherever OUTDIR
# points (scratch, which has hundreds of TB but PURGES), and the companion -- which is
# everything the paper's tables and figures are actually built from -- lives in project
# space, which is backed up and nearly full. Nothing the paper needs is lost to a purge;
# what a purge takes is the freedom to recompute per-draw theta diagnostics later.
durabledir = get(ENV, "DURABLEDIR", "")
if durabledir != ""
    mkpath(durabledir)
    dfile = joinpath(durabledir, "summary_chain$(chainSeed).jld")
    src = load(outfile)
    delete!(src, "th")
    # JLD.save takes alternating name/value arguments, not splatted Pairs
    sargs = Any[]
    for (k, v) in src; push!(sargs, k); push!(sargs, v); end
    save(dfile, sargs...)
    @printf("durable companion (%.0f MB, no theta trace) -> %s\n",
            filesize(dfile)/1e6, dfile); flush(stdout)
end

rm(ckptfile, force=true)
@printf("done: %d sweeps in %.1f min -> %s\n", nsweeps, (time()-t0)/60, outfile)
