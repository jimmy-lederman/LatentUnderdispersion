println(Threads.nthreads())

using CSV
using DataFrames
using Random
using JLD
using Base.Filesystem

maskSeed = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[2])
D = parse(Int, ARGS[3])
type = parse(Int, ARGS[4])
g = parse(Int, ARGS[5])
T = 99 #because running on full

ROOTDIR = joinpath(@__DIR__, "../..")

datafile = joinpath(ROOTDIR, "data/flights/flights_final.csv")
df = CSV.read(datafile, DataFrame)
df = select(df, Not(1))
Y_NM = convert(Matrix{Int}, Matrix(select(df, 1)))
dist_NM = convert(Matrix{Int}, Matrix(select(df, 2)))
I_N3 = convert(Matrix{Int}, Matrix(select(df, 3:4)))

data = Dict("Y_NM"=>Y_NM)
info = Dict("I_N3"=>I_N3, "dist_NM"=>dist_NM)

home_N = info["I_N3"][:,1]
away_N = info["I_N3"][:,2]
dist_NM = info["dist_NM"]
N = length(home_N)

routes_R4 = Vector{Any}()
for t1 in 1:T
    for t2 in 1:T
        bitvector = home_N .== t1 .&& away_N .== t2
        if sum(bitvector) != 0
            indices = findall(bitvector)
            distances = dist_NM[indices, 1]
            @assert all(x -> x == distances[1], distances) "Not all distances are equal"
            push!(routes_R4, [t1,t2,indices, distances[1],Y_NM[indices,1],length(indices)])
        end
    end
end
routes_R4 = Matrix(hcat(routes_R4...)')
routes_N = zeros(Int, N)
for n in 1:N
    t1 = home_N[n]
    t2 = away_N[n]
    routes_N[n] = findfirst(r -> r[1] == t1 && r[2] == t2, eachrow(routes_R4))
end
I_N3 = hcat(I_N3, routes_N)


R = size(routes_R4)[1]

N = size(Y_NM)[1]
M = size(Y_NM)[2]
a =1
b =.01
alpha = beta = 1 #not used
tau2 = 50^2
gforwards = [x -> x, sqrt]
gbakwards = [x -> x, x -> x^2]

Nsamples = 500
nbunrin = 1000
nthin = 1


Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .2

#need to format things such that the model doesn't see the test data at all
Ntest = sum(mask_NM)
Ntrain = N - Ntest
routes_Ntrain = routes_N[mask_NM[:,1] .== 0]
#per-route training indices and counts, held as concrete types: routes_R4 is a Matrix{Any},
#so indexing it inside the models' route loops forces a dynamic dispatch on every access
route_idx = [findall(routes_Ntrain .== r) for r in 1:R]
route_n = [length(route_idx[r]) for r in 1:R]
datatrain = Dict{String,Any}(
    "Y_NM"=>reshape(data["Y_NM"][mask_NM .== 0],:,1),
    "route_idx"=>route_idx,
    "route_n"=>route_n,
    "routes_N"=>routes_Ntrain)
info = Dict("routes_R2"=>routes_R4[:,3:6], "routes_N"=>routes_Ntrain)
# type 1 = MedPoisson, type 2 = STAR. Both use the hierarchical mu_k prior (referee 2(a)):
# the prior on mu_k is learned across routes rather than fixed. Types 3 and 4 are the flat
# priors of the original submission, kept so the hierarchical-vs-flat comparison is
# reproducible.
Dmax = 9
a0 = 1.0; b0 = 0.01     # hyperprior on the population rate of mu_k
c0 = 1.0; d0 = 1.0      # hyperprior on the population variance of mu_k (STAR)
Dkw = D > 0 ? (constantinit=Dict("D_R"=>fill(D,R)), skipupdatealways=["D_R"]) : NamedTuple()

if type == 1
    include(joinpath(ROOTDIR, "models/flights/flights_hier.jl"))
    model = flights_hier(Ntrain, M, R, Dmax, a, a0, b0, alpha, beta)
elseif type == 2
    include(joinpath(ROOTDIR, "models/flights/flights_STAR_hier.jl"))
    model = flights_STAR_hier(Ntrain, M, R, alpha, beta, tau2, c0, d0, gforwards[g], gbakwards[g])
elseif type == 3
    include(joinpath(ROOTDIR, "models/flights/flights.jl"))
    model = flights(Ntrain, M, R, Dmax, a, b, alpha, beta)
elseif type == 4
    include(joinpath(ROOTDIR, "models/flights/flights_STAR.jl"))
    model = flights_STAR(Ntrain, M, R, alpha, beta, tau2, gforwards[g], gbakwards[g])
else
    error("unknown type $type (1=MedPois hier, 2=STAR hier, 3=MedPois flat, 4=STAR flat)")
end

# warm up the JIT so compile time is not inside the timed fit
fit(model, datatrain; nsamples=1, nburnin=1, nthin=1, mask=nothing, info=info,
    initseed=chainSeed, verbose=false, Dkw...)

t = @elapsed samples = fit(model, datatrain; nsamples=Nsamples, nburnin=nbunrin, nthin=nthin,
                           mask=nothing, info=info, initseed=chainSeed, verbose=true, Dkw...)

samplesnew = (type == 1 || type == 3) ?
    [Dict("U_R"=>s["U_R"], "D_R"=>s["D_R"]) for s in samples] :
    [Dict("U_R"=>s["U_R"], "sigma2_R"=>s["sigma2_R"]) for s in samples]

# ---- held-out predictive density, retained PER DRAW ------------------------------------
# This is the input to both the information gain in Table 2 and the ESS diagnostics, so it
# is computed once here and saved rather than recomputed from `samples` for every chain.
# The full model (all N rows) is used because the fitted state is route-level.
modelfull = type == 1 ? flights_hier(N, M, R, Dmax, a, a0, b0, alpha, beta) :
            type == 2 ? flights_STAR_hier(N, M, R, alpha, beta, tau2, c0, d0, gforwards[g], gbakwards[g]) :
            type == 3 ? flights(N, M, R, Dmax, a, b, alpha, beta) :
                        flights_STAR(N, M, R, alpha, beta, tau2, gforwards[g], gbakwards[g])
route_idx_full = [findall(routes_N .== r) for r in 1:R]
datafull = Dict{String,Any}("Y_NM"=>Y_NM, "route_idx"=>route_idx_full,
                            "route_n"=>[length(route_idx_full[r]) for r in 1:R],
                            "routes_N"=>routes_N)

heldout_rows = findall(mask_NM[:,1])
S = length(samplesnew)
t_eval = @elapsed begin
    # loglik_SH[s, h] = log p(y_h | theta_s) for held-out flight h and draw s
    loglik_SH = Matrix{Float64}(undef, S, length(heldout_rows))
    for (h, row) in enumerate(heldout_rows), s in 1:S
        loglik_SH[s, h] = evalulateLogLikelihood(modelfull, samplesnew[s], datafull, info, row, 1)
    end
end
# per-draw scalar functional (for ESS / Rhat on the predictive density)
heldout_loglik_draws = vec(sum(loglik_SH, dims=2))
# per-cell log posterior-mean predictive density (for information gain, incl. on subsets)
heldout_logmeanexp_cells = [logsumexpvec(loglik_SH[:, h]) - log(S) for h in 1:length(heldout_rows)]
inforate = sum(heldout_logmeanexp_cells) / length(heldout_rows)

gitcommit = try readchomp(`git -C $ROOTDIR rev-parse --short HEAD`) catch; "unknown" end

params = [maskSeed,chainSeed,D,type,g]
folder = joinpath(ROOTDIR, "output/flights/revisionsamples/")
mkpath(folder)  #the samples subfolder itself, not just output/flights/
name = type == 1 ? "MedPoissonD$(D)" :
       type == 2 ? "STARg$(g)" :
       type == 3 ? "MedPoissonFlatD$(D)" : "STARFlatg$(g)"
save(folder*"$(name)mask$(maskSeed)chain$(chainSeed).jld",
     "params", params,                 # [maskSeed, chainSeed, D, type, g]
     "samples", samplesnew,
     "mask", mask_NM,
     # --- timing: fit only; the JIT warm-up and the evaluation below are excluded from `time`
     "time", t,
     "time_eval", t_eval,
     # --- sampler effort, so `time` can be converted to per-iteration cost after the fact
     "nsamples", Nsamples, "nburnin", nbunrin, "nthin", nthin,
     "nsweeps", nbunrin + nthin*Nsamples,
     # --- model/prior settings, so the model can be rebuilt for any later analysis
     "hypers", Dict("a"=>a, "b"=>b, "a0"=>a0, "b0"=>b0, "alpha"=>alpha, "beta"=>beta,
                    "tau2"=>tau2, "c0"=>c0, "d0"=>d0, "Dmax"=>Dmax, "maskfrac"=>0.2),
     # --- held-out predictive quantities (per draw, and per cell)
     "heldout_loglik_draws", heldout_loglik_draws,
     "heldout_logmeanexp_cells", heldout_logmeanexp_cells,
     "heldout_rows", heldout_rows,
     "inforate", inforate,
     "ntrain", Ntrain, "ntest", Ntest,
     # --- environment, without which a timing table is not interpretable
     "nthreads", Threads.nthreads(), "julia", string(VERSION),
     "cpu", Sys.CPU_NAME, "ncores", Sys.CPU_THREADS, "gitcommit", gitcommit,
     "hostname", gethostname(), "slurmjob", get(ENV, "SLURM_JOB_ID", ""),
     # Set FLIGHTS_DEDICATED=1 only when this run had the machine to itself. Chains run
     # concurrently (e.g. under xargs -P) inflate wall-clock by 25-35% through contention,
     # so the summary script must use only dedicated runs for the timing column. The
     # statistical results (samples, information gain, ESS) are unaffected by contention.
     "dedicated", get(ENV, "FLIGHTS_DEDICATED", "0") == "1")
