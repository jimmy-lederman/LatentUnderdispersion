# Library: one cell of the Section 6.3 production run (see run_task.jl): fit (dataset, model, seed),
# evaluate, and write a single CSV row. Designed for cluster array jobs or
# local parallel driving; cells are independent.
#
# Usage: julia --project=. run_cell.jl <dataset> <model> <seed> [nsamples] [nburnin] [nthin] [nchains]
#   dataset: CMP | GC | Poisson | NB
#   model:   poisson | medpois | nb | mednb | star_id | star_sqrt
#            | starnnf_id | starnnf_sqrt
#   seed:    mask/replicate seed (1..25)
# Defaults: nsamples=1000 nburnin=5000 nthin=5 nchains=4  (2x the paper's
#   retained samples, so every model clears ESS >= 1000 on evaluated
#   functionals with margin)
#
# Output row (results/<dataset>_<model>_<seed>.csv):
#   dataset,model,seed,inforate,essmu_med,essmu_min,rhatmu_med,rhatmu_max,
#   essll,rhatll,cosine_to_truth,time_s,iters_per_chain
# cosine_to_truth is NaN when the dataset has no saved true factors.

include("../../../models/STARmodels/revision2/STARMFNNfloor.jl")
include("../../../models/STARmodels/STARMF.jl")
include("../../../models/other_models/OrderStatisticPoissonMFprior.jl")
include("../../../models/other_models/OrderStatisticNegBinMF.jl")
include("../../../models/other_models/PoissonMF.jl")
include("../../../models/other_models/NegBinMF.jl")
include("../../../revison_experiments/mcmc_stuff.jl")
using JLD
using CSV, DataFrames
using Combinatorics
using Random
using Statistics
using Printf

const K = 3

function run_one(dataset::String, mname::String, seed::Int;
                 NSAMPLES::Int=5000, NBURNIN::Int=5000, NTHIN::Int=1, NCHAINS::Int=4)


    datadir = joinpath(@__DIR__, "../../../revison_experiments/factorexperiment/data")
    datafiles = Dict("CMP" => "CMPfactor.csv", "GC" => "GCfactor.csv",
                     "Poisson" => "Poisfactor.csv", "NB" => "NBfactor.csv")
    truthfiles = Dict("CMP" => "CMPU_NK2.csv", "GC" => "GCU_NK.csv",
                      "Poisson" => "PoisU_NK.csv", "NB" => "NBU_NK.csv")

    mat = Matrix(CSV.read(joinpath(datadir, datafiles[dataset]), DataFrame))[1:end, 2:end]
    N, M = size(mat)
    data = Dict("Y_NM" => mat)

    # model registry: (constructor, fit kwargs, mean functional)
    uv(st) = vec(st["U_NK"] * st["V_KM"])
    uvp(st) = vec(st["U_NK"] * st["V_KM"]) .* (1 - st["p"]) ./ st["p"]  # NB identified mean
    registry = Dict(
        "poisson"      => (() -> PoissonMF(N, M, K, 1.0, 1.0, 1.0, 0.01), Dict(), uv),
        "medpois"      => (() -> OrderStatisticPoissonMF(N, M, K, 1.0, 1.0, 1.0, 0.01, 15, 2, 0.5),
                           Dict(), uv),
        "nb"           => (() -> NegBinMF(N, M, K, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0), Dict(), uvp),
        "mednb"        => (() -> OrderStatisticNegBinMF(N, M, K, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 3, 2),
                           Dict(), uvp),
        "star_id"      => (() -> STARMF(N, M, K, 1.0, 1.0, identity, identity), Dict(), uv),
        "star_sqrt"    => (() -> STARMF(N, M, K, 1.0, 1.0, sqrt, x -> x^2), Dict(), uv),
        "starnnf_id"   => (() -> STARMFNNF(N, M, K, 1.0, 1.0, 0.01, 1.0, 1.0, identity, identity), Dict(), uv),
        "starnnf_sqrt" => (() -> STARMFNNF(N, M, K, 1.0, 1.0, 0.01, 1.0, 1.0, sqrt, x -> x^2), Dict(), uv),
    )
    makemodel, opts, meanfun = registry[mname]

    Random.seed!(seed)
    mask = rand(N, M) .< 0.2
    heldout = [(i, j) for i in 1:N for j in 1:M if mask[i, j]]

    model = makemodel()
    samples_lst = []
    t = @elapsed for ch in 1:NCHAINS
        s = fit(model, copy(data); nsamples=NSAMPLES, nburnin=NBURNIN, nthin=NTHIN,
                mask=mask, initseed=seed * 1000 + ch, verbose=false, opts...)
        push!(samples_lst, s)
    end

    chain_mu = zeros(NSAMPLES, N * M, NCHAINS)
    chain_ll = zeros(NSAMPLES, 1, NCHAINS)
    for ch in 1:NCHAINS, s in 1:NSAMPLES
        st = samples_lst[ch][s]
        chain_mu[s, :, ch] = meanfun(st)
        chain_ll[s, 1, ch] = mean(evalulateLogLikelihood(model, st, data, nothing, i, j)
                                  for (i, j) in heldout)
    end
    essmu = ess_multichain(chain_mu)
    rhmu = rhat_split_multivariate(chain_mu)
    essll = ess_multichain(chain_ll)[1]
    rhll = rhat_split_multivariate(chain_ll)[1]
    ir = evaluateInfoRate(model, data, vcat(samples_lst...), mask=mask, verbose=false)

    # cosine similarity to true factors (permutation-aligned; K=3 brute force)
    cosine = NaN
    if haskey(truthfiles, dataset)
        Utr_raw = Matrix(CSV.read(joinpath(datadir, truthfiles[dataset]), DataFrame))[1:end, 2:end]
        Utrue = size(Utr_raw, 1) == N ? Float64.(Utr_raw) : Float64.(permutedims(Utr_raw))
        best_perm(Ur, Ut) = argmax_perm = begin
            S = abs.(cor(Ur, Ut)); best = -Inf; bp = collect(1:K)
            for p in permutations(1:K)
                sc = sum(S[k, p[k]] for k in 1:K)
                if sc > best; best = sc; bp = collect(p) end
            end
            bp
        end
        U_all = zeros(NCHAINS * NSAMPLES, N, K)
        for ch in 1:NCHAINS
            refc = samples_lst[ch][1]["U_NK"]
            for s in 1:NSAMPLES
                U = samples_lst[ch][s]["U_NK"]
                U_all[(ch-1)*NSAMPLES+s, :, :] = U[:, best_perm(refc, U)]
            end
        end
        ref = U_all[1, :, :]
        for ch in 2:NCHAINS
            lo = (ch - 1) * NSAMPLES + 1
            p = best_perm(ref, dropdims(mean(U_all[lo:(ch*NSAMPLES), :, :], dims=1), dims=1))
            for s in lo:(ch*NSAMPLES)
                U_all[s, :, :] = U_all[s, :, :][:, p]
            end
        end
        Um = dropdims(mean(U_all, dims=1), dims=1)
        Um = Um[:, best_perm(Utrue, Um)]
        cosine = mean(abs(dot(Utrue[:, k], Um[:, k])) / (norm(Utrue[:, k]) * norm(Um[:, k])) for k in 1:K)
    end

    # save posterior samples (Float32, chain-major: global index (ch-1)*NSAMPLES+s)
    # so metrics can be recomputed later without refitting. ~2-4 MB/cell.
    sampdir = joinpath(@__DIR__, "results", "samples")
    mkpath(sampdir)
    U_save = zeros(Float32, NCHAINS * NSAMPLES, N, K)
    V_save = zeros(Float32, NCHAINS * NSAMPLES, K, M)
    scalars = Dict{String,Vector{Float64}}()
    for key in ["sigma2", "p", "D"]
        if haskey(samples_lst[1][1], key)
            scalars[key] = [samples_lst[ch][s][key] for ch in 1:NCHAINS for s in 1:NSAMPLES]
        end
    end
    for ch in 1:NCHAINS, s in 1:NSAMPLES
        U_save[(ch-1)*NSAMPLES+s, :, :] = samples_lst[ch][s]["U_NK"]
        V_save[(ch-1)*NSAMPLES+s, :, :] = samples_lst[ch][s]["V_KM"]
    end
    sampdict = Dict{String,Any}("U" => U_save, "V" => V_save,
                                "llik_chain" => chain_ll, "mask" => collect(mask),
                                "nchains" => NCHAINS, "nsamples" => NSAMPLES,
                                "nthreads" => Threads.nthreads())
    for (k, v) in scalars
        sampdict["scalar_" * k] = v
    end
    JLD.save(joinpath(sampdir, "$(dataset)_$(mname)_$(seed)_samples.jld"), sampdict, compress=true)

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    open(joinpath(outdir, "$(dataset)_$(mname)_$(seed).csv"), "w") do io
        println(io, "dataset,model,seed,inforate,essmu_med,essmu_min,rhatmu_med,rhatmu_max,essll,rhatll,cosine_to_truth,time_s,iters_per_chain")
        @printf(io, "%s,%s,%d,%.6f,%.1f,%.1f,%.4f,%.4f,%.1f,%.4f,%.5f,%.1f,%d\n",
                dataset, mname, seed, ir, median(essmu), minimum(essmu),
                median(rhmu), maximum(rhmu), essll, rhll, cosine, t,
                NBURNIN + NTHIN * NSAMPLES)
    end
    @printf("done %s %s seed %d: ir %.4f essmu %.0f essll %.0f cos %.4f (%.0fs)\n",
            dataset, mname, seed, ir, median(essmu), essll, cosine, t)
    return ir

end
