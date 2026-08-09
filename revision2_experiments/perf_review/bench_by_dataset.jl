# Per-iteration cost of all eight §6.3 models on ALL FOUR datasets, plus the
# structural reason the ranking moves with the data.
#
# Motivation: the earlier fair-timing benchmark used CMPfactor only. The models
# scale with different features of the data:
#   * PoissonMF / NegBinMF thin ONLY the nonzero cells (the `if Y > 0` guard),
#     so their cost tracks nnz, not N*M -- the sparsity advantage of Poisson
#     factorization.
#   * the STAR models draw one truncated normal for EVERY cell, zeros included,
#     so their cost is fixed at N*M regardless of sparsity.
#   * the order-statistic models augment D latent draws per cell.
# A single sparse dataset therefore flatters Poisson; a dense one narrows the
# gap. This reports all four so the paper's timing column can be honest about
# which regime it describes.
#
# Usage: julia --project=. revision2_experiments/perf_review/bench_by_dataset.jl [nreps] [niters]

include("../../models/STARmodels/revision2/STARMFNNfloor.jl")
include("../../models/STARmodels/STARMF.jl")
include("../../models/other_models/OrderStatisticPoissonMFprior.jl")
include("../../models/other_models/OrderStatisticNegBinMF.jl")
include("../../models/other_models/PoissonMF.jl")
include("../../models/other_models/NegBinMF.jl")
using CSV, DataFrames
using Random, Statistics, Printf

nreps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
niters = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 60

datadir = joinpath(@__DIR__, "../../revison_experiments/factorexperiment/data")
datasets = [("CMP", "CMPfactor.csv"), ("GC", "GCfactor.csv"),
            ("Poisson", "Poisfactor.csv"), ("NB", "NBfactor.csv")]
K = 3

println("=== dataset characteristics ===")
@printf("%-9s %8s %8s %10s %10s\n", "dataset", "mean", "max", "%% zeros", "nnz/N*M")
for (name, f) in datasets
    m = Matrix(CSV.read(joinpath(datadir, f), DataFrame))[1:end, 2:end]
    @printf("%-9s %8.2f %8d %9.0f%% %9.2f\n", name, mean(m), maximum(m),
            100 * mean(m .== 0), mean(m .> 0))
end

results = Dict{String,Dict{String,Float64}}()

for (dname, f) in datasets
    mat = Matrix(CSV.read(joinpath(datadir, f), DataFrame))[1:end, 2:end]
    N, M = size(mat)
    data = Dict("Y_NM" => mat)
    Random.seed!(1)
    mask = rand(N, M) .< 0.2

    builders = [
        ("poisson",      () -> PoissonMF(N, M, K, 1.0, 1.0, 1.0, 0.01), NamedTuple()),
        ("medpois",      () -> OrderStatisticPoissonMF(N, M, K, 1.0, 1.0, 1.0, 0.01, 15, 2, 0.5), NamedTuple()),
        ("nb",           () -> NegBinMF(N, M, K, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0), NamedTuple()),
        ("mednb",        () -> OrderStatisticNegBinMF(N, M, K, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 3, 2), NamedTuple()),
        ("star_id",      () -> STARMF(N, M, K, 1.0, 1.0, identity, identity), NamedTuple()),
        ("star_sqrt",    () -> STARMF(N, M, K, 1.0, 1.0, sqrt, x -> x^2), NamedTuple()),
        ("starnnf_id",   () -> STARMFNNF(N, M, K, 1.0, 1.0, 0.01, 1.0, 1.0, identity, identity), NamedTuple()),
        ("starnnf_sqrt", () -> STARMFNNF(N, M, K, 1.0, 1.0, 0.01, 1.0, 1.0, sqrt, x -> x^2), NamedTuple()),
    ]

    function time_iters(build, n)
        model = build()
        Random.seed!(12345)
        state = sample_prior(model)
        ~, state = backward_sample(model, data, state, mask)
        t = @elapsed for _ in 1:n
            ~, state = backward_sample(model, data, state, mask)
        end
        return t / n * 1000
    end

    for (_, b, _) in builders
        time_iters(b, 1)
    end

    times = Dict(nm => Float64[] for (nm, _, _) in builders)
    for rep in 1:nreps
        for i in randperm(length(builders))     # randomized order within replicate
            nm, b, _ = builders[i]
            push!(times[nm], time_iters(b, niters))
        end
    end
    results[dname] = Dict(nm => median(v) for (nm, v) in times)
    println("  $dname done"); flush(stdout)
end

println("\n=== ms/iteration by dataset (median of $nreps reps x $niters iters, randomized order) ===")
names = ["poisson", "nb", "starnnf_id", "starnnf_sqrt", "star_id", "star_sqrt", "medpois", "mednb"]
@printf("%-14s", "model")
for (d, _) in datasets
    @printf("%12s", d)
end
println()
for nm in names
    @printf("%-14s", nm)
    for (d, _) in datasets
        @printf("%12.3f", results[d][nm])
    end
    println()
end

println("\n=== relative to PoissonMF on the same dataset ===")
@printf("%-14s", "model")
for (d, _) in datasets
    @printf("%12s", d)
end
println()
for nm in names
    @printf("%-14s", nm)
    for (d, _) in datasets
        @printf("%12.2f", results[d][nm] / results[d]["poisson"])
    end
    println()
end
