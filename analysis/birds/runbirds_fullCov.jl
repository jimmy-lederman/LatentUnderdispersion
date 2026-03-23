println(Threads.nthreads())

using CSV
using DataFrames
using Random
using JLD
using Base.Filesystem

chainSeed = parse(Int, ARGS[1])
D = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])

using CSV, DataFrames

ROOTDIR = joinpath(@__DIR__, "../..")
data = Dict("Y_NM" => Matrix(CSV.read(joinpath(ROOTDIR, "data/birds/birds_final_counts.csv"), DataFrame))[:,2:end]);
info = Dict("X_NP" => Matrix(CSV.read(joinpath(ROOTDIR, "data/birds/birds_final_covariates.csv"), DataFrame))[:,2:end]);
Q = size(info["X_NP"])[2]
N, M = size(data["Y_NM"])


a = 1
b = 1
c = 1
d = 1
include(joinpath(ROOTDIR, "models/birds/birds_final_covariate.jl"))
model = birdsCov(N, M, K, D, a, b, c, d)
@time samples = fit(model, data, nsamples = 500, nburnin=4000, nthin=20, initseed=chainSeed,info=info,
skipupdate=["D_NM"], constantinit=Dict("D_NM"=>ones(Int, model.N, model.M)))

params = [chainSeed,D,K]
folder = joinpath(ROOTDIR, "output/birds/")
mkpath(folder)
save(folder*"fullsamplesCov/sampleD$(D)K$(K)seedChain$(chainSeed).jld", "params", params, "samples", samples)
