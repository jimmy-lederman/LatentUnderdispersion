println(Threads.nthreads())

using CSV
using DataFrames
using Random
using JLD
using Base.Filesystem

K = parse(Int, ARGS[1])
# P = parse(Int, ARGS[3])
chainSeed = parse(Int, ARGS[2])

using CSV, DataFrames

data = Dict("Y_NM" => Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/birds/birds_dunson.csv", DataFrame))[:,2:end]);
info = Dict("X_NP" => Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/birds/birds_dunsonX.csv", DataFrame))[:,2:end]);
P = size(info["X_NP"])[2]
N, M = size(data["Y_NM"])


a = 1
b = 1
c = 1
d = 1
Dmax = 5

include("/home/jlederman/DiscreteOrderStatistics/models/birds/birds.jl")
model = birdsCov(N, M, K, Dmax, a, b, c, d)
@time samples = fit(model, data, nsamples = 500, nburnin=4000, nthin=20, initseed=chainSeed,
skipupdate=["D_NM"], constantinit=Dict("D_NM"=>ones(Int, model.N, model.M)))

params = [K,chainSeed]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/birds/"
save(folder*"fullsamplesCov/sampleD$(D)seedChain$(chainSeed).jld", "params", params, "samples", samples)
