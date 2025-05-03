println(Threads.nthreads())

using CSV
using DataFrames
using Random
using JLD
using Base.Filesystem

D = parse(Int, ARGS[1])
K = parse(Int, ARGS[2])
P = parse(Int, ARGS[3])
chainSeed = parse(Int, ARGS[4])

using CSV, DataFrames

data = Dict("Y_NM" => Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/birds/birds_dunson.csv", DataFrame))[:,2:end]);
N, M = size(data["Y_NM"])


a = 1
b = 1
c = 1
d = 1

if D == 0
    Dmax = 5
    include("/home/jlederman/DiscreteOrderStatistics/models/birds/birds_simple2.jl")
    model = birds(N, M, K, P, Dmax, a, b, c, d)
    @time samples = fit(model, data, nsamples = 500, nburnin=4000, nthin=20, initseed=chainSeed,
    skipupdate=["D_NM"], constantinit=Dict("D_NM"=>ones(Int, model.N, model.M)))
elseif D == 1
    include("/home/jlederman/DiscreteOrderStatistics/models/PoissonMF.jl")
    model = PoissonMF(N, M, K, a, b, c, d)
    @time samples = fit(model, data, nsamples = 500, nburnin=4000, nthin=20, initseed=chainSeed)
else #D > 1
    include("/home/jlederman/DiscreteOrderStatistics/models/MaxPoissonMF.jl")
    model = MaxPoissonMF(N, M, K, D, a, b, c, d)
    @time samples = fit(model, data, nsamples = 500, nburnin=4000, nthin=20, initseed=chainSeed)
end

params = [D,K,P,chainSeed]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/birds/"
save(folder*"fullsamplesDmore/sampleD$(D)K$(K)P$(P)seedChain$(chainSeed).jld", "params", params, "samples", samples)
