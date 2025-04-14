println(Threads.nthreads())

using CSV
using DataFrames
using Random
using JLD
using Base.Filesystem

D = parse(Int, ARGS[1])
K = parse(Int, ARGS[2])
P = parse(Int, ARGS[3])
maskSeed = parse(Int, ARGS[4])
chainSeed = parse(Int, ARGS[5])

using CSV, DataFrames

data = Dict("Y_NM" => Matrix(CSV.read("../../data/birds/birds_dunson.csv", DataFrame))[:,2:end]);
N, M = size(data["Y_NM"])


using Random
Random.seed!(1)
mask = rand(N,M) .< .05;

a = 1
b = 1
c = 1
d = 1

Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .05

if D == 0
    Dmax = 5
    include("/home/jlederman/DiscreteOrderStatistics/models/flights/birds_simple.jl")
    model = birds_simple(N, M, K, P, Dmax, a, b, c, d)
    @time samples = fit(model, data, nsamples = 100, nburnin=4000, nthin=10, mask=mask_NM, info=info,initseed=chainSeed)
elseif D == 1
    include("/home/jlederman/DiscreteOrderStatistics/models/PoissonMF.jl")
    model = Birds_simple_base(N, M, K, a, b, c, d)
    @time samples = fit(model, data, nsamples = 100, nburnin=4000, nthin=10, mask=mask_NM, info=info,initseed=chainSeed)
else #D > 1
    include("/home/jlederman/DiscreteOrderStatistics/models/MaxPoissonMF.jl")
    model = Birds_simple_base(N, M, K, D, a, b, c, d)
    @time samples = fit(model, data, nsamples = 100, nburnin=4000, nthin=10, mask=mask_NM, info=info,initseed=chainSeed)
end

params = [D,K,p,maskSeed,chainSeed]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/birds/"
save(folder*"heldoutsamplesD/sampleD$(D)seedMask$(maskSeed)seedChain$(chainSeed).jld", "params", params, "samples", samples)
