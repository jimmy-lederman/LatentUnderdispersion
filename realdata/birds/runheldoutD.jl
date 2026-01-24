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

data = Dict("Y_NM" => Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/birds/birds_dunson.csv", DataFrame))[:,2:end]);
N, M = size(data["Y_NM"])


# using Random
# Random.seed!(1)
# mask = rand(N,M) .< .05;

a = 1
b = 1
c = 1
d = 1

Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .05

if D == 0
    Dmax = 5
    include("/home/jlederman/DiscreteOrderStatistics/models/birds/birds_simple.jl")
    model = birds(N, M, K, P, Dmax, a, b, c, d)
    @time samples = fit(model, data, nsamples = 500, nburnin=4000, nthin=20, mask=mask_NM, initseed=chainSeed,
    skipupdate=["D_NM"], constantinit=Dict("D_NM"=>ones(Int, model.N, model.M)))
elseif D == 1
    include("/home/jlederman/DiscreteOrderStatistics/models/PoissonMF.jl")
    model = PoissonMF(N, M, K, a, b, c, d)
    @time samples = fit(model, data, nsamples = 500, nburnin=4000, nthin=20, mask=mask_NM, initseed=chainSeed)
else #D > 1
    include("/home/jlederman/DiscreteOrderStatistics/models/MaxPoissonMF.jl")
    model = MaxPoissonMF(N, M, K, D, a, b, c, d)
    @time samples = fit(model, data, nsamples = 500, nburnin=4000, nthin=20, mask=mask_NM, initseed=chainSeed)
end

params = [D,K,P,maskSeed,chainSeed]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/birds/"
save(folder*"heldoutsamplesDmore/sampleD$(D)K$(K)P$(P)seedMask$(maskSeed)seedChain$(chainSeed).jld", "params", params, "samples", samples, "mask", mask_NM)
