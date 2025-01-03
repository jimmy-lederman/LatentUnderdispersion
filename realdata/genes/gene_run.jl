using Dates
using CSV
using DataFrames
using Random
using JLD
using NMF
println(Threads.nthreads())
include("/home/jlederman/DiscreteOrderStatistics/models/genes/genes.jl")

file_path = "/home/jlederman/DiscreteOrderStatistics/data/hammer.csv"
df = CSV.read(file_path, DataFrame)
df = select(df, Not(1))
Y_NM = Matrix(df);
N = size(Y_NM, 1)
M = size(Y_NM, 2)
#full experiment
# nlil = 2000
# random_indices = randperm(N)[1:nlil]  # Generate random indices
# Y_NM = Y_NM[random_indices,:]
data = Dict("Y_NM"=>Y_NM)
# N = nlil;

maskSeed = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[2])
D = parse(Int, ARGS[3])
j = parse(Int, ARGS[4])
K = parse(Int, ARGS[5])
type = parse(Int, ARGS[6])
nburnin = parse(Int, ARGS[7])

Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .2

initasNMF = true
if initasNMF
    Random.seed!(chainSeed)
    r = nnmf(Float64.(Y_NM), K)
    eps = 0.01
    W = r.W  # Assuming W is a matrix
    W .= ifelse.(W .<= eps, eps, W)
    H = r.H
    H .= ifelse.(H .<= eps, eps, H)
    constantinit = Dict("U_NK"=>W,"V_KM"=>H)
else
    constantinit = nothing
end

a = 1
b = 1
c = 1
d = 1
alpha = 1
beta = 1
if type == 1
    #constantinit = nothing
    dist = x -> Poisson(x)
    @assert
else
    if isnothing(constantinit)
        constantinit = Dict("p_N"=>fill(.5,N))
    else
        constantinit["p_N"] = fill(.5,N)
    end
    dist = (x,y) -> NegativeBinomial(x,y)
end
model = genes(N,M,K,a,b,c,d,alpha,beta,D,j,dist)


@time samples = fit(model, data, initseed=seed2, nsamples = 100, nburnin=nburnin, nthin=10, initseed = chainSeed, mask=mask_NM,constantinit=constantinit)
inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=false)
results = [K,D,maskSeed,chainSeed,j,type,nburnin,inforate]
println(inforate)
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/genes/heldoutsamples/"
save(folder*"/sample_seed1_$(seed1)seed2_$(seed2)D$(D)j$(j)K$(K)Type$(type)Burnin$(nburnin).jld", "results", results, "samples", samples)
