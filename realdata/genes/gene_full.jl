using CSV
using DataFrames
using Random
using JLD

println(Threads.nthreads())
include("/home/jlederman/DiscreteOrderStatistics/models/genes_polya/genes.jl")

function sparsify_format_mask(data, mask)
    D = size(data, 1)
    V = size(data, 2)
    nztrain = findall(!iszero, data)
    test = findall(mask .== 1)
    nztrain = union(nztrain,test)
    Y = data[nztrain]
    nztrain_mat = zeros(Int, length(nztrain), 2)
    @views for i in eachindex(nztrain)
        ind = nztrain[i]
        nztrain_mat[i, 1] = ind[1]
        nztrain_mat[i, 2] = ind[2]
    end
    nztrain = nztrain_mat
    @assert size(nztrain, 2) == 2
    return nztrain, Y
end

#maskSeed = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[1])
D = parse(Int, ARGS[2])
K = parse(Int, ARGS[3])
Q = parse(Int, ARGS[4])
nburnin = parse(Int, ARGS[5])

file_path = "/home/jlederman/DiscreteOrderStatistics/data/cancer.csv"
df = CSV.read(file_path, DataFrame)
df = select(df, Not(1))
Y_NM = transpose(Matrix(df));
N = size(Y_NM, 1)
M = size(Y_NM, 2)

seed = 101
Random.seed!(seed)
nlil = 3000
random_indices = randperm(N)[1:nlil]  # Generate random indices
Y_NMsmall = Y_NM[random_indices,:]
N = nlil

#Random.seed!(maskSeed)
#mask_NM = rand(N, M) .< .025
mask_NM = zeros(N,M)

Ysparse, _ = sparsify_format_mask(Y_NMsmall, mask_NM)

data = Dict("Y_NM"=>Y_NMsmall, "Ysparse"=>Ysparse)

a = 1
b = 1
c = 1
d = 1
model = genes(N,M,K,Q,a,b,c,d,D)


@time samples = fit(model, data, nsamples = 1, nburnin=nburnin, nthin=1, initseed = chainSeed)
#inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=true)
results = [K,Q,D,chainSeed,nburnin]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/genes_polya/fullsamples/"
save(folder*"/sample_seed2_$(chainSeed)D$(D)K$(K)Q$(Q)Burnin$(nburnin).jld", "results", results, "samples", samples)
