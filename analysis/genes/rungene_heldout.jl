println("opened code")
using CSV
using DataFrames
using Random
using JLD

ROOTDIR = joinpath(@__DIR__, "../..")

include(joinpath(ROOTDIR, "models/genes/genes_final.jl"))
println("imported packages")
println(Threads.nthreads())
flush(stdout)
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

maskSeed = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[2])
D = parse(Int, ARGS[3])
K = parse(Int, ARGS[4])
Q = parse(Int, ARGS[5])
nburnin = parse(Int, ARGS[6])
j = parse(Int, ARGS[7])

file_path = joinpath(ROOTDIR, "data/genes/cancer_small_final.csv")
df = CSV.read(file_path, DataFrame)
Y_NM = Matrix(df)
N = size(Y_NM, 1)
M = size(Y_NM, 2)

Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .001

Ysparse, _ = sparsify_format_mask(Y_NM, mask_NM)

data = Dict("Y_NM"=>Y_NM, "Ysparse"=>Ysparse)


a = 1
b = 1
c = 1
d = 1
model = genes(N,M,K,Q,a,b,c,d,D,j)


@time samples = fit(model, data, nsamples = 100, nburnin=nburnin, nthin=10, initseed = chainSeed, mask=mask_NM)
results = [K,Q,D,maskSeed,chainSeed,nburnin,NaN]

folder = joinpath(ROOTDIR, "output/genes/heldoutsamples/")
mkpath(folder)
save(folder*"/sample_seed1_$(maskSeed)seed2_$(chainSeed)D$(D)j$(j)K$(K)Q$(Q)Burnin$(nburnin).jld", "results", results, "samples", samples)
