println("opened code")
using CSV
using DataFrames
using Random
using JLD


include("/home/jlederman/DiscreteOrderStatistics/models/genes_polya/genes.jl")
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

file_path = "/home/jlederman/DiscreteOrderStatistics/data/cancer.csv"
df = CSV.read(file_path, DataFrame)
df = select(df, Not(1))
Y_NM = transpose(Matrix(df));
N = size(Y_NM, 1)
M = size(Y_NM, 2)

seed = 101
Random.seed!(seed)
nlil = 1000
random_indices = randperm(N)[1:nlil]  # Generate random indices
Y_NMsmall = Y_NM[random_indices,:]
N = nlil

Random.seed!(maskSeed)
mask_NM = rand(N, M) .< .025

Ysparse, _ = sparsify_format_mask(Y_NMsmall, mask_NM)

data = Dict("Y_NM"=>Y_NMsmall, "Ysparse"=>Ysparse)





# initasNMF = true
# if initasNMF
#     Random.seed!(chainSeed)
#     r = nnmf(Float64.(Y_NMsmall), K)
#     eps = 0.01
#     W = r.W  # Assuming W is a matrix
#     W .= ifelse.(W .<= eps, eps, W)
#     H = r.H
#     H .= ifelse.(H .<= eps, eps, H)
#     constantinit = Dict("U_NK"=>W,"V_KM"=>H)
# else
#     constantinit = nothing
# end

a = 1
b = 1
c = 1
d = 1
model = genes(N,M,K,Q,a,b,c,d,D)


@time samples = fit(model, data, nsamples = 100, nburnin=nburnin, nthin=10, initseed = chainSeed, mask=mask_NM)
inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=true)
results = [K,Q,D,maskSeed,chainSeed,nburnin,inforate]
println(inforate)
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/genes_polya/heldoutsamples/"
save(folder*"/sample_seed1_$(maskSeed)seed2_$(chainSeed)D$(D)K$(K)Q$(Q)Burnin$(nburnin).jld", "results", results, "samples", samples)
