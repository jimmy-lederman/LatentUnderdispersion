using CSV
using DataFrames
using Random
using JLD

println(Threads.nthreads())
include("/home/jlederman/DiscreteOrderStatistics/models/genes_polyaD/genes3.jl")

chainSeed = parse(Int, ARGS[1])
K = parse(Int, ARGS[2])
Q = parse(Int, ARGS[3])

file_path = "/home/jlederman/DiscreteOrderStatistics/data/cancer_small2.csv"
df = CSV.read(file_path, DataFrame)

Y_NM = Matrix(df)
N = size(Y_NM, 1)
M = size(Y_NM, 2)


data = Dict("Y_NM"=>Y_NM)

a = 1
b = 1
c = 1
d = 1
alpha0 = beta0 = 1
alpha = beta = 1
constant = -50
sigma2 = 10
Dmax = 9

model = genes(N,M,K,Q,Dmax,a,b,c,d,alpha0,beta0,alpha,beta,constant,sigma2)


@time samples = fit(model, data, nsamples =500, nburnin=4000, nthin=20, initseed = chainSeed,
skipupdate=["D_NM"],constantinit=Dict("D_NM"=>ones(Int, N, M)))
#inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=true)
params = [chainSeed,K,Q]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/genes_polya/fullsamples_subset/"
save(folder*"/sample_seed2_$(chainSeed)K$(K)Q$(Q).jld", "params", params, "samples", samples)
