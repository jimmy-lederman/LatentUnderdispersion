using CSV
using DataFrames
using Random
using JLD

println(Threads.nthreads())

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/genes/genes_final.jl"))

chainSeed = parse(Int, ARGS[1])
K = parse(Int, ARGS[2])
Q = parse(Int, ARGS[3])

file_path = joinpath(ROOTDIR, "data/genes/cancer_small_final.csv")
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
params = [chainSeed,K,Q]
folder = joinpath(ROOTDIR, "output/genes/fullsamples/")
mkpath(folder)
save(folder*"/sample_seed2_$(chainSeed)K$(K)Q$(Q).jld", "params", params, "samples", samples)
