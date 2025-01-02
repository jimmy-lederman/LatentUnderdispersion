include("/home/jlederman/DiscreteOrderStatistics/models/flights/flights.jl")
println(Threads.nthreads())

using CSV
using DataFrames
using Random
using JLD2
using Base.Filesystem

#datafile = "/net/projects/schein-lab/jimmy/PoissonMax/flights/airtime.csv"
datafile = "/home/jlederman/DiscreteOrderStatistics/data/airtime.csv"


K = parse(Int, ARGS[1])
D = parse(Int, ARGS[2])
maskSeed = parse(Int, ARGS[3])
chainSeed = parse(Int, ARGS[4])
T = 99 #because running on sub
#thread = Bool(parse(Int, ARGS[1]))


df = CSV.read(datafile, DataFrame)
df = select(df, Not(1))
Y_NM = convert(Matrix{Int}, Matrix(select(df, 1)))
dist_NM = convert(Matrix{Int}, Matrix(select(df, 2)))
I_NM = convert(Matrix{Int}, Matrix(select(df, 3:4)))
data = Dict("Y_NM"=>Y_NM)
info = Dict("I_NM"=>I_NM, "dist_NM"=>dist_NM)
#newdirectory = "samples/samplesSeed$(maskSeed)K$(K)D$(D)"
#mkdir(newdirectory)
#outdir=  "/net/projects/schein-lab/jimmy/PoissonMax/realdata/trans/"*newdirectory

N = size(Y_NM)[1]
M = size(Y_NM)[2]
a =20
b =100
c =10
d=1
alpha = beta = 1 #not used
j = D #using max model
dist = x -> Poisson(x)

model = flights(N, M, T, K, a, b, c, d, alpha, beta, D, j, dist)

# Random.seed!(maskSeed)
# mask_NM = rand(N, M) .< .2

@time samples = fit(model, data, nsamples = 100, nburnin=10000, nthin=20, info=info,initseed=chainSeed)
# inforate = evaluateInfoRate(model,data,samples, info=info, verbose=false)
# results = [K,D,maskSeed,inforate]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/flights/"
save(folder*"fullsamples/sampleK$(K)D$(D)seedMask$(maskSeed)seedChain$(chainSeed).jld", "samples", samples)
