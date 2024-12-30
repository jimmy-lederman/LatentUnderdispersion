using Dates
using CSV
using DataFrames
using Random
using JLD
println(Threads.nthreads())
include("/home/jlederman/DiscreteOrderStatistics/models/genes/gene.jl")

#cumdf = Matrix(CSV.read("/Users/jimmy/Desktop/OrderStats/data/CTFL.csv",DataFrame))
#cumdf = Matrix(CSV.read("../data/CTFL.csv",DataFrame))
cumdf = Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/hammer.csv",DataFrame))
df = CSV.read(file_path, DataFrame)
df = select(df, Not(1))
Y_NM = Matrix(df);
N = size(Y_NM, 1)
M = size(Y_NM, 2)
nlil = 2000
random_indices = randperm(N)[1:nlil]  # Generate random indices
Y_NM = Y_NM[random_indices,:]
data = Dict("Y_NM"=>Y_NM)
N = nlil;
data = Dict("Y_NM"=>cumdf)

seed1 = parse(Int, ARGS[1])
seed2 = parse(Int, ARGS[2])
D = parse(Int, ARGS[3])
j = parse(Int, ARGS[4])
K = parse(Int, ARGS[5])
type = parse(Int, ARGS[6])
nburnin = parse(Int, ARGS[7])

Random.seed!(seed1)
mask_NM = rand(N, M) .< .2


a = 1
b = 1
c = 1
d = 1
alpha = .1
beta = .1
if type == 1
    constantinit = nothing
    dist = x -> Poisson(x)
else
    constantinit = Dict("p_N"=>fill(.5,N))
    dist = (x,y) -> NegativeBinomial(x,y)
end
model = genes(N,M,K,a,b,c,d,alpha,beta,D,j,dist)


@time samples = fit(model, data, initseed=seed2, nsamples = 100, nburnin=nburnin, nthin=1, mask=mask_NM)
inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=false)
results = [seed1,seed2,D,j,K,type,nburnin,inforate]
println(inforate)

# samples = [Dict("eps"=>sample["eps"], "alpha"=>sample["alpha"], "V_KM"=>sample["V_KM"], "U_NK"=>sample["U_NK"], "R_KTS"=>sample["R_KTS"]) for sample in samples]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/genes/heldoutsamples/"
save(folder*"/sample_seed$(seed)D$(D)j$(j)K$(K)Type$(type)Burnin$(nburnin).jld", "results", results, "samples", samples)
