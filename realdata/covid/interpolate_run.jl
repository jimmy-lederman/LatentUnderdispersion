using Dates
using CSV
using DataFrames
using Random
using JLD
println(Threads.nthreads())

#cumdf = Matrix(CSV.read("/Users/jimmy/Desktop/OrderStats/data/CTFL.csv",DataFrame))
#cumdf = Matrix(CSV.read("../data/CTFL.csv",DataFrame))
cumdf = Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/usafull.csv",DataFrame))
days = cumdf[1,4:end]
state = cumdf[2:end,1]
fips = cumdf[2:end,2]
pops = log.(cumdf[2:end,3])
cumdf = cumdf[2:end,4:end]
data = Dict("Y_NM"=>cumdf)
N = size(cumdf)[1]
M = size(cumdf)[2]
Y_N1 = Int.(reshape(cumdf[:,1], :, 1))
info = Dict("Y_N1"=>Y_N1,"pop_N"=>pops,"days_M"=>Vector(days), "state_N"=>state)

seed = parse(Int, ARGS[1])
D = parse(Int, ARGS[2])
j = parse(Int, ARGS[3])
K = parse(Int, ARGS[4])

Random.seed!(seed)
mask_NMminus1 = rand(N,M-1) .< .2
mask_NM = hcat(fill(false,N), mask_NMminus1);


# cumdf = Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/CTFL.csv",DataFrame))
T = 7
S = 49
a = .1
b = 1
c = 100
d = .01
starta = .01
startb = 1
e = 1000
f = 1000
g = .1
h = 1
scale_shape = .5
scale_rate = 1
# D = 3
# j = 2

if D == 1 && j == 1
    include("/home/jlederman/DiscreteOrderStatistics/models/PoissonTimeDayMF3.jl")
    model = PoissonTimeDayMF(N,M,T,S,K,a,b,c,d,starta,startb,e,f,g,h,scale_shape,scale_rate)
else
    include("/home/jlederman/DiscreteOrderStatistics/models/OrderStatisticPoissonTimeDayMF3.jl")
    model = OrderStatisticPoissonTimeDayMF(N,M,T,S,K,a,b,c,d,starta,startb,e,f,g,h,scale_shape,scale_rate,D,j)
end

@time samples = fit(model, data, initseed=seed, nsamples = 500, nburnin=4000, nthin=20, mask=mask_NM,info=info,constantinit=Dict("V_KM"=>fill(1.0, K, M), "R_KTS"=>fill(1.0, K,T,S)),skipupdate=["R_KTS"])
# inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=false,info=info)
results = [K,D,j,seed]

samples = [Dict("eps"=>sample["eps"], "alpha"=>sample["alpha"], "V_KM"=>sample["V_KM"], "U_NK"=>sample["U_NK"], "R_KTS"=>sample["R_KTS"]) for sample in samples]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/covid/medianD3/usa/interpolate_samples/"
save(folder*"/sample_seed$(seed)D$(D)j$(j)K$(K).jld", "results", results, "samples", samples)
