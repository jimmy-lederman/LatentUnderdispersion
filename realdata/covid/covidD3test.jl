using Dates
using CSV
using DataFrames
using Random
println(Threads.nthreads())

#cumdf = Matrix(CSV.read("/Users/jimmy/Desktop/OrderStats/data/CTFL.csv",DataFrame))
#cumdf = Matrix(CSV.read("../data/CTFL.csv",DataFrame))
cumdf = Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/CTFL.csv",DataFrame))
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

Random.seed!(seed)
mask_NMminus1 = rand(N,M-1) .< .2
mask_NM = hcat(fill(false,N), mask_NMminus1);

include("../models/OrderStatisticPoissonTimeDayMF3.jl")
T = 7
S = 2
K = 2
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
    model = PoissonTimeDayMF(N,M,T,S,K,a,b,c,d,starta,startb,e,f,g,h,scale_shape,scale_rate)
else
    model = OrderStatisticPoissonTimeDayMF(N,M,T,S,K,a,b,c,d,starta,startb,e,f,g,h,scale_shape,scale_rate,D,j)
end

samples2 = fit(model, data, nsamples=1000, nthin=10, nburnin=10000, info=info,mask=mask_NM,verbose=true,initseed=2)#,constantinit=Dict("V_KM"=>fill(1.0, K, M), "R_KTS"=>fill(1.0,K,T,S)), skipupdate=["R_KTS"])


save(folder*"samples_seed$(seed)D$(D)j$j.jld", "samples", samples)
