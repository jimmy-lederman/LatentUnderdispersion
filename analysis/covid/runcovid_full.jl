println("opened file")
flush(stdout)

ROOTDIR = joinpath(@__DIR__, "../..")
include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
using Dates
using CSV
using DataFrames
using Random
using JLD
println(Threads.nthreads())
println("imported packages")
flush(stdout)

cumdf = Matrix(CSV.read(joinpath(ROOTDIR, "data/covid/usafull_final.csv"),DataFrame))
days = cumdf[1,4:end]
state = cumdf[2:end,1]
fips = cumdf[2:end,2]
pops = log.(cumdf[2:end,3])
cumdf = cumdf[2:end,4:end]
data = Dict("Y_NM"=>cumdf)
N = size(cumdf)[1]
M = size(cumdf)[2]
Y_N1 = Int.(reshape(cumdf[:,1], :, 1))
info = Dict("Y0_N"=>Y_N1,"pop_N"=>pops,"days_M"=>Vector(days), "state_N"=>state)

seed = parse(Int, ARGS[1])
D = parse(Int, ARGS[2])
Q = parse(Int, ARGS[3])
K = parse(Int, ARGS[4])

a = 1
b = 1
c = 100
d = .1
g = .5
h = 1
scale_shape = 2
scale_rate = 1
Dmax = 9
alpha = 1
beta = 1
start_tau = 0
start_V1 = .01
start_V2 = 1
tauc = 1
taud = 0

nsamples = 1
nburnin = 2000
nthin = 1

include(joinpath(ROOTDIR, "models/covid/covid_final.jl"))
model = covid4(N,M,K,Q,Dmax,a,b,c,d,g,h,start_V1,start_V2,alpha,beta,tauc,taud,start_tau)
@time samples = fit(model, data, initseed=seed, nsamples = nsamples, nburnin=nburnin, nthin=nthin,
    info=info,constantinit=Dict("V_KM"=>fill(1.0, K, M),"D_NM"=>ones(Int, N, M)), skipupdate=["D_NM"])

params = [K,Q,D,seed]

folder = joinpath(ROOTDIR, "output/covid/full_samples/")
mkpath(folder)
save(folder*"/fullsample_seed$(seed)D$(D)K$(K)Q$(Q).jld", "params", params, "samples", samples)
