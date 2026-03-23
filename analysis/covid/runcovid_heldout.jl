println("opened file")
flush(stdout)
using Dates
using CSV
using DataFrames
using Random
using JLD
println(Threads.nthreads())
println("imported packages")
flush(stdout)

ROOTDIR = joinpath(@__DIR__, "../..")

#get data
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
info = Dict("Y0_N"=>Y_N1,"pop_N"=>pops)

#get parameters
maskSeed = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[2])
D = parse(Int, ARGS[3])
K = parse(Int, ARGS[4])
Q = parse(Int, ARGS[5])
type = parse(Int, ARGS[6])

#make mask (forecast)
endlength = 10
county_pct = .2
function make_forecasting_mask(endlength,county_pct,seed,N,M)
    Random.seed!(seed)
    missing_counties = rand(N) .< county_pct

    mask_NM = zeros(N, M)
    for i in (M-endlength+1):M
        mask_NM[:,i] = missing_counties
    end
    return BitArray(mask_NM .!= 0)
end
mask_NM = make_forecasting_mask(endlength,county_pct,maskSeed,N,M)

a = 1
b = .01

d = .01
g = 1
h = 1
v1 = 1
v2 = 1
alpha = 1
beta = 1
start_tau = 0
tauc = 1
taud = 0
Nsamples = 100
nburnin = 2000
nthin = 20
constantinit = nothing

if type == 1
    c = .01
    include(joinpath(ROOTDIR, "models/covid/ablation/covid1.jl"))
    model = covid1(N,M,K,a,b,c,d)
elseif type == 2
    c = 100
    include(joinpath(ROOTDIR, "models/covid/ablation/covid2.jl"))
    model = covid2(N,M,K,a,b,c,d,g,h,v1,v2)
elseif type == 3
    c = 100
    include(joinpath(ROOTDIR, "models/covid/ablation/covid3.jl"))
    model = covid3(N,M,K,D,a,b,c,d,g,h,v1,v2)
elseif type == 4
    c = 100
    include(joinpath(ROOTDIR, "models/covid/ablation/covid3.jl"))
    model = covid4(N,M,K,Q,D,a,b,c,d,g,h,v1,v2,alpha,beta,tauc,taud,start_tau)
end

time_result = @elapsed samples = fit(model, data, nsamples=Nsamples, nburnin=nburnin, nthin=nthin, mask=mask_NM, initseed=chainSeed,verbose=true,info=info)

params = [maskSeed, chainSeed, D, K, Q, type]
folder = joinpath(ROOTDIR, "output/covid/ablation/samples/")
mkpath(folder)
save(folder*"/covid$(type)gamma_maskSeed$(maskSeed)chainSeed$(chainSeed)D$(D)K$(K)Q$(Q).jld", "params", params, "samples", samples, "mask", mask_NM, "time_result", time_result)
