println("opened file")
flush(stdout)
# using Pkg
# Pkg.activate(".")
using Dates
using CSV
using DataFrames
using Random
using JLD
println(Threads.nthreads())
println("imported packages")
#include("/home/jlederman/DiscreteOrderStatistics/revison_experiments/mcmc_stuff.jl");
flush(stdout)

#get data
cumdf = Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/covid_data/usafull_final.csv",DataFrame))
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
type1 = parse(Int, ARGS[3])
type2 = parse(Int, ARGS[4])
K = parse(Int, ARGS[5])
D = parse(Int, ARGS[6])
Q = parse(Int, ARGS[7])

#make mask (forecast)
endlength = 10
county_pct = .2
function make_forecasting_mask(endlength,county_pct,seed,N,M)
    Random.seed!(seed)
    missing_counties = rand(N) .< county_pct

    mask_NM = zeros(N, M)
    # for i in 0:(middlelength-1)
    #     mask_NM[:,middlestart + i] = missing_counties
    # end
    for i in (M-endlength+1):M
        mask_NM[:,i] = missing_counties
    end
    return BitArray(mask_NM .!= 0)
end
mask_NM = make_forecasting_mask(endlength,county_pct,maskSeed,N,M)

a = 1
b = 1
c = 1
d = .01
g = .5
h = 2
v1 = 1
v2 = 1

Nsamples = 500
nburnin = 2000
nthin = 1
constantinit = nothing

if type1 == 1
    if type2 == 1
        include("/home/jlederman/DiscreteOrderStatistics/models/covid_final/ablation/dirichlet/covid1.jl")
        model = covid1(N,M,K,a,c,d)

    elseif type2 == 2
        include("/home/jlederman/DiscreteOrderStatistics/models/covid_final/ablation/dirichlet/covid2.jl")
        model = covid2(N,M,K,a,c,d,g,h,v1,v2)

    elseif type2 == 3
        include("/home/jlederman/DiscreteOrderStatistics/models/covid_final/ablation/dirichlet/covid3.jl")
        model = covid3(N,M,K,D,a,c,d,g,h,v1,v2)
    end
elseif type1 == 2
    if type2 == 1
        include("/home/jlederman/DiscreteOrderStatistics/models/covid_final/ablation/gamma/covid1.jl")
        model = covid1(N,M,K,a,b,c,d)
    elseif type2 == 2
        include("/home/jlederman/DiscreteOrderStatistics/models/covid_final/ablation/gamma/covid2.jl")
        model = covid2(N,M,K,a,b,c,d,g,h,v1,v2)
    elseif type2 == 3
        include("/home/jlederman/DiscreteOrderStatistics/models/covid_final/ablation/gamma/covid3.jl")
        model = covid3(N,M,K,D,a,b,c,d,g,h,v1,v2)
    end
end

time_result = @elapsed samples = fit(model, data, nsamples=Nsamples, nburnin=nburnin, nthin=nthin, mask=mask_NM, initseed=chainSeed,verbose=true,info=info)

params = [maskSeed, chainSeed, type1, type2, K, D, Q]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/covid/ablation/samples/"
if type1 == 1
    save(folder*"/covid$(type2)dirichlet_maskSeed$(maskSeed)chainSeed$(chainSeed)D$(D)K$(K)Q$(Q).jld", "params", params, "samples", samples, "mask", mask_NM)
elseif type2 == 2
    save(folder*"/covid$(type2)gamma_maskSeed$(maskSeed)chainSeed$(chainSeed)D$(D)K$(K)Q$(Q).jld", "params", params, "samples", samples, "mask", mask_NM, "time_result", time_result)
end

