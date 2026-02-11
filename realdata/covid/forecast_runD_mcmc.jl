println("opended file")
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
D = parse(Int, ARGS[3])
Q = parse(Int, ARGS[4])
K = parse(Int, ARGS[5])
covidType = parse(Int, ARGS[6])

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
d = 1

Nsamples = 500
nburnin = 2000
nthin = 1


if covidType == 1
    if D == 0
        include("/home/jlederman/DiscreteOrderStatistics/models/covid_final/ablation/covid1Poisson.jl")
        model = covid1Poisson(N,M,K,a,b,c,d)
        @time samples = fit(model, data, nsamples=Nsamples, nburnin=nburnin, nthin=nthin, mask=mask_NM, initseed=chainSeed,verbose=true,info=info)
    else
        include("/home/jlederman/DiscreteOrderStatistics/models/covid_final/ablation/covid1.jl")
        model = covid1(N,M,K,D,a,b,c,d)
        @time samples = fit(model, data, nsamples=Nsamples, nburnin=nburnin, nthin=nthin, mask=mask_NM,
         initseed=chainSeed,verbose=true,info=info,constantinit=Dict("D"=>D))
    end

    # elseif MFtype == 2
    #     include("/home/jlederman/DiscreteOrderStatistics/models/other_models/PoissonMF2.jl")
    #     model = PoissonMF2(N,M,K,a,b,c,d)
    #     @time samples = fit(model, data, nsamples=Nsamples, nburnin=nburnin, nthin=nthin, mask=mask_NM, initseed=chainSeed,verbose=true,info=info)
    # elseif MFtype == 3
    #     include("/home/jlederman/DiscreteOrderStatistics/models/other_models/PoissonMF3.jl")
    #     model = PoissonMF3(N,M,K,a,b,c,d)
    #     @time samples = fit(model, data, nsamples=Nsamples, nburnin=nburnin, nthin=nthin, mask=mask_NM, initseed=chainSeed,verbose=true,info=info)
    # end
else
    @assert 1 == 2
end


#inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=false,info=info)
params = [K,Q,D,maskSeed,chainSeed,covidType]

# samples = [Dict("eps"=>sample["eps"], "alpha"=>sample["alpha"], "V_KM"=>sample["V_KM"], "U_NK"=>sample["U_NK"]) for sample in samples]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/covid/ablation/samples/"
save(folder*"/covidType$(covidType)sample_maskSeed$(maskSeed)chainSeed$(chainSeed)D$(D)K$(K)Q$(Q).jld", "params", params, "samples", samples, "mask", mask_NM)
