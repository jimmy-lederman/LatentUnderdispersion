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
flush(stdout)

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
info = Dict("Y0_N"=>Y_N1,"pop_N"=>pops)

maskSeed = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[2])
D = parse(Int, ARGS[3])
Q = parse(Int, ARGS[4])
K = parse(Int, ARGS[5])

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


# cumdf = Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/CTFL.csv",DataFrame))
# a = .1
# b = 1
# c = 100
# d = .01
# starta = .01
# startb = 1
# e = 1000
# f = 1000
# g = .1
# h = 1
# scale_shape = .5
# scale_rate = 1
a = 1
b = 1
c = 100
d = .1
starta = .01
startb = 1
g = .5
h = 1
scale_shape = 2
scale_rate = 1
Dmax = 9
alpha = 1
beta = 1
if D == 0
    include("/home/jlederman/DiscreteOrderStatistics/models/covid/covidsimpleD.jl")
    model = covidsimple(N,M,K,Q,Dmax,a,b,c,d,g,h,scale_shape,scale_rate,starta,startb,alpha,beta)
    @time samples = fit(model, data, initseed=chainSeed, nsamples = 100, nburnin=4000, nthin=20,
     mask=mask_NM,info=info,constantinit=Dict("V_KM"=>fill(1.0, K, M),"D_NM"=>ones(Int, N, M)), skipupdate=["D_NM"])
else
    include("/home/jlederman/DiscreteOrderStatistics/models/covid/covidsimple.jl")
    j = div(D,2)+1
    model = covidsimplebase(N,M,K,a,b,c,d,g,h,scale_shape,scale_rate,starta,startb,D,j)
    @time samples = fit(model, data, initseed=chainSeed, nsamples = 100, nburnin=4000, nthin=20,
     mask=mask_NM,info=info,constantinit=Dict("V_KM"=>fill(1.0, K, M),))
end
    


#inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=false,info=info)
params = [K,Q,D,maskSeed,chainSeed]

# samples = [Dict("eps"=>sample["eps"], "alpha"=>sample["alpha"], "V_KM"=>sample["V_KM"], "U_NK"=>sample["U_NK"]) for sample in samples]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/covid/medians/heldout_samplesD/"
save(folder*"/sample_maskSeed$(maskSeed)chainSeed$(chainSeed)D$(D)K$(K)Q$(Q).jld", "params", params, "samples", samples, "mask", mask_NM)
