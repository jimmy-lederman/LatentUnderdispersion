println("opended file")
flush(stdout)
# using Pkg
# Pkg.instantiate()
# Pkg.status()
# using Pkg
# Pkg.precompile()
include("/home/jlederman/DiscreteOrderStatistics/models/covid/covidsimple.jl")
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
info = Dict("Y0_N"=>Y_N1,"pop_N"=>pops,"days_M"=>Vector(days), "state_N"=>state)

seed = parse(Int, ARGS[1])
D = parse(Int, ARGS[2])
Q = parse(Int, ARGS[3])
K = parse(Int, ARGS[4])

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
    info=info,constantinit=Dict("V_KM"=>fill(1.0, K, M),"D_NM"=>ones(Int, N, M)), skipupdate=["D_NM"])
else
    include("/home/jlederman/DiscreteOrderStatistics/models/covid/covidsimple.jl")
    j = div(D,2)+1
    model = covidsimple(N,M,K,a,b,c,d,g,h,scale_shape,scale_rate,starta,startb,D,j)
    @time samples = fit(model, data, initseed=chainSeed, nsamples = 100, nburnin=4000, nthin=20,
    info=info,constantinit=Dict("V_KM"=>fill(1.0, K, M),))
end


# @time samples = fit(model, data, initseed=seed, nsamples = 1, nburnin=4000, nthin=1,info=info,constantinit=Dict("V_KM"=>fill(1.0, K, M)))
# inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=false,info=info)
params = [K,Q,D,seed]

# samples = [Dict("eps"=>sample["eps"], "alpha"=>sample["alpha"], "V_KM"=>sample["V_KM"], "U_NK"=>sample["U_NK"]) for sample in samples]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/covid/medians/full_samplesD/"
save(folder*"/sample_seed$(seed)D$(D)K$(K)Q$(Q).jld", "params", params, "samples", samples)
