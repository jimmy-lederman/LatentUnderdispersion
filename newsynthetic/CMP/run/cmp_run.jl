#import packages
using JLD
using CSV
using DataFrames
using Base.Threads
println("ahhhhh")
println(Threads.nthreads())
println("AHHHHHHH")
flush(stdout)

#set arguments
datasetnum = parse(Int, ARGS[1]) #datset to do experiment on
dist = parse(Int, ARGS[2])
type = parse(Int, ARGS[3])
D = parse(Int, ARGS[4])
if length(ARGS) > 4
    griddy = parse(Int, ARGS[5])
    annealstrat = parse(Int, ARGS[6])
else
    griddy = false
    annealstrat = nothing
end
if griddy == 1
    griddy = true
else
    griddy = false
end
if annealstrat == 0
    annealstrat = nothing
end
println(griddy)
println(annealstrat)
#get data
data_all = CSV.read("/home/jlederman/DiscreteOrderStatistics/newsynthetic/CMP/data/CMPdata.csv", DataFrame)
data_all = select(data_all, Not(1))
data_all = Matrix(data_all)
Y_NM = Int.(reshape(data_all[3:1002,datasetnum], :,1))
data = Dict("Y_NM"=>Y_NM)
nu = data_all[1,datasetnum]
seed = Int(data_all[2,datasetnum])

#set folder
folder = "/net/projects/schein-lab/jimmy/OrderStats/synthetic/CMP/samples_burnin/"

N = 1000
M = 1  

if dist == 1 #Poisson case
    #hyperparameters
    a = .0001
    b = .0001 
    include("/home/jlederman/DiscreteOrderStatistics/models/OrderStatisticPoissonUnivariate.jl")
    if type == 1 #min
        model = OrderStatisticPoissonUnivariate(N, M, a, b, D, 1)
    elseif type == 2 #median 
        model = OrderStatisticPoissonUnivariate(N, M, a, b, D, div(D,2)+1)
    elseif type == 3 #max 
        model = OrderStatisticPoissonUnivariate(N, M, a, b, D, D)
    end
    samples = fit(model, data, nsamples = 1000, nburnin=10000, nthin=10, constantinit=Dict("mu"=>100))
elseif dist == 2
    #hyperparameters
    a = .0001
    b = .0001 
    alpha = 1
    beta = 1
    include("/home/jlederman/DiscreteOrderStatistics/models/OrderStatisticNegBinUnivariate.jl")
    if type == 1 #min
        model = OrderStatisticNegBinUnivariate(N, M, a, b, alpha, beta, D, 1)
        F = var(rand(OrderStatistic(Normal(0,1),D,1),1000))
    elseif type == 2 #median 
        model = OrderStatisticNegBinUnivariate(N, M, a, b, alpha, beta, D, div(D,2)+1)
        F = var(rand(OrderStatistic(Normal(0,1),D,div(D,2)+1),1000))
    elseif type == 3 #max 
        model = OrderStatisticNegBinUnivariate(N, M, a, b, alpha, beta, D, D)
        F = var(rand(OrderStatistic(Normal(0,1),D,D),1000))
    end
    samples = fit(model, data, nsamples = 1000, nburnin=10000, nthin=10, constantinit=Dict("mu"=>100,"p"=>1-F),annealStrat=annealstrat,griddy=griddy)
end

outfile = folder * "samples_Dist$(dist)Type$(type)D$(D)NU$(nu)Seed$(seed).jld"

save(outfile, "samples", samples)
