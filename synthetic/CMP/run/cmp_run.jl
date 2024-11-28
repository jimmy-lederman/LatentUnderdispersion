#import packages
using JLD
using CSV
using DataFrames
using Base.Threads
println(Threads.nthreads())

#set arguments
datasetnum = parse(Int, ARGS[1]) #datset to do experiment on
dist = parse(Int, ARGS[2])
type = parse(Int, ARGS[3])
D = parse(Int, ARGS[4])

#get data
data_all = CSV.read("/home/jlederman/DiscreteOrderStatistics/synthetic/CMP/data/CMP.csv", DataFrame)
data_all = select(data_all, Not(1))
data_all = Matrix(data_all)
Y_NM = Int.(reshape(data_all[3:1002,datasetnum], :,1))
data = Dict("Y_NM"=>Y_NM)
nu = data_all[1,datasetnum]
seed = Int(data_all[2,datasetnum])

#set folder
folder = "/net/projects/schein-lab/jimmy/OrderStats/synthetic/CMP/samples/"

N = 1000
M = 1  

if dist == 1 #Poisson case
    #hyperparameters
    a = .0001
    b = .0001 
    if D == 1
        include("/home/jlederman/DiscreteOrderStatistics/models/PoissonUnivariate.jl")
        model = PoissonUnivariate(N, M, a, b)
    else
        @assert D > 1
        include("/home/jlederman/DiscreteOrderStatistics/models/OrderStatisticPoissonUnivariate.jl")
        if type == 1 #min
            model = OrderStatisticPoissonUnivariate(N, M, a, b, D, 1)
        elseif type == 2 #median 
            model = OrderStatisticPoissonUnivariate(N, M, a, b, D, div(D,2)+1)
        elseif type == 3 #max 
            model = OrderStatisticPoissonUnivariate(N, M, a, b, D, D)
        end
    end
elseif dist == 2
    #hyperparameters
    a = .0001
    b = .0001 
    alpha = 1
    beta = 1
    if D == 1
        include("/home/jlederman/DiscreteOrderStatistics/models/NegBinUnivariate.jl")
        model = NegBinUnivariate(N, M, a, b, alpha, beta)
    else
        @assert D > 1
        include("/home/jlederman/DiscreteOrderStatistics/models/OrderStatisticNegBinUnivariate.jl")
        if type == 1 #min
            model = OrderStatisticNegBinUnivariate(N, M, a, b, alpha, beta, D, 1)
        elseif type == 2 #median 
            model = OrderStatisticNegBinUnivariate(N, M, a, b, alpha, beta, D, div(D,2)+1)
        elseif type == 3 #max 
            model = OrderStatisticNegBinUnivariate(N, M, a, b, alpha, beta, D, D)
        end
    end
end

outfile = folder * "samples_Dist$(dist)Type$(type)D$(D)NU$(nu)Seed$(seed).jld"
samples = fit(model, data, nsamples = 1000, nburnin=10000, nthin=10)
save(outfile, "samples", samples)
