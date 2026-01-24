include("../helper/MatrixMF.jl")
include("../helper/PoissonMaxFunctions.jl")
using Distributions

struct MaxFirstPoissonUnivariate <: MatrixMF
    N::Int64
    M::Int64
    a::Float64
    b::Float64
    D::Int64
end

function evalulateLogLikelihood(model::MaxFirstPoissonUnivariate, state, data, info, row, col)
    Y = data["Y_NM"][row,col]
    mu = state["mu"]
    return logpmfMaxPoisson(Y,mu,model.D)
end

function sample_prior(model::MaxFirstPoissonUnivariate, info=nothing)
    mu = rand(Gamma(model.a, 1/model.b))
    state = Dict("mu" => mu)
    return state
end

function forward_sample(model::MaxFirstPoissonUnivariate; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model)
    end
    mu = state["mu"]
    Y_NM = rand(OrderStatistic(Poisson(mu), model.D, model.D), model.N, model.M)
    data = Dict("Y_NM" => Y_NM)
    return data, state 
end

function backward_sample(model::MaxFirstPoissonUnivariate, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    mu = copy(state["mu"])
    Z_N = zeros(model.N)

    for n in 1:model.N
        if Y_NM[n,1] > 0
            Z_N[n] = sampleFirstMax(Y_NM[n,1], model.D, Poisson(mu))
        end
    end

    post_shape = model.a + sum(Z_N)
    post_rate = model.b + model.N
    mu = rand(Gamma(post_shape, 1/post_rate))
    state = Dict("mu" => mu)
    return data, state
end

# N = 100
# M = 100
# K = 2
# a = b = c = d = 1
# D = 10
# model = maxPoissonMF(N,M,K,a,b,c,d,D)
# data, state = forward_sample(model)
# posteriorsamples = fit(model, data, nsamples=100,nburnin=100,nthin=1)
# print(evaluateInfoRate(model, data, posteriorsamples))
#fsamples, bsamples = gewekeTest(model, ["U_NK", "V_KM"], nsamples=1000, nburnin=100, nthin=1)